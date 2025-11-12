import os
import time
import traceback
import tempfile
from collections import OrderedDict

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue, TensorboardWriter
import lib.utils.misc as misc

# Optional HF upload + plotting
from huggingface_hub import upload_file, HfFolder, upload_folder, hf_hub_download
import matplotlib.pyplot as plt


class LTRTrainer(BaseTrainer):
    """
    Full-featured trainer for SeqTrack (restores original behaviour) with:
      - phase-based checkpoint saving (local)
      - resume from previous checkpoint support
      - per-phase IoU plotting
      - optional upload to Hugging Face
      - mixed precision (AMP) support
      - restore IoU values across resumed checkpoints
    """

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        # restore default params that original trainer relied on
        self._set_default_settings()

        # keep the loader->stats mapping (same as original)
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Tensorboard on main process
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir, exist_ok=True)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings

        # AMP
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None

        # IoU and Loss bookkeeping
        self.iou_values = []  # append one value per epoch (if found in stats)
        self.loss_values = []  # append one value per epoch (if found in stats)
        self.repo_id = getattr(self.settings, "repo_id", None)
        self.phase_name = getattr(self.settings, "phase_name", "phase_1")
        self.resume_checkpoint = getattr(self.settings, "resume_checkpoint", None)
        # Removed skipped_epochs - we want to retrain for reproducible results

        # Phase directory resolution - integrate with base trainer's checkpoint system
        if hasattr(self, '_checkpoint_dir') and self._checkpoint_dir:
            # Use base trainer's checkpoint directory structure
            self.phase_dir = os.path.join(self._checkpoint_dir, self.phase_name)
        else:
            # Fallback to settings-based directory
            default_base = r"D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack"
            workspace_base = getattr(self.settings, "save_dir", None) or getattr(self.settings.env, "workspace_dir", None) or default_base
            self.phase_dir = os.path.join(os.path.abspath(workspace_base), "checkpoints", self.phase_name)
        
        os.makedirs(self.phase_dir, exist_ok=True)

        # Logging file path
        if getattr(self.settings, 'log_file', None) is None:
            # Use the specific log filename requested
            self.settings.log_file = os.path.join(self.phase_dir, "seqtrack-seqtrack_b256.log")

        # Initialize epoch tracking
        if not hasattr(self, 'epoch') or self.epoch is None:
            self.epoch = 0
        
        # If resume_checkpoint provided, attempt to load weights, optimizer, epoch, and iou_values
        if self.resume_checkpoint:
            try:
                # Check if resume_checkpoint is a Hugging Face path or needs downloading
                checkpoint_path = self._resolve_checkpoint_path(self.resume_checkpoint)
                
                # If resolution failed but original path exists, use it
                if checkpoint_path is None and os.path.exists(self.resume_checkpoint):
                    checkpoint_path = self.resume_checkpoint
                
                if checkpoint_path and os.path.exists(checkpoint_path):
                    print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
                    try:
                        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                        net = self.actor.net.module if hasattr(self.actor.net, 'module') else self.actor.net
                        
                        # Load network weights with better error handling
                        if 'net' in ckpt:
                            missing_keys, unexpected_keys = net.load_state_dict(ckpt['net'], strict=False)
                            if missing_keys:
                                print(f"⚠️ Missing keys in network: {missing_keys[:5]}...")  # Show first 5
                            if unexpected_keys:
                                print(f"⚠️ Unexpected keys in network: {unexpected_keys[:5]}...")  # Show first 5
                        elif 'state_dict' in ckpt:
                            missing_keys, unexpected_keys = net.load_state_dict(ckpt['state_dict'], strict=False)
                            if missing_keys:
                                print(f"⚠️ Missing keys in state_dict: {missing_keys[:5]}...")
                            if unexpected_keys:
                                print(f"⚠️ Unexpected keys in state_dict: {unexpected_keys[:5]}...")
                        elif 'model_state' in ckpt:
                            missing_keys, unexpected_keys = net.load_state_dict(ckpt['model_state'], strict=False)
                            if missing_keys:
                                print(f"⚠️ Missing keys in model_state: {missing_keys[:5]}...")
                            if unexpected_keys:
                                print(f"⚠️ Unexpected keys in model_state: {unexpected_keys[:5]}...")
                        else:
                            if os.path.isdir(self.resume_checkpoint):
                                self.load_state_dict(self.resume_checkpoint)

                        # Load optimizer state
                        if 'optimizer' in ckpt:
                            try:
                                self.optimizer.load_state_dict(ckpt['optimizer'])
                                print("✅ Optimizer state restored")
                            except Exception as e:
                                print("⚠️ Could not load optimizer state:", e)

                        # Load epoch
                        if 'epoch' in ckpt:
                            self.epoch = ckpt.get('epoch', self.epoch)
                            print(f"[{self.phase_name}] ✅ Resuming from epoch: {self.epoch}")
                            
                            # Set epoch in data samplers for reproducible data loading
                            for loader in self.loaders:
                                if isinstance(loader.sampler, DistributedSampler):
                                    loader.sampler.set_epoch(self.epoch)
                            print(f"[{self.phase_name}] ✅ Data sampler epoch set for reproducible data loading")

                        # --- RESTORE IoU VALUES ---
                        if 'iou_values' in ckpt and ckpt['iou_values'] is not None:
                            self.iou_values = ckpt['iou_values']
                            print(f"[{self.phase_name}] ✅ IoU values restored: {len(self.iou_values)} epochs of history")
                            print(f"   IoU values: {self.iou_values[:5]}..." if len(self.iou_values) > 5 else f"   IoU values: {self.iou_values}")
                            
                            # Note: We will retrain all epochs from resume point for reproducible results
                        else:
                            print("⚠️ No IoU values found in checkpoint, starting fresh")
                            self.iou_values = []

                        # --- RESTORE LOSS VALUES ---
                        if 'loss_values' in ckpt and ckpt['loss_values'] is not None:
                            self.loss_values = ckpt['loss_values']
                            print(f"[{self.phase_name}] ✅ Loss values restored: {len(self.loss_values)} epochs of history")
                            print(f"   Loss values: {self.loss_values[:5]}..." if len(self.loss_values) > 5 else f"   Loss values: {self.loss_values}")
                        else:
                            print("⚠️ No Loss values found in checkpoint, starting fresh")
                            self.loss_values = []

                        # Load stats if available
                        if 'stats' in ckpt and ckpt['stats'] is not None:
                            self.stats = ckpt['stats']
                            print("✅ Training stats restored")

                        # Load random state if available
                        if 'random_state' in ckpt and ckpt['random_state'] is not None:
                            import random
                            import numpy as np
                            random.setstate(ckpt['random_state']['python'])
                            np.random.set_state(ckpt['random_state']['numpy'])
                            torch.set_rng_state(ckpt['random_state']['torch'])
                            if torch.cuda.is_available():
                                torch.cuda.set_rng_state(ckpt['random_state']['torch_cuda'])
                            print("✅ Random state restored for reproducible training")

                        print("✅ Resume checkpoint loaded successfully")
                    except Exception as e:
                        try:
                            # Try fallback with resolved path
                            if checkpoint_path:
                                self.load_state_dict(checkpoint_path)
                            else:
                                self.load_state_dict(self.resume_checkpoint)
                            print("✅ Fallback: loaded using base trainer method")
                        except Exception as inner_e:
                            print("⚠️ Error while loading resume checkpoint:", e, inner_e)
                else:
                    print(f"⚠️ Resume checkpoint path does not exist: {checkpoint_path if checkpoint_path else self.resume_checkpoint}")
                    print(f"   Original input: {self.resume_checkpoint}")
            except Exception as e:
                print("⚠️ Error while loading resume checkpoint:", e)
                import traceback
                traceback.print_exc()
        else:
            # No resume checkpoint - starting training from scratch
            print(f"[{self.phase_name}] 🚀 Starting training from scratch (epoch 1)")
            print(f"[{self.phase_name}]    IoU and Loss history will be collected during training")
    
    def _resolve_checkpoint_path(self, checkpoint_input):
        """
        Resolve checkpoint path from various input formats:
        - Local file path: "/path/to/checkpoint.pth.tar"
        - Hugging Face format: "hf://repo_id/phase_name/checkpoint_name.pth.tar"
        - Hugging Face format: "repo_id/phase_name/checkpoint_name.pth.tar"
        - Just checkpoint name (assumes Hugging Face if repo_id is set)
        
        Returns: Local file path to checkpoint (downloaded if from HF)
        """
        # If it's already a local file that exists, return it
        if os.path.exists(checkpoint_input):
            return checkpoint_input
        
        # Check if it's a Hugging Face path
        hf_path = None
        repo_id = None
        filename = None
        
        # Format 1: "hf://user/repo/<path-inside-repo>"
        hf_prefix = getattr(self.settings, 'hf_train_prefix', 'member_10_abdelrahman_ahmed/training')

        if checkpoint_input.startswith("hf://"):
            raw = checkpoint_input[5:]  # strip "hf://"
            parts = raw.split("/")
            if len(parts) >= 3:
                # repo_id is always first two segments: user/repo
                repo_id = f"{parts[0]}/{parts[1]}"
                # the rest is the file path inside the repo
                filename = "/".join(parts[2:])
                hf_path = filename
        
        # Format 2: "repo_id/phase_name/checkpoint_name.pth.tar" (no hf:// prefix)
        elif "/" in checkpoint_input and not os.path.isabs(checkpoint_input):
            # Try to parse as repo_id/phase/filename
            # repo_id can contain "/" (e.g., "user/repo-name")
            # We need to find where phase_name starts
            # Common pattern: user/repo-name/phase_name/filename.pth.tar
            parts = checkpoint_input.split("/")
            if len(parts) >= 3:
                # Try different splits to find repo_id
                # If we have repo_id set, check if first parts match
                if self.repo_id:
                    repo_parts = self.repo_id.split("/")
                    if len(parts) >= len(repo_parts) + 1:
                        # Check if first parts match repo_id
                        if "/".join(parts[:len(repo_parts)]) == self.repo_id:
                            repo_id = self.repo_id
                            filename = "/".join(parts[len(repo_parts):])
                            hf_path = filename
                else:
                    # Try to detect: assume repo_id is first part(s) if it contains "/"
                    # For now, try 2-part repo_id (user/repo)
                    if len(parts) >= 3:
                        potential_repo = f"{parts[0]}/{parts[1]}"
                        filename = "/".join(parts[2:])
                        repo_id = potential_repo
                        hf_path = filename
        
        # Format 3: Just filename - use repo_id and hf_train_prefix from settings
        if not hf_path and self.repo_id:
            # Check if it's just a filename (no slashes)
            if "/" not in checkpoint_input:
                repo_id = self.repo_id
                filename = checkpoint_input
                hf_path = f"{hf_prefix}/{filename}" if hf_prefix else filename
            else:
                repo_id = self.repo_id
                hf_path = checkpoint_input
        
        # Download from Hugging Face if we identified an HF path
        if hf_path and repo_id:
            try:
                token = HfFolder.get_token()
                if not token:
                    print("⚠️ Hugging Face token not found. Run `huggingface-cli login` first.")
                    print(f"   Attempting to download without token (public repos only)...")
                
                print(f"📥 Downloading checkpoint from Hugging Face: {repo_id}/{hf_path}")
                
                # Download to temp location (for Kaggle - no local saving)
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=hf_path,
                    repo_type="model",
                    token=token,
                    cache_dir=tempfile.gettempdir()
                )
                
                print(f"✅ Checkpoint downloaded to: {downloaded_path}")
                return downloaded_path
                
            except Exception as e:
                print(f"⚠️ Failed to download checkpoint from Hugging Face: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # If we couldn't resolve it, return None
        return None

    def _set_default_settings(self):
        default = {
            'print_interval': 10,
            'print_stats': None,
            'description': '',
            'grad_clip_norm': getattr(self.settings, 'grad_clip_norm', 0),
        }
        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)
        self._init_timing()

        for i, data in enumerate(loader, 1):
            if self.move_data_to_gpu and isinstance(data, dict):
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings

            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if getattr(self.settings, 'grad_clip_norm', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    if getattr(self.settings, 'grad_clip_norm', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            batch_size = data['template_images'].shape[loader.stack_dim] if 'template_images' in data else 1
            self._update_stats(stats, batch_size, loader)
            self._print_stats(i, loader, batch_size)

    def train_epoch(self):
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                if isinstance(getattr(loader, 'sampler', None), DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        # Collect IoU and Loss BEFORE resetting meters
        # Always collect metrics for current epoch (will overwrite if resuming)
        iou_collected = False
        loss_collected = False
        
        for loader_name, loader_stats in self.stats.items():
            if loader_stats is None:
                continue
            for stat_name, stat_val in loader_stats.items():
                # Collect IoU values - try multiple patterns including exact match
                if not iou_collected and (stat_name.lower() == 'iou' or 
                    'iou' in stat_name.lower() or 
                    'intersection' in stat_name.lower()):
                    try:
                        # Only initialize if it doesn't exist, don't reset if it's None
                        if not hasattr(self, 'iou_values'):
                            self.iou_values = []
                        elif self.iou_values is None:
                            self.iou_values = []
                        
                        # Only append if we have a valid stat value
                        if hasattr(stat_val, 'avg') and stat_val.avg is not None:
                            iou_value = float(stat_val.avg)
                            self.iou_values.append(iou_value)
                            print(f"[{self.phase_name}] IoU collected: Epoch {self.epoch}, Value: {iou_value:.4f}, Total IoU history: {len(self.iou_values)} values")
                            iou_collected = True
                    except Exception as e:
                        print(f"Warning: Could not collect IoU value: {e}")
                        pass
                
                # Collect Loss values - try multiple patterns including exact match
                if not loss_collected and (stat_name.lower() == 'loss/total' or 
                    stat_name.lower() == 'loss' or 
                    'loss/total' in stat_name.lower() or 
                    ('loss' in stat_name.lower() and 'total' in stat_name.lower())):
                    try:
                        # Only initialize if it doesn't exist, don't reset if it's None
                        if not hasattr(self, 'loss_values'):
                            self.loss_values = []
                        elif self.loss_values is None:
                            self.loss_values = []
                        
                        # Only append if we have a valid stat value
                        if hasattr(stat_val, 'avg') and stat_val.avg is not None:
                            loss_value = float(stat_val.avg)
                            self.loss_values.append(loss_value)
                            print(f"[{self.phase_name}] Loss collected: Epoch {self.epoch}, Value: {loss_value:.4f}, Total Loss history: {len(self.loss_values)} values")
                            loss_collected = True
                    except Exception as e:
                        print(f"Warning: Could not collect Loss value: {e}")
                        pass
        
        # If no IoU was collected, try to extract from log file as fallback
        if not iou_collected:
            try:
                # This is a fallback method - extract IoU from the log file
                import re
                log_file = getattr(self.settings, 'log_file', None)
                if log_file and os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Look for the last IoU value in the log
                    for line in reversed(lines[-100:]):  # Check last 100 lines
                        if f'[{self.phase_name}]' in line and 'IoU:' in line:
                            iou_match = re.search(r'IoU:\s*([0-9.]+)', line)
                            if iou_match:
                                iou_value = float(iou_match.group(1))
                                if not hasattr(self, 'iou_values'):
                                    self.iou_values = []
                                elif self.iou_values is None:
                                    self.iou_values = []
                                self.iou_values.append(iou_value)
                                print(f"[{self.phase_name}] IoU collected (from log): Epoch {self.epoch}, Value: {iou_value:.4f}, Total IoU history: {len(self.iou_values)} values")
                                iou_collected = True
                                break
            except Exception as e:
                print(f"Warning: Could not extract IoU from log: {e}")
                pass

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

        # Save checkpoint and IoU plot
        if misc.is_main_process():
            # Local checkpoint saving - COMMENTED OUT for Kaggle
            # try:
            #     # Use consistent naming with base trainer
            #     net = self.actor.net.module if hasattr(self.actor.net, 'module') else self.actor.net
            #     net_type = type(net).__name__
            #     ckpt_name = f"{net_type}_ep{self.epoch:04d}.pth.tar"
            #     ckpt_path = os.path.join(self.phase_dir, ckpt_name)

            #     # Include all necessary fields for proper resuming
            #     state = {
            #         'epoch': self.epoch,
            #         'actor_type': type(self.actor).__name__,
            #         'net_type': net_type,
            #         'net': net.state_dict(),
            #         'net_info': getattr(net, 'info', None),
            #         'constructor': getattr(net, 'constructor', None),
            #         'optimizer': self.optimizer.state_dict(),
            #         'stats': self.stats,
            #         'iou_values': self.iou_values,  # SAVE IoU values for continuity
            #         'loss_values': self.loss_values,  # SAVE Loss values for continuity
            #         'settings': self.settings
            #     }
                
            #     # Save random state for reproducible resuming
            #     import random
            #     import numpy as np
            #     state['random_state'] = {
            #         'python': random.getstate(),
            #         'numpy': np.random.get_state(),
            #         'torch': torch.get_rng_state(),
            #         'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            #     }

            #     # Atomic save operation
            #     tmp_path = ckpt_path + ".tmp"
            #     torch.save(state, tmp_path)
            #     if os.path.exists(ckpt_path):
            #         os.remove(ckpt_path)
            #     os.rename(tmp_path, ckpt_path)

            # except Exception as e:
            #     print("⚠️ Failed to save checkpoint:", e)
            #     traceback.print_exc()

            # Prepare checkpoint path for Hugging Face upload (only when needed)
            ckpt_path = None
            if self.repo_id and self.epoch % 5 == 0:
                try:
                    net = self.actor.net.module if hasattr(self.actor.net, 'module') else self.actor.net
                    net_type = type(net).__name__
                    ckpt_name = f"{net_type}_ep{self.epoch:04d}.pth.tar"
                    # Use a temporary location for checkpoint (won't be saved locally)
                    temp_dir = tempfile.gettempdir()
                    ckpt_path = os.path.join(temp_dir, ckpt_name)
                    
                    # Include all necessary fields for proper resuming
                    state = {
                        'epoch': self.epoch,
                        'actor_type': type(self.actor).__name__,
                        'net_type': net_type,
                        'net': net.state_dict(),
                        'net_info': getattr(net, 'info', None),
                        'constructor': getattr(net, 'constructor', None),
                        'optimizer': self.optimizer.state_dict(),
                        'stats': self.stats,
                        'iou_values': self.iou_values,
                        'loss_values': self.loss_values,
                        'settings': self.settings
                    }
                    
                    # Save random state for reproducible resuming
                    import random
                    import numpy as np
                    state['random_state'] = {
                        'python': random.getstate(),
                        'numpy': np.random.get_state(),
                        'torch': torch.get_rng_state(),
                        'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                    }

                    # Save to temp location for upload
                    torch.save(state, ckpt_path)
                except Exception as e:
                    print("⚠️ Failed to prepare checkpoint for upload:", e)
                    traceback.print_exc()
                    ckpt_path = None

            # IoU and Loss plots - show complete training history
            iou_fig_path = None
            loss_fig_path = None
            try:
                # IoU Plot
                if hasattr(self, 'iou_values') and len(self.iou_values) > 0:
                    iou_fig_path = os.path.join(self.phase_dir, f"{self.phase_name}_iou.png")
                    plt.figure(figsize=(10, 6))
                    
                    # Plot IoU values with proper epoch numbering
                    epochs = range(1, len(self.iou_values) + 1)
                    plt.plot(epochs, self.iou_values, marker='o', linewidth=2, markersize=4, color='blue')
                    
                    # Add current epoch marker
                    if len(self.iou_values) > 0:
                        plt.axvline(x=self.epoch, color='red', linestyle='--', alpha=0.7, 
                                  label=f'Current Epoch: {self.epoch}')
                    
                    plt.xlabel("Epoch")
                    plt.ylabel("IoU")
                    plt.title(f"IoU Progress - {self.phase_name} (Total Epochs: {len(self.iou_values)})")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(iou_fig_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"[{self.phase_name}] IoU plot saved at: {iou_fig_path} (showing {len(self.iou_values)} epochs)")
                    print(f"[{self.phase_name}]    IoU values range: {min(self.iou_values):.4f} to {max(self.iou_values):.4f}")
                    print(f"[{self.phase_name}]    Current epoch: {self.epoch}, Total epochs in history: {len(self.iou_values)}")
                else:
                    print("NO IOU")
                    iou_fig_path = None

                # Loss Plot
                if hasattr(self, 'loss_values') and len(self.loss_values) > 0:
                    loss_fig_path = os.path.join(self.phase_dir, f"{self.phase_name}_loss.png")
                    plt.figure(figsize=(10, 6))
                    
                    # Plot Loss values with proper epoch numbering
                    epochs = range(1, len(self.loss_values) + 1)
                    plt.plot(epochs, self.loss_values, marker='o', linewidth=2, markersize=4, color='red')
                    
                    # Add current epoch marker
                    if len(self.loss_values) > 0:
                        plt.axvline(x=self.epoch, color='red', linestyle='--', alpha=0.7, 
                                  label=f'Current Epoch: {self.epoch}')
                    
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title(f"Loss Progress - {self.phase_name} (Total Epochs: {len(self.loss_values)})")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(loss_fig_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"[{self.phase_name}] Loss plot saved at: {loss_fig_path} (showing {len(self.loss_values)} epochs)")
                    print(f"[{self.phase_name}]    Loss values range: {min(self.loss_values):.4f} to {max(self.loss_values):.4f}")
                    print(f"[{self.phase_name}]    Current epoch: {self.epoch}, Total epochs in history: {len(self.loss_values)}")
                else:
                    loss_fig_path = None
            except Exception as e:
                print("⚠️ Failed to save plots:", e)
                iou_fig_path = None
                loss_fig_path = None


            # Optional Hugging Face upload
            if self.repo_id:
                try:
                    token = HfFolder.get_token()
                    if not token:
                        print("⚠️ Hugging Face token not found. Run `huggingface-cli login` first.")
                    else:
                        hf_prefix = getattr(self.settings, 'hf_train_prefix', 'member_10_abdelrahman_ahmed/training').rstrip('/')
                        # Upload checkpoint only every 5 epochs
                        if ckpt_path and self.epoch % 5 == 0:
                            try:
                                upload_file(
                                    path_or_fileobj=ckpt_path,
                                    path_in_repo=f"{hf_prefix}/{os.path.basename(ckpt_path)}",
                                    repo_id=self.repo_id,
                                    repo_type="model",
                                    token=token,
                                )
                                print(f"Uploaded checkpoint to Hugging Face: {self.repo_id}/{hf_prefix}/{os.path.basename(ckpt_path)}")
                                # Clean up temp checkpoint file after upload
                                try:
                                    if os.path.exists(ckpt_path):
                                        os.remove(ckpt_path)
                                except:
                                    pass
                            except Exception as e:
                                print("⚠️ Failed uploading checkpoint to Hugging Face:", e)

                        # Upload IoU plot every epoch
                        if iou_fig_path:
                            try:
                                upload_file(
                                    path_or_fileobj=iou_fig_path,
                                    path_in_repo=f"{hf_prefix}/{os.path.basename(iou_fig_path)}",
                                    repo_id=self.repo_id,
                                    repo_type="model",
                                    token=token,
                                )
                                print(f"Uploaded IoU plot to Hugging Face: {self.repo_id}/{hf_prefix}/{os.path.basename(iou_fig_path)}")
                            except Exception as e:
                                print("⚠️ Failed uploading IoU plot to Hugging Face:", e)

                        # Upload Loss plot every epoch
                        if loss_fig_path:
                            try:
                                upload_file(
                                    path_or_fileobj=loss_fig_path,
                                    path_in_repo=f"{hf_prefix}/{os.path.basename(loss_fig_path)}",
                                    repo_id=self.repo_id,
                                    repo_type="model",
                                    token=token,
                                )
                                print(f"Uploaded Loss plot to Hugging Face: {self.repo_id}/{hf_prefix}/{os.path.basename(loss_fig_path)}")
                            except Exception as e:
                                print("⚠️ Failed uploading Loss plot to Hugging Face:", e)

                except Exception as e:
                    print("⚠️ Hugging Face upload block error:", e)

    # ----------------- Helpers copied / restored -----------------
    @staticmethod
    def _format_time(seconds):
        seconds = int(round(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:01}:{minutes:02}:{secs:02} hours"

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.last_log_time = self.start_time
        self.next_log_point = 50
        self.last_log_frames = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})
        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        if self.num_frames >= self.next_log_point or i == loader.__len__():
            current_time = time.time()
            time_for_last_interval = current_time - self.last_log_time
            time_since_beginning = current_time - self.start_time

            total_samples = loader.__len__() * batch_size
            samples_left = max(0, total_samples - self.num_frames)
            avg_time_per_sample = time_since_beginning / self.num_frames if self.num_frames > 0 else 0
            time_left = samples_left * avg_time_per_sample

            log_str = (
                f"[{self.phase_name}] Epoch {self.epoch}: {self.num_frames} / {total_samples} samples,\n"
                f"time for last {self.num_frames - self.last_log_frames} samples: {self._format_time(time_for_last_interval)}\n"
                f"time since beginning {self._format_time(time_since_beginning)},\n"
                f"time left to finish the epoch {self._format_time(time_left)}\n"
            )

            stats_str = ""
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    stats_str += f"{name}: {val.avg:.5f}, "
            log_str += stats_str.strip(', ') + "\n"

            print(log_str)
            if misc.is_main_process():
                try:
                    # Enhanced logging - save all training logs
                    with open(self.settings.log_file, 'a', encoding='utf-8') as f:
                        f.write(log_str)
                        f.flush()  # Ensure immediate write
                except Exception as e:
                    print(f"⚠️ Failed to write to log file: {e}")
                    pass

            self.last_log_time = current_time
            self.last_log_frames = self.num_frames
            self.next_log_point += 50

    def _stats_new_epoch(self):
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_lr()
                except Exception:
                    try:
                        lr_list = self.lr_scheduler._get_lr(self.epoch)
                    except Exception:
                        lr_list = [None]
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            try:
                self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)
            except Exception:
                pass
        try:
            self.tensorboard_writer.write_epoch(self.stats, self.epoch)
        except Exception:
            pass

    def train(self, max_epochs, load_latest=False, fail_safe=True, load_previous_ckpt=False, distill=False):
        """Override base trainer's train method to handle epoch skipping"""
        epoch = -1
        num_tries = 1
        for i in range(num_tries):
            try:
                if load_latest:
                    self.load_checkpoint()
                if load_previous_ckpt:
                    directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path_prv)
                    self.load_state_dict(directory)
                if distill:
                    directory_teacher = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path_teacher)
                    self.load_state_dict(directory_teacher, distill=True)
                
                # Modified training loop - start from epoch+1 to max_epochs
                # If no resume checkpoint, self.epoch is 0, so training starts from epoch 1
                start_epoch = self.epoch + 1
                if start_epoch == 1:
                    print(f"[{self.phase_name}] 🚀 Starting training from epoch 1 to {max_epochs}")
                else:
                    print(f"[{self.phase_name}] 🔁 Resuming training from epoch {start_epoch} to {max_epochs}")
                
                for epoch in range(start_epoch, max_epochs+1):
                    self.epoch = epoch
                    
                    # No epoch skipping - retrain for reproducible results
                    
                    self.train_epoch()

                    if self.lr_scheduler is not None:
                        if self.settings.scheduler_type != 'cosine':
                            self.lr_scheduler.step()
                        else:
                            self.lr_scheduler.step(epoch - 1)
                    
                    # Save checkpoint at the end of every epoch - COMMENTED OUT for Kaggle
                    # if self._checkpoint_dir:
                    #     if self.settings.local_rank in [-1, 0]:
                    #         self.save_checkpoint()
                            
            except:
                print(f'[{self.phase_name}] Training crashed at epoch {epoch}')
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print(f'[{self.phase_name}] Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')

    def save_checkpoint(self):
        """Override base trainer's save_checkpoint to use phase-specific directory"""
        # Local checkpoint saving - COMMENTED OUT for Kaggle
        # if not misc.is_main_process():
        #     return
            
        # try:
        #     net = self.actor.net.module if hasattr(self.actor.net, 'module') else self.actor.net
        #     net_type = type(net).__name__
        #     ckpt_name = f"{net_type}_ep{self.epoch:04d}.pth.tar"
        #     ckpt_path = os.path.join(self.phase_dir, ckpt_name)

        #     state = {
        #         'epoch': self.epoch,
        #         'actor_type': type(self.actor).__name__,
        #         'net_type': net_type,
        #         'net': net.state_dict(),
        #         'net_info': getattr(net, 'info', None),
        #         'constructor': getattr(net, 'constructor', None),
        #         'optimizer': self.optimizer.state_dict(),
        #         'stats': self.stats,
        #         'iou_values': self.iou_values,
        #         'loss_values': self.loss_values,
        #         'settings': self.settings
        #     }
            
        #     # Save random state for reproducible resuming
        #     import random
        #     import numpy as np
        #     state['random_state'] = {
        #         'python': random.getstate(),
        #         'numpy': np.random.get_state(),
        #         'torch': torch.get_rng_state(),
        #         'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        #     }

        #     tmp_path = ckpt_path + ".tmp"
        #     torch.save(state, tmp_path)
        #     if os.path.exists(ckpt_path):
        #         os.remove(ckpt_path)
        #     os.rename(tmp_path, ckpt_path)
            
        #     print(f"Phase checkpoint saved at: {ckpt_path}")
        # except Exception as e:
        #     print("⚠️ Failed to save phase checkpoint:", e)
        pass  # Local checkpoint saving disabled for Kaggle
