"""
Assignment 4: Test and Evaluation Script
==========================================

This script:
1. Downloads checkpoints from Hugging Face (for both phases)
2. Runs inference on LaSOT test sequences for each checkpoint
3. Evaluates performance metrics (IoU, Precision, AUC, FPS)
4. Generates tables and graphs for results
5. Does NOT save checkpoints locally (uses temp files for Kaggle)

Usage:
    python tracking/assignment_4_test.py --repo_id USER/seqtrack-checkpoints --phase phase_1 --start_epoch 1 --end_epoch 10
"""

import os
import sys
import argparse
import tempfile
import shutil
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add paths
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from huggingface_hub import hf_hub_download, HfFolder, list_repo_files, upload_file
from lib.test.evaluation import get_dataset, trackerlist
from lib.test.evaluation.running import run_dataset
from lib.test.analysis.plot_results import print_results
from lib.test.analysis.extract_results import extract_results


class Assignment4Evaluator:
    """Evaluator for Assignment 4 - Test and Evaluation"""
    
    def __init__(self, repo_id, phase_name, start_epoch=1, end_epoch=10, 
                 dataset_name='lasot', temp_dir=None, upload_prefix="member_10_abdelrahman_ahmed/test",
                 resume_epoch=None):
        self.repo_id = repo_id
        self.phase_name = phase_name
        self.safe_phase = phase_name.replace('/', '_').replace('\\', '_')
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.dataset_name = dataset_name
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.upload_prefix = upload_prefix
        self.resume_epoch = resume_epoch
        self.results_dir = os.path.join(self.temp_dir, f"assignment_4_results_{self.safe_phase}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Storage for results
        self.evaluation_results = {}  # epoch -> metrics dict
        self.inference_rates = {}     # epoch -> FPS/ms_per_frame
        self.per_sequence_metrics = {}  # sequence -> {epoch -> metrics}

        # Load previous summary if resuming
        self.summary_path = os.path.join(self.results_dir, f"summary.json")
        if os.path.exists(self.summary_path):
            try:
                with open(self.summary_path, 'r') as f:
                    summary = json.load(f)
                prev_eval = summary.get('evaluation_results', {})
                prev_inf = summary.get('inference_rates', {})
                prev_seq = summary.get('per_sequence_metrics', {})
                # keys in summary may be strings, convert to int
                for k, v in prev_eval.items():
                    try:
                        epoch = int(k)
                    except Exception:
                        continue
                    self.evaluation_results[epoch] = v
                for k, v in prev_inf.items():
                    try:
                        epoch = int(k)
                    except Exception:
                        continue
                    self.inference_rates[epoch] = v
                for seq_name, epoch_dict in prev_seq.items():
                    seq_store = self.per_sequence_metrics.setdefault(seq_name, {})
                    for k, v in epoch_dict.items():
                        try:
                            epoch = int(k)
                        except Exception:
                            continue
                        seq_store[epoch] = v
                if self.evaluation_results or self.per_sequence_metrics:
                    print(f"🔄 Loaded previous evaluation history from summary (epochs: {len(self.evaluation_results)}, sequences: {len(self.per_sequence_metrics)}).")
            except Exception as e:
                print(f"⚠️ Failed to load previous summary: {e}")
        
    def list_checkpoints(self):
        """List available checkpoints in Hugging Face repo for this phase"""
        try:
            token = HfFolder.get_token()
            if not token:
                print("⚠️ Hugging Face token not found. Run `huggingface-cli login` first.")
                return []
            
            files = list_repo_files(self.repo_id, repo_type="model", token=token)
            phase_files = [f for f in files if f.startswith(f"{self.phase_name}/") and f.endswith(".pth.tar")]
            
            # Extract epoch numbers
            checkpoints = []
            for f in phase_files:
                # Expected format: phase_1/SEQTRACK_ep0005.pth.tar
                filename = os.path.basename(f)
                if "ep" in filename:
                    try:
                        epoch_str = filename.split("ep")[1].split(".")[0]
                        epoch = int(epoch_str)
                        if self.start_epoch <= epoch <= self.end_epoch:
                            checkpoints.append((epoch, f))
                    except:
                        continue
            
            checkpoints.sort(key=lambda x: x[0])

            # Apply resume filter if requested
            if self.resume_epoch is not None:
                checkpoints = [(epoch, path) for epoch, path in checkpoints if epoch >= self.resume_epoch]
                if not checkpoints:
                    print(f"⚠️ Resume epoch {self.resume_epoch} is higher than available checkpoints. Nothing to do.")
                    checkpoints = []
            print(f"📋 Found {len(checkpoints)} checkpoints for {self.phase_name}: epochs {[c[0] for c in checkpoints]}")
            return checkpoints
            
        except Exception as e:
            print(f"⚠️ Error listing checkpoints: {e}")
            return []
    
    def download_checkpoint(self, checkpoint_path_in_repo, epoch):
        """Download checkpoint from Hugging Face to temp location"""
        try:
            token = HfFolder.get_token()
            if not token:
                raise RuntimeError("Hugging Face token not found")
            
            # Download to temp file
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=checkpoint_path_in_repo,
                repo_type="model",
                token=token,
                cache_dir=self.temp_dir
            )
            
            print(f"✅ Downloaded checkpoint for epoch {epoch}")
            return local_path
            
        except Exception as e:
            print(f"⚠️ Failed to download checkpoint for epoch {epoch}: {e}")
            return None
    
    def run_inference_for_checkpoint(self, checkpoint_path, epoch):
        """Run inference on test dataset for a specific checkpoint"""
        print(f"\n{'='*60}")
        print(f"Running inference for {self.phase_name} - Epoch {epoch}")
        print(f"{'='*60}")
        
        try:
            # Get dataset
            dataset = get_dataset(self.dataset_name)
            
            # Create a custom tracker class that uses our checkpoint
            # We'll monkey-patch the get_parameters method
            from lib.test.evaluation import tracker as tracker_module
            
            # Store original get_parameters
            original_get_parameters = tracker_module.Tracker.get_parameters
            
            def patched_get_parameters(self):
                """Patched get_parameters that uses our checkpoint"""
                params = original_get_parameters(self)
                # Override checkpoint path with downloaded checkpoint
                params.checkpoint = checkpoint_path
                return params
            
            # Apply patch
            tracker_module.Tracker.get_parameters = patched_get_parameters
            
            try:
                # Create tracker with this checkpoint
                trackers = trackerlist(
                    name='seqtrack',
                    parameter_name='seqtrack_b256',
                    dataset_name=self.dataset_name,
                    run_ids=None,
                    display_name=f'{self.phase_name}_ep{epoch:04d}'
                )
                
                print(f"Using checkpoint: {checkpoint_path}")
            
                # Run inference
                run_dataset(dataset, trackers, debug=False, threads=0, num_gpus=1)
                
                # Extract timing information from results
                fps = self._extract_fps_from_results(trackers[0], dataset)
                self.inference_rates[epoch] = fps
                
                print(f"✅ Completed inference for epoch {epoch}")
                
            finally:
                # Restore original get_parameters
                tracker_module.Tracker.get_parameters = original_get_parameters
            
        except Exception as e:
            print(f"⚠️ Error during inference for epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            # Make sure to restore even on error
            try:
                tracker_module.Tracker.get_parameters = original_get_parameters
            except:
                pass
    
    def _extract_fps_from_results(self, tracker, dataset):
        """Extract FPS from saved time files"""
        try:
            total_time = 0
            total_frames = 0
            
            for seq in dataset:
                if seq.dataset in ['lasot', 'trackingnet', 'got10k', 'otb', 'uav', 'nfs', 'tnl2k']:
                    time_file = os.path.join(tracker.results_dir, seq.dataset, f"{seq.name}_time.txt")
                else:
                    time_file = os.path.join(tracker.results_dir, f"{seq.name}_time.txt")
                
                if os.path.exists(time_file):
                    times = np.loadtxt(time_file)
                    total_time += np.sum(times)
                    total_frames += len(times)
            
            if total_frames > 0:
                fps = total_frames / total_time if total_time > 0 else 0
                ms_per_frame = (total_time / total_frames * 1000) if total_frames > 0 else 0
                return {'fps': fps, 'ms_per_frame': ms_per_frame, 'total_frames': total_frames}
            else:
                return {'fps': 0, 'ms_per_frame': 0, 'total_frames': 0}
                
        except Exception as e:
            print(f"⚠️ Error extracting FPS: {e}")
            return {'fps': 0, 'ms_per_frame': 0, 'total_frames': 0}
    
    def evaluate_checkpoint(self, epoch, checkpoint_path):
        """Evaluate a checkpoint and extract metrics"""
        print(f"\nEvaluating epoch {epoch}...")
        
        try:
            # Use the same patching approach as inference
            from lib.test.evaluation import tracker as tracker_module
            original_get_parameters = tracker_module.Tracker.get_parameters
            
            def patched_get_parameters(self):
                """Patched get_parameters that uses our checkpoint"""
                params = original_get_parameters(self)
                params.checkpoint = checkpoint_path
                return params
            
            tracker_module.Tracker.get_parameters = patched_get_parameters
            
            try:
                # Get tracker
                trackers = trackerlist(
                    name='seqtrack',
                    parameter_name='seqtrack_b256',
                    dataset_name=self.dataset_name,
                    run_ids=None,
                    display_name=f'{self.phase_name}_ep{epoch:04d}'
                )
                
                dataset = get_dataset(self.dataset_name)
                
                # Extract results
                # Compute results directly (skip_missing_seq=True to tolerate partial datasets)
                eval_data = extract_results(trackers, dataset, self.dataset_name, skip_missing_seq=True)
                
                # Extract metrics (robust to empty/partial results)
                avg_iou = 0.0
                auc = 0.0
                precision = 0.0

                if eval_data and 'avg_overlap_all' in eval_data:
                    avg_list = eval_data['avg_overlap_all']
                    if isinstance(avg_list, list) and len(avg_list) > 0:
                        # avg_list is [num_seq][num_trackers]; take mean over sequences and trackers
                        try:
                            arr = np.array(avg_list, dtype=float)
                            if arr.size > 0:
                                avg_iou = float(np.nanmean(arr))
                        except Exception:
                            pass
                
                # Extract success rate (AUC) and precision
                if eval_data and 'ave_success_rate_plot_overlap' in eval_data:
                    success_plot = eval_data['ave_success_rate_plot_overlap']
                    try:
                        arr = np.array(success_plot, dtype=float)
                        if arr.size > 0:
                            # arr shape: [num_seq, num_trackers, bins]; take mean over seq+trackers, then mean over bins
                            auc = float(np.nanmean(arr))
                    except Exception:
                        pass
                
                if eval_data and 'ave_success_rate_plot_center' in eval_data:
                    prec_plot = eval_data['ave_success_rate_plot_center']
                    try:
                        arr = np.array(prec_plot, dtype=float)
                        if arr.size > 0:
                            # pick bin 20 if exists; else mean
                            if arr.shape[-1] > 20:
                                precision = float(np.nanmean(arr[..., 20]))
                            else:
                                precision = float(np.nanmean(arr))
                    except Exception:
                        pass

                # Per-sequence metrics accumulation
                if eval_data and 'avg_overlap_all' in eval_data:
                    seq_names = eval_data.get('sequences', [])
                    per_overlap = eval_data.get('avg_overlap_all', [])
                    per_prec = eval_data.get('ave_success_rate_plot_center', [])
                    per_auc = eval_data.get('ave_success_rate_plot_overlap', [])
                    for seq_idx, seq_name in enumerate(seq_names):
                        seq_metrics = self.per_sequence_metrics.setdefault(seq_name, {})
                        entry = {}
                        try:
                            entry['iou'] = float(np.nanmean(np.array(per_overlap[seq_idx], dtype=float)))
                        except Exception:
                            entry['iou'] = 0.0
                        try:
                            arr_prec = np.array(per_prec[seq_idx], dtype=float)
                            if arr_prec.size > 0:
                                entry['precision'] = float(np.nanmean(arr_prec[..., 20])) if arr_prec.shape[-1] > 20 else float(np.nanmean(arr_prec))
                            else:
                                entry['precision'] = 0.0
                        except Exception:
                            entry['precision'] = 0.0
                        try:
                            entry['auc'] = float(np.nanmean(np.array(per_auc[seq_idx], dtype=float)))
                        except Exception:
                            entry['auc'] = 0.0
                        seq_metrics[epoch] = entry
                
                metrics = {
                    'epoch': epoch,
                    'iou': float(avg_iou),
                    'precision': float(precision),
                    'auc': float(auc)
                }
                
                self.evaluation_results[epoch] = metrics
                print(f"✅ Epoch {epoch} - IoU: {avg_iou:.4f}, Precision: {precision:.4f}, AUC: {auc:.4f}")
                
                return metrics
                
            finally:
                # Restore original get_parameters
                tracker_module.Tracker.get_parameters = original_get_parameters
            
        except Exception as e:
            print(f"⚠️ Error evaluating epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            # Make sure to restore even on error
            try:
                tracker_module.Tracker.get_parameters = original_get_parameters
            except:
                pass
            return None
    
    def generate_tables(self):
        """Generate tables for results"""
        print("\n" + "="*60)
        print("Generating Tables")
        print("="*60)
        
        # Table 1: Inference Rate Results
        table1_data = []
        for epoch in sorted(self.inference_rates.keys()):
            rates = self.inference_rates[epoch]
            table1_data.append({
                'Epoch': epoch,
                'FPS': f"{rates['fps']:.2f}",
                'ms/frame': f"{rates['ms_per_frame']:.2f}"
            })
        
        df_table1 = pd.DataFrame(table1_data)
        table1_path = os.path.join(self.results_dir, f"table1_inference_rate.csv")
        df_table1.to_csv(table1_path, index=False)
        print(f"\n📊 Table 1 (Inference Rate) saved to: {table1_path}")
        print(df_table1.to_string(index=False))
        
        # Table 2: Evaluation Results
        table2_data = []
        for epoch in sorted(self.evaluation_results.keys()):
            metrics = self.evaluation_results[epoch]
            table2_data.append({
                'Epoch': epoch,
                'IoU': f"{metrics['iou']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'AUC': f"{metrics['auc']:.4f}"
            })
        
        df_table2 = pd.DataFrame(table2_data)
        table2_path = os.path.join(self.results_dir, f"table2_evaluation.csv")
        df_table2.to_csv(table2_path, index=False)
        print(f"\n📊 Table 2 (Evaluation Results) saved to: {table2_path}")
        print(df_table2.to_string(index=False))

        # Table 3: Per-sequence metrics across epochs
        seq_rows = []
        for seq_name, epoch_dict in self.per_sequence_metrics.items():
            for epoch, metrics in epoch_dict.items():
                seq_rows.append({
                    'Sequence': seq_name,
                    'Epoch': epoch,
                    'IoU': f"{metrics.get('iou', 0.0):.4f}",
                    'Precision': f"{metrics.get('precision', 0.0):.4f}",
                    'AUC': f"{metrics.get('auc', 0.0):.4f}",
                })
        if seq_rows:
            df_table3 = pd.DataFrame(seq_rows).sort_values(by=['Sequence', 'Epoch'])
        else:
            df_table3 = pd.DataFrame(columns=['Sequence', 'Epoch', 'IoU', 'Precision', 'AUC'])
        table3_path = os.path.join(self.results_dir, f"table3_per_sequence.csv")
        df_table3.to_csv(table3_path, index=False)
        print(f"\n📊 Table 3 (Per-sequence Metrics) saved to: {table3_path}")
        
        # Optional: upload tables to the same repo/phase path
        try:
            token = HfFolder.get_token()
            if token:
                for p in [table1_path, table2_path, table3_path]:
                    try:
                        upload_file(
                            path_or_fileobj=p,
                            path_in_repo=f"{self.upload_prefix}/{os.path.basename(p)}",
                            repo_id=self.repo_id,
                            repo_type="model",
                            token=token,
                        )
                        print(f"⬆️ Uploaded table to Hugging Face: {self.repo_id}/{self.upload_prefix}/{os.path.basename(p)}")
                    except Exception as e:
                        print("⚠️ Failed uploading table:", e)
        except Exception as e:
            print("⚠️ Upload block (tables) error:", e)

        return table1_path, table2_path
    
    def generate_graphs(self):
        """Generate graphs for IoU, Precision, AUC vs Epoch"""
        print("\n" + "="*60)
        print("Generating Graphs")
        print("="*60)
        
        epochs = sorted(self.evaluation_results.keys())
        if not epochs:
            print("⚠️ No evaluation results to plot")
            return
        
        iou_values = [self.evaluation_results[e]['iou'] for e in epochs]
        precision_values = [self.evaluation_results[e]['precision'] for e in epochs]
        auc_values = [self.evaluation_results[e]['auc'] for e in epochs]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # IoU plot
        axes[0].plot(epochs, iou_values, marker='o', linewidth=2, markersize=6, color='blue')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('IoU', fontsize=12)
        axes[0].set_title(f'IoU vs Epoch', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(epochs)
        
        # Precision plot
        axes[1].plot(epochs, precision_values, marker='s', linewidth=2, markersize=6, color='green')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Precision', fontsize=12)
        axes[1].set_title(f'Precision vs Epoch', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(epochs)
        
        # AUC plot
        axes[2].plot(epochs, auc_values, marker='^', linewidth=2, markersize=6, color='red')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('AUC', fontsize=12)
        axes[2].set_title(f'AUC vs Epoch', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(epochs)
        
        plt.tight_layout()
        graph_path = os.path.join(self.results_dir, f"metrics_vs_epoch.png")
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Graph saved to: {graph_path}")
        
        # Combined graph
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(epochs, iou_values, marker='o', linewidth=2, markersize=6, label='IoU', color='blue')
        ax.plot(epochs, precision_values, marker='s', linewidth=2, markersize=6, label='Precision', color='green')
        ax.plot(epochs, auc_values, marker='^', linewidth=2, markersize=6, label='AUC', color='red')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title(f'Evaluation Metrics vs Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_xticks(epochs)
        
        plt.tight_layout()
        combined_graph_path = os.path.join(self.results_dir, f"combined_metrics.png")
        plt.savefig(combined_graph_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Combined graph saved to: {combined_graph_path}")

        # Optional: upload graphs to the same repo/phase path
        try:
            token = HfFolder.get_token()
            if token:
                for p in [graph_path, combined_graph_path]:
                    try:
                        upload_file(
                            path_or_fileobj=p,
                            path_in_repo=f"{self.upload_prefix}/{os.path.basename(p)}",
                            repo_id=self.repo_id,
                            repo_type="model",
                            token=token,
                        )
                        print(f"⬆️ Uploaded graph to Hugging Face: {self.repo_id}/{self.upload_prefix}/{os.path.basename(p)}")
                    except Exception as e:
                        print("⚠️ Failed uploading graph:", e)
        except Exception as e:
            print("⚠️ Upload block (graphs) error:", e)

        return graph_path, combined_graph_path
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print(f"\n{'='*80}")
        print(f"Assignment 4: Test and Evaluation")
        print(f"{'='*80}\n")
        
        # Step 1: List checkpoints
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            print("⚠️ No checkpoints found. Exiting.")
            return
        
        # Step 2: Download and run inference for each checkpoint
        for epoch, checkpoint_path_in_repo in checkpoints:
            # Download checkpoint
            checkpoint_path = self.download_checkpoint(checkpoint_path_in_repo, epoch)
            if checkpoint_path is None:
                continue
            
            # Run inference
            self.run_inference_for_checkpoint(checkpoint_path, epoch)
            
            # Evaluate
            self.evaluate_checkpoint(epoch, checkpoint_path)
            
            # Clean up checkpoint file (don't save locally for Kaggle)
            try:
                if os.path.exists(checkpoint_path):
                    # Only delete if it's in temp dir (not cached by hf_hub)
                    if self.temp_dir in checkpoint_path:
                        os.remove(checkpoint_path)
            except:
                pass
        
        # Step 3: Generate tables
        self.generate_tables()
        
        # Step 4: Generate graphs
        self.generate_graphs()
        
        # Save summary JSON
        summary = {
            'phase': self.phase_name,
            'evaluation_results': self.evaluation_results,
            'inference_rates': self.inference_rates,
            'per_sequence_metrics': self.per_sequence_metrics
        }
        summary_path = os.path.join(self.results_dir, f"summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 Summary saved to: {summary_path}")

        # Optional: upload summary JSON
        try:
            token = HfFolder.get_token()
            if token:
                upload_file(
                    path_or_fileobj=summary_path,
                    path_in_repo=f"{self.upload_prefix}/{os.path.basename(summary_path)}",
                    repo_id=self.repo_id,
                    repo_type="model",
                    token=token,
                )
                print(f"⬆️ Uploaded summary to Hugging Face: {self.repo_id}/{self.upload_prefix}/{os.path.basename(summary_path)}")
        except Exception as e:
            print("⚠️ Failed uploading summary JSON:", e)
        
        print(f"\n✅ Assignment 4 evaluation complete for {self.phase_name}!")
        print(f"📁 Results directory: {self.results_dir}")


def main():
    parser = argparse.ArgumentParser(description='Assignment 4: Test and Evaluation')
    parser.add_argument('--repo_id', type=str, required=True,
                        help='Hugging Face repository ID (e.g., USER/seqtrack-checkpoints)')
    parser.add_argument('--phase', type=str, required=True,
                        help='Phase name (e.g., phase_1, phase_2)')
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='Starting epoch (default: 1)')
    parser.add_argument('--end_epoch', type=int, default=10,
                        help='Ending epoch (default: 10)')
    parser.add_argument('--dataset_name', type=str, default='lasot',
                        help='Dataset name (default: lasot)')
    parser.add_argument('--temp_dir', type=str, default=None,
                        help='Temporary directory for downloads (default: system temp)')
    parser.add_argument('--upload_prefix', type=str, default="member_10_abdelrahman_ahmed/test",
                        help='Subfolder path inside repo to store test artifacts (tables, graphs)')
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help='Resume from this epoch (skip earlier checkpoints)')
    
    args = parser.parse_args()
    
    evaluator = Assignment4Evaluator(
        repo_id=args.repo_id,
        phase_name=args.phase,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
        dataset_name=args.dataset_name,
        temp_dir=args.temp_dir,
        upload_prefix=args.upload_prefix,
        resume_epoch=args.resume_epoch
    )
    
    evaluator.run_full_evaluation()


if __name__ == '__main__':
    main()

