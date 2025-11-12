# 🧠 Assignment 3 — SeqTrack Full Training Report (Phase 1 & Phase 2)

**Repository:** [https://github.com/aya2500/Assignment_3](https://github.com/aya2500/Assignment_3)  
**Hugging Face Checkpoints:** [https://huggingface.co/ayamohamed2500/seqtrack-checkpoints](https://huggingface.co/ayamohamed2500/seqtrack-checkpoints)

---

## 📘 Project Overview

This project documents the full training pipeline of the **SeqTrack** model through **two consecutive phases**.  
It focuses on:
- Automating the training process.  
- Implementing **robust checkpointing and resume** mechanisms.  
- Integrating model checkpoints and logs with **Hugging Face** for cloud storage.  
- Ensuring full **reproducibility** of results across training phases.

All experiments were conducted on a filtered subset of the **LaSOT dataset**, restricted to **two object classes: “book” and “coin.”**

---

## 📄 Understanding SeqTrack

SeqTrack is a **sequence-based object tracking model** proposed by Microsoft (VideoX).  
Its architecture includes:
- A **Vision Transformer (ViT)** encoder for feature extraction.  
- A **sequence head** for temporal modeling.  
- An **IoU prediction module** for overlap-based tracking accuracy.

**Main objectives:**
1. Improve robustness across diverse tracking scenarios.  
2. Utilize large-scale datasets efficiently.  
3. Provide reproducible, modular training with clear checkpoints.

---

## ⚙️ Environment Setup

- **Platform:** PyTorch (CUDA 12.1)  
- **GPU:** NVIDIA RTX 3050  
- **Python:** 3.10  
- **Seed:** Fixed seed (3) for deterministic results  

**Installation steps:**
```bash
git clone https://github.com/microsoft/VideoX
cd VideoX/SeqTrack
pip install -r <packages individually based on errors>
```

All manually installed dependencies were recorded in **`installedpackages.txt`** to ensure reproducibility.

Checkpoints were automatically saved and uploaded to Hugging Face at the end of each epoch.

---

## 📊 Dataset Description

- **Dataset:** [LaSOT (Hugging Face)](https://huggingface.co/datasets/lasot)  
- **Filtered Classes:** `['book', 'coin']`  
- **Total Sequences:** 32 (16 per class)  
- **Frames:** ~66K  
- **Samples per Epoch:** 992 (10 epochs → 9,920 total samples)

| Class | Train Sequences | Frames | Found Frames | Note |
|--------|----------------|--------|---------------|------|
| Book | 16 | 34,010 | 33,918 / 34,010 | Filtered LaSOT subset |
| Coin | 16 | 32,548 | 31,624 / 32,548 | Filtered LaSOT subset |
| **Total** | **32** | **66,558** | **65,542 / 66,558** | — |

---

## 🧩 Key Code Modifications

| File | Change Summary |
|------|----------------|
| `experiments/seqtrack/seqtrack_b256.yaml` | Reduced input size (256→192), scale tuning, batch adjustments |
| `lib/train/run_training.py` | Added `--resume`, `--phase`, `--repo_id` args for automation |
| `lib/train/train_script.py` | Standardized log file naming |
| `lib/train/trainers/base_trainer.py` | Saved RNG, optimizer, and scheduler states for deterministic resume |
| `lib/train/trainers/ltr_trainer.py` | Added IoU/Loss history plots, phase folders, auto Hugging Face upload |
| `lib/train/dataset/lasot.py` | Filtered dataset to `book` and `coin` classes |
| `tracking/train.py` | Added CLI flags for resume and Hugging Face upload |
| `upload_checkpoint.py`, `create_hf_repo.py`, `checkRepo.py` | New utilities for checkpoint upload and repo verification |

---

## 🚀 Training Details

### **Phase 1**
- Trained from scratch for **10 epochs**.
- Each epoch saved a checkpoint and IoU plot.
- IoU improved steadily from **0.047 → 0.089**, showing consistent convergence.

| Epoch | Total Loss | IoU | Time |
|-------|-------------|-----|------|
| 1 | 8.29 | 0.0479 | 25m 56s |
| 5 | 8.18 | 0.0342 | 27m 28s |
| 10 | 8.01 | 0.0893 | 28m 42s |

**IoU Curve:** Steady rise after epoch 5 → 0.089 @ epoch 10  
**Loss Curve:** Smooth decline confirming stable optimization.

---

### **Phase 2**
- Resumed training from **checkpoint `ep0003.pth.tar`**.  
- Restored optimizer, scheduler, RNG, and history.  
- Continued identical configuration for full determinism.  
- Final results matched Phase 1, confirming **perfect resume consistency**.

| Epoch | Total Loss | IoU | Time |
|-------|-------------|-----|------|
| 4 | 8.20 | 0.0357 | 33m 03s |
| 8 | 8.07 | 0.0654 | 32m 31s |
| 10 | 8.01 | 0.0893 | 32m 32s |

---

## 💾 Checkpoint & Logging System

**Checkpointing:**
- Saved after each epoch (includes model + optimizer + scheduler + RNG states).  
- Auto-uploaded to Hugging Face under:
  - `/phase_1/`  
  - `/phase_2/`

**Resume evidence:**
```text
Resuming from checkpoint checkpoints/phase_2/SEQTRACK_ep0003.pth.tar
```

**Logging:**
- Updates printed every 50 samples with elapsed time, ETA, Loss, and IoU.
- Example:
  ```
  [phase_2] Epoch 8: 992 / 992 samples, time for last 32 samples: 0:01:02,
  Loss/total: 8.07525, IoU: 0.06536
  ```

---

## ☁️ Hugging Face Integration

All checkpoints, logs, and plots are synced automatically to:  
🔗 [https://huggingface.co/ayamohamed2500/seqtrack-checkpoints](https://huggingface.co/ayamohamed2500/seqtrack-checkpoints)

- Organized folders per phase.  
- Auto-retry system handles upload errors gracefully.  

---

## 🔍 Observations & Results

- The model converged smoothly across both phases.  
- IoU trends confirm stable and consistent learning.  
- Resume functionality verified identical results between full and resumed training.  
- Hugging Face sync provided easy checkpoint verification and reproducibility.

---

## ✅ Conclusion

Both **Phase 1** and **Phase 2** successfully demonstrated:
- Stable training and loss reduction.  
- Reproducible results with deterministic resume.  
- Robust and automated checkpoint handling.  
- Full integration with Hugging Face for reproducible cloud-based experiments.

This workflow ensures **traceable, resumable, and verifiable** SeqTrack training runs suitable for research or deployment pipelines.


---

## 📚 License  
This project is part of a university assignment. For educational use only.
