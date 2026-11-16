# Multi-Label Knee Abnormality Detection from Multi-Plane MRI Scans

**Team KNOCK** — Department of Computer Science & Engineering, St Aloysius (Deemed to be University), Mangaluru  
**Course:** Fundamentals of Image Processing (`256NUPC273`) | **Guide:** Dr. Aldrin Claytus Vaz  
**Competition:** [RSNA Knee Abnormality Detection (Kaggle)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)  
**Final Score:** `0.965` Public Macro ROC-AUC / `0.953` Private Macro ROC-AUC

## Team Members
* **Preemal Simona Pinto** (`25190139`) — Problem Formulation & Hybrid Model Architecture
* **Sanvi B K** (`25190145`) — Clinical Scope & 12-Pathology Classification
* **Punarvi J** (`25190156`) — Medical DICOM & Multi-Plane Preprocessing
* **Gautam Naveen Rao** (`25190113`) — Dual-GPU Inference & Kaggle Leaderboard Evaluation
* **Keora Pais** (`25190118`) — Clinical Triage Analysis & Validation Findings

## Project Links
* **Kaggle Inference Notebook (`RSNA KNEE - Version 1`):** (https://www.kaggle.com/code/preemalsimonapinto/rsna-knee)
* **Video Presentation:** (https://drive.google.com/file/d/15lkF_rWgcacIco9KX3OryM11uxVqQka0/view?usp=drive_link)

---

## 1. Hardware & Compute Environment (B2)
* **GPUs:** 2× NVIDIA Tesla T4 (16 GB VRAM each, 32 GB total VRAM, CUDA 12.x)
* **CPU:** Intel Xeon CPU @ 2.00 GHz (4 vCPUs / 2 physical cores)
* **System RAM:** 30 GB DDR4
* **Storage:** 73 GB SSD scratch disk (`/kaggle/working`)

## 2. Operating System & Platform
* **OS / Runtime:** Ubuntu 22.04 LTS (Kaggle Docker Container Environment)
* **Python Version:** Python `3.12`
* **Deep Learning Framework:** PyTorch `2.10.0` + `timm`

## 3. Third-Party Software & Offline Installation
Because Kaggle code competitions disable internet access during evaluation, lossless medical DICOM decompression packages (`pylibjpeg`, `libjpeg`, `openjpeg`) are installed via pre-packaged offline `.whl` files:
```bash
pip install -r requirements.txt
# For offline Kaggle runtime:
pip install --no-index --find-links=/kaggle/input/rsna-offline-wheels pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
