# Robust Deepfake Detection Using Learning-Based Computer Vision Methods

Binary deepfake detection with a focus on building models that **generalize across datasets**. The goal isn't just high accuracy on FaceForensics++ — it's strong **zero-shot performance on unseen sources** like Celeb-DF, the property that actually matters for real-world content moderation.

**Course:** CS 668 — Analytics Capstone
**Instructor:** Dr. Krishna Bathula
**Status:** ✅ Complete

> 📌 **About this repository:** This is my copy of a **4-person team capstone**. The original team repository is **[abraraltaf92/deepfake-detection](https://github.com/abraraltaf92/deepfake-detection)**.
> **My role:** co-developer — one of the two engineers who built the end-to-end ML pipeline (data ingestion, preprocessing, model training, and cross-dataset evaluation).

---

## Team Members

| Member | Role |
|--------|------|
| Krishna Kirit Maniyar | Model development & ML pipeline (co-developer) |
| Abrar Altaf Lone | Model development & ML pipeline (co-developer) |
| Sajan Singh Shergill | Literature review & research |
| Muhammad Abdullah Zahid | Documentation & presentations |

---

## Approach

- **Train / validate / test** on FaceForensics++ (C23) — 6 manipulation types (Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures, + originals).
- **Zero-shot out-of-distribution test** on **Celeb-DF v2** (6,529 videos) — models never see this data during training.
- Compare **5 architectures + a soft-vote ensemble**, scored by **cross-dataset Celeb-DF AUC** and the **generalization gap** (Δgen = FF++ AUC − Celeb-DF AUC; lower = better).

Face crops (224×224) are extracted per frame; models are trained with class-weighted loss to handle the real/fake imbalance.

---

## Results

### Final Leaderboard (sorted by Celeb-DF AUC — the deployment-relevant metric)

| Rank | Model | FF++ Acc | FF++ F1 | FF++ AUC | Celeb-DF Acc | Celeb-DF F1 | Celeb-DF AUC | Δgen (AUC) |
|------|-------|----------|---------|----------|--------------|-------------|--------------|------------|
| 🥇 #1 | **Soft-Vote Ensemble** | 1.000 | 1.000 | 1.000 | 0.638 | 0.737 | **0.892** | **0.109** |
| #2 | ViT-B/16 | 0.978 | 0.987 | 0.999 | 0.621 | 0.722 | 0.884 | 0.116 |
| #3 | EfficientNet-B4 | 0.999 | 0.999 | 1.000 | 0.641 | 0.742 | 0.840 | 0.160 |
| #4 | ResNet-18 (baseline) | 0.999 | 0.999 | 1.000 | 0.524 | 0.626 | 0.828 | 0.172 |
| #5 | R3D-18 + RAFT (optical flow) | 0.884 | 0.926 | 0.961 | 0.446 | 0.530 | 0.802 | 0.159 |
| #6 | R3D-18 (3D-CNN) | 0.999 | 0.999 | 1.000 | 0.635 | 0.744 | 0.756 | 0.244 |

*Ensemble = equal-weight probability averaging over EfficientNet-B4, R3D-18, ViT-B/16, and R3D-18+RAFT.*

### Key Findings

- **Every model is near-perfect in-dataset (FF++ AUC ≈ 1.0) but drops sharply zero-shot on Celeb-DF** — the central, well-known challenge in deepfake detection. In-dataset accuracy alone is misleading; cross-dataset AUC is what counts.
- **The soft-vote ensemble generalizes best** (Celeb-DF AUC **0.892**, smallest gap **0.109**) — combining diverse architectures beats any single model out-of-distribution.
- **Among single models, ViT-B/16 wins** (Celeb-DF AUC 0.884, best generalization Δgen 0.116) — attention-based features transfer to unseen data better than 3D-CNNs here.
- **R3D-18 (3D-CNN) had top in-dataset scores but the worst generalization** (Δgen 0.244) — a textbook case of overfitting to dataset-specific artifacts.
- **Adding RAFT optical flow** to R3D-18 lowered in-dataset accuracy and did not improve cross-dataset AUC — temporal motion cues didn't transfer here.

---

## Models

| Notebook | Model | Type |
|----------|-------|------|
| `04_model_baseline` | ResNet-18 | 2D CNN (baseline) |
| `05_model_advanced` | EfficientNet-B4 | 2D CNN |
| `06_advanced_model_r3d18` | R3D-18 | 3D CNN (spatiotemporal) |
| `07_advanced_model_vit` | ViT-B/16 | Vision Transformer |
| `08_advanced_model_raft` | R3D-18 + RAFT | 3D CNN + optical flow |
| `09_evaluation` | Soft-vote ensemble | Combines all of the above |

---

## Datasets

| Dataset | Use | Link |
|---------|-----|------|
| FaceForensics++ (C23) | Train / validation / test | [link](https://github.com/ondyari/FaceForensics) |
| Celeb-DF v2 | Zero-shot cross-dataset test (6,529 videos) | [link](https://github.com/yuezunli/celeb-deepfakeforensics) |
| DFDC | Reference | [link](https://ai.meta.com/datasets/dfdc/) |

---

## Repository Structure

```
deepfake-detection/
├── README.md                          # This file
├── CONTRIBUTING.md                    # Team collaboration workflow
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Excludes data, models, credentials
├── configs/
│   └── paths.py                       # Centralized path configuration
├── notebooks/
│   ├── 01_data_ingestion.ipynb        # Dataset discovery, indexing, splitting
│   ├── 02_eda.ipynb                   # Exploratory data analysis
│   ├── 03_preprocessing.ipynb         # Frame extraction, face detection, tensor prep
│   ├── 04_model_baseline.ipynb        # ResNet-18 baseline
│   ├── 05_model_advanced.ipynb        # EfficientNet-B4
│   ├── 06_advanced_model_r3d18.ipynb  # R3D-18 (3D CNN)
│   ├── 07_advanced_model_vit.ipynb    # ViT-B/16
│   ├── 08_advanced_model_raft.ipynb   # R3D-18 + RAFT optical flow
│   └── 09_evaluation.ipynb            # Cross-dataset evaluation, ensemble, leaderboard
└── docs/
    └── references.md                  # Research papers and resources
```

---

## Quick Start

The notebooks are designed for **Google Colab** (GPU) with data on Google Drive.

1. **Clone the repo** into your Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !git clone https://github.com/krishnamaniyar2209/deepfake-detection.git /content/drive/MyDrive/deepfake-detection
   ```
2. **Download datasets** into `data/raw/` (see the Datasets table). Model weights and data stay on Drive — they are **not** committed (see `.gitignore`).
3. **Set `REPO_ROOT`** in `configs/paths.py` if your Drive path differs.
4. **Run the notebooks in order** `01 → 09`. `09_evaluation.ipynb` reproduces the leaderboard above.

---

## Tech Stack

`Python` · `PyTorch` · `torchvision` · `OpenCV` · `facenet-pytorch` · `scikit-learn` · `pandas` · `NumPy`

---

## Key Resources

- [FaceForensics++](https://github.com/ondyari/FaceForensics) · [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics) · [DFDC](https://ai.meta.com/datasets/dfdc/)
- [Research papers & references](docs/references.md)

---

*CS 668 Analytics Capstone — Pace University. See [CONTRIBUTING.md](CONTRIBUTING.md) for the team's collaboration workflow.*
