# MediEdge 
> **Efficient Deep Learning for Medical Image Diagnosis on Resource-Constrained Hardware**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MedMNIST](https://img.shields.io/badge/dataset-MedMNIST-green.svg)](https://medmnist.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Motivation

Medical AI models are accurate but heavy. They assume a GPU in the room. The clinics that need them most, rural and primary-care settings in Pakistan and similar low-resource countries, run on basic CPUs and often no internet. MediEdge answers one practical question:

> **Can a lightweight model match a standard model's diagnostic accuracy at a fraction of the size and latency cost?**

---

## Dataset

We use [MedMNIST v2](https://medmnist.com/) — the standard lightweight medical imaging benchmark.

| Dataset | Task | Classes | Train / Val / Test |
|---|---|---|---|
| **DermaMNIST** | Skin lesion classification | 7 | 7,007 / 1,003 / 2,005 |
| **PathMNIST** | Colon histology | 9 | 89,996 / 10,004 / 7,180 |

Images are 28×28 RGB — fast to train on a laptop CPU, clinically meaningful, and citable.

---

## Project Structure

```
mediedge/
├── configs/                  # YAML experiment configs
│   └── default.yaml
├── data/                     # Dataset download & loaders
│   ├── __init__.py
│   └── dataloader.py
├── models/                   # Model definitions
│   ├── __init__.py
│   ├── cnn_baseline.py       # Small custom CNN
│   ├── resnet18.py           # ResNet-18 (baseline)
│   ├── mobilenetv2.py        # MobileNetV2 (efficient)
│   └── efficientnet_lite.py  # EfficientNet-Lite (efficient)
├── compression/              # Model compression techniques
│   ├── __init__.py
│   ├── quantization.py       # INT8 post-training quantization
│   ├── pruning.py            # Magnitude pruning
│   └── distillation.py      # Knowledge distillation
├── training/                 # Training & evaluation loops
│   ├── __init__.py
│   ├── trainer.py
│   └── metrics.py
├── experiments/              # Saved model weights & logs (git-ignored)
│   └── .gitkeep
├── results/                  # Tables, figures, reports
│   └── .gitkeep
├── notebooks/                # Exploratory analysis
│   └── 01_eda.ipynb
├── scripts/                  # Convenience shell scripts
│   ├── run_baseline.py
│   ├── run_efficient.py
│   └── run_compression.py
├── tests/                    # Unit tests
│   └── test_models.py
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## Experiment Plan

### Phase 1 — Setup (Days 1–2)
- [x] Project scaffold & clean README
- [x] Install dependencies
- [x] Download DermaMNIST / PathMNIST via MedMNIST API

### Phase 2 — Baseline (Week 1)
- [x] Train small CNN → record accuracy, AUC, size, CPU latency
- [x] Train ResNet-18 → same metrics

### Phase 3 — Efficient Models (Week 2)
- [x] Train MobileNetV2
- [x] Train EfficientNet-Lite

### Phase 4 — Compression (Weeks 2–3)
- [x] INT8 post-training quantization
- [x] Magnitude pruning (20% sparsity)
- [x] Knowledge distillation (teacher=ResNet-18, student=MobileNetV2)
- [x] Combined: distill + prune + quantize

### Phase 5 — Report (Week 3)
- [x] Results table & ablation analysis
- [x] Accuracy-vs-size tradeoff curve (`results/figures/fig1_accuracy_vs_size.png`)
- [x] Latency comparison bar chart (`results/figures/fig2_latency_comparison.png`)
- [x] Confusion matrix for best models (`results/figures/fig3_confusion_matrix.png`)
- [x] Complete Proposal & Report document (`results/MediEdge_Final_Report.md`)

---

## Benchmark Results (DermaMNIST Test Set)

| Model | Compression / Stage | Accuracy (%) | Macro AUC | Size (MB) | CPU Latency (ms/img) | Size vs. ResNet-18 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **ResNet-18 (Baseline)** | Standard Reference (Float32) | **77.91%** | **0.9351** | **42.70 MB** | **22.82 ms** | 1.0× (Reference) |
| **EfficientNet-Lite** | Efficient CNN (Float32) | 74.86% | 0.9068 | 15.61 MB | 31.29 ms | 2.7× smaller |
| **Small CNN (Floor)** | 3-Block Custom CNN | **76.51%** | **0.9262** | **0.37 MB** | **1.83 ms** | **116.0× smaller** |
| **MobileNetV2 (Base)** | Efficient Edge Backbone | 73.07% | 0.9043 | 8.75 MB | 16.75 ms | 4.9× smaller |
| **MobileNetV2 + Distill** | Knowledge Distillation | **74.71%** | **0.9179** | 8.75 MB | 16.96 ms | 4.9× smaller |
| **MobileNetV2 + INT8** | FX Graph Mode PTQ | 73.22% | 0.8977 | **2.53 MB** | 16.64 ms | **16.9× smaller** |
| **MobileNetV2 + Prune** | 20% L1 Sparsity | 72.37% | 0.8989 | 8.75 MB | 25.75 ms | 4.9× smaller |
| **MobileNetV2 Combined** | **Distill + 20% Prune + INT8** | **73.07%** | **0.9078** | **2.53 MB** | 29.65 ms | **16.9× smaller** |

**Headline Finding:** The compressed model operates within 5 percentage points (4.84 points: 73.07% vs. 77.91%) of ResNet-18 at **16.9× smaller size**, matching the uncompressed MobileNetV2 baseline with zero accuracy drop.

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/adnaan512/mediedge.git
cd mediedge
pip install -r requirements.txt

# 2. Download dataset (auto via MedMNIST)
python -c "from data.dataloader import get_loaders; get_loaders('dermamnist')"

# 3. Train baseline models
python scripts/run_baseline.py --dataset dermamnist --model cnn
python scripts/run_baseline.py --dataset dermamnist --model resnet18

# 4. Train efficient models
python scripts/run_efficient.py --dataset dermamnist --model mobilenetv2
python scripts/run_efficient.py --dataset dermamnist --model efficientnet_lite

# 5. Apply compression
python scripts/run_compression.py --model mobilenetv2 --method int8
python scripts/run_compression.py --model mobilenetv2 --method prune
python scripts/run_compression.py --model mobilenetv2 --method distill
python scripts/run_compression.py --model mobilenetv2 --method combined
```

---

## Hardware Target

| Spec | Value |
|---|---|
| CPU | Any x86 (tested on Intel Core i5) |
| RAM | ≥ 4 GB |
| GPU | **Not required** |
| OS | Windows / Linux / macOS |

---

## Citation

```bibtex
@article{medmnistv2,
  title={MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
  author={Yang, Jiancheng and others},
  journal={Scientific Data},
  year={2023}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
