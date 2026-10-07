# MetaDerm-XA
## Field-Wise Metadata Conditioning and Image-Fixed Intervention Analysis for CNN-Based Skin Lesion Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange.svg)](https://pytorch.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/Dataset%20License-CC--BY--NC-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Overview

MetaDerm-XA is a **multi-modal skin lesion classification system** that fuses dermoscopic images with patient clinical metadata (age, sex, anatomical site) using a novel **field-token cross-attention mechanism**. Each metadata variable is encoded as a separate learnable token, enabling spatially selective, field-specific conditioning of CNN feature maps.

The system includes a **counterfactual XAI engine** that answers: *"How would the predicted score change if the patient were older, or if the lesion were on a different site?"*

### Key Contributions

1. **Field-Token Cross-Attention** — 4 separate metadata tokens (age, sex, site, missingness) enable spatially varying, field-selective image conditioning
2. **Image-Fixed Intervention Protocol** — full test-set counterfactual sweep (1,327 melanoma cases) with population-mean curves and 95% confidence bands
3. **Cross-Dataset Evaluation** — trained on ISIC 2019, externally validated on ISIC 2020
4. **4-Model Ablation** with DeLong pairwise tests and calibration analysis

---

## Results

### ISIC 2019 Internal Test (8,238 images)

| Model | AUROC | 95% CI | Recall | ECE ↓ |
|-------|-------|--------|--------|-------|
| Baseline (vision-only) | 0.880 | [0.870–0.890] | 0.802 | 0.164 |
| Cross-Attn (1-token) | 0.883 | [0.873–0.892] | 0.769 | 0.133 |
| Concat Fusion | 0.895 | [0.886–0.904] | 0.729 | 0.237 |
| **Field-Token (ours)** | **0.896** | **[0.887–0.904]** | **0.855** | 0.186 |

> Field-token significantly outperforms baseline (p<0.001) and 1-token cross-attention (p<0.001) via DeLong test.

### ISIC 2020 External Validation (33,126 images)

| Model | AUROC | AUROC Drop |
|-------|-------|------------|
| Concat Fusion | **0.796** | −11.1% |
| Cross-Attn (1-token) | 0.795 | −10.0% |
| Baseline | 0.763 | −13.3% |
| Field-Token (ours) | 0.746 | −16.7% |

---

## Datasets

**Training:** ISIC 2019 Challenge (25,331 images, 8 diagnostic categories)
- Source: [ISIC Challenge](https://challenge.isic-archive.com/data/#2019)
- Binary mapping: MEL → malignant (1), all others → benign (0)
- Patient-level stratified split: 20,160 train / 5,171 val / 8,238 test (via `lesion_id`)

**External Validation:** SIIM-ISIC 2020 (33,126 images)
- Source: [ISIC Archive](https://doi.org/10.34970/2020-DS01)
- License: CC BY-NC 4.0

---

## Project Structure

```
skin-lesion-project/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── isic2019/               # Primary experiment configs
│   │   ├── baseline.yaml
│   │   ├── concat_fusion.yaml
│   │   ├── cross_attention.yaml
│   │   └── field_token.yaml
│   ├── external_eval/          # ISIC 2020 external validation
│   │   ├── baseline.yaml
│   │   ├── concat_fusion.yaml
│   │   ├── cross_attention.yaml
│   │   └── field_token.yaml
│   └── archive/                # Legacy ISIC 2020-only experiments
│
├── src/
│   ├── train.py                # Config-driven training loop
│   ├── evaluate.py             # Full evaluation + bootstrap CIs
│   ├── calibration_analysis.py # ECE, Brier, DeLong pairwise tests
│   ├── counterfactual_sweep.py # Full test-set counterfactual analysis
│   ├── data/                   # Dataset, transforms, loaders
│   ├── models/                 # All architectures
│   │   ├── baseline.py
│   │   ├── concat_fusion.py
│   │   ├── cross_attention_fusion.py
│   │   ├── field_token_model.py       # ← Novel architecture
│   │   └── field_token_metadata_encoder.py
│   ├── utils/                  # Metrics, visualization, seed
│   └── xai/                    # Counterfactual engine, Grad-CAM
│
├── checkpoints/                # 8 trained models (gitignored)
├── data/
│   ├── raw/                    # ISIC 2019 + 2020 images (gitignored)
│   ├── processed_isic2019/     # Primary processed splits
│   └── processed/              # ISIC 2020 processed data
│
├── reports/
│   ├── isic2019_internal/      # Internal test metrics + plots
│   ├── isic2020_external/      # External validation results
│   ├── calibration/            # Reliability diagrams, ECE/Brier
│   ├── counterfactual/         # Full sweep: age/sex/site curves
│   └── xai_examples/           # Grad-CAM + counterfactual panels
│
├── runs/                       # TensorBoard logs
├── app/                        # Streamlit demo
├── notebooks/                  # EDA notebook
├── scripts/                    # Utility scripts
└── tests/                      # Unit tests (pytest)
```

---

## Quick Start

### 1. Install
```bash
conda create -n skin-lesion python=3.10
conda activate skin-lesion
pip install -r requirements.txt
```

### 2. Train (ISIC 2019)
```bash
python src/train.py --config configs/isic2019/field_token.yaml
```

### 3. Evaluate
```bash
python src/evaluate.py \
    --config configs/isic2019/field_token.yaml \
    --checkpoint checkpoints/field_token_isic2019_best.pth
```

### 4. Counterfactual Analysis
```bash
python src/counterfactual_sweep.py \
    --config configs/isic2019/field_token.yaml \
    --checkpoint checkpoints/field_token_isic2019_best.pth \
    --target_class melanoma
```

### 5. Explainability (Grad-CAM + Counterfactuals)
```bash
python src/xai/explain.py \
    --config configs/isic2019/field_token.yaml \
    --checkpoint checkpoints/field_token_isic2019_best.pth \
    --n_cases 10 --melanoma_only
```

### 6. Calibration & DeLong Tests
```bash
python src/calibration_analysis.py
```

### 7. Demo App
```bash
streamlit run app/streamlit_app.py
```

> ⚠️ **Disclaimer:** Research prototype only. Not a medical device. Not for diagnostic use.

---

## Architecture

```
Image (224×224) ──► EfficientNet-B2 ──► Spatial Feature Tokens [B, H*W, C]
                                                          │
                    ┌─ age  ──► MLP  ──► token_age  ──┐   │
Metadata ──────────►├─ sex  ──► OHE  ──► token_sex  ──┤── M = [B, 4, D]
                    ├─ site ──► Emb  ──► token_site ──┤   │
                    └─ miss ──► MLP  ──► token_miss ──┘   │
                                                          │
                         Cross-Attention (Q=image, K/V=M) │
                         Softmax over 4 metadata fields   │
                                                          │
                         Global Average Pool ──► Classifier ──► P(malignant)
```

---

## Reproducing Results

All experiments use seed `42`. To reproduce:
1. Use patient-level splits saved in `data/processed_isic2019/`
2. Use configs from `configs/isic2019/`
3. PyTorch 2.2.2, CUDA 12.x

---

## License

Code: MIT License
Dataset: CC BY-NC 4.0 (ISIC 2019/2020) — non-commercial research only
