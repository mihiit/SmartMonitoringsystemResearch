# Behavioural Robustness as an Evaluation Criterion for Temporal Disease Risk Models

**IEEE JBHI — Manuscript Under Review**  
Mihit Nanda† and Hannah Nagpall‡

---

## Overview

This repository provides fully reproducible code for the paper:

> *"Behavioural Robustness as an Evaluation Criterion for Temporal Disease Risk Models under Lifestyle Intervention"*

The primary contribution is the **BR (Behavioural Robustness) evaluation framework**, not a novel model architecture. The hybrid LSTM–Transformer is the evaluation vehicle.

---

## Repository Structure

```
br_paper/
├── data/
│   ├── download_data.py          # Instructions & auto-download for public datasets
│   └── synthetic_cohorts.py      # Pima & NHANES-calibrated synthetic cohort generation
├── models/
│   ├── lstm_model.py             # 2-layer LSTM with MC-Dropout
│   ├── transformer_model.py      # Transformer with optional identity embedding
│   └── baselines.py              # LR, XGBoost, RETAIN-style, Med-BERT-style, std-LSTM
├── evaluation/
│   ├── br_metric.py              # BR, BRvelocity, Wilcoxon test, sensitivity analysis
│   ├── calibration.py            # ECE across 10 bins, reliability diagram
│   └── uncertainty.py            # MC-Dropout, CoV
├── utils/
│   ├── data_loader.py            # Dataset loaders for all four datasets
│   ├── intervention.py           # Lifestyle intervention simulation (Eq. 2)
│   └── shap_attribution.py       # SHAP feature attribution & delta computation
├── figures/
│   └── plot_all.py               # Reproduces all paper figures (Fig 2–9)
├── run_all.py                    # Master script: runs full pipeline end-to-end
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Role | N | Source |
|---|---|---|---|
| Pima Indians Diabetes | Tier-1: Metric validation | 768 | UCI ML Repository (CC BY 4.0) |
| NHANES 2015–2018 | Tier-1: Metric validation | 2,000 | CDC open data |
| UCI Diabetes CGM | Tier-2: Exploratory real | 70 | UCI ML Repository (public domain) |
| Diabetes 130-US Hospitals | Tier-2: Primary real | 5,246 | UCI ML Repository (CC BY 4.0) |

All datasets are publicly available and de-identified. No IRB required (45 CFR 46.104(d)(4)).

### Auto-download
```bash
python data/download_data.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

Tested on Python 3.9+.

---

## Reproducing All Results

```bash
python run_all.py
```

This runs:
1. Synthetic cohort generation (Pima & NHANES-calibrated)
2. Training all model variants (LSTM, Transformer, Transformer No-Pers.) and 5 baselines
3. BR metric computation on all four datasets
4. Wilcoxon signed-rank tests
5. MC-Dropout uncertainty estimation
6. SHAP attribution
7. All figures and tables (saved to `results/`)

---

## Key Results (reproduced)

| Dataset | AUC | BR | Wilcoxon p |
|---|---|---|---|
| Pima (tier-1) | 0.755 | 0.0069 | 1.6×10⁻¹¹ |
| NHANES (tier-1) | 0.677 | 0.0036 | 6.0×10⁻¹⁰ |
| UCI (tier-2, exploratory) | 0.844 | 0.0217 | 1.8×10⁻⁸ |
| Diabetes-130 (tier-2, primary) | 0.612 | 0.0217 | 1.7×10⁻²⁸⁴ |

---

## Ethics

This study uses exclusively publicly available, de-identified datasets. All data processing followed FAIR principles. The authors declare no conflict of interest.

---

## Citation

```bibtex
@article{nanda2024behavioural,
  title={Behavioural Robustness as an Evaluation Criterion for Temporal Disease Risk Models under Lifestyle Intervention},
  author={Nanda, Mihit and Nagpall, Hannah},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  note={Under Review}
}
```

---

## Code Repository

https://github.com/mihiit/SmartMonitoringsystemResearch
