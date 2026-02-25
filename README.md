# HFENN: Driver Fatigue & Emotion Detection via Wearable Biosignals

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A dual-channel deep learning system for **real-time driver fatigue regression** and **emotion classification** using wearable pulse and blink signals, designed for V2X (Vehicle-to-Everything) safety integration.

---

## Overview

This repository contains the full implementation of:

- **HFENN** (Hierarchical Feature Extraction Neural Network) — a wavelet-enhanced regression model that continuously estimates a fatigue index (0–1) from blink envelope and pulse signals.
- **WCNN** (Wavelet Convolutional Neural Network) — an emotion recognition model (Normal / Negative) from pulse data.
- A **V2X integration layer** that maps detection outputs to standardised risk levels for autonomous vehicle communication.

### Key Results (HFENN Regression, Random Split)

| Metric | Value |
|--------|-------|
| R² | 0.866 |
| Pearson r | 0.932 |
| MAE | 0.082 |
| RMSE | 0.109 |
| Cross-subject R² std | 0.043 (n=13) |

---

## Repository Structure

```
.
├── notebooks/
│   ├── HFENN_regression_random_split.ipynb   # Main regression model (canonical)
│   ├── HFENN_enhaced.ipynb                   # Enhanced classification model
│   ├── HFENN_regression_sigmoid_single_test.ipynb
│   ├── HFENN_affine_calibration_exp.ipynb    # Affine post-hoc calibration
│   ├── HFENN_affine_calibration_exp_kalman.ipynb
│   ├── HFENN_affine_calibration_exp_quality.ipynb
│   ├── HFENN_temporal_features_exp.ipynb     # Temporal feature experiments
│   ├── HFENN_kalman_exp.ipynb                # Kalman filter experiments
│   └── HFENN_quality_gate_analysis.ipynb
│
├── src/
│   ├── HFENN_visualization_v2.py             # Publication-quality figure generation
│   ├── HFENN_ablation_study_v2.py            # Ablation study (model variants)
│   ├── HFENN_hyperparameter_sensitivity.py   # Hyperparameter sensitivity analysis
│   ├── hfenn_interface.py                    # Inference API wrapper
│   ├── v2x_integration_example.py            # V2X system integration demo
│   ├── generate_continuous_fatigue.py        # Continuous fatigue signal generation
│   ├── data.py                               # Data loading utilities
│   └── fix_wcnn_model_v2.py                  # WCNN model compatibility fix
│
├── integration/
│   ├── dnsr_integration.py                   # DNSR system integration
│   ├── enhanced_hfenn_integration.py         # Enhanced HFENN integration module
│   ├── demo_dnsr_integration.py              # Integration demo
│   ├── test_dnsr_integration.py              # Integration tests
│   ├── test_enhanced_hfenn_integration.py
│   └── requirements.txt
│
├── figures/                                  # Publication-ready figures (PNG/PDF/SVG)
│   ├── Figure1_Model_Performance.*           # Time-series, scatter, Bland-Altman
│   ├── Figure3_Preprocessing_Validation.*    # Signal preprocessing
│   ├── Figure4A_Wavelet_Decomposition.*      # Wavelet analysis
│   ├── Figure4BC_Importance_Analysis.*       # Band & feature importance
│   ├── Figure6_Ablation_Study.*              # Ablation results
│   ├── Figure7_Training_Stability.*          # Training curves
│   └── Figure8_Per_Participant_Performance.* # Cross-subject results
│
├── plot_data/                                # Intermediate data for figures
│   ├── per_participant_metrics.csv           # Per-subject R², RMSE, MAE
│   ├── plot_data_feature_importance.csv      # Permutation feature importance
│   ├── plot_data_band_importance.csv         # Frequency band importance
│   ├── ablation_results.csv                  # Ablation study summary
│   ├── ablation_study_results.csv
│   └── ablation_full_results.json
│
├── selected_feature_names.txt                # Final feature set (50 features)
├── training_history.csv                      # Training loss/metric curves
├── ablation_results.csv                      # Top-level ablation summary
├── .gitignore
├── README.md
└── LICENSE
```

---

## Model Architecture

### HFENN (Fatigue Regression)

```
Blink Signal  ──► Wavelet cD1–cD4 ──► Conv1D × 2 ──┐
                                                      ├──► Attention ──► Dense ──► Fatigue Index [0,1]
Pulse Signal  ──► Wavelet cD1–cD4 ──► Conv1D × 2 ──┘
                         +
              Handcrafted Features (50-dim):
              Time-domain, Frequency-domain, Wavelet stats,
              Nonlinear (Hilbert envelope), Differential
```

- **Dual-channel wavelet input**: 4-level DWT decomposition of both blink and pulse signals
- **Attention mechanism**: Cross-channel attention for adaptive feature fusion
- **Training**: Random split (80/20), `random_state=42`, 100 epochs with early stopping
- **Best checkpoint**: saved at epoch 24 (val_loss = 0.0634)

### WCNN (Emotion Classification)

- Wavelet-based CNN for binary emotion classification (Normal / Negative)
- Input: 30-second pulse segments
- Output: softmax probability over 2 classes

---

## Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow>=2.10 numpy scipy pandas scikit-learn matplotlib seaborn pywavelets
```

Or use the integration requirements:

```bash
pip install -r integration/requirements.txt
```

### 2. Run Inference

```python
from hfenn_interface import HFENNInterface

interface = HFENNInterface(
    model_path="HFENN_Regression_RandomSplit_best.keras",
    scaler_path="feature_scaler.pkl",
    feature_names_path="selected_feature_names.txt"
)

# blink_signal, pulse_signal: numpy arrays of shape (N,)
fatigue_index = interface.predict(blink_signal, pulse_signal)
print(f"Fatigue index: {fatigue_index:.3f}")
```

### 3. Reproduce Figures

```bash
python src/HFENN_visualization_v2.py
```

Figures are saved to `figures/` and intermediate data to `plot_data/`.

### 4. Run Ablation Study

```bash
python src/HFENN_ablation_study_v2.py
```

### 5. V2X Integration Demo

```bash
python v2x_integration_example.py
```

---

## Data

Raw data is **not included** in this repository (participant privacy). The dataset consists of:

- **13 participants**, 30-min driving sessions per participant
- **Sensors**: smart glasses (blink IR), wristband (photoplethysmography pulse)
- **Labels**: continuous fatigue index derived from reaction-time tasks, thresholded at 0.35 / 0.75 for 3-class discretisation

For access to anonymised data, please contact the authors.

---

## Feature Importance (Top 5 — Permutation ΔRMSE)

| Rank | Feature | ΔRMSE | Contribution |
|------|---------|-------|-------------|
| 1 | Fatigue_Envelope_Mean | 0.02896 | 5.8% |
| 2 | Pulse_PSD_Max | 0.02749 | 5.5% |
| 3 | Fatigue_Wavelet_cD2_Std | 0.02459 | 4.9% |
| 4 | Fatigue_Median | 0.02381 | 4.7% |
| 5 | Fatigue_Diff_MeanAbsChange | 0.02345 | 4.7% |

---

## Citation

If you use this code, please cite:

```bibtex
@article{hfenn2026,
  title   = {HFENN: Hierarchical Feature Extraction Neural Network for
             Continuous Driver Fatigue Index Estimation},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {2026}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
