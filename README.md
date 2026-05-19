# Arabic BCI — EEG Classification of Imagined Arabic Letters

A brain-computer interface (BCI) pipeline for classifying imagined Arabic letters from EEG signals. The system uses the EMOTIV Epoc X headset (14 channels) and a multi-band feature extraction + LDA classification approach.

## Overview

Subjects imagine writing each of 28 Arabic letters while EEG is recorded. The pipeline:

1. **Preprocessing** — bandpass filter (0.5–40 Hz), notch filter (50 Hz), ICA artifact removal, Laplacian spatial filter, adaptive artifact subspace reconstruction (ASR)
2. **Feature extraction** — Riemannian covariance, Band Power (BP), Phase Locking Value (PLV), adaptive CSP
3. **Band optimization** — per-subject frequency band selection (delta+theta, alpha, beta, gamma, broadband) via ANOVA feature selection
4. **Classification** — LDA with cross-validated evaluation

## Dataset

**Raw Imagined Arabic Letters Dataset** — 30 subjects, 28 letters, 10 trials each, 14 EEG channels at 256 Hz.

Each trial: 5 s relax → 5 s observe → 8 s imagine.

The dataset is not included in this repository. Place it at `data/` with the structure:

```
data/
  S01/
    L01/   ← letter 1
      S01_L01_T1.mat
      ...
      S01_L01_T10.mat
    L02/
    ...
    L28/
  S02/
  ...
```

For new subjects recorded with EMOTIV Epoc X (EDF format), use the structure:

```
real time/
  S01/
    L01/
      *T01*.edf
      ...
    ...
    L28/
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train on existing dataset subject
```bash
python run_band_optimization.py          # runs band optimization on all subjects
```

### New subject (EMOTIV Epoc X / EDF files)
```bash
python run_new_subject.py                # full pipeline
python run_new_subject.py --preview      # check files only
python run_new_subject.py --letters 8   # first N letters only
python run_new_subject.py --data_root /path/to/data
```

### Save a trained model for a subject
```bash
python train_subject_model.py
```

### Web demo
```bash
python app.py
# then open interface.html in a browser
```

## Project Structure

```
├── config.py                        # central config (paths, hyperparameters)
├── preprocessing.py                 # EEG preprocessing pipeline
├── features.py                      # feature extraction (Riem, BP, PLV, CSP)
├── models.py                        # classifier definitions
├── final_model.py                   # validated final pipeline
├── utils.py                         # I/O and printing utilities
├── handedness.py                    # subject handedness metadata
│
├── run_band_optimization.py         # main training script (per-subject band opt)
├── run_new_subject.py               # new subject inference (EDF input)
├── train_subject_model.py           # train + save subject model (.pkl)
│
├── app.py                           # Flask backend for web demo
├── interface.html                   # web demo frontend
│
├── plot_scripts/                    # visualisation scripts
│   ├── generate_plots.py
│   ├── plot_features.py
│   ├── plot_compare_eeg.py
│   ├── plot_same_letter_diff_subjects.py
│   └── plot_topomap_comparison.py
│
├── plots/                           # generated figures
├── results/                         # JSON experiment outputs
├── models/                          # saved subject models (.pkl)
│
└── experiments/                     # exploratory scripts (not part of final pipeline)
    ├── main.py
    ├── run_adaptive_pipeline.py
    ├── run_best_models.py
    ├── run_calibration_curve.py
    ├── run_cross_session.py
    ├── run_csp_riemannian.py
    ├── run_deep_learning.py
    ├── run_ensemble.py
    ├── run_improved_pipeline_test.py
    ├── run_loso.py
    ├── run_model_experiments.py
    ├── run_s22.py
    ├── run_subset_crosssubject.py
    └── run_voting_ensemble.py
```

## Key Hyperparameters (`config.py`)

| Parameter | Value |
|---|---|
| Channels | 14 (AF3, AF4, F7, F8, F3, F4, FC5, FC6, T7, T8, P7, P8, O1, O2) |
| Sampling rate | 256 Hz |
| Bandpass | 0.5–40 Hz |
| Epoch | 0–6 s (imagination window) |
| CV folds | 10 |
| Adaptive CSP components | 2 / 4 / 6 (based on trial count) |

## Requirements

- Python 3.10+
- See `requirements.txt` for full dependency list (numpy, scipy, scikit-learn, mne, pyriemann, PyWavelets, torch, xgboost)
