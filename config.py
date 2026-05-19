"""
config.py — Central configuration for the EEG Arabic BCI pipeline.
Edit the paths and hyperparameters here; everything else adapts automatically.
"""

from pathlib import Path

CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    "data_root":   Path(__file__).parent / "data",
    "results_dir": Path("results"),

    # ── Dataset constants ──────────────────────────────────────────────────
    "n_subjects":  30,
    "n_letters":   28,
    "n_trials":    10,
    "n_channels":  14,
    "sfreq":       256,
    "channel_names": [
        "AF3", "AF4", "F7", "F8", "F3", "F4",
        "FC5", "FC6", "T7", "T8", "P7", "P8", "O1", "O2"
    ],

    # ── Trial timing (seconds) ─────────────────────────────────────────────
    "t_relax":   5.0,
    "t_observe": 5.0,
    "t_imagine": 8.0,

    # ── Preprocessing ──────────────────────────────────────────────────────
    "bandpass":         (0.5, 40.0),
    "notch_freqs":      [50.0],
    "epoch_tmin":       0.0,
    "epoch_tmax":       6.0,
    "ica_n_components": 14,
    "ica_random_state": 42,
    "asr_cutoff":       20,
    "re_reference":     "average",

    # ── Segmentation ───────────────────────────────────────────────────────
    "segment_len_samples": 128,
    "segment_overlap":     0.5,

    # ── Feature extraction ─────────────────────────────────────────────────
    "wavelet":    "morl",
    "cwt_scales": range(1, 33),
    "freq_bands": {
        "delta": (0.5, 4),
        "theta": (4,   8),
        "alpha": (8,  13),
        "beta":  (13, 30),
        "gamma": (30, 40),
    },

    # ── Adaptive CSP ───────────────────────────────────────────────────────
    # n_components is chosen automatically based on available trial count.
    # Thresholds are conservative: CSP needs enough trials per class
    # (28 classes OvR) to fit reliable spatial filters without overfitting.
    #
    #   < 200 trials → 2 components  (very data-sparse, minimal overfitting)
    #   200–249      → 4 components
    #   250+         → 6 components  (original setting, well-powered)
    #
    # To disable adaptive CSP entirely, set "csp_enabled": False.
    "csp_enabled": True,
    "csp_components_map": [
        (200, 2),   # (min_trials_threshold, n_components)
        (250, 4),
        (999, 6),   # catch-all for well-powered subjects
    ],

    # ── Models ─────────────────────────────────────────────────────────────
    "cv_folds":    10,
    "random_state": 42,
    "n_jobs":      -1,

    # Deep-learning specific
    "dl_epochs":   80,
    "dl_batch":    32,
    "dl_lr":       1e-3,
    "dl_patience": 15,
}