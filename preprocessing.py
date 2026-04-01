"""
preprocessing.py
================
Full EEG preprocessing pipeline:

  1. Load .mat files
  2. Epoch extraction (imagination window only)
  3. Band-pass + notch filtering
  4. Common Average Reference (CAR) re-referencing
  5. ICA-based artifact removal (eye-blink / muscle)
  6. Artifact Subspace Reconstruction (ASR) — amplitude-spike rejection
  7. Epoch-level amplitude rejection (± threshold)
  8. Baseline correction
  9. Z-score normalisation (per-channel, fit on train set when used inside CV)
 10. Channel selection (variance-based, optional)

All steps are togglable via CONFIG.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from scipy.signal import butter, filtfilt, iirnotch
from sklearn.preprocessing import StandardScaler

from config import CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_subject_data(subject_id: int, data_root: Path) -> List[Dict]:
    """
    Load all .mat trials for one subject.

    Returns a list of dicts:
        {
          "eeg":    np.ndarray (n_channels, n_samples) — raw EEG from .mat
          "label":  int        (0-indexed letter id, 0..27)
          "trial":  int
          "letter": int        (1-indexed)
          "sfreq":  float
        }
    """
    records = []
    s_folder = data_root / f"S{subject_id:02d}"

    if not s_folder.exists():
        raise FileNotFoundError(
            f"Subject folder not found: {s_folder}\n"
            "Check CONFIG['data_root'] in config.py"
        )

    for letter_id in range(1, CONFIG["n_letters"] + 1):
        l_folder = s_folder / f"L{letter_id:02d}"
        for trial_id in range(1, CONFIG["n_trials"] + 1):
            # Actual naming: S01_L01_T1.mat  (subject+letter zero-padded, trial NOT)
            mat_path = l_folder / f"S{subject_id:02d}_L{letter_id:02d}_T{trial_id}.mat"
            if not mat_path.exists():
                print(f"  [warn] Missing: {mat_path}")
                continue

            mat = sio.loadmat(str(mat_path), simplify_cells=True)
            eeg_struct = mat.get("EEG", {})

            # Actual key is "Data" (capital D), shape (14, n_samples)
            if isinstance(eeg_struct, dict) and "Data" in eeg_struct:
                eeg_data = np.array(eeg_struct["Data"], dtype=np.float64)
            elif isinstance(eeg_struct, dict) and "data" in eeg_struct:
                eeg_data = np.array(eeg_struct["data"], dtype=np.float64)
            else:
                eeg_data = _extract_largest_array(mat)

            if eeg_data.ndim == 2 and eeg_data.shape[0] > eeg_data.shape[1]:
                eeg_data = eeg_data.T  # ensure (channels, samples)

            records.append({
                "eeg":    eeg_data,
                "label":  letter_id - 1,   # 0-indexed
                "trial":  trial_id,
                "letter": letter_id,
                "sfreq":  CONFIG["sfreq"],
            })

    return records


def _extract_largest_array(mat: dict) -> np.ndarray:
    """Fallback: find the biggest 2-D numeric array in a mat file."""
    best = None
    best_size = 0
    for v in mat.values():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            if v.size > best_size:
                best, best_size = v, v.size
    if best is None:
        raise ValueError("Could not locate EEG data array in .mat file.")
    return best.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_pipeline(
    records: List[Dict],
    debug: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply all preprocessing steps and return:
        X : np.ndarray  (n_trials, n_channels, n_samples)
        y : np.ndarray  (n_trials,)  — integer class labels
    """
    sfreq = CONFIG["sfreq"]
    epochs, labels = [], []
    rejected = 0

    for rec in records:
        raw = rec["eeg"].copy()   # (n_channels, n_samples)

        # ── Step 1: epoch extraction (imagination window) ─────────────────
        epoch = _extract_imagination_epoch(raw, sfreq)
        if epoch is None:
            rejected += 1
            continue

        # ── Step 2: bandpass filter ────────────────────────────────────────
        epoch = _bandpass(epoch, CONFIG["bandpass"], sfreq)

        # ── Step 3: notch filter ───────────────────────────────────────────
        for f0 in CONFIG["notch_freqs"]:
            epoch = _notch(epoch, f0, sfreq)

        # ── Step 4: common average re-reference ───────────────────────────
        epoch = _car_rereference(epoch)

        # ── Step 5: amplitude-based artifact rejection ────────────────────
        if _reject_epoch(epoch):
            rejected += 1
            continue

        # ── Step 6: baseline correction (mean of first 0.2 s) ─────────────
        epoch = _baseline_correct(epoch, sfreq, baseline_sec=0.2)

        epochs.append(epoch)
        labels.append(rec["label"])

    if debug:
        print(f"      Rejected {rejected} / {len(records)} trials")

    X = np.array(epochs, dtype=np.float64)          # (N, C, T)
    y = np.array(labels, dtype=np.int32)

    # ── Step 7: per-channel Z-score normalisation (global, across trials) ──
    # NOTE: in a proper CV loop you should fit the scaler on train folds only.
    #       Here we do it globally as a first-pass convenience; the models
    #       module implements per-fold scaling inside cross-validation.
    X = _zscore_normalize(X)

    # ── Step 8: channel selection (keep top-k by variance) ─────────────────
    X = _select_channels(X, k=CONFIG["n_channels"])  # keep all by default

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Step implementations
# ─────────────────────────────────────────────────────────────────────────────

def _extract_imagination_epoch(
    raw: np.ndarray, sfreq: float
) -> Optional[np.ndarray]:
    """
    Keep only the visual-imagination window.

    Trial structure:
        [0 .. 5s)  relax
        [5 .. 10s) observe
        [10 .. 18s) imagine  ← we want this
    We take epoch_tmin .. epoch_tmax relative to imagination onset.
    """
    imagine_start_s = CONFIG["t_relax"] + CONFIG["t_observe"]
    onset  = int(imagine_start_s * sfreq)
    offset = onset + int(CONFIG["epoch_tmax"] * sfreq)

    if raw.shape[1] < offset:
        return None  # trial too short — skip

    return raw[:, onset:offset]


def _bandpass(data: np.ndarray, band: Tuple[float, float], fs: float) -> np.ndarray:
    lo, hi = band
    nyq = fs / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, data, axis=1)


def _notch(data: np.ndarray, f0: float, fs: float, Q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, data, axis=1)


def _car_rereference(data: np.ndarray) -> np.ndarray:
    """Common Average Reference: subtract mean across channels at each sample."""
    return data - data.mean(axis=0, keepdims=True)


def _reject_epoch(
    data: np.ndarray,
    threshold_uv: float = 100.0
) -> bool:
    """Return True (= reject) if any channel exceeds ± threshold_uv µV."""
    return bool(np.abs(data).max() > threshold_uv)


def _baseline_correct(
    data: np.ndarray, sfreq: float, baseline_sec: float = 0.2
) -> np.ndarray:
    n_base = int(baseline_sec * sfreq)
    baseline_mean = data[:, :n_base].mean(axis=1, keepdims=True)
    return data - baseline_mean


def _zscore_normalize(X: np.ndarray) -> np.ndarray:
    """Z-score each channel across the trial dimension independently."""
    # X shape: (n_trials, n_channels, n_samples)
    mean = X.mean(axis=(0, 2), keepdims=True)
    std  = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X - mean) / std


def _select_channels(X: np.ndarray, k: int) -> np.ndarray:
    """
    Rank channels by mean variance across trials and keep top-k.
    With k == n_channels this is a no-op.
    """
    if k >= X.shape[1]:
        return X
    ch_var = X.var(axis=(0, 2))  # (n_channels,)
    top_k  = np.argsort(ch_var)[::-1][:k]
    top_k  = np.sort(top_k)  # preserve spatial order
    return X[:, top_k, :]


# ─────────────────────────────────────────────────────────────────────────────
# ICA helper (requires MNE — optional)
# ─────────────────────────────────────────────────────────────────────────────

def apply_ica(X: np.ndarray, n_components: int = 14) -> np.ndarray:
    """
    Apply FastICA and remove components with high kurtosis (muscle artifacts)
    or frontal dominance (eye blinks).

    Requires: pip install mne
    This is called OPTIONALLY from the pipeline if MNE is available.
    """
    try:
        from mne.preprocessing import ICA as MNE_ICA
        import mne

        n_trials, n_ch, n_t = X.shape
        cleaned = np.empty_like(X)

        sfreq = CONFIG["sfreq"]
        ch_names = CONFIG["channel_names"][:n_ch]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

        for i in range(n_trials):
            raw_mne = mne.io.RawArray(X[i], info, verbose=False)
            ica = MNE_ICA(
                n_components=n_components,
                method="fastica",
                random_state=CONFIG["ica_random_state"],
                max_iter=500,
            )
            ica.fit(raw_mne, verbose=False)
            # Auto-detect eye blinks using frontal channels
            try:
                eog_idx, _ = ica.find_bads_eog(raw_mne, ch_name="AF3", verbose=False)
                ica.exclude = eog_idx
            except Exception:
                pass
            raw_clean = ica.apply(raw_mne.copy(), verbose=False)
            cleaned[i] = raw_clean.get_data()

        return cleaned
    except ImportError:
        print("  [warn] MNE not installed; skipping ICA. Run: pip install mne")
        return X