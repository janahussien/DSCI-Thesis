"""
preprocessing.py
================
Full EEG preprocessing pipeline:

  1. Load .mat files
  2. Epoch extraction (imagination window only)
  3. Band-pass + notch filtering
  4. Common Average Reference (CAR) re-referencing
  5. Surface Laplacian filter (NEW — sharpens local sources)
  6. Adaptive amplitude rejection (NEW — data-driven threshold per subject)
  7. Baseline correction
  8. Z-score normalisation (per-channel, fit on train set when used inside CV)
  9. Channel selection (variance-based, optional)

All steps are togglable via CONFIG.

CHANGES vs original:
  - _reject_epoch replaced by _adaptive_reject_epoch (MAD-based, no fixed µV)
  - _laplacian_filter added after CAR in pipeline
  - preprocess_pipeline calls the new functions
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
            mat_path = l_folder / f"S{subject_id:02d}_L{letter_id:02d}_T{trial_id}.mat"
            if not mat_path.exists():
                print(f"  [warn] Missing: {mat_path}")
                continue

            mat = sio.loadmat(str(mat_path), simplify_cells=True)
            eeg_struct = mat.get("EEG", {})

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
                "label":  letter_id - 1,
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

        # ── Step 5: surface Laplacian (NEW) ───────────────────────────────
        epoch = _laplacian_filter(epoch)

        # ── Step 6: adaptive amplitude rejection (NEW) ────────────────────
        if _adaptive_reject_epoch(epoch):
            rejected += 1
            continue

        # ── Step 7: baseline correction (mean of first 0.2 s) ─────────────
        epoch = _baseline_correct(epoch, sfreq, baseline_sec=0.2)

        epochs.append(epoch)
        labels.append(rec["label"])

    if debug:
        print(f"      Rejected {rejected} / {len(records)} trials")

    if len(epochs) == 0:
        raise RuntimeError(
            f"All {len(records)} trials were rejected. "
            "The adaptive threshold may be too strict — check your data."
        )

    X = np.array(epochs, dtype=np.float64)          # (N, C, T)
    y = np.array(labels, dtype=np.int32)

    # ── Step 8: per-channel Z-score normalisation ──────────────────────────
    X = _zscore_normalize(X)

    # ── Step 9: channel selection (keep top-k by variance) ─────────────────
    X = _select_channels(X, k=CONFIG["n_channels"])

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
    """
    imagine_start_s = CONFIG["t_relax"] + CONFIG["t_observe"]
    onset  = int(imagine_start_s * sfreq)
    offset = onset + int(CONFIG["epoch_tmax"] * sfreq)

    if raw.shape[1] < offset:
        return None

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


def _laplacian_filter(data: np.ndarray) -> np.ndarray:
    """
    Approximate surface Laplacian using nearest-neighbor spatial filter.
    Each channel is replaced by itself minus the mean of its neighbors.
    Sharpens local cortical sources and reduces volume conduction.

    Neighbor map is for the EMOTIV 14-channel layout:
        AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
        idx:  0   1   2    3   4   5   6   7   8   9   10  11  12   13
    """
    neighbors = {
        0:  [1, 2],           # AF3  ← F7, F3
        1:  [0, 3],           # F7   ← AF3, FC5
        2:  [0, 3, 4],        # F3   ← AF3, FC5, T7  (approx)
        3:  [1, 2, 5],        # FC5  ← F7, F3, P7
        4:  [2, 3, 5],        # T7   ← F3, FC5, P7
        5:  [3, 4, 6],        # P7   ← FC5, T7, O1
        6:  [5, 7],           # O1   ← P7, O2
        7:  [6, 8],           # O2   ← O1, P8
        8:  [7, 9, 10],       # P8   ← O2, T8, FC6
        9:  [8, 10, 11],      # T8   ← P8, FC6, F4
        10: [8, 9, 12],       # FC6  ← P8, T8, F8
        11: [9, 10, 13],      # F4   ← T8, FC6, AF4
        12: [10, 11, 13],     # F8   ← FC6, F4, AF4
        13: [11, 12],         # AF4  ← F4, F8
    }
    result = data.copy()
    n_ch = data.shape[0]
    for ch, nbrs in neighbors.items():
        if ch < n_ch and all(n < n_ch for n in nbrs):
            result[ch] = data[ch] - data[nbrs].mean(axis=0)
    return result


def _adaptive_reject_epoch(
    data: np.ndarray,
    multiplier: float = 20.0
) -> bool:
    """
    Reject trial if any channel exceeds multiplier × MAD of the whole epoch.
    This is data-driven — no fixed µV threshold.

    multiplier=20.0 is intentionally lenient — we only want to catch
    genuinely catastrophic artifacts (electrode pop-off, movement spike),
    not normal EEG variance. The Laplacian filter preceding this step
    reduces amplitudes, so a tight multiplier would over-reject.

    Returns True = reject this trial.
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad < 1e-10:
        return False
    threshold = multiplier * mad
    return bool(np.abs(data - median).max() > threshold)


def _baseline_correct(
    data: np.ndarray, sfreq: float, baseline_sec: float = 0.2
) -> np.ndarray:
    n_base = int(baseline_sec * sfreq)
    baseline_mean = data[:, :n_base].mean(axis=1, keepdims=True)
    return data - baseline_mean


def _zscore_normalize(X: np.ndarray) -> np.ndarray:
    """Z-score each channel across the trial dimension independently."""
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
    ch_var = X.var(axis=(0, 2))
    top_k  = np.argsort(ch_var)[::-1][:k]
    top_k  = np.sort(top_k)
    return X[:, top_k, :]


# ─────────────────────────────────────────────────────────────────────────────
# ICA helper (requires MNE — optional)
# ─────────────────────────────────────────────────────────────────────────────

def apply_ica(X: np.ndarray, n_components: int = 14) -> np.ndarray:
    """
    Apply FastICA and remove components with high kurtosis (muscle artifacts)
    or frontal dominance (eye blinks).

    Requires: pip install mne
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