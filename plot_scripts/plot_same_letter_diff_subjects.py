"""
plot_same_letter_diff_subjects.py
==================================
Plots real EEG waveforms for the same Arabic letter (ج = L05)
across 3 different subjects to show inter-subject variability.

Usage:
    python plot_same_letter_diff_subjects.py

Output:
    plots/same_letter_diff_subjects.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch

# ─────────────────────────────────────────────────────────────────────────────
# Config — edit these if needed
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT   = Path("/Users/janahussien/Desktop/eeg_project/data")
LETTER_ID   = 5          # ج is letter 5
TRIAL_ID    = 1          # which trial to use (1-10)
SUBJECTS    = [1, 12, 27]  # S01 (average), S12 (best), S27 (weakest)
CHANNEL_IDX = 6           # O1 — occipital, most relevant for visual imagery
SFREQ       = 256
OUTPUT_PATH = Path("plots/same_letter_diff_subjects.png")
OUTPUT_PATH.parent.mkdir(exist_ok=True)

# Trial timing
T_RELAX   = 5.0
T_OBSERVE = 5.0
T_IMAGINE = 6.0   # we plot 6s imagination window

# Thesis palette
BG      = '#f0ece4'
WHITE   = '#faf8f4'
TEXT    = '#2c2824'
MID     = '#6b6258'
SOFT    = '#b5a99a'
COLORS  = ['#8c7b6e', '#6b8c7a', '#7a6b8c']   # bark, sage, dusty purple

ARABIC_LETTERS = [
    'ا','ب','ت','ث','ج','ح','خ','د','ذ','ر',
    'ز','س','ش','ص','ض','ط','ظ','ع','غ','ف',
    'ق','ك','ل','م','ن','ه','و','ي'
]

CHANNEL_NAMES = [
    "AF3","F7","F3","FC5","T7","P7","O1",
    "O2","P8","T8","FC6","F4","F8","AF4"
]

# ─────────────────────────────────────────────────────────────────────────────
# Load one trial
# ─────────────────────────────────────────────────────────────────────────────

def load_trial(subject_id, letter_id, trial_id):
    s_folder = DATA_ROOT / f"S{subject_id:02d}" / f"L{letter_id:02d}"
    mat_path = s_folder / f"S{subject_id:02d}_L{letter_id:02d}_T{trial_id}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Not found: {mat_path}")
    mat = sio.loadmat(str(mat_path), simplify_cells=True)
    eeg = mat.get("EEG", {})
    if isinstance(eeg, dict) and "Data" in eeg:
        data = np.array(eeg["Data"], dtype=np.float64)
    elif isinstance(eeg, dict) and "data" in eeg:
        data = np.array(eeg["data"], dtype=np.float64)
    else:
        # fallback: find largest 2D array
        data = max(
            (v for v in mat.values() if isinstance(v, np.ndarray) and v.ndim == 2),
            key=lambda x: x.size
        ).astype(np.float64)
    if data.shape[0] > data.shape[1]:
        data = data.T  # ensure (channels, samples)
    return data


def preprocess(data):
    """Bandpass + notch on (n_ch, n_samples)."""
    sfreq = SFREQ
    nyq = sfreq / 2.0
    b, a = butter(4, [0.5/nyq, 40.0/nyq], btype='band')
    data = filtfilt(b, a, data, axis=1)
    b, a = iirnotch(50.0, Q=30.0, fs=sfreq)
    data = filtfilt(b, a, data, axis=1)
    # CAR
    data = data - data.mean(axis=0, keepdims=True)
    return data


def extract_imagination_window(data):
    """Extract the 6-second imagination window."""
    start = int((T_RELAX + T_OBSERVE) * SFREQ)
    end   = start + int(T_IMAGINE * SFREQ)
    if data.shape[1] < end:
        raise ValueError(f"Trial too short: {data.shape[1]} samples, need {end}")
    return data[:, start:end]


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading EEG data...")
    signals = []
    for sid in SUBJECTS:
        try:
            raw   = load_trial(sid, LETTER_ID, TRIAL_ID)
            proc  = preprocess(raw)
            epoch = extract_imagination_window(proc)
            ch    = epoch[CHANNEL_IDX]     # single channel (n_samples,)
            signals.append((sid, ch))
            print(f"  S{sid:02d} loaded — shape {epoch.shape}")
        except Exception as e:
            print(f"  S{sid:02d} FAILED: {e}")

    if not signals:
        print("No data loaded — check DATA_ROOT and subject/letter IDs")
        return

    letter   = ARABIC_LETTERS[LETTER_ID - 1]
    t        = np.linspace(0, T_IMAGINE, int(T_IMAGINE * SFREQ))

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        len(signals), 1,
        figsize=(13, 3.2 * len(signals)),
        facecolor=BG
    )
    if len(signals) == 1:
        axes = [axes]

    fig.suptitle(
        f'Same Letter  ·  {letter}  ·  {len(signals)} Different Subjects\n'
        f'Channel: {CHANNEL_NAMES[CHANNEL_IDX]}   |   Imagination window (0–{int(T_IMAGINE)}s)',
        fontsize=13,
        fontweight='bold',
        color=TEXT,
        y=0.98,
        fontfamily='serif'
    )

    for ax, (sid, sig), color in zip(axes, signals, COLORS):
        ax.set_facecolor(WHITE)

        # Shade background subtly
        ax.fill_between(t, sig.min()*1.2, sig.max()*1.2,
                        alpha=0.04, color=color)

        # Plot waveform
        ax.plot(t, sig, color=color, linewidth=1.2, alpha=0.92)

        # Zero line
        ax.axhline(0, color=SOFT, linewidth=0.6, linestyle='--', alpha=0.5)

        # Styling
        ax.set_xlim(0, T_IMAGINE)
        ax.set_ylabel('Amplitude (µV)', fontsize=9, color=MID)
        ax.tick_params(colors=MID, labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(SOFT)
        ax.spines['bottom'].set_color(SOFT)
        ax.set_facecolor(WHITE)

        # Subject label
        ax.text(0.01, 0.90, f'Subject {sid:02d}',
                transform=ax.transAxes,
                fontsize=11, fontweight='bold',
                color=TEXT, va='top',
                fontfamily='serif')

        # Stats annotation
        rms = np.sqrt(np.mean(sig**2))
        ax.text(0.99, 0.90,
                f'RMS: {rms:.1f} µV',
                transform=ax.transAxes,
                fontsize=8, color=MID,
                ha='right', va='top',
                fontfamily='monospace')

        # Remove x labels except last
        if ax != axes[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)', fontsize=9, color=MID)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.patch.set_facecolor(BG)

    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight', facecolor=BG)
    print(f"\nSaved → {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()