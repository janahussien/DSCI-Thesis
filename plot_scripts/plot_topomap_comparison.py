"""
plot_topomap_comparison.py
==========================
Generates real EEG topomap heatmaps for 2 subjects across 4 Arabic letters,
showing how the same letter produces completely different spatial patterns
across subjects — demonstrating why per-subject models are necessary.

Usage:
    python plot_topomap_comparison.py

Output:
    plots/topomap_cross_subject.png

Requires: mne, scipy, matplotlib, numpy
    pip install mne --break-system-packages
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RBFInterpolator
from scipy.signal import butter, filtfilt, iirnotch
from pathlib import Path
import scipy.io as sio

Path("plots").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT  = Path("/Users/janahussien/Desktop/eeg_project/data")
SUBJECTS   = [12, 1]          # S12 (best), S01 (average)
LETTERS    = [1, 2, 5, 8]     # ا  ب  ج  د
N_TRIALS   = 5                 # average over this many trials per letter

SFREQ = 256

ARABIC_LETTERS = [
    'Alef','Ba','Ta','Tha','Jeem','Ha','Kha','Dal','Dhal','Ra',
    'Zay','Seen','Sheen','Sad','Dad','Ta2','Dha','Ain','Ghain','Fa',
    'Qaf','Kaf','Lam','Meem','Noon','Ha2','Waw','Ya'
]

CH_NAMES = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]

# 2D positions on unit circle (x, y) — standard 10-20 layout for EMOTIV 14ch
# x: left=-1, right=+1  y: front=+1, back=-1
CH_POS_NORM = {
    "AF3": (-0.30,  0.82),
    "AF4": ( 0.30,  0.82),
    "F7":  (-0.71,  0.52),
    "F8":  ( 0.71,  0.52),
    "F3":  (-0.37,  0.58),
    "F4":  ( 0.37,  0.58),
    "FC5": (-0.58,  0.27),
    "FC6": ( 0.58,  0.27),
    "T7":  (-0.88,  0.00),
    "T8":  ( 0.88,  0.00),
    "P7":  (-0.68, -0.44),
    "P8":  ( 0.68, -0.44),
    "O1":  (-0.27, -0.84),
    "O2":  ( 0.27, -0.84),
}

# Thesis palette
BG    = '#f0ece4'
WHITE = '#faf8f4'
TEXT  = '#2c2824'
MID   = '#6b6258'
SOFT  = '#b5a99a'


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_trial(subject_id, letter_id, trial_id):
    path = (DATA_ROOT / f"S{subject_id:02d}" / f"L{letter_id:02d}"
            / f"S{subject_id:02d}_L{letter_id:02d}_T{trial_id}.mat")
    mat  = sio.loadmat(str(path), simplify_cells=True)
    eeg  = mat.get("EEG", {})
    if isinstance(eeg, dict) and "Data" in eeg:
        data = np.array(eeg["Data"], dtype=np.float64)
    elif isinstance(eeg, dict) and "data" in eeg:
        data = np.array(eeg["data"], dtype=np.float64)
    else:
        data = max(
            (v for v in mat.values() if isinstance(v, np.ndarray) and v.ndim == 2),
            key=lambda x: x.size
        ).astype(np.float64)
    if data.shape[0] > data.shape[1]:
        data = data.T
    return data   # (14, n_samples)


def preprocess(data):
    nyq  = SFREQ / 2.0
    b, a = butter(4, [0.5/nyq, 40.0/nyq], btype='band')
    data = filtfilt(b, a, data, axis=1)
    b, a = iirnotch(50.0, Q=30.0, fs=SFREQ)
    data = filtfilt(b, a, data, axis=1)
    data = data - data.mean(axis=0, keepdims=True)
    return data


def get_imagination_epoch(subject_id, letter_id, trial_id):
    raw   = load_trial(subject_id, letter_id, trial_id)
    proc  = preprocess(raw)
    start = int((5 + 5) * SFREQ)
    end   = start + int(6 * SFREQ)
    return proc[:, start:end]   # (14, 1536)


def get_mean_amplitude(subject_id, letter_id, n_trials=5):
    """
    Average RMS amplitude per channel across n_trials.
    Returns array of shape (14,) — one value per electrode.
    """
    channel_rms = []
    loaded = 0
    for t in range(1, 11):
        try:
            ep = get_imagination_epoch(subject_id, letter_id, t)
            # RMS per channel
            rms = np.sqrt(np.mean(ep**2, axis=1))   # (14,)
            channel_rms.append(rms)
            loaded += 1
            if loaded >= n_trials:
                break
        except Exception:
            pass

    if not channel_rms:
        return np.zeros(14)

    arr = np.array(channel_rms)   # (n_loaded, 14)
    mean_rms = arr.mean(axis=0)   # (14,)
    return mean_rms


# ─────────────────────────────────────────────────────────────────────────────
# Topomap interpolation
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_topomap(values, resolution=100):
    """
    Interpolate 14 channel values onto a 2D grid using RBF.
    Returns (xx, yy, zz, mask) where mask is the head circle.
    """
    points = np.array([CH_POS_NORM[ch] for ch in CH_NAMES])   # (14, 2)
    vals   = np.array(values, dtype=np.float64)

    # Grid
    x = np.linspace(-1.1, 1.1, resolution)
    y = np.linspace(-1.1, 1.1, resolution)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

    # RBF interpolation
    rbf = RBFInterpolator(points, vals, kernel='thin_plate_spline', smoothing=0.0)
    zz  = rbf(grid_pts).reshape(resolution, resolution)

    # Circular mask (head boundary)
    mask = (xx**2 + yy**2) > 1.0

    return xx, yy, zz, mask


def draw_topomap(ax, values, title, cmap, vmin, vmax):
    """Draw one topomap on the given axis."""
    xx, yy, zz, mask = interpolate_topomap(values)

    # Apply mask
    zz_masked = np.ma.array(zz, mask=mask)

    # Plot interpolated surface
    im = ax.contourf(xx, yy, zz_masked, levels=40,
                     cmap=cmap, vmin=vmin, vmax=vmax)

    # Contour lines
    ax.contour(xx, yy, zz_masked, levels=8,
               colors='white', linewidths=0.4, alpha=0.5)

    # Head circle
    head = plt.Circle((0, 0), 1.0, fill=False, color=TEXT,
                       linewidth=2.5, zorder=5)
    ax.add_patch(head)

    # Nose
    nose_x = [-.09, 0, .09]
    nose_y = [1.0,  1.15, 1.0]
    ax.plot(nose_x, nose_y, color=TEXT, linewidth=2.5, zorder=5)

    # Ears
    for side in [-1, 1]:
        ear_x = [side*1.0, side*1.1, side*1.1, side*1.0]
        ear_y = [0.1, 0.1, -0.1, -0.1]
        ax.plot(ear_x, ear_y, color=TEXT, linewidth=2.5, zorder=5)

    # Electrode dots
    for i, ch in enumerate(CH_NAMES):
        x, y = CH_POS_NORM[ch]
        ax.scatter(x, y, s=18, color='white', zorder=6,
                   edgecolors=TEXT, linewidths=0.8)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.28)
    ax.set_aspect('equal')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=13, fontweight='bold',
                     color=TEXT, pad=8, fontfamily='serif')

    return im


# ─────────────────────────────────────────────────────────────────────────────
# Main plot
# ─────────────────────────────────────────────────────────────────────────────

def main():
    n_subj   = len(SUBJECTS)
    n_lett   = len(LETTERS)

    print("Loading EEG data...")
    # Collect all values first so we can set consistent colormap range
    all_vals = {}
    for sid in SUBJECTS:
        for lid in LETTERS:
            print(f"  S{sid:02d} · L{lid:02d} ({ARABIC_LETTERS[lid-1]})...", end=" ")
            v = get_mean_amplitude(sid, lid, N_TRIALS)
            all_vals[(sid, lid)] = v
            print(f"max={v.max():.2f} µV")

    # Global vmin/vmax for consistent colormap across all plots
    all_flat = np.concatenate(list(all_vals.values()))
    vmax = np.percentile(all_flat, 95)
    vmin = 0

    # Colormap — red/blue diverging but start from white (all positive RMS)
    cmap = LinearSegmentedColormap.from_list(
        'topo', ['#3B6DB5', '#AABFE8', '#FAFAFA', '#F0A88A', '#C03020'], N=256
    )

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(4.2 * n_lett, 4.5 * n_subj + 0.8), facecolor=BG)

    # Main title
    fig.suptitle(
        'Same Letter · Different Subjects — Spatial EEG Patterns\n'
        'RMS amplitude per channel · Imagination window',
        fontsize=14, fontweight='bold', color=TEXT,
        fontfamily='serif', y=0.98
    )

    gs = gridspec.GridSpec(
        n_subj, n_lett,
        hspace=0.15, wspace=0.08,
        left=0.04, right=0.88,
        top=0.88, bottom=0.04
    )

    last_im = None
    for r, sid in enumerate(SUBJECTS):
        for c, lid in enumerate(LETTERS):
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor(WHITE)

            vals  = all_vals[(sid, lid)]
            title = ARABIC_LETTERS[lid - 1] if r == 0 else ""
            im    = draw_topomap(ax, vals, title, cmap, vmin, vmax)
            last_im = im

            # Subject label on left
            if c == 0:
                ax.text(-1.6, 0, f'Subject {sid:02d}',
                        fontsize=12, fontweight='bold',
                        color=TEXT, va='center', ha='center',
                        rotation=90, fontfamily='serif')

    # Colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.018, 0.65])
    cb = fig.colorbar(last_im, cax=cbar_ax)
    cb.set_label('RMS amplitude (µV)', fontsize=10, color=MID)
    cb.ax.tick_params(colors=MID, labelsize=8)
    cb.outline.set_edgecolor(SOFT)

    out = Path("plots/topomap_cross_subject.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()