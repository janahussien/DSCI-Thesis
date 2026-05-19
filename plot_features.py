"""
plot_features.py
================
Generates three feature visualization plots for the thesis:

  1. Riemannian — covariance matrix heatmap (14x14) for one trial
  2. PLV — connectivity matrix heatmap between all channel pairs
  3. CSP — spatial filter weights + log-variance separation

Usage:
    python plot_features.py

Output:
    plots/feature_riemannian.png
    plots/feature_plv.png
    plots/feature_csp.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch, hilbert
from sklearn.covariance import OAS

# Fix Arabic font rendering in matplotlib
import matplotlib.font_manager as fm
import os

# Try to register Noto Naskh Arabic if available on macOS
_arabic_font = None
for _candidate in [
    "/Library/Fonts/NotoNaskhArabic-Regular.ttf",
    "/System/Library/Fonts/Supplemental/NotoNaskhArabic-Regular.ttf",
    os.path.expanduser("~/Library/Fonts/NotoNaskhArabic-Regular.ttf"),
]:
    if os.path.exists(_candidate):
        fm.fontManager.addfont(_candidate)
        _arabic_font = fm.FontProperties(fname=_candidate)
        break

def ar_text(ax, x, y, text, **kwargs):
    """Draw Arabic text using the Arabic font if available, else use transliteration."""
    AR_LATIN = {
        'ا': 'Alef', 'ب': 'Ba', 'ت': 'Ta', 'ث': 'Tha', 'ج': 'Jeem',
        'ح': 'Ha', 'خ': 'Kha', 'د': 'Dal', 'ذ': 'Dhal', 'ر': 'Ra',
        'ز': 'Zay', 'س': 'Seen', 'ش': 'Sheen', 'ص': 'Sad', 'ض': 'Dad',
        'ط': 'Ta2', 'ظ': 'Dha', 'ع': 'Ain', 'غ': 'Ghain', 'ف': 'Fa',
        'ق': 'Qaf', 'ك': 'Kaf', 'ل': 'Lam', 'م': 'Meem', 'ن': 'Noon',
        'ه': 'Ha2', 'و': 'Waw', 'ي': 'Ya',
    }
    if _arabic_font:
        kwargs['fontproperties'] = _arabic_font
        ax.text(x, y, text, **kwargs)
    else:
        display = ''.join(AR_LATIN.get(c, c) for c in text)
        ax.text(x, y, display, **kwargs)

def ar_label(text):
    """Return Latin transliteration for use in titles/labels if no Arabic font."""
    AR_LATIN = {
        'ا': 'Alef', 'ب': 'Ba', 'ت': 'Ta', 'ث': 'Tha', 'ج': 'Jeem',
        'ح': 'Ha', 'خ': 'Kha', 'د': 'Dal', 'ذ': 'Dhal', 'ر': 'Ra',
        'ز': 'Zay', 'س': 'Seen', 'ش': 'Sheen', 'ص': 'Sad', 'ض': 'Dad',
        'ط': 'Ta2', 'ظ': 'Dha', 'ع': 'Ain', 'غ': 'Ghain', 'ف': 'Fa',
        'ق': 'Qaf', 'ك': 'Kaf', 'ل': 'Lam', 'م': 'Meem', 'ن': 'Noon',
        'ه': 'Ha2', 'و': 'Waw', 'ي': 'Ya',
    }
    if _arabic_font:
        return text
    return ''.join(AR_LATIN.get(c, c) for c in text)

Path("plots").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT  = Path("/Users/janahussien/Desktop/eeg_project/data")
SUBJECT_ID = 12       # S12 — best subject, clearest signal
LETTER_A   = 1        # ا  — class A for CSP contrast
LETTER_B   = 2        # ب  — class B for CSP contrast
SFREQ      = 256

# Thesis palette
BG      = '#f0ece4'
WHITE   = '#faf8f4'
PARCH   = '#e8e2d8'
TEXT    = '#2c2824'
MID     = '#6b6258'
SOFT    = '#b5a99a'
ACCENT  = '#8c7b6e'

CH_NAMES = [
    "AF3","F7","F3","FC5","T7","P7","O1",
    "O2","P8","T8","FC6","F4","F8","AF4"
]

ARABIC_LETTERS = [
    'ا','ب','ت','ث','ج','ح','خ','د','ذ','ر',
    'ز','س','ش','ص','ض','ط','ظ','ع','غ','ف',
    'ق','ك','ل','م','ن','ه','و','ي'
]

# Approximate 2D electrode positions for topomap (x, y)
# Normalized to [-1, 1] range, left-right symmetric
CH_POS = {
    "AF3": (-0.3,  0.85), "AF4": ( 0.3,  0.85),
    "F7":  (-0.7,  0.55), "F8":  ( 0.7,  0.55),
    "F3":  (-0.35, 0.60), "F4":  ( 0.35, 0.60),
    "FC5": (-0.55, 0.30), "FC6": ( 0.55, 0.30),
    "T7":  (-0.90, 0.00), "T8":  ( 0.90, 0.00),
    "P7":  (-0.70,-0.45), "P8":  ( 0.70,-0.45),
    "O1":  (-0.25,-0.85), "O2":  ( 0.25,-0.85),
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_trial(subject_id, letter_id, trial_id=1):
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
    return data


def preprocess(data):
    nyq  = SFREQ / 2.0
    b, a = butter(4, [0.5/nyq, 40.0/nyq], btype='band')
    data = filtfilt(b, a, data, axis=1)
    b, a = iirnotch(50.0, Q=30.0, fs=SFREQ)
    data = filtfilt(b, a, data, axis=1)
    data = data - data.mean(axis=0, keepdims=True)
    return data


def get_epoch(subject_id, letter_id, trial_id=1):
    raw   = load_trial(subject_id, letter_id, trial_id)
    proc  = preprocess(raw)
    start = int((5 + 5) * SFREQ)
    end   = start + int(6 * SFREQ)
    return proc[:, start:end]   # (14, 1536)


def load_all_trials(subject_id, letter_id, n_trials=10):
    """Load all available trials for a letter."""
    epochs = []
    for t in range(1, n_trials + 1):
        try:
            ep = get_epoch(subject_id, letter_id, t)
            epochs.append(ep)
        except Exception:
            pass
    return np.array(epochs)  # (N, 14, 1536)


# ─────────────────────────────────────────────────────────────────────────────
# Custom colormap — thesis warm tones
# ─────────────────────────────────────────────────────────────────────────────

def thesis_cmap(diverging=False):
    if diverging:
        return LinearSegmentedColormap.from_list(
            'thesis_div',
            ['#4a6c7a', WHITE, '#8c5a3a'], N=256
        )
    return LinearSegmentedColormap.from_list(
        'thesis_seq',
        [WHITE, PARCH, '#c9b99a', '#8c7b6e', TEXT], N=256
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Riemannian: covariance matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_riemannian():
    print("  [1/3] Riemannian covariance heatmap...")

    epoch = get_epoch(SUBJECT_ID, LETTER_A, trial_id=1)  # (14, 1536)

    # Compute covariance using OAS estimator (same as pyriemann uses internally)
    oas  = OAS()
    oas.fit(epoch.T)   # fit on (n_samples, n_features) = (1536, 14)
    cov  = oas.covariance_   # (14, 14)

    # Normalize to correlation matrix for visualization
    std  = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
    ax.set_facecolor(WHITE)

    cmap = thesis_cmap(diverging=True)
    im   = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Correlation', fontsize=10, color=MID)
    cbar.ax.tick_params(colors=MID, labelsize=8)

    ax.set_xticks(range(14))
    ax.set_yticks(range(14))
    ax.set_xticklabels(CH_NAMES, fontsize=8, rotation=45, ha='right', color=MID)
    ax.set_yticklabels(CH_NAMES, fontsize=8, color=MID)

    # Annotate cells with values
    for i in range(14):
        for j in range(14):
            v = corr[i, j]
            c = 'white' if abs(v) > 0.5 else TEXT
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=5.5, color=c, fontweight='bold')

    ax.set_title(
        f'Riemannian Geometry — Channel Covariance Matrix\n'
        f'S{SUBJECT_ID:02d} · Letter {ar_label(ARABIC_LETTERS[LETTER_A-1])} · Trial 1 · '
        f'Imagination window',
        fontsize=12, fontweight='bold', color=TEXT, pad=14,
        fontfamily='serif'
    )

    # Spine styling
    for spine in ax.spines.values():
        spine.set_color(SOFT)

    ax.tick_params(colors=MID)

    plt.tight_layout()
    out = Path("plots/feature_riemannian.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"     Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — PLV connectivity matrix + topomap
# ─────────────────────────────────────────────────────────────────────────────

def compute_plv(epoch, lo, hi):
    """Compute PLV matrix for epoch (14, n_samples) in given band."""
    nyq  = SFREQ / 2.0
    b, a = butter(4, [lo/nyq, hi/nyq], btype='band')
    filt = filtfilt(b, a, epoch, axis=1)
    phase = np.angle(hilbert(filt, axis=1))   # (14, n_samples)
    n_ch = epoch.shape[0]
    plv  = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i, n_ch):
            diff     = phase[i] - phase[j]
            plv[i,j] = abs(np.mean(np.exp(1j * diff)))
            plv[j,i] = plv[i,j]
    return plv


def plot_plv():
    print("  [2/3] PLV connectivity matrix + topomap...")

    epoch = get_epoch(SUBJECT_ID, LETTER_A, trial_id=1)

    # Average PLV across alpha and beta (as in your pipeline)
    plv_alpha = compute_plv(epoch, 8.0, 13.0)
    plv_beta  = compute_plv(epoch, 13.0, 30.0)
    plv_avg   = (plv_alpha + plv_beta) / 2.0

    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.1], wspace=0.35)

    # ── Left: PLV heatmap ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(WHITE)

    cmap = thesis_cmap(diverging=False)
    im   = ax1.imshow(plv_avg, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    cbar = plt.colorbar(im, ax=ax1, fraction=0.04, pad=0.02)
    cbar.set_label('PLV (0 = no sync, 1 = perfect)', fontsize=9, color=MID)
    cbar.ax.tick_params(colors=MID, labelsize=8)

    ax1.set_xticks(range(14))
    ax1.set_yticks(range(14))
    ax1.set_xticklabels(CH_NAMES, fontsize=8, rotation=45, ha='right', color=MID)
    ax1.set_yticklabels(CH_NAMES, fontsize=8, color=MID)
    ax1.set_title('PLV Matrix\n(alpha + beta average)',
                  fontsize=11, fontweight='bold', color=TEXT, pad=10,
                  fontfamily='serif')
    for spine in ax1.spines.values():
        spine.set_color(SOFT)

    # ── Right: topomap-style connectivity diagram ──────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(WHITE)
    ax2.set_aspect('equal')

    # Draw head circle
    head = plt.Circle((0, 0), 1.0, fill=False, color=SOFT,
                       linewidth=1.5, zorder=1)
    ax2.add_patch(head)

    # Nose
    ax2.plot([0, 0], [1.0, 1.12], color=SOFT, linewidth=1.5, zorder=1)

    # Draw strong connections (PLV > threshold)
    threshold = np.percentile(plv_avg[np.triu_indices(14, k=1)], 75)
    max_plv   = plv_avg.max()

    for i in range(14):
        for j in range(i+1, 14):
            v = plv_avg[i, j]
            if v > threshold:
                x1, y1 = CH_POS[CH_NAMES[i]]
                x2, y2 = CH_POS[CH_NAMES[j]]
                alpha  = (v - threshold) / (max_plv - threshold + 1e-10)
                lw     = 0.5 + 3.0 * alpha
                color  = ACCENT
                ax2.plot([x1, x2], [y1, y2],
                         color=color, alpha=0.3 + 0.5 * alpha,
                         linewidth=lw, zorder=2)

    # Draw electrodes
    for ch in CH_NAMES:
        x, y = CH_POS[ch]
        ax2.scatter(x, y, s=120, color=ACCENT, zorder=4,
                    edgecolors=WHITE, linewidths=1.5)
        offset_x = x * 0.18
        offset_y = y * 0.18
        ax2.text(x + offset_x, y + offset_y, ch,
                 fontsize=7, ha='center', va='center',
                 color=TEXT, fontweight='bold', zorder=5)

    ax2.set_xlim(-1.35, 1.35)
    ax2.set_ylim(-1.25, 1.35)
    ax2.axis('off')
    ax2.set_title(f'Connectivity Topomap\n(top 25% strongest PLV links)',
                  fontsize=11, fontweight='bold', color=TEXT, pad=10,
                  fontfamily='serif')

    fig.suptitle(
        f'Phase-Locking Value (PLV) — Neural Synchrony\n'
        f'S{SUBJECT_ID:02d} · Letter {ar_label(ARABIC_LETTERS[LETTER_A-1])} · Alpha & Beta bands',
        fontsize=13, fontweight='bold', color=TEXT, y=1.02,
        fontfamily='serif'
    )

    plt.tight_layout()
    out = Path("plots/feature_plv.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"     Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — CSP: spatial filter weights + log-variance separation
# ─────────────────────────────────────────────────────────────────────────────

def compute_csp_filters(X_a, X_b, n_components=4):
    """Simple OvO CSP between class A and class B."""
    def cov_mean(X):
        covs = np.array([np.cov(t) for t in X])
        return covs.mean(axis=0)

    C_a = cov_mean(X_a)
    C_b = cov_mean(X_b)
    Cc  = C_a + C_b

    vals, vecs = np.linalg.eigh(Cc)
    W   = vecs @ np.diag(1.0 / np.sqrt(vals + 1e-10)) @ vecs.T
    S_a = W @ C_a @ W.T
    evals, evecs = np.linalg.eigh(S_a)
    order   = np.argsort(evals)[::-1]
    filters = evecs[:, order[:n_components]].T @ W
    return filters   # (n_components, 14)


def plot_csp():
    print("  [3/3] CSP spatial filters + variance separation...")

    X_a = load_all_trials(SUBJECT_ID, LETTER_A)
    X_b = load_all_trials(SUBJECT_ID, LETTER_B)

    if len(X_a) == 0 or len(X_b) == 0:
        print("     Not enough trials — skipping CSP plot")
        return

    n_comp   = 4
    filters  = compute_csp_filters(X_a, X_b, n_comp)

    # Log-variance of projected signals
    def log_var(X, filt):
        proj = np.einsum('cd,tds->tcs', filt[np.newaxis], X)[0]
        return np.log(proj.var(axis=1) + 1e-10)   # (n_trials, n_comp)

    lv_a = np.array([np.log(np.var(filters @ ep, axis=1) + 1e-10)
                     for ep in X_a])   # (Na, n_comp)
    lv_b = np.array([np.log(np.var(filters @ ep, axis=1) + 1e-10)
                     for ep in X_b])   # (Nb, n_comp)

    fig = plt.figure(figsize=(15, 6), facecolor=BG)
    gs  = gridspec.GridSpec(2, n_comp, hspace=0.45, wspace=0.3)

    colors_ab = ['#7a9478', '#8c5a3a']   # sage vs bark
    ar_a = ARABIC_LETTERS[LETTER_A - 1]
    ar_b = ARABIC_LETTERS[LETTER_B - 1]

    # ── Top row: filter weight topomaps ───────────────────────────────────
    for c in range(n_comp):
        ax = fig.add_subplot(gs[0, c])
        ax.set_facecolor(WHITE)
        ax.set_aspect('equal')

        weights = filters[c]   # (14,)
        w_norm  = weights / (np.abs(weights).max() + 1e-10)

        # Head circle
        head = plt.Circle((0,0), 1.0, fill=False,
                          color=SOFT, linewidth=1.2, zorder=1)
        ax.add_patch(head)
        ax.plot([0,0],[1.0,1.1], color=SOFT, linewidth=1.2)

        # Electrodes colored by weight
        for i, ch in enumerate(CH_NAMES):
            x, y = CH_POS[ch]
            w    = w_norm[i]
            # positive = warm, negative = cool
            if w >= 0:
                color = '#8c5a3a'
                alpha = 0.2 + 0.7 * abs(w)
            else:
                color = '#4a6c7a'
                alpha = 0.2 + 0.7 * abs(w)
            size = 80 + 180 * abs(w)
            ax.scatter(x, y, s=size, color=color, alpha=alpha,
                       zorder=3, edgecolors=WHITE, linewidths=1)
            ax.text(x, y, ch, fontsize=6, ha='center', va='center',
                    color=TEXT, fontweight='bold', zorder=4)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.2, 1.3)
        ax.axis('off')
        ax.set_title(f'CSP {c+1}', fontsize=10,
                     fontweight='bold', color=TEXT,
                     fontfamily='serif', pad=6)

    # ── Bottom row: log-variance bars ─────────────────────────────────────
    for c in range(n_comp):
        ax = fig.add_subplot(gs[1, c])
        ax.set_facecolor(WHITE)

        mean_a = lv_a[:, c].mean()
        mean_b = lv_b[:, c].mean()
        std_a  = lv_a[:, c].std()
        std_b  = lv_b[:, c].std()

        x = [0, 1]
        means  = [mean_a, mean_b]
        stds   = [std_a,  std_b]
        clrs   = colors_ab

        bars = ax.bar(x, means, color=clrs, width=0.5,
                      alpha=0.82, zorder=2)
        ax.errorbar(x, means, yerr=stds, fmt='none',
                    color=TEXT, capsize=4, linewidth=1.2, zorder=3)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([ar_label(ar_a), ar_label(ar_b)], fontsize=12)
        ax.set_ylabel('Log-variance', fontsize=8, color=MID)
        ax.tick_params(colors=MID, labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(SOFT)
        ax.spines['bottom'].set_color(SOFT)

        # Separation annotation
        sep = abs(mean_a - mean_b)
        ax.text(0.5, max(means) + max(stds) * 0.5,
                f'Δ={sep:.2f}',
                ha='center', fontsize=8, color=MID,
                transform=ax.transData)

    # Legend
    patch_a = mpatches.Patch(color=colors_ab[0], alpha=0.82, label=ar_label(ar_a))
    patch_b = mpatches.Patch(color=colors_ab[1], alpha=0.82, label=ar_label(ar_b))
    fig.legend(handles=[patch_a, patch_b], loc='lower center',
               ncol=2, fontsize=10, framealpha=0.9,
               facecolor=WHITE, edgecolor=SOFT)

    fig.suptitle(
        f'Adaptive CSP — Spatial Filters & Log-Variance Separation\n'
        f'S{SUBJECT_ID:02d} · {ar_label(ar_a)} vs {ar_label(ar_b)} · {n_comp} components',
        fontsize=13, fontweight='bold', color=TEXT,
        fontfamily='serif', y=1.01
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = Path("plots/feature_csp.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"     Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nGenerating feature visualizations for S{SUBJECT_ID:02d}...")
    print(f"Data root: {DATA_ROOT}\n")

    try:
        plot_riemannian()
    except Exception as e:
        print(f"  Riemannian FAILED: {e}")

    try:
        plot_plv()
    except Exception as e:
        print(f"  PLV FAILED: {e}")

    try:
        plot_csp()
    except Exception as e:
        import traceback
        print(f"  CSP FAILED: {e}")
        traceback.print_exc()

    print("\nDone! Check plots/ folder.")