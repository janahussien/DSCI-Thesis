"""
generate_plots.py
=================
Thesis visualisation plots — updated for band optimization pipeline.
Matches the presentation colour scheme (warm beige / taupe / charcoal).

Plots produced (saved to plots/ folder):
  1. per_subject_accuracy.png   — baseline vs band-opt accuracy, sorted
  2. band_wins.png              — which frequency band won per subject
  3. confusion_matrix.png       — letter confusion matrix (aggregated CV)
  4. class_distribution.png     — trials per Arabic letter after preprocessing
  5. accuracy_distribution.png  — violin / strip plot baseline vs band-opt
  6. letter_difficulty.png      — per-letter recognition accuracy bar chart
  7. confused_pairs.png         — top confused letter pairs
  8. letter_bubble_map.png      — Arabic alphabet grid coloured by accuracy

Usage:
    python generate_plots.py                   # all plots
    python generate_plots.py --skip_cm         # skip confusion matrix (slow)
    python generate_plots.py --skip_letters    # skip letter insight plots
    python generate_plots.py --subject 12      # confusion matrix single subject
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import shutil
import argparse
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def ar(text):
        return get_display(arabic_reshaper.reshape(text))
except ImportError:
    def ar(text):
        return text

from config import CONFIG
from preprocessing import load_subject_data, preprocess_pipeline
from features import (
    riemannian_features, band_power_features,
    connectivity_features, adaptive_csp_features,
)
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

# ── Output folder ──────────────────────────────────────────────────────────
PLOTS_DIR = Path("plots")
if PLOTS_DIR.exists():
    shutil.rmtree(PLOTS_DIR)
PLOTS_DIR.mkdir()

EXCLUDED = {22, 29}

# ── Thesis colour palette ──────────────────────────────────────────────────
BG       = '#f0ece4'   # warm beige background
BG2      = '#e8e2d8'   # card surface
WHITE    = '#faf8f4'   # lightest surface
TEXT     = '#2c2824'   # charcoal
TEXT_MED = '#6b6258'   # mid brown
SPINE    = '#b5a99a'   # soft border
GRID     = '#ddd5c8'   # grid lines

C_BASE   = '#9aacb5'   # muted steel — baseline bars
C_BAND   = '#8c7b6e'   # accent brown — band-opt bars
C_PAPER  = '#a89e8e'   # tan — paper baseline line
C_MEAN   = '#6b6258'   # dark — our mean line
C_NEG    = '#b07878'   # muted red — hard letters
C_MID    = '#a89878'   # tan — medium letters
C_POS    = '#7a9478'   # muted green — easy letters

BAND_COLORS = {
    'broadband':   '#9aacb5',
    'delta_theta': '#a89898',
    'alpha':       '#8c9e7e',
    'beta':        '#8c7b6e',
    'gamma':       '#b5a99a',
    'alpha_beta':  '#7e8c7e',
    'multiband':   '#2c2824',
}

# ── Matplotlib theme ───────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    WHITE,
    'axes.edgecolor':    SPINE,
    'axes.labelcolor':   TEXT_MED,
    'axes.titlecolor':   TEXT,
    'xtick.color':       TEXT_MED,
    'ytick.color':       TEXT_MED,
    'text.color':        TEXT,
    'grid.color':        GRID,
    'grid.linewidth':    0.6,
    'font.family':       'serif',
    'font.size':         10,
    'axes.titlesize':    13,
    'axes.labelsize':    10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'legend.facecolor':  WHITE,
    'legend.edgecolor':  SPINE,
    'legend.labelcolor': TEXT_MED,
    'legend.framealpha': 0.9,
})

# ── Arabic letters ─────────────────────────────────────────────────────────
RAW_ARABIC = [
    'ا','ب','ت','ث','ج','ح','خ','د','ذ','ر',
    'ز','س','ش','ص','ض','ط','ظ','ع','غ','ف',
    'ق','ك','ل','م','ن','ه','و','ي'
]
ARABIC_LETTERS = [ar(l) for l in RAW_ARABIC]

# ── Band optimization results (your final pipeline) ────────────────────────
# Format: subject_id → {base, band_best, band_config, hand}
RESULTS = {
    1:  {'base': 75.17, 'band': 82.69, 'config': 'multiband',   'hand': 'R'},
    2:  {'base': 83.69, 'band': 93.65, 'config': 'beta',        'hand': 'R'},
    3:  {'base': 74.78, 'band': 79.63, 'config': 'multiband',   'hand': 'R'},
    4:  {'base': 63.61, 'band': 73.25, 'config': 'broadband',   'hand': 'L'},
    5:  {'base': 67.75, 'band': 84.45, 'config': 'multiband',   'hand': 'R'},
    6:  {'base': 69.04, 'band': 72.13, 'config': 'multiband',   'hand': 'R'},
    7:  {'base': 66.52, 'band': 71.77, 'config': 'multiband',   'hand': 'R'},
    8:  {'base': 81.58, 'band': 90.12, 'config': 'alpha_beta',  'hand': 'R'},
    9:  {'base': 78.64, 'band': 81.86, 'config': 'alpha_beta',  'hand': 'L'},
    10: {'base': 55.19, 'band': 66.00, 'config': 'multiband',   'hand': 'R'},
    11: {'base': 78.37, 'band': 83.05, 'config': 'gamma',       'hand': 'R'},
    12: {'base': 91.02, 'band': 95.10, 'config': 'beta',        'hand': 'R'},
    13: {'base': 71.34, 'band': 78.85, 'config': 'multiband',   'hand': 'R'},
    14: {'base': 81.30, 'band': 83.48, 'config': 'multiband',   'hand': 'R'},
    15: {'base': 49.50, 'band': 58.33, 'config': 'broadband',   'hand': 'L'},
    16: {'base': 54.94, 'band': 71.74, 'config': 'multiband',   'hand': 'R'},
    17: {'base': 63.12, 'band': 78.14, 'config': 'beta',        'hand': 'L'},
    18: {'base': 68.09, 'band': 81.52, 'config': 'multiband',   'hand': 'R'},
    19: {'base': 71.58, 'band': 82.57, 'config': 'beta',        'hand': 'R'},
    20: {'base': 49.89, 'band': 71.81, 'config': 'multiband',   'hand': 'R'},
    21: {'base': 69.76, 'band': 75.64, 'config': 'multiband',   'hand': 'R'},
    23: {'base': 67.37, 'band': 84.58, 'config': 'multiband',   'hand': 'R'},
    24: {'base': 53.56, 'band': 58.95, 'config': 'broadband',   'hand': 'R'},
    25: {'base': 56.05, 'band': 78.57, 'config': 'multiband',   'hand': 'R'},
    26: {'base': 72.39, 'band': 81.58, 'config': 'multiband',   'hand': 'R'},
    27: {'base': 46.11, 'band': 62.60, 'config': 'alpha_beta',  'hand': 'R'},
    28: {'base': 64.94, 'band': 76.72, 'config': 'multiband',   'hand': 'R'},
    30: {'base': 60.77, 'band': 80.77, 'config': 'multiband',   'hand': 'R'},
}

PAPER_BASELINE = 74.80


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bandpass(X, lo, hi, fs):
    nyq = fs / 2.0
    b, a = butter(4, [max(lo, 0.1) / nyq, min(hi, nyq - 0.1) / nyq], btype='band')
    return filtfilt(b, a, X, axis=2)

def _extract(X, y):
    Xr = np.nan_to_num(riemannian_features(X))
    Xb = np.nan_to_num(band_power_features(X))
    Xp = np.nan_to_num(connectivity_features(X))
    Xc = np.nan_to_num(adaptive_csp_features(X, y))
    return np.concatenate([Xr, Xb, Xp, Xc], axis=1)

def _savefig(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"     Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Per-subject accuracy: baseline vs band-opt, sorted
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_subject_accuracy():
    print("  [1] Per-subject accuracy chart...")

    sids  = sorted(RESULTS.keys())
    bases = [RESULTS[s]['base'] for s in sids]
    bands = [RESULTS[s]['band'] for s in sids]
    hands = [RESULTS[s]['hand'] for s in sids]

    # Sort by band-opt accuracy
    order = np.argsort(bands)
    sids  = [sids[i]  for i in order]
    bases = [bases[i] for i in order]
    bands = [bands[i] for i in order]
    hands = [hands[i] for i in order]
    deltas = [b - a for a, b in zip(bases, bands)]

    x = np.arange(len(sids))
    fig, ax = plt.subplots(figsize=(17, 6.5))
    fig.patch.set_facecolor(BG)

    # Stacked bars: base + delta
    ax.bar(x, bases,  color=C_BASE, width=0.62,
           label='Baseline (broadband + LDA svd)', zorder=2, alpha=0.9)
    ax.bar(x, deltas, bottom=bases, color=C_BAND, width=0.62,
           label='Band optimization gain', zorder=2, alpha=0.88)

    # Paper baseline
    ax.axhline(PAPER_BASELINE, color=C_PAPER, linewidth=1.8,
               linestyle='--', zorder=3,
               label=f'Dataset paper baseline ({PAPER_BASELINE}%)')

    # Our mean
    our_mean = np.mean(bands)
    ax.axhline(our_mean, color=C_MEAN, linewidth=1.8,
               linestyle=':', zorder=3,
               label=f'Our mean ({our_mean:.1f}%)')

    # Left-handed markers
    for i, h in enumerate(hands):
        if h == 'L':
            ax.text(i, bands[i] + 1.0, '◄', ha='center',
                    fontsize=7, color=TEXT_MED)

    # Value labels on top
    for i, v in enumerate(bands):
        ax.text(i, v + 0.5, f'{v:.0f}', ha='center', va='bottom',
                fontsize=6.5, color=TEXT, fontweight='bold',
                fontfamily='sans-serif')

    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s:02d}' for s in sids],
                       rotation=45, ha='right', fontsize=8,
                       fontfamily='sans-serif')
    ax.set_ylabel('Accuracy (%)', fontfamily='sans-serif')
    ax.set_ylim(0, 104)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title('Per-Subject Classification Accuracy\nBaseline vs Band Optimization Pipeline',
                 fontweight='bold', pad=14)
    ax.legend(fontsize=9, loc='upper left')

    # Annotation: 28/28
    ax.text(0.98, 0.04, '28 / 28 subjects improved',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color=C_BAND, fontfamily='sans-serif',
            fontstyle='italic')

    plt.tight_layout()
    _savefig(fig, 'per_subject_accuracy.png')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Which band won per subject (horizontal stacked legend)
# ─────────────────────────────────────────────────────────────────────────────

def plot_band_wins():
    print("  [2] Band wins chart...")

    from collections import Counter
    configs = [RESULTS[s]['config'] for s in RESULTS]
    counts  = Counter(configs)
    # clean labels
    label_map = {
        'broadband':   'Broadband (0.5–40)',
        'delta_theta': 'Delta+Theta (0.5–8)',
        'alpha':       'Alpha (8–13)',
        'beta':        'Beta (13–30)',
        'gamma':       'Gamma (30–40)',
        'alpha_beta':  'Alpha+Beta (8–30)',
        'multiband':   'Multiband (all bands)',
    }
    # Sort by count
    bands_sorted = sorted(counts.keys(), key=lambda b: counts[b], reverse=True)
    labels = [label_map.get(b, b) for b in bands_sorted]
    values = [counts[b] for b in bands_sorted]
    colors = [BAND_COLORS.get(b, C_BAND) for b in bands_sorted]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)

    y = np.arange(len(bands_sorted))
    bars = ax.barh(y, values, color=colors, height=0.58,
                   alpha=0.88, zorder=2)

    for i, (val, band) in enumerate(zip(values, bands_sorted)):
        ax.text(val + 0.15, i, str(val), va='center',
                fontsize=12, color=TEXT, fontweight='bold',
                fontfamily='sans-serif')
        # show % of subjects
        ax.text(val + 0.65, i - 0.25,
                f'{val/len(RESULTS)*100:.0f}% of subjects',
                va='center', fontsize=8, color=TEXT_MED,
                fontfamily='sans-serif')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, fontfamily='sans-serif')
    ax.set_xlabel('Number of subjects', fontfamily='sans-serif')
    ax.set_xlim(0, max(values) + 3)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.set_title('Most Discriminative Frequency Band Per Subject\n'
                 'Winning band selected by per-subject CV accuracy',
                 fontweight='bold', pad=14)

    # Key insight annotation
    ax.text(0.98, 0.04,
            'No single band dominates — per-subject adaptation is essential',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8.5, color=TEXT_MED, fontstyle='italic',
            fontfamily='sans-serif')

    plt.tight_layout()
    _savefig(fig, 'band_wins.png')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

BANDS_DICT = {
    'broadband':   (0.5, 40.0),
    'delta_theta': (0.5,  8.0),
    'alpha':       (8.0, 13.0),
    'beta':        (13.0, 30.0),
    'gamma':       (30.0, 40.0),
    'alpha_beta':  (8.0,  30.0),
}

def _run_cv_for_subject(sid):
    """Run band-opt pipeline CV for one subject, return (y_true, y_pred)."""
    config = RESULTS[sid]['config']
    records = load_subject_data(sid, CONFIG['data_root'])
    X, y    = preprocess_pipeline(records)
    sfreq   = CONFIG['sfreq']

    if config == 'multiband':
        feats_list = []
        for lo, hi in BANDS_DICT.values():
            Xf = _bandpass(X, lo, hi, sfreq)
            feats_list.append(_extract(Xf, y))
        X_feat = np.concatenate(feats_list, axis=1)
        k = 500
    else:
        lo, hi = BANDS_DICT[config]
        X_filt = _bandpass(X, lo, hi, sfreq)
        X_feat = _extract(X_filt, y)
        k = None   # no ANOVA for single band — already fast

    n_splits = max(min(CONFIG['cv_folds'], int(np.bincount(y).min())), 2)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=CONFIG['random_state'])
    all_true, all_pred = [], []
    for tr_idx, te_idx in skf.split(X_feat, y):
        Xtr, Xte = X_feat[tr_idx], X_feat[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        if k is not None:
            sel  = SelectKBest(f_classif, k=min(k, Xtr.shape[1]))
            Xtr  = sel.fit_transform(Xtr, ytr)
            Xte  = sel.transform(Xte)
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf.fit(Xtr, ytr)
        all_true.extend(yte.tolist())
        all_pred.extend(clf.predict(Xte).tolist())
    return np.array(all_true), np.array(all_pred)


def collect_predictions(subject_ids):
    all_true, all_pred = [], []
    for sid in subject_ids:
        if sid in EXCLUDED:
            continue
        try:
            yt, yp = _run_cv_for_subject(sid)
            all_true.extend(yt.tolist())
            all_pred.extend(yp.tolist())
            print(f"     S{sid:02d} ✓", end='  ', flush=True)
        except Exception as e:
            print(f"     S{sid:02d} ✗ ({e})", end='  ', flush=True)
    print()
    return np.array(all_true), np.array(all_pred)


def plot_confusion_matrix(subject_ids=None):
    print("  [3] Confusion matrix...")
    if subject_ids is None:
        subject_ids = [12, 2, 8, 14, 3]

    y_true, y_pred = collect_predictions(subject_ids)
    if len(y_true) == 0:
        print("     No data — skipping.")
        return

    n = CONFIG['n_letters']
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=(13, 11))
    fig.patch.set_facecolor(BG)

    cmap = LinearSegmentedColormap.from_list(
        'thesis', [WHITE, BG2, C_BAND, TEXT], N=256)
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Normalised recall', fontsize=10, color=TEXT_MED)
    cbar.ax.tick_params(colors=TEXT_MED)

    ticks = list(range(n))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ARABIC_LETTERS, fontsize=9)
    ax.set_yticklabels(ARABIC_LETTERS, fontsize=9)
    ax.set_xlabel('Predicted letter', fontsize=11, labelpad=10,
                  fontfamily='sans-serif')
    ax.set_ylabel('True letter',      fontsize=11, labelpad=10,
                  fontfamily='sans-serif')

    subj_str = ', '.join([f'S{s:02d}' for s in subject_ids])
    ax.set_title(f'Confusion Matrix — Arabic Alphabet Classification\n'
                 f'Band optimization pipeline  ·  {subj_str}',
                 fontweight='bold', pad=14)

    plt.tight_layout()
    _savefig(fig, 'confusion_matrix.png')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Class distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(sample_subjects=None):
    print("  [4] Class distribution...")
    if sample_subjects is None:
        sample_subjects = [1, 5, 10, 12, 27]

    colors = [C_BASE, C_BAND, '#9e9e7a', '#a89880', '#b5a99a']
    fig, axes = plt.subplots(len(sample_subjects), 1,
                             figsize=(15, 3.2 * len(sample_subjects)))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Trials per Arabic Letter After Preprocessing',
                 fontsize=14, fontweight='bold', y=1.01)

    for idx, sid in enumerate(sample_subjects):
        ax = axes[idx]
        ax.set_facecolor(WHITE)
        try:
            records = load_subject_data(sid, CONFIG['data_root'])
            X, y    = preprocess_pipeline(records)
            counts  = np.bincount(y, minlength=CONFIG['n_letters'])
            x = np.arange(CONFIG['n_letters'])
            ax.bar(x, counts, color=colors[idx % len(colors)],
                   alpha=0.85, width=0.65, zorder=2)
            ax.axhline(counts.mean(), color=TEXT, linewidth=1.2,
                       linestyle='--', alpha=0.55,
                       label=f'Mean: {counts.mean():.1f} trials', zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(ARABIC_LETTERS, fontsize=9)
            ax.set_ylabel('Trials', fontfamily='sans-serif')
            h = RESULTS.get(sid, {}).get('hand', '?')
            ax.set_title(f'S{sid:02d} ({h}-handed) — {X.shape[0]} trials kept',
                         fontsize=10, fontweight='bold', pad=6)
            ax.yaxis.grid(True, zorder=0)
            ax.set_axisbelow(True)
            ax.legend(fontsize=8, framealpha=0.9)
        except Exception as e:
            ax.text(0.5, 0.5, f'S{sid:02d}: {e}',
                    transform=ax.transAxes, ha='center', color=TEXT_MED)

    plt.tight_layout()
    _savefig(fig, 'class_distribution.png')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Accuracy distribution: baseline vs band-opt violin
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_distribution():
    print("  [5] Accuracy distribution...")

    bases = [RESULTS[s]['base'] for s in RESULTS]
    bands = [RESULTS[s]['band'] for s in RESULTS]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(WHITE)

    parts = ax.violinplot([bases, bands], positions=[1, 2],
                          showmeans=True, showmedians=True,
                          widths=0.55)

    colors = [C_BASE, C_BAND]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor(SPINE)
        pc.set_alpha(0.72)

    for key in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
        parts[key].set_color(TEXT)
        parts[key].set_linewidth(1.4)

    np.random.seed(42)
    jitter = 0.06
    for i, (data, color) in enumerate([(bases, C_BASE), (bands, C_BAND)]):
        xpos = (i + 1) + np.random.uniform(-jitter, jitter, len(data))
        ax.scatter(xpos, data, color=color, alpha=0.65,
                   s=30, zorder=4, edgecolors=TEXT, linewidths=0.4)

    ax.axhline(PAPER_BASELINE, color=C_PAPER, linewidth=1.8,
               linestyle='--', zorder=3,
               label=f'Dataset paper baseline ({PAPER_BASELINE}%)')
    ax.axhline(3.57, color=SPINE, linewidth=1.2,
               linestyle=':', zorder=2, alpha=0.7,
               label='Chance level (3.6%)')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        ['Baseline\n(broadband + LDA svd)',
         'Band Optimization\n(per-subject adaptive)'],
        fontsize=10, fontfamily='sans-serif')
    ax.set_ylabel('Accuracy (%)', fontfamily='sans-serif')
    ax.set_ylim(30, 104)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    # Mean labels
    for pos, vals in [(1, bases), (2, bands)]:
        ax.text(pos, np.mean(vals) + 2.5, f'Mean: {np.mean(vals):.1f}%',
                ha='center', fontsize=9.5, color=TEXT, fontweight='bold',
                fontfamily='sans-serif')

    ax.set_title('Accuracy Distribution — 28 Subjects\nBaseline vs Band Optimization',
                 fontweight='bold', pad=14)
    ax.legend(fontsize=9)

    plt.tight_layout()
    _savefig(fig, 'accuracy_distribution.png')


# ─────────────────────────────────────────────────────────────────────────────
# Per-letter accuracy collection (for plots 6, 7, 8)
# ─────────────────────────────────────────────────────────────────────────────

def collect_letter_accuracies(subject_ids=None):
    if subject_ids is None:
        subject_ids = [s for s in sorted(RESULTS.keys()) if s not in EXCLUDED]

    print("     Collecting per-letter predictions...")
    all_true, all_pred = [], []
    for sid in subject_ids:
        try:
            yt, yp = _run_cv_for_subject(sid)
            all_true.extend(yt.tolist())
            all_pred.extend(yp.tolist())
            print(f"     S{sid:02d} ✓", end='  ', flush=True)
        except Exception as e:
            print(f"     S{sid:02d} ✗ ({e})", end='  ', flush=True)
    print()

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    n = CONFIG['n_letters']
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    letter_accs = np.diag(cm_norm)
    return letter_accs, cm_norm


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Letter difficulty bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_letter_difficulty(letter_accs):
    print("  [6] Letter difficulty chart...")

    letters = ARABIC_LETTERS
    order   = np.argsort(letter_accs)
    s_lett  = [letters[i] for i in order]
    s_accs  = letter_accs[order]
    colors  = [C_NEG if a < 0.5 else C_MID if a < 0.7 else C_POS
               for a in s_accs]

    fig, ax = plt.subplots(figsize=(15, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(WHITE)

    bars = ax.bar(range(len(s_lett)), s_accs * 100,
                  color=colors, width=0.68, zorder=2, alpha=0.88)

    # Arabic letter on top of each bar
    for i, (l, a) in enumerate(zip(s_lett, s_accs)):
        ax.text(i, a * 100 + 0.8, l, ha='center', va='bottom',
                fontsize=14, color=TEXT, fontweight='bold')
        ax.text(i, a * 100 * 0.5, f'{a*100:.0f}',
                ha='center', va='center', fontsize=7,
                color=WHITE if a < 0.65 else TEXT,
                fontfamily='sans-serif', fontweight='bold')

    ax.axhline(np.mean(s_accs) * 100, color=TEXT_MED, linewidth=1.4,
               linestyle='--', zorder=3,
               label=f'Mean: {np.mean(s_accs)*100:.1f}%')

    easy  = mpatches.Patch(color=C_POS, label='Easy  (≥ 70%)', alpha=0.88)
    mid_p = mpatches.Patch(color=C_MID, label='Medium (50–70%)', alpha=0.88)
    hard  = mpatches.Patch(color=C_NEG, label='Hard   (< 50%)', alpha=0.88)
    ax.legend(handles=[easy, mid_p, hard], fontsize=9)

    ax.set_xticks([])
    ax.set_ylabel('Recognition Accuracy (%)', fontfamily='sans-serif')
    ax.set_ylim(0, 112)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title('Arabic Letter Recognition Difficulty\n'
                 'Sorted easiest → hardest  ·  aggregated across all subjects',
                 fontweight='bold', pad=14)

    plt.tight_layout()
    _savefig(fig, 'letter_difficulty.png')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7 — Most confused letter pairs
# ─────────────────────────────────────────────────────────────────────────────

def plot_confused_pairs(cm_norm):
    print("  [7] Confused pairs chart...")

    letters = ARABIC_LETTERS
    pairs   = []
    n = cm_norm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm_norm[i, j] > 0.01:
                pairs.append((i, j, cm_norm[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top = pairs[:18]

    pair_labels = [f'{letters[i]}  →  {letters[j]}' for i, j, _ in top]
    pair_vals   = [v * 100 for _, _, v in top]
    colors      = [C_NEG if v > 15 else C_MID if v > 8 else C_BASE
                   for v in pair_vals]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(WHITE)

    ax.barh(range(len(top)), pair_vals, color=colors,
            height=0.62, alpha=0.88, zorder=2)

    for i, val in enumerate(pair_vals):
        ax.text(val + 0.2, i, f'{val:.1f}%',
                va='center', fontsize=9, color=TEXT,
                fontweight='bold', fontfamily='sans-serif')

    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=12)
    ax.set_xlabel('Confusion rate (%)', fontfamily='sans-serif')
    ax.set_xlim(0, max(pair_vals) * 1.18)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.set_title('Most Commonly Confused Letter Pairs\n'
                 'True letter  →  mistaken as',
                 fontweight='bold', pad=14)

    h_p = mpatches.Patch(color=C_NEG,  label='High (> 15%)',  alpha=0.88)
    m_p = mpatches.Patch(color=C_MID,  label='Medium (8–15%)', alpha=0.88)
    l_p = mpatches.Patch(color=C_BASE, label='Low (< 8%)',    alpha=0.88)
    ax.legend(handles=[h_p, m_p, l_p], fontsize=9)

    plt.tight_layout()
    _savefig(fig, 'confused_pairs.png')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 8 — Letter bubble map
# ─────────────────────────────────────────────────────────────────────────────

def plot_letter_bubble_map(letter_accs):
    print("  [8] Letter bubble map...")

    letters = ARABIC_LETTERS
    n_cols  = 7
    n_rows  = int(np.ceil(len(letters) / n_cols))

    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-0.8, n_cols - 0.2)
    ax.set_ylim(-0.8, n_rows - 0.2)
    ax.axis('off')

    for idx, (letter, acc) in enumerate(zip(letters, letter_accs)):
        col = idx % n_cols
        row = n_rows - 1 - (idx // n_cols)

        size  = 2000 * acc + 300
        color = C_NEG if acc < 0.5 else C_MID if acc < 0.7 else C_POS

        ax.scatter(col, row, s=size, color=color, alpha=0.78,
                   edgecolors=SPINE, linewidths=1.2, zorder=2)

        ax.text(col, row + 0.08, letter,
                ha='center', va='center',
                fontsize=17, color=TEXT, fontweight='bold', zorder=3)

        ax.text(col, row - 0.32, f'{acc*100:.0f}%',
                ha='center', va='center',
                fontsize=8, color=TEXT_MED, zorder=3,
                fontfamily='sans-serif')

    ax.set_title(
        'Arabic Alphabet — EEG Recognition Accuracy\n'
        'Bubble size = accuracy  ·  Green = easy  ·  Red = hard',
        fontsize=12, fontweight='bold', color=TEXT, pad=14
    )

    # Legend
    for acc_v, lbl, clr in [(0.85, 'Easy (≥70%)', C_POS),
                             (0.60, 'Medium (50–70%)', C_MID),
                             (0.35, 'Hard (<50%)', C_NEG)]:
        ax.scatter([], [], s=2000 * acc_v + 300, color=clr,
                   alpha=0.78, edgecolors=SPINE, linewidths=1.2, label=lbl)
    ax.legend(loc='lower right', fontsize=9, scatterpoints=1,
              labelspacing=1.1, framealpha=0.9)

    plt.tight_layout()
    _savefig(fig, 'letter_bubble_map.png')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate thesis visualisation plots (band optimization pipeline)"
    )
    parser.add_argument("--subject",      type=int, default=None,
                        help="Run confusion matrix for one subject only")
    parser.add_argument("--skip_cm",      action="store_true",
                        help="Skip confusion matrix (slow)")
    parser.add_argument("--skip_letters", action="store_true",
                        help="Skip letter insight plots (also slow)")
    args = parser.parse_args()

    print("\n" + "═" * 55)
    print("  Generating thesis visualisations")
    print(f"  Pipeline: band optimization + LDA lsqr+auto")
    print(f"  Output:   {PLOTS_DIR.resolve()}")
    print("═" * 55)

    # ── Fast plots (no CV needed) ──────────────────────────────────────────
    plot_per_subject_accuracy()
    plot_band_wins()
    plot_accuracy_distribution()
    plot_class_distribution()

    # ── Confusion matrix ──────────────────────────────────────────────────
    if not args.skip_cm:
        if args.subject:
            plot_confusion_matrix([args.subject])
        else:
            plot_confusion_matrix([12, 2, 8, 14, 3])
    else:
        print("  [3] Confusion matrix skipped (--skip_cm)")

    # ── Letter insight plots (slow — run full CV per subject) ─────────────
    if not args.skip_letters:
        print("\n  Collecting per-letter accuracy (runs CV on all subjects)...")
        sample = sorted(RESULTS.keys())
        letter_accs, cm_norm = collect_letter_accuracies(sample)
        plot_letter_difficulty(letter_accs)
        plot_confused_pairs(cm_norm)
        plot_letter_bubble_map(letter_accs)
    else:
        print("  [6–8] Letter plots skipped (--skip_letters)")

    print("\n" + "═" * 55)
    print(f"  Done! Plots saved to: {PLOTS_DIR.resolve()}")
    print("═" * 55)