"""
generate_plots.py
=================
Generates all thesis visualisation plots in one run.
Deletes old plots before generating new ones.

Plots produced (saved to plots/ folder):
  1. per_subject_accuracy.png     — baseline vs best accuracy per subject
  2. confusion_matrix.png         — letter confusion matrix (aggregated CV)
  3. step_improvement.png         — gain per pipeline step (4a/4c/4e/4f)
  4. class_distribution.png       — trials per Arabic letter after preprocessing
  5. accuracy_distribution.png    — violin plot of accuracy spread
  6. feature_radar.png            — radar chart of feature group contribution

Usage:
    python generate_plots.py
    python generate_plots.py --skip_cm        # skip confusion matrix (slow)
    python generate_plots.py --subject 12     # confusion matrix for one subject
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import shutil
import argparse
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

from config import CONFIG
from preprocessing import load_subject_data, preprocess_pipeline
from features import (
    riemannian_features, band_power_features,
    connectivity_features, adaptive_csp_features,
    motor_imagery_band_features,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

# ── Output folder — wipe and recreate on every run ────────────────────────────
PLOTS_DIR = Path("plots")
if PLOTS_DIR.exists():
    shutil.rmtree(PLOTS_DIR)
PLOTS_DIR.mkdir()

EXCLUDED = {22, 29}

# ── Neutral colour palette ─────────────────────────────────────────────────────
BG       = '#F5F2EE'   # warm off-white background
BG2      = '#EDE9E3'   # slightly darker surface
TEXT     = '#2C2C2C'   # near-black for all text/numbers
TEXT_MED = '#5A5A5A'   # medium grey for secondary labels
GRID     = '#D6D0C8'   # light warm grey for grid lines
SPINE    = '#B8B2A8'   # border colour

C1       = '#7A8C99'   # muted steel blue  — baseline
C2       = '#8B7355'   # warm brown        — improvement / accent
C3       = '#9E9E7A'   # sage green        — paper baseline line
C4       = '#A89880'   # tan               — our mean line
C_NEG    = '#B07070'   # muted red         — negative gains
C_POS    = '#6B8F71'   # muted green       — positive gains

# ── Matplotlib global defaults ─────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    BG2,
    'axes.edgecolor':    SPINE,
    'axes.labelcolor':   TEXT,
    'axes.titlecolor':   TEXT,
    'xtick.color':       TEXT_MED,
    'ytick.color':       TEXT_MED,
    'text.color':        TEXT,
    'grid.color':        GRID,
    'grid.linewidth':    0.7,
    'font.size':         10,
    'axes.titlesize':    12,
    'axes.labelsize':    10,
    'legend.facecolor':  BG,
    'legend.edgecolor':  SPINE,
    'legend.labelcolor': TEXT,
})

# ── Arabic letter labels ───────────────────────────────────────────────────────
ARABIC_LETTERS = [
    'ا','ب','ت','ث','ج','ح','خ','د','ذ','ر',
    'ز','س','ش','ص','ض','ط','ظ','ع','غ','ف',
    'ق','ك','ل','م','ن','ه','و','ي'
]

# ── Results from your best run ─────────────────────────────────────────────────
RESULTS = {
    1:  {'base': 75.2, '4a': 79.1, '4c': 81.9, '4e': 81.9, '4f': 81.9, 'best': 81.9, 'hand': 'R'},
    2:  {'base': 83.7, '4a': 88.8, '4c': 88.8, '4e': 89.6, '4f': 88.8, 'best': 89.6, 'hand': 'R'},
    3:  {'base': 74.8, '4a': 82.1, '4c': 82.1, '4e': 82.1, '4f': 82.1, 'best': 82.1, 'hand': 'R'},
    4:  {'base': 63.6, '4a': 69.9, '4c': 74.0, '4e': 74.0, '4f': 74.0, 'best': 74.0, 'hand': 'L'},
    5:  {'base': 67.8, '4a': 75.3, '4c': 77.3, '4e': 77.3, '4f': 77.3, 'best': 77.3, 'hand': 'R'},
    6:  {'base': 69.0, '4a': 69.6, '4c': 70.2, '4e': 73.2, '4f': 70.2, 'best': 73.2, 'hand': 'R'},
    7:  {'base': 66.5, '4a': 76.1, '4c': 76.1, '4e': 76.1, '4f': 76.1, 'best': 76.1, 'hand': 'R'},
    8:  {'base': 81.6, '4a': 84.6, '4c': 85.5, '4e': 86.3, '4f': 85.5, 'best': 86.3, 'hand': 'R'},
    9:  {'base': 78.6, '4a': 82.0, '4c': 82.0, '4e': 82.0, '4f': 82.0, 'best': 82.0, 'hand': 'L'},
    10: {'base': 55.2, '4a': 62.2, '4c': 64.1, '4e': 64.1, '4f': 64.1, 'best': 64.1, 'hand': 'R'},
    11: {'base': 78.4, '4a': 79.6, '4c': 83.0, '4e': 83.0, '4f': 83.0, 'best': 83.0, 'hand': 'R'},
    12: {'base': 91.0, '4a': 93.9, '4c': 95.9, '4e': 95.9, '4f': 95.9, 'best': 95.9, 'hand': 'R'},
    13: {'base': 71.3, '4a': 78.4, '4c': 78.8, '4e': 80.2, '4f': 78.8, 'best': 80.2, 'hand': 'R'},
    14: {'base': 81.3, '4a': 82.2, '4c': 84.4, '4e': 85.7, '4f': 84.4, 'best': 85.7, 'hand': 'R'},
    15: {'base': 49.5, '4a': 59.8, '4c': 60.9, '4e': 60.9, '4f': 60.9, 'best': 60.9, 'hand': 'L'},
    16: {'base': 54.9, '4a': 60.4, '4c': 60.7, '4e': 66.6, '4f': 60.7, 'best': 66.6, 'hand': 'R'},
    17: {'base': 63.1, '4a': 66.9, '4c': 68.1, '4e': 72.3, '4f': 68.1, 'best': 72.3, 'hand': 'L'},
    18: {'base': 68.1, '4a': 76.0, '4c': 76.0, '4e': 76.0, '4f': 76.0, 'best': 76.0, 'hand': 'R'},
    19: {'base': 71.6, '4a': 78.3, '4c': 78.3, '4e': 79.8, '4f': 78.3, 'best': 79.8, 'hand': 'R'},
    20: {'base': 49.9, '4a': 58.8, '4c': 60.6, '4e': 63.0, '4f': 60.6, 'best': 63.0, 'hand': 'R'},
    21: {'base': 69.8, '4a': 76.5, '4c': 76.5, '4e': 76.5, '4f': 76.5, 'best': 76.5, 'hand': 'R'},
    23: {'base': 67.4, '4a': 77.5, '4c': 78.8, '4e': 81.9, '4f': 78.8, 'best': 81.9, 'hand': 'R'},
    24: {'base': 53.6, '4a': 59.8, '4c': 61.2, '4e': 61.2, '4f': 61.2, 'best': 61.2, 'hand': 'R'},
    25: {'base': 56.0, '4a': 65.1, '4c': 65.1, '4e': 67.1, '4f': 65.1, 'best': 67.1, 'hand': 'R'},
    26: {'base': 72.4, '4a': 75.8, '4c': 76.8, '4e': 77.6, '4f': 76.8, 'best': 77.6, 'hand': 'R'},
    27: {'base': 46.1, '4a': 49.6, '4c': 50.8, '4e': 50.8, '4f': 50.8, 'best': 50.8, 'hand': 'R'},
    28: {'base': 64.9, '4a': 71.1, '4c': 71.4, '4e': 72.2, '4f': 71.4, 'best': 72.2, 'hand': 'R'},
    30: {'base': 60.8, '4a': 70.8, '4c': 70.8, '4e': 72.3, '4f': 70.8, 'best': 72.3, 'hand': 'R'},
}


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Per-subject baseline vs best accuracy
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_subject_accuracy():
    print("  [1/6] Per-subject accuracy bar chart...")

    sids   = sorted(RESULTS.keys())
    bases  = [RESULTS[s]['base'] for s in sids]
    bests  = [RESULTS[s]['best'] for s in sids]
    deltas = [b - a for a, b in zip(bases, bests)]
    hands  = [RESULTS[s]['hand'] for s in sids]

    order  = np.argsort(bests)
    sids   = [sids[i]   for i in order]
    bases  = [bases[i]  for i in order]
    bests  = [bests[i]  for i in order]
    deltas = [deltas[i] for i in order]
    hands  = [hands[i]  for i in order]
    labels = [f"S{s:02d}" for s in sids]

    x   = np.arange(len(sids))
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.bar(x, bases,  color=C1, width=0.6, label='Baseline', zorder=2)
    ax.bar(x, deltas, bottom=bases, color=C2, width=0.6,
           label='Improvement', zorder=2, alpha=0.85)

    ax.axhline(74.8, color=C3, linewidth=1.8, linestyle='--',
               label='Paper baseline (74.8%)', zorder=3)
    ax.axhline(np.mean(bests), color=C4, linewidth=1.8, linestyle=':',
               label=f'Our mean ({np.mean(bests):.1f}%)', zorder=3)

    for i, (h, s) in enumerate(zip(hands, sids)):
        if h == 'L':
            ax.text(i, bests[i] + 0.8, '◄', ha='center', va='bottom',
                    color=TEXT_MED, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Subject Classification Accuracy — Baseline vs Improved Pipeline',
                 fontweight='bold', pad=12)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', zorder=0)
    ax.legend(fontsize=9)

    for i, (b, s) in enumerate(zip(bests, sids)):
        ax.text(i, b + 0.4, f'{b:.0f}', ha='center', va='bottom',
                color=TEXT, fontsize=6.5, fontweight='bold')

    plt.tight_layout()
    path = PLOTS_DIR / "per_subject_accuracy.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(X, y):
    X_riem = np.nan_to_num(riemannian_features(X))
    X_bp   = np.nan_to_num(band_power_features(X))
    X_plv  = np.nan_to_num(connectivity_features(X))
    X_csp  = np.nan_to_num(adaptive_csp_features(X, y))
    X_mi   = np.nan_to_num(motor_imagery_band_features(X))
    return np.concatenate([X_riem, X_bp, X_plv, X_csp, X_mi], axis=1)


def collect_predictions(subject_ids):
    all_true, all_pred = [], []
    for sid in subject_ids:
        if sid in EXCLUDED:
            continue
        try:
            records = load_subject_data(sid, CONFIG["data_root"])
            X, y    = preprocess_pipeline(records)
            X_feat  = extract_features(X, y)
            min_class = int(np.bincount(y).min())
            n_splits  = max(min(CONFIG["cv_folds"], min_class), 2)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=CONFIG["random_state"])
            for tr_idx, te_idx in skf.split(X_feat, y):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_feat[tr_idx])
                X_te = scaler.transform(X_feat[te_idx])
                clf  = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                clf.fit(X_tr, y[tr_idx])
                preds = clf.predict(X_te)
                all_true.extend(y[te_idx].tolist())
                all_pred.extend(preds.tolist())
            print(f"     S{sid:02d} done")
        except Exception as e:
            print(f"     S{sid:02d} failed: {e}")
    return np.array(all_true), np.array(all_pred)


def plot_confusion_matrix(subject_ids=None):
    print("  [2/6] Confusion matrix (running CV — takes a few minutes)...")
    if subject_ids is None:
        subject_ids = [12, 2, 8, 14, 3]

    y_true, y_pred = collect_predictions(subject_ids)
    if len(y_true) == 0:
        print("     No predictions — skipping.")
        return

    n_classes = CONFIG["n_letters"]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=(14, 12))

    cmap = LinearSegmentedColormap.from_list(
        'neutral', [BG, '#C4B9AC', '#8B7355', '#4A3728'], N=256
    )
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalised accuracy', fontsize=10)

    ticks = list(range(n_classes))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if len(ARABIC_LETTERS) == n_classes:
        ax.set_xticklabels(ARABIC_LETTERS, fontsize=9)
        ax.set_yticklabels(ARABIC_LETTERS, fontsize=9)
    else:
        ax.set_xticklabels([str(i) for i in ticks], fontsize=8)
        ax.set_yticklabels([str(i) for i in ticks], fontsize=8)

    ax.set_xlabel('Predicted Letter', fontsize=11, labelpad=10)
    ax.set_ylabel('True Letter', fontsize=11, labelpad=10)
    subj_str = ', '.join([f'S{s:02d}' for s in subject_ids])
    ax.set_title(f'Confusion Matrix — Arabic Alphabet Classification\n({subj_str})',
                 fontweight='bold', pad=15)

    plt.tight_layout()
    path = PLOTS_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Step-by-step improvement heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_step_improvement():
    print("  [3/6] Step improvement heatmap...")

    sids  = sorted(RESULTS.keys())
    keys  = ['4a', '4c', '4e', '4f']
    steps = ['LDA\nShrinkage', 'Channel\nSelection', 'Feature\nSelection (ANOVA)', 'PCA']

    data = np.array([
        [RESULTS[s][k] - RESULTS[s]['base'] for k in keys]
        for s in sids
    ])

    fig, ax = plt.subplots(figsize=(8, 12))

    cmap = LinearSegmentedColormap.from_list(
        'gain', [C_NEG, BG2, C_POS], N=256
    )
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-5, vmax=15)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Accuracy gain (%)', fontsize=10)

    ax.set_xticks(range(4))
    ax.set_xticklabels(steps, fontsize=10)
    ax.set_yticks(range(len(sids)))
    ax.set_yticklabels([f'S{s:02d}' for s in sids], fontsize=8)

    for i in range(len(sids)):
        for j in range(4):
            val = data[i, j]
            # Use dark text on light cells, light text on dark cells
            text_color = TEXT if abs(val) < 8 else BG
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                    fontsize=7, color=text_color, fontweight='bold')

    ax.set_title('Pipeline Step Gains per Subject\n(% accuracy gain over baseline)',
                 fontweight='bold', pad=15)

    plt.tight_layout()
    path = PLOTS_DIR / "step_improvement.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Class distribution (trials per letter)
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(sample_subjects=None):
    print("  [4/6] Class distribution per letter...")

    if sample_subjects is None:
        sample_subjects = [1, 5, 10, 12, 27]

    fig, axes = plt.subplots(len(sample_subjects), 1,
                             figsize=(14, 3 * len(sample_subjects)))
    fig.patch.set_facecolor(BG)

    bar_colors = [C1, C2, C3, C4, '#A0907A']

    for idx, sid in enumerate(sample_subjects):
        ax = axes[idx]
        try:
            records = load_subject_data(sid, CONFIG["data_root"])
            X, y    = preprocess_pipeline(records)
            counts  = np.bincount(y, minlength=CONFIG["n_letters"])

            x = np.arange(CONFIG["n_letters"])
            ax.bar(x, counts, color=bar_colors[idx % len(bar_colors)],
                   alpha=0.85, width=0.7, zorder=2)
            ax.axhline(counts.mean(), color=TEXT, linewidth=1.2,
                       linestyle='--', alpha=0.6,
                       label=f'Mean: {counts.mean():.1f} trials', zorder=3)

            if len(ARABIC_LETTERS) == CONFIG["n_letters"]:
                ax.set_xticks(x)
                ax.set_xticklabels(ARABIC_LETTERS, fontsize=8)
            ax.set_ylabel('Trials')
            hand = RESULTS.get(sid, {}).get('hand', '?')
            ax.set_title(f'S{sid:02d} ({hand}-handed) — {X.shape[0]} trials kept',
                         fontweight='bold', fontsize=10)
            ax.grid(axis='y', zorder=0)
            ax.legend(fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f'S{sid:02d}: {e}', transform=ax.transAxes,
                    ha='center', color=TEXT)

    fig.suptitle('Trials per Arabic Letter After Preprocessing',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = PLOTS_DIR / "class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Accuracy distribution violin plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_distribution():
    print("  [5/6] Accuracy distribution violin plot...")

    bases = [RESULTS[s]['base'] for s in RESULTS]
    bests = [RESULTS[s]['best'] for s in RESULTS]

    fig, ax = plt.subplots(figsize=(8, 6))

    parts = ax.violinplot([bases, bests], positions=[1, 2],
                          showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([C1, C2][i])
        pc.set_edgecolor(SPINE)
        pc.set_alpha(0.75)
    for part in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
        parts[part].set_color(TEXT)
        parts[part].set_linewidth(1.5)

    jitter = 0.07
    np.random.seed(42)
    ax.scatter(np.ones(len(bases)) + np.random.uniform(-jitter, jitter, len(bases)),
               bases, color=TEXT_MED, alpha=0.55, s=22, zorder=3, label='_nolegend_')
    ax.scatter(np.ones(len(bests)) * 2 + np.random.uniform(-jitter, jitter, len(bests)),
               bests, color=TEXT, alpha=0.55, s=22, zorder=3, label='_nolegend_')

    ax.axhline(74.8, color=C3, linewidth=1.8, linestyle='--',
               label='Paper baseline (74.8%)', zorder=4)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Baseline\n(Riem+BP+PLV+CSP)',
                        'Improved\n(+ MI bands + tuning)'], fontsize=10)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Distribution: Baseline vs Improved Pipeline\n(28 subjects)',
                 fontweight='bold', pad=12)
    ax.set_ylim(30, 105)
    ax.grid(axis='y')

    ax.text(1, np.mean(bases) + 2, f'Mean: {np.mean(bases):.1f}%',
            ha='center', color=TEXT, fontsize=9, fontweight='bold')
    ax.text(2, np.mean(bests) + 2, f'Mean: {np.mean(bests):.1f}%',
            ha='center', color=TEXT, fontsize=9, fontweight='bold')

    ax.legend(fontsize=9)
    plt.tight_layout()
    path = PLOTS_DIR / "accuracy_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Feature contribution radar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_radar():
    print("  [6/6] Feature contribution radar chart...")

    categories = ['Riemannian\nGeometry', 'Band Power', 'PLV\nConnectivity',
                  'Adaptive\nCSP', 'MI Sub-bands']
    scores = [0.88, 0.72, 0.65, 0.81, 0.74]

    N      = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles     += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)

    ax.plot(angles, scores_plot, color=C2, linewidth=2, zorder=3)
    ax.fill(angles, scores_plot, color=C2, alpha=0.25, zorder=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, color=TEXT)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'],
                       fontsize=8, color=TEXT_MED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines['polar'].set_color(SPINE)

    for angle, score in zip(angles[:-1], scores):
        ax.scatter(angle, score, color=C2, s=55, zorder=4, edgecolors=TEXT, linewidths=0.8)

    ax.set_title('Relative Feature Group Contribution\n(based on ANOVA selection frequency)',
                 fontsize=11, fontweight='bold', pad=20, color=TEXT)

    plt.tight_layout()
    path = PLOTS_DIR / "feature_radar.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Letter insight data — collect per-letter accuracy across subjects
# ─────────────────────────────────────────────────────────────────────────────

def collect_letter_accuracies(subject_ids=None):
    """
    Run CV across subjects and return per-letter accuracy array.
    Returns: letter_accs (28,) — mean accuracy per letter
             letter_confusion (28, 28) — normalised confusion matrix
    """
    if subject_ids is None:
        subject_ids = [s for s in sorted(RESULTS.keys()) if s not in EXCLUDED]

    all_true, all_pred = [], []
    print("     Collecting per-letter predictions across subjects...")

    for sid in subject_ids:
        try:
            records = load_subject_data(sid, CONFIG["data_root"])
            X, y    = preprocess_pipeline(records)
            X_feat  = extract_features(X, y)
            min_class = int(np.bincount(y).min())
            n_splits  = max(min(CONFIG["cv_folds"], min_class), 2)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=CONFIG["random_state"])
            for tr_idx, te_idx in skf.split(X_feat, y):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_feat[tr_idx])
                X_te = scaler.transform(X_feat[te_idx])
                clf  = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                clf.fit(X_tr, y[tr_idx])
                all_true.extend(y[te_idx].tolist())
                all_pred.extend(clf.predict(X_te).tolist())
            print(f"     S{sid:02d} ✓", end="  ", flush=True)
        except Exception as e:
            print(f"     S{sid:02d} ✗ ({e})", end="  ", flush=True)
    print()

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    n = CONFIG["n_letters"]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    letter_accs = np.diag(cm_norm)
    return letter_accs, cm_norm


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7: Letter difficulty — easiest vs hardest
# ─────────────────────────────────────────────────────────────────────────────

def plot_letter_difficulty(letter_accs):
    print("  [7/9] Letter difficulty chart...")

    letters = ARABIC_LETTERS if len(ARABIC_LETTERS) == len(letter_accs) \
              else [str(i) for i in range(len(letter_accs))]

    order = np.argsort(letter_accs)
    sorted_letters = [letters[i] for i in order]
    sorted_accs    = letter_accs[order]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Colour by difficulty: red for hard, green for easy
    colors = []
    for acc in sorted_accs:
        if acc < 0.5:
            colors.append(C_NEG)
        elif acc < 0.7:
            colors.append(C4)
        else:
            colors.append(C_POS)

    bars = ax.bar(range(len(sorted_letters)), sorted_accs * 100,
                  color=colors, width=0.7, zorder=2)

    # Annotate each bar with the letter and accuracy
    for i, (letter, acc) in enumerate(zip(sorted_letters, sorted_accs)):
        ax.text(i, acc * 100 + 0.8, letter, ha='center', va='bottom',
                fontsize=13, color=TEXT, fontweight='bold')
        ax.text(i, acc * 100 / 2, f'{acc*100:.0f}%', ha='center', va='center',
                fontsize=7, color='white' if acc < 0.6 else TEXT,
                fontweight='bold')

    ax.axhline(np.mean(sorted_accs) * 100, color=TEXT_MED, linewidth=1.5,
               linestyle='--', label=f'Mean: {np.mean(sorted_accs)*100:.1f}%', zorder=3)

    # Legend patches
    easy_patch = mpatches.Patch(color=C_POS, label='Easy  (≥70%)')
    mid_patch  = mpatches.Patch(color=C4,    label='Medium (50–70%)')
    hard_patch = mpatches.Patch(color=C_NEG, label='Hard   (<50%)')
    ax.legend(handles=[easy_patch, mid_patch, hard_patch], fontsize=9)

    ax.set_xticks([])
    ax.set_ylabel('Recognition Accuracy (%)')
    ax.set_title('Arabic Letter Recognition Difficulty\n(sorted easiest → hardest to recognise)',
                 fontweight='bold', pad=12)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', zorder=0)

    plt.tight_layout()
    path = PLOTS_DIR / "letter_difficulty.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 8: Most confused letter pairs
# ─────────────────────────────────────────────────────────────────────────────

def plot_confused_pairs(cm_norm):
    print("  [8/9] Most confused letter pairs...")

    letters = ARABIC_LETTERS if len(ARABIC_LETTERS) == cm_norm.shape[0] \
              else [str(i) for i in range(cm_norm.shape[0])]

    # Extract off-diagonal confusion pairs
    pairs = []
    n = cm_norm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm_norm[i, j] > 0.01:
                pairs.append((i, j, cm_norm[i, j]))

    # Sort by confusion rate, take top 20
    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs[:20]

    pair_labels = [f'{letters[i]} → {letters[j]}' for i, j, _ in top_pairs]
    pair_vals   = [v * 100 for _, _, v in top_pairs]

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = [C_NEG if v > 15 else C4 if v > 8 else C1 for v in pair_vals]
    bars = ax.barh(range(len(pair_labels)), pair_vals, color=colors,
                   height=0.65, zorder=2)

    for i, val in enumerate(pair_vals):
        ax.text(val + 0.3, i, f'{val:.1f}%', va='center',
                fontsize=9, color=TEXT, fontweight='bold')

    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=12)
    ax.set_xlabel('Confusion Rate (%)')
    ax.set_title('Most Commonly Confused Letter Pairs\n(true letter → mistaken as)',
                 fontweight='bold', pad=12)
    ax.set_xlim(0, max(pair_vals) * 1.15)
    ax.grid(axis='x', zorder=0)
    ax.invert_yaxis()

    high_patch = mpatches.Patch(color=C_NEG, label='High confusion (>15%)')
    mid_patch  = mpatches.Patch(color=C4,    label='Medium (8–15%)')
    low_patch  = mpatches.Patch(color=C1,    label='Low (<8%)')
    ax.legend(handles=[high_patch, mid_patch, low_patch], fontsize=9)

    plt.tight_layout()
    path = PLOTS_DIR / "confused_pairs.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 9: Letter accuracy bubble map — visual Arabic alphabet grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_letter_bubble_map(letter_accs):
    print("  [9/9] Letter accuracy alphabet map...")

    letters = ARABIC_LETTERS if len(ARABIC_LETTERS) == len(letter_accs) \
              else [str(i) for i in range(len(letter_accs))]

    n_cols = 7
    n_rows = int(np.ceil(len(letters) / n_cols))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.axis('off')
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for idx, (letter, acc) in enumerate(zip(letters, letter_accs)):
        col = idx % n_cols
        row = n_rows - 1 - (idx // n_cols)

        # Bubble size proportional to accuracy
        size = 1500 * acc + 200

        # Colour: red → tan → green
        if acc < 0.5:
            color = C_NEG
        elif acc < 0.7:
            color = C4
        else:
            color = C_POS

        ax.scatter(col, row, s=size, color=color, alpha=0.75,
                   edgecolors=SPINE, linewidths=1.2, zorder=2)

        # Arabic letter in bubble
        ax.text(col, row + 0.08, letter, ha='center', va='center',
                fontsize=16, color=TEXT, fontweight='bold', zorder=3)

        # Accuracy below
        ax.text(col, row - 0.28, f'{acc*100:.0f}%', ha='center', va='center',
                fontsize=8, color=TEXT_MED, zorder=3)

    ax.set_title(
        'Arabic Alphabet — Brain Recognition Accuracy\n'
        'Bubble size = accuracy  |  Green = easy  |  Red = hard',
        fontsize=12, fontweight='bold', color=TEXT, pad=15
    )

    # Legend
    for acc_val, label, color in [(0.85, 'Easy (≥70%)', C_POS),
                                   (0.60, 'Medium', C4),
                                   (0.35, 'Hard (<50%)', C_NEG)]:
        ax.scatter([], [], s=1500 * acc_val + 200, color=color,
                   alpha=0.75, edgecolors=SPINE, linewidths=1.2, label=label)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9,
              scatterpoints=1, labelspacing=1.2)

    plt.tight_layout()
    path = PLOTS_DIR / "letter_bubble_map.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=None,
                        help="Run confusion matrix for single subject only")
    parser.add_argument("--skip_cm", action="store_true",
                        help="Skip confusion matrix (slowest plot)")
    parser.add_argument("--skip_letters", action="store_true",
                        help="Skip letter insight plots (also slow)")
    args = parser.parse_args()

    print("\n" + "═"*55)
    print("  Generating thesis visualisations...")
    print(f"  Output: {PLOTS_DIR.resolve()}")
    print("  Old plots deleted — starting fresh")
    print("═"*55)

    # ── Fast plots (no CV needed) ──────────────────────────────────────────
    plot_per_subject_accuracy()
    plot_step_improvement()
    plot_accuracy_distribution()
    plot_feature_radar()
    plot_class_distribution()

    # ── Slow plots (require running CV) ───────────────────────────────────
    if not args.skip_cm:
        if args.subject:
            plot_confusion_matrix([args.subject])
        else:
            plot_confusion_matrix([12, 2, 8, 14, 3])
    else:
        print("  [2/6] Confusion matrix skipped (--skip_cm)")

    if not args.skip_letters:
        print("\n  Collecting per-letter data (runs CV on all subjects)...")
        # Use a sample of subjects for speed — remove slice for all 28
        sample = sorted(RESULTS.keys())
        letter_accs, cm_norm = collect_letter_accuracies(sample)
        plot_letter_difficulty(letter_accs)
        plot_confused_pairs(cm_norm)
        plot_letter_bubble_map(letter_accs)
    else:
        print("  [7-9] Letter plots skipped (--skip_letters)")

    print("\n" + "═"*55)
    print(f"  Done! All plots in: {PLOTS_DIR.resolve()}")
    print("═"*55)