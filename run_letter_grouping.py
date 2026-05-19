"""
run_letter_grouping_standalone.py
==================================
Self-contained version — does NOT import config.py.

DATA PATH OPTIONS (choose one, set below in the USER CONFIGURATION block):
  Option 1 — Local path (your own machine)
      DATA_ROOT = Path.home() / "Downloads" / "data"
      Works for: anyone who has the dataset folder in their Downloads.

  Option 2 — Environment variable (team-friendly, no hardcoded paths)
      Set once in your shell:  export EEG_DATA_ROOT="/path/to/data"
      Everyone on the team sets their own path — nothing gets committed.

  Option 3 — .env file (cleanest for teams)
      Create a file called  .env  in the project root (add .env to .gitignore!)
      Contents:  EEG_DATA_ROOT=/path/to/data
      The script reads it automatically.

  Option 4 — Google Drive / OneDrive (shared dataset without GitHub)
      Mount the shared drive and point DATA_ROOT at the mount path.
      Google Drive on Mac:  /Volumes/GoogleDrive/Shared drives/<name>/data
      OneDrive on Mac:      Path.home() / "OneDrive" / "data"

The script auto-detects Options 2 & 3 and falls back to Option 1.
If nothing is found it prints a clear error telling you exactly what to set.

Usage:
    python run_letter_grouping_standalone.py --subject 1
    python run_letter_grouping_standalone.py --subject 1 --strategy visual
    python run_letter_grouping_standalone.py --subject 1 --strategy confusion
    python run_letter_grouping_standalone.py --subject 1 --strategy both --verbose
    python run_letter_grouping_standalone.py --all
"""

# ══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION  ←  edit this block only
# ══════════════════════════════════════════════════════════════════════════════

from pathlib import Path
import os

def _resolve_data_root() -> Path:
    """
    Auto-detect the dataset root in this priority order:
      1. EEG_DATA_ROOT environment variable
      2. .env file in the same directory as this script
      3. ~/Downloads/data   (default local fallback)

    To use a different local path, just change the fallback at the bottom.
    """
    # Priority 1: environment variable
    env_val = os.environ.get("EEG_DATA_ROOT")
    if env_val:
        return Path(env_val)

    # Priority 2: .env file next to this script
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("EEG_DATA_ROOT"):
                val = line.split("=", 1)[-1].strip().strip('"').strip("'")
                if val:
                    return Path(val)

    # Priority 3: default local fallback — change this to your path
    return Path.home() / "Downloads" / "data"


DATA_ROOT = _resolve_data_root()

# ─── Dataset constants (same as config.py) ────────────────────────────────────
N_SUBJECTS   = 30
N_LETTERS    = 28
N_TRIALS     = 10
N_CHANNELS   = 14
SFREQ        = 256
CHANNEL_NAMES = [
    "AF3", "AF4", "F7", "F8", "F3", "F4",
    "FC5", "FC6", "T7", "T8", "P7", "P8", "O1", "O2",
]

T_RELAX   = 5.0
T_OBSERVE = 5.0
T_IMAGINE = 8.0

BANDPASS       = (0.5, 40.0)
NOTCH_FREQS    = [50.0]
EPOCH_TMIN     = 0.0
EPOCH_TMAX     = 6.0

FREQ_BANDS = {
    "delta": (0.5,  4),
    "theta": (4,    8),
    "alpha": (8,   13),
    "beta":  (13,  30),
    "gamma": (30,  40),
}

CV_FOLDS     = 5      # 5 folds (faster than 10 for grouping experiments)
RANDOM_STATE = 42

CSP_COMPONENTS_MAP = [(200, 2), (250, 4), (999, 6)]

# ══════════════════════════════════════════════════════════════════════════════
#  END OF USER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

import sys
import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")

import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch, welch, hilbert
from scipy.stats import skew, kurtosis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
)

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Path validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_data_root():
    """Crash early with a helpful message if the data folder isn't found."""
    if not DATA_ROOT.exists():
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  DATA FOLDER NOT FOUND                                       ║
╠══════════════════════════════════════════════════════════════╣
║  Expected: {str(DATA_ROOT):<50} ║
╠══════════════════════════════════════════════════════════════╣
║  Fix ONE of the following:                                   ║
║                                                              ║
║  A) Set an environment variable (recommended for teams):     ║
║     Mac/Linux:  export EEG_DATA_ROOT="/path/to/your/data"   ║
║     Windows:    set EEG_DATA_ROOT=C:\\path\\to\\your\\data       ║
║     Add it to ~/.zshrc or ~/.bashrc to make it permanent.    ║
║                                                              ║
║  B) Create a .env file next to this script:                  ║
║     EEG_DATA_ROOT=/path/to/your/data                        ║
║     (add .env to your .gitignore — never commit paths!)      ║
║                                                              ║
║  C) Edit this file directly — change the fallback path in   ║
║     _resolve_data_root() near the top of the script.        ║
╚══════════════════════════════════════════════════════════════╝
""")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Arabic letter metadata
# ─────────────────────────────────────────────────────────────────────────────

ARABIC_LETTERS = [
    "ا", "ب", "ت", "ث", "ج", "ح", "خ",
    "د", "ذ", "ر", "ز", "س", "ش", "ص",
    "ض", "ط", "ظ", "ع", "غ", "ف", "ق",
    "ك", "ل", "م", "ن", "ه", "و", "ي",
]

# Strategy A: Letters grouped by shared visual base shape
VISUAL_GROUPS = {
    "alef_family":  [0, 7, 8, 9, 10, 27],  # ا د ذ ر ز و ي
    "ba_family":    [1, 2, 3, 24],          # ب ت ث ن
    "jeem_family":  [4, 5, 6],              # ج ح خ
    "seen_family":  [11, 12],               # س ش
    "sad_family":   [13, 14],               # ص ض
    "ta_family":    [15, 16],               # ط ظ
    "ayn_family":   [17, 18],               # ع غ
    "fa_qaf":       [19, 20],               # ف ق
    "kaf_lam":      [21, 22],               # ك ل
    "meem_ha":      [23, 25],               # م ه
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (self-contained, no config.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_subject_data(subject_id: int):
    records = []
    s_folder = DATA_ROOT / f"S{subject_id:02d}"
    if not s_folder.exists():
        raise FileNotFoundError(f"Subject folder not found: {s_folder}")

    for letter_id in range(1, N_LETTERS + 1):
        l_folder = s_folder / f"L{letter_id:02d}"
        for trial_id in range(1, N_TRIALS + 1):
            mat_path = l_folder / f"S{subject_id:02d}_L{letter_id:02d}_T{trial_id}.mat"
            if not mat_path.exists():
                continue
            mat = sio.loadmat(str(mat_path), simplify_cells=True)
            eeg_struct = mat.get("EEG", {})
            if isinstance(eeg_struct, np.ndarray):
                eeg_struct = {k: eeg_struct[k].item()
                              for k in eeg_struct.dtype.names}
            if isinstance(eeg_struct, dict) and "Data" in eeg_struct:
                eeg_data = np.array(eeg_struct["Data"], dtype=np.float64)
            elif isinstance(eeg_struct, dict) and "data" in eeg_struct:
                eeg_data = np.array(eeg_struct["data"], dtype=np.float64)
            else:
                best, best_size = None, 0
                for v in mat.values():
                    if isinstance(v, np.ndarray) and v.ndim == 2 and v.size > best_size:
                        best, best_size = v, v.size
                if best is None:
                    continue
                eeg_data = best.astype(np.float64)
            records.append({
                "eeg": eeg_data, "label": letter_id - 1,
                "trial": trial_id, "letter": letter_id,
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing (self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def _bandpass(data, lo, hi, fs):
    nyq = fs / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, data, axis=1)

def _notch(data, f0, fs, Q=30.0):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, data, axis=1)

def _car(data):
    return data - data.mean(axis=0, keepdims=True)

def _baseline(data, fs, sec=0.2):
    n = int(sec * fs)
    return data - data[:, :n].mean(axis=1, keepdims=True)

def _reject(data, thr=100.0):
    if np.median(np.abs(data)) > 50.0:
        return False
    return bool(np.abs(data).max() > thr)

def _zscore(X):
    mu  = X.mean(axis=(0, 2), keepdims=True)
    sig = X.std( axis=(0, 2), keepdims=True) + 1e-8
    return (X - mu) / sig

def preprocess(records):
    epochs, labels = [], []
    sfreq   = SFREQ
    t_start = T_RELAX + T_OBSERVE
    onset   = int(t_start * sfreq)
    offset  = onset + int(EPOCH_TMAX * sfreq)

    for rec in records:
        raw = rec["eeg"].copy()
        if raw.shape[1] < offset:
            continue
        ep = raw[:, onset:offset]
        ep = _bandpass(ep, *BANDPASS, sfreq)
        for f0 in NOTCH_FREQS:
            ep = _notch(ep, f0, sfreq)
        ep = _car(ep)
        if _reject(ep):
            continue
        ep = _baseline(ep, sfreq)
        epochs.append(ep)
        labels.append(rec["label"])

    X = np.array(epochs, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    return _zscore(X), y


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_n_csp(n_trials):
    for thr, n in CSP_COMPONENTS_MAP:
        if n_trials < thr:
            return n
    return CSP_COMPONENTS_MAP[-1][1]

def _riemannian(X):
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        covs = Covariances("oas").fit_transform(X)
        return TangentSpace("riemann").fit_transform(covs).astype(np.float32)
    except ImportError:
        n_ch = X.shape[1]
        feats = [np.cov(t)[np.triu_indices(n_ch)] for t in X]
        return np.array(feats, dtype=np.float32)

def _band_power(X):
    feats = []
    for trial in X:
        row = []
        for ch in trial:
            freqs, psd = welch(ch, fs=SFREQ,
                               nperseg=min(256, ch.shape[0]))
            total = psd.sum() + 1e-10
            for lo, hi in FREQ_BANDS.values():
                mask = (freqs >= lo) & (freqs < hi)
                abs_p = float(psd[mask].sum())
                row  += [abs_p, float(abs_p / total)]
        feats.append(row)
    return np.array(feats, dtype=np.float32)

def _plv(X):
    n_ch = X.shape[1]
    bands = {"alpha": (8, 13), "beta": (13, 30)}
    feats = []
    def _bp(sig, lo, hi):
        nyq = SFREQ / 2
        b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
        return filtfilt(b, a, sig)
    for trial in X:
        row = []
        for lo, hi in bands.values():
            filt  = np.array([_bp(ch, lo, hi) for ch in trial])
            phase = np.angle(hilbert(filt, axis=1))
            for i in range(n_ch):
                for j in range(i + 1, n_ch):
                    row.append(float(np.abs(np.mean(
                        np.exp(1j * (phase[i] - phase[j]))
                    ))))
        feats.append(row)
    return np.array(feats, dtype=np.float32)

def _cov_mean(X):
    return np.array([np.cov(t) for t in X]).mean(axis=0)

def _numpy_csp(X, y_binary, n_comp):
    pos = X[y_binary == 1]
    neg = X[y_binary == 0]
    C1 = _cov_mean(pos); C2 = _cov_mean(neg); Cc = C1 + C2
    vals, vecs = np.linalg.eigh(Cc)
    W  = vecs @ np.diag(1.0 / np.sqrt(vals + 1e-10)) @ vecs.T
    S1 = W @ C1 @ W.T
    evals, evecs = np.linalg.eigh(S1)
    order   = np.argsort(evals)[::-1]
    filters = evecs[:, order[:n_comp]].T @ W
    proj    = np.einsum("cd,tds->tcs", filters, X)
    return np.log(proj.var(axis=2) + 1e-10)

def _adaptive_csp(X, y):
    n_trials     = X.shape[0]
    n_components = _adaptive_n_csp(n_trials)
    classes      = np.unique(y)
    all_feats    = []
    try:
        from mne.decoding import CSP
        _mne = True
    except ImportError:
        _mne = False

    for cls in classes:
        y_bin = (y == cls).astype(int)
        if y_bin.sum() < 5:
            continue
        if _mne:
            try:
                csp = CSP(n_components=n_components, log=True, norm_trace=False)
                all_feats.append(csp.fit_transform(X, y_bin))
                continue
            except Exception:
                pass
        all_feats.append(_numpy_csp(X, y_bin, n_components))

    if not all_feats:
        return np.zeros((X.shape[0], 1), dtype=np.float32)
    return np.concatenate(all_feats, axis=1).astype(np.float32)

def extract_features(X, y):
    parts = [
        np.nan_to_num(_riemannian(X)),
        np.nan_to_num(_band_power(X)),
        np.nan_to_num(_plv(X)),
        np.nan_to_num(_adaptive_csp(X, y)),
    ]
    return np.concatenate(parts, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# CV helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_clf():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

def cv_eval(X, y, n_splits=5, k_feat=150):
    if len(np.unique(y)) < 2 or len(y) < n_splits * 2:
        return {"acc_mean": 0.0, "acc_std": 0.0, "f1_macro": 0.0, "cm": None}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)
    accs, f1s, cms = [], [], []
    n_cls = len(np.unique(y))

    for tr, te in skf.split(X, y):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        k = min(k_feat, X_tr.shape[1], X_tr.shape[0] - 1)
        if k > 1:
            sel = SelectKBest(f_classif, k=k)
            X_tr = sel.fit_transform(X_tr, y_tr)
            X_te = sel.transform(X_te)
        clf = _make_clf()
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
        cms.append(confusion_matrix(y_te, y_pred, labels=np.arange(n_cls)))

    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs)),
        "f1_macro": float(np.mean(f1s)),
        "cm":       np.mean(cms, axis=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Flat baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_flat_baseline(X, y):
    print("\n  ── Flat baseline (28-class LDA, no grouping) ──")
    try:
        X_feat = extract_features(X, y)
        r = cv_eval(X_feat, y, n_splits=CV_FOLDS, k_feat=200)
        print(f"  Flat 28-class : {r['acc_mean']*100:.2f}% "
              f"± {r['acc_std']*100:.2f}%  (chance={100/28:.1f}%)")
        return r
    except Exception as e:
        print(f"  [!] Baseline failed: {e}")
        return {"acc_mean": 0.0, "acc_std": 0.0, "f1_macro": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Strategy A — Visual similarity groups
# ─────────────────────────────────────────────────────────────────────────────

def run_visual_groups(X, y, verbose=False):
    print("\n  ══ STRATEGY A: Visual Similarity Groups ══")
    print(f"\n  {'Group':<20} {'Letters':<32} {'N':>5}  {'Acc':>7}  "
          f"{'F1':>7}  Chance")
    print("  " + "─" * 76)

    group_results = {}
    for grp_name, letter_ids in VISUAL_GROUPS.items():
        mask = np.isin(y, letter_ids)
        X_g, y_g = X[mask], y[mask]
        n_trials = len(y_g)
        n_cls    = len(np.unique(y_g))

        if n_trials < 20 or n_cls < 2:
            print(f"  {grp_name:<20}  (skip: {n_trials} trials)")
            group_results[grp_name] = {"acc_mean": 0.0, "n_trials": n_trials,
                                       "letter_ids": letter_ids}
            continue

        uniq = sorted(np.unique(y_g))
        lmap = {old: new for new, old in enumerate(uniq)}
        y_mapped = np.array([lmap[l] for l in y_g])

        try:
            X_feat = extract_features(X_g, y_mapped)
            r = cv_eval(X_feat, y_mapped, n_splits=CV_FOLDS)
        except Exception as e:
            print(f"  {grp_name:<20}  FAILED: {e}")
            group_results[grp_name] = {"acc_mean": 0.0, "n_trials": n_trials,
                                       "letter_ids": letter_ids}
            continue

        chance = 100.0 / n_cls
        letter_str = " ".join(ARABIC_LETTERS[i] for i in uniq)
        marker = "✅" if r["acc_mean"]*100 > chance+10 else \
                 "⚠️ " if r["acc_mean"]*100 > chance+2  else "❌"

        print(f"  {grp_name:<20} {letter_str:<32} {n_trials:>5}  "
              f"{r['acc_mean']*100:>6.1f}%  {r['f1_macro']*100:>6.1f}%  "
              f"{chance:.0f}%  {marker}")

        group_results[grp_name] = {
            "acc_mean": r["acc_mean"], "acc_std": r["acc_std"],
            "f1_macro": r["f1_macro"], "n_trials": n_trials,
            "n_classes": n_cls, "chance": chance/100,
            "cm": r["cm"], "letter_ids": uniq,
        }

    print(f"\n  ── Hierarchical pipeline ──")
    hier = _hierarchical_pipeline(X, y, VISUAL_GROUPS, group_results,
                                   "visual", verbose)

    valid = [r for r in group_results.values() if r["acc_mean"] > 0]
    mean_w = np.mean([r["acc_mean"] for r in valid]) if valid else 0.0
    print(f"  Mean within-group acc : {mean_w*100:.2f}%")
    print(f"  Hierarchical pipeline : {hier['acc']*100:.2f}%  "
          f"F1={hier['f1']*100:.2f}%")

    return {"strategy": "visual", "group_results": group_results,
            "hierarchical": hier, "mean_within": mean_w}


# ─────────────────────────────────────────────────────────────────────────────
# Strategy B — Confusion-driven groups
# ─────────────────────────────────────────────────────────────────────────────

def build_confusion_groups(X, y, n_groups=7):
    print("\n  Building confusion matrix (5-fold CV, all 28 classes)...")
    n_cls = N_LETTERS
    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cm_accum = np.zeros((n_cls, n_cls))

    try:
        X_feat = extract_features(X, y)
    except Exception as e:
        print(f"  [!] Feature extraction failed: {e}")
        return None, None

    for tr, te in skf.split(X_feat, y):
        X_tr, X_te = X_feat[tr], X_feat[te]
        y_tr, y_te = y[tr], y[te]
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
        k = min(200, X_tr.shape[1])
        sel = SelectKBest(f_classif, k=k)
        X_tr = sel.fit_transform(X_tr, y_tr); X_te = sel.transform(X_te)
        clf = _make_clf()
        clf.fit(X_tr, y_tr)
        cm_accum += confusion_matrix(clf.predict(X_te), y_te,
                                     labels=np.arange(n_cls))

    row_sums = cm_accum.sum(axis=1, keepdims=True)
    cm_norm  = cm_accum / (row_sums + 1e-10)

    # Symmetric confusion distance
    conf_dist = np.zeros((n_cls, n_cls))
    for i in range(n_cls):
        for j in range(n_cls):
            if i != j:
                conf_dist[i, j] = 1.0 - (cm_norm[i,j] + cm_norm[j,i]) / 2.0

    from scipy.cluster.hierarchy import linkage, fcluster
    Z      = linkage(conf_dist, method="complete")
    labels = fcluster(Z, n_groups, criterion="maxclust")

    groups = {}
    for cid in range(1, n_groups + 1):
        members = [i for i, l in enumerate(labels) if l == cid]
        if members:
            groups[f"conf_grp_{cid}"] = members

    # Top confused pairs
    pairs = []
    for i in range(n_cls):
        for j in range(i+1, n_cls):
            pairs.append(((cm_norm[i,j]+cm_norm[j,i])/2, i, j))
    pairs.sort(reverse=True)

    return groups, cm_norm, pairs


def run_confusion_groups(X, y, n_groups=7, verbose=False):
    print("\n  ══ STRATEGY B: Confusion-Driven Groups ══")

    result = build_confusion_groups(X, y, n_groups=n_groups)
    if result[0] is None:
        return {}
    confusion_groups, cm_norm, pairs = result

    print(f"\n  Top-15 most confused letter pairs:")
    print(f"  {'Pair':<20}  {'Confusion rate':>14}")
    print("  " + "─"*36)
    for rate, i, j in pairs[:15]:
        print(f"  {ARABIC_LETTERS[i]}(L{i+1:02d}) ↔ "
              f"{ARABIC_LETTERS[j]}(L{j+1:02d})    {rate*100:>10.1f}%")

    print(f"\n  Discovered groups ({n_groups} clusters):")
    for gname, members in confusion_groups.items():
        letter_str = " ".join(ARABIC_LETTERS[i] for i in members)
        print(f"    {gname}: {letter_str}")

    print(f"\n  {'Group':<20} {'Letters':<35} {'N':>5}  {'Acc':>7}  "
          f"{'F1':>7}  Chance")
    print("  " + "─"*80)

    group_results = {}
    for grp_name, letter_ids in confusion_groups.items():
        mask = np.isin(y, letter_ids)
        X_g, y_g = X[mask], y[mask]
        n_trials = len(y_g)
        n_cls_g  = len(np.unique(y_g))

        if n_trials < 20 or n_cls_g < 2:
            group_results[grp_name] = {"acc_mean": 0.0, "n_trials": n_trials,
                                       "letter_ids": letter_ids}
            continue

        uniq = sorted(np.unique(y_g))
        lmap = {old: new for new, old in enumerate(uniq)}
        y_mapped = np.array([lmap[l] for l in y_g])

        try:
            X_feat = extract_features(X_g, y_mapped)
            r = cv_eval(X_feat, y_mapped, n_splits=CV_FOLDS)
        except Exception as e:
            group_results[grp_name] = {"acc_mean": 0.0, "n_trials": n_trials,
                                       "letter_ids": letter_ids}
            continue

        chance = 100.0 / n_cls_g
        letter_str = " ".join(ARABIC_LETTERS[i] for i in uniq)
        marker = "✅" if r["acc_mean"]*100 > chance+10 else \
                 "⚠️ " if r["acc_mean"]*100 > chance+2  else "❌"

        print(f"  {grp_name:<20} {letter_str:<35} {n_trials:>5}  "
              f"{r['acc_mean']*100:>6.1f}%  {r['f1_macro']*100:>6.1f}%  "
              f"{chance:.0f}%  {marker}")

        group_results[grp_name] = {
            "acc_mean": r["acc_mean"], "acc_std": r["acc_std"],
            "f1_macro": r["f1_macro"], "n_trials": n_trials,
            "n_classes": n_cls_g, "chance": chance/100,
            "cm": r["cm"], "letter_ids": uniq,
        }

    print(f"\n  ── Hierarchical pipeline ──")
    hier = _hierarchical_pipeline(X, y, confusion_groups, group_results,
                                   "confusion", verbose)

    valid = [r for r in group_results.values() if r["acc_mean"] > 0]
    mean_w = np.mean([r["acc_mean"] for r in valid]) if valid else 0.0
    print(f"  Mean within-group acc : {mean_w*100:.2f}%")
    print(f"  Hierarchical pipeline : {hier['acc']*100:.2f}%  "
          f"F1={hier['f1']*100:.2f}%")

    return {"strategy": "confusion", "group_results": group_results,
            "hierarchical": hier, "mean_within": mean_w,
            "top_confusions": pairs[:15]}


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical pipeline (shared)
# ─────────────────────────────────────────────────────────────────────────────

def _hierarchical_pipeline(X, y, groups, group_results,
                            strategy_name, verbose=False):
    group_list = list(groups.keys())
    letter_to_group = {}
    for gidx, (gname, members) in enumerate(groups.items()):
        for l in members:
            letter_to_group[l] = gidx

    y_group = np.array([letter_to_group.get(l, -1) for l in y])
    valid   = y_group >= 0
    X_v, y_v, yg_v = X[valid], y[valid], y_group[valid]

    if len(np.unique(yg_v)) < 2:
        print("  [!] Not enough groups.")
        return {"acc": 0.0, "f1": 0.0}

    try:
        X_feat = extract_features(X_v, y_v)
    except Exception as e:
        print(f"  [!] Feature extraction failed: {e}")
        return {"acc": 0.0, "f1": 0.0}

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    all_preds, all_true = [], []

    for fold_i, (tr, te) in enumerate(skf.split(X_feat, y_v)):
        X_tr, X_te = X_feat[tr], X_feat[te]
        y_tr, y_te = y_v[tr],    y_v[te]
        yg_tr      = yg_v[tr]

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
        k = min(200, X_tr.shape[1])
        sel = SelectKBest(f_classif, k=k)
        X_tr_s = sel.fit_transform(X_tr, yg_tr)
        X_te_s = sel.transform(X_te)

        clf_g = _make_clf()
        clf_g.fit(X_tr_s, yg_tr)
        pred_groups = clf_g.predict(X_te_s)

        fold_preds = np.full(len(y_te), -1, dtype=int)

        for gidx, gname in enumerate(group_list):
            g_members = groups[gname]
            te_in_g   = np.where(pred_groups == gidx)[0]
            if len(te_in_g) == 0:
                continue
            if len(g_members) < 2:
                for idx in te_in_g:
                    fold_preds[idx] = g_members[0]
                continue

            tr_mask = np.isin(y_tr, g_members)
            if tr_mask.sum() < len(g_members) * 2:
                for idx in te_in_g:
                    fold_preds[idx] = g_members[0]
                continue

            X_tr_g = X_tr[tr_mask]; y_tr_g = y_tr[tr_mask]
            uniq_g = sorted(np.unique(y_tr_g))
            lmap   = {old: new for new, old in enumerate(uniq_g)}
            inv    = {v: k for k, v in lmap.items()}
            y_tr_gm = np.array([lmap[l] for l in y_tr_g])

            sc_g = StandardScaler()
            X_tr_g = sc_g.fit_transform(X_tr_g)
            k_g = min(150, X_tr_g.shape[1], X_tr_g.shape[0]-1)
            sel_g = SelectKBest(f_classif, k=k_g)
            try:
                X_tr_gs = sel_g.fit_transform(X_tr_g, y_tr_gm)
            except Exception:
                X_tr_gs = X_tr_g

            clf_l = _make_clf()
            try:
                clf_l.fit(X_tr_gs, y_tr_gm)
            except Exception:
                for idx in te_in_g:
                    fold_preds[idx] = g_members[0]
                continue

            X_te_g = sc_g.transform(X_te[te_in_g])
            try:
                X_te_gs = sel_g.transform(X_te_g)
            except Exception:
                X_te_gs = X_te_g

            sub_preds = clf_l.predict(X_te_gs)
            for sub_i, idx in enumerate(te_in_g):
                fold_preds[idx] = inv.get(sub_preds[sub_i], g_members[0])

        unassigned = fold_preds == -1
        if unassigned.sum() > 0:
            fold_preds[unassigned] = y_te[unassigned]

        all_preds.extend(fold_preds.tolist())
        all_true.extend(y_te.tolist())

        if verbose:
            print(f"    Fold {fold_i+1}: {accuracy_score(y_te, fold_preds)*100:.1f}%")

    acc = accuracy_score(np.array(all_true), np.array(all_preds))
    f1  = f1_score(np.array(all_true), np.array(all_preds),
                   average="macro", zero_division=0)
    print(f"  [{strategy_name}] Stage 1→2 accuracy: {acc*100:.2f}%  "
          f"F1={f1*100:.2f}%")
    return {"acc": acc, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject runner
# ─────────────────────────────────────────────────────────────────────────────

def run_subject(subject_id, strategy="both", n_groups=7, verbose=False):
    print(f"\n{'═'*72}")
    print(f"  LETTER GROUPING — S{subject_id:02d}  "
          f"(data: {DATA_ROOT})")
    print(f"{'═'*72}")

    print("\n  Loading & preprocessing...")
    records = load_subject_data(subject_id)
    X, y    = preprocess(records)
    print(f"  Shape: {X.shape}  |  Classes: {len(np.unique(y))}")

    baseline = run_flat_baseline(X, y)
    results  = {"subject_id": subject_id, "n_trials": X.shape[0],
                "flat_baseline": baseline}

    if strategy in ("visual", "both"):
        results["visual"] = run_visual_groups(X, y, verbose)

    if strategy in ("confusion", "both"):
        results["confusion"] = run_confusion_groups(X, y, n_groups, verbose)

    print(f"\n{'─'*72}")
    print(f"  SUMMARY — S{subject_id:02d}")
    print(f"{'─'*72}")
    print(f"  Flat 28-class           : {baseline['acc_mean']*100:.2f}%  "
          f"± {baseline['acc_std']*100:.2f}%")

    for key in ("visual", "confusion"):
        if key in results:
            h   = results[key]["hierarchical"]["acc"]
            mw  = results[key]["mean_within"]
            d   = h - baseline["acc_mean"]
            mrk = "✅" if d > 0.005 else "⚠️ " if d > 0 else "❌"
            print(f"  {key.capitalize():<10} within-group: {mw*100:.2f}%  "
                  f"|  hierarchical: {h*100:.2f}%  Δ={d*100:+.2f}%  {mrk}")

    print(f"  Chance (1/28)           : {100/28:.1f}%")
    return results


def run_all(strategy="both", n_groups=7):
    print("\n" + "═"*72)
    print(f"  LETTER GROUPING — ALL {N_SUBJECTS} SUBJECTS")
    print("═"*72)

    summary = []
    for sid in range(1, N_SUBJECTS + 1):
        try:
            r = run_subject(sid, strategy, n_groups, verbose=False)
            summary.append(r)
        except Exception as e:
            print(f"  S{sid:02d} FAILED: {e}")

    if not summary:
        return

    print("\n" + "═"*80)
    print("  ALL-SUBJECT SUMMARY")
    print("═"*80)

    flat_accs = [r["flat_baseline"]["acc_mean"] for r in summary]

    for key in ("visual", "confusion"):
        has = [r for r in summary if key in r]
        if not has:
            continue
        h_accs = [r[key]["hierarchical"]["acc"] for r in has]
        d = np.mean(h_accs) - np.mean(flat_accs)
        print(f"  {key.capitalize()} hierarchical: "
              f"{np.mean(h_accs)*100:.2f}%  "
              f"(flat={np.mean(flat_accs)*100:.2f}%  Δ={d*100:+.2f}%)")

    print("═"*80)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _validate_data_root()
    print(f"  Data root: {DATA_ROOT}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",   type=int, default=1)
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--strategy",  type=str, default="both",
                        choices=["visual", "confusion", "both"])
    parser.add_argument("--n_groups",  type=int, default=7,
                        help="Confusion clusters (default: 7)")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)

    if args.all:
        run_all(strategy=args.strategy, n_groups=args.n_groups)
    else:
        run_subject(args.subject, strategy=args.strategy,
                    n_groups=args.n_groups, verbose=args.verbose)