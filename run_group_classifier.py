"""
run_group_classifier.py
========================
Classifies EEG trials into LETTER GROUPS instead of individual letters.

Two grouping strategies:
  Strategy A — Visual similarity (human-defined by letter shape)
  Strategy B — Confusion-driven (data-driven from confusion matrix)

For each strategy we ask:
  "If the task were to predict which GROUP a letter belongs to,
   how accurately can we do that?"

This reduces the problem from 28 classes to ~7-10 classes,
which should be substantially easier.

Key metrics reported:
  - Group-level accuracy (how well we classify into groups)
  - Chance level for that number of groups
  - Per-group accuracy (which groups are easiest/hardest)
  - Comparison to flat 28-class baseline

Usage:
    python run_group_classifier.py --subject 1
    python run_group_classifier.py --subject 1 --strategy visual
    python run_group_classifier.py --subject 1 --strategy confusion
    python run_group_classifier.py --subject 1 --strategy both --verbose
    python run_group_classifier.py --all
"""

from pathlib import Path
import os, sys, numpy as np, warnings, argparse
warnings.filterwarnings("ignore")

import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch, welch, hilbert
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

try:
    import mne; mne.set_log_level("ERROR")
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
#  PATH RESOLUTION  (no config.py dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_data_root() -> Path:
    # 1. Environment variable
    env = os.environ.get("EEG_DATA_ROOT")
    if env:
        return Path(env)
    # 2. .env file next to this script
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("EEG_DATA_ROOT"):
                val = line.split("=", 1)[-1].strip().strip('"').strip("'")
                if val:
                    return Path(val)
    # 3. Default fallback
    return Path.home() / "Downloads" / "Raw_Imagined_Arabic_Letters_Dataset"

DATA_ROOT = _resolve_data_root()

def _check_data_root():
    if not DATA_ROOT.exists():
        print(f"""
  DATA FOLDER NOT FOUND: {DATA_ROOT}
  Fix: export EEG_DATA_ROOT="/path/to/your/data"
  Or:  create a .env file with EEG_DATA_ROOT=/path/to/your/data
""")
        sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

N_SUBJECTS  = 30
N_LETTERS   = 28
N_TRIALS    = 10
SFREQ       = 256
CHANNEL_NAMES = ["AF3","AF4","F7","F8","F3","F4","FC5","FC6",
                 "T7","T8","P7","P8","O1","O2"]
T_RELAX     = 5.0
T_OBSERVE   = 5.0
EPOCH_TMAX  = 6.0
BANDPASS    = (0.5, 40.0)
NOTCH_FREQS = [50.0]
FREQ_BANDS  = {"delta":(0.5,4),"theta":(4,8),"alpha":(8,13),
               "beta":(13,30),"gamma":(30,40)}
CV_FOLDS    = 10
RANDOM_STATE = 42
CSP_MAP     = [(200,2),(250,4),(999,6)]

# ══════════════════════════════════════════════════════════════════════════════
#  ARABIC LETTER METADATA
# ══════════════════════════════════════════════════════════════════════════════

ARABIC_LETTERS = [
    "ا","ب","ت","ث","ج","ح","خ",
    "د","ذ","ر","ز","س","ش","ص",
    "ض","ط","ظ","ع","غ","ف","ق",
    "ك","ل","م","ن","ه","و","ي",
]

# Strategy A: groups by shared visual base shape (indices 0-27)
VISUAL_GROUPS = {
    "alef_family" : [0, 7, 8, 9, 10, 26, 27],  # ا د ذ ر ز و ي
    "ba_family"   : [1, 2, 3, 24],              # ب ت ث ن
    "jeem_family" : [4, 5, 6],                  # ج ح خ
    "seen_family" : [11, 12],                   # س ش
    "sad_family"  : [13, 14],                   # ص ض
    "ta_family"   : [15, 16],                   # ط ظ
    "ayn_family"  : [17, 18],                   # ع غ
    "fa_qaf"      : [19, 20],                   # ف ق
    "kaf_lam"     : [21, 22],                   # ك ل
    "meem_ha"     : [23, 25],                   # م ه
}

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_subject(subject_id: int):
    records = []
    s_folder = DATA_ROOT / f"S{subject_id:02d}"
    if not s_folder.exists():
        raise FileNotFoundError(f"Not found: {s_folder}")
    for letter_id in range(1, N_LETTERS + 1):
        l_folder = s_folder / f"L{letter_id:02d}"
        for trial_id in range(1, N_TRIALS + 1):
            mat_path = l_folder / f"S{subject_id:02d}_L{letter_id:02d}_T{trial_id}.mat"
            if not mat_path.exists():
                continue
            mat = sio.loadmat(str(mat_path), simplify_cells=True)
            es  = mat.get("EEG", {})
            if isinstance(es, np.ndarray):
                es = {k: es[k].item() for k in es.dtype.names}
            if isinstance(es, dict) and "Data" in es:
                eeg = np.array(es["Data"], dtype=np.float64)
            elif isinstance(es, dict) and "data" in es:
                eeg = np.array(es["data"], dtype=np.float64)
            else:
                best, bsz = None, 0
                for v in mat.values():
                    if isinstance(v, np.ndarray) and v.ndim==2 and v.size>bsz:
                        best, bsz = v, v.size
                if best is None: continue
                eeg = best.astype(np.float64)
            records.append({"eeg": eeg, "label": letter_id - 1})
    return records

def preprocess(records):
    sfreq  = SFREQ
    onset  = int((T_RELAX + T_OBSERVE) * sfreq)
    offset = onset + int(EPOCH_TMAX * sfreq)
    epochs, labels = [], []
    for rec in records:
        raw = rec["eeg"].copy()
        if raw.shape[1] < offset: continue
        ep = raw[:, onset:offset]
        # bandpass
        nyq = sfreq / 2
        b, a = butter(4, [BANDPASS[0]/nyq, BANDPASS[1]/nyq], btype="band")
        ep = filtfilt(b, a, ep, axis=1)
        # notch
        for f0 in NOTCH_FREQS:
            b, a = iirnotch(f0, 30.0, sfreq)
            ep = filtfilt(b, a, ep, axis=1)
        # CAR
        ep = ep - ep.mean(axis=0, keepdims=True)
        # amplitude rejection
        if np.median(np.abs(ep)) <= 50.0 and np.abs(ep).max() > 100.0:
            continue
        # baseline
        n_base = int(0.2 * sfreq)
        ep = ep - ep[:, :n_base].mean(axis=1, keepdims=True)
        epochs.append(ep)
        labels.append(rec["label"])
    X = np.array(epochs, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    # z-score
    mu  = X.mean(axis=(0,2), keepdims=True)
    sig = X.std( axis=(0,2), keepdims=True) + 1e-8
    return (X - mu) / sig, y

# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _n_csp(n_trials):
    for thr, n in CSP_MAP:
        if n_trials < thr: return n
    return CSP_MAP[-1][1]

def feat_riemannian(X):
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        covs = Covariances("oas").fit_transform(X)
        return TangentSpace("riemann").fit_transform(covs).astype(np.float32)
    except ImportError:
        n_ch = X.shape[1]
        return np.array([np.cov(t)[np.triu_indices(n_ch)] for t in X],
                        dtype=np.float32)

def feat_band_power(X):
    feats = []
    for trial in X:
        row = []
        for ch in trial:
            freqs, psd = welch(ch, fs=SFREQ, nperseg=min(256, ch.shape[0]))
            total = psd.sum() + 1e-10
            for lo, hi in FREQ_BANDS.values():
                mask = (freqs >= lo) & (freqs < hi)
                ap   = float(psd[mask].sum())
                row += [ap, ap / total]
        feats.append(row)
    return np.array(feats, dtype=np.float32)

def feat_plv(X):
    n_ch = X.shape[1]
    bands = {"alpha": (8,13), "beta": (13,30)}
    feats = []
    def _bp(sig, lo, hi):
        nyq = SFREQ/2
        b, a = butter(4, [lo/nyq, hi/nyq], btype="band")
        return filtfilt(b, a, sig)
    for trial in X:
        row = []
        for lo, hi in bands.values():
            filt  = np.array([_bp(ch, lo, hi) for ch in trial])
            phase = np.angle(__import__("scipy.signal",fromlist=["hilbert"])
                             .hilbert(filt, axis=1))
            for i in range(n_ch):
                for j in range(i+1, n_ch):
                    row.append(float(np.abs(np.mean(
                        np.exp(1j*(phase[i]-phase[j]))))))
        feats.append(row)
    return np.array(feats, dtype=np.float32)

def feat_csp(X, y):
    n_comp = _n_csp(X.shape[0])
    classes, all_f = np.unique(y), []
    try:
        from mne.decoding import CSP as MNE_CSP
        _mne = True
    except ImportError:
        _mne = False
    for cls in classes:
        y_bin = (y == cls).astype(int)
        if y_bin.sum() < 5: continue
        if _mne:
            try:
                csp = MNE_CSP(n_components=n_comp, log=True, norm_trace=False)
                all_f.append(csp.fit_transform(X, y_bin))
                continue
            except Exception:
                pass
        # numpy fallback
        pos, neg = X[y_bin==1], X[y_bin==0]
        C1 = np.array([np.cov(t) for t in pos]).mean(0)
        C2 = np.array([np.cov(t) for t in neg]).mean(0)
        Cc = C1 + C2
        vals, vecs = np.linalg.eigh(Cc)
        W = vecs @ np.diag(1/np.sqrt(vals+1e-10)) @ vecs.T
        S1 = W @ C1 @ W.T
        evals, evecs = np.linalg.eigh(S1)
        filters = evecs[:, np.argsort(evals)[::-1][:n_comp]].T @ W
        proj = np.einsum("cd,tds->tcs", filters, X)
        all_f.append(np.log(proj.var(axis=2) + 1e-10))
    if not all_f:
        return np.zeros((X.shape[0], 1), dtype=np.float32)
    return np.concatenate(all_f, axis=1).astype(np.float32)

def extract_features(X, y):
    return np.concatenate([
        np.nan_to_num(feat_riemannian(X)),
        np.nan_to_num(feat_band_power(X)),
        np.nan_to_num(feat_plv(X)),
        np.nan_to_num(feat_csp(X, y)),
    ], axis=1)

# ══════════════════════════════════════════════════════════════════════════════
#  CV HELPER
# ══════════════════════════════════════════════════════════════════════════════

def cv_eval(X_feat, y_group, n_splits=10, k_feat=200):
    """
    Stratified k-fold CV on the GROUP labels.
    Returns accuracy, std, f1, and per-group accuracy.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)
    accs, f1s = [], []
    n_grps = len(np.unique(y_group))
    cm_accum = np.zeros((n_grps, n_grps))

    for tr, te in skf.split(X_feat, y_group):
        X_tr, X_te = X_feat[tr], X_feat[te]
        y_tr, y_te = y_group[tr], y_group[te]

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        k = min(k_feat, X_tr.shape[1], X_tr.shape[0] - 1)
        if k > 1:
            sel  = SelectKBest(f_classif, k=k)
            X_tr = sel.fit_transform(X_tr, y_tr)
            X_te = sel.transform(X_te)

        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
        cm_accum += confusion_matrix(y_te, y_pred, labels=np.arange(n_grps))

    # Per-group accuracy from the accumulated CM
    row_sums = cm_accum.sum(axis=1)
    per_group_acc = np.divide(np.diag(cm_accum), row_sums,
                              where=row_sums > 0,
                              out=np.zeros(n_grps))
    return {
        "acc_mean":     float(np.mean(accs)),
        "acc_std":      float(np.std(accs)),
        "f1_macro":     float(np.mean(f1s)),
        "per_group_acc": per_group_acc,
        "cm":           cm_accum,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY A — Visual groups
# ══════════════════════════════════════════════════════════════════════════════

def run_visual_strategy(X, y, X_feat, verbose=False):
    """
    Relabel every trial from its letter label (0-27)
    to its visual GROUP label, then classify.
    Groups whose letters were all excluded are dropped automatically.
    """
    print("\n  ══ STRATEGY A: Visual Similarity Groups ══")

    active_letters = set(np.unique(y).tolist())

    # Only keep groups that have at least one letter with trials
    active_groups = {
        gname: members
        for gname, members in VISUAL_GROUPS.items()
        if any(l in active_letters for l in members)
    }
    dropped = set(VISUAL_GROUPS.keys()) - set(active_groups.keys())
    if dropped:
        print(f"  [info] Dropped {len(dropped)} empty group(s): "
              f"{', '.join(dropped)}")

    # Build letter → contiguous group index (0, 1, 2, ...)
    group_names = list(active_groups.keys())
    letter_to_group = {}
    for gidx, (gname, members) in enumerate(active_groups.items()):
        for l in members:
            letter_to_group[l] = gidx

    # Relabel y; letters not in any active group get -1
    y_group    = np.array([letter_to_group.get(int(l), -1) for l in y])
    unassigned = (y_group == -1).sum()
    if unassigned > 0:
        print(f"  [warn] {unassigned} trials not in any group — skipped.")
    mask  = y_group >= 0
    X_f_v = X_feat[mask]
    y_grp = y_group[mask]

    n_groups = len(np.unique(y_grp))
    n_trials = len(y_grp)
    chance   = 100.0 / n_groups

    print(f"\n  Groups   : {n_groups}")
    print(f"  Trials   : {n_trials}")
    print(f"  Chance   : {chance:.1f}%")

    # Group composition table
    print(f"\n  {'Group':<20} {'Letters':<35} {'Trials':>7}")
    print("  " + "─" * 62)
    for gidx, (gname, members) in enumerate(active_groups.items()):
        n       = (y_grp == gidx).sum()
        letters = " ".join(ARABIC_LETTERS[i] for i in members
                           if i in active_letters)
        print(f"  {gname:<20} {letters:<35} {n:>7}")

    # Run CV on group labels
    print(f"\n  Running {CV_FOLDS}-fold CV on group labels...")
    r = cv_eval(X_f_v, y_grp, n_splits=CV_FOLDS)

    print(f"\n  Group-level accuracy : {r['acc_mean']*100:.2f}% "
          f"± {r['acc_std']*100:.2f}%")
    print(f"  Group-level F1 macro : {r['f1_macro']*100:.2f}%")
    print(f"  Chance level         : {chance:.1f}%")
    print(f"  Above chance         : {r['acc_mean']*100 - chance:+.1f}%")

    # Per-group accuracy — gidx always in bounds because we re-indexed
    print(f"\n  Per-group accuracy:")
    print(f"  {'Group':<20} {'Letters':<35} {'Acc':>7}  {'Trials':>7}")
    print("  " + "─" * 70)
    for gidx, (gname, members) in enumerate(active_groups.items()):
        n       = (y_grp == gidx).sum()
        ga      = r["per_group_acc"][gidx] * 100
        letters = " ".join(ARABIC_LETTERS[i] for i in members
                           if i in active_letters)
        marker  = "✅" if ga > 70 else "⚠️ " if ga > 50 else "❌"
        print(f"  {gname:<20} {letters:<35} {ga:>6.1f}%  {n:>7}  {marker}")

    return {
        "strategy":      "visual",
        "n_groups":      n_groups,
        "acc_mean":      r["acc_mean"],
        "acc_std":       r["acc_std"],
        "f1_macro":      r["f1_macro"],
        "chance":        chance / 100,
        "per_group_acc": r["per_group_acc"],
        "group_names":   group_names,
        "cm":            r["cm"],
    }

# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY B — Confusion-driven groups
# ══════════════════════════════════════════════════════════════════════════════

def build_confusion_groups(X_feat, y, n_groups=7):
    """
    1. Run CV on all 28 letter labels → confusion matrix
    2. Cluster letters by how often the model confuses them
    3. Return the cluster assignments
    """
    print(f"\n  Building confusion matrix (5-fold CV, all 28 letters)...")
    n_cls    = N_LETTERS
    skf      = StratifiedKFold(n_splits=5, shuffle=True,
                               random_state=RANDOM_STATE)
    cm_accum = np.zeros((n_cls, n_cls))

    for tr, te in skf.split(X_feat, y):
        X_tr, X_te = X_feat[tr], X_feat[te]
        y_tr, y_te = y[tr], y[te]
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
        k    = min(200, X_tr.shape[1])
        sel  = SelectKBest(f_classif, k=k)
        X_tr = sel.fit_transform(X_tr, y_tr); X_te = sel.transform(X_te)
        clf  = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X_tr, y_tr)
        cm_accum += confusion_matrix(clf.predict(X_te), y_te,
                                     labels=np.arange(n_cls))

    row_sums = cm_accum.sum(axis=1, keepdims=True)
    cm_norm  = cm_accum / (row_sums + 1e-10)

    # Symmetric confusion distance: high confusion → small distance
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
    pairs = sorted([
        ((cm_norm[i,j]+cm_norm[j,i])/2, i, j)
        for i in range(n_cls) for j in range(i+1, n_cls)
    ], reverse=True)

    return groups, cm_norm, pairs


def run_confusion_strategy(X, y, X_feat, n_groups=7, verbose=False):
    """
    Build confusion-driven groups, relabel trials to group IDs,
    then run a group-level classifier.
    """
    print(f"\n  ══ STRATEGY B: Confusion-Driven Groups ({n_groups} clusters) ══")

    groups, cm_norm, pairs = build_confusion_groups(X_feat, y, n_groups)

    # Print top confused pairs
    print(f"\n  Top-15 most confused letter pairs:")
    print(f"  {'Pair':<22}  {'Confusion rate':>14}")
    print("  " + "─"*38)
    for rate, i, j in pairs[:15]:
        print(f"  {ARABIC_LETTERS[i]}(L{i+1:02d}) ↔ "
              f"{ARABIC_LETTERS[j]}(L{j+1:02d})    {rate*100:>10.1f}%")

    # Show discovered groups
    print(f"\n  Discovered groups:")
    group_names = list(groups.keys())
    for gname, members in groups.items():
        letters = " ".join(ARABIC_LETTERS[i] for i in members)
        print(f"    {gname}: {letters}")

    # Relabel y
    letter_to_group = {}
    for gidx, (gname, members) in enumerate(groups.items()):
        for l in members:
            letter_to_group[l] = gidx

    y_group = np.array([letter_to_group.get(int(l), -1) for l in y])
    mask    = y_group >= 0
    X_f_c   = X_feat[mask]
    y_grp   = y_group[mask]

    n_grps   = len(np.unique(y_grp))
    n_trials = len(y_grp)
    chance   = 100.0 / n_grps

    print(f"\n  Groups   : {n_grps}")
    print(f"  Trials   : {n_trials}  (all letters included)")
    print(f"  Chance   : {chance:.1f}%")

    # Group composition table
    print(f"\n  {'Group':<20} {'Letters':<40} {'Trials':>7}")
    print("  " + "─" * 68)
    for gidx, (gname, members) in enumerate(groups.items()):
        n       = (y_grp == gidx).sum()
        letters = " ".join(ARABIC_LETTERS[i] for i in members)
        print(f"  {gname:<20} {letters:<40} {n:>7}")

    # Run CV on group labels
    print(f"\n  Running {CV_FOLDS}-fold CV on group labels...")
    r = cv_eval(X_f_c, y_grp, n_splits=CV_FOLDS)

    print(f"\n  Group-level accuracy : {r['acc_mean']*100:.2f}% "
          f"± {r['acc_std']*100:.2f}%")
    print(f"  Group-level F1 macro : {r['f1_macro']*100:.2f}%")
    print(f"  Chance level         : {chance:.1f}%")
    print(f"  Above chance         : {r['acc_mean']*100 - chance:+.1f}%")

    # Per-group accuracy
    print(f"\n  Per-group accuracy:")
    print(f"  {'Group':<20} {'Letters':<40} {'Acc':>7}  {'Trials':>7}")
    print("  " + "─" * 76)
    for gidx, (gname, members) in enumerate(groups.items()):
        n       = (y_grp == gidx).sum()
        ga      = r["per_group_acc"][gidx] * 100
        letters = " ".join(ARABIC_LETTERS[i] for i in members)
        marker  = "✅" if ga > 70 else "⚠️ " if ga > 50 else "❌"
        print(f"  {gname:<20} {letters:<40} {ga:>6.1f}%  {n:>7}  {marker}")

    return {
        "strategy":      "confusion",
        "n_groups":      n_grps,
        "acc_mean":      r["acc_mean"],
        "acc_std":       r["acc_std"],
        "f1_macro":      r["f1_macro"],
        "chance":        chance / 100,
        "per_group_acc": r["per_group_acc"],
        "group_names":   group_names,
        "groups":        groups,
        "cm":            r["cm"],
        "top_confusions": pairs[:15],
    }

# ══════════════════════════════════════════════════════════════════════════════
#  FLAT BASELINE (28-class, same features — reference point)
# ══════════════════════════════════════════════════════════════════════════════

def run_flat_baseline(X_feat, y):
    print("\n  ── Flat baseline (28-class LDA) ──")
    r = cv_eval(X_feat, y, n_splits=CV_FOLDS, k_feat=200)
    print(f"  28-class accuracy : {r['acc_mean']*100:.2f}% "
          f"± {r['acc_std']*100:.2f}%  (chance={100/28:.1f}%)")
    return r

# ══════════════════════════════════════════════════════════════════════════════
#  PER-SUBJECT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_subject(subject_id, strategy="both", n_groups=7, verbose=False):
    print(f"\n{'═'*72}")
    print(f"  GROUP CLASSIFIER — S{subject_id:02d}")
    print(f"  Data: {DATA_ROOT}")
    print(f"{'═'*72}")

    print("\n  Loading & preprocessing...")
    records = load_subject(subject_id)
    X, y    = preprocess(records)

    # ── Letter trial audit ────────────────────────────────────────────────
    # Only exclude letters with 0 trials; keep incomplete ones as-is
    expected_per_letter = N_TRIALS
    trial_counts = {i: int((y == i).sum()) for i in range(N_LETTERS)}
    zero_letters = [i for i, n in trial_counts.items() if n == 0]

    print(f"\n  ── Letter trial audit ──")
    print(f"  {'Letter':<6} {'Name':<6} {'Trials':>7}  Status")
    print("  " + "─" * 35)
    for i in range(N_LETTERS):
        n      = trial_counts[i]
        status = "❌ EXCLUDED (0 trials)" if n == 0 else \
                 f"⚠️  incomplete ({n}/{expected_per_letter})" \
                 if n < expected_per_letter else "✅ ok"
        if n != expected_per_letter:
            print(f"  L{i+1:02d}   {ARABIC_LETTERS[i]:<6} {n:>7}  {status}")

    if zero_letters:
        excluded_names = " ".join(
            f"{ARABIC_LETTERS[i]}(L{i+1:02d})" for i in zero_letters)
        print(f"\n  Excluding {len(zero_letters)} letter(s) with 0 trials: "
              f"{excluded_names}")
        keep = ~np.isin(y, zero_letters)
        X, y = X[keep], y[keep]
        print(f"  Trials after exclusion: {len(y)}")
    else:
        print(f"  No letters with 0 trials ✅")

    print(f"\n  Shape: {X.shape}  |  Letters kept: {len(np.unique(y))}/28")

    print("\n  Extracting features (this takes ~1–2 min)...")
    X_feat = extract_features(X, y)
    print(f"  Feature shape: {X_feat.shape}")

    # Flat 28-class baseline
    baseline = run_flat_baseline(X_feat, y)

    results = {
        "subject_id": subject_id,
        "n_trials":   X.shape[0],
        "flat_28":    baseline,
    }

    if strategy in ("visual", "both"):
        results["visual"] = run_visual_strategy(X, y, X_feat, verbose)

    if strategy in ("confusion", "both"):
        results["confusion"] = run_confusion_strategy(
            X, y, X_feat, n_groups, verbose)

    # ── Final comparison ──────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  FINAL COMPARISON — S{subject_id:02d}")
    print(f"{'═'*72}")
    print(f"  {'Method':<35} {'Acc':>8}  {'Chance':>8}  {'Above chance':>13}")
    print(f"  {'─'*65}")
    print(f"  {'Flat 28-class (no grouping)':<35} "
          f"{baseline['acc_mean']*100:>7.2f}%  "
          f"{100/28:>7.1f}%  "
          f"{baseline['acc_mean']*100 - 100/28:>+12.1f}%")

    for key, label in [("visual",    "Visual groups"),
                        ("confusion", "Confusion groups")]:
        if key in results:
            r      = results[key]
            chance = r["chance"] * 100
            acc    = r["acc_mean"] * 100
            n_grps = r["n_groups"]
            print(f"  {label+f' ({n_grps} groups)':<35} "
                  f"{acc:>7.2f}%  "
                  f"{chance:>7.1f}%  "
                  f"{acc - chance:>+12.1f}%")

    print(f"{'═'*72}")
    return results

# ══════════════════════════════════════════════════════════════════════════════
#  ALL SUBJECTS
# ══════════════════════════════════════════════════════════════════════════════

def run_all(strategy="both", n_groups=7):
    print("\n" + "═"*72)
    print(f"  GROUP CLASSIFIER — ALL {N_SUBJECTS} SUBJECTS")
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

    flat_accs = [r["flat_28"]["acc_mean"] for r in summary]

    print("\n" + "═"*80)
    print("  SUMMARY — ALL SUBJECTS")
    print("═"*80)

    header = f"  {'Subj':<6} {'Flat-28':>8}"
    if strategy in ("visual", "both"):
        header += f"  {'Visual':>8}  {'V-chance':>9}  {'V-above':>8}"
    if strategy in ("confusion", "both"):
        header += f"  {'Conf':>8}  {'C-chance':>9}  {'C-above':>8}"
    print(header)
    print("  " + "─"*72)

    for r in summary:
        line = f"  S{r['subject_id']:02d}   {r['flat_28']['acc_mean']*100:>7.1f}%"
        if "visual" in r:
            v  = r["visual"]
            ch = v["chance"] * 100
            line += (f"  {v['acc_mean']*100:>7.1f}%  {ch:>8.1f}%  "
                     f"{v['acc_mean']*100 - ch:>+7.1f}%")
        if "confusion" in r:
            c  = r["confusion"]
            ch = c["chance"] * 100
            line += (f"  {c['acc_mean']*100:>7.1f}%  {ch:>8.1f}%  "
                     f"{c['acc_mean']*100 - ch:>+7.1f}%")
        print(line)

    print("  " + "─"*72)
    means_line = f"  {'MEAN':<6} {np.mean(flat_accs)*100:>7.1f}%"
    if strategy in ("visual", "both"):
        v_accs = [r["visual"]["acc_mean"] for r in summary if "visual" in r]
        v_ch   = [r["visual"]["chance"]   for r in summary if "visual" in r]
        means_line += (f"  {np.mean(v_accs)*100:>7.1f}%  "
                       f"{np.mean(v_ch)*100:>8.1f}%  "
                       f"{np.mean(v_accs)*100 - np.mean(v_ch)*100:>+7.1f}%")
    if strategy in ("confusion", "both"):
        c_accs = [r["confusion"]["acc_mean"] for r in summary if "confusion" in r]
        c_ch   = [r["confusion"]["chance"]   for r in summary if "confusion" in r]
        means_line += (f"  {np.mean(c_accs)*100:>7.1f}%  "
                       f"{np.mean(c_ch)*100:>8.1f}%  "
                       f"{np.mean(c_accs)*100 - np.mean(c_ch)*100:>+7.1f}%")
    print(means_line)
    print("═"*80)

# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _check_data_root()
    print(f"  Data root: {DATA_ROOT}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",  type=int, default=1)
    parser.add_argument("--all",      action="store_true")
    parser.add_argument("--strategy", type=str, default="both",
                        choices=["visual", "confusion", "both"])
    parser.add_argument("--n_groups", type=int, default=7,
                        help="Number of confusion-driven clusters (default: 7)")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)

    if args.all:
        run_all(strategy=args.strategy, n_groups=args.n_groups)
    else:
        run_subject(args.subject, strategy=args.strategy,
                    n_groups=args.n_groups, verbose=args.verbose)
