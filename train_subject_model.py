"""
train_subject_model.py
======================
Trains the full band optimization pipeline on a subject's EDF recordings
and saves the fitted model to a .pkl file named after the subject.

Usage:
    python train_subject_model.py --name omar
    python train_subject_model.py --name omar --data_root /custom/path
    python train_subject_model.py --name omar --letters 8   # partial recording

After running, a file like:
    models/omar.pkl
is saved and can be used by the Flask app for real-time prediction.
"""

import numpy as np
import pickle
import argparse
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    raise ImportError("Run: pip install mne --break-system-packages")

from scipy.signal import butter, filtfilt, iirnotch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from config import CONFIG
from features import (
    riemannian_features, band_power_features,
    connectivity_features, adaptive_csp_features,
)

DEFAULT_DATA_ROOT = Path.home() / "Desktop" / "real time" / "S02"
MODELS_DIR = Path("models")

EMOTIV_CH_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

BANDS = {
    "broadband":   (0.5, 40.0),
    "delta_theta": (0.5,  8.0),
    "alpha":       (8.0, 13.0),
    "beta":        (13.0, 30.0),
    "gamma":       (30.0, 40.0),
    "alpha_beta":  (8.0,  30.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# EDF loading (same as run_new_subject.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_edf_trial(edf_path, sfreq_target=256):
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    available = raw.ch_names
    eeg_picks = []
    for ch in EMOTIV_CH_NAMES:
        if ch in available:
            eeg_picks.append(ch)
        else:
            matches = [c for c in available if c.upper() == ch.upper()]
            if matches:
                eeg_picks.append(matches[0])
    if len(eeg_picks) < 10:
        raise ValueError(f"Only {len(eeg_picks)} EEG channels found")
    raw.pick_channels(eeg_picks, ordered=True)
    if abs(raw.info["sfreq"] - sfreq_target) > 1:
        raw.resample(sfreq_target, verbose=False)
    data = raw.get_data()
    if np.abs(data).max() < 1.0:
        data = data * 1e6
    return data


def load_subject_edf(data_root, n_letters=28, n_trials=10):
    records = []
    for letter_id in range(1, n_letters + 1):
        l_folder = data_root / f"L{letter_id:02d}"
        if not l_folder.exists():
            continue
        edf_files = sorted(l_folder.glob("*.edf"))
        for trial_idx, edf_path in enumerate(edf_files[:n_trials]):
            try:
                eeg_data = load_edf_trial(edf_path)
                records.append({
                    "eeg":    eeg_data,
                    "label":  letter_id - 1,
                    "letter": letter_id,
                    "trial":  trial_idx + 1,
                })
            except Exception as e:
                print(f"  [warn] {edf_path.name}: {e}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _bp(data, lo, hi, fs):
    nyq = fs / 2.0
    b, a = butter(4, [max(lo,0.1)/nyq, min(hi,nyq-0.1)/nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)

def _notch(data, f0, fs):
    b, a = iirnotch(f0, Q=30.0, fs=fs)
    return filtfilt(b, a, data, axis=-1)

def _car(data):
    return data - data.mean(axis=0, keepdims=True)

def _laplacian(data):
    nb = {0:[1,2],1:[0,3],2:[0,3,4],3:[1,2,5],4:[2,3,5],5:[3,4,6],
          6:[5,7],7:[6,8],8:[7,9,10],9:[8,10,11],10:[8,9,12],
          11:[9,10,13],12:[10,11,13],13:[11,12]}
    r = data.copy()
    for ch, ns in nb.items():
        if ch < data.shape[0] and all(n < data.shape[0] for n in ns):
            r[ch] = data[ch] - data[ns].mean(axis=0)
    return r

def _reject(data, mult=50.0):
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    if mad < 1e-10:
        return False
    return bool(np.abs(data - med).max() > mult * mad)

def _baseline(data, fs, sec=0.2):
    n = int(sec * fs)
    return data - data[:, :n].mean(axis=1, keepdims=True)

def _zscore(X):
    m = X.mean(axis=(0,2), keepdims=True)
    s = X.std(axis=(0,2),  keepdims=True) + 1e-8
    return (X - m) / s

def preprocess(records):
    sfreq = CONFIG["sfreq"]
    t_start = int((CONFIG["t_relax"] + CONFIG["t_observe"]) * sfreq)
    t_end   = t_start + int(CONFIG["epoch_tmax"] * sfreq)
    epochs, labels = [], []
    rejected = 0
    for rec in records:
        raw = rec["eeg"].copy()
        if raw.shape[1] < t_end:
            rejected += 1
            continue
        ep = raw[:, t_start:t_end]
        ep = _bp(ep, *CONFIG["bandpass"], sfreq)
        for f0 in CONFIG["notch_freqs"]:
            ep = _notch(ep, f0, sfreq)
        ep = _car(ep)
        ep = _laplacian(ep)
        if _reject(ep):
            rejected += 1
            continue
        ep = _baseline(ep, sfreq)
        epochs.append(ep)
        labels.append(rec["label"])
    print(f"  Preprocessing: {len(epochs)} kept, {rejected} rejected")
    if not epochs:
        raise RuntimeError("All trials rejected")
    X = _zscore(np.array(epochs, dtype=np.float64))
    y = np.array(labels, dtype=np.int32)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract(X, y, include_csp=True):
    Xr = np.nan_to_num(riemannian_features(X))
    Xb = np.nan_to_num(band_power_features(X))
    Xp = np.nan_to_num(connectivity_features(X))
    if include_csp:
        Xc = np.nan_to_num(adaptive_csp_features(X, y))
        return np.concatenate([Xr, Xb, Xp, Xc], axis=1)
    return np.concatenate([Xr, Xb, Xp], axis=1)

def bp3d(X, lo, hi, fs):
    nyq = fs / 2.0
    b, a = butter(4, [max(lo,0.1)/nyq, min(hi,nyq-0.1)/nyq], btype="band")
    return filtfilt(b, a, X, axis=2)


# ─────────────────────────────────────────────────────────────────────────────
# Band optimization + find best config
# ─────────────────────────────────────────────────────────────────────────────

def split_score(X_feat, y, k=None, n_repeats=10):
    """
    Repeated stratified 80/20 train/test split evaluation.
    More honest than CV for small datasets — model trains on 80%
    and is evaluated on 20% it has never seen.
    Repeated n_repeats times with different random seeds for stability.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    min_c = int(np.bincount(y).min())
    if min_c < 2:
        return 0.0

    # Need at least 1 sample per class in test set
    # With 20% test and min_c samples, need min_c >= 5 for safety
    test_size = 0.2
    accs = []

    for seed in range(n_repeats):
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size,
            random_state=CONFIG["random_state"] + seed
        )
        try:
            tr, te = next(sss.split(X_feat, y))
        except ValueError:
            # Not enough samples for stratified split — fall back to random
            rng = np.random.RandomState(CONFIG["random_state"] + seed)
            n   = len(y)
            idx = rng.permutation(n)
            split = int(n * (1 - test_size))
            tr, te = idx[:split], idx[split:]

        Xtr, Xte = X_feat[tr], X_feat[te]
        ytr, yte = y[tr], y[te]

        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        if k:
            sel = SelectKBest(f_classif, k=min(k, Xtr.shape[1]))
            Xtr = sel.fit_transform(Xtr, ytr)
            Xte = sel.transform(Xte)

        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        try:
            clf.fit(Xtr, ytr)
            accs.append(accuracy_score(yte, clf.predict(Xte)))
        except Exception:
            accs.append(0.0)

    return float(np.mean(accs))


def find_best_config(X, y):
    """Find best band + K combination via 80/20 split evaluation."""
    sfreq    = CONFIG["sfreq"]
    best_acc = 0.0
    best_cfg = {"type": "broadband", "band": "broadband", "k": None}

    band_feats = {}
    for band_name, (lo, hi) in BANDS.items():
        Xf = bp3d(X, lo, hi, sfreq)
        band_feats[band_name] = extract(Xf, y, include_csp=False)

    # Single bands
    for band_name, Xf in band_feats.items():
        acc = split_score(Xf, y)
        if acc > best_acc:
            best_acc = acc
            best_cfg = {"type": "single", "band": band_name, "k": None}
        for k in [100, 150, 200, 250, 300]:
            acc = split_score(Xf, y, k=k)
            if acc > best_acc:
                best_acc = acc
                best_cfg = {"type": "single", "band": band_name, "k": k}

    # Multiband
    X_multi = np.concatenate(list(band_feats.values()), axis=1)
    for k in [100, 150, 200, 250, 300, 400, 500]:
        acc = split_score(X_multi, y, k=k)
        if acc > best_acc:
            best_acc = acc
            best_cfg = {"type": "multi", "band": "multiband", "k": k}

    print(f"  Best config: {best_cfg['type']} band={best_cfg['band']} "
          f"K={best_cfg['k']} → 80/20 acc={best_acc*100:.2f}%")
    return best_cfg, best_acc, band_feats, X_multi


def build_features_from_config(X, y, cfg, band_feats, X_multi):
    """Return the feature matrix for the winning config."""
    if cfg["type"] == "multi":
        return X_multi
    else:
        return band_feats[cfg["band"]]


def per_class_split(X, y, n_test_per_class=2):
    """
    Hold out n_test_per_class trials per class for testing.
    The rest go to training.
    Returns X_train, y_train, X_test, y_test.
    """
    train_idx, test_idx = [], []
    rng = np.random.RandomState(CONFIG["random_state"])

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = min(n_test_per_class, len(idx) - 2)  # keep at least 2 for training
        n_test = max(n_test, 1)
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    return (X[train_idx], y[train_idx],
            X[test_idx],  y[test_idx],
            len(test_idx))


# ─────────────────────────────────────────────────────────────────────────────
# Train final model on ALL data (no CV — this is the saved model)
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(X_feat, y, k=None):
    """Fit scaler + ANOVA + LDA on the full dataset."""
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_feat)

    selector = None
    if k:
        selector = SelectKBest(f_classif, k=min(k, X_sc.shape[1]))
        X_sc = selector.fit_transform(X_sc, y)

    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(X_sc, y)

    return scaler, selector, clf


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train and save subject model from EDF recordings"
    )
    parser.add_argument("--name", type=str, required=True,
                        help="Subject name (e.g. omar)")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--letters", type=int, default=28)
    parser.add_argument("--n_test", type=int, default=2,
                        help="Trials per class to hold out for testing (default: 2)")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"{args.name.lower()}.pkl"

    print("\n" + "═"*60)
    print(f"  TRAINING MODEL FOR: {args.name.upper()}")
    print(f"  Data: {data_root}")
    print(f"  Holding out {args.n_test} trials per class for testing")
    print("═"*60)

    # Load
    print("\n  [1/5] Loading EDF files...")
    records = load_subject_edf(data_root, n_letters=args.letters)
    print(f"  Loaded: {len(records)} trials across "
          f"{len(set(r['letter'] for r in records))} letters")

    # Preprocess
    print("\n  [2/5] Preprocessing...")
    X, y = preprocess(records)
    n_classes = len(np.unique(y))
    print(f"  Shape: {X.shape} | Classes: {n_classes}")

    # Per-class held-out split
    print(f"\n  [3/5] Splitting — {args.n_test} test trials per class...")
    X_train, y_train, X_test, y_test, n_test_total = per_class_split(
        X, y, n_test_per_class=args.n_test
    )
    print(f"  Train: {len(y_train)} trials | Test: {n_test_total} trials")

    # Find best config on TRAINING data only
    print("\n  [4/5] Finding best band config on training data...")
    best_cfg, best_acc, band_feats, X_multi = find_best_config(X_train, y_train)

    # Evaluate on held-out TEST set
    X_feat_train = build_features_from_config(
        X_train, y_train, best_cfg, band_feats, X_multi)

    # Build test features using same band config
    sfreq = CONFIG["sfreq"]
    if best_cfg["type"] == "multi":
        test_feats = []
        for lo, hi in BANDS.values():
            Xf = bp3d(X_test, lo, hi, sfreq)
            test_feats.append(extract(Xf, y_test, include_csp=False))
        X_feat_test = np.concatenate(test_feats, axis=1)
    else:
        lo, hi = BANDS[best_cfg["band"]]
        Xf = bp3d(X_test, lo, hi, sfreq)
        X_feat_test = extract(Xf, y_test, include_csp=False)

    # Scale + select + predict on held-out test
    sc  = StandardScaler()
    Xtr_sc = sc.fit_transform(X_feat_train)
    Xte_sc = sc.transform(X_feat_test)

    sel = None
    if best_cfg["k"]:
        sel  = SelectKBest(f_classif, k=min(best_cfg["k"], Xtr_sc.shape[1]))
        Xtr_sc = sel.fit_transform(Xtr_sc, y_train)
        Xte_sc = sel.transform(Xte_sc)

    clf_eval = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf_eval.fit(Xtr_sc, y_train)
    y_pred    = clf_eval.predict(Xte_sc)
    test_acc  = accuracy_score(y_test, y_pred)

    print(f"\n  ─── Held-out test results ───")
    print(f"  Test trials    : {n_test_total} ({args.n_test} per class)")
    print(f"  Test accuracy  : {test_acc*100:.2f}%")
    print(f"  Chance level   : {100/n_classes:.1f}%")

    # Per-letter test breakdown
    arabic_letters = [
        'ا','ب','ت','ث','ج','ح','خ','د','ذ','ر',
        'ز','س','ش','ص','ض','ط','ظ','ع','غ','ف',
        'ق','ك','ل','م','ن','ه','و','ي'
    ]
    print(f"\n  Per-letter test accuracy:")
    for cls in np.unique(y_test):
        mask    = y_test == cls
        correct = (y_pred[mask] == cls).sum()
        total   = mask.sum()
        ar      = arabic_letters[cls] if cls < len(arabic_letters) else f"L{cls}"
        status  = "✅" if correct == total else "❌" if correct == 0 else "⚠️ "
        print(f"    {ar} (L{cls+1:02d})  {correct}/{total}  {status}")

    # Now retrain on ALL data for the saved model
    print(f"\n  [5/5] Retraining on ALL data for saved model...")
    best_cfg2, _, band_feats2, X_multi2 = find_best_config(X, y)
    X_feat_all = build_features_from_config(X, y, best_cfg2, band_feats2, X_multi2)
    scaler, selector, clf = train_final_model(X_feat_all, y, k=best_cfg2["k"])

    model_bundle = {
        "name":            args.name,
        "config":          best_cfg2,
        "scaler":          scaler,
        "selector":        selector,
        "clf":             clf,
        "cv_accuracy":     test_acc,   # held-out test accuracy
        "n_classes":       n_classes,
        "n_letters":       args.letters,
        "arabic_letters":  arabic_letters[:n_classes],
        "sfreq":           CONFIG["sfreq"],
        "emotiv_channels": EMOTIV_CH_NAMES,
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"\n  ✅ Model saved: {model_path}")
    print(f"  Held-out accuracy : {test_acc*100:.2f}%")
    print(f"  Band config       : {best_cfg2['band']} K={best_cfg2['k']}")
    print(f"  Classes           : {n_classes} Arabic letters")
    print("═"*60)

    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"{args.name.lower()}.pkl"

    print("\n" + "═"*60)
    print(f"  TRAINING MODEL FOR: {args.name.upper()}")
    print(f"  Data: {data_root}")
    print("═"*60)

    # Load
    print("\n  [1/4] Loading EDF files...")
    records = load_subject_edf(data_root, n_letters=args.letters)
    print(f"  Loaded: {len(records)} trials across "
          f"{len(set(r['letter'] for r in records))} letters")

    # Preprocess
    print("\n  [2/4] Preprocessing...")
    X, y = preprocess(records)
    n_classes = len(np.unique(y))
    print(f"  Shape: {X.shape} | Classes: {n_classes}")

    # Find best config
    print("\n  [3/4] Finding best band configuration...")
    best_cfg, best_acc, band_feats, X_multi = find_best_config(X, y)

    # Build feature matrix for best config
    X_feat = build_features_from_config(X, y, best_cfg, band_feats, X_multi)

    # Train final model on ALL data
    print("\n  [4/4] Training final model on full dataset...")
    scaler, selector, clf = train_final_model(X_feat, y, k=best_cfg["k"])

    # Arabic letter names
    arabic_letters = [
        'ا','ب','ت','ث','ج','ح','خ','د','ذ','ر',
        'ز','س','ش','ص','ض','ط','ظ','ع','غ','ف',
        'ق','ك','ل','م','ن','ه','و','ي'
    ]

    # Save everything needed for inference
    model_bundle = {
        "name":           args.name,
        "config":         best_cfg,
        "scaler":         scaler,
        "selector":       selector,
        "clf":            clf,
        "cv_accuracy":    best_acc,   # now 80/20 split accuracy
        "n_classes":      n_classes,
        "n_letters":      args.letters,
        "arabic_letters": arabic_letters[:n_classes],
        "sfreq":          CONFIG["sfreq"],
        "emotiv_channels": EMOTIV_CH_NAMES,
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"\n  ✅ Model saved: {model_path}")
    print(f"  80/20 accuracy : {best_acc*100:.2f}%")
    print(f"  Band config    : {best_cfg['band']} K={best_cfg['k']}")
    print(f"  Classes        : {n_classes} Arabic letters")
    print(f"\n  Note: final model is trained on ALL data.")
    print(f"  The 80/20 accuracy was used only to select the best config.")
    print("═"*60)


if __name__ == "__main__":
    main()