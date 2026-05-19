"""
run_new_subject.py
==================
Runs the full band optimization pipeline on a new subject recorded
with the EMOTIV Epoc X headset and saved as EDF files.
Now includes detailed per-letter accuracy and misclassification analysis.

Folder structure expected (on your Desktop):
    real time/
        S01/
            L01/
                *T01*.edf
                *T02*.edf
                ... (up to 10 trials)
            L02/
                ...
            ...
            L28/

Usage:
    python run_new_subject.py                        # run full pipeline
    python run_new_subject.py --preview              # check files only
    python run_new_subject.py --letters 8            # only first N letters
    python run_new_subject.py --data_root /custom    # custom path
"""

import numpy as np
import warnings
import argparse
from pathlib import Path
from collections import Counter
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
from sklearn.metrics import accuracy_score, f1_score

from config import CONFIG
from features import (
    riemannian_features, band_power_features,
    connectivity_features, adaptive_csp_features,
)

DEFAULT_DATA_ROOT = Path.home() / "Desktop" / "real time" / "S02"

EMOTIV_CH_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

ARABIC_LETTERS = [
    'ا','ب','ت','ث','ج','ح','خ','د','ذ','ر',
    'ز','س','ش','ص','ض','ط','ظ','ع','غ','ف',
    'ق','ك','ل','م','ن','ه','و','ي'
]

BANDS = {
    "broadband":   (0.5, 40.0),
    "delta_theta": (0.5,  8.0),
    "alpha":       (8.0, 13.0),
    "beta":        (13.0, 30.0),
    "gamma":       (30.0, 40.0),
    "alpha_beta":  (8.0,  30.0),
}

MULTIBAND_K_VALUES   = [100, 150, 200, 250, 300, 400, 500]
SINGLE_BAND_K_VALUES = [100, 150, 200, 250, 300]


# ─────────────────────────────────────────────────────────────────────────────
# EDF loading
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


def load_subject_edf(data_root, n_letters=28, n_trials=10, verbose=True):
    records = []
    missing = 0
    for letter_id in range(1, n_letters + 1):
        l_folder = data_root / f"L{letter_id:02d}"
        if not l_folder.exists():
            if verbose:
                print(f"  [skip] L{letter_id:02d} not found")
            continue
        edf_files = sorted(l_folder.glob("*.edf"))
        if not edf_files:
            continue
        for trial_idx, edf_path in enumerate(edf_files[:n_trials]):
            try:
                eeg_data = load_edf_trial(edf_path)
                records.append({
                    "eeg":    eeg_data,
                    "label":  letter_id - 1,
                    "letter": letter_id,
                    "trial":  trial_idx + 1,
                    "sfreq":  CONFIG["sfreq"],
                    "path":   str(edf_path),
                })
            except Exception as e:
                if verbose:
                    print(f"  [warn] {edf_path.name}: {e}")
                missing += 1
    if verbose:
        print(f"\n  Loaded: {len(records)} trials across "
              f"{len(set(r['letter'] for r in records))} letters")
        if missing:
            print(f"  Failed: {missing}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _bp(data, lo, hi, fs):
    nyq = fs / 2.0
    b, a = butter(4, [max(lo, 0.1)/nyq, min(hi, nyq-0.1)/nyq], btype="band")
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

def preprocess_edf_records(records, verbose=True):
    sfreq       = CONFIG["sfreq"]
    t_start     = int((CONFIG["t_relax"] + CONFIG["t_observe"]) * sfreq)
    t_end       = t_start + int(CONFIG["epoch_tmax"] * sfreq)
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

    if verbose:
        print(f"  Preprocessing: {len(epochs)} kept, {rejected} rejected")

    if not epochs:
        raise RuntimeError("All trials rejected — check your EDF files")

    X = _zscore(np.array(epochs, dtype=np.float64))
    y = np.array(labels, dtype=np.int32)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract(X, y):
    Xr = np.nan_to_num(riemannian_features(X))
    Xb = np.nan_to_num(band_power_features(X))
    Xp = np.nan_to_num(connectivity_features(X))
    Xc = np.nan_to_num(adaptive_csp_features(X, y))
    return np.concatenate([Xr, Xb, Xp, Xc], axis=1)

def _bp3d(X, lo, hi, fs):
    nyq = fs / 2.0
    b, a = butter(4, [max(lo,0.1)/nyq, min(hi,nyq-0.1)/nyq], btype="band")
    return filtfilt(b, a, X, axis=2)


# ─────────────────────────────────────────────────────────────────────────────
# CV evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _cv_eval(X_feat, y, k=None):
    min_c    = int(np.bincount(y).min())
    if min_c < 2:
        return {"acc_mean": 0.0, "acc_std": 0.0, "f1_macro": 0.0}
    n_splits = max(min(CONFIG["cv_folds"], min_c), 2)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=CONFIG["random_state"])
    accs, f1s = [], []
    for tr, te in skf.split(X_feat, y):
        Xtr, Xte = X_feat[tr], X_feat[te]
        ytr, yte = y[tr], y[te]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        if k:
            sel  = SelectKBest(f_classif, k=min(k, Xtr.shape[1]))
            Xtr  = sel.fit_transform(Xtr, ytr)
            Xte  = sel.transform(Xte)
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        try:
            clf.fit(Xtr, ytr)
            accs.append(accuracy_score(yte, clf.predict(Xte)))
            f1s.append(f1_score(yte, clf.predict(Xte),
                                average="macro", zero_division=0))
        except Exception:
            accs.append(0.0); f1s.append(0.0)
    return {"acc_mean": float(np.mean(accs)),
            "acc_std":  float(np.std(accs)),
            "f1_macro": float(np.mean(f1s))}


def _cv_collect(X_feat, y, k=None):
    """Same as _cv_eval but returns all true/pred labels for confusion analysis."""
    min_c    = int(np.bincount(y).min())
    if min_c < 2:
        return np.array([]), np.array([])
    n_splits = max(min(CONFIG["cv_folds"], min_c), 2)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=CONFIG["random_state"])
    all_true, all_pred = [], []
    for tr, te in skf.split(X_feat, y):
        Xtr, Xte = X_feat[tr], X_feat[te]
        ytr, yte = y[tr], y[te]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        if k:
            sel  = SelectKBest(f_classif, k=min(k, Xtr.shape[1]))
            Xtr  = sel.fit_transform(Xtr, ytr)
            Xte  = sel.transform(Xte)
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        try:
            clf.fit(Xtr, ytr)
            preds = clf.predict(Xte)
            all_true.extend(yte.tolist())
            all_pred.extend(preds.tolist())
        except Exception:
            pass
    return np.array(all_true), np.array(all_pred)


# ─────────────────────────────────────────────────────────────────────────────
# Band optimization
# ─────────────────────────────────────────────────────────────────────────────

def run_band_optimization(X, y, verbose=True):
    sfreq    = CONFIG["sfreq"]
    base_acc = _cv_eval(_extract(X, y), y)["acc_mean"]

    if verbose:
        print(f"\n  Baseline (broadband + LDA svd): {base_acc*100:.2f}%")
        print(f"\n  Extracting features per band...")

    band_feats = {}
    for band_name, (lo, hi) in BANDS.items():
        try:
            Xf = _bp3d(X, lo, hi, sfreq)
            band_feats[band_name] = _extract(Xf, y)
            if verbose:
                print(f"    {band_name:<14} ({lo:.1f}-{hi:.1f}Hz)  "
                      f"shape={band_feats[band_name].shape}")
        except Exception as e:
            if verbose:
                print(f"    {band_name:<14} FAILED: {e}")

    if verbose:
        print(f"\n  Single band accuracy (lsqr+auto):")
    band_results = {}
    for band_name, Xf in band_feats.items():
        r = _cv_eval(Xf, y)
        delta  = r["acc_mean"] - base_acc
        marker = "✅" if delta > 0.005 else "❌" if delta < -0.005 else "·"
        if verbose:
            print(f"    {band_name:<14} {r['acc_mean']*100:.2f}%  "
                  f"Δ={delta*100:+.2f}%  {marker}")
        band_results[band_name] = r

    if verbose:
        print(f"\n  Multi-band + ANOVA:")
    X_multi = np.concatenate(list(band_feats.values()), axis=1)
    best_multi_acc = base_acc
    best_multi_k   = None
    for k in MULTIBAND_K_VALUES:
        if k > X_multi.shape[1]:
            break
        r = _cv_eval(X_multi, y, k=k)
        delta  = r["acc_mean"] - base_acc
        marker = "✅" if delta > 0.005 else "❌" if delta < -0.005 else "·"
        if verbose:
            print(f"    ANOVA K={k:<4}  {r['acc_mean']*100:.2f}%  "
                  f"Δ={delta*100:+.2f}%  {marker}")
        if r["acc_mean"] > best_multi_acc:
            best_multi_acc = r["acc_mean"]
            best_multi_k   = k

    best_band_acc  = base_acc
    best_band_name = None
    best_band_k    = None
    for band_name, Xf in band_feats.items():
        for k in SINGLE_BAND_K_VALUES:
            if k > Xf.shape[1]:
                break
            r = _cv_eval(Xf, y, k=k)
            if r["acc_mean"] > best_band_acc:
                best_band_acc  = r["acc_mean"]
                best_band_name = band_name
                best_band_k    = k

    overall_best  = max(base_acc, best_multi_acc, best_band_acc)
    delta_overall = overall_best - base_acc
    marker = "✅" if delta_overall > 0.005 else "❌" if delta_overall < -0.005 else "·"

    if verbose:
        print(f"\n  {'─'*55}")
        print(f"  Baseline              : {base_acc*100:.2f}%")
        print(f"  Best single band      : "
              f"{max(r['acc_mean'] for r in band_results.values())*100:.2f}%  "
              f"({max(band_results, key=lambda b: band_results[b]['acc_mean'])})")
        print(f"  Best multi+ANOVA      : {best_multi_acc*100:.2f}%  (K={best_multi_k})")
        print(f"  Best band+ANOVA       : {best_band_acc*100:.2f}%  "
              f"({best_band_name} K={best_band_k})")
        print(f"  Overall best          : {overall_best*100:.2f}%  "
              f"Δ={delta_overall*100:+.2f}%  {marker}")
        print(f"  {'─'*55}")

    return {
        "base_acc":        base_acc,
        "best_multi_acc":  best_multi_acc,
        "best_multi_k":    best_multi_k,
        "best_band_acc":   best_band_acc,
        "best_band_name":  best_band_name,
        "best_band_k":     best_band_k,
        "overall_best":    overall_best,
        "delta":           delta_overall,
        "band_results":    {k: v["acc_mean"] for k, v in band_results.items()},
        "band_feats":      band_feats,
        "X_multi":         X_multi,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-letter analysis
# ─────────────────────────────────────────────────────────────────────────────

def per_letter_analysis(X, y, results):
    """Run CV on the winning config and show per-letter accuracy + confusions."""

    # Pick the winning feature matrix + K
    if results["best_multi_acc"] >= results["best_band_acc"] and results["best_multi_k"]:
        X_best = results["X_multi"]
        best_k = results["best_multi_k"]
        config_str = f"multiband K={best_k}"
    elif results["best_band_name"]:
        X_best = results["band_feats"][results["best_band_name"]]
        best_k = results["best_band_k"]
        config_str = f"{results['best_band_name']} K={best_k}"
    else:
        X_best = results["band_feats"]["broadband"]
        best_k = None
        config_str = "broadband"

    print(f"\n  Running per-letter CV on winning config: {config_str}")
    all_true, all_pred = _cv_collect(X_best, y, k=best_k)

    if len(all_true) == 0:
        print("  Not enough data for per-letter analysis")
        return

    classes = np.unique(y)

    print("\n" + "═"*65)
    print("  PER-LETTER ACCURACY")
    print("═"*65)
    print(f"  {'L#':<6} {'Arabic':<8} {'Correct':<10} {'Total':<8} {'Acc':>7}   Status")
    print(f"  {'─'*58}")

    letter_accs = {}
    for cls in classes:
        mask    = all_true == cls
        total   = int(mask.sum())
        correct = int((all_pred[mask] == cls).sum())
        acc     = correct / total if total > 0 else 0
        ar      = ARABIC_LETTERS[cls] if cls < len(ARABIC_LETTERS) else f"L{cls}"
        status  = "✅" if acc >= 0.7 else "⚠️ " if acc >= 0.4 else "❌"
        letter_accs[cls] = acc
        print(f"  L{cls+1:02d}   {ar:<8} {correct}/{total:<8}  {acc*100:5.1f}%    {status}")

    print(f"\n{'═'*65}")
    print("  MISCLASSIFICATION DETAILS")
    print("  (which letter did the model confuse each one with?)")
    print("═"*65)

    any_errors = False
    for cls in classes:
        mask = (all_true == cls) & (all_pred != cls)
        if mask.sum() == 0:
            continue
        any_errors = True
        ar = ARABIC_LETTERS[cls] if cls < len(ARABIC_LETTERS) else f"L{cls}"
        confused = Counter(all_pred[mask].tolist())
        confused_str = "  ".join(
            f"{ARABIC_LETTERS[c] if c < len(ARABIC_LETTERS) else f'L{c}'} ({n}×)"
            for c, n in confused.most_common(4)
        )
        total_wrong = mask.sum()
        print(f"  {ar} (L{cls+1:02d})  →  {confused_str}   [{total_wrong} errors]")

    if not any_errors:
        print("  No misclassifications! Perfect CV accuracy.")

    # Summary stats
    best_cls  = max(letter_accs, key=letter_accs.get)
    worst_cls = min(letter_accs, key=letter_accs.get)
    n_perfect = sum(1 for a in letter_accs.values() if a == 1.0)
    n_zero    = sum(1 for a in letter_accs.values() if a == 0.0)
    overall   = (all_true == all_pred).mean()

    print(f"\n{'═'*65}")
    print("  SUMMARY")
    print("═"*65)
    print(f"  Overall CV accuracy  : {overall*100:.2f}%")
    print(f"  Easiest letter       : {ARABIC_LETTERS[best_cls]} (L{best_cls+1:02d}) "
          f"@ {letter_accs[best_cls]*100:.0f}%")
    print(f"  Hardest letter       : {ARABIC_LETTERS[worst_cls]} (L{worst_cls+1:02d}) "
          f"@ {letter_accs[worst_cls]*100:.0f}%")
    if n_perfect:
        print(f"  Perfect letters      : {n_perfect}")
    if n_zero:
        print(f"  Zero accuracy        : {n_zero} letters")


# ─────────────────────────────────────────────────────────────────────────────
# Preview
# ─────────────────────────────────────────────────────────────────────────────

def preview_files(data_root):
    print(f"\n  Scanning: {data_root}")
    total = 0
    for letter_id in range(1, 29):
        l_folder = data_root / f"L{letter_id:02d}"
        if not l_folder.exists():
            continue
        edfs = sorted(l_folder.glob("*.edf"))
        n     = len(edfs)
        total += n
        status = "✅" if n == 10 else f"⚠️  {n}/10"
        print(f"    L{letter_id:02d}  {status}")
    print(f"\n  Total EDF files: {total}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--preview",   action="store_true")
    parser.add_argument("--letters",   type=int, default=28)
    parser.add_argument("--trials",    type=int, default=10)
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT

    print("\n" + "═"*65)
    print("  NEW SUBJECT — EDF PIPELINE")
    print(f"  Data: {data_root}")
    print("═"*65)

    if not data_root.exists():
        print(f"\n  ERROR: {data_root} not found")
        return

    if args.preview:
        preview_files(data_root)
        return

    print("\n  [1/4] Loading EDF files...")
    records = load_subject_edf(data_root, n_letters=args.letters,
                               n_trials=args.trials, verbose=True)
    if not records:
        print("  No trials loaded.")
        return

    print("\n  [2/4] Preprocessing...")
    X, y = preprocess_edf_records(records, verbose=True)
    n_classes = len(np.unique(y))
    print(f"  Shape: X={X.shape}  y={y.shape}")
    print(f"  Classes: {n_classes} letters")
    print(f"  Trials per class (min/max): "
          f"{np.bincount(y).min()} / {np.bincount(y).max()}")

    if n_classes < 2:
        print(f"\n  WARNING: Only {n_classes} class — need at least 2.")
        return

    print("\n  [3/4] Running band optimization...")
    results = run_band_optimization(X, y, verbose=True)

    print("\n  [4/4] Per-letter analysis...")
    per_letter_analysis(X, y, results)

    print(f"\n  Band breakdown:")
    for band, acc in sorted(results["band_results"].items(),
                            key=lambda x: x[1], reverse=True):
        bar = "█" * int(acc * 30)
        print(f"    {band:<14} {acc*100:5.1f}%  {bar}")

    print(f"\n  Context vs Alazrai dataset:")
    print(f"    Dataset mean : 77.84%")
    print(f"    Dataset best : 95.10% (S12)")
    print(f"    Dataset worst: 58.33% (S15)")
    print(f"    This subject : {results['overall_best']*100:.2f}%")
    print("═"*65)


if __name__ == "__main__":
    main()