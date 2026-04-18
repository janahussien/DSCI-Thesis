"""
run_band_optimization.py
========================
IMPROVEMENT: Per-subject frequency band optimization.

The current pipeline uses a fixed bandpass of 0.5-40Hz for all subjects.
But different subjects' visual imagery EEG signal may be concentrated in
different frequency ranges — one subject's discriminative signal might live
in alpha (8-13Hz), another's in beta (13-30Hz).

This script:
  1. Extracts features from multiple frequency sub-bands in parallel
  2. Concatenates them all into one feature matrix
  3. Lets ANOVA feature selection naturally keep the bands that are
     discriminative for THIS subject and discard the rest
  4. Compares vs the fixed broadband baseline

Sub-bands tested (on top of full broadband):
  - delta+theta : 0.5-8 Hz   (slow waves)
  - alpha       : 8-13 Hz    (visual imagery, relaxation)
  - beta        : 13-30 Hz   (active cognition)
  - gamma       : 30-40 Hz   (high-frequency processing)
  - alpha+beta  : 8-30 Hz    (combined cognitive)
  - broadband   : 0.5-40 Hz  (current default)

For each band, we extract the full feature set (Riem+BP+PLV+CSP) on
the band-filtered signal. The model then selects which band's features
are most useful per subject.

Usage:
    python run_band_optimization.py                # all subjects
    python run_band_optimization.py --subject 1    # single subject
    python run_band_optimization.py --subject 1 --verbose
"""

import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    pass

from scipy.signal import butter, filtfilt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score

from config import CONFIG
from preprocessing import load_subject_data, preprocess_pipeline
from features import (
    riemannian_features, band_power_features,
    connectivity_features, adaptive_csp_features,
)
from final_model import run_final_pipeline
from handedness import get_handedness

EXCLUDED_SUBJECTS = {22, 29}

# ─────────────────────────────────────────────────────────────────────────────
# Frequency bands to extract features from
# ─────────────────────────────────────────────────────────────────────────────

BANDS = {
    "broadband":   (0.5, 40.0),   # current default
    "delta_theta": (0.5,  8.0),
    "alpha":       (8.0, 13.0),
    "beta":        (13.0, 30.0),
    "gamma":       (30.0, 40.0),
    "alpha_beta":  (8.0,  30.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Bandpass filter
# ─────────────────────────────────────────────────────────────────────────────

def _bandpass(X: np.ndarray, lo: float, hi: float, fs: float) -> np.ndarray:
    """Apply bandpass filter to X (n_trials, n_channels, n_samples)."""
    nyq = fs / 2.0
    # Clip to valid range
    lo = max(lo, 0.1)
    hi = min(hi, nyq - 0.1)
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, X, axis=2)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Extract Riem+BP+PLV+CSP features — matches improved pipeline."""
    X_riem = np.nan_to_num(riemannian_features(X))
    X_bp   = np.nan_to_num(band_power_features(X))
    X_plv  = np.nan_to_num(connectivity_features(X))
    X_csp  = np.nan_to_num(adaptive_csp_features(X, y))
    return np.concatenate([X_riem, X_bp, X_plv, X_csp], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# CV evaluator with ANOVA feature selection
# ─────────────────────────────────────────────────────────────────────────────

def _cv_eval(X_feat: np.ndarray, y: np.ndarray,
             feat_k: int = None,
             lda_solver: str = "lsqr",
             lda_shrinkage: str = "auto") -> dict:
    """10-fold CV with optional ANOVA feature selection inside each fold."""
    n_splits = min(CONFIG["cv_folds"], int(np.bincount(y).min()))
    n_splits = max(n_splits, 2)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=CONFIG["random_state"])
    accs, f1s = [], []

    kwargs = {"solver": lda_solver}
    if lda_shrinkage:
        kwargs["shrinkage"] = lda_shrinkage

    for tr_idx, te_idx in skf.split(X_feat, y):
        X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        if feat_k is not None:
            k    = min(feat_k, X_tr.shape[1])
            sel  = SelectKBest(f_classif, k=k)
            X_tr = sel.fit_transform(X_tr, y_tr)
            X_te = sel.transform(X_te)

        clf = LinearDiscriminantAnalysis(**kwargs)
        try:
            clf.fit(X_tr, y_tr)
            accs.append(accuracy_score(y_te, clf.predict(X_te)))
            f1s.append(f1_score(y_te, clf.predict(X_te),
                                average="macro", zero_division=0))
        except Exception:
            accs.append(0.0)
            f1s.append(0.0)

    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs)),
        "f1_macro": float(np.mean(f1s)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject band optimization
# ─────────────────────────────────────────────────────────────────────────────

def run_subject(subject_id: int, verbose: bool = True) -> dict:
    hand    = get_handedness(subject_id)
    records = load_subject_data(subject_id, CONFIG["data_root"])
    X, y    = preprocess_pipeline(records)
    sfreq   = CONFIG["sfreq"]

    print(f"\n{'═'*65}")
    print(f"  S{subject_id:02d} ({hand}-handed)  |  {X.shape[0]} trials")
    print(f"{'═'*65}")

    # ── Baseline (broadband, current pipeline) ────────────────────────────
    baseline = run_final_pipeline(X, y)
    base_acc = baseline["acc_mean"]
    print(f"  Baseline (broadband 0.5-40Hz, LDA svd): "
          f"{base_acc*100:.2f}% ± {baseline['acc_std']*100:.2f}%")

    # ── Per-band feature extraction ───────────────────────────────────────
    print(f"\n  Extracting features per band...")
    band_feats = {}
    for band_name, (lo, hi) in BANDS.items():
        try:
            X_filt = _bandpass(X, lo, hi, sfreq)
            feats  = _extract(X_filt, y)
            band_feats[band_name] = feats
            print(f"    {band_name:<14} ({lo:.1f}-{hi:.1f}Hz)  "
                  f"shape={feats.shape}")
        except Exception as e:
            print(f"    {band_name:<14} FAILED: {e}")

    # ── Single band evaluation ─────────────────────────────────────────────
    print(f"\n  Single band accuracy (lsqr+auto, no feat sel):")
    band_results = {}
    for band_name, X_feat in band_feats.items():
        r = _cv_eval(X_feat, y)
        delta  = r["acc_mean"] - base_acc
        marker = "✅" if delta > 0.005 else "❌" if delta < -0.005 else "·"
        print(f"    {band_name:<14} {r['acc_mean']*100:.2f}% ± "
              f"{r['acc_std']*100:.2f}%  Δ={delta*100:+.2f}%  {marker}")
        band_results[band_name] = r

    # ── Multi-band concatenation ───────────────────────────────────────────
    # Concatenate all band features and let ANOVA select
    print(f"\n  Multi-band concatenation + ANOVA feature selection:")
    X_multiband = np.concatenate(list(band_feats.values()), axis=1)
    print(f"    Total features: {X_multiband.shape[1]}")

    best_multi_acc = base_acc
    best_multi_k   = None
    best_multi_r   = None

    for k in [100, 150, 200, 250, 300, 400, 500]:
        if k > X_multiband.shape[1]:
            break
        r = _cv_eval(X_multiband, y, feat_k=k)
        delta  = r["acc_mean"] - base_acc
        marker = "✅" if delta > 0.005 else "❌" if delta < -0.005 else "·"
        print(f"    ANOVA K={k:<4}  {r['acc_mean']*100:.2f}% ± "
              f"{r['acc_std']*100:.2f}%  Δ={delta*100:+.2f}%  {marker}")
        if r["acc_mean"] > best_multi_acc:
            best_multi_acc = r["acc_mean"]
            best_multi_k   = k
            best_multi_r   = r

    # ── Best individual band + ANOVA ──────────────────────────────────────
    print(f"\n  Best single band + ANOVA feature selection:")
    best_band_acc = base_acc
    best_band_name = None
    best_band_k    = None

    for band_name, X_feat in band_feats.items():
        for k in [100, 150, 200, 250, 300]:
            if k > X_feat.shape[1]:
                break
            r = _cv_eval(X_feat, y, feat_k=k)
            if r["acc_mean"] > best_band_acc:
                best_band_acc  = r["acc_mean"]
                best_band_name = band_name
                best_band_k    = k

    if best_band_name:
        print(f"    Best: {best_band_name} K={best_band_k} "
              f"@ {best_band_acc*100:.2f}%  "
              f"Δ={(best_band_acc-base_acc)*100:+.2f}%")
    else:
        print(f"    No single band + ANOVA beat baseline")

    # ── Summary ───────────────────────────────────────────────────────────
    overall_best = max(base_acc, best_multi_acc, best_band_acc)
    delta_overall = overall_best - base_acc
    marker = "✅" if delta_overall > 0.005 else "❌" if delta_overall < -0.005 else "·"

    print(f"\n  {'─'*55}")
    print(f"  Baseline              : {base_acc*100:.2f}%")
    print(f"  Best single band      : {max(r['acc_mean'] for r in band_results.values())*100:.2f}%  "
          f"({max(band_results, key=lambda b: band_results[b]['acc_mean'])})")
    print(f"  Best multi-band+ANOVA : {best_multi_acc*100:.2f}%  "
          f"(K={best_multi_k})")
    print(f"  Best band+ANOVA       : {best_band_acc*100:.2f}%  "
          f"({best_band_name} K={best_band_k})")
    print(f"  Overall best          : {overall_best*100:.2f}%  "
          f"Δ={delta_overall*100:+.2f}%  {marker}")
    print(f"  {'─'*55}")

    return {
        "subject_id":      subject_id,
        "handedness":      hand,
        "n_trials":        X.shape[0],
        "base_acc":        base_acc,
        "band_results":    {k: v["acc_mean"] for k, v in band_results.items()},
        "best_multi_acc":  best_multi_acc,
        "best_multi_k":    best_multi_k,
        "best_band_acc":   best_band_acc,
        "best_band_name":  best_band_name,
        "best_band_k":     best_band_k,
        "overall_best":    overall_best,
        "delta":           delta_overall,
    }


# ─────────────────────────────────────────────────────────────────────────────
# All subjects
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    print("\n" + "═"*65)
    print("  FREQUENCY BAND OPTIMIZATION — ALL SUBJECTS")
    print("═"*65)

    results, failed = [], []

    for sid in range(1, CONFIG["n_subjects"] + 1):
        if sid in EXCLUDED_SUBJECTS:
            print(f"  S{sid:02d}... EXCLUDED")
            continue
        try:
            print(f"  S{sid:02d}...", end=" ", flush=True)
            r = run_subject(sid, verbose=False)
            results.append(r)
            marker = "✅" if r["delta"] >  0.005 else \
                     "❌" if r["delta"] < -0.005 else "·"
            print(f"base={r['base_acc']*100:.2f}%  "
                  f"best={r['overall_best']*100:.2f}%  "
                  f"Δ={r['delta']*100:+.2f}%  "
                  f"({r['best_band_name']})  {marker}")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(sid)

    if not results:
        print("No results.")
        return

    base_accs = [r["base_acc"]     for r in results]
    best_accs = [r["overall_best"] for r in results]
    deltas    = [r["delta"]        for r in results]
    n_helped  = sum(1 for r in results if r["delta"] >  0.005)
    n_hurt    = sum(1 for r in results if r["delta"] < -0.005)
    n_neutral = len(results) - n_helped - n_hurt

    # Which band wins most often
    band_win_counts = {}
    for r in results:
        b = r.get("best_band_name")
        if b:
            band_win_counts[b] = band_win_counts.get(b, 0) + 1

    print("\n" + "═"*65)
    print("  BAND OPTIMIZATION SUMMARY")
    print("═"*65)
    print(f"  Mean baseline     : {np.mean(base_accs)*100:.2f}%")
    print(f"  Mean best         : {np.mean(best_accs)*100:.2f}%")
    print(f"  Mean Δ            : {np.mean(deltas)*100:+.2f}%")
    print(f"  Helped            : {n_helped}/{len(results)}")
    print(f"  Neutral           : {n_neutral}/{len(results)}")
    print(f"  Hurt              : {n_hurt}/{len(results)}")
    print(f"\n  Most winning bands:")
    for band, count in sorted(band_win_counts.items(),
                               key=lambda x: x[1], reverse=True):
        print(f"    {band:<14}: {count} subjects")
    print(f"  Paper baseline    : 74.80%")
    if failed:
        print(f"  Failed            : {failed}")
    print("═"*65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-subject frequency band optimization"
    )
    parser.add_argument("--subject", type=int, default=None,
                        help="Single subject (default: all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.subject:
        if args.subject in EXCLUDED_SUBJECTS:
            print(f"S{args.subject:02d} is excluded.")
        else:
            run_subject(args.subject, verbose=args.verbose)
    else:
        run_all()