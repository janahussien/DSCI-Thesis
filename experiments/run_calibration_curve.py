"""
run_calibration_curve.py
========================
IMPROVEMENT 3: Calibration curve — how much data does a new user need?

The core product question for a per-subject BCI is:
    "How long does a user need to sit and imagine letters before
     the system works well enough to be useful?"

Currently the pipeline assumes 10 trials × 28 letters = 280 trials
(full dataset). This script answers: what happens if you only have
3, 5, or 7 trials per letter?

For each subject and each training size (n_trials_per_class), it:
  1. Randomly subsamples n trials per class from the full dataset
  2. Runs the full pipeline (Riem+BP+PLV+CSP+MI + dynamic LDA tuning)
  3. Repeats 10 times with different random seeds (to get stable estimates)
  4. Reports mean accuracy ± std vs training size

Output:
  - Per-subject accuracy vs calibration size table
  - Group-level curve (mean ± std across subjects)
  - "Useful threshold" line: at what calibration size does mean accuracy
    cross 60% / 70% / 75% (paper baseline)?

This directly informs the product: minimum viable session length.

Usage:
    python run_calibration_curve.py                    # all subjects
    python run_calibration_curve.py --subject 1        # single subject
    python run_calibration_curve.py --max_trials 8     # limit max trials
    python run_calibration_curve.py --n_repeats 5      # faster, less stable
"""

import numpy as np
import warnings
import argparse
import json
from datetime import datetime
from pathlib import Path
warnings.filterwarnings("ignore")

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    pass

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score

from config import CONFIG
from preprocessing import load_subject_data, preprocess_pipeline
from features import (
    riemannian_features, band_power_features,
    connectivity_features, adaptive_csp_features,
    motor_imagery_band_features,
)
from handedness import get_handedness

EXCLUDED_SUBJECTS = {22, 29}

# Calibration sizes to test (trials per class)
CALIBRATION_SIZES = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Accuracy thresholds we care about for the product
USEFUL_THRESHOLDS = [0.60, 0.70, 0.748]   # 74.8% = paper baseline


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_riem = np.nan_to_num(riemannian_features(X))
    X_bp   = np.nan_to_num(band_power_features(X))
    X_plv  = np.nan_to_num(connectivity_features(X))
    X_csp  = np.nan_to_num(adaptive_csp_features(X, y))
    X_mi   = np.nan_to_num(motor_imagery_band_features(X))
    return np.concatenate([X_riem, X_bp, X_plv, X_csp, X_mi], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# CV for a given training subset
# ─────────────────────────────────────────────────────────────────────────────

def _eval_subset(X: np.ndarray, y: np.ndarray,
                 n_per_class: int,
                 seed: int) -> float:
    """
    Subsample n_per_class trials per class, extract features,
    run leave-one-out (for tiny N) or stratified k-fold CV.
    Returns mean accuracy.
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y)

    # ── Subsample ─────────────────────────────────────────────────────────
    sel_idx = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        n     = min(n_per_class, len(c_idx))
        if n == 0:
            continue
        chosen = rng.choice(c_idx, size=n, replace=False)
        sel_idx.extend(chosen.tolist())

    sel_idx = np.array(sel_idx)
    X_sub   = X[sel_idx]
    y_sub   = y[sel_idx]

    # ── Feature extraction ────────────────────────────────────────────────
    try:
        X_feat = _extract(X_sub, y_sub)
    except Exception:
        return np.nan

    # ── Choose CV strategy based on sample size ───────────────────────────
    min_class_count = int(np.bincount(y_sub).min())

    if min_class_count < 2:
        return np.nan

    n_splits = min(5, min_class_count)   # max 5-fold for speed
    n_splits = max(n_splits, 2)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=CONFIG["random_state"])
    accs = []

    for tr_idx, te_idx in skf.split(X_feat, y_sub):
        X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
        y_tr, y_te = y_sub[tr_idx], y_sub[te_idx]

        if len(np.unique(y_tr)) < 2:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Light feature selection to avoid overfitting with few trials
        n_feat = min(100, X_tr_s.shape[1], X_tr_s.shape[0] - 1)
        if n_feat > 1:
            sel    = SelectKBest(f_classif, k=n_feat)
            X_tr_s = sel.fit_transform(X_tr_s, y_tr)
            X_te_s = sel.transform(X_te_s)

        try:
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(X_tr_s, y_tr)
            accs.append(accuracy_score(y_te, clf.predict(X_te_s)))
        except Exception:
            continue

    return float(np.mean(accs)) if accs else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject calibration curve
# ─────────────────────────────────────────────────────────────────────────────

def run_subject(subject_id: int,
                max_trials: int = 10,
                n_repeats: int = 10,
                verbose: bool = True,
                X: np.ndarray = None,
                y: np.ndarray = None) -> dict:
    hand = get_handedness(subject_id)
    if X is None or y is None:
        records = load_subject_data(subject_id, CONFIG["data_root"])
        X, y    = preprocess_pipeline(records)

    # Clip calibration sizes to what's actually available
    min_avail = int(np.bincount(y).min())
    sizes     = [s for s in CALIBRATION_SIZES
                 if s <= min(max_trials, min_avail)]

    if verbose:
        print(f"\n{'═'*65}")
        print(f"  S{subject_id:02d} ({hand}-handed)  |  {X.shape[0]} trials  "
              f"|  min class={min_avail}")
        print(f"  Testing calibration sizes: {sizes}")
        print(f"{'═'*65}")
        print(f"  {'Trials/class':<14} {'Mean acc':>9} {'± Std':>8}  "
              f"{'Min':>8}  {'Max':>8}")
        print(f"  {'─'*52}")

    curve = {}
    for n in sizes:
        rep_accs = []
        for seed in range(n_repeats):
            acc = _eval_subset(X, y, n_per_class=n, seed=seed * 7 + subject_id)
            if not np.isnan(acc):
                rep_accs.append(acc)

        if rep_accs:
            mean_acc = float(np.mean(rep_accs))
            std_acc  = float(np.std(rep_accs))
        else:
            mean_acc = std_acc = np.nan

        curve[n] = {"mean": mean_acc, "std": std_acc}

        if verbose:
            if np.isnan(mean_acc):
                print(f"  {n:<14} {'N/A':>9}")
            else:
                bar = "█" * int(mean_acc * 30)
                print(f"  {n:<14} {mean_acc*100:>8.2f}%"
                      f" {std_acc*100:>7.2f}%  "
                      f"{min(rep_accs)*100:>7.2f}%  "
                      f"{max(rep_accs)*100:>7.2f}%  {bar}")

    # ── Find minimum calibration size for each useful threshold ───────────
    thresholds_hit = {}
    for thresh in USEFUL_THRESHOLDS:
        hit = None
        for n in sorted(curve.keys()):
            if not np.isnan(curve[n]["mean"]) and curve[n]["mean"] >= thresh:
                hit = n
                break
        thresholds_hit[thresh] = hit

    if verbose:
        print(f"\n  Minimum calibration to reach threshold:")
        for thresh, n in thresholds_hit.items():
            label = f"{thresh*100:.0f}%"
            if n is not None:
                minutes = n * 28 * 15 / 60   # ~15 sec per trial
                print(f"    {label}: {n} trials/class "
                      f"= {n*28} total trials "
                      f"≈ {minutes:.0f} min calibration")
            else:
                print(f"    {label}: not reached within {max_trials} trials/class")

    return {
        "subject_id":      subject_id,
        "handedness":      hand,
        "n_available":     int(min_avail),
        "curve":           curve,
        "thresholds_hit":  thresholds_hit,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Group-level summary
# ─────────────────────────────────────────────────────────────────────────────

def run_all(max_trials: int = 10, n_repeats: int = 10,
            min_class_size: int = 4):
    """
    Only include subjects where every class has >= min_class_size trials
    after preprocessing. Default=4 based on data quality analysis:
      - S03: has empty classes after artifact rejection
      - S06, S07, S09, S23: have single-trial classes
    These subjects produce unreliable near-chance results and are excluded.
    """
    print("\n" + "═"*65)
    print("  CALIBRATION CURVE — CLEAN SUBJECTS ONLY")
    print(f"  Max trials/class: {max_trials}  |  Repeats: {n_repeats}")
    print(f"  Min class size filter: >= {min_class_size} trials per class")
    print("═"*65)

    results          = []
    failed           = []
    excluded_quality = []

    for sid in range(1, CONFIG["n_subjects"] + 1):
        if sid in EXCLUDED_SUBJECTS:
            print(f"  S{sid:02d}... EXCLUDED (data corruption)")
            continue
        try:
            # Quality check first
            records = load_subject_data(sid, CONFIG["data_root"])
            X, y    = preprocess_pipeline(records)
            counts  = np.bincount(y)
            n_empty = int((counts == 0).sum())
            min_cls = int(counts.min())

            if n_empty > 0 or min_cls < min_class_size:
                reason = (f"empty_classes={n_empty}" if n_empty > 0
                          else f"min_class={min_cls}<{min_class_size}")
                print(f"  S{sid:02d}... EXCLUDED (quality: {reason})")
                excluded_quality.append(sid)
                continue

            print(f"  S{sid:02d}...", end=" ", flush=True)
            r = run_subject(sid, max_trials=max_trials,
                            n_repeats=n_repeats, verbose=False,
                            X=X, y=y)
            results.append(r)

            avail_sizes = sorted(r["curve"].keys())
            if avail_sizes:
                full_acc = r["curve"][avail_sizes[-1]]["mean"]
                min_acc  = r["curve"][avail_sizes[0]]["mean"]
                print(f"{avail_sizes[0]} trials={min_acc*100:.1f}%  "
                      f"{avail_sizes[-1]} trials={full_acc*100:.1f}%")
            else:
                print("no data")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(sid)

    if not results:
        print("No results.")
        return

    n_clean = len(results)
    print(f"\n  Clean subjects       : {n_clean}")
    print(f"  Excluded (corruption): S22, S29")
    print(f"  Excluded (quality)   : {excluded_quality}")

    # ── Group-level accuracy by calibration size ──────────────────────────
    print("\n" + "═"*65)
    print(f"  GROUP CALIBRATION CURVE  (n={n_clean} clean subjects)")
    print("═"*65)
    print(f"  {'Trials/class':<14} {'Total trials':>13} "
          f"{'≈ Min':>7} {'N subj':>7} {'Mean acc':>10} {'± Std':>8}")
    print(f"  {'─'*63}")

    all_sizes = sorted(CALIBRATION_SIZES)
    for n in all_sizes:
        group_accs = []
        for r in results:
            if n in r["curve"] and not np.isnan(r["curve"][n]["mean"]):
                group_accs.append(r["curve"][n]["mean"])
        if not group_accs:
            continue
        total_trials = n * 28
        minutes      = total_trials * 15 / 60
        bar = "█" * int(np.mean(group_accs) * 20)
        print(f"  {n:<14} {total_trials:>13}  "
              f"{minutes:>5.0f}m  "
              f"{len(group_accs):>5}  "
              f"{np.mean(group_accs)*100:>9.2f}%  "
              f"{np.std(group_accs)*100:>7.2f}%  {bar}")

    # ── Threshold summary ──────────────────────────────────────────────────
    print(f"\n  PRODUCT IMPLICATION (out of {n_clean} clean subjects):")
    print(f"  How many subjects reach each accuracy threshold?")
    print(f"\n  {'Trials/class':<14} {'≈ Time':>7}", end="")
    for t in USEFUL_THRESHOLDS:
        print(f"  >=  {t*100:.0f}%", end="")
    print()
    print(f"  {'─'*55}")

    for n in all_sizes:
        row = [r for r in results if n in r["curve"]
               and not np.isnan(r["curve"][n]["mean"])]
        if not row:
            continue
        minutes = n * 28 * 15 / 60
        print(f"  {n:<14} {minutes:>5.0f}m ", end="")
        for t in USEFUL_THRESHOLDS:
            count = sum(1 for r in row if r["curve"][n]["mean"] >= t)
            pct   = count / n_clean * 100
            print(f"  {count:>2}/{n_clean} ({pct:>3.0f}%)", end="")
        print()

    # ── Save results ──────────────────────────────────────────────────────
    out_dir = CONFIG["results_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir) / f"calibration_curve_{ts}.json"

    def _serial(obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32,  np.int64)):    return int(obj)
        if isinstance(obj, dict):   return {str(k): _serial(v) for k, v in obj.items()}
        if isinstance(obj, list):   return [_serial(i) for i in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_serial(results), f, indent=2)
    print(f"\n  Results saved → {out_path}")

    if failed:
        print(f"  Failed: {failed}")
    print("═"*65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibration curve: accuracy vs training data size"
    )
    parser.add_argument("--subject",    type=int, default=None,
                        help="Single subject (default: all)")
    parser.add_argument("--max_trials", type=int, default=10,
                        help="Max trials per class to test (default: 10)")
    parser.add_argument("--n_repeats",  type=int, default=10,
                        help="Random repeats per size for stability (default: 10)")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    if args.subject:
        if args.subject in EXCLUDED_SUBJECTS:
            print(f"S{args.subject:02d} is excluded.")
        else:
            run_subject(args.subject,
                        max_trials=args.max_trials,
                        n_repeats=args.n_repeats,
                        verbose=True)
    else:
        run_all(max_trials=args.max_trials, n_repeats=args.n_repeats)