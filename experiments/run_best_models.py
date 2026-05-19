"""
run_best_models.py
==================
Runs basic LDA on each subject with the full feature set:
    Riem + BP + PLV + CSP(adaptive) + MI bands

No hyperparameter tuning. No deep learning.
S22 and S29 excluded due to data quality issues.

Run:
    python run_best_models.py              # all subjects
    python run_best_models.py --subject 1  # single subject
"""

import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from config import CONFIG
from preprocessing import load_subject_data, preprocess_pipeline
from features import (
    band_power_features, riemannian_features,
    adaptive_csp_features, connectivity_features,
    motor_imagery_band_features,
    _adaptive_n_components,
)
from handedness import get_handedness

EXCLUDED_SUBJECTS = {22, 29}


# ─────────────────────────────────────────────────────────────────────────────
# CV with basic LDA — no tuning
# ─────────────────────────────────────────────────────────────────────────────

def run_lda(X_feat: np.ndarray, y: np.ndarray) -> dict:
    """10-fold CV with basic LDA (solver=svd, no shrinkage)."""
    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True,
                          random_state=CONFIG["random_state"])
    accs, f1s = [], []
    for tr_idx, te_idx in skf.split(X_feat, y):
        X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        clf = LinearDiscriminantAnalysis(solver="svd")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs)),
        "f1_macro": float(np.mean(f1s)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def build_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Full feature set: Riem + BP + PLV + CSP(adaptive) + MI bands."""
    X_riem = np.nan_to_num(riemannian_features(X))
    X_bp   = np.nan_to_num(band_power_features(X))
    X_plv  = np.nan_to_num(connectivity_features(X))
    X_csp  = np.nan_to_num(adaptive_csp_features(X, y))
    X_mi   = np.nan_to_num(motor_imagery_band_features(X))
    return np.concatenate([X_riem, X_bp, X_plv, X_csp, X_mi], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Single subject
# ─────────────────────────────────────────────────────────────────────────────

def run_subject(subject_id: int, verbose: bool = True) -> dict:
    hand    = get_handedness(subject_id)
    records = load_subject_data(subject_id, CONFIG["data_root"])
    X, y    = preprocess_pipeline(records)
    n_comp  = _adaptive_n_components(X.shape[0])

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  S{subject_id:02d} ({hand}-handed) | "
              f"{X.shape[0]} trials | CSP n_comp={n_comp}")
        print(f"{'═'*60}")

    X_feat = build_features(X, y)
    r      = run_lda(X_feat, y)

    if verbose:
        print(f"  Riem+BP+PLV+CSP+MI   "
              f"acc={r['acc_mean']*100:.2f}% ± {r['acc_std']*100:.2f}%  "
              f"F1={r['f1_macro']*100:.2f}%")

    return {
        "subject_id": subject_id,
        "handedness": hand,
        "n_trials":   X.shape[0],
        "n_comp":     n_comp,
        "acc_mean":   r["acc_mean"],
        "acc_std":    r["acc_std"],
        "f1_macro":   r["f1_macro"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# All subjects
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    print("\n" + "═"*65)
    print("  LDA + Riem+BP+PLV+CSP+MI — ALL SUBJECTS")
    print("  S22 & S29 excluded due to data quality")
    print("═"*65)

    summary, failed = [], []

    for sid in range(1, CONFIG["n_subjects"] + 1):
        if sid in EXCLUDED_SUBJECTS:
            print(f"  S{sid:02d}... EXCLUDED")
            continue
        try:
            print(f"  S{sid:02d}...", end=" ", flush=True)
            r = run_subject(sid, verbose=False)
            summary.append(r)
            hand_tag = "L" if r["handedness"] == "left" else "R"
            print(f"({hand_tag})  acc={r['acc_mean']*100:.2f}% ± "
                  f"{r['acc_std']*100:.2f}%")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(sid)

    if not summary:
        print("No results.")
        return

    # ── Summary table ─────────────────────────────────────────────────────
    accs = [r["acc_mean"] for r in summary]
    f1s  = [r["f1_macro"] for r in summary]

    print("\n" + "═"*65)
    print("  FULL RESULTS")
    print("═"*65)
    print(f"  {'Subj':<6} {'H':<3} {'N':>5}  {'Accuracy':>10}  "
          f"{'±':>6}  {'F1':>8}")
    print("  " + "─"*52)

    for r in summary:
        hand_marker = "◄" if r["handedness"] == "left" else " "
        print(f"  S{r['subject_id']:02d}{hand_marker}  "
              f"{r['handedness'][0]:<3} {r['n_trials']:>5}  "
              f"{r['acc_mean']*100:>9.2f}%  "
              f"{r['acc_std']*100:>5.2f}%  "
              f"{r['f1_macro']*100:>7.2f}%")

    print("  " + "─"*52)
    print(f"  {'MEAN':<10} {len(summary):>5}  "
          f"{np.mean(accs)*100:>9.2f}%  "
          f"{np.std(accs)*100:>5.2f}%  "
          f"{np.mean(f1s)*100:>7.2f}%")

    print(f"\n  Subjects analysed : {len(summary)}/30")
    print(f"  Excluded          : S22, S29 (data quality)")
    print(f"  Paper baseline    : 74.80%")
    print(f"  Our mean          : {np.mean(accs)*100:.2f}%")
    print(f"  Δ vs paper        : {(np.mean(accs) - 0.748)*100:+.2f}%")

    right = [r for r in summary if r["handedness"] == "right"]
    left  = [r for r in summary if r["handedness"] == "left"]
    if right:
        print(f"\n  Right-handed ({len(right)}): "
              f"mean={np.mean([r['acc_mean'] for r in right])*100:.2f}%")
    if left:
        print(f"  Left-handed  ({len(left)}): "
              f"mean={np.mean([r['acc_mean'] for r in left])*100:.2f}%")

    if failed:
        print(f"\n  Failed: {failed}")

    print("═"*65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=None,
                        help="Single subject (default: all)")
    args = parser.parse_args()

    if args.subject:
        if args.subject in EXCLUDED_SUBJECTS:
            print(f"S{args.subject:02d} is excluded due to data quality issues.")
        else:
            run_subject(args.subject, verbose=True)
    else:
        run_all()