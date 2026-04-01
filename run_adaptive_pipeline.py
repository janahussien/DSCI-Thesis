"""
run_adaptive_pipeline.py
========================
Answers the key research question:

    Does adaptive CSP on top of Riem + BP + PLV
    improve classification across subjects — universally?

For each subject this script:
  1. Loads & preprocesses data
  2. Selects CSP n_components automatically from trial count
  3. Runs 10-fold CV on both combos with LDA (fastest, most stable)
  4. Records whether CSP helped, was neutral, or hurt

Final output:
  - Per-subject table with base vs extended accuracy
  - Group-level verdict: add CSP to pipeline or not
  - Breakdown by handedness (informational only, not a branching condition)
  - JSON results saved to results_dir

Run:
    python run_adaptive_pipeline.py                  # all 30 subjects
    python run_adaptive_pipeline.py --subject 1      # single subject
    python run_adaptive_pipeline.py --classifier lda # lda | svm | rf (default: lda)
"""

import numpy as np
import warnings
import argparse
import json
from datetime import datetime
from pathlib import Path

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

from config import CONFIG
from preprocessing import load_subject_data, preprocess_pipeline
from features import (
    riemannian_features, band_power_features,
    connectivity_features, adaptive_csp_features,
    _adaptive_n_components,
)
from handedness import get_handedness, is_left_handed
from utils import print_banner


# ─────────────────────────────────────────────────────────────────────────────
# CV evaluator (single classifier, two feature sets)
# ─────────────────────────────────────────────────────────────────────────────

def cv_eval(X: np.ndarray, y: np.ndarray, clf_name: str = "lda") -> dict:
    """10-fold stratified CV. Returns acc_mean, acc_std, f1_macro."""
    skf = StratifiedKFold(
        n_splits=CONFIG["cv_folds"], shuffle=True,
        random_state=CONFIG["random_state"]
    )

    def _make_clf():
        if clf_name == "lda":
            return LinearDiscriminantAnalysis(solver="svd")
        elif clf_name == "svm":
            return SVC(kernel="rbf", C=10.0, gamma="scale")
        elif clf_name == "rf":
            return RandomForestClassifier(
                n_estimators=300, random_state=CONFIG["random_state"]
            )
        raise ValueError(f"Unknown classifier: {clf_name}")

    accs, f1s = [], []
    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        clf = _make_clf()
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
# Per-subject analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_subject(subject_id: int, clf_name: str = "lda", verbose: bool = True) -> dict:
    hand = get_handedness(subject_id)

    # Load & preprocess
    raw  = load_subject_data(subject_id, CONFIG["data_root"])
    X, y = preprocess_pipeline(raw, debug=False)

    n_trials     = X.shape[0]
    n_components = _adaptive_n_components(n_trials)

    if verbose:
        print(f"\n  S{subject_id:02d} ({hand[0]})  "
              f"{n_trials} trials  n_comp={n_components}")

    # Extract features
    X_riem = np.nan_to_num(riemannian_features(X))
    X_bp   = np.nan_to_num(band_power_features(X))
    X_plv  = np.nan_to_num(connectivity_features(X))
    X_csp  = np.nan_to_num(adaptive_csp_features(X, y))

    X_base     = np.concatenate([X_riem, X_bp, X_plv], axis=1)
    X_extended = np.concatenate([X_riem, X_bp, X_plv, X_csp], axis=1)

    # Evaluate
    r_base = cv_eval(X_base,     y, clf_name)
    r_ext  = cv_eval(X_extended, y, clf_name)

    csp_delta = r_ext["acc_mean"] - r_base["acc_mean"]
    verdict   = ("helped" if csp_delta >  0.005 else
                 "hurt"   if csp_delta < -0.005 else "neutral")

    if verbose:
        marker = "✅" if verdict == "helped" else "❌" if verdict == "hurt" else "·"
        print(f"    Base  (Riem+BP+PLV)      : "
              f"{r_base['acc_mean']*100:.2f}% ± {r_base['acc_std']*100:.2f}%")
        print(f"    +CSP  (n_comp={n_components})         : "
              f"{r_ext['acc_mean']*100:.2f}% ± {r_ext['acc_std']*100:.2f}%  "
              f"Δ={csp_delta*100:+.2f}%  {marker}")

    return {
        "subject_id":   subject_id,
        "handedness":   hand,
        "n_trials":     n_trials,
        "n_components": n_components,
        "base_acc":     r_base["acc_mean"],
        "base_std":     r_base["acc_std"],
        "base_f1":      r_base["f1_macro"],
        "ext_acc":      r_ext["acc_mean"],
        "ext_std":      r_ext["acc_std"],
        "ext_f1":       r_ext["f1_macro"],
        "csp_delta":    csp_delta,
        "verdict":      verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary & verdict
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list, clf_name: str):
    print("\n" + "═" * 72)
    print(f"  ADAPTIVE CSP ANALYSIS — ALL SUBJECTS  (classifier: {clf_name.upper()})")
    print("═" * 72)
    print(f"  {'Subj':<6} {'Hand':<5} {'Trials':<7} {'Comp':<5} "
          f"{'Base':>8} {'±':>6} {'+CSP':>8} {'±':>6} {'Δ':>7}  {'Verdict'}")
    print("  " + "─" * 68)

    for r in results:
        marker = "✅ helped" if r["verdict"] == "helped" \
            else "❌ hurt"   if r["verdict"] == "hurt" \
            else "·  neutral"
        print(
            f"  S{r['subject_id']:02d}   {r['handedness'][0]:<5} "
            f"{r['n_trials']:<7} {r['n_components']:<5} "
            f"{r['base_acc']*100:>7.2f}% "
            f"{r['base_std']*100:>5.2f}% "
            f"{r['ext_acc']*100:>7.2f}% "
            f"{r['ext_std']*100:>5.2f}% "
            f"{r['csp_delta']*100:>+6.2f}%  {marker}"
        )

    # ── Group stats ────────────────────────────────────────────────────────
    base_accs = [r["base_acc"]  for r in results]
    ext_accs  = [r["ext_acc"]   for r in results]
    deltas    = [r["csp_delta"] for r in results]
    n_helped  = sum(1 for r in results if r["verdict"] == "helped")
    n_hurt    = sum(1 for r in results if r["verdict"] == "hurt")
    n_neutral = sum(1 for r in results if r["verdict"] == "neutral")

    print("  " + "─" * 68)
    print(f"  Mean base accuracy    : {np.mean(base_accs)*100:.2f}%")
    print(f"  Mean +CSP accuracy    : {np.mean(ext_accs)*100:.2f}%")
    print(f"  Mean CSP Δ            : {np.mean(deltas)*100:+.2f}%")
    print(f"  CSP helped            : {n_helped}/{len(results)} subjects")
    print(f"  CSP neutral           : {n_neutral}/{len(results)} subjects")
    print(f"  CSP hurt              : {n_hurt}/{len(results)} subjects")
    print(f"  Paper baseline        : 74.80%")

    # ── Handedness breakdown (informational) ───────────────────────────────
    left_res  = [r for r in results if r["handedness"] == "left"]
    right_res = [r for r in results if r["handedness"] == "right"]

    print(f"\n  BY HANDEDNESS (informational — same pipeline used for all):")
    if right_res:
        rd = [r["csp_delta"] for r in right_res]
        print(f"    Right-handed ({len(right_res)}):  "
              f"mean Δ = {np.mean(rd)*100:+.2f}%  "
              f"helped={sum(1 for d in rd if d>0.005)}")
    if left_res:
        ld = [r["csp_delta"] for r in left_res]
        print(f"    Left-handed  ({len(left_res)}):  "
              f"mean Δ = {np.mean(ld)*100:+.2f}%  "
              f"helped={sum(1 for d in ld if d>0.005)}")

    # ── Final verdict ──────────────────────────────────────────────────────
    mean_delta = np.mean(deltas)
    print("\n" + "═" * 72)
    print("  PIPELINE RECOMMENDATION")
    print("═" * 72)
    if mean_delta > 0.005:
        print(f"  ✅ Add adaptive CSP to the universal pipeline.")
        print(f"     Average gain: {mean_delta*100:+.2f}%")
        print(f"     Recommended pipeline: Riem + BP + PLV + CSP(adaptive)")
    elif mean_delta < -0.005:
        print(f"  ❌ Do NOT add CSP — it hurts on average ({mean_delta*100:+.2f}%).")
        print(f"     Stick with: Riem + BP + PLV")
    else:
        print(f"  ⚠️  CSP is neutral on average ({mean_delta*100:+.2f}%).")
        print(f"     Optional: add it since it helps some subjects without")
        print(f"     consistently hurting others.")
        print(f"     Recommended: keep Riem + BP + PLV as default,")
        print(f"     offer CSP as an opt-in via config.")
    print("═" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_summary(results: list, clf_name: str):
    out_dir = CONFIG["results_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir) / f"adaptive_csp_{clf_name}_{ts}.json"

    def _serial(obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32,  np.int64)):    return int(obj)
        if isinstance(obj, dict):  return {k: _serial(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_serial(i) for i in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_serial(results), f, indent=2)
    print(f"\n  Results saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",    type=int,  default=None,
                        help="Single subject (default: all)")
    parser.add_argument("--classifier", type=str,  default="lda",
                        choices=["lda", "svm", "rf"],
                        help="Classifier for CV (default: lda)")
    args = parser.parse_args()

    clf = args.classifier

    if args.subject:
        print_banner(f"Adaptive CSP — Subject S{args.subject:02d}")
        r = analyse_subject(args.subject, clf_name=clf, verbose=True)
        save_summary([r], clf)
    else:
        print_banner("Adaptive CSP — All 30 Subjects")
        print(f"  Classifier: {clf.upper()}  |  "
              f"CSP thresholds: {CONFIG['csp_components_map']}\n")

        all_results = []
        for sid in range(1, CONFIG["n_subjects"] + 1):
            try:
                r = analyse_subject(sid, clf_name=clf, verbose=True)
                all_results.append(r)
            except Exception as e:
                print(f"  S{sid:02d} FAILED: {e}")

        if all_results:
            print_summary(all_results, clf)
            save_summary(all_results, clf)