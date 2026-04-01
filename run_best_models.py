"""
run_best_models.py
==================
Unified best-feature pipeline — same strategy for ALL subjects.

Feature combo (data-validated):
    Riemannian + Band Power + PLV  (baseline, proven best)
    Riemannian + Band Power + PLV + CSP(adaptive)  (extended)

PLV is whole-scalp for everyone.  Handedness analysis showed standard
PLV helped both right- and left-handed subjects; hemisphere-aware
variants offered no consistent advantage.

CSP n_components is chosen automatically from trial count (see config.py).
No per-subject or per-handedness hardcoding.

Run:
    python run_best_models.py --subject 1
    python run_best_models.py --subject 1 --skip_dl
    python run_best_models.py --subject 1 --dl_only
    python run_best_models.py --all          # all 30 subjects
"""

import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")

from config import CONFIG
from preprocessing import load_subject_data, preprocess_pipeline
from features import (
    band_power_features, riemannian_features,
    adaptive_csp_features, connectivity_features,
    _adaptive_n_components,
)
from handedness import get_handedness, is_left_handed
from models import run_classical_models, _print_result
from utils import print_banner, save_results


# ─────────────────────────────────────────────────────────────────────────────
# Unified feature pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_features(X: np.ndarray, y: np.ndarray, subject_id: int):
    """
    Extract and return both feature combos for comparison:
      A) Riem + BP + PLV            (proven baseline)
      B) Riem + BP + PLV + CSP      (extended, adaptive components)

    Returns a dict of {combo_name: X_features}.
    """
    n_trials     = X.shape[0]
    n_components = _adaptive_n_components(n_trials)
    hand         = get_handedness(subject_id)

    print(f"\n  Subject S{subject_id:02d} | {hand}-handed | "
          f"{n_trials} trials | CSP n_components = {n_components}")
    print("  Extracting features...")

    X_riem = np.nan_to_num(riemannian_features(X))
    X_bp   = np.nan_to_num(band_power_features(X))
    X_plv  = np.nan_to_num(connectivity_features(X))
    X_csp  = np.nan_to_num(adaptive_csp_features(X, y))

    print(f"    ✓ Riemannian   {X_riem.shape}")
    print(f"    ✓ Band Power   {X_bp.shape}")
    print(f"    ✓ PLV          {X_plv.shape}")
    print(f"    ✓ Adaptive CSP {X_csp.shape}  (n_comp={n_components})")

    base    = np.concatenate([X_riem, X_bp, X_plv], axis=1)
    extended = np.concatenate([X_riem, X_bp, X_plv, X_csp], axis=1)

    return {
        "Riem+BP+PLV":           base,
        "Riem+BP+PLV+CSP(auto)": extended,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single subject
# ─────────────────────────────────────────────────────────────────────────────

def run_subject(subject_id: int, skip_dl: bool = False, dl_only: bool = False):
    hand = get_handedness(subject_id)
    print_banner(f"Best Model Search — Subject S{subject_id:02d}  ({hand}-handed)")

    print("\n[1/3] Loading & preprocessing...")
    raw  = load_subject_data(subject_id, CONFIG["data_root"])
    X, y = preprocess_pipeline(raw)
    print(f"      Shape: {X.shape}  |  Classes: {len(np.unique(y))}")

    all_results = {}

    # ── Classical models ──────────────────────────────────────────────────
    if not dl_only:
        combos = build_features(X, y, subject_id)

        print("\n[2/3] Running classical models on each feature combo...")
        best_acc, best_label = 0.0, ""

        for combo_name, X_combo in combos.items():
            print(f"\n  ── {combo_name}  ({X_combo.shape[1]} dims) ──")
            results = run_classical_models(X_combo, y)
            all_results[combo_name] = results

            combo_best_acc   = max(r["acc_mean"] for r in results.values())
            combo_best_model = max(results, key=lambda k: results[k]["acc_mean"])
            print(f"  → Best: {combo_best_model} @ {combo_best_acc*100:.2f}%")

            if combo_best_acc > best_acc:
                best_acc   = combo_best_acc
                best_label = f"{combo_name}  →  {combo_best_model}"

        print(f"\n  🏆 Best: {best_label}")
        print(f"     Accuracy  : {best_acc*100:.2f}%")
        print(f"     Paper baseline: 74.80%  |  Δ = {(best_acc - 0.748)*100:+.2f}%")

        # Did CSP help?
        base_best = max(r["acc_mean"] for r in all_results["Riem+BP+PLV"].values())
        ext_best  = max(r["acc_mean"] for r in all_results["Riem+BP+PLV+CSP(auto)"].values())
        csp_delta = ext_best - base_best
        verdict   = "✅ CSP helped" if csp_delta > 0.005 \
               else "⚠️  CSP neutral" if abs(csp_delta) <= 0.005 \
               else "❌ CSP hurt"
        print(f"\n  CSP impact: {csp_delta*100:+.2f}%  →  {verdict}")

    # ── Deep learning ──────────────────────────────────────────────────────
    if not skip_dl:
        from models import run_deep_models
        print("\n[3/3] Deep learning on raw EEG...")
        dl_res = run_deep_models(X, y)
        all_results["deep_learning"] = dl_res

    save_results(all_results, subject_id)
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# All subjects
# ─────────────────────────────────────────────────────────────────────────────

def run_all(skip_dl: bool = True):
    """
    Run the unified pipeline across all 30 subjects and print a summary table.
    Deep learning is skipped by default for speed; pass skip_dl=False to include.
    """
    print_banner("Unified Pipeline — All 30 Subjects")
    summary = []

    for sid in range(1, CONFIG["n_subjects"] + 1):
        try:
            hand         = get_handedness(sid)
            raw          = load_subject_data(sid, CONFIG["data_root"])
            X, y         = preprocess_pipeline(raw)
            combos       = build_features(X, y, sid)
            n_components = _adaptive_n_components(X.shape[0])

            base_res = run_classical_models(combos["Riem+BP+PLV"], y)
            ext_res  = run_classical_models(combos["Riem+BP+PLV+CSP(auto)"], y)

            base_best = max(r["acc_mean"] for r in base_res.values())
            ext_best  = max(r["acc_mean"] for r in ext_res.values())
            csp_delta = ext_best - base_best
            best_acc  = max(base_best, ext_best)

            summary.append({
                "sid":        sid,
                "hand":       hand,
                "n_trials":   X.shape[0],
                "n_comp":     n_components,
                "base_acc":   base_best,
                "ext_acc":    ext_best,
                "csp_delta":  csp_delta,
                "best_acc":   best_acc,
            })
            print(f"  S{sid:02d} ({hand[0]})  base={base_best*100:.1f}%  "
                  f"+CSP={ext_best*100:.1f}%  Δ={csp_delta*100:+.1f}%  "
                  f"n_comp={n_components}")

        except Exception as e:
            print(f"  S{sid:02d} FAILED: {e}")

    if not summary:
        return

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  SUMMARY — ALL SUBJECTS")
    print("═" * 70)
    print(f"  {'Subj':<6} {'Hand':<6} {'Trials':<8} {'n_comp':<8} "
          f"{'Base':>8} {'+CSP':>8} {'Δ':>8}")
    print("  " + "─" * 58)
    for r in summary:
        marker = " ✅" if r["csp_delta"] > 0.005 \
            else " ❌" if r["csp_delta"] < -0.005 else "  ·"
        print(f"  S{r['sid']:02d}   {r['hand'][0]:<6} {r['n_trials']:<8} "
              f"{r['n_comp']:<8} "
              f"{r['base_acc']*100:>7.1f}% "
              f"{r['ext_acc']*100:>7.1f}%"
              f"  {r['csp_delta']*100:>+6.1f}%{marker}")

    base_accs = [r["base_acc"]  for r in summary]
    ext_accs  = [r["ext_acc"]   for r in summary]
    deltas    = [r["csp_delta"] for r in summary]
    n_helped  = sum(1 for d in deltas if d >  0.005)
    n_hurt    = sum(1 for d in deltas if d < -0.005)

    print("  " + "─" * 58)
    print(f"  Mean (base)     : {np.mean(base_accs)*100:.2f}%")
    print(f"  Mean (+CSP)     : {np.mean(ext_accs)*100:.2f}%")
    print(f"  Mean CSP Δ      : {np.mean(deltas)*100:+.2f}%")
    print(f"  CSP helped      : {n_helped}/30 subjects")
    print(f"  CSP hurt        : {n_hurt}/30 subjects")
    print(f"  Paper baseline  : 74.80%")
    print("═" * 70)

    # Global verdict
    if np.mean(deltas) > 0.005:
        print("\n  ✅ Adaptive CSP is a net positive — keep it in the pipeline.")
    elif np.mean(deltas) < -0.005:
        print("\n  ❌ Adaptive CSP hurts on average — remove from pipeline.")
    else:
        print("\n  ⚠️  Adaptive CSP is neutral on average — optional addition.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",  type=int,  default=1)
    parser.add_argument("--all",      action="store_true")
    parser.add_argument("--skip_dl",  action="store_true")
    parser.add_argument("--dl_only",  action="store_true")
    args = parser.parse_args()

    if args.all:
        run_all(skip_dl=args.skip_dl)
    else:
        run_subject(args.subject, skip_dl=args.skip_dl, dl_only=args.dl_only)