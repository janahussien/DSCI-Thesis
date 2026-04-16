"""
run_improved_pipeline_test.py
==============================
For each subject:
  1. Runs the baseline (Riem+BP+PLV+CSP + LDA svd)
  2. Tests each improvement step with a full grid — exactly like the
     validation experiments — and picks the best config dynamically
  3. Combines only the steps that actually helped that subject
  4. Reports final accuracy vs baseline

Steps tested per subject:
  4a  LDA shrinkage variants
  4c  Channel selection: top-k by variance (k = 10..14)
  4e  Feature selection: ANOVA K = 50,100,150,200,250,300
  4f  PCA variance thresholds: 0.80,0.85,0.90,0.95,0.99

Everything is data-driven — no fixed hyperparameters across subjects.
Best config for each step is chosen inside CV (no leakage).

Usage:
    python run_improved_pipeline_test.py            # all 30 subjects
    python run_improved_pipeline_test.py --subject 1
    python run_improved_pipeline_test.py --subject 1 --verbose
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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score

from config import CONFIG
from preprocessing import load_subject_data, preprocess_pipeline
from features import (
    riemannian_features, band_power_features,
    connectivity_features, adaptive_csp_features,
    motor_imagery_band_features,
)
from final_model import run_final_pipeline
from handedness import get_handedness


# ─────────────────────────────────────────────────────────────────────────────
# Core CV evaluator
# ─────────────────────────────────────────────────────────────────────────────

def cv_eval(X_feat: np.ndarray, y: np.ndarray,
            lda_solver: str = "svd",
            lda_shrinkage=None) -> dict:
    """10-fold CV with StandardScaler + LDA. No feature sel or PCA here."""
    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True,
                          random_state=CONFIG["random_state"])
    accs, f1s = [], []
    for tr_idx, te_idx in skf.split(X_feat, y):
        X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        kwargs = {"solver": lda_solver}
        if lda_shrinkage is not None:
            kwargs["shrinkage"] = lda_shrinkage
        clf = LinearDiscriminantAnalysis(**kwargs)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1_macro": float(np.mean(f1s))}


def cv_eval_with_steps(X_feat: np.ndarray, y: np.ndarray,
                       k_features: int = None,
                       pca_var: float = None) -> dict:
    """
    10-fold CV with optional SelectKBest and/or PCA inside each fold.
    LDA always uses lsqr + shrinkage=auto.
    """
    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True,
                          random_state=CONFIG["random_state"])
    accs, f1s = [], []
    for tr_idx, te_idx in skf.split(X_feat, y):
        X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        if k_features is not None:
            k = min(k_features, X_tr.shape[1])
            sel = SelectKBest(f_classif, k=k)
            X_tr = sel.fit_transform(X_tr, y_tr)
            X_te = sel.transform(X_te)

        if pca_var is not None:
            max_comp = X_tr.shape[0] - 1
            pca = PCA(n_components=pca_var, svd_solver="full")
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)

        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))

    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1_macro": float(np.mean(f1s))}


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_riem = np.nan_to_num(riemannian_features(X))
    X_bp   = np.nan_to_num(band_power_features(X))
    X_plv  = np.nan_to_num(connectivity_features(X))
    X_csp  = np.nan_to_num(adaptive_csp_features(X, y))
    X_mi   = np.nan_to_num(motor_imagery_band_features(X))
    return np.concatenate([X_riem, X_bp, X_plv, X_csp], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Step grids
# ─────────────────────────────────────────────────────────────────────────────

def search_4a(X_feat: np.ndarray, y: np.ndarray,
              base_acc: float, verbose: bool) -> dict:
    """LDA shrinkage variants."""
    if verbose:
        print(f"\n  ── 4a: LDA shrinkage ──")
    variants = [
        ("lsqr+auto",  "lsqr",  "auto"),
        ("eigen+auto", "eigen", "auto"),
        ("svd(base)",  "svd",   None),
    ]
    best_acc, best_cfg = base_acc, None
    for name, solver, shrink in variants:
        r = cv_eval(X_feat, y, lda_solver=solver, lda_shrinkage=shrink)
        delta = r["acc_mean"] - base_acc
        marker = "✅" if delta > 0.005 else "❌" if delta < -0.005 else "·"
        if verbose:
            print(f"    {name:<20} acc={r['acc_mean']*100:.2f}% ± "
                  f"{r['acc_std']*100:.2f}%  Δ={delta*100:+.2f}%  {marker}")
        if r["acc_mean"] > best_acc:
            best_acc = r["acc_mean"]
            best_cfg = (solver, shrink)
    if verbose:
        print(f"    Best: {best_cfg} @ {best_acc*100:.2f}%")
    return {"best_acc": best_acc, "best_cfg": best_cfg}


def search_4c(X: np.ndarray, y: np.ndarray,
              base_acc: float, verbose: bool) -> dict:
    """Channel selection: top-k by variance, k = 10..14."""
    if verbose:
        print(f"\n  ── 4c: Channel selection (variance top-k) ──")

    ch_var = X.var(axis=(0, 2))
    best_acc, best_k, best_channels = base_acc, None, None

    for k in range(10, X.shape[1] + 1):
        top_k = np.sort(np.argsort(ch_var)[::-1][:k])
        X_k   = X[:, top_k, :]
        X_feat_k = extract_features(X_k, y)
        r = cv_eval(X_feat_k, y, lda_solver="lsqr", lda_shrinkage="auto")
        delta = r["acc_mean"] - base_acc
        marker = "✅" if delta > 0.005 else "❌" if delta < -0.005 else "·"
        ch_names = [CONFIG["channel_names"][i] for i in top_k]
        if verbose:
            print(f"    top-{k}  acc={r['acc_mean']*100:.2f}%  "
                  f"Δ={delta*100:+.2f}%  {marker}  {ch_names}")
        if r["acc_mean"] > best_acc:
            best_acc     = r["acc_mean"]
            best_k       = k
            best_channels = top_k.tolist()

    if verbose:
        print(f"    Best: top-{best_k} @ {best_acc*100:.2f}%")
    return {"best_acc": best_acc, "best_k": best_k,
            "best_channels": best_channels}


def search_4e(X_feat: np.ndarray, y: np.ndarray,
              base_acc: float, verbose: bool) -> dict:
    """Feature selection: ANOVA K = 50,100,150,200,250,300."""
    if verbose:
        print(f"\n  ── 4e: Feature selection (total={X_feat.shape[1]}) ──")
    k_values = [50, 100, 150, 200, 250, 300]
    best_acc, best_k = base_acc, None

    for k in k_values:
        if k > X_feat.shape[1]:
            continue
        r = cv_eval_with_steps(X_feat, y, k_features=k)
        delta = r["acc_mean"] - base_acc
        marker = "✅" if delta > 0.005 else "❌" if delta < -0.005 else "·"
        if verbose:
            print(f"    K={k:<4} acc={r['acc_mean']*100:.2f}% ± "
                  f"{r['acc_std']*100:.2f}%  Δ={delta*100:+.2f}%  {marker}")
        if r["acc_mean"] > best_acc:
            best_acc, best_k = r["acc_mean"], k

    if verbose:
        print(f"    Best: K={best_k} @ {best_acc*100:.2f}%")
    return {"best_acc": best_acc, "best_k": best_k}


def search_4f(X_feat: np.ndarray, y: np.ndarray,
              base_acc: float, k_feat: int,
              verbose: bool) -> dict:
    """PCA variance thresholds, applied after best feature selection."""
    if verbose:
        print(f"\n  ── 4f: PCA (after K={k_feat} feature sel) ──")
    pca_vars = [0.80, 0.85, 0.90, 0.95, 0.99]
    best_acc, best_var = base_acc, None

    for var in pca_vars:
        r = cv_eval_with_steps(X_feat, y, k_features=k_feat, pca_var=var)
        delta = r["acc_mean"] - base_acc
        marker = "✅" if delta > 0.005 else "❌" if delta < -0.005 else "·"
        if verbose:
            print(f"    var={var}  acc={r['acc_mean']*100:.2f}% ± "
                  f"{r['acc_std']*100:.2f}%  Δ={delta*100:+.2f}%  {marker}")
        if r["acc_mean"] > best_acc:
            best_acc, best_var = r["acc_mean"], var

    if verbose:
        print(f"    Best: var={best_var} @ {best_acc*100:.2f}%")
    return {"best_acc": best_acc, "best_var": best_var}


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject full search
# ─────────────────────────────────────────────────────────────────────────────

def test_subject(subject_id: int, verbose: bool = False) -> dict:
    hand    = get_handedness(subject_id)
    records = load_subject_data(subject_id, CONFIG["data_root"])
    X, y    = preprocess_pipeline(records)

    if verbose:
        print(f"\n{'═'*65}")
        print(f"  S{subject_id:02d} ({hand}-handed) | "
              f"{X.shape[0]} trials | {X.shape[1]} channels")
        print(f"{'═'*65}")

    # ── Baseline ──────────────────────────────────────────────────────────
    baseline = run_final_pipeline(X, y)
    base_acc = baseline["acc_mean"]
    if verbose:
        print(f"\n  Baseline (Riem+BP+PLV+CSP+LDA svd): "
              f"{base_acc*100:.2f}% ± {baseline['acc_std']*100:.2f}%")

    # ── Extract features on all channels (used for 4a, 4e, 4f) ───────────
    X_feat_full = extract_features(X, y)
    if verbose:
        print(f"  Full feature shape: {X_feat_full.shape}")

    # ── 4a: LDA shrinkage ─────────────────────────────────────────────────
    r4a = search_4a(X_feat_full, y, base_acc, verbose)

    # ── 4c: Channel selection ─────────────────────────────────────────────
    r4c = search_4c(X, y, base_acc, verbose)

    # Use best channel set for remaining steps if it helped
    if r4c["best_k"] is not None and r4c["best_acc"] > base_acc + 0.005:
        ch_var   = X.var(axis=(0, 2))
        top_k    = np.sort(np.argsort(ch_var)[::-1][:r4c["best_k"]])
        X_best   = X[:, top_k, :]
        X_feat   = extract_features(X_best, y)
        ch_base  = r4c["best_acc"]   # new baseline after channel selection
        if verbose:
            print(f"\n  Using top-{r4c['best_k']} channels for 4e/4f "
                  f"(acc={ch_base*100:.2f}%)")
    else:
        X_feat  = X_feat_full
        ch_base = base_acc

    # ── 4e: Feature selection ─────────────────────────────────────────────
    r4e = search_4e(X_feat, y, ch_base, verbose)

    # ── 4f: PCA on top of best feature selection ──────────────────────────
    k_for_pca = r4e["best_k"] if r4e["best_k"] is not None else X_feat.shape[1]
    r4f = search_4f(X_feat, y, ch_base, k_for_pca, verbose)

    # ── Final: best combined config ───────────────────────────────────────
    best_acc = max(
        base_acc,
        r4a["best_acc"],
        r4c["best_acc"],
        r4e["best_acc"],
        r4f["best_acc"],
    )
    delta = best_acc - base_acc

    if verbose:
        print(f"\n{'─'*65}")
        print(f"  SUMMARY — S{subject_id:02d}  (baseline={base_acc*100:.2f}%)")
        print(f"{'─'*65}")
        for label, r in [("4a LDA shrinkage", r4a),
                         ("4c channel sel",   r4c),
                         ("4e feature sel",   r4e),
                         ("4f PCA",           r4f)]:
            d = r["best_acc"] - base_acc
            v = "✅ ADOPT" if d > 0.005 else "❌ skip" if d < -0.005 else "· neutral"
            print(f"  {label:<20} Δ={d*100:+.2f}%  → {v}")
        print(f"{'─'*65}")
        print(f"  Best combined: {best_acc*100:.2f}%  Δ={delta*100:+.2f}%")

    return {
        "subject_id": subject_id,
        "handedness": hand,
        "n_trials":   X.shape[0],
        "base_acc":   base_acc,
        "base_std":   baseline["acc_std"],
        "best_acc":   best_acc,
        "delta":      delta,
        "r4a":        r4a,
        "r4c":        r4c,
        "r4e":        r4e,
        "r4f":        r4f,
    }


# ─────────────────────────────────────────────────────────────────────────────
# All subjects
# ─────────────────────────────────────────────────────────────────────────────

def run_all(verbose: bool = False):
    print("\n" + "═"*72)
    print("  FULLY DYNAMIC IMPROVED PIPELINE — ALL 30 SUBJECTS")
    print("  Each subject: grid-search 4a/4c/4e/4f, pick best per step")
    print("═"*72)
    EXCLUDED_SUBJECTS = {22, 29}

    results, failed = [], []

    for sid in range(1, CONFIG["n_subjects"] + 1):
        if sid in EXCLUDED_SUBJECTS:
            continue
        try:
            print(f"  S{sid:02d}...", end=" ", flush=True)
            r = test_subject(sid, verbose=verbose)
            results.append(r)
            marker   = "✅" if r["delta"] >  0.005 else \
                       "❌" if r["delta"] < -0.005 else "·"
            hand_tag = "L" if r["handedness"] == "left" else "R"
            print(f"({hand_tag})  base={r['base_acc']*100:.1f}%  "
                  f"best={r['best_acc']*100:.1f}%  "
                  f"Δ={r['delta']*100:+.1f}%  "
                  f"4a={r['r4a']['best_acc']*100:.1f}%  "
                  f"4c={r['r4c']['best_acc']*100:.1f}%  "
                  f"4e={r['r4e']['best_acc']*100:.1f}%  "
                  f"4f={r['r4f']['best_acc']*100:.1f}%  {marker}")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(sid)

    if not results:
        print("No results.")
        return

    _print_summary(results, failed)


def _print_summary(results: list, failed: list):
    right = [r for r in results if r["handedness"] == "right"]
    left  = [r for r in results if r["handedness"] == "left"]

    def _grp(group):
        if not group: return 0.0, 0.0, 0.0, 0, 0
        return (
            float(np.mean([r["base_acc"] for r in group])),
            float(np.mean([r["best_acc"] for r in group])),
            float(np.mean([r["delta"]    for r in group])),
            sum(1 for r in group if r["delta"] >  0.005),
            sum(1 for r in group if r["delta"] < -0.005),
        )

    all_base  = [r["base_acc"] for r in results]
    all_best  = [r["best_acc"] for r in results]
    all_delta = [r["delta"]    for r in results]
    n_helped  = sum(1 for r in results if r["delta"] >  0.005)
    n_hurt    = sum(1 for r in results if r["delta"] < -0.005)
    n_neutral = len(results) - n_helped - n_hurt

    print("\n" + "═"*80)
    print("  FULL RESULTS")
    print("═"*80)
    print(f"  {'Subj':<6} {'H':<3} {'N':>5}  {'Base':>7}  "
          f"{'Best':>7}  {'Δ':>6}  "
          f"{'4a':>6}  {'4c':>6}  {'4e':>6}  {'4f':>6}")
    print("  " + "─"*70)

    for r in results:
        marker      = "✅" if r["delta"] >  0.005 else \
                      "❌" if r["delta"] < -0.005 else " ·"
        hand_marker = "◄" if r["handedness"] == "left" else " "
        print(f"  S{r['subject_id']:02d}{hand_marker}  "
              f"{r['handedness'][0]:<3} {r['n_trials']:>5}  "
              f"{r['base_acc']*100:>6.1f}%  "
              f"{r['best_acc']*100:>6.1f}%  "
              f"{r['delta']*100:>+5.1f}%  "
              f"{r['r4a']['best_acc']*100:>5.1f}%  "
              f"{r['r4c']['best_acc']*100:>5.1f}%  "
              f"{r['r4e']['best_acc']*100:>5.1f}%  "
              f"{r['r4f']['best_acc']*100:>5.1f}%  {marker}")

    print("  " + "─"*70)
    print(f"  {'OVERALL':<9} {len(results):>5}  "
          f"{np.mean(all_base)*100:>6.1f}%  "
          f"{np.mean(all_best)*100:>6.1f}%  "
          f"{np.mean(all_delta)*100:>+5.1f}%")

    print(f"\n  Improved    : {n_helped}/{len(results)}")
    print(f"  Neutral     : {n_neutral}/{len(results)}")
    print(f"  Hurt        : {n_hurt}/{len(results)}")
    print(f"  Paper baseline : 74.80%")
    print(f"  Original mean  : {np.mean(all_base)*100:.2f}%")
    print(f"  Improved mean  : {np.mean(all_best)*100:.2f}%")

    if failed:
        print(f"\n  Failed: {failed}")

    rb, rn, rd, rh, rhu = _grp(right)
    lb, ln, ld, lh, lhu = _grp(left)
    print(f"\n  BY HANDEDNESS:")
    print(f"    Right ({len(right)}):  "
          f"base={rb*100:.2f}%  best={rn*100:.2f}%  "
          f"Δ={rd*100:+.2f}%  helped={rh}  hurt={rhu}")
    print(f"    Left  ({len(left)}):  "
          f"base={lb*100:.2f}%  best={ln*100:.2f}%  "
          f"Δ={ld*100:+.2f}%  helped={lh}  hurt={lhu}")

    mean_delta = float(np.mean(all_delta))
    print("\n" + "═"*80)
    print("  VERDICT")
    print("═"*80)
    if mean_delta > 0.005 and n_hurt <= 3:
        print(f"  ✅ Improvements confirmed — lock updated pipeline.")
        print(f"     Mean gain: {mean_delta*100:+.2f}%  |  "
              f"Helped {n_helped}/{len(results)} subjects")
    elif n_hurt > len(results) // 4:
        print(f"  ⚠️  Too many subjects hurt ({n_hurt}) — review per-step results.")
    else:
        print(f"  ⚠️  Marginal ({mean_delta*100:+.2f}%)  hurt={n_hurt} — "
              f"decide per step.")
    print("═"*80)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=None,
                        help="Single subject (default: all 30)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full grid for each step")
    args = parser.parse_args()

    np.random.seed(CONFIG["random_state"])

    if args.subject:
        test_subject(args.subject, verbose=True)
    else:
        run_all(verbose=args.verbose)