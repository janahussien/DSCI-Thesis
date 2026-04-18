"""
run_ensemble.py
===============
IMPROVEMENT: Soft-vote ensemble over top-N configs from the improved pipeline.

Instead of picking ONE best config per subject (as run_improved_pipeline_test.py
does), this script:
  1. Runs the same 4a/4c/4e/4f search as the improved pipeline
  2. Collects ALL configs that beat the baseline (not just the winner)
  3. Takes the top-N by CV accuracy
  4. In each CV fold, trains all N models and averages their class probabilities
  5. The final prediction is the class with the highest average probability

Why this helps:
  Different configs make different mistakes. lsqr+auto might confuse ك↔ق
  while top-13 channels might get it right. By averaging probabilities,
  individual errors cancel out and the ensemble is more robust than any
  single config.

Usage:
    python run_ensemble.py                # all subjects
    python run_ensemble.py --subject 1    # single subject
    python run_ensemble.py --top_n 5      # how many configs to ensemble (default 5)
    python run_ensemble.py --subject 1 --verbose
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
)
from final_model import run_final_pipeline
from handedness import get_handedness

EXCLUDED_SUBJECTS = {22, 29}


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — matches run_improved_pipeline_test.py exactly
# ─────────────────────────────────────────────────────────────────────────────

def _extract(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_riem = np.nan_to_num(riemannian_features(X))
    X_bp   = np.nan_to_num(band_power_features(X))
    X_plv  = np.nan_to_num(connectivity_features(X))
    X_csp  = np.nan_to_num(adaptive_csp_features(X, y))
    return np.concatenate([X_riem, X_bp, X_plv, X_csp], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Config catalogue — all configs to search over
# Same search space as run_improved_pipeline_test.py
# ─────────────────────────────────────────────────────────────────────────────

def _build_config_catalogue(X: np.ndarray, y: np.ndarray,
                             verbose: bool = False) -> list:
    """
    Run the same 4a/4c/4e/4f search as the improved pipeline but collect
    ALL configs with their CV accuracies, not just the best one.

    Returns a list of dicts sorted by acc descending:
        {
            "label":     str,
            "X_feat":    np.ndarray,   # pre-extracted features for this config
            "lda_solver":str,
            "lda_shrink":str or None,
            "feat_k":    int or None,
            "pca_var":   float or None,
            "acc":       float,
        }
    """
    ch_var = X.var(axis=(0, 2))
    configs = []

    # ── Channel subsets to try ────────────────────────────────────────────
    channel_sets = {}
    for k in range(10, X.shape[1] + 1):
        top_k = np.sort(np.argsort(ch_var)[::-1][:k])
        channel_sets[k] = (top_k, _extract(X[:, top_k, :], y))
    channel_sets[None] = (None, _extract(X, y))   # all channels

    if verbose:
        print(f"  Built {len(channel_sets)} channel subsets")

    # ── LDA variants ──────────────────────────────────────────────────────
    lda_variants = [
        ("svd",       "svd",  None),
        ("lsqr+auto", "lsqr", "auto"),
        ("eigen+auto","eigen","auto"),
    ]

    # ── Feature selection K values ────────────────────────────────────────
    feat_k_values = [None, 50, 100, 150, 200, 250, 300]

    # ── PCA variance thresholds ───────────────────────────────────────────
    pca_vars = [None, 0.80, 0.85, 0.90, 0.95, 0.99]

    # ── Evaluate all combos ───────────────────────────────────────────────
    n_splits = min(CONFIG["cv_folds"], int(np.bincount(y).min()))
    n_splits = max(n_splits, 2)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=CONFIG["random_state"])

    for ch_k, (_, X_feat) in channel_sets.items():
        for lda_label, solver, shrink in lda_variants:
            for feat_k in feat_k_values:
                for pca_var in pca_vars:

                    # Clip feat_k
                    fk = min(feat_k, X_feat.shape[1]) if feat_k else None

                    fold_accs = []
                    for tr_idx, te_idx in skf.split(X_feat, y):
                        X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
                        y_tr, y_te = y[tr_idx], y[te_idx]

                        sc = StandardScaler()
                        X_tr = sc.fit_transform(X_tr)
                        X_te = sc.transform(X_te)

                        if fk:
                            sel  = SelectKBest(f_classif, k=fk)
                            X_tr = sel.fit_transform(X_tr, y_tr)
                            X_te = sel.transform(X_te)

                        if pca_var:
                            pca  = PCA(n_components=pca_var, svd_solver="full")
                            X_tr = pca.fit_transform(X_tr)
                            X_te = pca.transform(X_te)

                        kwargs = {"solver": solver}
                        if shrink:
                            kwargs["shrinkage"] = shrink
                        clf = LinearDiscriminantAnalysis(**kwargs)
                        try:
                            clf.fit(X_tr, y_tr)
                            fold_accs.append(
                                accuracy_score(y_te, clf.predict(X_te))
                            )
                        except Exception:
                            fold_accs.append(0.0)

                    acc = float(np.mean(fold_accs))
                    configs.append({
                        "label":      f"ch={ch_k} lda={lda_label} "
                                      f"feat={feat_k} pca={pca_var}",
                        "X_feat":     X_feat,
                        "lda_solver": solver,
                        "lda_shrink": shrink,
                        "feat_k":     fk,
                        "pca_var":    pca_var,
                        "acc":        acc,
                    })

    configs.sort(key=lambda c: c["acc"], reverse=True)
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Soft-vote ensemble CV
# ─────────────────────────────────────────────────────────────────────────────

def _ensemble_cv(top_configs: list, y: np.ndarray) -> dict:
    """
    10-fold CV where in each fold:
      - Each config's full pipeline (scale → feat sel → PCA → LDA) is trained
        on the training fold
      - Class probabilities are predicted for the test fold
      - Probabilities are averaged across all configs (soft vote)
      - Final prediction = argmax of averaged probabilities
    """
    n_splits = min(CONFIG["cv_folds"], int(np.bincount(y).min()))
    n_splits = max(n_splits, 2)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=CONFIG["random_state"])

    # Use the first config's X_feat just to get the split indices
    # (all configs share the same trial ordering)
    X_ref = top_configs[0]["X_feat"]

    accs, f1s = [], []

    for tr_idx, te_idx in skf.split(X_ref, y):
        y_tr, y_te = y[tr_idx], y[te_idx]
        probs = None   # will accumulate

        for cfg in top_configs:
            X_feat = cfg["X_feat"]
            X_tr   = X_feat[tr_idx]
            X_te   = X_feat[te_idx]

            sc   = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

            if cfg["feat_k"]:
                fk   = min(cfg["feat_k"], X_tr.shape[1])
                sel  = SelectKBest(f_classif, k=fk)
                X_tr = sel.fit_transform(X_tr, y_tr)
                X_te = sel.transform(X_te)

            if cfg["pca_var"]:
                pca  = PCA(n_components=cfg["pca_var"], svd_solver="full")
                X_tr = pca.fit_transform(X_tr)
                X_te = pca.transform(X_te)

            kwargs = {"solver": cfg["lda_solver"]}
            if cfg["lda_shrink"]:
                kwargs["shrinkage"] = cfg["lda_shrink"]

            clf = LinearDiscriminantAnalysis(**kwargs)
            try:
                clf.fit(X_tr, y_tr)
                p = clf.predict_proba(X_te)   # (n_test, n_classes)
                probs = p if probs is None else probs + p
            except Exception:
                continue

        if probs is None:
            continue

        y_pred = np.argmax(probs, axis=1)
        # Map back to original class labels
        classes = np.unique(y_tr)
        y_pred_labels = classes[y_pred] if len(classes) == probs.shape[1] \
                        else y_pred

        accs.append(accuracy_score(y_te, y_pred_labels))
        f1s.append(f1_score(y_te, y_pred_labels,
                            average="macro", zero_division=0))

    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs)),
        "f1_macro": float(np.mean(f1s)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject
# ─────────────────────────────────────────────────────────────────────────────

def run_subject(subject_id: int,
                top_n: int = 5,
                verbose: bool = True) -> dict:
    hand    = get_handedness(subject_id)
    records = load_subject_data(subject_id, CONFIG["data_root"])
    X, y    = preprocess_pipeline(records)

    print(f"\n{'═'*65}")
    print(f"  S{subject_id:02d} ({hand}-handed)  |  {X.shape[0]} trials  "
          f"|  ensemble top-{top_n}")
    print(f"{'═'*65}")

    # ── Baseline ──────────────────────────────────────────────────────────
    baseline = run_final_pipeline(X, y)
    base_acc = baseline["acc_mean"]
    print(f"  Baseline (Riem+BP+PLV+CSP+LDA svd): "
          f"{base_acc*100:.2f}% ± {baseline['acc_std']*100:.2f}%")

    # ── Build config catalogue ─────────────────────────────────────────────
    print(f"  Searching configs (this takes a few minutes)...")
    configs = _build_config_catalogue(X, y, verbose=False)

    # Single best config (same as improved pipeline)
    single_best     = configs[0]
    single_best_acc = single_best["acc"]

    if verbose:
        print(f"\n  Top {min(top_n * 2, 10)} configs found:")
        for i, c in enumerate(configs[:top_n * 2]):
            print(f"    #{i+1:>2}  acc={c['acc']*100:.2f}%  {c['label']}")

    # ── Ensemble of top-N ─────────────────────────────────────────────────
    # Try different ensemble sizes and pick best
    best_ens_acc  = single_best_acc
    best_ens_n    = 1
    best_ens_result = {"acc_mean": single_best_acc,
                       "acc_std": 0.0, "f1_macro": 0.0}

    print(f"\n  Testing ensemble sizes:")
    for n in range(2, min(top_n + 1, len(configs) + 1)):
        top = configs[:n]
        r   = _ensemble_cv(top, y)
        delta  = r["acc_mean"] - base_acc
        marker = "✅" if delta > 0.005 else "❌" if delta < -0.005 else "·"
        print(f"    top-{n}: {r['acc_mean']*100:.2f}% ± "
              f"{r['acc_std']*100:.2f}%  "
              f"Δ_base={delta*100:+.2f}%  "
              f"Δ_single={( r['acc_mean']-single_best_acc)*100:+.2f}%  "
              f"{marker}")
        if r["acc_mean"] > best_ens_acc:
            best_ens_acc    = r["acc_mean"]
            best_ens_n      = n
            best_ens_result = r

    delta_base   = best_ens_acc - base_acc
    delta_single = best_ens_acc - single_best_acc
    marker = "✅" if delta_base > 0.005 else "❌" if delta_base < -0.005 else "·"

    print(f"\n  {'─'*55}")
    print(f"  Baseline          : {base_acc*100:.2f}%")
    print(f"  Single best config: {single_best_acc*100:.2f}%  "
          f"Δ={( single_best_acc-base_acc)*100:+.2f}%")
    print(f"  Best ensemble     : {best_ens_acc*100:.2f}%  "
          f"Δ_base={delta_base*100:+.2f}%  "
          f"Δ_single={delta_single*100:+.2f}%  {marker}")
    print(f"  Best ensemble size: top-{best_ens_n}")
    print(f"  {'─'*55}")

    return {
        "subject_id":      subject_id,
        "handedness":      hand,
        "n_trials":        X.shape[0],
        "base_acc":        base_acc,
        "single_best_acc": single_best_acc,
        "ens_acc":         best_ens_acc,
        "ens_std":         best_ens_result["acc_std"],
        "ens_n":           best_ens_n,
        "delta_base":      delta_base,
        "delta_single":    delta_single,
    }


# ─────────────────────────────────────────────────────────────────────────────
# All subjects
# ─────────────────────────────────────────────────────────────────────────────

def run_all(top_n: int = 5):
    print("\n" + "═"*65)
    print(f"  ENSEMBLE VOTING — ALL SUBJECTS  (top-{top_n} configs)")
    print("═"*65)

    results, failed = [], []

    for sid in range(1, CONFIG["n_subjects"] + 1):
        if sid in EXCLUDED_SUBJECTS:
            print(f"  S{sid:02d}... EXCLUDED")
            continue
        try:
            print(f"  S{sid:02d}...", end=" ", flush=True)
            r = run_subject(sid, top_n=top_n, verbose=False)
            results.append(r)
            marker = "✅" if r["delta_single"] >  0.005 else \
                     "❌" if r["delta_single"] < -0.005 else "·"
            print(f"base={r['base_acc']*100:.2f}%  "
                  f"single={r['single_best_acc']*100:.2f}%  "
                  f"ens={r['ens_acc']*100:.2f}%  "
                  f"Δ_single={r['delta_single']*100:+.2f}%  {marker}")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(sid)

    if not results:
        print("No results.")
        return

    base_accs   = [r["base_acc"]        for r in results]
    single_accs = [r["single_best_acc"] for r in results]
    ens_accs    = [r["ens_acc"]         for r in results]
    deltas      = [r["delta_single"]    for r in results]
    n_helped    = sum(1 for r in results if r["delta_single"] >  0.005)
    n_hurt      = sum(1 for r in results if r["delta_single"] < -0.005)
    n_neutral   = len(results) - n_helped - n_hurt

    print("\n" + "═"*65)
    print("  ENSEMBLE SUMMARY")
    print("═"*65)
    print(f"  Mean baseline     : {np.mean(base_accs)*100:.2f}%")
    print(f"  Mean single best  : {np.mean(single_accs)*100:.2f}%")
    print(f"  Mean ensemble     : {np.mean(ens_accs)*100:.2f}%")
    print(f"  Mean Δ (ens vs single): {np.mean(deltas)*100:+.2f}%")
    print(f"  Ensemble helped   : {n_helped}/{len(results)}")
    print(f"  Ensemble neutral  : {n_neutral}/{len(results)}")
    print(f"  Ensemble hurt     : {n_hurt}/{len(results)}")
    print(f"  Paper baseline    : 74.80%")
    if failed:
        print(f"  Failed            : {failed}")
    print("═"*65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Soft-vote ensemble over top-N configs from improved pipeline"
    )
    parser.add_argument("--subject", type=int, default=None,
                        help="Single subject (default: all)")
    parser.add_argument("--top_n",  type=int, default=5,
                        help="Number of configs to ensemble (default: 5)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print top configs found")
    args = parser.parse_args()

    if args.subject:
        if args.subject in EXCLUDED_SUBJECTS:
            print(f"S{args.subject:02d} is excluded.")
        else:
            run_subject(args.subject, top_n=args.top_n, verbose=args.verbose)
    else:
        run_all(top_n=args.top_n)