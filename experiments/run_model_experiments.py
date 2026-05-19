"""
run_model_experiments.py
========================
Tests model improvements 4a–4f against the final model baseline.
Preprocessing is LOCKED (standard pipeline from Step 3).
All experiments use the same final feature set: Riem+BP+PLV+CSP(adaptive).

Each experiment runs on S01 first, validates on S03.
Exit criteria: improvement confirmed on BOTH subjects.

Usage:
    python run_model_experiments.py --subject 1
    python run_model_experiments.py --subject 1 --step 4a
    python run_model_experiments.py --validate          # S01 + S03, all steps
    python run_model_experiments.py --validate --step 4e
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from config import CONFIG
from preprocessing import load_subject_data, preprocess_pipeline
from final_model import extract_final_features, run_final_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_subject(subject_id: int):
    records = load_subject_data(subject_id, CONFIG["data_root"])
    return preprocess_pipeline(records)


def cv_eval(X_feat: np.ndarray, y: np.ndarray, clf_fn) -> dict:
    """10-fold stratified CV with per-fold StandardScaler. clf_fn() → fresh clf."""
    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True,
                          random_state=CONFIG["random_state"])
    accs, f1s = [], []
    for tr_idx, te_idx in skf.split(X_feat, y):
        X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        clf = clf_fn()
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1_macro": float(np.mean(f1s))}


def _delta(base: float, new: float) -> str:
    d = new - base
    marker = "✅" if d > 0.005 else "❌" if d < -0.005 else "·"
    return f"base={base*100:.2f}%  new={new*100:.2f}%  Δ={d*100:+.2f}%  {marker}"


# ─────────────────────────────────────────────────────────────────────────────
# 4a — LDA shrinkage (Ledoit-Wolf)
# ─────────────────────────────────────────────────────────────────────────────

def experiment_4a(X_feat: np.ndarray, y: np.ndarray, base: float) -> dict:
    print("\n  ── 4a: LDA shrinkage (Ledoit-Wolf) ──")

    variants = {
        "lsqr+auto":  lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "eigen+auto": lambda: LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"),
        "svd (base)": lambda: LinearDiscriminantAnalysis(solver="svd"),
    }
    best_acc, best_name = base, "svd (base)"
    results = {}
    for name, fn in variants.items():
        r = cv_eval(X_feat, y, fn)
        results[name] = r
        marker = "✅" if r["acc_mean"] - base > 0.005 else \
                 "❌" if r["acc_mean"] - base < -0.005 else "·"
        print(f"    {name:<20} acc={r['acc_mean']*100:.2f}% ± {r['acc_std']*100:.2f}%  "
              f"Δ={( r['acc_mean']-base)*100:+.2f}%  {marker}")
        if r["acc_mean"] > best_acc:
            best_acc, best_name = r["acc_mean"], name

    print(f"    Best: {best_name} @ {best_acc*100:.2f}%")
    return {"best_name": best_name, "best_acc": best_acc, "acc_mean": best_acc,
            "all": results}


# ─────────────────────────────────────────────────────────────────────────────
# 4b — Nested CV hyperparameter optimisation
# ─────────────────────────────────────────────────────────────────────────────

def experiment_4b(X_feat: np.ndarray, y: np.ndarray, base: float) -> dict:
    print("\n  ── 4b: Nested CV hyperparameter optimisation ──")

    outer_skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True,
                                random_state=CONFIG["random_state"])
    inner_skf = StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=CONFIG["random_state"])

    search_configs = {
        "LDA_shrink": {
            "estimator": LinearDiscriminantAnalysis(solver="lsqr"),
            "param_grid": {"shrinkage": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, "auto"]},
        },
        "LR": {
            "estimator": LogisticRegression(max_iter=1000, solver="lbfgs"),
            "param_grid": {"C": [0.01, 0.1, 1.0, 5.0, 10.0]},
        },
        "SVM_RBF": {
            "estimator": SVC(kernel="rbf", decision_function_shape="ovr"),
            "param_grid": {"C": [1.0, 5.0, 10.0], "gamma": ["scale", "auto"]},
        },
    }

    best_overall_acc, best_overall_name = base, "baseline"
    results = {}

    for clf_name, cfg in search_configs.items():
        fold_accs = []
        for tr_idx, te_idx in outer_skf.split(X_feat, y):
            X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            gs = GridSearchCV(cfg["estimator"], cfg["param_grid"],
                              cv=inner_skf, scoring="accuracy", n_jobs=-1)
            gs.fit(X_tr, y_tr)
            y_pred = gs.best_estimator_.predict(X_te)
            fold_accs.append(accuracy_score(y_te, y_pred))

        acc = float(np.mean(fold_accs))
        std = float(np.std(fold_accs))
        results[clf_name] = {"acc_mean": acc, "acc_std": std}
        marker = "✅" if acc - base > 0.005 else "❌" if acc - base < -0.005 else "·"
        print(f"    {clf_name:<15} acc={acc*100:.2f}% ± {std*100:.2f}%  "
              f"Δ={(acc-base)*100:+.2f}%  {marker}")
        if acc > best_overall_acc:
            best_overall_acc, best_overall_name = acc, clf_name

    print(f"    Best: {best_overall_name} @ {best_overall_acc*100:.2f}%")
    return {"best_name": best_overall_name, "best_acc": best_overall_acc,
            "acc_mean": best_overall_acc, "all": results}


# ─────────────────────────────────────────────────────────────────────────────
# 4c — Channel attention via RF feature importances
# ─────────────────────────────────────────────────────────────────────────────

def experiment_4c(X: np.ndarray, y: np.ndarray, base: float) -> dict:
    """
    Use Random Forest importances on the RAW EEG (per-channel variance)
    to weight channels, then re-extract features from reweighted signal.
    Tries: top 10, top 12, all 14 channels.
    """
    print("\n  ── 4c: Channel attention (RF importance) ──")

    # Compute per-channel discriminability: mean variance across trials
    ch_var = X.var(axis=(0, 2))  # (n_channels,)

    # Also use RF on simple per-channel means as a quick discriminability proxy
    X_ch_feats = X.mean(axis=2)  # (n_trials, n_channels)
    scaler_tmp = StandardScaler()
    X_ch_scaled = scaler_tmp.fit_transform(X_ch_feats)
    rf = RandomForestClassifier(n_estimators=200, random_state=CONFIG["random_state"],
                                n_jobs=-1)
    rf.fit(X_ch_scaled, y)
    importances = rf.feature_importances_  # (n_channels,)

    print(f"    Channel importances (RF):")
    ch_names = CONFIG["channel_names"]
    for i, (name, imp) in enumerate(zip(ch_names, importances)):
        bar = "█" * int(imp * 200)
        print(f"      {name:<6} {imp:.4f}  {bar}")

    best_acc, best_k = base, 14
    results = {}

    for k in [10, 11, 12, 13, 14]:
        top_k = np.argsort(importances)[::-1][:k]
        top_k = np.sort(top_k)
        X_k = X[:, top_k, :]
        X_feat_k = extract_final_features(X_k, y)
        r = cv_eval(X_feat_k, y,
                    lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        results[k] = r
        marker = "✅" if r["acc_mean"] - base > 0.005 else \
                 "❌" if r["acc_mean"] - base < -0.005 else "·"
        kept = [ch_names[i] for i in top_k]
        print(f"    top-{k} channels: acc={r['acc_mean']*100:.2f}%  "
              f"Δ={(r['acc_mean']-base)*100:+.2f}%  {marker}  {kept}")
        if r["acc_mean"] > best_acc:
            best_acc, best_k = r["acc_mean"], k

    print(f"    Best: top-{best_k} @ {best_acc*100:.2f}%")
    return {"best_k": best_k, "best_acc": best_acc, "acc_mean": best_acc,
            "all": results}


# ─────────────────────────────────────────────────────────────────────────────
# 4d — Temporal attention / windowing
# ─────────────────────────────────────────────────────────────────────────────

def experiment_4d(X: np.ndarray, y: np.ndarray, base: float) -> dict:
    """
    Divide 6s epoch into 0.5s windows, extract Riem+BP per window,
    concatenate all windows → LDA. Also reports per-window accuracy
    so you can see which time segments are most discriminative.
    """
    print("\n  ── 4d: Temporal windowing (0.5s windows) ──")

    from features import riemannian_features, band_power_features

    sfreq = CONFIG["sfreq"]
    win_samples = int(0.5 * sfreq)   # 128 samples
    n_samples = X.shape[2]
    n_windows = n_samples // win_samples

    print(f"    Epoch: {n_samples} samples → {n_windows} windows × {win_samples} samples")

    # Per-window accuracy (discriminability map)
    print(f"    Per-window accuracy:")
    window_accs = []
    for w in range(n_windows):
        s = w * win_samples
        e = s + win_samples
        X_w = X[:, :, s:e]
        try:
            F_w = np.nan_to_num(np.concatenate([
                riemannian_features(X_w),
                band_power_features(X_w),
            ], axis=1))
            r_w = cv_eval(F_w, y,
                          lambda: LinearDiscriminantAnalysis(solver="lsqr",
                                                             shrinkage="auto"))
            window_accs.append(r_w["acc_mean"])
            t_start = w * 0.5
            bar = "█" * int(r_w["acc_mean"] * 40)
            print(f"      t={t_start:.1f}–{t_start+0.5:.1f}s  "
                  f"acc={r_w['acc_mean']*100:.1f}%  {bar}")
        except Exception as ex:
            print(f"      window {w} failed: {ex}")
            window_accs.append(0.0)

    # All windows concatenated
    all_window_feats = []
    for w in range(n_windows):
        s, e = w * win_samples, (w + 1) * win_samples
        X_w = X[:, :, s:e]
        F_w = np.nan_to_num(np.concatenate([
            riemannian_features(X_w),
            band_power_features(X_w),
        ], axis=1))
        all_window_feats.append(F_w)
    X_all_windows = np.concatenate(all_window_feats, axis=1)

    r_all = cv_eval(X_all_windows, y,
                    lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    print(f"    All windows concat: acc={r_all['acc_mean']*100:.2f}%  "
          f"Δ={(r_all['acc_mean']-base)*100:+.2f}%")

    # Top-N most discriminative windows only
    best_acc, best_top_n = base, None
    results = {"all_windows": r_all}
    for top_n in [2, 3, 4, 6]:
        top_wins = np.argsort(window_accs)[::-1][:top_n]
        feats = []
        for w in sorted(top_wins):
            s, e = w * win_samples, (w + 1) * win_samples
            X_w = X[:, :, s:e]
            F_w = np.nan_to_num(np.concatenate([
                riemannian_features(X_w),
                band_power_features(X_w),
            ], axis=1))
            feats.append(F_w)
        X_top = np.concatenate(feats, axis=1)
        r_top = cv_eval(X_top, y,
                        lambda: LinearDiscriminantAnalysis(solver="lsqr",
                                                           shrinkage="auto"))
        marker = "✅" if r_top["acc_mean"] - base > 0.005 else \
                 "❌" if r_top["acc_mean"] - base < -0.005 else "·"
        print(f"    Top-{top_n} windows: acc={r_top['acc_mean']*100:.2f}%  "
              f"Δ={(r_top['acc_mean']-base)*100:+.2f}%  {marker}")
        results[f"top_{top_n}"] = r_top
        if r_top["acc_mean"] > best_acc:
            best_acc, best_top_n = r_top["acc_mean"], top_n

    print(f"    Best: top-{best_top_n} windows @ {best_acc*100:.2f}%")
    return {"best_top_n": best_top_n, "best_acc": best_acc,
            "acc_mean": best_acc, "window_accs": window_accs, "all": results}


# ─────────────────────────────────────────────────────────────────────────────
# 4e — Feature selection (ANOVA F-test + mutual information)
# ─────────────────────────────────────────────────────────────────────────────

def experiment_4e(X_feat: np.ndarray, y: np.ndarray, base: float) -> dict:
    """
    SelectKBest with ANOVA F-test and mutual information.
    Tries K = 50, 100, 150, 200, 250, 300.
    Selection is fit inside each CV fold (no leakage).
    """
    print(f"\n  ── 4e: Feature selection (total features: {X_feat.shape[1]}) ──")

    k_values = [50, 100, 150, 200, 250, 300]
    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True,
                          random_state=CONFIG["random_state"])

    best_acc, best_cfg = base, None
    results = {}

    for method_name, score_fn in [("ANOVA", f_classif), ("MutualInfo", mutual_info_classif)]:
        print(f"    {method_name}:")
        for k in k_values:
            if k > X_feat.shape[1]:
                continue
            fold_accs = []
            for tr_idx, te_idx in skf.split(X_feat, y):
                X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)

                # Feature selection fit on train only
                selector = SelectKBest(score_fn, k=k)
                X_tr_sel = selector.fit_transform(X_tr, y_tr)
                X_te_sel = selector.transform(X_te)

                clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                clf.fit(X_tr_sel, y_tr)
                fold_accs.append(accuracy_score(y_te, clf.predict(X_te_sel)))

            acc = float(np.mean(fold_accs))
            std = float(np.std(fold_accs))
            results[f"{method_name}_k{k}"] = {"acc_mean": acc, "acc_std": std}
            marker = "✅" if acc - base > 0.005 else \
                     "❌" if acc - base < -0.005 else "·"
            print(f"      K={k:<4} acc={acc*100:.2f}% ± {std*100:.2f}%  "
                  f"Δ={(acc-base)*100:+.2f}%  {marker}")
            if acc > best_acc:
                best_acc = acc
                best_cfg = f"{method_name}_K={k}"

    print(f"    Best: {best_cfg} @ {best_acc*100:.2f}%")
    return {"best_cfg": best_cfg, "best_acc": best_acc,
            "acc_mean": best_acc, "all": results}


# ─────────────────────────────────────────────────────────────────────────────
# 4f — PCA before LDA
# ─────────────────────────────────────────────────────────────────────────────

def experiment_4f(X_feat: np.ndarray, y: np.ndarray, base: float) -> dict:
    """
    PCA dimensionality reduction before LDA.
    Tries explained variance thresholds: 0.80, 0.85, 0.90, 0.95, 0.99
    and fixed component counts: 50, 100, 150, 200.
    PCA is fit inside each CV fold (no leakage).
    """
    print(f"\n  ── 4f: PCA before LDA (features: {X_feat.shape[1]}) ──")

    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True,
                          random_state=CONFIG["random_state"])

    best_acc, best_cfg = base, None
    results = {}
    
    max_comp = min(X_feat.shape[1], int(len(y) * (CONFIG["cv_folds"]-1) / CONFIG["cv_folds"])) - 1

    configs = [
    ("var_0.80", {"n_components": 0.80, "svd_solver": "full"}),
    ("var_0.85", {"n_components": 0.85, "svd_solver": "full"}),
    ("var_0.90", {"n_components": 0.90, "svd_solver": "full"}),
    ("var_0.95", {"n_components": 0.95, "svd_solver": "full"}),
    ("var_0.99", {"n_components": 0.99, "svd_solver": "full"}),
    ("n=50",     {"n_components": min(50,  max_comp)}),
    ("n=100",    {"n_components": min(100, max_comp)}),
    ("n=150",    {"n_components": min(150, max_comp)}),
    ("n=200",    {"n_components": min(200, max_comp)}),
]



    for cfg_name, pca_kwargs in configs:
        fold_accs = []
        n_comp_used = []
        for tr_idx, te_idx in skf.split(X_feat, y):
            X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            pca = PCA(**pca_kwargs)
            X_tr_pca = pca.fit_transform(X_tr)
            X_te_pca = pca.transform(X_te)
            n_comp_used.append(X_tr_pca.shape[1])

            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(X_tr_pca, y_tr)
            fold_accs.append(accuracy_score(y_te, clf.predict(X_te_pca)))

        acc = float(np.mean(fold_accs))
        std = float(np.std(fold_accs))
        avg_comp = int(np.mean(n_comp_used))
        results[cfg_name] = {"acc_mean": acc, "acc_std": std, "avg_components": avg_comp}
        marker = "✅" if acc - base > 0.005 else \
                 "❌" if acc - base < -0.005 else "·"
        print(f"    {cfg_name:<10} components≈{avg_comp:<5} "
              f"acc={acc*100:.2f}% ± {std*100:.2f}%  "
              f"Δ={(acc-base)*100:+.2f}%  {marker}")
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg_name

    print(f"    Best: {best_cfg} @ {best_acc*100:.2f}%")
    return {"best_cfg": best_cfg, "best_acc": best_acc,
            "acc_mean": best_acc, "all": results}


# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiments(subject_id: int, steps: list = None) -> dict:
    all_steps = ["4a", "4b", "4c", "4d", "4e", "4f"]
    steps = steps or all_steps

    print(f"\n{'═'*65}")
    print(f"  MODEL EXPERIMENTS — Subject S{subject_id:02d}")
    print(f"{'═'*65}")

    print("\n[1/2] Loading & preprocessing...")
    X, y = load_subject(subject_id)
    print(f"      Shape: {X.shape}  |  Classes: {len(np.unique(y))}")

    print("\n[2/2] Baseline (Riem+BP+PLV+CSP+LDA svd)...")
    baseline = run_final_pipeline(X, y)
    base_acc = baseline["acc_mean"]
    print(f"      Baseline: {base_acc*100:.2f}% ± {baseline['acc_std']*100:.2f}%")

    # Pre-extract final features (reused by 4a, 4b, 4e, 4f)
    print("\n      Extracting final features...")
    X_feat = extract_final_features(X, y)
    X_feat = np.nan_to_num(X_feat)
    print(f"      Feature shape: {X_feat.shape}")

    results = {"subject": subject_id, "baseline": baseline, "experiments": {}}

    if "4a" in steps:
        results["experiments"]["4a"] = experiment_4a(X_feat, y, base_acc)

    if "4b" in steps:
        results["experiments"]["4b"] = experiment_4b(X_feat, y, base_acc)

    if "4c" in steps:
        results["experiments"]["4c"] = experiment_4c(X, y, base_acc)

    if "4d" in steps:
        results["experiments"]["4d"] = experiment_4d(X, y, base_acc)

    if "4e" in steps:
        results["experiments"]["4e"] = experiment_4e(X_feat, y, base_acc)

    if "4f" in steps:
        results["experiments"]["4f"] = experiment_4f(X_feat, y, base_acc)

    # Summary
    print(f"\n{'─'*65}")
    print(f"  SUMMARY — S{subject_id:02d}  (baseline={base_acc*100:.2f}%)")
    print(f"{'─'*65}")
    for step, r in results["experiments"].items():
        new_acc = r.get("acc_mean", base_acc)
        delta = new_acc - base_acc
        verdict = "✅ ADOPT" if delta > 0.005 else \
                  "❌ skip"  if delta < -0.005 else "· neutral"
        print(f"  Step {step}: Δ={delta*100:+.2f}%  → {verdict}")
    print(f"{'─'*65}")

    return results


def run_validation(steps: list = None):
    steps = steps or ["4a", "4b", "4c", "4d", "4e", "4f"]

    print("\n" + "═"*65)
    print("  VALIDATION RUN — S01 + S03 (exit criteria check)")
    print("═"*65)

    r01 = run_experiments(1, steps)
    r03 = run_experiments(3, steps)

    base01 = r01["baseline"]["acc_mean"]
    base03 = r03["baseline"]["acc_mean"]

    print("\n" + "═"*65)
    print("  FINAL VERDICT — adopt into pipeline?")
    print("  (must help S01 AND S03 to pass exit criteria)")
    print("═"*65)
    print(f"  {'Step':<6} {'S01 Δ':>8}  {'S03 Δ':>8}  {'Verdict'}")
    print(f"  {'─'*52}")

    for step in steps:
        e01 = r01["experiments"].get(step, {})
        e03 = r03["experiments"].get(step, {})
        acc01 = e01.get("acc_mean", base01)
        acc03 = e03.get("acc_mean", base03)
        d01 = acc01 - base01
        d03 = acc03 - base03
        passed = d01 > 0.005 and d03 > 0.005
        partial = (d01 > 0) or (d03 > 0)
        verdict = "✅ ADOPT" if passed else \
                  "⚠️  partial" if partial else "❌ skip"
        print(f"  {step:<6} {d01*100:>+7.2f}%  {d03*100:>+7.2f}%  {verdict}")

    print("═"*65)
    print("  Lock the pipeline after confirming winners here.")
    print("═"*65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",  type=int, default=1)
    parser.add_argument("--step",     type=str, default=None,
                        help="Single step: 4a | 4b | 4c | 4d | 4e | 4f")
    parser.add_argument("--validate", action="store_true",
                        help="Run S01+S03 and check exit criteria")
    args = parser.parse_args()

    np.random.seed(CONFIG["random_state"])

    if args.validate:
        steps = [args.step] if args.step else None
        run_validation(steps)
    else:
        steps = [args.step] if args.step else None
        run_experiments(args.subject, steps)