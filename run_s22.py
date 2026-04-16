"""
run_s22.py
==========
Standalone analysis for Subject 22.

S22 has only 109 usable trials after artifact rejection, with some
classes having fewer than 10 samples — making standard 10-fold CV
impossible. This script runs the same pipeline as
run_improved_pipeline_test.py but with folds reduced to match the
smallest class size in S22's data.

Nothing in this file affects any other subject or any shared file.

Usage:
    python run_s22.py
"""

import numpy as np
import warnings
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
from handedness import get_handedness

SUBJECT_ID = 22


def get_n_splits(y):
    """Return safe fold count for this subject's class distribution."""
    min_class = int(np.bincount(y).min())
    n = min(CONFIG["cv_folds"], min_class)
    return max(n, 2)


def cv_eval(X_feat, y, lda_solver="svd", lda_shrinkage=None):
    n_splits = get_n_splits(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
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
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs))}


def cv_eval_with_steps(X_feat, y, k_features=None, pca_var=None):
    n_splits = get_n_splits(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
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
            pca = PCA(n_components=pca_var, svd_solver="full")
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs))}


def baseline_cv(X_feat, y):
    """Same as run_final_pipeline but with adaptive folds."""
    n_splits = get_n_splits(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=CONFIG["random_state"])
    accs = []
    for tr_idx, te_idx in skf.split(X_feat, y):
        X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        clf = LinearDiscriminantAnalysis(solver="svd")
        clf.fit(X_tr, y_tr)
        accs.append(accuracy_score(y_te, clf.predict(X_te)))
    return float(np.mean(accs)), float(np.std(accs))


def extract_features(X, y):
    X_riem = np.nan_to_num(riemannian_features(X))
    X_bp   = np.nan_to_num(band_power_features(X))
    X_plv  = np.nan_to_num(connectivity_features(X))
    X_csp  = np.nan_to_num(adaptive_csp_features(X, y))
    X_mi   = np.nan_to_num(motor_imagery_band_features(X))
    return np.concatenate([X_riem, X_bp, X_plv, X_csp, X_mi], axis=1)


if __name__ == "__main__":
    print(f"\n{'═'*65}")
    print(f"  S22 STANDALONE ANALYSIS (adaptive CV folds)")
    print(f"{'═'*65}")

    records = load_subject_data(SUBJECT_ID, CONFIG["data_root"])
    X, y    = preprocess_pipeline(records)
    n_splits = get_n_splits(y)

    print(f"  Trials: {X.shape[0]}  |  Channels: {X.shape[1]}")
    print(f"  Min class size: {np.bincount(y).min()}  |  Using {n_splits}-fold CV")

    # Baseline — same feature set as final_model.py
    X_riem = np.nan_to_num(riemannian_features(X))
    X_bp   = np.nan_to_num(band_power_features(X))
    X_plv  = np.nan_to_num(connectivity_features(X))
    X_csp  = np.nan_to_num(adaptive_csp_features(X, y))
    X_base = np.concatenate([X_riem, X_bp, X_plv, X_csp], axis=1)
    base_acc, base_std = baseline_cv(X_base, y)
    print(f"\n  Baseline (Riem+BP+PLV+CSP+LDA svd): {base_acc*100:.2f}% ± {base_std*100:.2f}%")

    # Improvement steps
    X_feat_full = extract_features(X, y)

    # 4a LDA shrinkage
    best_4a = base_acc
    for solver, shrink in [("lsqr","auto"), ("eigen","auto")]:
        r = cv_eval(X_feat_full, y, lda_solver=solver, lda_shrinkage=shrink)
        if r["acc_mean"] > best_4a:
            best_4a = r["acc_mean"]

    # 4c channel selection
    ch_var = X.var(axis=(0, 2))
    best_4c, best_k, X_best = base_acc, None, X
    for k in range(10, X.shape[1] + 1):
        top_k = np.sort(np.argsort(ch_var)[::-1][:k])
        X_k   = X[:, top_k, :]
        r = cv_eval(extract_features(X_k, y), y, lda_solver="lsqr", lda_shrinkage="auto")
        if r["acc_mean"] > best_4c:
            best_4c, best_k = r["acc_mean"], k
            X_best = X_k

    X_feat = extract_features(X_best, y) if best_k else X_feat_full
    ch_base = best_4c if best_k else base_acc

    # 4e feature selection
    best_4e, best_ke = ch_base, None
    for k in [50, 100, 150, 200, 250, 300]:
        if k > X_feat.shape[1]:
            continue
        r = cv_eval_with_steps(X_feat, y, k_features=k)
        if r["acc_mean"] > best_4e:
            best_4e, best_ke = r["acc_mean"], k

    # 4f PCA
    best_4f = ch_base
    k_pca = best_ke if best_ke else X_feat.shape[1]
    for var in [0.80, 0.85, 0.90, 0.95, 0.99]:
        r = cv_eval_with_steps(X_feat, y, k_features=k_pca, pca_var=var)
        if r["acc_mean"] > best_4f:
            best_4f = r["acc_mean"]

    best_acc = max(base_acc, best_4a, best_4c, best_4e, best_4f)
    delta = best_acc - base_acc

    print(f"\n{'─'*65}")
    print(f"  SUMMARY — S22")
    print(f"{'─'*65}")
    print(f"  4a LDA shrinkage   Δ={( best_4a-base_acc)*100:+.2f}%")
    print(f"  4c channel sel     Δ={( best_4c-base_acc)*100:+.2f}%")
    print(f"  4e feature sel     Δ={( best_4e-base_acc)*100:+.2f}%")
    print(f"  4f PCA             Δ={( best_4f-base_acc)*100:+.2f}%")
    print(f"{'─'*65}")
    print(f"  Baseline : {base_acc*100:.2f}%")
    print(f"  Best     : {best_acc*100:.2f}%")
    print(f"  Delta    : {delta*100:+.2f}%")
    print(f"  CV folds : {n_splits} (reduced from 10 due to small class sizes)")
    print(f"{'═'*65}\n")