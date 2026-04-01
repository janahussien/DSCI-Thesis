# final_model.py
# ══════════════════════════════════════════════════════
# FINAL VALIDATED PIPELINE — do not change without versioning
# Validated: Riem + BP + PLV + CSP(adaptive) + LDA
# Mean CSP Δ: +1.62% | Helped: 18/27 subjects
# ══════════════════════════════════════════════════════

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from config import CONFIG
from features import (
    riemannian_features,
    band_power_features,
    connectivity_features,
    adaptive_csp_features,
)


PIPELINE_NAME = "Riem+BP+PLV+CSP(adaptive)"
CLASSIFIER_NAME = "LDA"


def extract_final_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Extract the validated final feature set. Always call this."""
    X_riem = np.nan_to_num(riemannian_features(X))
    X_bp   = np.nan_to_num(band_power_features(X))
    X_plv  = np.nan_to_num(connectivity_features(X))
    X_csp  = np.nan_to_num(adaptive_csp_features(X, y))
    return np.concatenate([X_riem, X_bp, X_plv, X_csp], axis=1)


def get_final_classifier():
    """Return a fresh instance of the final classifier."""
    return LinearDiscriminantAnalysis(solver="svd")


def run_final_pipeline(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Full 10-fold CV with the final pipeline.
    Returns acc_mean, acc_std, f1_macro, per_fold_accs.
    """
    X_feat = extract_final_features(X, y)

    skf = StratifiedKFold(
        n_splits=CONFIG["cv_folds"], shuffle=True,
        random_state=CONFIG["random_state"]
    )
    accs, f1s = [], []

    for tr_idx, te_idx in skf.split(X_feat, y):
        X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = get_final_classifier()
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))

    return {
        "pipeline":     PIPELINE_NAME,
        "classifier":   CLASSIFIER_NAME,
        "acc_mean":     float(np.mean(accs)),
        "acc_std":      float(np.std(accs)),
        "f1_macro":     float(np.mean(f1s)),
        "per_fold_accs": accs,
    }