"""
features.py
===========
All feature extraction groups.  Each function takes
    X : np.ndarray  (n_trials, n_channels, n_samples)
and returns a 2-D array  (n_trials, n_features).

Groups implemented
------------------
  1. Statistical (time-domain)         → stat_features
  2. Time-frequency / CWT              → cwt_features
  3. Frequency-band power              → band_power_features
  4. Motor imagery narrow bands (NEW)  → motor_imagery_band_features
  5. Spatial / Hjorth                  → hjorth_features
  6. Adaptive CSP (OvR, data-driven)   → adaptive_csp_features
  7. Riemannian geometry               → riemannian_features
  8. P300 (Pz proxy amplitude)         → p300_features
  9. Connectivity (PLV, coherence)     → connectivity_features
 10. Per-block soft-vote ensemble (NEW)→ ensemble_predict

CHANGES vs original:
  - motor_imagery_band_features added (mu + beta sub-bands, more targeted
    than the general band_power for motor imagery tasks)
  - ensemble_predict added (soft-vote across feature blocks, helps subjects
    where one block is noisy and drags down the concatenated feature set)
  - Both are used automatically in the improved pipeline for low performers
"""

import numpy as np
from typing import Dict, List
from scipy.signal import welch
import warnings

from config import CONFIG

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_n_components(n_trials: int) -> int:
    for (threshold, n_comp) in CONFIG["csp_components_map"]:
        if n_trials < threshold:
            return n_comp
    return CONFIG["csp_components_map"][-1][1]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Statistical (time-domain)
# ─────────────────────────────────────────────────────────────────────────────

def stat_features(X: np.ndarray) -> np.ndarray:
    """
    Per-channel: mean, variance, std, skewness, kurtosis,
                 RMS, IQR, zero-crossing rate, peak-to-peak
    → (n_trials, n_channels × 9)
    """
    from scipy.stats import skew, kurtosis

    feats = []
    for trial in X:
        row = []
        for ch in trial:
            row += [
                ch.mean(),
                ch.var(),
                ch.std(),
                float(skew(ch)),
                float(kurtosis(ch)),
                float(np.sqrt(np.mean(ch ** 2))),
                float(np.percentile(ch, 75) - np.percentile(ch, 25)),
                float(np.mean(np.diff(np.sign(ch)) != 0)),
                float(ch.max() - ch.min()),
            ]
        feats.append(row)
    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Time-frequency (CWT)
# ─────────────────────────────────────────────────────────────────────────────

def cwt_features(X: np.ndarray) -> np.ndarray:
    """
    Continuous Wavelet Transform per channel — 12 features per channel.
    → (n_trials, n_channels × 12)
    """
    try:
        import pywt
    except ImportError:
        warnings.warn("PyWavelets not found. Run: pip install PyWavelets. Returning zeros.")
        return np.zeros((X.shape[0], X.shape[1] * 12), dtype=np.float32)

    from scipy.stats import skew as _skew, kurtosis as _kurtosis

    scales  = np.array(list(CONFIG["cwt_scales"]))
    wavelet = CONFIG["wavelet"]
    feats   = []

    for trial in X:
        row = []
        for ch in trial:
            coeffs, _ = pywt.cwt(ch, scales, wavelet)
            mag = np.abs(coeffs)

            log_amp_sum = float(np.sum(np.log1p(mag)))
            mad         = float(np.mean(np.abs(mag - mag.mean())))
            rms         = float(np.sqrt(np.mean(mag ** 2)))
            iqr         = float(np.percentile(mag, 75) - np.percentile(mag, 25))
            mean        = float(mag.mean())
            var         = float(mag.var())
            sk          = float(_skew(mag.ravel()))
            kurt        = float(_kurtosis(mag.ravel()))
            energy      = float(np.sum(mag ** 2))
            flux        = float(np.mean(np.diff(mag, axis=1) ** 2))
            p           = mag.ravel() / (mag.sum() + 1e-10)
            renyi       = float(-np.log2(np.sum(p ** 2) + 1e-10))
            flat_sorted = np.sort(mag.ravel())[::-1]
            top10_n     = max(1, len(flat_sorted) // 10)
            e_conc      = float(flat_sorted[:top10_n].sum() / (energy + 1e-10))

            row += [log_amp_sum, mad, rms, iqr, mean, var,
                    sk, kurt, energy, flux, renyi, e_conc]
        feats.append(row)

    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Frequency-band power (Welch PSD)
# ─────────────────────────────────────────────────────────────────────────────

def band_power_features(X: np.ndarray) -> np.ndarray:
    """
    Per channel per band: absolute & relative PSD power.
    Bands: delta, theta, alpha, beta, gamma.
    → (n_trials, n_channels × n_bands × 2)
    """
    sfreq = CONFIG["sfreq"]
    bands = CONFIG["freq_bands"]
    feats = []

    for trial in X:
        row = []
        for ch in trial:
            freqs, psd = welch(ch, fs=sfreq, nperseg=min(256, ch.shape[0]))
            total_power = psd.sum() + 1e-10
            for (lo, hi) in bands.values():
                mask    = (freqs >= lo) & (freqs < hi)
                abs_pow = float(psd[mask].sum())
                rel_pow = float(abs_pow / total_power)
                row += [abs_pow, rel_pow]
        feats.append(row)

    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Motor imagery narrow-band power (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def motor_imagery_band_features(X: np.ndarray) -> np.ndarray:
    """
    Narrow-band PSD features specifically tuned for motor imagery.
    Splits the mu rhythm (8-12 Hz) and beta (13-30 Hz) into sub-bands
    to capture per-subject peak frequency variation.

    Sub-bands:
        mu_low   : 8–10 Hz
        mu_high  : 10–12 Hz
        beta_low : 13–20 Hz
        beta_mid : 20–25 Hz
        beta_high: 25–30 Hz

    → (n_trials, n_channels × 5_bands × 2)   [abs + relative power]
    """
    sfreq = CONFIG["sfreq"]
    mi_bands = {
        "mu_low":    (8,  10),
        "mu_high":   (10, 12),
        "beta_low":  (13, 20),
        "beta_mid":  (20, 25),
        "beta_high": (25, 30),
    }
    feats = []

    for trial in X:
        row = []
        for ch in trial:
            freqs, psd = welch(ch, fs=sfreq, nperseg=min(256, ch.shape[0]))
            total = psd.sum() + 1e-10
            for (lo, hi) in mi_bands.values():
                mask = (freqs >= lo) & (freqs < hi)
                abs_pow = float(psd[mask].sum())
                rel_pow = float(abs_pow / total)
                row += [abs_pow, rel_pow]
        feats.append(row)

    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Spatial / Hjorth parameters
# ─────────────────────────────────────────────────────────────────────────────

def hjorth_features(X: np.ndarray) -> np.ndarray:
    """
    Hjorth Activity, Mobility, Complexity per channel.
    → (n_trials, n_channels × 3)
    """
    feats = []
    for trial in X:
        row = []
        for ch in trial:
            d1 = np.diff(ch)
            d2 = np.diff(d1)
            activity   = float(ch.var())
            mobility   = float(np.sqrt(d1.var() / (activity + 1e-10)))
            complexity = float(
                np.sqrt(d2.var() / (d1.var() + 1e-10)) / (mobility + 1e-10)
            )
            row += [activity, mobility, complexity]
        feats.append(row)
    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Adaptive CSP — One-vs-Rest, data-driven n_components
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_csp_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    One-vs-Rest CSP with automatically chosen n_components based on
    available trial count.
    → (n_trials, n_classes × n_components)
    """
    if not CONFIG.get("csp_enabled", True):
        return np.zeros((X.shape[0], 1), dtype=np.float32)

    n_trials     = X.shape[0]
    n_components = _adaptive_n_components(n_trials)

    try:
        from mne.decoding import CSP
        _use_mne = True
    except ImportError:
        _use_mne = False

    classes   = np.unique(y)
    all_feats = []

    for cls in classes:
        y_binary = (y == cls).astype(int)
        if y_binary.sum() < 5:
            continue

        if _use_mne:
            try:
                csp = CSP(n_components=n_components, log=True, norm_trace=False)
                feats_cls = csp.fit_transform(X, y_binary)
                all_feats.append(feats_cls)
                continue
            except Exception:
                pass

        all_feats.append(_numpy_csp(X, y_binary, n_components))

    if not all_feats:
        return np.zeros((X.shape[0], 1), dtype=np.float32)

    return np.concatenate(all_feats, axis=1).astype(np.float32)


def _numpy_csp(X: np.ndarray, y_binary: np.ndarray, n_comp: int) -> np.ndarray:
    """Pure-NumPy CSP fallback."""
    pos = X[y_binary == 1]
    neg = X[y_binary == 0]

    C1 = _cov_mean(pos)
    C2 = _cov_mean(neg)
    Cc = C1 + C2

    vals, vecs = np.linalg.eigh(Cc)
    W   = vecs @ np.diag(1.0 / np.sqrt(vals + 1e-10)) @ vecs.T
    S1  = W @ C1 @ W.T
    evals, evecs = np.linalg.eigh(S1)
    order   = np.argsort(evals)[::-1]
    filters = evecs[:, order[:n_comp]].T @ W

    projected = np.einsum("cd,tds->tcs", filters, X)
    log_var   = np.log(projected.var(axis=2) + 1e-10)
    return log_var


def _cov_mean(X: np.ndarray) -> np.ndarray:
    covs = np.array([np.cov(t) for t in X])
    return covs.mean(axis=0)


def csp_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Alias for adaptive_csp_features."""
    return adaptive_csp_features(X, y)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Riemannian geometry
# ─────────────────────────────────────────────────────────────────────────────

def riemannian_features(X: np.ndarray) -> np.ndarray:
    """
    Project trial covariance matrices onto the tangent space of their
    Riemannian mean.  Requires: pip install pyriemann
    → (n_trials, n_channels*(n_channels+1)/2)
    """
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace

        cov_est = Covariances(estimator="oas")
        covs    = cov_est.fit_transform(X)
        ts      = TangentSpace(metric="riemann")
        feats   = ts.fit_transform(covs)
        return feats.astype(np.float32)

    except ImportError:
        warnings.warn("pyriemann not found. Using covariance upper-tri fallback.")
        return _covariance_features(X)


def _covariance_features(X: np.ndarray) -> np.ndarray:
    """Fallback: upper-triangular of covariance matrix per trial."""
    n_trials, n_ch, _ = X.shape
    feats = []
    for t in X:
        C   = np.cov(t)
        idx = np.triu_indices(n_ch)
        feats.append(C[idx])
    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 8. P300 proxy
# ─────────────────────────────────────────────────────────────────────────────

def p300_features(X: np.ndarray) -> np.ndarray:
    """
    Mean & peak amplitude in N200 / P300 / late-positive windows.
    → (n_trials, n_windows × n_target_channels × 2)
    """
    sfreq      = CONFIG["sfreq"]
    target_ch  = [10, 11, 12, 13]
    windows    = [(0.15, 0.25), (0.25, 0.50), (0.50, 0.80)]

    feats = []
    for trial in X:
        row = []
        for (t0, t1) in windows:
            s0, s1  = int(t0 * sfreq), min(int(t1 * sfreq), trial.shape[1])
            window  = trial[target_ch, s0:s1]
            row    += list(window.mean(axis=1))
            row    += list(window.max(axis=1))
        feats.append(row)

    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 9. PLV connectivity
# ─────────────────────────────────────────────────────────────────────────────

def connectivity_features(X: np.ndarray) -> np.ndarray:
    """
    Whole-scalp PLV between all channel pairs in alpha + beta.
    → (n_trials, n_pairs × 2_bands)
    """
    from scipy.signal import hilbert, butter, filtfilt

    sfreq  = CONFIG["sfreq"]
    n_ch   = X.shape[1]
    bands  = {"alpha": (8, 13), "beta": (13, 30)}
    feats  = []

    def _bandpass(sig, lo, hi):
        nyq = sfreq / 2
        b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
        return filtfilt(b, a, sig)

    for trial in X:
        row = []
        for (lo, hi) in bands.values():
            filtered = np.array([_bandpass(ch, lo, hi) for ch in trial])
            phase    = np.angle(hilbert(filtered, axis=1))
            for i in range(n_ch):
                for j in range(i + 1, n_ch):
                    plv = float(np.abs(np.mean(np.exp(1j * (phase[i] - phase[j])))))
                    row.append(plv)
        feats.append(row)

    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Per-block soft-vote ensemble (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_predict(
    X_feat_blocks: List[np.ndarray],
    y: np.ndarray,
) -> dict:
    """
    Soft-vote ensemble: one LDA (lsqr+auto) per feature block,
    probabilities averaged across blocks before argmax.

    More robust than concatenating all features when one block is noisy
    — a bad block averages down rather than flooding the feature space.

    Parameters
    ----------
    X_feat_blocks : list of np.ndarray, each (n_trials, n_features_k)
        One array per feature group, e.g. [X_riem, X_bp, X_plv, X_csp, X_mi]
    y : np.ndarray (n_trials,)

    Returns
    -------
    dict with keys: acc_mean, acc_std, f1_macro
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score

    min_class_count = int(np.bincount(y).min())
    n_splits = min(CONFIG["cv_folds"], min_class_count)
    n_splits = max(n_splits, 2)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=CONFIG["random_state"],
    )
    accs, f1s = [], []

    for tr_idx, te_idx in skf.split(X_feat_blocks[0], y):
        y_tr, y_te = y[tr_idx], y[te_idx]
        probs = None

        for X_block in X_feat_blocks:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_block[tr_idx])
            X_te = scaler.transform(X_block[te_idx])

            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(X_tr, y_tr)
            p = clf.predict_proba(X_te)
            probs = p if probs is None else probs + p

        pred = np.argmax(probs, axis=1)
        accs.append(accuracy_score(y_te, pred))
        f1s.append(f1_score(y_te, pred, average="macro", zero_division=0))

    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs)),
        "f1_macro": float(np.mean(f1s)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Master extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(
    X: np.ndarray,
    y: np.ndarray = None,
    debug: bool   = False,
) -> Dict[str, np.ndarray]:
    """
    Run all feature extractors and return a named dict.
    CSP and Riemannian need labels y — pass y when available.
    """
    results = {}

    def _run(name, fn, *args):
        try:
            f = fn(*args)
            f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
            results[name] = f
            if debug:
                print(f"    ✓ {name:<35} {f.shape}")
        except Exception as e:
            warnings.warn(f"Feature group '{name}' failed: {e}")
            if debug:
                import traceback; traceback.print_exc()

    _run("statistical",       stat_features,               X)
    _run("cwt_time_freq",     cwt_features,                X)
    _run("band_power",        band_power_features,         X)
    _run("motor_imagery_bp",  motor_imagery_band_features, X)   # NEW
    _run("hjorth_spatial",    hjorth_features,             X)
    _run("p300_erp",          p300_features,               X)
    _run("connectivity_plv",  connectivity_features,       X)
    _run("covariance",        _covariance_features,        X)

    if y is not None:
        _run("adaptive_csp",  adaptive_csp_features, X, y)
        _run("riemannian",    riemannian_features,   X)

    all_f = np.concatenate(list(results.values()), axis=1)
    results["combined"] = all_f

    return results