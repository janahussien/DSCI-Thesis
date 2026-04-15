"""
run_loso.py
===========
Leave-One-Subject-Out (LOSO) Cross-Subject Analysis.

Two modes:
  1. LOSO     : Pure cross-subject — train on 29, test on 1
  2. FewShot  : Take n_shots trials per class from test subject,
                add to training pool, test on remaining trials.
                This simulates a short real-world calibration session.

Pipeline: Riem+BP+PLV+CSP + ANOVA(200) + LDA(lsqr+auto)
No alignment — raw preprocessed features used directly.

Usage:
    python run_loso.py                    # both modes, all 30 subjects
    python run_loso.py --subject 1        # test S01 only
    python run_loso.py --mode loso        # pure cross-subject only
    python run_loso.py --mode fewshot     # few-shot only
    python run_loso.py --shots 5          # 5 trials per class calibration
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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score

from config import CONFIG
from preprocessing import load_subject_data, preprocess_pipeline
from features import (
    riemannian_features, band_power_features,
    connectivity_features, adaptive_csp_features,
)
from handedness import get_handedness


ALL_SUBJECTS = list(range(1, 31))


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_riem = np.nan_to_num(riemannian_features(X))
    X_bp   = np.nan_to_num(band_power_features(X))
    X_plv  = np.nan_to_num(connectivity_features(X))
    X_csp  = np.nan_to_num(adaptive_csp_features(X, y))
    return np.concatenate([X_riem, X_bp, X_plv, X_csp], axis=1)


def align_dims(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    n_tr, n_te = X_train.shape[1], X_test.shape[1]
    if n_te < n_tr:
        X_test = np.hstack([X_test,
                            np.zeros((X_test.shape[0], n_tr - n_te))])
    elif n_te > n_tr:
        X_test = X_test[:, :n_tr]
    return X_train, X_test


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(subject_id: int):
    try:
        records = load_subject_data(subject_id, CONFIG["data_root"])
        X, y = preprocess_pipeline(records)
        if len(X) == 0:
            return None, None
        return X, y
    except Exception as e:
        print(f"  [warn] S{subject_id:02d} failed: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Core classifier
# ─────────────────────────────────────────────────────────────────────────────

def classify(X_train_feat: np.ndarray, y_train: np.ndarray,
             X_test_feat: np.ndarray, y_test: np.ndarray) -> dict:
    """Scale → feature select → LDA → evaluate."""
    X_train_feat, X_test_feat = align_dims(X_train_feat, X_test_feat)

    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_test_feat  = scaler.transform(X_test_feat)

    k = min(200, X_train_feat.shape[1])
    sel = SelectKBest(f_classif, k=k)
    X_train_feat = sel.fit_transform(X_train_feat, y_train)
    X_test_feat  = sel.transform(X_test_feat)

    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(X_train_feat, y_train)
    y_pred = clf.predict(X_test_feat)

    return {
        "acc":      float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred,
                                   average="macro", zero_division=0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: Pure LOSO
# ─────────────────────────────────────────────────────────────────────────────

def run_loso_pure(test_id: int, all_data: dict,
                  verbose: bool = False) -> dict:
    """Train on 29 subjects, test on 1 unseen subject."""
    if test_id not in all_data:
        return None

    X_test, y_test = all_data[test_id]

    X_trains, y_trains = [], []
    for sid, (X_s, y_s) in all_data.items():
        if sid == test_id:
            continue
        X_trains.append(X_s)
        y_trains.append(y_s)

    X_train = np.concatenate(X_trains, axis=0)
    y_train = np.concatenate(y_trains, axis=0)

    if verbose:
        print(f"  [LOSO] Train: {len(X_train)} trials | "
              f"Test: {len(X_test)} trials")

    try:
        X_tr_feat = extract_features(X_train, y_train)
        X_te_feat = extract_features(X_test,  y_test)
    except Exception as e:
        if verbose:
            print(f"  [!] Features failed: {e}")
        return None

    metrics = classify(X_tr_feat, y_train, X_te_feat, y_test)

    if verbose:
        print(f"  [LOSO] Accuracy: {metrics['acc']*100:.2f}%")

    return {"subject_id": test_id,
            "handedness": get_handedness(test_id),
            "n_test": len(y_test),
            **metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: Few-Shot Calibration
# ─────────────────────────────────────────────────────────────────────────────

def run_fewshot(test_id: int, all_data: dict,
                n_shots: int = 5,
                verbose: bool = False) -> dict:
    """
    Few-shot calibration:
    - Take n_shots trials per class from test subject
    - Add to training pool alongside other 29 subjects
    - Test on remaining trials of test subject

    n_shots=5 means 5 × 28 = 140 calibration trials (~12 min session).
    n_shots=3 means 3 × 28 = 84 calibration trials (~7 min session).
    """
    if test_id not in all_data:
        return None

    X_raw, y_raw = all_data[test_id]

    # Split test subject: first n_shots per class = calibration
    calib_idx, test_idx = [], []
    for cls in np.unique(y_raw):
        cls_idx = np.where(y_raw == cls)[0]
        n = min(n_shots, len(cls_idx) - 1)
        calib_idx.extend(cls_idx[:n].tolist())
        test_idx.extend(cls_idx[n:].tolist())

    X_calib = X_raw[calib_idx]
    y_calib = y_raw[calib_idx]
    X_test  = X_raw[test_idx]
    y_test  = y_raw[test_idx]

    if len(y_test) == 0 or len(np.unique(y_test)) < 2:
        return None

    # Build training pool: other 29 subjects + calibration trials
    X_trains = [X_calib]
    y_trains = [y_calib]
    for sid, (X_s, y_s) in all_data.items():
        if sid == test_id:
            continue
        X_trains.append(X_s)
        y_trains.append(y_s)

    X_train = np.concatenate(X_trains, axis=0)
    y_train = np.concatenate(y_trains, axis=0)

    if verbose:
        print(f"  [FS-{n_shots}] Calib: {len(y_calib)} | "
              f"Train total: {len(y_train)} | Test: {len(y_test)}")

    try:
        X_tr_feat = extract_features(X_train, y_train)
        X_te_feat = extract_features(X_test,  y_test)
    except Exception as e:
        if verbose:
            print(f"  [!] Features failed: {e}")
        return None

    metrics = classify(X_tr_feat, y_train, X_te_feat, y_test)

    if verbose:
        print(f"  [FS-{n_shots}] Accuracy: {metrics['acc']*100:.2f}%")

    return {"subject_id": test_id,
            "handedness": get_handedness(test_id),
            "n_shots": n_shots,
            "n_calib": len(y_calib),
            "n_test":  len(y_test),
            **metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Full run — all subjects
# ─────────────────────────────────────────────────────────────────────────────

def run_all(mode: str = "both", n_shots: int = 5,
            verbose: bool = False):

    print("\n" + "═"*72)
    print("  CROSS-SUBJECT ANALYSIS — LOSO")
    print(f"  Mode: {mode.upper()}  |  "
          f"Pipeline: Riem+BP+PLV+CSP + ANOVA(200) + LDA")
    print("═"*72)

    # Load all subjects once
    print("\n  Loading all subjects...")
    all_data = {}
    for sid in ALL_SUBJECTS:
        X, y = load_and_preprocess(sid)
        if X is not None:
            all_data[sid] = (X, y)
            print(f"  S{sid:02d} ✅  {X.shape[0]} trials")
        else:
            print(f"  S{sid:02d} ❌  skipped")

    print(f"\n  Loaded: {len(all_data)}/30 subjects\n")
    print("─"*72)

    loso_res = []
    fs_res   = []
    failed   = []

    for sid in sorted(all_data.keys()):
        hand = "L" if get_handedness(sid) == "left" else "R"
        print(f"  S{sid:02d} ({hand})...", end=" ", flush=True)
        try:
            parts = []

            if mode in ("loso", "both"):
                r = run_loso_pure(sid, all_data, verbose=verbose)
                if r:
                    loso_res.append(r)
                    parts.append(f"LOSO={r['acc']*100:.1f}%")

            if mode in ("fewshot", "both"):
                r = run_fewshot(sid, all_data,
                                n_shots=n_shots, verbose=verbose)
                if r:
                    fs_res.append(r)
                    parts.append(f"FS-{n_shots}={r['acc']*100:.1f}%")

            print("  ".join(parts))

        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(sid)

    # Summaries
    if loso_res:
        _print_summary(loso_res, failed, "PURE LOSO (no calibration)")
    if fs_res:
        _print_summary(fs_res, failed,
                       f"FEW-SHOT ({n_shots} trials/class calibration)")

    # Final comparison
    print("\n" + "═"*72)
    print("  FINAL COMPARISON")
    print("═"*72)
    print(f"  Chance level (1/28)           : 3.57%")
    if loso_res:
        lm = np.mean([r["acc"] for r in loso_res])
        print(f"  Pure LOSO                     : {lm*100:.2f}%")
    if fs_res:
        fm = np.mean([r["acc"] for r in fs_res])
        print(f"  Few-shot ({n_shots} trials/class)      : {fm*100:.2f}%")
    print(f"  Within-subject (improved)     : 73.08%")
    print(f"  Paper baseline                : 74.80%")
    print("═"*72)


def _print_summary(results: list, failed: list, label: str):
    right   = [r for r in results if r["handedness"] == "right"]
    left    = [r for r in results if r["handedness"] == "left"]
    all_acc = [r["acc"] for r in results]
    all_f1  = [r["f1_macro"] for r in results]

    print("\n" + "═"*72)
    print(f"  {label}")
    print("═"*72)
    print(f"  {'Subj':<6} {'H':<3} {'N_test':>7}  {'Acc':>7}  {'F1':>7}")
    print("  " + "─"*45)

    for r in results:
        hm = "◄" if r["handedness"] == "left" else " "
        print(f"  S{r['subject_id']:02d}{hm}  "
              f"{r['handedness'][0]:<3} {r['n_test']:>7}  "
              f"{r['acc']*100:>6.1f}%  "
              f"{r['f1_macro']*100:>6.1f}%")

    print("  " + "─"*45)
    print(f"  {'OVERALL':<9} "
          f"{sum(r['n_test'] for r in results):>7}  "
          f"{np.mean(all_acc)*100:>6.1f}%  "
          f"{np.mean(all_f1)*100:>6.1f}%")

    print(f"\n  Mean : {np.mean(all_acc)*100:.2f}% ± {np.std(all_acc)*100:.2f}%")
    print(f"  Best : S{max(results, key=lambda r: r['acc'])['subject_id']:02d} "
          f"@ {max(all_acc)*100:.1f}%")
    print(f"  Worst: S{min(results, key=lambda r: r['acc'])['subject_id']:02d} "
          f"@ {min(all_acc)*100:.1f}%")

    if right and left:
        print(f"\n  BY HANDEDNESS:")
        print(f"    Right ({len(right)}): "
              f"{np.mean([r['acc'] for r in right])*100:.2f}%")
        print(f"    Left  ({len(left)}): "
              f"{np.mean([r['acc'] for r in left])*100:.2f}%")

    if failed:
        print(f"\n  Failed: {failed}")
    print("═"*72)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--mode", type=str, default="both",
                        choices=["loso", "fewshot", "both"])
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    np.random.seed(CONFIG["random_state"])

    if args.subject:
        print(f"\n  Loading all subjects for test of S{args.subject:02d}...")
        all_data = {}
        for sid in ALL_SUBJECTS:
            X, y = load_and_preprocess(sid)
            if X is not None:
                all_data[sid] = (X, y)

        if args.subject not in all_data:
            print(f"  [!] S{args.subject:02d} could not be loaded.")
        else:
            if args.mode in ("loso", "both"):
                r = run_loso_pure(args.subject, all_data, verbose=True)
                if r:
                    print(f"\n  S{args.subject:02d} LOSO: {r['acc']*100:.2f}%")

            if args.mode in ("fewshot", "both"):
                r = run_fewshot(args.subject, all_data,
                                n_shots=args.shots, verbose=True)
                if r:
                    print(f"\n  S{args.subject:02d} Few-shot "
                          f"({args.shots}/class): {r['acc']*100:.2f}%")
    else:
        run_all(mode=args.mode, n_shots=args.shots, verbose=args.verbose)