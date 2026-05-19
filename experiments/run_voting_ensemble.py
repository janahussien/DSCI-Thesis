"""
run_voting_ensemble.py
======================
Voting Ensemble — Full 28-Class Cross-Subject Analysis.

Strategy:
  - Load all 30 subjects
  - Exclude S22 and S29 from the source (training) pool entirely
  - For each test subject, train one LDA model per remaining source subject
  - Average all probability outputs → pick highest class (equal-weight vote)
  - Test on ALL 28 classes (no letter filtering)
  - Report per-subject accuracy and overall average

Usage:
    python run_voting_ensemble.py                  # all 30 subjects
    python run_voting_ensemble.py --subject 1      # test S01 only
    python run_voting_ensemble.py --verbose        # show per-model details
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


ALL_SUBJECTS  = list(range(1, 31))
EXCLUDE_FROM_SOURCES = {22, 29}   # never used as training subjects


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_riem = np.nan_to_num(riemannian_features(X))
    X_bp   = np.nan_to_num(band_power_features(X))
    X_plv  = np.nan_to_num(connectivity_features(X))
    X_csp  = np.nan_to_num(adaptive_csp_features(X, y))
    return np.concatenate([X_riem, X_bp, X_plv, X_csp], axis=1)


def augment_features(X: np.ndarray, y: np.ndarray,
                     factor: int = 5) -> tuple:
    """Augment features with small Gaussian noise (5% std)."""
    rng = np.random.RandomState(42)
    noise_scale = X.std(axis=0) * 0.05
    X_aug, y_aug = [X], [y]
    for _ in range(factor - 1):
        X_aug.append(X + rng.randn(*X.shape) * noise_scale)
        y_aug.append(y)
    return np.concatenate(X_aug), np.concatenate(y_aug)


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
# Core: Equal-weight voting ensemble — all 28 classes
# ─────────────────────────────────────────────────────────────────────────────

def run_voting_subject(test_id: int,
                       all_data: dict,
                       verbose: bool = False) -> dict:
    """
    For one test subject:
      1. Collect all source subjects (excluding test subject + S22 + S29)
      2. Extract + normalize features per source subject independently
      3. Train one LDA per source subject (with augmentation)
      4. Average all probability outputs equally
      5. Predict and evaluate on all 28 classes
    """
    if test_id not in all_data:
        return None

    X_test, y_test = all_data[test_id]

    # Extract and normalize test features
    try:
        X_te_feat = extract_features(X_test, y_test)
        te_scaler = StandardScaler()
        X_te_feat = te_scaler.fit_transform(X_te_feat)
    except Exception as e:
        if verbose:
            print(f"  [!] Test feature extraction failed: {e}")
        return None

    n_feat    = X_te_feat.shape[1]
    n_classes = CONFIG["n_letters"]  # 28

    # ── Build one model per source subject ───────────────────────────────
    prob_sum   = np.zeros((len(y_test), n_classes))
    n_models   = 0
    skipped    = []

    source_ids = [
        sid for sid in sorted(all_data.keys())
        if sid != test_id and sid not in EXCLUDE_FROM_SOURCES
    ]

    if verbose:
        print(f"  Source subjects ({len(source_ids)}): {source_ids}")

    for sid in source_ids:
        X_src, y_src = all_data[sid]

        # Check source has all 28 classes with enough trials
        classes, counts = np.unique(y_src, return_counts=True)
        if len(classes) < n_classes or counts.min() < 2:
            skipped.append(sid)
            continue

        try:
            # Extract features for this source subject
            X_src_feat = extract_features(X_src, y_src)

            # Align feature dimensions to test subject
            if X_src_feat.shape[1] < n_feat:
                pad = np.zeros((X_src_feat.shape[0],
                                n_feat - X_src_feat.shape[1]))
                X_src_feat = np.hstack([X_src_feat, pad])
            elif X_src_feat.shape[1] > n_feat:
                X_src_feat = X_src_feat[:, :n_feat]

            # Normalize this source subject independently
            src_scaler = StandardScaler()
            X_src_feat = src_scaler.fit_transform(X_src_feat)

            # Feature selection fitted on this source subject
            k = min(200, X_src_feat.shape[1])
            sel = SelectKBest(f_classif, k=k)
            X_src_sel = sel.fit_transform(X_src_feat, y_src)
            X_te_sel  = sel.transform(X_te_feat)

            # Augment source features
            X_src_aug, y_src_aug = augment_features(X_src_sel, y_src,
                                                     factor=5)

            # Train LDA
            clf = LinearDiscriminantAnalysis(solver="lsqr",
                                             shrinkage="auto")
            clf.fit(X_src_aug, y_src_aug)

            # Get probabilities — only add if all 28 classes predicted
            probs = clf.predict_proba(X_te_sel)
            if probs.shape[1] == n_classes:
                prob_sum += probs
                n_models += 1
                if verbose:
                    src_pred = clf.predict(X_src_sel)
                    src_acc  = accuracy_score(y_src, src_pred)
                    print(f"    S{sid:02d} — train acc: {src_acc*100:.1f}%  ✓")
            else:
                skipped.append(sid)

        except Exception as e:
            skipped.append(sid)
            if verbose:
                print(f"    S{sid:02d} — failed: {e}")
            continue

    if n_models == 0:
        return None

    # ── Final prediction: equal-weight average ────────────────────────────
    y_pred = np.argmax(prob_sum / n_models, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    return {
        "subject_id": test_id,
        "handedness": get_handedness(test_id),
        "n_models":   n_models,
        "n_test":     len(y_test),
        "acc":        float(acc),
        "f1":         float(f1),
        "skipped":    skipped,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full run
# ─────────────────────────────────────────────────────────────────────────────

def run_all(target_subject: int = None, verbose: bool = False):

    print("\n" + "═"*72)
    print("  VOTING ENSEMBLE — FULL 28-CLASS CROSS-SUBJECT")
    print(f"  One model per source subject  |  Equal-weight probability vote")
    print(f"  Excluded from training: S22, S29")
    print("═"*72)

    print("\n  Loading all subjects...")
    all_data = {}
    for sid in ALL_SUBJECTS:
        X, y = load_and_preprocess(sid)
        if X is not None:
            all_data[sid] = (X, y)
            print(f"  S{sid:02d} ✅  {X.shape[0]} trials")
        else:
            print(f"  S{sid:02d} ❌  skipped")

    print(f"\n  Loaded: {len(all_data)}/30 subjects")
    print(f"  Chance level (1/28): 3.57%")
    print("─"*72)

    results = []
    failed  = []
    subjects = ([target_subject] if target_subject
                else sorted(all_data.keys()))

    for sid in subjects:
        hand = "L" if get_handedness(sid) == "left" else "R"
        print(f"  S{sid:02d} ({hand})...", end=" ", flush=True)
        try:
            r = run_voting_subject(sid, all_data, verbose=verbose)
            if r is None:
                raise ValueError("returned None")
            results.append(r)
            print(f"acc={r['acc']*100:.2f}%  "
                  f"f1={r['f1']*100:.2f}%  "
                  f"models={r['n_models']}")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(sid)

    if results:
        _print_summary(results, failed)


def _print_summary(results: list, failed: list):
    all_acc = [r["acc"] for r in results]
    all_f1  = [r["f1"]  for r in results]
    right   = [r for r in results if r["handedness"] == "right"]
    left    = [r for r in results if r["handedness"] == "left"]

    print("\n" + "═"*72)
    print("  VOTING ENSEMBLE — 28-CLASS RESULTS")
    print("═"*72)
    print(f"  {'Subj':<6} {'H':<3} {'N_test':>7}  {'Models':>7}  "
          f"{'Acc':>8}  {'F1':>8}")
    print("  " + "─"*50)

    for r in results:
        hm = "◄" if r["handedness"] == "left" else " "
        print(f"  S{r['subject_id']:02d}{hm}  "
              f"{r['handedness'][0]:<3} {r['n_test']:>7}  "
              f"{r['n_models']:>7}  "
              f"{r['acc']*100:>7.2f}%  "
              f"{r['f1']*100:>7.2f}%")

    print("  " + "─"*50)
    print(f"\n  Mean accuracy : {np.mean(all_acc)*100:.2f}% "
          f"± {np.std(all_acc)*100:.2f}%")
    print(f"  Mean F1-macro : {np.mean(all_f1)*100:.2f}%")
    print(f"  Chance level  : 3.57%  (1/28)")
    print(f"  Above chance  : {(np.mean(all_acc)*100 - 3.57):+.2f}%")
    print(f"  Best          : S{max(results, key=lambda r: r['acc'])['subject_id']:02d}"
          f" @ {max(all_acc)*100:.2f}%")
    print(f"  Worst         : S{min(results, key=lambda r: r['acc'])['subject_id']:02d}"
          f" @ {min(all_acc)*100:.2f}%")

    if right and left:
        print(f"\n  BY HANDEDNESS:")
        print(f"    Right ({len(right)}): "
              f"{np.mean([r['acc'] for r in right])*100:.2f}%")
        print(f"    Left  ({len(left)}): "
              f"{np.mean([r['acc'] for r in left])*100:.2f}%")

    if failed:
        print(f"\n  Failed: {failed}")

    print(f"\n  COMPARISON:")
    print(f"    Chance (1/28)             : 3.57%")
    print(f"    Voting ensemble (28-class): {np.mean(all_acc)*100:.2f}%")
    print(f"    Within-subject (28-class) : 73.08%")
    print(f"    Paper baseline            : 74.80%")
    print("═"*72)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=None,
                        help="Test a single subject (default: all 30)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-model details")
    args = parser.parse_args()

    np.random.seed(CONFIG["random_state"])
    run_all(target_subject=args.subject, verbose=args.verbose)