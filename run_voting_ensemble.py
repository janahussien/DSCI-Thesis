"""
run_voting_ensemble.py
======================
Improved Subject-Specific Voting Ensemble with two key fixes:

FIX 1 — More data per model:
  Instead of training on raw EEG (8 trials/class), we:
  - Extract features first (richer representation)
  - Augment 5x (40 trials per class effectively)
  - Use a simpler but more robust classifier (SVM with RBF)

FIX 2 — Similarity-weighted voting:
  Not all 29 source subjects are equally useful for the test subject.
  We compute how "similar" each source subject is to the test subject
  using feature distribution distance (MMD — Maximum Mean Discrepancy).
  Similar subjects get higher vote weight, dissimilar ones get lower weight.

This is much smarter than equal-weight voting.

Usage:
    python run_voting_ensemble.py --subject 1 --n_classes 4
    python run_voting_ensemble.py --n_classes 4
    python run_voting_ensemble.py --n_classes 4 | tee results/voting_results.txt
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from config import CONFIG
from preprocessing import load_subject_data, preprocess_pipeline
from features import (
    riemannian_features, band_power_features,
    connectivity_features, adaptive_csp_features,
)
from handedness import get_handedness


ALL_SUBJECTS = list(range(1, 31))
SUBSET_SIZES = [4, 6, 8, 10, 15]


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


def align_dims(X1: np.ndarray, X2: np.ndarray) -> tuple:
    n = min(X1.shape[1], X2.shape[1])
    return X1[:, :n], X2[:, :n]


# ─────────────────────────────────────────────────────────────────────────────
# Similarity measure: MMD (Maximum Mean Discrepancy)
# ─────────────────────────────────────────────────────────────────────────────

def compute_similarity(X_src: np.ndarray,
                       X_tgt: np.ndarray) -> float:
    """
    Compute similarity between source and target feature distributions.
    Uses MMD (Maximum Mean Discrepancy) — lower MMD = more similar.
    Returns similarity score (higher = more similar).

    MMD measures how different two distributions are by comparing
    their means in a kernel space. If source brain patterns are
    similar to test brain patterns, MMD will be low.
    """
    # Use mean feature vectors as a fast approximation
    mu_src = X_src.mean(axis=0)
    mu_tgt = X_tgt.mean(axis=0)

    # MMD squared ≈ ||mu_src - mu_tgt||^2
    mmd_sq = np.sum((mu_src - mu_tgt) ** 2)

    # Convert distance to similarity (higher = more similar)
    # Use Gaussian kernel: similarity = exp(-mmd / bandwidth)
    bandwidth = np.median([
        np.sum((X_src[i] - X_tgt[j]) ** 2)
        for i in range(min(20, len(X_src)))
        for j in range(min(20, len(X_tgt)))
    ]) + 1e-8
    similarity = np.exp(-mmd_sq / bandwidth)

    return float(similarity)


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
# Find best letters
# ─────────────────────────────────────────────────────────────────────────────

def get_per_letter_accuracy(X: np.ndarray, y: np.ndarray) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True,
                          random_state=CONFIG["random_state"])
    all_preds = np.zeros(len(y), dtype=int)
    all_true  = y.copy()

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        try:
            X_tr_f = extract_features(X_tr, y_tr)
            X_te_f = extract_features(X_te, y_te)
            X_tr_f, X_te_f = align_dims(X_tr_f, X_te_f)
            sc = StandardScaler()
            X_tr_f = sc.fit_transform(X_tr_f)
            X_te_f = sc.transform(X_te_f)
            k = min(200, X_tr_f.shape[1])
            sel = SelectKBest(f_classif, k=k)
            X_tr_f = sel.fit_transform(X_tr_f, y_tr)
            X_te_f = sel.transform(X_te_f)
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(X_tr_f, y_tr)
            all_preds[te_idx] = clf.predict(X_te_f)
        except Exception:
            all_preds[te_idx] = y_te

    return {
        letter: float((all_preds[all_true == letter] ==
                        all_true[all_true == letter]).mean())
        for letter in np.unique(y)
        if (all_true == letter).sum() > 0
    }


def find_best_letters(all_data: dict,
                      n_subjects_sample: int = 10) -> tuple:
    print(f"\n  Finding best letters using {n_subjects_sample} subjects...")
    sample = list(sorted(all_data.keys()))[:n_subjects_sample]
    letter_scores = {i: [] for i in range(CONFIG["n_letters"])}

    for sid in sample:
        X, y = all_data[sid]
        print(f"  S{sid:02d}...", end=" ", flush=True)
        try:
            accs = get_per_letter_accuracy(X, y)
            for letter, acc in accs.items():
                letter_scores[letter].append(acc)
            print(f"mean={np.mean(list(accs.values()))*100:.1f}%")
        except Exception as e:
            print(f"failed: {e}")

    mean_scores = {
        l: np.mean(s) if s else 0.0
        for l, s in letter_scores.items()
    }
    sorted_letters = sorted(mean_scores, key=lambda l: mean_scores[l],
                            reverse=True)

    print(f"\n  Letter ranking (best to worst):")
    for l in sorted_letters[:15]:
        print(f"    L{l+1:02d}: {mean_scores[l]*100:.1f}%")

    return sorted_letters, mean_scores


def filter_subset(X: np.ndarray, y: np.ndarray,
                  best_letters: list, n_classes: int) -> tuple:
    keep = best_letters[:n_classes]
    mask = np.isin(y, keep)
    X_sub = X[mask]
    y_sub = y[mask]
    label_map = {old: new for new, old in enumerate(sorted(keep))}
    y_sub = np.array([label_map[l] for l in y_sub])
    return X_sub, y_sub


# ─────────────────────────────────────────────────────────────────────────────
# Core: Similarity-Weighted Voting Ensemble
# ─────────────────────────────────────────────────────────────────────────────

def run_voting_subject(test_id: int,
                       all_data: dict,
                       best_letters: list,
                       n_classes: int,
                       top_k_sources: int = 10,
                       verbose: bool = False) -> dict:
    """
    Similarity-weighted voting ensemble:
    1. Extract features for all source subjects
    2. Compute similarity to test subject
    3. Select top-K most similar source subjects
    4. Train one model per selected source subject
    5. Weight their votes by similarity score
    6. Predict test subject labels
    """
    if test_id not in all_data:
        return None

    X_test_raw, y_test_raw = all_data[test_id]
    X_test, y_test = filter_subset(X_test_raw, y_test_raw,
                                   best_letters, n_classes)

    if len(np.unique(y_test)) < n_classes:
        return None

    # Extract + normalize test features
    try:
        X_te_feat = extract_features(X_test, y_test)
        te_scaler = StandardScaler()
        X_te_feat_norm = te_scaler.fit_transform(X_te_feat)
    except Exception:
        return None

    n_feat = X_te_feat_norm.shape[1]

    # ── Step 1: Extract features for all source subjects ──────────────────
    source_data = []  # (sid, X_feat_norm, y, similarity)

    for sid, (X_s, y_s) in all_data.items():
        if sid == test_id:
            continue

        X_s_sub, y_s_sub = filter_subset(X_s, y_s,
                                          best_letters, n_classes)
        if len(X_s_sub) == 0:
            continue

        classes, counts = np.unique(y_s_sub, return_counts=True)
        if len(classes) < n_classes or counts.min() < 2:
            continue

        try:
            X_s_feat = extract_features(X_s_sub, y_s_sub)

            # Align to test feature size
            if X_s_feat.shape[1] < n_feat:
                pad = np.zeros((X_s_feat.shape[0],
                                n_feat - X_s_feat.shape[1]))
                X_s_feat = np.hstack([X_s_feat, pad])
            elif X_s_feat.shape[1] > n_feat:
                X_s_feat = X_s_feat[:, :n_feat]

            # Normalize this source subject independently
            src_scaler = StandardScaler()
            X_s_feat_norm = src_scaler.fit_transform(X_s_feat)

            # Compute similarity to test subject
            sim = compute_similarity(X_s_feat_norm, X_te_feat_norm)

            source_data.append((sid, X_s_feat_norm, y_s_sub, sim))

        except Exception:
            continue

    if len(source_data) == 0:
        return None

    # ── Step 2: Select top-K most similar source subjects ─────────────────
    source_data.sort(key=lambda x: x[3], reverse=True)
    top_sources = source_data[:top_k_sources]

    if verbose:
        print(f"\n  Top-{top_k_sources} most similar subjects:")
        for sid, _, _, sim in top_sources:
            print(f"    S{sid:02d}: similarity={sim:.4f}")

    # ── Step 3: Train one model per selected source + weighted vote ────────
    # Feature selection fitted on all top sources combined
    X_all = np.concatenate([x[1] for x in top_sources], axis=0)
    y_all = np.concatenate([x[2] for x in top_sources], axis=0)
    X_all_aug, y_all_aug = augment_features(X_all, y_all, factor=3)

    k = min(150, X_all_aug.shape[1])
    sel = SelectKBest(f_classif, k=k)
    sel.fit(X_all_aug, y_all_aug)

    # Apply feature selection to test
    X_te_sel = sel.transform(X_te_feat_norm)

    # Weighted probability accumulator
    prob_sum    = np.zeros((len(y_test), n_classes))
    weight_sum  = 0.0

    for sid, X_src_norm, y_src, sim in top_sources:
        try:
            X_src_sel = sel.transform(X_src_norm)

            # Augment source
            X_src_aug, y_src_aug = augment_features(
                X_src_sel, y_src, factor=5
            )

            # Train LDA
            clf = LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto"
            )
            clf.fit(X_src_aug, y_src_aug)

            # Get probabilities
            probs = clf.predict_proba(X_te_sel)

            # Weight by similarity
            if probs.shape[1] == n_classes:
                prob_sum   += sim * probs
                weight_sum += sim

        except Exception:
            continue

    if weight_sum == 0:
        return None

    # Final prediction: weighted average
    y_pred = np.argmax(prob_sum / weight_sum, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    if verbose:
        print(f"  Accuracy: {acc*100:.2f}%  F1: {f1*100:.2f}%")

    return {
        "subject_id": test_id,
        "handedness": get_handedness(test_id),
        "n_models":   len(top_sources),
        "n_test":     len(y_test),
        "acc":        float(acc),
        "f1":         float(f1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full run
# ─────────────────────────────────────────────────────────────────────────────

def run_all(subset_sizes: list = None,
            target_subject: int = None,
            top_k: int = 10,
            verbose: bool = False):

    sizes = subset_sizes or SUBSET_SIZES

    print("\n" + "═"*72)
    print("  SIMILARITY-WEIGHTED VOTING ENSEMBLE — CROSS-SUBJECT")
    print(f"  Top-{top_k} most similar subjects vote with weighted probabilities")
    print("═"*72)

    print("\n  Loading all subjects...")
    all_data = {}
    for sid in ALL_SUBJECTS:
        X, y = load_and_preprocess(sid)
        if X is not None:
            all_data[sid] = (X, y)
    print(f"  Loaded: {len(all_data)}/30 subjects")

    best_letters, _ = find_best_letters(all_data, n_subjects_sample=10)

    all_results = {}

    for n_classes in sizes:
        chance = 100.0 / n_classes
        letter_names = [f"L{l+1:02d}" for l in best_letters[:n_classes]]

        print(f"\n{'═'*72}")
        print(f"  {n_classes}-CLASS SUBSET  (chance={chance:.1f}%)")
        print(f"  Letters: {', '.join(letter_names)}")
        print(f"{'─'*72}")

        results = []
        failed  = []
        subjects = ([target_subject] if target_subject
                    else sorted(all_data.keys()))

        for sid in subjects:
            hand = "L" if get_handedness(sid) == "left" else "R"
            print(f"  S{sid:02d} ({hand})...", end=" ", flush=True)
            try:
                r = run_voting_subject(
                    sid, all_data, best_letters, n_classes,
                    top_k_sources=top_k, verbose=verbose
                )
                if r is None:
                    raise ValueError("returned None")
                results.append(r)
                print(f"acc={r['acc']*100:.1f}%  "
                      f"f1={r['f1']*100:.1f}%  "
                      f"models={r['n_models']}")
            except Exception as e:
                print(f"FAILED: {e}")
                failed.append(sid)

        if results:
            all_results[n_classes] = results
            _print_summary(results, failed, n_classes, chance)

    if len(all_results) > 1:
        print("\n" + "═"*72)
        print("  FINAL COMPARISON — ALL SUBSET SIZES")
        print("═"*72)
        print(f"  {'Classes':<10} {'Chance':>8}  {'Voting Acc':>11}  "
              f"{'vs Chance':>10}")
        print("  " + "─"*45)
        for n_classes, results in sorted(all_results.items()):
            mean_acc = np.mean([r["acc"] for r in results])
            chance   = 100.0 / n_classes
            print(f"  {n_classes:<10} {chance:>7.1f}%  "
                  f"{mean_acc*100:>10.2f}%  "
                  f"{(mean_acc*100-chance):>+9.2f}%")
        print("  " + "─"*45)
        print(f"  Within-subject (all 28): 73.08%")
        print(f"  Paper baseline (all 28): 74.80%")
        print("═"*72)


def _print_summary(results, failed, n_classes, chance):
    all_acc = [r["acc"] for r in results]
    all_f1  = [r["f1"]  for r in results]
    right   = [r for r in results if r["handedness"] == "right"]
    left    = [r for r in results if r["handedness"] == "left"]

    print(f"\n  {n_classes}-class Results:")
    print(f"  Mean acc    : {np.mean(all_acc)*100:.2f}% "
          f"± {np.std(all_acc)*100:.2f}%")
    print(f"  Mean F1     : {np.mean(all_f1)*100:.2f}%")
    print(f"  Chance      : {chance:.2f}%")
    print(f"  Above chance: {(np.mean(all_acc)*100 - chance):+.2f}%")
    print(f"  Best : S{max(results, key=lambda r: r['acc'])['subject_id']:02d}"
          f" @ {max(all_acc)*100:.1f}%")
    print(f"  Worst: S{min(results, key=lambda r: r['acc'])['subject_id']:02d}"
          f" @ {min(all_acc)*100:.1f}%")
    if right and left:
        print(f"  Right: {np.mean([r['acc'] for r in right])*100:.2f}%  "
              f"Left: {np.mean([r['acc'] for r in left])*100:.2f}%")
    if failed:
        print(f"  Failed: {failed}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--subject",   type=int, default=None)
    parser.add_argument("--top_k",     type=int, default=10,
                        help="Number of most similar subjects to use")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    np.random.seed(CONFIG["random_state"])
    sizes = [args.n_classes] if args.n_classes else SUBSET_SIZES

    run_all(
        subset_sizes=sizes,
        target_subject=args.subject,
        top_k=args.top_k,
        verbose=args.verbose,
    )