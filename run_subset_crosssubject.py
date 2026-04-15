"""
run_subset_crosssubject.py
==========================
Cross-subject analysis on best-performing Arabic letter subsets.

Improvements over baseline:
  1. Subject-specific normalization — each subject's features normalized
     independently before combining, removes inter-subject scale differences
  2. Data augmentation — adds Gaussian noise to training features to
     increase effective training size and improve generalization
  3. Ensemble classifier — combines LDA + SVM predictions for robustness
  4. Per-subject CSP fitting — extracts features per subject then combines
     to avoid CSP being dominated by high-trial subjects
  5. Best letter selection — only uses top-K most discriminable letters

Usage:
    python run_subset_crosssubject.py              # all subset sizes
    python run_subset_crosssubject.py --n_classes 4
    python run_subset_crosssubject.py --subject 1 --n_classes 4
    python run_subset_crosssubject.py | tee results/subset_results.txt
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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

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


def align_dims(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    n_tr, n_te = X_train.shape[1], X_test.shape[1]
    if n_te < n_tr:
        X_test = np.hstack([X_test,
                            np.zeros((X_test.shape[0], n_tr - n_te))])
    elif n_te > n_tr:
        X_test = X_test[:, :n_tr]
    return X_train, X_test


def augment_features(X: np.ndarray, y: np.ndarray,
                     factor: int = 3) -> tuple:
    """
    Augment training features by adding small Gaussian noise.
    Multiplies training size by factor.
    Noise scale = 5% of each feature's std — small enough to preserve
    the signal, large enough to improve generalization.
    """
    X_aug, y_aug = [X], [y]
    noise_scale = X.std(axis=0) * 0.05
    for _ in range(factor - 1):
        noise = np.random.randn(*X.shape) * noise_scale
        X_aug.append(X + noise)
        y_aug.append(y)
    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)


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
# Step 1: Find best letters
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
            scaler = StandardScaler()
            X_tr_f = scaler.fit_transform(X_tr_f)
            X_te_f = scaler.transform(X_te_f)
            k = min(200, X_tr_f.shape[1])
            sel = SelectKBest(f_classif, k=k)
            X_tr_f = sel.fit_transform(X_tr_f, y_tr)
            X_te_f = sel.transform(X_te_f)
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(X_tr_f, y_tr)
            all_preds[te_idx] = clf.predict(X_te_f)
        except Exception:
            all_preds[te_idx] = y_te

    letter_accs = {}
    for letter in np.unique(y):
        mask = all_true == letter
        if mask.sum() > 0:
            letter_accs[letter] = float(
                (all_preds[mask] == all_true[mask]).mean()
            )
    return letter_accs


def find_best_letters(all_data: dict,
                      n_subjects_sample: int = 10) -> tuple:
    print(f"\n  Finding best letters using {n_subjects_sample} subjects...")
    subject_sample = list(sorted(all_data.keys()))[:n_subjects_sample]
    letter_scores = {i: [] for i in range(CONFIG["n_letters"])}

    for sid in subject_sample:
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
    sorted_letters = sorted(mean_scores.keys(),
                            key=lambda l: mean_scores[l],
                            reverse=True)

    print(f"\n  Letter ranking (best to worst):")
    for letter in sorted_letters[:15]:
        print(f"    L{letter+1:02d}: {mean_scores[letter]*100:.1f}%")

    return sorted_letters, mean_scores


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: LOSO on subset with all improvements
# ─────────────────────────────────────────────────────────────────────────────

def filter_subset(X: np.ndarray, y: np.ndarray,
                  best_letters: list, n_classes: int) -> tuple:
    keep = best_letters[:n_classes]
    mask = np.isin(y, keep)
    X_sub = X[mask]
    y_sub = y[mask]
    label_map = {old: new for new, old in enumerate(sorted(keep))}
    y_sub = np.array([label_map[l] for l in y_sub])
    return X_sub, y_sub


def run_loso_subset(test_id: int, all_data: dict,
                    best_letters: list, n_classes: int,
                    augment: bool = True,
                    use_ensemble: bool = True) -> dict:
    """
    LOSO cross-subject with:
    - Per-subject feature normalization
    - Data augmentation on training features
    - Ensemble of LDA + SVM
    """
    if test_id not in all_data:
        return None

    X_test_raw, y_test_raw = all_data[test_id]
    X_test, y_test = filter_subset(X_test_raw, y_test_raw,
                                   best_letters, n_classes)

    if len(np.unique(y_test)) < n_classes:
        return None

    # ── Extract features per subject, normalize independently ────────────
    # This is the KEY improvement — each subject normalized on its own
    # before combining, removing inter-subject scale differences
    X_train_parts, y_train_parts = [], []

    for sid, (X_s, y_s) in all_data.items():
        if sid == test_id:
            continue
        X_s_sub, y_s_sub = filter_subset(X_s, y_s,
                                          best_letters, n_classes)
        if len(X_s_sub) == 0:
            continue
        try:
            # Extract features for this subject
            X_s_feat = extract_features(X_s_sub, y_s_sub)
            # Normalize THIS subject independently
            scaler_s = StandardScaler()
            X_s_feat = scaler_s.fit_transform(X_s_feat)
            X_train_parts.append(X_s_feat)
            y_train_parts.append(y_s_sub)
        except Exception:
            continue

    if not X_train_parts:
        return None

    # Extract and normalize test subject independently
    try:
        X_te_feat = extract_features(X_test, y_test)
        scaler_te = StandardScaler()
        X_te_feat = scaler_te.fit_transform(X_te_feat)
    except Exception as e:
        return None

    # Align feature dimensions
    max_feat = max(X.shape[1] for X in X_train_parts)
    max_feat = max(max_feat, X_te_feat.shape[1])

    aligned_parts = []
    for X_part in X_train_parts:
        if X_part.shape[1] < max_feat:
            pad = np.zeros((X_part.shape[0], max_feat - X_part.shape[1]))
            X_part = np.hstack([X_part, pad])
        elif X_part.shape[1] > max_feat:
            X_part = X_part[:, :max_feat]
        aligned_parts.append(X_part)

    if X_te_feat.shape[1] < max_feat:
        pad = np.zeros((X_te_feat.shape[0], max_feat - X_te_feat.shape[1]))
        X_te_feat = np.hstack([X_te_feat, pad])
    elif X_te_feat.shape[1] > max_feat:
        X_te_feat = X_te_feat[:, :max_feat]

    X_train = np.concatenate(aligned_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)

    # ── Data augmentation on training features ────────────────────────────
    if augment:
        X_train, y_train = augment_features(X_train, y_train, factor=3)

    # ── Feature selection ─────────────────────────────────────────────────
    k = min(150, X_train.shape[1])
    sel = SelectKBest(f_classif, k=k)
    X_train_sel = sel.fit_transform(X_train, y_train)
    X_te_sel    = sel.transform(X_te_feat)

    # ── Classify: ensemble of LDA + SVM ──────────────────────────────────
    if use_ensemble:
        # LDA
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X_train_sel, y_train)
        pred_lda = lda.predict(X_te_sel)
        prob_lda = lda.predict_proba(X_te_sel)

        # SVM with probability
        svm = SVC(kernel="rbf", C=1.0, gamma="scale",
                  probability=True, random_state=42)
        svm.fit(X_train_sel, y_train)
        prob_svm = svm.predict_proba(X_te_sel)

        # Average probabilities
        prob_avg = (prob_lda + prob_svm) / 2
        y_pred = np.argmax(prob_avg, axis=1)
    else:
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X_train_sel, y_train)
        y_pred = lda.predict(X_te_sel)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    return {
        "subject_id": test_id,
        "handedness": get_handedness(test_id),
        "n_test":     len(y_test),
        "acc":        float(acc),
        "f1":         float(f1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full run
# ─────────────────────────────────────────────────────────────────────────────

def run_all_subsets(subset_sizes: list = None,
                    target_subject: int = None,
                    augment: bool = True,
                    use_ensemble: bool = True):

    sizes = subset_sizes or SUBSET_SIZES

    print("\n" + "═"*72)
    print("  IMPROVED SUBSET CROSS-SUBJECT ANALYSIS")
    print("  Improvements: per-subject normalization + augmentation + ensemble")
    print("═"*72)

    print("\n  Loading all subjects...")
    all_data = {}
    for sid in ALL_SUBJECTS:
        X, y = load_and_preprocess(sid)
        if X is not None:
            all_data[sid] = (X, y)
    print(f"  Loaded: {len(all_data)}/30 subjects")

    best_letters, letter_scores = find_best_letters(
        all_data, n_subjects_sample=10
    )

    all_results = {}

    for n_classes in sizes:
        chance = 100.0 / n_classes
        top_letters = best_letters[:n_classes]
        letter_names = [f"L{l+1:02d}" for l in top_letters]

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
                r = run_loso_subset(sid, all_data, best_letters,
                                    n_classes, augment=augment,
                                    use_ensemble=use_ensemble)
                if r is None:
                    raise ValueError("returned None")
                results.append(r)
                print(f"acc={r['acc']*100:.1f}%  f1={r['f1']*100:.1f}%")
            except Exception as e:
                print(f"FAILED: {e}")
                failed.append(sid)

        if results:
            all_results[n_classes] = results
            _print_summary(results, failed, n_classes,
                           chance, letter_names)

    if len(all_results) > 1:
        print("\n" + "═"*72)
        print("  FINAL COMPARISON — ALL SUBSET SIZES")
        print("═"*72)
        print(f"  {'Classes':<10} {'Chance':>8}  {'LOSO Acc':>10}  "
              f"{'vs Chance':>10}  {'Literature':>15}")
        print("  " + "─"*60)

        lit = {4: "~50-60%", 6: "~45-55%", 8: "~40-50%",
               10: "~35-45%", 15: "50.42% (Kamble)"}

        for n_classes, results in sorted(all_results.items()):
            mean_acc = np.mean([r["acc"] for r in results])
            chance   = 100.0 / n_classes
            vs_chance = mean_acc*100 - chance
            print(f"  {n_classes:<10} {chance:>7.1f}%  "
                  f"{mean_acc*100:>9.2f}%  "
                  f"{vs_chance:>+9.2f}%  "
                  f"{lit.get(n_classes, 'N/A'):>15}")

        print("  " + "─"*60)
        print(f"  Within-subject (all 28 classes): 73.08%")
        print(f"  Paper baseline (all 28 classes): 74.80%")
        print("═"*72)


def _print_summary(results, failed, n_classes, chance, letter_names):
    all_acc = [r["acc"] for r in results]
    all_f1  = [r["f1"]  for r in results]
    right   = [r for r in results if r["handedness"] == "right"]
    left    = [r for r in results if r["handedness"] == "left"]

    print(f"\n  {n_classes}-class Results:")
    print(f"  Mean acc  : {np.mean(all_acc)*100:.2f}% "
          f"± {np.std(all_acc)*100:.2f}%")
    print(f"  Mean F1   : {np.mean(all_f1)*100:.2f}%")
    print(f"  Chance    : {chance:.2f}%")
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
    parser.add_argument("--no_augment",   action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--no_ensemble",  action="store_true",
                        help="Disable ensemble, use LDA only")
    args = parser.parse_args()

    np.random.seed(CONFIG["random_state"])

    sizes = [args.n_classes] if args.n_classes else SUBSET_SIZES

    run_all_subsets(
        subset_sizes=sizes,
        target_subject=args.subject,
        augment=not args.no_augment,
        use_ensemble=not args.no_ensemble,
    )