"""
run_cross_session.py
====================
Simulates cross-session generalisation using trial order as a proxy.

Since this dataset has only one recording session per subject, we split
the 10 trials per letter chronologically:

    "Session 1" = trials 1–5   (train)
    "Session 2" = trials 6–10  (test)

This is scientifically meaningful because:
  - Trials recorded earlier vs later in the session reflect natural
    drift in attention, fatigue, and electrode impedance over time
  - It tests whether features learned early generalise to later brain states
  - It's a stricter test than random CV folds (no temporal leakage)

Uses the winning feature combo: Riemannian + Band Power + PLV + LDA (84.08%) + Adaptive CSP (no leakage).

Run:
    python run_cross_session.py --subject 1
    python run_cross_session.py --subject 2
    python run_cross_session.py --all        # all 30 subjects
"""

import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

from config import CONFIG
from preprocessing import preprocess_pipeline, _bandpass, _notch, \
    _car_rereference, _reject_epoch, _baseline_correct, _zscore_normalize, \
    _extract_imagination_epoch
from features import (
    band_power_features,
    riemannian_features,
    connectivity_features,
    adaptive_csp_features,
    _adaptive_n_components
)
from utils import print_banner
import scipy.io as sio


# ─────────────────────────────────────────────────────────────────────────────
# Load with session split preserved
# ─────────────────────────────────────────────────────────────────────────────

def load_subject_split(subject_id: int, data_root: Path):
    """
    Load all trials for a subject and return them split by session.

    Returns:
        session1_records : list of dicts  (trials 1–5 per letter)
        session2_records : list of dicts  (trials 6–10 per letter)
    """
    session1, session2 = [], []
    sfreq = CONFIG["sfreq"]

    s_folder = data_root / f"S{subject_id:02d}"
    if not s_folder.exists():
        raise FileNotFoundError(f"Subject folder not found: {s_folder}")

    for letter_id in range(1, CONFIG["n_letters"] + 1):
        l_folder = s_folder / f"L{letter_id:02d}"

        for trial_id in range(1, CONFIG["n_trials"] + 1):
            mat_path = l_folder / f"S{subject_id:02d}_L{letter_id:02d}_T{trial_id}.mat"
            if not mat_path.exists():
                continue

            mat        = sio.loadmat(str(mat_path), simplify_cells=True)
            eeg_struct = mat.get("EEG", {})

            if isinstance(eeg_struct, dict) and "Data" in eeg_struct:
                eeg_data = np.array(eeg_struct["Data"], dtype=np.float64)
            else:
                continue

            if eeg_data.ndim == 2 and eeg_data.shape[0] > eeg_data.shape[1]:
                eeg_data = eeg_data.T

            record = {
                "eeg":    eeg_data,
                "label":  letter_id - 1,
                "trial":  trial_id,
                "letter": letter_id,
                "sfreq":  sfreq,
            }

            if trial_id <= 5:
                session1.append(record)
            else:
                session2.append(record)

    print(f"      Session 1 (trials 1–5):  {len(session1)} raw trials")
    print(f"      Session 2 (trials 6–10): {len(session2)} raw trials")
    return session1, session2


def preprocess_records(records):
    """
    Apply full preprocessing pipeline to a list of records.
    Returns X (n_trials, n_ch, n_t), y (n_trials,)
    """
    sfreq    = CONFIG["sfreq"]
    epochs, labels = [], []
    rejected = 0

    for rec in records:
        raw = rec["eeg"].copy()

        epoch = _extract_imagination_epoch(raw, sfreq)
        if epoch is None:
            rejected += 1
            continue

        epoch = _bandpass(epoch, CONFIG["bandpass"], sfreq)
        for f0 in CONFIG["notch_freqs"]:
            epoch = _notch(epoch, f0, sfreq)
        epoch = _car_rereference(epoch)

        if _reject_epoch(epoch):
            rejected += 1
            continue

        epoch = _baseline_correct(epoch, sfreq)
        epochs.append(epoch)
        labels.append(rec["label"])

    X = np.array(epochs, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    X = _zscore_normalize(X)
    return X, y, rejected


def extract_winning_features(X_train, y_train, X_test):
    """
    Extract Riem + BP + PLV + Adaptive CSP (NO leakage).
    CSP is fit on training data only.
    """

    # ── Base features ─────────────────────────────
    Xr_tr = np.nan_to_num(riemannian_features(X_train))
    Xb_tr = np.nan_to_num(band_power_features(X_train))
    Xp_tr = np.nan_to_num(connectivity_features(X_train))

    Xr_te = np.nan_to_num(riemannian_features(X_test))
    Xb_te = np.nan_to_num(band_power_features(X_test))
    Xp_te = np.nan_to_num(connectivity_features(X_test))

    # ── Adaptive CSP ──────────────────────────────
    n_comp = _adaptive_n_components(X_train.shape[0])

    Xc_tr, csp_model = adaptive_csp_features(X_train, y_train, return_model=True)
    Xc_te            = adaptive_csp_features(X_test,  y=None, model=csp_model)

    Xc_tr = np.nan_to_num(Xc_tr)
    Xc_te = np.nan_to_num(Xc_te)

    # ── Concatenate ───────────────────────────────
    F_train = np.concatenate([Xr_tr, Xb_tr, Xp_tr, Xc_tr], axis=1)
    F_test  = np.concatenate([Xr_te, Xb_te, Xp_te, Xc_te], axis=1)

    return F_train, F_test, n_comp


# ─────────────────────────────────────────────────────────────────────────────
# Cross-session evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_session(subject_id: int, verbose: bool = True):
    if verbose:
        print_banner(f"Cross-Session — Subject S{subject_id:02d}")
        print("\n  Strategy: Train on trials 1–5, Test on trials 6–10")
        print("  Features: Riemannian + Band Power + PLV")
        print("  Classifiers: LDA, LR, SVM_Linear, SVM_RBF, RF\n")

    # ── Load ──────────────────────────────────────────────────────────────
    if verbose: print("[1/4] Loading data...")
    s1_records, s2_records = load_subject_split(
        subject_id, CONFIG["data_root"]
    )

    # ── Preprocess ────────────────────────────────────────────────────────
    if verbose: print("\n[2/4] Preprocessing...")
    X_train, y_train, rej1 = preprocess_records(s1_records)
    X_test,  y_test,  rej2 = preprocess_records(s2_records)

    if verbose:
        print(f"      Train: {X_train.shape[0]} trials "
              f"({rej1} rejected)  |  "
              f"Test: {X_test.shape[0]} trials ({rej2} rejected)")

    # Check we have enough trials
    if X_train.shape[0] < 10 or X_test.shape[0] < 10:
        print(f"  [!] Not enough trials for S{subject_id:02d}, skipping.")
        return None

    # ── Features ──────────────────────────────────────────────────────────
    if verbose: print("\n[3/4] Extracting features...")
    F_train, F_test, n_comp = extract_winning_features(X_train, y_train, X_test)

    if verbose:
        print(f"      Train features: {F_train.shape}")
        print(f"      Test  features: {F_test.shape}")

    # Per-feature normalisation (fit on train only — no leakage)
    scaler  = StandardScaler()
    F_train = scaler.fit_transform(F_train)
    F_test  = scaler.transform(F_test)

    # ── Classify ──────────────────────────────────────────────────────────
    if verbose: print("\n[4/4] Training & evaluating classifiers...")

    classifiers = {
        "LDA":        LinearDiscriminantAnalysis(solver="svd"),
        "LR":         LogisticRegression(max_iter=1000, C=1.0),
        "SVM_Linear": SVC(kernel="linear", C=1.0),
        "SVM_RBF":    SVC(kernel="rbf",    C=10.0, gamma="scale"),
        "RF":         RandomForestClassifier(n_estimators=300,
                                             random_state=CONFIG["random_state"]),
    }

    results = {}
    best_acc, best_name = 0, ""

    if verbose:
        print(f"\n  {'Classifier':<15} {'Accuracy':>10}  {'F1 Macro':>10}")
        print("  " + "─"*38)

    for name, clf in classifiers.items():
        try:
            clf.fit(F_train, y_train)
            y_pred = clf.predict(F_test)
            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
            results[name] = {"acc": acc, "f1": f1, "y_pred": y_pred.tolist()}

            if verbose:
                marker = " ←" if acc > best_acc else ""
                print(f"  {name:<15} {acc*100:>9.2f}%  {f1*100:>9.2f}%{marker}")

            if acc > best_acc:
                best_acc, best_name = acc, name

        except Exception as e:
            if verbose: print(f"  {name:<15} FAILED: {e}")

    # ── Compare to within-subject CV ──────────────────────────────────────
    if verbose:
        print("  " + "─"*38)
        print(f"\n  Best cross-session:    {best_acc*100:.2f}%  ({best_name})")
        print(f"  Within-subject CV:     84.08%  (10-fold, same subject)")
        drop = 84.08 - best_acc * 100
        print(f"  Performance drop:      {drop:.2f}%")

        if drop < 10:
            verdict = "✅ Excellent — model generalises well across time"
        elif drop < 20:
            verdict = "⚠️  Moderate drop — some temporal drift present"
        else:
            verdict = "❌ Large drop — significant session drift detected"
        print(f"\n  Verdict: {verdict}")

        # ── Per-class breakdown for best classifier ────────────────────────
        best_clf = classifiers[best_name]
        best_clf.fit(F_train, y_train)
        y_pred_best = best_clf.predict(F_test)

        print(f"\n  Per-class accuracy (best classifier: {best_name}):")
        print("  " + "─"*50)
        letter_names = [
            "ا","ب","ت","ث","ج","ح","خ","د","ذ","ر",
            "ز","س","ش","ص","ض","ط","ظ","ع","غ","ف",
            "ق","ك","ل","م","ن","ه","و","ي"
        ]
        for cls_id in range(CONFIG["n_letters"]):
            mask     = y_test == cls_id
            if mask.sum() == 0:
                continue
            cls_acc  = accuracy_score(y_test[mask], y_pred_best[mask])
            n_trials = mask.sum()
            bar      = "█" * int(cls_acc * 20)
            letter   = letter_names[cls_id] if cls_id < len(letter_names) else str(cls_id)
            print(f"  {letter}  (L{cls_id+1:02d})  [{bar:<20}]  "
                  f"{cls_acc*100:5.1f}%  ({n_trials} trials)")

    return {
        "subject_id":    subject_id,
        "train_trials":  int(X_train.shape[0]),
        "test_trials":   int(X_test.shape[0]),
        "results":       {k: {"acc": v["acc"], "f1": v["f1"]}
                          for k, v in results.items()},
        "best_acc":      best_acc,
        "best_model":    best_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# All subjects
# ─────────────────────────────────────────────────────────────────────────────

def run_all_subjects():
    print_banner("Cross-Session — All 30 Subjects")
    all_results = {}

    for sid in range(1, CONFIG["n_subjects"] + 1):
        print(f"\n  Subject S{sid:02d}...", end=" ")
        try:
            r = run_cross_session(sid, verbose=False)
            if r:
                all_results[sid] = r
                print(f"Best: {r['best_acc']*100:.1f}% ({r['best_model']})")
            else:
                print("skipped")
        except Exception as e:
            print(f"FAILED: {e}")

    if not all_results:
        print("No results collected.")
        return

    accs = [r["best_acc"] for r in all_results.values()]
    print("\n" + "═"*55)
    print("  CROSS-SESSION SUMMARY — ALL SUBJECTS")
    print("═"*55)
    print(f"  Subjects completed : {len(all_results)}/30")
    print(f"  Mean accuracy      : {np.mean(accs)*100:.2f}%")
    print(f"  Std deviation      : {np.std(accs)*100:.2f}%")
    print(f"  Min                : {np.min(accs)*100:.2f}%")
    print(f"  Max                : {np.max(accs)*100:.2f}%")
    print(f"  Within-subject CV  : 84.08%  (for reference)")
    print(f"  Paper baseline     : 74.80%")
    print("═"*55)

    # Per-subject table
    print(f"\n  {'Subject':<10} {'Acc':>8}  {'Best Model':<15}")
    print("  " + "─"*36)
    for sid, r in sorted(all_results.items()):
        print(f"  S{sid:02d}       {r['best_acc']*100:>7.2f}%  {r['best_model']}")

    # Save
    import json
    from datetime import datetime
    out_dir = CONFIG["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"cross_session_all_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\n  Results saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--all", action="store_true",
                        help="Run all 30 subjects")
    args = parser.parse_args()

    if args.all:
        run_all_subjects()
    else:
        run_cross_session(args.subject, verbose=True)