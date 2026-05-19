"""
EEG Arabic Alphabet BCI Pipeline
==================================
Main entry point. Runs the full pipeline on a single subject first,
then optionally scales to all subjects.

Usage:
    python main.py --subject 1          # single subject test
    python main.py --all                # all 30 subjects
    python main.py --subject 1 --debug  # verbose output + plots
"""

import argparse
import numpy as np
from pathlib import Path

from config import CONFIG
from preprocessing import load_subject_data, preprocess_pipeline
from features import extract_all_features
from models import run_all_models
from utils import print_banner, save_results


def run_single_subject(subject_id: int, debug: bool = False):
    print_banner(f"Processing Subject S{subject_id:02d}")

    # 1. Load raw .mat files
    print("\n[1/4] Loading raw EEG data...")
    raw_data = load_subject_data(subject_id, CONFIG["data_root"])
    print(f"      Loaded {len(raw_data)} trials across {CONFIG['n_letters']} letters")

    # 2. Full preprocessing pipeline
    print("\n[2/4] Preprocessing...")
    X, y = preprocess_pipeline(raw_data, debug=debug)
    print(f"      Final data shape: {X.shape}  |  Labels: {np.unique(y)}")

    # 3. Feature extraction
    print("\n[3/4] Extracting features...")
    features = extract_all_features(X, debug=debug)
    for fname, fmat in features.items():
        print(f"      {fname:<30} shape: {fmat.shape}")

    # 4. Run all models
    print("\n[4/4] Running models...")
    results = run_all_models(features, y, subject_id=subject_id, debug=debug)

    # 5. Save & summarise
    save_results(results, subject_id)
    return results


def run_all_subjects():
    all_results = {}
    for sid in range(1, CONFIG["n_subjects"] + 1):
        try:
            all_results[sid] = run_single_subject(sid)
        except Exception as e:
            print(f"  [!] Subject {sid} failed: {e}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Arabic BCI Pipeline")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.all:
        run_all_subjects()
    else:
        run_single_subject(args.subject, debug=args.debug)