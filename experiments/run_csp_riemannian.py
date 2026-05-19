"""
run_csp_riemannian.py
=====================
Runs ONLY CSP and Riemannian features + all classical models
on a subject without redoing preprocessing or other features.
"""

import numpy as np
import warnings
from pathlib import Path

from config import CONFIG
from preprocessing import load_subject_data, preprocess_pipeline
from features import csp_features, riemannian_features, _covariance_features
from models import run_classical_models, _print_result
from utils import print_banner

def run_csp_riemannian(subject_id: int = 1):
    print_banner(f"CSP + Riemannian — Subject S{subject_id:02d}")

    # ── Load & preprocess (fast, already know it works) ───────────────────
    print("\n[1/3] Loading & preprocessing...")
    raw_data = load_subject_data(subject_id, CONFIG["data_root"])
    X, y = preprocess_pipeline(raw_data)
    print(f"      Data shape: {X.shape}")

    # ── CSP features ──────────────────────────────────────────────────────
    print("\n[2/3] Extracting CSP features...")
    try:
        X_csp = csp_features(X, y)
        X_csp = np.nan_to_num(X_csp)
        print(f"      CSP shape: {X_csp.shape}")

        print("\n  ── Feature set: CSP ──")
        warnings.filterwarnings("ignore")
        csp_results = run_classical_models(X_csp, y)
    except Exception as e:
        print(f"  [!] CSP failed: {e}")
        X_csp = None

    # ── Riemannian features ───────────────────────────────────────────────
    print("\n[3/3] Extracting Riemannian features...")
    try:
        X_riem = riemannian_features(X)
        X_riem = np.nan_to_num(X_riem)
        print(f"      Riemannian shape: {X_riem.shape}")

        print("\n  ── Feature set: Riemannian ──")
        riem_results = run_classical_models(X_riem, y)
    except Exception as e:
        print(f"  [!] Riemannian failed: {e}")
        X_riem = None

    # ── CSP + Riemannian combined ─────────────────────────────────────────
    if X_csp is not None and X_riem is not None:
        print("\n  ── Feature set: CSP + Riemannian combined ──")
        X_combined = np.concatenate([X_csp, X_riem], axis=1)
        print(f"      Combined shape: {X_combined.shape}")
        combined_results = run_classical_models(X_combined, y)

    # ── Also try Riemannian + band_power (your current best) ─────────────
    if X_riem is not None:
        from features import band_power_features
        print("\n  ── Feature set: Riemannian + Band Power ──")
        X_bp = band_power_features(X)
        X_bp = np.nan_to_num(X_bp)
        X_super = np.concatenate([X_riem, X_bp], axis=1)
        print(f"      Shape: {X_super.shape}")
        super_results = run_classical_models(X_super, y)

    print("\n✅ Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    args = parser.parse_args()
    run_csp_riemannian(args.subject)