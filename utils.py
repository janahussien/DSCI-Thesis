"""
utils.py
========
Printing, saving results, and optional plotting.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from config import CONFIG


def print_banner(text: str):
    width = 60
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def save_results(results: Dict, subject_id: int):
    """Save results as JSON to results_dir."""
    out_dir = CONFIG["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"S{subject_id:02d}_{ts}.json"

    # Convert numpy types for JSON serialisation
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_convert(results), f, indent=2)

    print(f"\n  Results saved → {out_path}")
    _print_summary_table(results)


def _print_summary_table(results: Dict):
    """Print a quick best-model-per-feature-set summary."""
    print("\n" + "─" * 75)
    print(f"  {'Feature Set':<25}  {'Best Model':<30}  {'Accuracy':>10}")
    print("─" * 75)

    for feat_name, model_dict in results.items():
        if not isinstance(model_dict, dict):
            continue
        best_name, best_acc = "", 0.0
        for model_name, r in model_dict.items():
            if isinstance(r, dict) and r.get("acc_mean", 0) > best_acc:
                best_acc = r["acc_mean"]
                best_name = model_name
        if best_name:
            print(f"  {feat_name:<25}  {best_name:<30}  {best_acc*100:>9.2f}%")
    print("─" * 75)


def plot_results(results: Dict, subject_id: int, save: bool = True):
    """
    Generate:
      1. Bar chart — accuracy by model & feature set
      2. Confusion matrix for best model/feature combo
    Requires: matplotlib
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [warn] matplotlib not found. Skipping plots.")
        return

    out_dir = CONFIG["results_dir"] / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Accuracy bar chart ─────────────────────────────────────────────────
    feat_sets, model_names, accs, stds = [], [], [], []
    for feat_name, model_dict in results.items():
        if not isinstance(model_dict, dict):
            continue
        for model_name, r in model_dict.items():
            if isinstance(r, dict) and "acc_mean" in r:
                feat_sets.append(feat_name)
                model_names.append(f"{feat_name}\n{model_name}")
                accs.append(r["acc_mean"] * 100)
                stds.append(r["acc_std"] * 100)

    if not accs:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(accs) * 0.5), 6))
    x = np.arange(len(accs))
    colors = plt.cm.tab20(np.linspace(0, 1, len(set(feat_sets))))
    color_map = {fs: colors[i] for i, fs in enumerate(sorted(set(feat_sets)))}
    bar_colors = [color_map[fs] for fs in feat_sets]

    ax.bar(x, accs, yerr=stds, color=bar_colors, alpha=0.85,
           capsize=4, edgecolor="white", linewidth=0.5)
    ax.axhline(100 / CONFIG["n_letters"], color="red", linestyle="--",
               alpha=0.6, label=f"Chance ({100/CONFIG['n_letters']:.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Subject S{subject_id:02d} — Classification Accuracy by Model & Feature Set")
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()

    if save:
        fig.savefig(out_dir / f"S{subject_id:02d}_accuracy.png", dpi=150)
        print(f"  Plot saved → {out_dir / f'S{subject_id:02d}_accuracy.png'}")
    plt.close(fig)

    # ── Confusion matrix for best combo ───────────────────────────────────
    best_acc, best_cm = 0.0, None
    for feat_name, model_dict in results.items():
        if not isinstance(model_dict, dict):
            continue
        for r in model_dict.values():
            if isinstance(r, dict) and r.get("acc_mean", 0) > best_acc:
                if r.get("confusion_matrix") is not None:
                    best_acc = r["acc_mean"]
                    best_cm  = np.array(r["confusion_matrix"])

    if best_cm is not None:
        fig, ax = plt.subplots(figsize=(12, 10))
        cm_norm = best_cm / (best_cm.sum(axis=1, keepdims=True) + 1e-10)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Normalised count")
        ax.set_xlabel("Predicted letter")
        ax.set_ylabel("True letter")
        ax.set_title(f"S{subject_id:02d} — Confusion Matrix (best model, acc={best_acc*100:.1f}%)")
        plt.tight_layout()
        if save:
            fig.savefig(out_dir / f"S{subject_id:02d}_confusion.png", dpi=150)
        plt.close(fig)