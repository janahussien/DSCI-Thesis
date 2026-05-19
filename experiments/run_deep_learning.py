"""
run_deep_learning.py
====================
Deep learning only — with proper window size and data augmentation.

Key differences from previous run:
  - Shorter epoch window (2s instead of 6s) → 512 samples, better for CNNs
  - Data augmentation (3x training data via noise, dropout, time shift)
  - AdamW optimiser + label smoothing + cosine LR schedule
  - MPS (Apple Silicon) support

Architectures:
  1. EEGNet          (Lawhern et al. 2018)
  2. ShallowConvNet  (Schirrmeister et al. 2017)
  3. DeepConvNet     (Schirrmeister et al. 2017)
  4. CNN + LSTM      (spatial-temporal hybrid)
  5. EEG Transformer (patch-based self-attention)

Run:
    python run_deep_learning.py --subject 1
    python run_deep_learning.py --subject 1 --epochs 100
    python run_deep_learning.py --subject 1 --window 3   # 3s window
"""

import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from config import CONFIG
from preprocessing import load_subject_data, preprocess_pipeline
from utils import print_banner


# ─────────────────────────────────────────────────────────────────────────────
# Data augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_eeg(X: np.ndarray, y: np.ndarray, factor: int = 3) -> tuple:
    """
    Augment EEG training data to multiply dataset size by `factor`.

    Techniques:
      1. Gaussian noise injection  — adds small random noise scaled to signal
      2. Channel dropout           — zeros out 1 random channel per trial
      3. Time shift                — rolls signal ±50 samples along time axis

    Only applied to TRAINING data inside each CV fold — never to test data.
    """
    X_aug, y_aug = [X], [y]

    for _ in range(factor - 1):
        X_new = X.copy()

        # 1. Gaussian noise (~5% of signal std)
        noise_std = 0.05 * X_new.std(axis=(1, 2), keepdims=True)
        X_new += np.random.randn(*X_new.shape) * noise_std

        # 2. Random channel dropout (zero 1 channel per trial)
        drop_ch = np.random.randint(0, X_new.shape[1], size=len(X_new))
        for i, ch in enumerate(drop_ch):
            X_new[i, ch] = 0.0

        # 3. Random time shift ±50 samples
        shift = np.random.randint(-50, 50)
        X_new = np.roll(X_new, shift, axis=2)

        X_aug.append(X_new)
        y_aug.append(y)

    return np.concatenate(X_aug), np.concatenate(y_aug)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch architectures
# ─────────────────────────────────────────────────────────────────────────────

def build_models(n_ch, n_t, n_cls, sfreq, device):
    """Return dict of model name → constructor lambda."""
    import torch
    import torch.nn as nn

    # ── EEGNet ────────────────────────────────────────────────────────────
    class EEGNet(nn.Module):
        def __init__(self, F1=8, D=2, F2=16):
            super().__init__()
            half = sfreq // 2
            pad  = half // 2
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, half), padding=(0, pad), bias=False),
                nn.BatchNorm2d(F1),
                nn.Conv2d(F1, F1*D, (n_ch, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1*D),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(0.5),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8),
                          groups=F1*D, bias=False),
                nn.Conv2d(F1*D, F2, 1, bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(0.5),
            )
            with torch.no_grad():
                d = torch.zeros(1, 1, n_ch, n_t)
                flat = self.block2(self.block1(d)).numel()
            self.fc = nn.Linear(flat, n_cls)

        def forward(self, x):
            return self.fc(self.block2(self.block1(x)).flatten(1))

    # ── ShallowConvNet ────────────────────────────────────────────────────
    class ShallowConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal = nn.Conv2d(1,  40, (1, 25), bias=False)
            self.spatial  = nn.Conv2d(40, 40, (n_ch, 1), bias=False)
            self.bn       = nn.BatchNorm2d(40)
            self.pool     = nn.AvgPool2d((1, 75), stride=(1, 15))
            self.drop     = nn.Dropout(0.5)
            with torch.no_grad():
                d    = torch.zeros(1, 1, n_ch, n_t)
                d    = self.pool(self.spatial(self.temporal(d)))
                flat = d.numel()
            self.fc = nn.Linear(flat, n_cls)

        def forward(self, x):
            x = self.spatial(self.temporal(x))
            x = self.bn(x)
            x = torch.log(torch.clamp(self.pool(x ** 2), min=1e-6))
            return self.fc(self.drop(x).flatten(1))

    # ── DeepConvNet ───────────────────────────────────────────────────────
    class DeepConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            def blk(i, o, k):
                return nn.Sequential(
                    nn.Conv2d(i, o, (1, k), bias=False),
                    nn.BatchNorm2d(o), nn.ELU(), nn.Dropout(0.5),
                    nn.MaxPool2d((1, 3), stride=(1, 3))
                )
            self.b0 = nn.Sequential(
                nn.Conv2d(1,  25, (1, 10), bias=False),
                nn.Conv2d(25, 25, (n_ch, 1), bias=False),
                nn.BatchNorm2d(25), nn.ELU(), nn.Dropout(0.5),
                nn.MaxPool2d((1, 3), stride=(1, 3))
            )
            self.b1 = blk(25,  50, 10)
            self.b2 = blk(50, 100, 10)
            self.b3 = blk(100,200, 10)
            with torch.no_grad():
                try:
                    d    = torch.zeros(1, 1, n_ch, n_t)
                    flat = self.b3(self.b2(self.b1(self.b0(d)))).numel()
                except Exception:
                    flat = 200
            self.fc = nn.Linear(flat, n_cls)

        def forward(self, x):
            return self.fc(self.b3(self.b2(self.b1(self.b0(x)))).flatten(1))

    # ── CNN + LSTM ────────────────────────────────────────────────────────
    class CNN_LSTM(nn.Module):
        def __init__(self, hidden=128, n_layers=2):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(n_ch, 64,  7, padding=3, bias=False),
                nn.BatchNorm1d(64),  nn.GELU(),
                nn.Conv1d(64,  128, 5, padding=2, bias=False),
                nn.BatchNorm1d(128), nn.GELU(),
                nn.MaxPool1d(4), nn.Dropout(0.3),
            )
            self.lstm = nn.LSTM(128, hidden, n_layers,
                                batch_first=True, dropout=0.3,
                                bidirectional=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden*2, 256), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(256, n_cls)
            )

        def forward(self, x):
            x = self.cnn(x.squeeze(1))
            out, _ = self.lstm(x.permute(0, 2, 1))
            return self.fc(out[:, -1])

    # ── EEG Transformer ───────────────────────────────────────────────────
    class EEGTransformer(nn.Module):
        def __init__(self, patch=64, d=64, heads=4, layers=2, drop=0.3):
            super().__init__()
            self.patch = patch
            n_patches   = n_t // patch
            self.embed  = nn.Sequential(
                nn.Linear(n_ch * patch, d), nn.LayerNorm(d)
            )
            self.pos = nn.Parameter(torch.zeros(1, n_patches, d))
            nn.init.trunc_normal_(self.pos, std=0.02)
            enc = nn.TransformerEncoderLayer(
                d_model=d, nhead=heads, dim_feedforward=d*4,
                dropout=drop, batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(
                enc, num_layers=layers, norm=nn.LayerNorm(d)
            )
            self.head = nn.Sequential(
                nn.Linear(d, d//2), nn.GELU(),
                nn.Dropout(drop), nn.Linear(d//2, n_cls)
            )

        def forward(self, x):
            x = x.squeeze(1)                          # (B, C, T)
            B, C, T = x.shape
            p = self.patch
            n = T // p
            x = x[:, :, :n*p].reshape(B, C, n, p)
            x = x.permute(0, 2, 1, 3).reshape(B, n, C*p)
            x = self.embed(x) + self.pos
            x = self.transformer(x)
            return self.head(x.mean(dim=1))

    return {
        "EEGNet":         EEGNet,
        "ShallowConvNet": ShallowConvNet,
        "DeepConvNet":    DeepConvNet,
        "CNN_LSTM":       CNN_LSTM,
        "EEGTransformer": EEGTransformer,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_and_eval(arch_name, arch_cls, X_raw, y,
                   epochs, device, augment_factor=3):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    skf = StratifiedKFold(
        n_splits=CONFIG["cv_folds"], shuffle=True,
        random_state=CONFIG["random_state"]
    )
    fold_accs, fold_f1s = [], []

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_raw, y)):
        X_tr = X_raw[tr_idx].astype(np.float32)
        X_te = X_raw[te_idx].astype(np.float32)
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Normalise (fit on train only)
        mu  = X_tr.mean(axis=(0, 2), keepdims=True)
        sig = X_tr.std( axis=(0, 2), keepdims=True) + 1e-8
        X_tr = (X_tr - mu) / sig
        X_te = (X_te - mu) / sig

        # Augment training data only
        X_tr_aug, y_tr_aug = augment_eeg(X_tr, y_tr, factor=augment_factor)

        # Tensors
        Xt = torch.tensor(X_tr_aug[:, None]).to(device)
        yt = torch.tensor(y_tr_aug, dtype=torch.long).to(device)
        Xv = torch.tensor(X_te[:, None]).to(device)
        yv = torch.tensor(y_te, dtype=torch.long)

        loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=CONFIG["dl_batch"], shuffle=True
        )

        model     = arch_cls().to(device)
        optimizer = optim.AdamW(model.parameters(),
                                lr=CONFIG["dl_lr"], weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        best_loss    = np.inf
        best_weights = None
        patience_ctr = 0

        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(
                    model(Xv.to(device)),
                    yv.to(device)
                ).item()

            if val_loss < best_loss:
                best_loss    = val_loss
                best_weights = {k: v.clone()
                                for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= CONFIG["dl_patience"]:
                    break

        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            preds = model(Xv.to(device)).argmax(1).cpu().numpy()

        fold_accs.append(accuracy_score(y_te, preds))
        fold_f1s.append(f1_score(y_te, preds, average="macro",
                                  zero_division=0))

        print(f"    Fold {fold_i+1:2d}/10  "
              f"acc={fold_accs[-1]*100:.1f}%  "
              f"(stopped @ epoch {epoch+1})",
              end="\r")

    print()  # newline after fold progress
    return {
        "model":        arch_name,
        "acc_mean":     float(np.mean(fold_accs)),
        "acc_std":      float(np.std(fold_accs)),
        "f1_macro":     float(np.mean(fold_f1s)),
        "per_fold_acc": fold_accs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",  type=int,   default=1)
    parser.add_argument("--epochs",   type=int,   default=CONFIG["dl_epochs"])
    parser.add_argument("--window",   type=float, default=2.0,
                        help="Epoch length in seconds (default: 2.0)")
    parser.add_argument("--augment",  type=int,   default=3,
                        help="Augmentation factor (default: 3x)")
    args = parser.parse_args()

    # ── Override epoch window ─────────────────────────────────────────────
    original_tmax = CONFIG["epoch_tmax"]
    CONFIG["epoch_tmax"] = args.window
    print(f"\n  Using {args.window}s window "
          f"→ {int(args.window * CONFIG['sfreq'])} timepoints")

    print_banner(f"Deep Learning — Subject S{args.subject:02d}")

    # ── PyTorch setup ─────────────────────────────────────────────────────
    try:
        import torch
    except ImportError:
        print("PyTorch not found. Run: pip install torch")
        return

    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs}  (+ early stopping)")
    print(f"  Augment: {args.augment}x training data")

    # ── Load & preprocess ─────────────────────────────────────────────────
    print("\n[1/2] Loading & preprocessing...")
    raw = load_subject_data(args.subject, CONFIG["data_root"])
    X, y = preprocess_pipeline(raw)
    n_ch, n_t = X.shape[1], X.shape[2]
    n_cls = len(np.unique(y))
    print(f"      Shape: {X.shape}  →  {n_ch}ch × {n_t}t  |  {n_cls} classes")
    print(f"      Training trials per fold: ~{int(len(y)*0.9*args.augment)}"
          f" (after {args.augment}x augmentation)")

    # ── Train all architectures ───────────────────────────────────────────
    print("\n[2/2] Training deep learning models...")
    print("  " + "─"*50)

    arch_map = build_models(n_ch, n_t, n_cls, CONFIG["sfreq"], device)
    results  = {}

    for arch_name, arch_cls in arch_map.items():
        print(f"\n  ▶ {arch_name}")
        try:
            r = train_and_eval(
                arch_name, arch_cls, X, y,
                epochs=args.epochs,
                device=device,
                augment_factor=args.augment,
            )
            results[arch_name] = r
        except Exception as e:
            print(f"  [!] {arch_name} failed: {e}")
            import traceback; traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  DEEP LEARNING RESULTS SUMMARY")
    print("═"*60)
    print(f"  {'Model':<22} {'Accuracy':>10}  {'F1 Macro':>10}")
    print("  " + "─"*46)

    for name, r in sorted(results.items(),
                           key=lambda x: -x[1]["acc_mean"]):
        marker = " ← best" if r["acc_mean"] == max(
            v["acc_mean"] for v in results.values()) else ""
        print(f"  {name:<22} "
              f"{r['acc_mean']*100:>9.2f}%  "
              f"{r['f1_macro']*100:>9.2f}%"
              f"{marker}")

    print("  " + "─"*46)
    print(f"  Chance level:          {100/n_cls:>9.2f}%")
    print(f"  Classical best:        {'81.58%':>10}  (Riem+BP+PLV, LDA)")
    print(f"  Paper baseline:        {'74.80%':>10}")

    # Restore original config
    CONFIG["epoch_tmax"] = original_tmax

    # Save
    out_dir = CONFIG["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"S{args.subject:02d}_deeplearning_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({k: {kk: (vv if not isinstance(vv, float)
                             else round(vv, 4))
                        for kk, vv in v.items()}
                   for k, v in results.items()}, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()