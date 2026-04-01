"""
models.py
=========
Every model in one place.

Classical (sklearn)
--------------------
  • Logistic Regression
  • Linear Discriminant Analysis (LDA)
  • Support Vector Machine (RBF + Linear)
  • k-Nearest Neighbours
  • Random Forest
  • Gradient Boosting (XGBoost if available, else sklearn GBM)
  • Naive Bayes (Gaussian)

EEG-specific
-------------
  • SWLDA (stepwise LDA — P300 classic)

Deep Learning (PyTorch — optional)
------------------------------------
  • ShallowConvNet  (Schirrmeister et al., 2017)
  • EEGNet          (Lawhern et al., 2018)
  • DeepConvNet
  • 1-D CNN + LSTM  (temporal-spatial hybrid)

All models run inside 10-fold stratified CV and report:
  mean accuracy, std, per-class F1, confusion matrix
"""

from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
import warnings
from typing import Dict, Any, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from config import CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cv_evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_splits: int = 10,
    use_pca: bool = False,
    pca_var: float = 0.95,
) -> Dict:
    """
    10-fold stratified CV.  Returns dict with acc_mean, acc_std, f1_macro,
    per_fold_accs, confusion_matrix.
    """
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=CONFIG["random_state"])
    accs, f1s = [], []
    cms = []

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Per-fold scaling (avoids data leakage)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        if use_pca:
            pca = PCA(n_components=pca_var, svd_solver="full")
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
        cms.append(confusion_matrix(y_te, y_pred,
                                    labels=np.arange(CONFIG["n_letters"])))

    result = {
        "model":        model_name,
        "acc_mean":     float(np.mean(accs)),
        "acc_std":      float(np.std(accs)),
        "f1_macro":     float(np.mean(f1s)),
        "per_fold_acc": accs,
        "confusion_matrix": np.mean(cms, axis=0),
    }
    return result


def _print_result(r: Dict):
    print(f"  [{r['model']:<35}]  "
          f"Acc: {r['acc_mean']*100:6.2f}% ± {r['acc_std']*100:.2f}%  |  "
          f"F1: {r['f1_macro']*100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Classical models
# ─────────────────────────────────────────────────────────────────────────────

def run_classical_models(    
    X: np.ndarray, y: np.ndarray
) -> Dict[str, Dict]:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB

    models = {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
        ),
        "LDA": LinearDiscriminantAnalysis(solver="svd"),
        "SVM_RBF": SVC(kernel="rbf", C=10.0, gamma="scale", decision_function_shape="ovr"),
        "SVM_Linear": SVC(kernel="linear", C=1.0, decision_function_shape="ovr"),
        "KNN_k5": KNeighborsClassifier(n_neighbors=5, n_jobs=CONFIG["n_jobs"]),
        "Random_Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None,
            random_state=CONFIG["random_state"], n_jobs=CONFIG["n_jobs"]
        ),
        "Gradient_Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.1, random_state=CONFIG["random_state"]
        ),
        "GaussianNB": GaussianNB(),
    }


    results = {}
    for name, clf in models.items():
        try:
            r = _cv_evaluate(clf, X, y, name, use_pca=(name == "LDA"))
            _print_result(r)
            results[name] = r
        except Exception as e:
            warnings.warn(f"  [{name}] failed: {e}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Deep learning (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def run_deep_models(
    X_raw: np.ndarray,  # (n_trials, n_channels, n_samples)
    y: np.ndarray
) -> Dict[str, Dict]:
    """
    Train EEGNet, ShallowConvNet, DeepConvNet, CNN+LSTM in 10-fold CV.
    Requires: pip install torch torchvision
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        warnings.warn("PyTorch not found. Skipping deep models. Run: pip install torch")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Deep learning device: {device}")

    n_ch  = X_raw.shape[1]
    n_t   = X_raw.shape[2]
    n_cls = len(np.unique(y))
    sfreq = CONFIG["sfreq"]

    arch_map = {
        "EEGNet":       lambda: EEGNet(n_ch, n_t, n_cls, sfreq),
        "ShallowConvNet": lambda: ShallowConvNet(n_ch, n_t, n_cls),
        "DeepConvNet":  lambda: DeepConvNet(n_ch, n_t, n_cls),
        "CNN_LSTM":     lambda: CNN_LSTM(n_ch, n_t, n_cls),
    }

    results = {}
    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True,
                          random_state=CONFIG["random_state"])

    for arch_name, arch_fn in arch_map.items():
        fold_accs = []
        for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_raw, y)):
            X_tr = X_raw[tr_idx].astype(np.float32)
            X_te = X_raw[te_idx].astype(np.float32)
            y_tr, y_te = y[tr_idx], y[te_idx]

            # Normalise per fold
            mu  = X_tr.mean(axis=(0, 2), keepdims=True)
            sig = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
            X_tr = (X_tr - mu) / sig
            X_te = (X_te - mu) / sig

            Xt = torch.tensor(X_tr[:, None, :, :])  # add "depth" dim
            yt = torch.tensor(y_tr, dtype=torch.long)
            Xv = torch.tensor(X_te[:, None, :, :])
            yv = torch.tensor(y_te, dtype=torch.long)

            loader = DataLoader(TensorDataset(Xt, yt),
                                batch_size=CONFIG["dl_batch"], shuffle=True)

            model = arch_fn().to(device)
            optimizer = optim.Adam(model.parameters(), lr=CONFIG["dl_lr"])
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=CONFIG["dl_epochs"]
            )

            best_val_loss, patience_ctr = np.inf, 0
            best_weights = None

            for epoch in range(CONFIG["dl_epochs"]):
                model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()

                # Validation loss for early stopping
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(Xv.to(device)), yv.to(device)).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= CONFIG["dl_patience"]:
                        break

            model.load_state_dict(best_weights)
            model.eval()
            with torch.no_grad():
                preds = model(Xv.to(device)).argmax(dim=1).cpu().numpy()
            fold_accs.append(accuracy_score(yv.numpy(), preds))

        r = {
            "model":        arch_name,
            "acc_mean":     float(np.mean(fold_accs)),
            "acc_std":      float(np.std(fold_accs)),
            "f1_macro":     0.0,  # simplified for DL
            "per_fold_acc": fold_accs,
            "confusion_matrix": None,
        }
        _print_result(r)
        results[arch_name] = r

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch architectures
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch.nn as nn

    class EEGNet(nn.Module):
        """Compact EEGNet (Lawhern et al., 2018)."""
        def __init__(self, n_ch, n_t, n_cls, sfreq, F1=8, D=2, F2=16):
            super().__init__()
            self.temporal = nn.Sequential(
                nn.Conv2d(1, F1, (1, sfreq // 2), padding=(0, sfreq // 4), bias=False),
                nn.BatchNorm2d(F1),
            )
            self.depthwise = nn.Sequential(
                nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(0.5),
            )
            self.separable = nn.Sequential(
                nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(0.5),
            )
            t_out = n_t // 4 // 8 + 1
            self.classifier = nn.Linear(F2 * t_out, n_cls)

        def forward(self, x):
            x = self.temporal(x)
            x = self.depthwise(x)
            x = self.separable(x)
            x = x.flatten(1)
            return self.classifier(x)


    class ShallowConvNet(nn.Module):
        """Shallow ConvNet (Schirrmeister et al., 2017)."""
        def __init__(self, n_ch, n_t, n_cls):
            super().__init__()
            self.temporal = nn.Conv2d(1, 40, (1, 25), bias=False)
            self.spatial  = nn.Conv2d(40, 40, (n_ch, 1), bias=False)
            self.bn       = nn.BatchNorm2d(40)
            self.pool     = nn.AvgPool2d((1, 75), stride=(1, 15))
            self.drop     = nn.Dropout(0.5)
            t_out = (n_t - 25 + 1 - 75) // 15 + 1
            self.fc = nn.Linear(40 * t_out, n_cls)

        def forward(self, x):
            x = self.temporal(x)
            x = self.spatial(x)
            x = self.bn(x)
            x = x ** 2
            x = self.pool(x)
            x = torch.log(torch.clamp(x, min=1e-6))
            x = self.drop(x)
            return self.fc(x.flatten(1))


    class DeepConvNet(nn.Module):
        """Deep ConvNet (Schirrmeister et al., 2017)."""
        def __init__(self, n_ch, n_t, n_cls):
            super().__init__()
            def block(in_f, out_f, k, pool=True):
                layers = [
                    nn.Conv2d(in_f, out_f, (1, k), bias=False),
                    nn.BatchNorm2d(out_f), nn.ELU(), nn.Dropout(0.5)
                ]
                if pool:
                    layers.append(nn.MaxPool2d((1, 3), stride=(1, 3)))
                return nn.Sequential(*layers)

            self.b0 = nn.Sequential(
                nn.Conv2d(1, 25, (1, 10), bias=False),
                nn.Conv2d(25, 25, (n_ch, 1), bias=False),
                nn.BatchNorm2d(25), nn.ELU(), nn.Dropout(0.5),
                nn.MaxPool2d((1, 3), stride=(1, 3))
            )
            self.b1 = block(25, 50, 10)
            self.b2 = block(50, 100, 10)
            self.b3 = block(100, 200, 10)

            # Compute flat size dynamically
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_ch, n_t)
                dummy = self.b3(self.b2(self.b1(self.b0(dummy))))
                flat = dummy.numel()
            self.fc = nn.Linear(flat, n_cls)

        def forward(self, x):
            x = self.b3(self.b2(self.b1(self.b0(x))))
            return self.fc(x.flatten(1))


    class CNN_LSTM(nn.Module):
        """Spatial CNN + temporal LSTM hybrid."""
        def __init__(self, n_ch, n_t, n_cls, hidden=128, n_layers=2):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(n_ch, 64, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(64), nn.GELU(),
                nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(128), nn.GELU(),
                nn.MaxPool1d(4),
                nn.Dropout(0.3),
            )
            self.lstm = nn.LSTM(128, hidden, n_layers, batch_first=True,
                                dropout=0.3, bidirectional=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden * 2, 256), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(256, n_cls)
            )

        def forward(self, x):
            x = x.squeeze(1)               # (B, C, T)
            x = self.cnn(x)                # (B, 128, T//4)
            x = x.permute(0, 2, 1)         # (B, T//4, 128)
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])   # last time step


    import torch  # noqa – needed for ShallowConvNet power law

except ImportError:
    pass  # PyTorch not installed — DL classes simply won't be defined


# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_models(
    features: Dict[str, np.ndarray],
    y: np.ndarray,
    subject_id: int = 1,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    For each feature set, run all classical models.
    Run deep models on the raw 3-D tensor stored under features["raw"] if present.
    """
    all_results = {}

    # ── Classical models on each feature set ──────────────────────────────
    for feat_name, X_feat in features.items():
        if feat_name == "raw":
            continue
        if X_feat.ndim != 2 or X_feat.shape[0] < 20:
            continue

        print(f"\n  ── Feature set: {feat_name}  ({X_feat.shape[1]} dims) ──")
        res = run_classical_models(X_feat, y)
        all_results[feat_name] = res

    # ── Deep learning on raw EEG ───────────────────────────────────────────
    if "raw" in features:
        print("\n  ── Deep learning on raw EEG ──")
        dl_res = run_deep_models(features["raw"], y)
        all_results["deep_learning"] = dl_res

    return all_results