"""
app.py
======
Flask backend for the Arabic BCI prediction interface.

Install dependencies:
    pip install flask --break-system-packages

Run:
    python app.py

Then open: http://localhost:5000
"""

import pickle
import numpy as np
import warnings
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

warnings.filterwarnings("ignore")

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    raise ImportError("Run: pip install mne --break-system-packages")

from scipy.signal import butter, filtfilt, iirnotch
from sklearn.feature_selection import SelectKBest, f_classif
from features import (
    riemannian_features, band_power_features,
    connectivity_features, adaptive_csp_features,
)
from config import CONFIG

app = Flask(__name__, static_folder="static")
MODELS_DIR = Path("models")

EMOTIV_CH_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

BANDS = {
    "broadband":   (0.5, 40.0),
    "delta_theta": (0.5,  8.0),
    "alpha":       (8.0, 13.0),
    "beta":        (13.0, 30.0),
    "gamma":       (30.0, 40.0),
    "alpha_beta":  (8.0,  30.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_edf_trial(edf_path, sfreq_target=256):
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    available = raw.ch_names
    eeg_picks = []
    for ch in EMOTIV_CH_NAMES:
        if ch in available:
            eeg_picks.append(ch)
        else:
            matches = [c for c in available if c.upper() == ch.upper()]
            if matches:
                eeg_picks.append(matches[0])
    if len(eeg_picks) < 10:
        raise ValueError(f"Only {len(eeg_picks)} EEG channels found")
    raw.pick_channels(eeg_picks, ordered=True)
    if abs(raw.info["sfreq"] - sfreq_target) > 1:
        raw.resample(sfreq_target, verbose=False)
    data = raw.get_data()
    if np.abs(data).max() < 1.0:
        data = data * 1e6
    return data


def preprocess_single_trial(raw):
    """Preprocess one raw EEG trial (n_ch, n_samples) → epoch (1, n_ch, n_samples).
    
    Always extracts the imagination window at seconds 10-16
    (after 5s relax + 5s observe), matching the training protocol exactly.
    """
    sfreq      = CONFIG["sfreq"]
    target_len = int(CONFIG["epoch_tmax"] * sfreq)  # 1536 samples
    t_start    = int((CONFIG["t_relax"] + CONFIG["t_observe"]) * sfreq)  # 2560
    t_end      = t_start + target_len  # 4096

    total_samples = raw.shape[1]

    if total_samples < t_end:
        raise ValueError(
            f"Trial too short: {total_samples/sfreq:.1f}s — "
            f"need at least {t_end/sfreq:.1f}s (5s relax + 5s observe + 6s imagine). "
            f"Make sure you record the full protocol."
        )

    ep = raw[:, t_start:t_end]

    # Bandpass
    nyq = sfreq / 2.0
    b, a = butter(4, [0.5/nyq, 40.0/nyq], btype="band")
    ep = filtfilt(b, a, ep, axis=1)

    # Notch 50Hz
    b, a = iirnotch(50.0, Q=30.0, fs=sfreq)
    ep = filtfilt(b, a, ep, axis=1)

    # CAR
    ep = ep - ep.mean(axis=0, keepdims=True)

    # Laplacian
    nb = {0:[1,2],1:[0,3],2:[0,3,4],3:[1,2,5],4:[2,3,5],5:[3,4,6],
          6:[5,7],7:[6,8],8:[7,9,10],9:[8,10,11],10:[8,9,12],
          11:[9,10,13],12:[10,11,13],13:[11,12]}
    r = ep.copy()
    for ch, ns in nb.items():
        if ch < ep.shape[0] and all(n < ep.shape[0] for n in ns):
            r[ch] = ep[ch] - ep[ns].mean(axis=0)
    ep = r

    # Baseline
    n_base = int(0.2 * sfreq)
    ep = ep - ep[:, :n_base].mean(axis=1, keepdims=True)

    # Z-score
    m = ep.mean()
    s = ep.std() + 1e-8
    ep = (ep - m) / s

    return ep[np.newaxis, :, :]   # (1, n_ch, n_samples)


def extract_features_inference(X, config):
    """
    Extract features for a SINGLE trial at inference time.

    CSP is intentionally excluded here — adaptive_csp_features needs
    multiple trials per class to fit spatial filters, so it produces a
    different feature count for a single trial vs the training set.
    This causes the StandardScaler dimension mismatch.

    Riemannian + BP + PLV are all trial-count independent and produce
    the same dimensions regardless of how many trials are passed in.
    The ANOVA selector (trained on the full feature set including CSP)
    will select the subset of features that overlap with what we provide.

    Note: accuracy may be slightly lower without CSP on single trials,
    but this is unavoidable for offline single-trial prediction.
    """
    sfreq    = CONFIG["sfreq"]
    cfg_type = config["type"]
    band     = config["band"]

    def _riem_bp_plv(Xf):
        fr = np.nan_to_num(riemannian_features(Xf))
        fb = np.nan_to_num(band_power_features(Xf))
        fp = np.nan_to_num(connectivity_features(Xf))
        return np.concatenate([fr, fb, fp], axis=1)

    if cfg_type == "multi":
        feats_list = []
        for lo, hi in BANDS.values():
            nyq = sfreq / 2.0
            b, a = butter(4, [max(lo,0.1)/nyq, min(hi,nyq-0.1)/nyq],
                          btype="band")
            Xf = filtfilt(b, a, X, axis=2)
            feats_list.append(_riem_bp_plv(Xf))
        return np.concatenate(feats_list, axis=1)
    else:
        lo, hi = BANDS[band]
        nyq = sfreq / 2.0
        b, a = butter(4, [max(lo,0.1)/nyq, min(hi,nyq-0.1)/nyq],
                      btype="band")
        Xf = filtfilt(b, a, X, axis=2)
        return _riem_bp_plv(Xf)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "interface.html")


@app.route("/models", methods=["GET"])
def list_models():
    """Return list of saved subject models."""
    MODELS_DIR.mkdir(exist_ok=True)
    models = []
    for pkl_file in sorted(MODELS_DIR.glob("*.pkl")):
        try:
            with open(pkl_file, "rb") as f:
                bundle = pickle.load(f)
            models.append({
                "id":           pkl_file.stem,
                "name":         bundle.get("name", pkl_file.stem),
                "n_classes":    bundle.get("n_classes", "?"),
                "cv_accuracy":  round(bundle.get("cv_accuracy", 0) * 100, 1),
                "band":         bundle.get("config", {}).get("band", "?"),
            })
        except Exception:
            pass
    return jsonify(models)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts:
        - model_id: name of the .pkl file (without extension)
        - file: uploaded EDF file
    Returns:
        - predicted_letter: Arabic letter string
        - predicted_index: integer class index
        - confidence: float 0-1
        - top3: list of {letter, index, probability}
        - all_probs: dict of all letter probabilities
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    model_id = request.form.get("model_id", "")
    model_path = MODELS_DIR / f"{model_id}.pkl"

    if not model_path.exists():
        return jsonify({"error": f"Model '{model_id}' not found"}), 404

    # Load model
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    scaler         = bundle["scaler"]
    selector       = bundle["selector"]
    clf            = bundle["clf"]
    config         = bundle["config"]
    arabic_letters = bundle["arabic_letters"]
    n_classes      = bundle["n_classes"]

    # Save uploaded file temporarily
    uploaded = request.files["file"]
    tmp_path = Path("/tmp/bci_trial.edf")
    uploaded.save(str(tmp_path))

    try:
        # Load and preprocess
        raw  = load_edf_trial(tmp_path)
        X    = preprocess_single_trial(raw)

        # Extract features — CSP excluded for single trial inference
        X_feat = extract_features_inference(X, config)

        # Apply saved scaler
        # The scaler was trained on Riem+BP+PLV+CSP features.
        # We only have Riem+BP+PLV here so we apply only the
        # matching columns using the scaler's learned mean/std.
        n_inference = X_feat.shape[1]
        n_train     = scaler.mean_.shape[0]

        if n_inference > n_train:
            # Trim to training size (shouldn't happen but be safe)
            X_feat = X_feat[:, :n_train]
            X_sc   = scaler.transform(X_feat)
        elif n_inference < n_train:
            # Pad with zeros for missing CSP columns
            X_pad  = np.zeros((X_feat.shape[0], n_train), dtype=np.float64)
            X_pad[:, :n_inference] = X_feat
            X_sc   = scaler.transform(X_pad)
        else:
            X_sc = scaler.transform(X_feat)

        # Apply ANOVA selector if present
        if selector is not None:
            X_sc = selector.transform(X_sc)

        # Predict
        probs     = clf.predict_proba(X_sc)[0]
        pred_idx  = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        # Top 3
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {
                "letter":      arabic_letters[i] if i < len(arabic_letters) else f"L{i}",
                "index":       int(i),
                "probability": float(probs[i]),
            }
            for i in top3_idx
        ]

        # All probabilities for alphabet display
        all_probs = {
            arabic_letters[i]: float(probs[i])
            for i in range(min(len(probs), len(arabic_letters)))
        }

        return jsonify({
            "predicted_letter": arabic_letters[pred_idx] if pred_idx < len(arabic_letters) else f"L{pred_idx}",
            "predicted_index":  pred_idx,
            "confidence":       confidence,
            "top3":             top3,
            "all_probs":        all_probs,
            "n_classes":        n_classes,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    print("\n" + "═"*50)
    print("  Arabic BCI — Prediction Interface")
    print("  Open: http://localhost:5000")
    print("═"*50 + "\n")
    app.run(debug=False, port=5000)