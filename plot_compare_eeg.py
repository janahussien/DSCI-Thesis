import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch
from pathlib import Path
from config import CONFIG

SFREQ    = CONFIG["sfreq"]
CH_NAMES = CONFIG["channel_names"]
CH       = 12  # O1

# Load
path = CONFIG["data_root"] / "S01" / "L01" / "S01_L01_T1.mat"
mat  = sio.loadmat(str(path), simplify_cells=True)
data = np.array(mat["EEG"]["Data"], dtype=np.float64)
if data.shape[0] > data.shape[1]:
    data = data.T  # (14, samples)

# Imagination window
onset  = int((CONFIG["t_relax"] + CONFIG["t_observe"]) * SFREQ)
offset = onset + int(CONFIG["epoch_tmax"] * SFREQ)

# Raw — DC removed for display only
raw = data[CH, onset:offset]
raw = raw - raw.mean()

# Preprocessed
epoch = data[:, onset:offset].copy()

# 1. Bandpass
lo, hi = CONFIG["bandpass"]
nyq = SFREQ / 2.0
b, a = butter(4, [lo/nyq, hi/nyq], btype="band")
epoch = filtfilt(b, a, epoch, axis=1)

# 2. Notch
for f0 in CONFIG["notch_freqs"]:
    b, a = iirnotch(f0, 50.0, SFREQ)
    epoch = filtfilt(b, a, epoch, axis=1)

# 3. CAR
epoch = epoch - epoch.mean(axis=0, keepdims=True)

# 4. Surface Laplacian
neighbors = {
    0:[1,2], 1:[0,3], 2:[0,3,4], 3:[1,2,5],
    4:[2,3,5], 5:[3,4,6], 6:[5,7], 7:[6,8],
    8:[7,9,10], 9:[8,10,11], 10:[8,9,12],
    11:[9,10,13], 12:[10,11,13], 13:[11,12],
}
lap = epoch.copy()
for ch, nbrs in neighbors.items():
    lap[ch] = epoch[ch] - epoch[nbrs].mean(axis=0)
epoch = lap

# 5. Baseline correction
n_base = int(0.2 * SFREQ)
epoch  = epoch - epoch[:, :n_base].mean(axis=1, keepdims=True)

# 6. Z-score
epoch = (epoch - epoch.mean(axis=1, keepdims=True)) / (epoch.std(axis=1, keepdims=True) + 1e-8)

clean = epoch[CH]
t     = np.arange(len(raw)) / SFREQ

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), facecolor="#0D1117")
fig.subplots_adjust(hspace=0.45, left=0.08, right=0.97, top=0.90, bottom=0.08)

for ax in [ax1, ax2]:
    ax.set_facecolor("#161B22")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363D")
    ax.tick_params(colors="#8B949E", labelsize=10)
    ax.grid(color="#30363D", lw=0.6, alpha=0.8)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Time (s)", color="#8B949E", fontsize=10)

# Raw
ax1.plot(t, raw, color="#FF7B72", lw=1.0)
ax1.set_title("RAW  —  Channel O1  (DC offset removed for display only)",
              color="#FF7B72", fontsize=12, fontweight="bold", pad=8)
ax1.set_ylabel("Amplitude (uV)", color="#8B949E", fontsize=10)
ax1.text(0.99, 0.95, f"RMS = {np.sqrt(np.mean(raw**2)):.1f} uV",
         transform=ax1.transAxes, ha="right", va="top",
         color="#FF7B72", fontsize=9, fontfamily="monospace")

# Preprocessed
ax2.plot(t, clean, color="#3FB950", lw=1.0)
ax2.set_title("PREPROCESSED  —  Channel O1  (bandpass + notch + CAR + Laplacian + baseline + z-score)",
              color="#3FB950", fontsize=12, fontweight="bold", pad=8)
ax2.set_ylabel("Amplitude (z-score)", color="#8B949E", fontsize=10)
ax2.text(0.99, 0.95, f"RMS = {np.sqrt(np.mean(clean**2)):.3f}",
         transform=ax2.transAxes, ha="right", va="top",
         color="#3FB950", fontsize=9, fontfamily="monospace")

fig.suptitle("EEG Signal Before vs After Preprocessing  .  S01  .  Letter 1  .  Trial 1  .  Channel O1",
             color="#E6EDF3", fontsize=13, fontweight="bold")

Path("plots").mkdir(exist_ok=True)
fig.savefig("plots/compare_eeg.png", dpi=150, bbox_inches="tight", facecolor="#0D1117")
plt.close(fig)
print("Saved -> plots/compare_eeg.png")