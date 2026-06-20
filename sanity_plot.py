import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.fetch import fetch_lightcurve
from tools.detrend import detrend

# KIC 11446443 = Kepler-1 (TrES-2b), a well-studied hot Jupiter
KEPID = 11446443

print(f"Fetching KIC {KEPID}...")
lc = fetch_lightcurve(KEPID)
print(f"  {len(lc)} cadences across {lc.time.min().value:.1f} – {lc.time.max().value:.1f} BKJD")

print("Detrending...")
lc_flat = detrend(lc)

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

axes[0].plot(lc.time.value, lc.flux.value, lw=0.3, color="steelblue", alpha=0.7)
axes[0].set_ylabel("Normalized flux")
axes[0].set_title(f"KIC {KEPID} (Kepler-1 / TrES-2b) — raw")

axes[1].plot(lc_flat.time.value, lc_flat.flux.value, lw=0.3, color="darkorange", alpha=0.7)
axes[1].set_ylabel("Detrended flux")
axes[1].set_xlabel("Time (BKJD)")
axes[1].set_title("Detrended (Savitzky-Golay, window=401)")

fig.tight_layout()
out = f"data/sanity_{KEPID}.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
