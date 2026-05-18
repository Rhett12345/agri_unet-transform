"""
Compare model-retrieved CLP/CTH vs AGRI L2 CLP/CTH vs MODIS MYD06.

Scene: 2019-05-05 04:00-04:30 UTC (FY-4A full-disk + Aqua over Australia).

Data sources:
  - Model retrieval:  AGRI → inference.py (2748x2748, 4km)
  - AGRI L2 CLP/CTH:   NetCDF  (2748x2748, 4km)
  - MODIS MYD06+MYD03: HDF4    (2030x1354, 1km swath)

Usage:
    conda run -n cloudunet python geo_match/comparison/compare_cloud_products.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree
import netCDF4
from pyhdf.SD import SD, SDC

# ── add parent so we can import the coordinate helpers from agri_viz ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agri_viz import linecolumn_to_lonlat, lonlat_to_linecolumn

# ══════════════════════════════════════════════════════════════════════
# Paths — MODIS 0430 UTC over Australia, AGRI 0400 UTC
# ══════════════════════════════════════════════════════════════════════
DATE = "20190505"
AGRI_TIME = "040000"
AGRI_TIME_END = "04001459"
MODIS_HHMM = "0430"

AGRI_L1_BASE = Path("/data/Data_yuq/FY4A") / DATE
L2_CLP_BASE = Path("/data/Data_yuq/FY4A_L2/CLP") / DATE
L2_CTH_BASE = Path("/data/Data_yuq/FY4A_L2/CTH") / DATE
RETRIEVAL_DIR = Path("/data/Data_yuq/unet_workdir/retrieval")
MYD06_BASE = Path("/data/Data_yuq/MYD06") / DATE
MYD03_BASE = Path("/data/Data_yuq/MYD03") / DATE

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ── file discovery ──
fdi_candidates = sorted(AGRI_L1_BASE.glob(f"*FDI*{DATE}{AGRI_TIME}*.HDF"))
geo_candidates = sorted(AGRI_L1_BASE.glob(f"*GEO*{DATE}{AGRI_TIME}*.HDF"))
clp_l2_candidates = sorted(L2_CLP_BASE.glob(f"*CLP*{DATE}{AGRI_TIME}*.NC"))
cth_l2_candidates = sorted(L2_CTH_BASE.glob(f"*CTH*{DATE}{AGRI_TIME}*.NC"))
ret_candidates = sorted(RETRIEVAL_DIR.glob(f"*{DATE}{AGRI_TIME}*_retrieval.npz"))
myd06_candidates = sorted(MYD06_BASE.glob(f"MYD06_L2.A2019125.{MODIS_HHMM}*.hdf"))
myd03_candidates = sorted(MYD03_BASE.glob(f"MYD03.A2019125.{MODIS_HHMM}*.hdf"))

print("=== File discovery ===")
for label, cands in [
    ("AGRI FDI", fdi_candidates), ("AGRI GEO", geo_candidates),
    ("L2 CLP", clp_l2_candidates), ("L2 CTH", cth_l2_candidates),
    ("Model retrieval", ret_candidates),
    ("MYD06", myd06_candidates), ("MYD03", myd03_candidates),
]:
    status = f"OK: {cands[0].name}" if cands else "MISSING"
    print(f"  {label:20s}: {status}")

assert all([fdi_candidates, clp_l2_candidates, cth_l2_candidates,
            ret_candidates, myd06_candidates, myd03_candidates]), \
    "Missing required data files"

ret_path = ret_candidates[0]
clp_l2_path = clp_l2_candidates[0]
cth_l2_path = cth_l2_candidates[0]
myd06_path = myd06_candidates[0]
myd03_path = myd03_candidates[0]

# ══════════════════════════════════════════════════════════════════════
# 1. Model retrieval
# ══════════════════════════════════════════════════════════════════════
print("\n=== Model retrieval ===")
ret = np.load(ret_path)
model_lat = ret["latitude"]
model_lon = ret["longitude"]
model_clp = ret["CLP_pred"].astype(float)
model_cth = ret["CTH_pred"].copy()

model_clp[model_clp < 0] = np.nan
print(f"  CLP: {dict(zip(['clear','water','ice'], np.unique(model_clp[~np.isnan(model_clp)], return_counts=True)[1]))}")
print(f"  CTH: valid={np.isfinite(model_cth).sum():,}, range=[{np.nanmin(model_cth):.0f},{np.nanmax(model_cth):.0f}]")

# ══════════════════════════════════════════════════════════════════════
# 2. AGRI L2 CLP (no DQF filter — DQF=3 uniformly in this product)
# ══════════════════════════════════════════════════════════════════════
print("\n=== AGRI L2 CLP ===")
nc = netCDF4.Dataset(clp_l2_path)
nc.set_auto_maskandscale(False)
l2_clp_raw = nc.variables["CLP"][:].astype(np.int16)
nc.close()

# Map: 0=Clear,1=Water,2=Supercooled,3=Mixed,4=Ice,5=Uncertain,126=Space,127=Fill
# →  0=Clear, 1=Water, 2=Ice
l2_clp = np.full(l2_clp_raw.shape, np.nan, dtype=float)
# Valid cloud phase values
l2_clp[l2_clp_raw == 0] = 0    # Clear
l2_clp[l2_clp_raw == 1] = 1    # Water
l2_clp[l2_clp_raw == 2] = 1    # Supercooled → Water
l2_clp[l2_clp_raw == 3] = 2    # Mixed → Ice
l2_clp[l2_clp_raw == 4] = 2    # Ice
# 5=Uncertain, 126=Space, 127=Fill → already NaN
print(f"  L2 CLP: {dict(zip(['clear','water','ice'], np.unique(l2_clp[~np.isnan(l2_clp)], return_counts=True)[1]))}")

# ══════════════════════════════════════════════════════════════════════
# 3. AGRI L2 CTH
# ══════════════════════════════════════════════════════════════════════
print("\n=== AGRI L2 CTH ===")
nc = netCDF4.Dataset(cth_l2_path)
nc.set_auto_maskandscale(False)
l2_cth = nc.variables["CTH"][:].astype(np.float32)
nc.close()
# Valid range [1, 20000], FillValue=-999
l2_cth[(l2_cth < 1.0) | (l2_cth > 20000.0)] = np.nan
print(f"  L2 CTH: valid={np.isfinite(l2_cth).sum():,}, range=[{np.nanmin(l2_cth):.0f},{np.nanmax(l2_cth):.0f}]")

# ══════════════════════════════════════════════════════════════════════
# 4. MODIS MYD06 + MYD03
# ══════════════════════════════════════════════════════════════════════
print("\n=== MODIS ===")
sd06 = SD(str(myd06_path), SDC.READ)
sd03 = SD(str(myd03_path), SDC.READ)

modis_clp_raw = sd06.select("Cloud_Phase_Infrared_1km")[:].astype(np.int16)
modis_cth_raw = sd06.select("cloud_top_height_1km")[:].astype(np.float32)
modis_lat = sd03.select("Latitude")[:].astype(np.float32)
modis_lon = sd03.select("Longitude")[:].astype(np.float32)
sd06.end(); sd03.end()

# CLP: 0=clear,1=water,2=ice,3=mixed,6=undetermined,127=fill
modis_clp = np.full(modis_clp_raw.shape, np.nan, dtype=float)
modis_clp[modis_clp_raw == 0] = 0
modis_clp[modis_clp_raw == 1] = 1
modis_clp[modis_clp_raw == 2] = 2
# CTH: FillValue=-999, valid [0, 18000]
modis_cth = modis_cth_raw.copy()
modis_cth[(modis_cth < 0) | (modis_cth > 18000)] = np.nan

modis_lat_ok = np.isfinite(modis_lat) & (modis_lat >= -90) & (modis_lat <= 90)
print(f"  MODIS region: lat=[{modis_lat[modis_lat_ok].min():.1f},{modis_lat[modis_lat_ok].max():.1f}]")
print(f"                lon=[{modis_lon[modis_lat_ok].min():.1f},{modis_lon[modis_lat_ok].max():.1f}]")
print(f"  MODIS CLP: {dict(zip(['clear','water','ice'], np.unique(modis_clp[~np.isnan(modis_clp)], return_counts=True)[1]))}")
print(f"  MODIS CTH: valid={np.isfinite(modis_cth).sum():,}")

# ══════════════════════════════════════════════════════════════════════
# 5. Spatial matching: MODIS → AGRI grid
# ══════════════════════════════════════════════════════════════════════
print("\n=== Spatial matching (MODIS → AGRI grid) ===")
modis_lat_f = modis_lat.ravel()
modis_lon_f = modis_lon.ravel()
modis_clp_f = modis_clp.ravel()
modis_cth_f = modis_cth.ravel()
ok = (np.isfinite(modis_lat_f) & np.isfinite(modis_lon_f) &
      np.isfinite(modis_clp_f))
ok_idx = np.where(ok)[0]
print(f"  MODIS valid CLP pixels: {ok.sum():,}")

# Build KDTree on AGRI grid
agri_lat_f = model_lat.ravel()
agri_lon_f = model_lon.ravel()
agri_valid = np.isfinite(agri_lat_f) & np.isfinite(agri_lon_f)
agri_valid_idx = np.where(agri_valid)[0]
agri_pts = np.column_stack([np.radians(agri_lat_f[agri_valid_idx]),
                             np.radians(agri_lon_f[agri_valid_idx])])
tree = cKDTree(agri_pts)

# MODIS points to match
modis_pts = np.column_stack([np.radians(modis_lat_f[ok_idx]),
                              np.radians(modis_lon_f[ok_idx])])

modis_on_agri_clp = np.full(len(agri_lat_f), np.nan)
modis_on_agri_cth = np.full(len(agri_lat_f), np.nan)

# ~3km = ~0.027 deg at equator, use 0.03 rad tolerance
MAX_DIST_RAD = 0.03
BS = 50000
for start in range(0, len(ok_idx), BS):
    end = min(start + BS, len(ok_idx))
    dists, idxs = tree.query(modis_pts[start:end], k=1, distance_upper_bound=MAX_DIST_RAD)
    keep = dists < MAX_DIST_RAD
    if keep.any():
        gi = agri_valid_idx[idxs[keep]]
        mi = ok_idx[start:end][keep]
        modis_on_agri_clp[gi] = modis_clp_f[mi]
        modis_on_agri_cth[gi] = modis_cth_f[mi]
    if (start // BS) % 10 == 0:
        print(f"  progress: {start}/{len(ok_idx)}")

modis_on_agri_clp = modis_on_agri_clp.reshape(2748, 2748)
modis_on_agri_cth = modis_on_agri_cth.reshape(2748, 2748)
n_clp = np.isfinite(modis_on_agri_clp).sum()
n_cth = np.isfinite(modis_on_agri_cth).sum()
print(f"  MODIS→AGRI CLP: {n_clp:,} pixels  |  CTH: {n_cth:,} pixels")

# ══════════════════════════════════════════════════════════════════════
# 6. Statistics on overlapping region
# ══════════════════════════════════════════════════════════════════════
print("\n=== Statistics ===")

# CLP — three-way common mask
clp_3way = (np.isfinite(model_clp) & np.isfinite(l2_clp) &
            np.isfinite(modis_on_agri_clp))
print(f"  3-way common CLP pixels: {clp_3way.sum():,}")

if clp_3way.sum() > 100:
    oa_m_l2 = (model_clp[clp_3way] == l2_clp[clp_3way]).mean() * 100
    oa_m_md = (model_clp[clp_3way] == modis_on_agri_clp[clp_3way]).mean() * 100
    oa_l2_md = (l2_clp[clp_3way] == modis_on_agri_clp[clp_3way]).mean() * 100
    print(f"  CLP OA  Model vs L2:    {oa_m_l2:.1f}%")
    print(f"  CLP OA  Model vs MODIS: {oa_m_md:.1f}%")
    print(f"  CLP OA  L2 vs MODIS:    {oa_l2_md:.1f}%")

# CTH pairwise
for label, mask in [
    ("Model vs L2", np.isfinite(model_cth) & np.isfinite(l2_cth)),
    ("Model vs MODIS", np.isfinite(model_cth) & np.isfinite(modis_on_agri_cth)),
    ("L2 vs MODIS", np.isfinite(l2_cth) & np.isfinite(modis_on_agri_cth)),
]:
    if mask.sum() < 50:
        print(f"  CTH {label}: <50 pixels, skipping")
        continue
    a = model_cth[mask] if "Model" in label.split(" vs ")[0] else l2_cth[mask]
    b = l2_cth[mask] if "L2" in label.split(" vs ")[1] else modis_on_agri_cth[mask]
    diff = a - b
    bias = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))
    corr = np.corrcoef(a, b)[0, 1]
    print(f"  CTH {label}: N={mask.sum():,}  bias={bias:.0f}m  RMSE={rmse:.0f}m  R={corr:.3f}")

# ══════════════════════════════════════════════════════════════════════
# 7. Visualization
# ══════════════════════════════════════════════════════════════════════
print("\n=== Generating plots ===")

CLP_CMAP = matplotlib.colors.ListedColormap(["#555555", "#4ECDC4", "#FF6B6B"])
CLP_NORM = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)

# Zoom to MODIS swath region for detail plots
modis_lat_vals = modis_lat[modis_lat_ok]
modis_lon_vals = modis_lon[modis_lat_ok]
pad = 3.0
detail_extent = [modis_lon_vals.min() - pad, modis_lon_vals.max() + pad,
                 modis_lat_vals.min() - pad, modis_lat_vals.max() + pad]

# ── Figure 1: Full-disk overview with MODIS swath outline ──
print("  [1/5] Full-disk overview...")
fig = plt.figure(figsize=(18, 10))
_PC_CRS = ccrs.PlateCarree()

# Model CLP on full disk
ax = fig.add_subplot(231, projection=_PC_CRS)
ax.set_title("Model CLP", fontsize=11, fontweight="bold")
ax.imshow(model_clp, origin="upper", cmap=CLP_CMAP, norm=CLP_NORM,
          extent=[-180, 180, -90, 90], interpolation="nearest", aspect="auto")
ax.coastlines(linewidth=0.5)
# Outline MODIS swath
ax.plot([modis_lon_vals.min(), modis_lon_vals.max(), modis_lon_vals.max(), modis_lon_vals.min(), modis_lon_vals.min()],
        [modis_lat_vals.min(), modis_lat_vals.min(), modis_lat_vals.max(), modis_lat_vals.max(), modis_lat_vals.min()],
        'r-', linewidth=1.5, label='MODIS swath')
ax.legend(fontsize=7, loc='lower left')

ax = fig.add_subplot(232, projection=_PC_CRS)
ax.set_title("AGRI L2 CLP", fontsize=11, fontweight="bold")
ax.imshow(l2_clp, origin="upper", cmap=CLP_CMAP, norm=CLP_NORM,
          extent=[-180, 180, -90, 90], interpolation="nearest", aspect="auto")
ax.coastlines(linewidth=0.5)

ax = fig.add_subplot(233, projection=_PC_CRS)
ax.set_title("MODIS CLP (swath)", fontsize=11, fontweight="bold")
ax.scatter(modis_lon[::5, ::5], modis_lat[::5, ::5],
           c=modis_clp[::5, ::5], cmap=CLP_CMAP, norm=CLP_NORM,
           s=0.5, transform=_PC_CRS)
ax.coastlines(linewidth=0.5)
ax.set_extent(detail_extent)

# CTH on full disk
for idx, (data, title) in enumerate([
    (model_cth, "Model CTH"), (l2_cth, "AGRI L2 CTH"),
    (modis_on_agri_cth, "MODIS CTH (on AGRI grid)"),
], start=4):
    ax = fig.add_subplot(2, 3, idx, projection=_PC_CRS)
    ax.set_title(title, fontsize=11, fontweight="bold")
    im = ax.imshow(data, origin="upper", cmap="viridis", vmin=0, vmax=16000,
                   extent=[-180, 180, -90, 90], interpolation="nearest", aspect="auto")
    ax.coastlines(linewidth=0.5)
    if idx == 6:
        plt.colorbar(im, ax=ax, fraction=0.04, label="m")

fig.suptitle(f"Cloud Products — {DATE} {AGRI_TIME} UTC  +  MODIS {MODIS_HHMM} UTC",
             fontsize=13, fontweight="bold")
fig.savefig(OUT_DIR / "01_full_disk_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figure 2: Detail zoom over MODIS swath ──
print("  [2/5] Detail zoom over swath...")
fig, axes = plt.subplots(2, 3, figsize=(21, 12))

for ax_idx, (data, title) in enumerate([
    (model_clp, "Model CLP"), (l2_clp, "AGRI L2 CLP"), (modis_on_agri_clp, "MODIS CLP (→AGRI)"),
]):
    ax = axes[0, ax_idx]
    ax.set_title(title, fontsize=11, fontweight="bold")
    im = ax.imshow(data, origin="upper", cmap=CLP_CMAP, norm=CLP_NORM,
                   interpolation="nearest", aspect="equal")
    ax.set_xlim(detail_extent[0] / 0.04 + 1373.5, detail_extent[1] / 0.04 + 1373.5)
    ax.set_ylim(detail_extent[2] / 0.04 + 1373.5, detail_extent[3] / 0.04 + 1373.5)
    ax.set_xticks([]); ax.set_yticks([])

for ax_idx, (data, title) in enumerate([
    (model_cth, "Model CTH"), (l2_cth, "AGRI L2 CTH"), (modis_on_agri_cth, "MODIS CTH (→AGRI)"),
]):
    ax = axes[1, ax_idx]
    ax.set_title(title, fontsize=11, fontweight="bold")
    im = ax.imshow(data, origin="upper", cmap="viridis", vmin=0, vmax=16000,
                   interpolation="nearest", aspect="equal")
    ax.set_xlim(detail_extent[0] / 0.04 + 1373.5, detail_extent[1] / 0.04 + 1373.5)
    ax.set_ylim(detail_extent[2] / 0.04 + 1373.5, detail_extent[3] / 0.04 + 1373.5)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m")

fig.suptitle(f"Detail over MODIS Swath — {DATE} {AGRI_TIME}+{MODIS_HHMM} UTC",
             fontsize=13, fontweight="bold")
fig.savefig(OUT_DIR / "02_detail_over_swath.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figure 3: CLP agreement maps and confusion matrices ──
print("  [3/5] CLP agreement...")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
agree_cmap = matplotlib.colors.ListedColormap(["#2ca02c", "#d62728", "#aaaaaa"])
agree_norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)

for ax_idx, (d1, d2, title) in enumerate([
    (model_clp, l2_clp, "Model vs L2 CLP"),
    (model_clp, modis_on_agri_clp, "Model vs MODIS CLP"),
    (l2_clp, modis_on_agri_clp, "L2 vs MODIS CLP"),
]):
    ax = axes[0, ax_idx]
    both = np.isfinite(d1) & np.isfinite(d2)
    agree_map = np.full_like(d1, 2.0)
    agree_map[both] = 1.0 - (d1[both] == d2[both]).astype(float)
    im = ax.imshow(agree_map, origin="upper", cmap=agree_cmap, norm=agree_norm,
                   interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    if both.sum() > 10:
        oa = (d1[both] == d2[both]).mean() * 100
        ax.set_xlabel(f"OA={oa:.1f}%  N={both.sum():,}", fontsize=9)

    # Confusion matrix in row 2
    ax = axes[1, ax_idx]
    if both.sum() < 10:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        continue
    from sklearn.metrics import confusion_matrix
    labels = [0, 1, 2]
    cm = confusion_matrix(d1[both].astype(int), d2[both].astype(int), labels=labels)
    # Normalize by row
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1e-9)
    im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.1%})", ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Clear", "Water", "Ice"])
    ax.set_yticklabels(["Clear", "Water", "Ice"])
    ax.set_ylabel("Reference" if ax_idx == 0 else "")
    ax.set_xlabel("Prediction" if ax_idx == 1 else "")
    ax.set_title(f"Confusion Matrix\n({title})", fontsize=9)

fig.suptitle(f"CLP Agreement — {DATE} {AGRI_TIME}+{MODIS_HHMM} UTC",
             fontsize=13, fontweight="bold")
fig.savefig(OUT_DIR / "03_clp_agreement.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figure 4: CTH scatter + histograms ──
print("  [4/5] CTH scatter...")
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

for ax_idx, (d1, d2, xlab, ylab) in enumerate([
    (l2_cth, model_cth, "AGRI L2 CTH (m)", "Model CTH (m)"),
    (modis_on_agri_cth, model_cth, "MODIS CTH (m)", "Model CTH (m)"),
    (modis_on_agri_cth, l2_cth, "MODIS CTH (m)", "AGRI L2 CTH (m)"),
]):
    ax = axes[ax_idx]
    mask = np.isfinite(d1) & np.isfinite(d2)
    x = d1[mask].ravel()
    y = d2[mask].ravel()
    if len(x) > 15000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x), 15000, replace=False)
        x, y = x[idx], y[idx]

    # Hexbin for density
    hb = ax.hexbin(x, y, gridsize=60, cmap="YlOrRd", bins="log", mincnt=1)
    plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="log10(count)")
    ax.plot([0, 18000], [0, 18000], "k--", linewidth=0.8)
    ax.set_xlabel(xlab, fontsize=11)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_xlim(0, 18000); ax.set_ylim(0, 18000)

    if len(x) > 10:
        bias = np.mean(y - x)
        rmse = np.sqrt(np.mean((y - x) ** 2))
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f"N={len(x):,}\nBias={bias:.0f}m\nRMSE={rmse:.0f}m\nR={r:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

fig.suptitle(f"CTH Comparison — {DATE} {AGRI_TIME}+{MODIS_HHMM} UTC",
             fontsize=13, fontweight="bold")
fig.savefig(OUT_DIR / "04_cth_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figure 5: CTH difference maps ──
print("  [5/5] CTH difference maps...")
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
DMAX = 4000

for ax_idx, (d1, d2, title) in enumerate([
    (model_cth, l2_cth, "Model CTH - L2 CTH (m)"),
    (model_cth, modis_on_agri_cth, "Model CTH - MODIS CTH (m)"),
    (l2_cth, modis_on_agri_cth, "L2 CTH - MODIS CTH (m)"),
]):
    ax = axes[ax_idx]
    diff = d1 - d2
    diff_c = np.clip(diff, -DMAX, DMAX)
    im = ax.imshow(diff_c, origin="upper", cmap="RdBu_r", vmin=-DMAX, vmax=DMAX,
                   interpolation="nearest", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    mask = np.isfinite(d1) & np.isfinite(d2)
    if mask.sum() > 50:
        dvals = diff[mask]
        ax.set_xlabel(f"bias={np.mean(dvals):.0f}m  std={np.std(dvals):.0f}m  N={mask.sum():,}", fontsize=9)

fig.suptitle(f"CTH Difference Maps — {DATE} {AGRI_TIME}+{MODIS_HHMM} UTC",
             fontsize=13, fontweight="bold")
fig.savefig(OUT_DIR / "05_cth_diff_maps.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\n=== Done. All outputs in {OUT_DIR} ===")
print("Files: 01_full_disk_overview.png, 02_detail_over_swath.png, "
      "03_clp_agreement.png, 04_cth_scatter.png, 05_cth_diff_maps.png")
