"""
data_fusion.py  (AGRI L2 label pairing, 多进程版)
=================================================
AGRI L1B FDI + L2 CLP/CTH 数据配对流水线的顶层调度器。

架构
----
  data_fusion.py   <- 本文件：调度、多进程、QC 图
  fusion_core.py   <- 纯数值工具（latlon_to_xyz 等）
  fusion_io.py     <- 文件读写（AGRI FDI/GEO / L2 CLP/CTH / HDF5 输出）
  fusion_config.py <- 质量控制阈值（可被环境变量覆盖）

AGRI L1B 与 L2 标签位于同一 2748×2748 网格上，无需空间匹配。
L1B FDI + GEO 文件与 L2 CLP/CTH 文件通过文件名时间戳配对。

多进程策略
----------
- 每个 AGRI L1B FDI 文件作为一个独立任务
- ProcessPoolExecutor：N 个 worker 并行，主进程调度
- 子进程之间无共享状态

用法
----
  python data_fusion.py --split train --day 20190401
  python data_fusion.py --split train --workers 8
"""
from __future__ import annotations
import argparse, csv, json, logging, sys, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import numpy as np

import config as cfg
import fusion_config as fc
from fusion_core import compute_tight_disk_mask
from fusion_io import (
    apply_quality_filter, find_day_folders,
    parse_agri_datetime,
    read_agri_scene,
    read_agri_l2_clp, read_agri_l2_cth,
    _find_matching_l2_file,
    write_fused_samples, write_full_disk_hdf5,
)
from sample_filters import get_patch_supervision_thresholds

log = logging.getLogger(__name__)

# compat alias used by main.py
_find_day_folders = find_day_folders


QC_DIAGNOSTIC_FIELDS = [
    "scene_id", "agri_file",
    "raw_clp_valid_px", "raw_cth_valid_px",
    "geo_ok_px",
    "reg_geo_ok_px",
    "final_clp_px", "final_cth_px",
]


def _json_safe(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _reset_qc_diagnostics(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["qc_gate_stats.csv", "qc_gate_stats.jsonl"]:
        path = out_dir / name
        if path.exists():
            path.unlink()


def _write_qc_diagnostics(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "qc_gate_stats.csv"
    jsonl_path = out_dir / "qc_gate_stats.jsonl"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QC_DIAGNOSTIC_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: _json_safe(row.get(k)) for k in QC_DIAGNOSTIC_FIELDS})
    with jsonl_path.open("a", encoding="utf-8") as f:
        for row in rows:
            payload = {k: _json_safe(row.get(k)) for k in QC_DIAGNOSTIC_FIELDS}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    log.info("QC diagnostics saved -> %s and %s", csv_path, jsonl_path)


def _unpack_scene_result(result):
    if len(result) == 3:
        ok, op, msg = result
        return ok, op, msg, None
    ok, op, msg, diag = result
    return ok, op, msg, diag


def _pair_one_scene(agri_file, out_path, mode, qc_diagnostics_enabled=False):
    """子进程任务：配对单个 AGRI L1B + L2 场景，返回 (ok, out_path, msg)。"""
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    agri_path = Path(agri_file)
    out = Path(out_path)
    diag_row = None
    try:
        agri_dt = parse_agri_datetime(agri_path.name)
        if agri_dt is None:
            return False, out_path, "Cannot parse AGRI datetime", diag_row

        # ── 读取 L1B FDI + GEO ──
        agri = read_agri_scene(agri_path)
        if agri is None:
            return False, out_path, "read_agri_scene None", diag_row

        # ── 保存完整圆盘经纬度（可视化用）──
        full_lat = agri["lat"].copy()
        full_lon = agri["lon"].copy()

        # ── 收紧 AGRI 圆盘边界 ──
        margin = float(getattr(fc, "AGRI_DISK_MARGIN_DEG", 5.0))
        tight_mask = np.ones(agri["lat"].shape, dtype=bool)
        if margin > 0:
            sub_lon = float(getattr(fc, "AGRI_SUB_LON", 105.0))
            tight_mask = compute_tight_disk_mask(agri["lat"], agri["lon"], margin, sub_lon=sub_lon)
            agri["lat"] = np.where(tight_mask, agri["lat"], np.nan)
            agri["lon"] = np.where(tight_mask, agri["lon"], np.nan)
            agri["VZA"] = np.where(tight_mask, agri["VZA"], np.nan)
            agri["SZA"] = np.where(tight_mask, agri["SZA"], np.nan)
            bt = agri["BT"]
            mask_3d = np.broadcast_to(tight_mask[..., np.newaxis], bt.shape)
            agri["BT"] = np.where(mask_3d, bt, np.nan)

        # ── 读取 L2 CLP ──
        clp_path = _find_matching_l2_file(agri_path, "CLP")
        if clp_path is None:
            return False, out_path, "No matching L2 CLP file", diag_row
        clp = read_agri_l2_clp(clp_path)
        if clp is None:
            return False, out_path, "read_agri_l2_clp None", diag_row

        # ── 读取 L2 CTH ──
        cth_path = _find_matching_l2_file(agri_path, "CTH")
        if cth_path is None:
            return False, out_path, "No matching L2 CTH file", diag_row
        cth = read_agri_l2_cth(cth_path)
        if cth is None:
            return False, out_path, "read_agri_l2_cth None", diag_row

        # ── 验证形状匹配 ──
        bt_shape = agri["BT"].shape[:2]
        if clp.shape != bt_shape or cth.shape != bt_shape:
            return False, out_path, (
                f"Shape mismatch: BT={bt_shape} CLP={clp.shape} CTH={cth.shape}"
            ), diag_row

        # ── L1+L2 时间完全对齐，MATCH_DT_MIN=0 表示完美配对 ──
        labels = {
            "CLP": clp.astype(np.float32),
            "CTH": cth.astype(np.float32),
            "MATCH_DT_MIN": np.zeros(bt_shape, dtype=np.float32),
            "_label_source": "agri_l2",
        }

        # ── 质量控制 ──
        diagnostics = None
        if qc_diagnostics_enabled:
            diagnostics = {
                "scene_id": agri_dt.strftime("%Y%m%d_%H%M%S"),
                "agri_file": agri_path.name,
            }
            raw_clp_valid = int(np.isfinite(clp).sum())
            raw_cth_valid = int(np.isfinite(cth).sum())

        labels = apply_quality_filter(agri, labels, diagnostics=diagnostics)

        if diagnostics is not None:
            diag_row = diagnostics.get("row", {})
            diag_row["raw_clp_valid_px"] = raw_clp_valid
            diag_row["raw_cth_valid_px"] = raw_cth_valid
            diag_row["scene_id"] = agri_dt.strftime("%Y%m%d_%H%M%S")
            diag_row["agri_file"] = agri_path.name

        # ── 场景级监督像素数检查 ──
        thresh = get_patch_supervision_thresholds(mode, tuple(cfg.PATCH_SIZE))
        n_clp = int(np.isfinite(labels["CLP"]).sum())
        n_cld = int((
            np.isfinite(labels["CLP"]) & (labels["CLP"] > 0) &
            np.isfinite(labels["CTH"])
        ).sum())
        if (n_clp < thresh["min_valid_label_pixels"] or
                n_cld < thresh["min_valid_cloudy_pixels"]):
            return False, out_path, (
                f"Too few: clp={n_clp}/{thresh['min_valid_label_pixels']} "
                f"cld={n_cld}/{thresh['min_valid_cloudy_pixels']}"
            ), diag_row

        # ── 写出 ──
        out.parent.mkdir(parents=True, exist_ok=True)
        if cfg.FUSION_OUTPUT_MODE == "samples_only":
            n_s = write_fused_samples(out, agri, labels, agri_dt, mode)
            _make_geo_figure(agri, labels, agri_dt,
                             out.with_name(out.stem + "_geo.png"),
                             full_lat=full_lat, full_lon=full_lon, tight_mask=tight_mask)
            return True, out_path, f"OK samples={n_s}", diag_row
        else:
            write_full_disk_hdf5(out, agri, labels, agri_dt)
            _make_geo_figure(agri, labels, agri_dt,
                             out.with_name(out.stem + "_geo.png"),
                             full_lat=full_lat, full_lon=full_lon, tight_mask=tight_mask)
            return True, out_path, "OK full_disk", diag_row

    except Exception:
        return False, out_path, f"Exception:\n{traceback.format_exc()}", diag_row


# ---------------------------------------------------------------------------
# 地理验证图（简化为单源版本）
# ---------------------------------------------------------------------------

def _draw_disk_outline(ax, lat, lon, sub_lon=104.7, **kwargs):
    """画出有效像元外轮廓（极角分箱近似凸包）。"""
    valid = np.isfinite(lat) & np.isfinite(lon)
    if valid.sum() < 3:
        return
    y = lat[valid]
    x_raw = lon[valid]
    x_rel = ((x_raw - sub_lon + 180.0) % 360.0) - 180.0

    center_lat = np.median(y)
    center_lon_rel = np.median(x_rel)
    dlon = x_rel - center_lon_rel
    angles = np.arctan2(y - center_lat, dlon)
    angles_2pi = np.where(angles < 0, angles + 2.0 * np.pi, angles)

    n_bins = 72
    bins = np.linspace(0, 2.0 * np.pi, n_bins + 1)
    hull_lat, hull_lon_rel = [], []
    for i in range(n_bins):
        mask = (angles_2pi >= bins[i]) & (angles_2pi < bins[i + 1])
        if not mask.any():
            continue
        dist = np.sqrt((y[mask] - center_lat) ** 2 + (x_rel[mask] - center_lon_rel) ** 2)
        idx = np.argmax(dist)
        hull_lat.append(y[mask][idx])
        hull_lon_rel.append(x_rel[mask][idx])

    if len(hull_lat) < 3:
        return
    hull_lat = np.array(hull_lat)
    hull_lon_rel = np.array(hull_lon_rel)
    hull_dlon = hull_lon_rel - center_lon_rel
    hull_angles = np.arctan2(hull_lat - center_lat, hull_dlon)
    hull_angles_2pi = np.where(hull_angles < 0, hull_angles + 2.0 * np.pi, hull_angles)
    order = np.argsort(hull_angles_2pi)
    plot_lon = np.append(hull_lon_rel[order], hull_lon_rel[order[0]])
    plot_lat = np.append(hull_lat[order], hull_lat[order[0]])
    ax.plot(plot_lon, plot_lat, **kwargs)


def _make_geo_figure(agri, labels, agri_dt, save_path,
                     full_lat=None, full_lon=None, tight_mask=None):
    """AGRI L2 监督覆盖验证图。"""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        sub_lon = float(getattr(fc, "AGRI_SUB_LON", 104.7))
        data_crs = ccrs.PlateCarree()
        map_crs = ccrs.PlateCarree(central_longitude=sub_lon)

        fig = plt.figure(figsize=(12, 11))
        ax = fig.add_subplot(1, 1, 1, projection=map_crs)

        ax.add_feature(cfeature.COASTLINE, lw=0.5, alpha=0.5, zorder=4)

        # 背景：完整 AGRI 全圆盘
        if full_lat is not None and full_lon is not None:
            valid_full = np.isfinite(full_lat) & np.isfinite(full_lon)
            if valid_full.any():
                y_f, x_f = full_lat[valid_full], full_lon[valid_full]
                step = max(1, len(y_f) // 4000)
                ax.scatter(x_f[::step], y_f[::step], s=0.25, alpha=0.35,
                           color="lightgrey", rasterized=True, zorder=1,
                           transform=data_crs, label="AGRI full disk")
                _draw_disk_outline(ax, full_lat, full_lon, sub_lon=sub_lon,
                                   color="lightgrey", lw=1.0, linestyle="--", alpha=0.5,
                                   label=None, transform=map_crs)

        # 保留区域轮廓
        lat, lon = agri["lat"], agri["lon"]
        valid_agri = np.isfinite(lat) & np.isfinite(lon)
        if valid_agri.any():
            _draw_disk_outline(ax, lat, lon, sub_lon=sub_lon,
                               color="royalblue", lw=2.2,
                               label="Retained region", transform=map_crs)

            # CLP 监督采样（网格中的有效点）
            y, x = lat[valid_agri], lon[valid_agri]
            step = max(1, len(y) // 3000)
            ax.scatter(x[::step], y[::step], s=0.8, alpha=0.7,
                       color="mediumseagreen", rasterized=True, zorder=2,
                       transform=data_crs, label="Valid AGRI pixels")

        ax.set_extent([sub_lon - 85.0, sub_lon + 85.0, -85.0, 85.0], crs=data_crs)

        gl = ax.gridlines(draw_labels=True, alpha=0.35, linestyle="--", linewidth=0.5)
        gl.top_labels = False
        gl.right_labels = False

        n_clp = int(np.isfinite(labels["CLP"]).sum())
        n_total = int(valid_agri.sum()) if valid_agri.any() else 1
        n_cth = int(np.isfinite(labels["CTH"]).sum())

        ax.set_title(
            f"AGRI L2 Supervision Coverage\n"
            f"{agri_dt:%Y-%m-%d %H:%M} UTC  |  "
            f"CLP: {n_clp}/{n_total} ({100.*n_clp/max(n_total,1):.1f}%)  |  "
            f"CTH: {n_cth}",
            fontsize=12, fontweight="bold")

        ax.legend(loc="upper left", fontsize=7, markerscale=2, ncol=1, framealpha=0.85)

        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info("Geo figure saved -> %s", save_path)
    except Exception as exc:
        log.warning("Geo figure failed for %s: %s",
                    agri_dt.strftime("%Y%m%d_%H%M%S") if agri_dt else "unknown", exc)


# ---------------------------------------------------------------------------
# 调度
# ---------------------------------------------------------------------------

def fuse_day(
    agri_day_dir: Path,
    out_dir: Path,
    mode: str = "train",
    overwrite: bool = False,
    n_workers: int = fc.N_FUSION_WORKERS,
    enable_qc_diagnostics: bool = fc.ENABLE_QC_DIAGNOSTICS,
    qc_diagnostics_dir: Path = Path(fc.QC_DIAGNOSTICS_DIR),
) -> int:
    """单日 L1B+L2 配对调度。"""
    agri_files = sorted([
        p for p in list(agri_day_dir.glob("*.HDF")) + list(agri_day_dir.glob("*.hdf"))
        if "_FDI-_" in p.name and not p.name.endswith(".db")
    ])
    log.info("Day %s | L1B FDI files=%d | workers=%d",
             agri_day_dir.name, len(agri_files), n_workers)

    if not agri_files:
        return 0

    tasks = []
    for agri_file in agri_files:
        agri_dt = parse_agri_datetime(agri_file.name)
        if agri_dt is None:
            continue
        out_name = f"AGRI_MYD06_{agri_dt:%Y%m%d_%H%M%S}.h5"
        out_path = out_dir / out_name
        if out_path.exists() and not overwrite:
            continue
        tasks.append((str(agri_file), str(out_path), mode, bool(enable_qc_diagnostics)))

    if not tasks:
        log.info("Day %s - no tasks", agri_day_dir.name)
        return 0

    log.info("Day %s - submitting %d tasks", agri_day_dir.name, len(tasks))
    success = 0
    diagnostic_rows = []

    if n_workers <= 1:
        for args in tasks:
            ok, op, msg, diag = _unpack_scene_result(_pair_one_scene(*args))
            if diag is not None:
                diagnostic_rows.append(diag)
            if ok:
                success += 1
            else:
                log.debug("Skip %s: %s", Path(args[1]).name, msg[:200])
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_pair_one_scene, *t): t for t in tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    ok, op, msg, diag = _unpack_scene_result(fut.result())
                except Exception as exc:
                    ok, op, msg, diag = False, task[1], str(exc), None
                if diag is not None:
                    diagnostic_rows.append(diag)
                if ok:
                    success += 1
                else:
                    log.debug("Skip %s: %s", Path(task[1]).name, msg[:200])

    log.info("Day %s - %d/%d ok", agri_day_dir.name, success, len(tasks))
    if enable_qc_diagnostics:
        _write_qc_diagnostics(diagnostic_rows, Path(qc_diagnostics_dir))
    return success


def fuse_day_compat(agri_day, modis_day, out_sub, overwrite=False, max_qc=3):
    """main.py stage_fuse 兼容包装器（modis_day 参数保留但不再使用）。"""
    parts = {p.lower() for p in out_sub.parts}
    mode = "val" if ("val" in parts or "valid" in parts) else ("test" if "test" in parts else "train")
    return fuse_day(agri_day, out_sub, mode=mode, overwrite=overwrite)


def _setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout,
    )


def main():
    _setup_logging(cfg.LOG_LEVEL)
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    choices=["train","val","test"], default="train")
    parser.add_argument("--day",      default=None)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--max_qc",  type=int, default=3)
    parser.add_argument("--workers", type=int, default=fc.N_FUSION_WORKERS)
    parser.add_argument("--enable-qc-diagnostics", action="store_true", default=None)
    parser.add_argument("--qc-diagnostics-dir", default=fc.QC_DIAGNOSTICS_DIR)
    args = parser.parse_args()

    split_out  = {"train":cfg.PAIRED_TRAIN_DIR,"val":cfg.PAIRED_VAL_DIR,"test":cfg.PAIRED_TEST_DIR}[args.split]
    dates      = {"train":cfg.TRAIN_DATES,"val":cfg.VAL_DATES,"test":cfg.TEST_DATES}[args.split]
    if args.day:
        dates = [args.day]

    agri_days  = find_day_folders(cfg.AGRI_ROOT, dates)

    total = 0
    qc_diag_enabled = (
        fc.ENABLE_QC_DIAGNOSTICS
        if args.enable_qc_diagnostics is None
        else args.enable_qc_diagnostics
    )
    qc_diag_dir = Path(args.qc_diagnostics_dir)
    if qc_diag_enabled:
        _reset_qc_diagnostics(qc_diag_dir)

    for agri_day in agri_days:
        out_sub = split_out / agri_day.name
        total += fuse_day(agri_day, out_sub,
                          mode=args.split, overwrite=args.overwrite,
                          n_workers=args.workers,
                          enable_qc_diagnostics=qc_diag_enabled,
                          qc_diagnostics_dir=qc_diag_dir)

    log.info("Pairing done - %d files total", total)


if __name__ == "__main__":
    main()
