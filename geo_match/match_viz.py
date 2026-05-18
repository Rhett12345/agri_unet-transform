"""
FY4A L2 vs MODIS MYD06 地理匹配与差异可视化工具
==================================================
功能：
  1. 读取 FY4A L2 NC 文件（CLP/CTH），通过官方行列号↔经纬度公式获取坐标
  2. 读取 MODIS MYD06 HDF4 文件（Cloud_Phase_Infrared / Cloud_Top_Height，热红外波段）
     + MYD03 HDF4 文件（Latitude / Longitude，1km 精确坐标）
  3. 将两者投影到同一经纬度网格，找出地理重叠区域
  4. 对重叠区域的 CLP / CTH 进行差异对比可视化

依赖：
    pip install numpy matplotlib cartopy scipy netCDF4 pyhdf

用法（命令行快速验证）：
    python geo_match_viz.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import griddata
import netCDF4 as nc

# pyhdf 读 HDF4（MODIS）
from pyhdf.SD import SD, SDC

# ─────────────────────────────────────────────────────────────────
# FY4A 坐标转换常量
# ─────────────────────────────────────────────────────────────────
_RES_PARAMS = {
    500:  dict(COFF=10991.5, CFAC=81865099, LOFF=10991.5, LFAC=81865099),
    1000: dict(COFF=5495.5,  CFAC=40932549, LOFF=5495.5,  LFAC=40932549),
    2000: dict(COFF=2747.5,  CFAC=20466274, LOFF=2747.5,  LFAC=20466274),
    4000: dict(COFF=1373.5,  CFAC=10233137, LOFF=1373.5,  LFAC=10233137),
}
_EA    = 6378.137
_EB    = 6356.7523
_H     = 42164.0
_LAM_D = 104.7
_PC    = ccrs.PlateCarree()


# ═══════════════════════════════════════════════════════════════════
# 坐标转换
# ═══════════════════════════════════════════════════════════════════

def linecolumn_to_lonlat(l: np.ndarray,
                         c: np.ndarray,
                         resolution: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
    """FY4A 标称行列号 → 地理经纬度，地球外像素返回 NaN。"""
    p = _RES_PARAMS[resolution]
    COFF, CFAC, LOFF, LFAC = p['COFF'], p['CFAC'], p['LOFF'], p['LFAC']
    l = np.asarray(l, dtype=float)
    c = np.asarray(c, dtype=float)

    x = np.deg2rad((c - COFF) / (2 ** -16 * CFAC))
    y = np.deg2rad((l - LOFF) / (2 ** -16 * LFAC))

    cos_x, cos_y = np.cos(x), np.cos(y)
    sin_x, sin_y = np.sin(x), np.sin(y)

    disc = ((_H * cos_x * cos_y)**2
            - (cos_y**2 + (_EA**2 / _EB**2) * sin_y**2) * (_H**2 - _EA**2))
    valid     = disc >= 0
    disc_safe = np.where(valid, disc, 0.0)

    sd  = np.sqrt(disc_safe)
    sn  = (_H * cos_x * cos_y - sd) / (cos_y**2 + (_EA**2 / _EB**2) * sin_y**2)
    s1  = _H - sn * cos_x * cos_y
    s2  = sn * sin_x * cos_y
    s3  = -sn * sin_y
    sxy = np.sqrt(s1**2 + s2**2)

    lon = np.rad2deg(np.arctan(s2 / s1)) + _LAM_D
    lat = np.rad2deg(np.arctan((_EA**2 / _EB**2) * (s3 / sxy)))
    lon = np.where(valid, lon, np.nan)
    lat = np.where(valid, lat, np.nan)
    return lon, lat


def _fy4a_full_lonlat(shape: Tuple[int, int],
                      resolution: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
    """为整幅 FY4A 图像计算每像素经纬度。"""
    rows, cols = shape
    l_arr, c_arr = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    return linecolumn_to_lonlat(l_arr, c_arr, resolution)


# ═══════════════════════════════════════════════════════════════════
# FY4A L2 NC 读取
# ═══════════════════════════════════════════════════════════════════

def read_fy4a_l2_nc(nc_path: str,
                    var_name: str,
                    resolution: int = 4000
                    ) -> Dict[str, np.ndarray]:
    """
    读取 FY4A L2 NC 文件中的指定变量，并计算其经纬度坐标网格。
    """
    with nc.Dataset(nc_path, 'r') as ds:
        var = ds.variables[var_name]

        # ★ 关闭自动掩码，避免 uint8 遇到负填充值时崩溃
        var.set_auto_mask(False)

        raw  = var[:]        # 获取原始 ndarray，保留原数据类型（如 uint8）
        fill = getattr(var, '_FillValue', None) or getattr(var, 'FillValue', None)
        attrs = {k: getattr(var, k) for k in var.ncattrs()}

    # 转成 (行, 列) 顺序并转换为 float32
    data = raw.T.copy().astype(np.float32)

    # 手动处理填充值
    if fill is not None:
        # 将 fill 值转换为与 raw 相同的 dtype，确保能正确匹配数据中的实际填充值
        fill_val = np.array(fill).astype(raw.dtype)
        data[data == fill_val] = np.nan

    print(f"[info] 计算 FY4A 经纬度网格（{resolution}m）...")
    lon, lat = _fy4a_full_lonlat(data.shape, resolution)

    return {'data': data, 'lon': lon, 'lat': lat, 'attrs': attrs}

# ═══════════════════════════════════════════════════════════════════
# MODIS HDF4 读取
# ═══════════════════════════════════════════════════════════════════

def _read_hdf4_sds(hdf_path: str, sds_name: str) -> Tuple[np.ndarray, dict]:
    """读取 HDF4 文件中的单个 SDS，自动应用 scale/offset，FillValue→NaN。"""
    hdf  = SD(hdf_path, SDC.READ)
    sds  = hdf.select(sds_name)
    data = sds.get().astype(np.float32)
    attr = sds.attributes()
    hdf.end()

    fill   = attr.get('_FillValue', None)
    scale  = attr.get('scale_factor', 1.0)
    offset = attr.get('add_offset',  0.0)

    if fill is not None:
        mask = data == float(fill)
    else:
        mask = np.zeros_like(data, dtype=bool)

    data = data * float(scale) + float(offset)
    data[mask] = np.nan
    return data, attr


def read_modis_myd06(myd06_path: str,
                     myd03_path: str
                     ) -> Dict[str, object]:
    """
    读取 MYD06 L2 文件中的热红外 CTP / CTH / CLP，以及 MYD03 的 1km 经纬度。

    热红外优先选择的 SDS：
      - 云相态 : Cloud_Phase_Infrared        (5km, IR 8.5/11μm)
      - 云顶高 : Cloud_Top_Height            (5km, IR 反演)
      - 云顶压 : Cloud_Top_Pressure          (5km, 备用)
    坐标     : MYD03 Latitude / Longitude    (1km，需降采样到 5km)

    Returns
    -------
    dict，含：
      'clp'         : Cloud_Phase_Infrared 物理值（5km 格点，NaN=无效）
      'cth'         : Cloud_Top_Height [m]（5km 格点）
      'ctp'         : Cloud_Top_Pressure [hPa]（5km 格点）
      'lon'         : 5km 经度（406×270）
      'lat'         : 5km 纬度（406×270）
      'clp_attrs'   : CLP 属性
      'cth_attrs'   : CTH 属性
    """
    print(f"[info] 读取 MYD06: {Path(myd06_path).name}")
    clp, clp_attrs = _read_hdf4_sds(myd06_path, 'Cloud_Phase_Infrared')
    cth, cth_attrs = _read_hdf4_sds(myd06_path, 'Cloud_Top_Height')
    ctp, ctp_attrs = _read_hdf4_sds(myd06_path, 'Cloud_Top_Pressure')

    print(f"[info] 读取 MYD03: {Path(myd03_path).name}")
    # MYD03 提供 1km 经纬度 (2030, 1354)
    # MYD06 5km 格点 (406, 270) 对应 MYD03 中每 5 行 5 列取中心点（offset=2）
    hdf03 = SD(myd03_path, SDC.READ)
    lat1km = hdf03.select('Latitude').get().astype(np.float32)
    lon1km = hdf03.select('Longitude').get().astype(np.float32)
    hdf03.end()

    # MYD06 5km 的 5km Lat/Lon 采样规则：从第3行/列开始，每5步
    # Cell_Along_Swath_Sampling: [3, 2028, 5]  → 行: 2,7,12,...（index=2+i*5）
    # Cell_Across_Swath_Sampling: [3, 1348, 5] → 列: 2,7,12,...
    row_idx = np.arange(2, 2030, 5)   # 406 个
    col_idx = np.arange(2, 1354, 5)   # 271 个 → 取前 270
    row_idx = row_idx[:406]
    col_idx = col_idx[:270]

    lat5km = lat1km[np.ix_(row_idx, col_idx)]
    lon5km = lon1km[np.ix_(row_idx, col_idx)]

    # 填充值处理
    lat5km[lat5km < -90]  = np.nan
    lon5km[lon5km < -180] = np.nan

    return {
        'clp':       clp,
        'cth':       cth,
        'ctp':       ctp,
        'lon':       lon5km,
        'lat':       lat5km,
        'clp_attrs': clp_attrs,
        'cth_attrs': cth_attrs,
    }


# ═══════════════════════════════════════════════════════════════════
# 地理匹配：两组散点 → 公共规则网格
# ═══════════════════════════════════════════════════════════════════

def geo_match(fy4a: Dict[str, np.ndarray],
              modis: Dict[str, object],
              fy4a_var: str = 'data',
              modis_var: str = 'cth',
              grid_res_deg: float = 0.05,
              interp_method: str = 'nearest'
              ) -> Dict[str, np.ndarray]:
    """
    将 FY4A 和 MODIS 的指定变量插值到同一规则经纬度网格，
    返回重叠区域的插值结果及差值。

    Parameters
    ----------
    fy4a         : read_fy4a_l2_nc 的返回值
    modis        : read_modis_myd06 的返回值
    fy4a_var     : 在 fy4a dict 中使用的字段名
    modis_var    : 在 modis dict 中使用的字段名 ('clp'/'cth'/'ctp')
    grid_res_deg : 公共网格分辨率（度），0.05° ≈ 5km，与 MODIS 5km 匹配
    interp_method: griddata 方法 'nearest'/'linear'

    Returns
    -------
    dict，含：
      'lon2d', 'lat2d' : 公共网格坐标
      'fy4a_grid'      : FY4A 插值结果
      'modis_grid'     : MODIS 插值结果
      'diff_grid'      : FY4A - MODIS 差值
      'overlap_mask'   : 两者均有效的像素布尔掩码
      'extent'         : [lon_min, lon_max, lat_min, lat_max]
    """
    # 展开散点
    fy4a_lon  = fy4a['lon'].ravel()
    fy4a_lat  = fy4a['lat'].ravel()
    fy4a_val  = fy4a[fy4a_var].ravel()

    mod_lon   = modis['lon'].ravel()
    mod_lat   = modis['lat'].ravel()
    mod_val   = modis[modis_var].ravel()

    # 去掉 NaN
    def _valid(lon, lat, val):
        m = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
        return lon[m], lat[m], val[m]

    fy4a_lon, fy4a_lat, fy4a_val = _valid(fy4a_lon, fy4a_lat, fy4a_val)
    mod_lon,  mod_lat,  mod_val  = _valid(mod_lon,  mod_lat,  mod_val)

    if fy4a_lon.size == 0 or mod_lon.size == 0:
        raise ValueError("有效数据点为空，请检查输入路径或数据范围。")

    # 求重叠范围（两者经纬度范围的交集）
    lon_min = max(np.nanmin(fy4a_lon), np.nanmin(mod_lon))
    lon_max = min(np.nanmax(fy4a_lon), np.nanmax(mod_lon))
    lat_min = max(np.nanmin(fy4a_lat), np.nanmin(mod_lat))
    lat_max = min(np.nanmax(fy4a_lat), np.nanmax(mod_lat))

    print(f"[info] 经纬度重叠范围: lon=[{lon_min:.2f}, {lon_max:.2f}]  "
          f"lat=[{lat_min:.2f}, {lat_max:.2f}]")

    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError(
            f"两数据集在地理上无重叠区域！\n"
            f"  FY4A  覆盖: lon=[{np.nanmin(fy4a_lon):.1f}, {np.nanmax(fy4a_lon):.1f}]  "
            f"lat=[{np.nanmin(fy4a_lat):.1f}, {np.nanmax(fy4a_lat):.1f}]\n"
            f"  MODIS 覆盖: lon=[{np.nanmin(mod_lon):.1f}, {np.nanmax(mod_lon):.1f}]  "
            f"lat=[{np.nanmin(mod_lat):.1f}, {np.nanmax(mod_lat):.1f}]"
        )

    extent = [lon_min, lon_max, lat_min, lat_max]

    # 构建公共规则网格
    lon_grid = np.arange(lon_min, lon_max, grid_res_deg)
    lat_grid = np.arange(lat_min, lat_max, grid_res_deg)
    lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)
    pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])

    print(f"[info] 公共网格大小: {lon2d.shape}  分辨率={grid_res_deg}°")
    print(f"[info] 插值 FY4A → 公共网格...")
    fy4a_grid = griddata(
        np.column_stack([fy4a_lon, fy4a_lat]), fy4a_val,
        pts, method=interp_method
    ).reshape(lon2d.shape)

    print(f"[info] 插值 MODIS → 公共网格...")
    modis_grid = griddata(
        np.column_stack([mod_lon, mod_lat]), mod_val,
        pts, method=interp_method
    ).reshape(lon2d.shape)

    overlap = np.isfinite(fy4a_grid) & np.isfinite(modis_grid)
    diff    = np.where(overlap, fy4a_grid - modis_grid, np.nan)

    n_overlap = overlap.sum()
    area_km2  = n_overlap * (grid_res_deg * 111)**2
    print(f"[info] 重叠像素数: {n_overlap}  约合面积: {area_km2:.0f} km²")

    return {
        'lon2d':        lon2d,
        'lat2d':        lat2d,
        'fy4a_grid':    fy4a_grid,
        'modis_grid':   modis_grid,
        'diff_grid':    diff,
        'overlap_mask': overlap,
        'extent':       extent,
    }


# ═══════════════════════════════════════════════════════════════════
# 地图底图辅助
# ═══════════════════════════════════════════════════════════════════

def _make_ax(fig, pos, extent,
             land_color='lightgray', ocean_color='lightblue',
             map_res='50m', gridlines=True) -> plt.Axes:
    ax = fig.add_subplot(pos, projection=_PC)
    ax.set_extent(extent, crs=_PC)
    ax.add_feature(cfeature.OCEAN.with_scale(map_res),   facecolor=ocean_color, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale(map_res),    facecolor=land_color,  zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale(map_res),
                   linewidth=0.7, edgecolor='black', zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale(map_res),
                   linewidth=0.5, edgecolor='gray', linestyle='--', zorder=3)
    ax.add_feature(cfeature.RIVERS.with_scale(map_res),
                   linewidth=0.3, edgecolor='steelblue', zorder=3)
    if gridlines:
        gl = ax.gridlines(crs=_PC, draw_labels=True,
                          linewidth=0.5, color='gray', alpha=0.6,
                          linestyle='--', zorder=4)
        gl.top_labels   = False
        gl.right_labels = False
        gl.xformatter   = LONGITUDE_FORMATTER
        gl.yformatter   = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 7}
    return ax


# ═══════════════════════════════════════════════════════════════════
# 可视化函数
# ═══════════════════════════════════════════════════════════════════

def plot_overlap_check(fy4a: Dict,
                       modis: Dict,
                       map_res: str = '50m',
                       save_path: Optional[str] = None,
                       show: bool = True) -> plt.Figure:
    """
    总览图：显示 FY4A 和 MODIS 各自的空间覆盖范围（散点），
    判断能否重合、重合在哪里。
    """
    fy4a_lon = fy4a['lon'].ravel()
    fy4a_lat = fy4a['lat'].ravel()
    mod_lon  = modis['lon'].ravel()
    mod_lat  = modis['lat'].ravel()

    # 有效点
    vm = np.isfinite(fy4a['data'].ravel())
    mm = np.isfinite(mod_lon) & np.isfinite(mod_lat)

    # 全局范围
    all_lon = np.concatenate([fy4a_lon[vm], mod_lon[mm]])
    all_lat = np.concatenate([fy4a_lat[vm], mod_lat[mm]])
    pad = 2.0
    extent = [float(np.nanmin(all_lon)) - pad, float(np.nanmax(all_lon)) + pad,
              float(np.nanmin(all_lat)) - pad, float(np.nanmax(all_lat)) + pad]

    fig = plt.figure(figsize=(12, 7), dpi=110)
    ax  = _make_ax(fig, 111, extent, map_res=map_res)

    # FY4A 覆盖（每 8 个点取一个，减少点数）
    sl = slice(None, None, 8)
    ax.scatter(fy4a_lon[vm][sl], fy4a_lat[vm][sl],
               s=0.3, c='steelblue', alpha=0.4,
               transform=_PC, zorder=2, label='FY4A L2')

    # MODIS 覆盖
    ax.scatter(mod_lon[mm], mod_lat[mm],
               s=8, c='tomato', alpha=0.6,
               transform=_PC, zorder=2, label='MODIS MYD06 (5km)')

    ax.legend(loc='lower right', fontsize=10, markerscale=5)
    ax.set_title("FY4A L2 vs MODIS MYD06 — space overlap\n"
                 "(bule=FY4A red=MODIS overlap = match)",
                 fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    return fig


def plot_cth_comparison(match: Dict,
                        fy4a_label: str = 'FY4A L2 CTH',
                        modis_label: str = 'MODIS MYD06 CTH (IR)',
                        map_res: str = '50m',
                        save_path: Optional[str] = None,
                        show: bool = True) -> plt.Figure:
    """
    三联图：FY4A CTH | MODIS CTH | 差值 (FY4A - MODIS)
    """
    lon2d  = match['lon2d']
    lat2d  = match['lat2d']
    fy4a_g = match['fy4a_grid']
    mod_g  = match['modis_grid']
    diff   = match['diff_grid']
    extent = match['extent']
    overlap = match['overlap_mask']

    # 色标范围（两者统一）
    vmin = np.nanpercentile(np.concatenate([fy4a_g[overlap], mod_g[overlap]]), 2)
    vmax = np.nanpercentile(np.concatenate([fy4a_g[overlap], mod_g[overlap]]), 98)
    diff_abs = np.nanpercentile(np.abs(diff[overlap]), 95) if overlap.any() else 1000
    diff_abs = max(diff_abs, 100)   # 至少 100m 范围

    fig = plt.figure(figsize=(18, 6), dpi=110)
    kw  = dict(extent=extent, map_res=map_res)

    # ── 左：FY4A CTH ──
    ax1 = _make_ax(fig, 131, **kw)
    fy4a_plot = np.where(overlap, fy4a_g, np.nan)
    pcm1 = ax1.pcolormesh(lon2d, lat2d, fy4a_plot,
                           cmap='plasma', vmin=vmin, vmax=vmax,
                           transform=_PC, shading='auto', zorder=1, alpha=0.92)
    fig.colorbar(pcm1, ax=ax1, fraction=0.046, pad=0.05, label='m')
    ax1.set_title(fy4a_label, fontsize=11, fontweight='bold')

    # ── 中：MODIS CTH ──
    ax2 = _make_ax(fig, 132, **kw)
    mod_plot = np.where(overlap, mod_g, np.nan)
    pcm2 = ax2.pcolormesh(lon2d, lat2d, mod_plot,
                           cmap='plasma', vmin=vmin, vmax=vmax,
                           transform=_PC, shading='auto', zorder=1, alpha=0.92)
    fig.colorbar(pcm2, ax=ax2, fraction=0.046, pad=0.05, label='m')
    ax2.set_title(modis_label, fontsize=11, fontweight='bold')

    # ── 右：差值 ──
    ax3 = _make_ax(fig, 133, **kw)
    pcm3 = ax3.pcolormesh(lon2d, lat2d, diff,
                           cmap='RdBu_r', vmin=-diff_abs, vmax=diff_abs,
                           transform=_PC, shading='auto', zorder=1, alpha=0.92)
    fig.colorbar(pcm3, ax=ax3, fraction=0.046, pad=0.05, label='m')

    # 差值统计
    if overlap.any():
        d_valid = diff[overlap]
        stats_str = (f"mean={np.nanmean(d_valid):.0f}m  "
                     f"std={np.nanstd(d_valid):.0f}m  "
                     f"RMSE={np.sqrt(np.nanmean(d_valid**2)):.0f}m")
    else:
        stats_str = "no overlap pixel"
    ax3.set_title(f"diff (FY4A − MODIS)\n{stats_str}", fontsize=11, fontweight='bold')

    fig.suptitle("CTH — overlap",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    return fig


def plot_clp_comparison(match: Dict,
                        fy4a_label: str = 'FY4A L2 CLP',
                        modis_label: str = 'MODIS MYD06 CLP (IR)',
                        map_res: str = '50m',
                        save_path: Optional[str] = None,
                        show: bool = True) -> plt.Figure:
    """
    云相态对比图（离散类别）：
    FY4A CLP | MODIS CLP | 一致/不一致掩码

    FY4A CLP 编码: 0=Clear,1=Water,2=SuperCooled,3=Mixed,4=Ice,5=Uncertain
    MODIS  CLP 编码: 0=CloudFree,1=Water,2=Ice,3=Mixed,6=Undetermined
    """
    lon2d   = match['lon2d']
    lat2d   = match['lat2d']
    fy4a_g  = match['fy4a_grid']
    mod_g   = match['modis_grid']
    overlap = match['overlap_mask']
    extent  = match['extent']

    # ── 统一映射到简化3类：1=水云, 2=冰云/混合, 0=晴/其他 ──
    def _fy4a_to_simple(v):
        # FY4A: 1=Water, 2=SuperCooled→水, 3=Mixed→冰混, 4=Ice→冰
        out = np.full_like(v, np.nan)
        out[v == 1] = 1   # 水云
        out[v == 2] = 1   # 超冷水云→水云
        out[v == 3] = 2   # 混合
        out[v == 4] = 2   # 冰云
        out[v == 0] = 0   # 晴空
        return out

    def _modis_to_simple(v):
        # MODIS IR: 0=CloudFree, 1=Water, 2=Ice, 3=Mixed, 6=Undetermined
        out = np.full_like(v, np.nan)
        out[v == 0] = 0   # 晴空
        out[v == 1] = 1   # 水云
        out[v == 2] = 2   # 冰云
        out[v == 3] = 2   # 混合→冰混
        return out

    fy4a_simple = _fy4a_to_simple(np.round(fy4a_g))
    mod_simple  = _modis_to_simple(np.round(mod_g))

    # 一致性图：0=晴空一致, 1=水云一致, 2=冰混一致, 3=不一致
    agree = np.where(
        overlap & (fy4a_simple == mod_simple),
        fy4a_simple,   # 0/1/2
        np.where(overlap, 3, np.nan)  # 3=不一致
    )

    # 配色
    phase_colors = {0: '#A8D5BA', 1: '#5B9BD5', 2: '#C4A0E8', 3: '#E8736A'}
    phase_labels = {0: 'clear', 1: 'water', 2: 'ice/mixed', 3: 'diff'}
    cmap_phase = mcolors.ListedColormap(['#A8D5BA', '#5B9BD5', '#C4A0E8', '#E8736A'])
    norm_phase = mcolors.BoundaryNorm([0, 1, 2, 3, 4], 4)
    cmap_agree = mcolors.ListedColormap(['#A8D5BA', '#5B9BD5', '#C4A0E8', '#E8736A'])

    fy4a_plot = np.where(overlap, fy4a_simple, np.nan)
    mod_plot  = np.where(overlap, mod_simple,  np.nan)

    fig = plt.figure(figsize=(18, 6), dpi=110)
    kw  = dict(extent=extent, map_res=map_res)

    for idx, (data, label) in enumerate(
            [(fy4a_plot, fy4a_label), (mod_plot, modis_label), (agree, '一致性')]):
        ax = _make_ax(fig, int(f"13{idx+1}"), **kw)
        ax.pcolormesh(lon2d, lat2d, data,
                      cmap=cmap_agree, norm=norm_phase,
                      transform=_PC, shading='auto', zorder=1, alpha=0.92)
        ax.set_title(label, fontsize=11, fontweight='bold')

        # 一致性统计（第三子图）
        if idx == 2 and overlap.any():
            total  = overlap.sum()
            agree_n= int(np.sum((agree >= 0) & (agree <= 2) & np.isfinite(agree)))
            pct    = 100 * agree_n / total if total > 0 else 0
            ax.set_title(f"con  (con rate={pct:.1f}%)", fontsize=11, fontweight='bold')

    # 共用图例
    patches = [mpatches.Patch(color=phase_colors[k], label=phase_labels[k])
               for k in [0, 1, 2, 3]]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("CLP — overlap",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    return fig


def plot_diff_histogram(match_cth: Dict,
                        save_path: Optional[str] = None,
                        show: bool = True) -> plt.Figure:
    """
    CTH 差值的统计直方图 + 累计分布曲线。
    """
    overlap = match_cth['overlap_mask']
    diff    = match_cth['diff_grid']
    d_valid = diff[overlap]
    d_valid = d_valid[np.isfinite(d_valid)]

    if d_valid.size == 0:
        print("[warn] 无有效差值数据，跳过直方图。")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)

    # ── 直方图 ──
    bins = np.linspace(np.percentile(d_valid, 1), np.percentile(d_valid, 99), 60)
    ax1.hist(d_valid, bins=bins, color='steelblue', edgecolor='white',
             linewidth=0.3, alpha=0.85)
    ax1.axvline(0, color='red', linewidth=1.5, linestyle='--', label='Zero')
    ax1.axvline(np.nanmean(d_valid), color='orange', linewidth=1.5,
                linestyle='-', label=f'Mean={np.nanmean(d_valid):.0f}m')
    ax1.set_xlabel("CTH diff FY4A − MODIS [m]", fontsize=11)
    ax1.set_ylabel("pixel", fontsize=11)
    ax1.set_title("CTH diff", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 统计标注
    stats_text = (f"N = {d_valid.size}\n"
                  f"Mean = {np.nanmean(d_valid):.0f} m\n"
                  f"Std  = {np.nanstd(d_valid):.0f} m\n"
                  f"RMSE = {np.sqrt(np.nanmean(d_valid**2)):.0f} m\n"
                  f"Bias = {np.nanmean(d_valid):.0f} m")
    ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes,
             va='top', ha='right', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ── 累计分布 ──
    sorted_d = np.sort(d_valid)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax2.plot(sorted_d, cdf * 100, color='steelblue', linewidth=1.5)
    ax2.axvline(0, color='red', linewidth=1.5, linestyle='--')
    for q in [10, 25, 50, 75, 90]:
        qv = np.percentile(d_valid, q)
        ax2.axvline(qv, color='gray', linewidth=0.8, linestyle=':')
        ax2.text(qv, q + 1, f'P{q}={qv:.0f}m', fontsize=7, ha='center')
    ax2.set_xlabel("CTH diff FY4A − MODIS [m]", fontsize=11)
    ax2.set_ylabel("rate [%]", fontsize=11)
    ax2.set_title("CTH diff", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    return fig


def plot_scatter_cth(match: Dict,
                     save_path: Optional[str] = None,
                     show: bool = True) -> plt.Figure:
    """
    CTH 散点密度图：FY4A vs MODIS（重叠区域逐像素比较）。
    """
    overlap = match['overlap_mask']
    fy4a_v  = match['fy4a_grid'][overlap]
    mod_v   = match['modis_grid'][overlap]
    valid   = np.isfinite(fy4a_v) & np.isfinite(mod_v)
    fy4a_v, mod_v = fy4a_v[valid], mod_v[valid]

    if fy4a_v.size == 0:
        print("[warn] 无有效配对点，跳过散点图。")
        return None

    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)

    vmin = min(np.percentile(fy4a_v, 1), np.percentile(mod_v, 1))
    vmax = max(np.percentile(fy4a_v, 99), np.percentile(mod_v, 99))

    # 2D 密度直方图（伪彩色）
    h, xe, ye = np.histogram2d(mod_v, fy4a_v, bins=80,
                                range=[[vmin, vmax], [vmin, vmax]])
    ax.pcolormesh(xe, ye, h.T, cmap='hot_r', shading='auto')

    # 1:1 线
    ax.plot([vmin, vmax], [vmin, vmax], 'b--', linewidth=1.5, label='1:1')

    # 线性回归
    coeffs = np.polyfit(mod_v, fy4a_v, 1)
    x_fit  = np.linspace(vmin, vmax, 200)
    ax.plot(x_fit, np.polyval(coeffs, x_fit),
            'r-', linewidth=1.5,
            label=f'fit: y={coeffs[0]:.2f}x+{coeffs[1]:.0f}m')

    corr = float(np.corrcoef(mod_v, fy4a_v)[0, 1])
    rmse = float(np.sqrt(np.mean((fy4a_v - mod_v)**2)))
    ax.set_xlabel("MODIS MYD06 CTH (IR) (m)", fontsize=11)
    ax.set_ylabel("FY4A L2 CTH [m]",          fontsize=11)
    ax.set_title(f"CTH scatter(N={fy4a_v.size})\n"
                 f"R={corr:.3f}   RMSE={rmse:.0f}m", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════
# 一键运行：完整流程
# ═══════════════════════════════════════════════════════════════════

def run_geo_match(fy4a_cth_path: str,
                  fy4a_clp_path: str,
                  myd06_path: str,
                  myd03_path: str,
                  grid_res_deg: float = 0.05,
                  interp_method: str = 'nearest',
                  map_res: str = '50m',
                  output_dir: str = '.',
                  show: bool = True) -> Dict[str, object]:
    """
    一键完成 FY4A L2 vs MODIS MYD06 地理匹配与可视化。

    Parameters
    ----------
    fy4a_cth_path : FY4A L2 CTH NC 文件路径
    fy4a_clp_path : FY4A L2 CLP NC 文件路径
    myd06_path    : MODIS MYD06 HDF4 文件路径
    myd03_path    : MODIS MYD03 HDF4 文件路径（1km 经纬度）
    grid_res_deg  : 公共网格分辨率（度），推荐 0.04~0.1
    interp_method : 'nearest'（保留原始类别，推荐离散量）或 'linear'
    map_res       : cartopy 底图分辨率 '10m'/'50m'/'110m'
    output_dir    : 图片输出目录
    show          : 是否调用 plt.show()

    Returns
    -------
    dict，含所有中间结果（match_cth, match_clp, fy4a_cth, fy4a_clp, modis）
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 读取数据
    print("\n===== 1. 读取 FY4A L2 CTH =====")
    fy4a_cth = read_fy4a_l2_nc(fy4a_cth_path, 'CTH')

    print("\n===== 2. 读取 FY4A L2 CLP =====")
    fy4a_clp = read_fy4a_l2_nc(fy4a_clp_path, 'CLP')

    print("\n===== 3. 读取 MODIS MYD06 + MYD03 =====")
    modis = read_modis_myd06(myd06_path, myd03_path)

    # 2. 空间覆盖总览
    print("\n===== 4. 空间覆盖总览 =====")
    plot_overlap_check(
        fy4a_cth, modis,
        map_res=map_res,
        save_path=str(out / 'overlap_coverage.png'),
        show=show
    )

    # 3. CTH 匹配
    print("\n===== 5. CTH 地理匹配 =====")
    try:
        match_cth = geo_match(fy4a_cth, modis,
                              fy4a_var='data', modis_var='cth',
                              grid_res_deg=grid_res_deg,
                              interp_method=interp_method)

        plot_cth_comparison(match_cth,
                            map_res=map_res,
                            save_path=str(out / 'cth_comparison.png'),
                            show=show)
        plot_diff_histogram(match_cth,
                            save_path=str(out / 'cth_diff_histogram.png'),
                            show=show)
        plot_scatter_cth(match_cth,
                         save_path=str(out / 'cth_scatter.png'),
                         show=show)
    except ValueError as e:
        print(f"[warn] CTH 匹配失败: {e}")
        match_cth = None

    # 4. CLP 匹配
    print("\n===== 6. CLP 地理匹配 =====")
    try:
        match_clp = geo_match(fy4a_clp, modis,
                              fy4a_var='data', modis_var='clp',
                              grid_res_deg=grid_res_deg,
                              interp_method='nearest')  # 相态必须用 nearest

        plot_clp_comparison(match_clp,
                            map_res=map_res,
                            save_path=str(out / 'clp_comparison.png'),
                            show=show)
    except ValueError as e:
        print(f"[warn] CLP 匹配失败: {e}")
        match_clp = None

    print("\n===== 完成 =====")
    return {
        'fy4a_cth':  fy4a_cth,
        'fy4a_clp':  fy4a_clp,
        'modis':     modis,
        'match_cth': match_cth,
        'match_clp': match_clp,
    }


# ═══════════════════════════════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # ── 修改为你的实际文件路径 ─────────────────────────────────────
    FY4A_CTH = ("/data/Data_yuq/testdata/fy4al2/"
                "FY4A-_AGRI--_N_DISK_1047E_L2-_CTH-_MULT_NOM_"
                "20190401000000_20190401001459_4000M_V0001.NC")
    FY4A_CLP = ("/data/Data_yuq/testdata/fy4al2/"
                "FY4A-_AGRI--_N_DISK_1047E_L2-_CLP-_MULT_NOM_"
                "20190401000000_20190401001459_4000M_V0001.NC")
    MYD06 = ("/data/Data_yuq/testdata/modis/"
             "MYD06_L2.A2019005.0100.061.2019005181910.hdf")
    MYD03 = ("/data/Data_yuq/testdata/modis/"
             "MYD03.A2019005.0100.061.2019005164819.hdf")

    results = run_geo_match(
        fy4a_cth_path = FY4A_CTH,
        fy4a_clp_path = FY4A_CLP,
        myd06_path    = MYD06,
        myd03_path    = MYD03,
        grid_res_deg  = 0.05,     # 0.05° ≈ 5km，与 MODIS 5km 分辨率匹配
        interp_method = 'nearest',
        map_res       = '50m',
        output_dir    = './geo_match_output',
        show          = False,    # 服务器运行时设 False，本地设 True
    )

    # 如需单独调用各步骤，示例如下：
    #
    # fy4a_cth = read_fy4a_l2_nc(FY4A_CTH, 'CTH')
    # modis    = read_modis_myd06(MYD06, MYD03)
    # match    = geo_match(fy4a_cth, modis, modis_var='cth',
    #                      grid_res_deg=0.05, interp_method='nearest')
    # plot_cth_comparison(match, show=True)
    # plot_diff_histogram(match, show=True)
    # plot_scatter_cth(match, show=True)
