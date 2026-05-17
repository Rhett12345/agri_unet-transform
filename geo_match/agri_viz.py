"""
FY-4A AGRI L1 HDF5 数据可视化工具包
支持 FDI（辐射定标）和 GEO（地理定位）两类文件的读取与可视化。
坐标转换基于官方文档《标称上行列号和经纬度的互相转换方法_V2》。

依赖安装：
    pip install h5py numpy matplotlib cartopy scipy
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Optional, Tuple, List

# cartopy（地图底图核心）
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# scipy 用于散点→网格插值（pcolormesh 显示）
from scipy.interpolate import griddata

# ─────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────
RESOLUTION_PARAMS = {
    500:  dict(COFF=10991.5, CFAC=81865099, LOFF=10991.5, LFAC=81865099),
    1000: dict(COFF=5495.5,  CFAC=40932549, LOFF=5495.5,  LFAC=40932549),
    2000: dict(COFF=2747.5,  CFAC=20466274, LOFF=2747.5,  LFAC=20466274),
    4000: dict(COFF=1373.5,  CFAC=10233137, LOFF=1373.5,  LFAC=10233137),
}

EA    = 6378.137     # 地球半长轴 [km]
EB    = 6356.7523    # 地球短半轴 [km]
H     = 42164.0      # 地心到卫星质心距离 [km]
LAM_D = 104.7        # FY-4A 星下点经度 [°]

_PC_CRS = ccrs.PlateCarree()   # 普通经纬度投影（底图通用）


# ═══════════════════════════════════════════════════════════════════
# 坐标转换（严格按官方文档公式）
# ═══════════════════════════════════════════════════════════════════

def lonlat_to_linecolumn(lon: np.ndarray,
                         lat: np.ndarray,
                         resolution: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
    """
    地理经纬度 → 标称行列号  (lon, lat) → (l, c)

    Parameters
    ----------
    lon        : 地理经度 [°]，标量或数组
    lat        : 地理纬度 [°]，标量或数组
    resolution : 分辨率，可选 500/1000/2000/4000 [m]

    Returns
    -------
    l : 行号 (float)
    c : 列号 (float)
    """
    p = RESOLUTION_PARAMS[resolution]
    COFF, CFAC, LOFF, LFAC = p['COFF'], p['CFAC'], p['LOFF'], p['LFAC']

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)

    # 地心经纬度
    lam_e = lon_r
    phi_e = np.arctan((EB**2 / EA**2) * np.tan(lat_r))

    # Re
    r_e = EB / np.sqrt(1 - ((EA**2 - EB**2) / EA**2) * np.cos(phi_e)**2)

    # r1, r2, r3
    lam_D_r = np.deg2rad(LAM_D)
    r1 = H - r_e * np.cos(phi_e) * np.cos(lam_e - lam_D_r)
    r2 = -r_e * np.cos(phi_e) * np.sin(lam_e - lam_D_r)
    r3 = r_e * np.sin(phi_e)

    # rn, x, y
    rn = np.sqrt(r1**2 + r2**2 + r3**2)
    x  = np.rad2deg(np.arctan(-r2 / r1))
    y  = np.rad2deg(np.arcsin(-r3 / rn))

    c = COFF + x * 2**-16 * CFAC
    l = LOFF + y * 2**-16 * LFAC
    return l, c


def linecolumn_to_lonlat(l: np.ndarray,
                         c: np.ndarray,
                         resolution: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
    """
    标称行列号 → 地理经纬度  (l, c) → (lon, lat)

    地球盘外的点返回 NaN。

    Parameters
    ----------
    l          : 行号，标量或数组
    c          : 列号，标量或数组
    resolution : 分辨率，可选 500/1000/2000/4000 [m]

    Returns
    -------
    lon : 地理经度 [°]
    lat : 地理纬度 [°]
    """
    p = RESOLUTION_PARAMS[resolution]
    COFF, CFAC, LOFF, LFAC = p['COFF'], p['CFAC'], p['LOFF'], p['LFAC']

    l = np.asarray(l, dtype=float)
    c = np.asarray(c, dtype=float)

    x = np.deg2rad((c - COFF) / (2 ** -16 * CFAC))
    y = np.deg2rad((l - LOFF) / (2 ** -16 * LFAC))

    cos_x, cos_y = np.cos(x), np.cos(y)
    sin_x, sin_y = np.sin(x), np.sin(y)

    disc = ((H * cos_x * cos_y)**2
            - (cos_y**2 + (EA**2 / EB**2) * sin_y**2) * (H**2 - EA**2))

    valid     = disc >= 0
    disc_safe = np.where(valid, disc, 0.0)

    sd  = np.sqrt(disc_safe)
    sn  = (H * cos_x * cos_y - sd) / (cos_y**2 + (EA**2 / EB**2) * sin_y**2)
    s1  = H - sn * cos_x * cos_y
    s2  = sn * sin_x * cos_y
    s3  = -sn * sin_y
    sxy = np.sqrt(s1**2 + s2**2)

    lon = np.rad2deg(np.arctan(s2 / s1)) + LAM_D
    lat = np.rad2deg(np.arctan((EA**2 / EB**2) * (s3 / sxy)))

    lon = np.where(valid, lon, np.nan)
    lat = np.where(valid, lat, np.nan)
    return lon, lat


# ═══════════════════════════════════════════════════════════════════
# HDF5 文件读取
# ═══════════════════════════════════════════════════════════════════

def _decode(b) -> str:
    if isinstance(b, (bytes, np.bytes_)):
        return b.decode(errors='replace')
    return str(b)


def read_fdi_channel(fdi_path: str,
                     channel: int,
                     calibrate: bool = True) -> Tuple[np.ndarray, dict]:
    """
    读取 FDI 文件的某个通道数据。

    Parameters
    ----------
    fdi_path  : FDI HDF5 文件路径
    channel   : 通道号 1~14
    calibrate : True→用 CALChannel 查表转为物理量；False→返回原始 DN

    Returns
    -------
    data  : float32 数组，shape=(rows, cols)，填充值→NaN
    attrs : 属性字典（含 center_wavelength, units 等）
    """
    ch_name  = f"NOMChannel{channel:02d}"
    cal_name = f"CALChannel{channel:02d}"

    with h5py.File(fdi_path, 'r') as f:
        raw  = f[ch_name][:]
        fill = int(f[ch_name].attrs.get('FillValue', [65535])[0])
        attrs = dict(f[ch_name].attrs)

        if calibrate and cal_name in f:
            cal_table = f[cal_name][:]
            mask = raw == fill
            idx  = np.clip(raw.astype(np.int32), 0, len(cal_table) - 1)
            data = cal_table[idx].astype(np.float32)
            data[mask] = np.nan
            attrs['calibrated'] = True
        else:
            data = raw.astype(np.float32)
            data[data == fill] = np.nan
            attrs['calibrated'] = False

    return data, attrs


def read_geo_dataset(geo_path: str,
                     dataset: str = 'NOMSunZenith') -> Tuple[np.ndarray, dict]:
    """
    读取 GEO 文件中的指定数据集。

    常用 dataset：
        NOMSunZenith, NOMSunAzimuth,
        NOMSatelliteZenith, NOMSatelliteAzimuth,
        NOMSunGlintAngle, LineNumber, ColumnNumber
    """
    with h5py.File(geo_path, 'r') as f:
        raw  = f[dataset][:]
        fill = float(f[dataset].attrs.get('FillValue', [65535])[0])
        attrs = dict(f[dataset].attrs)

    data = raw.astype(np.float32)
    data[data == fill] = np.nan
    return data, attrs


def list_channels(fdi_path: str) -> List[dict]:
    """列出 FDI 文件中所有可用通道及其基本信息。"""
    info = []
    with h5py.File(fdi_path, 'r') as f:
        for i in range(1, 15):
            key = f"NOMChannel{i:02d}"
            if key in f:
                a = f[key].attrs
                info.append({
                    'channel':    i,
                    'wavelength': _decode(a.get('center_wavelength', b'')),
                    'long_name':  _decode(a.get('long_name', b'')),
                    'shape':      tuple(f[key].shape),
                })
    return info


def print_file_summary(hdf_path: str):
    """打印 HDF5 文件基本信息摘要。"""
    with h5py.File(hdf_path, 'r') as f:
        print(f"\n{'='*60}")
        print(f"文件: {Path(hdf_path).name}")
        print(f"{'='*60}")
        print("【全局属性（部分）】")
        for key in ['Satellite Name', 'Sensor Name', 'OBIType',
                    'NOMCenterLon', 'NOMSatHeight', 'RegLength', 'RegWidth',
                    'Observing Beginning Date', 'Observing Beginning Time']:
            if key in f.attrs:
                v = f.attrs[key]
                if isinstance(v, (bytes, np.bytes_)):
                    v = v.decode()
                elif hasattr(v, '__len__') and len(v) == 1:
                    v = v[0]
                print(f"  {key}: {v}")
        print("\n【数据集列表】")
        for name in f.keys():
            ds = f[name]
            print(f"  {name:40s}  shape={str(ds.shape):20s}  dtype={ds.dtype}")


# ═══════════════════════════════════════════════════════════════════
# 内部辅助
# ═══════════════════════════════════════════════════════════════════

def _make_cartopy_ax(fig: plt.Figure,
                     pos,
                     extent: Optional[List[float]] = None,
                     projection: ccrs.Projection = _PC_CRS,
                     land_color:  str = 'lightgray',
                     ocean_color: str = 'lightblue',
                     map_resolution: str = '50m',
                     gridlines: bool = True) -> plt.Axes:
    """
    创建带 cartopy 地图底图的子图。

    Parameters
    ----------
    pos            : subplot 位置（如 111 或 (1,2,1)）
    extent         : [lon_min, lon_max, lat_min, lat_max]；None=全球
    projection     : cartopy 投影，默认 PlateCarree
    land_color     : 陆地填充色；'none' 表示透明（让卫星数据透出）
    ocean_color    : 海洋填充色；'none' 表示透明
    map_resolution : Natural Earth 要素分辨率 '10m'/'50m'/'110m'
    gridlines      : 是否绘制经纬度网格线及刻度
    """
    ax = fig.add_subplot(pos, projection=projection)

    if extent is not None:
        ax.set_extent(extent, crs=_PC_CRS)
    else:
        ax.set_global()

    # ── 底图要素（zorder=0 确保在卫星数据下方）──
    if ocean_color != 'none':
        ax.add_feature(cfeature.OCEAN.with_scale(map_resolution),
                       facecolor=ocean_color, zorder=0)
    if land_color != 'none':
        ax.add_feature(cfeature.LAND.with_scale(map_resolution),
                       facecolor=land_color, zorder=0)

    ax.add_feature(cfeature.COASTLINE.with_scale(map_resolution),
                   linewidth=0.7, edgecolor='black', zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale(map_resolution),
                   linewidth=0.5, edgecolor='gray', linestyle='--', zorder=3)
    ax.add_feature(cfeature.RIVERS.with_scale(map_resolution),
                   linewidth=0.3, edgecolor='steelblue', zorder=3)
    ax.add_feature(cfeature.LAKES.with_scale(map_resolution),
                   facecolor='lightblue', edgecolor='steelblue',
                   linewidth=0.3, zorder=2)

    # ── 经纬度网格线 ──
    if gridlines:
        gl = ax.gridlines(crs=_PC_CRS, draw_labels=True,
                          linewidth=0.5, color='gray',
                          alpha=0.6, linestyle='--', zorder=4)
        gl.top_labels   = False
        gl.right_labels = False
        gl.xformatter   = LONGITUDE_FORMATTER
        gl.yformatter   = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}

    return ax


def _data_to_scatter(data: np.ndarray,
                     resolution: int = 4000,
                     subsample: int = 1
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将像素数组展开为散点 (lon, lat, value)，去除 NaN。
    subsample：降采样步长，越大越快。
    """
    sl   = slice(None, None, subsample)
    rows = np.arange(data.shape[0])[sl]
    cols = np.arange(data.shape[1])[sl]
    c2d, r2d = np.meshgrid(cols, rows)
    lon, lat = linecolumn_to_lonlat(r2d, c2d, resolution)

    d_sub = data[sl, :][:, sl]
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(d_sub)
    return lon[valid], lat[valid], d_sub[valid]


def _scatter_to_grid(lon: np.ndarray,
                     lat: np.ndarray,
                     values: np.ndarray,
                     lon_range: Tuple[float, float],
                     lat_range: Tuple[float, float],
                     nx: int = 600,
                     ny: int = 600,
                     method: str = 'linear'
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将散点 (lon, lat, value) 插值到规则经纬度网格，供 pcolormesh 渲染。

    Returns
    -------
    lon2d, lat2d, val2d : 规则网格，shape=(ny, nx)
    """
    lon_grid = np.linspace(lon_range[0], lon_range[1], nx)
    lat_grid = np.linspace(lat_range[0], lat_range[1], ny)
    lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)

    val2d = griddata(
        np.column_stack([lon, lat]),
        values,
        (lon2d, lat2d),
        method=method
    )
    return lon2d, lat2d, val2d


def _clip_to_extent(lon, lat, val, extent):
    lon_min, lon_max, lat_min, lat_max = extent
    mask = ((lon >= lon_min) & (lon <= lon_max) &
            (lat >= lat_min) & (lat <= lat_max))
    return lon[mask], lat[mask], val[mask]


# ═══════════════════════════════════════════════════════════════════
# 可视化函数
# ═══════════════════════════════════════════════════════════════════

def plot_channel(fdi_path: str,
                 channel: int,
                 calibrate: bool = True,
                 cmap: str = 'gray',
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None,
                 title: Optional[str] = None,
                 save_path: Optional[str] = None,
                 show: bool = True) -> plt.Figure:
    """
    绘制 FDI 单通道原始图像（行列号坐标，无地图投影）。
    速度最快，适合快速检查数据质量。
    """
    data, attrs = read_fdi_channel(fdi_path, channel, calibrate)
    wl   = _decode(attrs.get('center_wavelength', ''))
    unit = _decode(attrs.get('units', 'DN'))
    lo   = vmin if vmin is not None else np.nanpercentile(data, 2)
    hi   = vmax if vmax is not None else np.nanpercentile(data, 98)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    im = ax.imshow(data, origin='upper', cmap=cmap, vmin=lo, vmax=hi,
                   interpolation='nearest', aspect='equal')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"{'Physical' if calibrate else 'DN'}  [{unit}]", fontsize=11)
    ax.set_title(title or f"FY-4A AGRI  Ch{channel:02d} ({wl})",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Column"); ax.set_ylabel("Line")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    return fig


def plot_channel_on_map(fdi_path: str,
                        channel: int,
                        calibrate: bool = True,
                        resolution: int = 4000,
                        subsample: int = 4,
                        interp_nx: int = 600,
                        interp_ny: int = 600,
                        interp_method: str = 'linear',
                        extent: Optional[List[float]] = None,
                        cmap: str = 'gray',
                        vmin: Optional[float] = None,
                        vmax: Optional[float] = None,
                        land_color: str = 'lightgray',
                        ocean_color: str = 'none',
                        map_resolution: str = '50m',
                        title: Optional[str] = None,
                        save_path: Optional[str] = None,
                        show: bool = True) -> plt.Figure:
    """
    将 FDI 通道数据叠加到 cartopy 地图底图（PlateCarree 投影）。

    工作流：
        行列号 → 经纬度（坐标转换）
        → 散点降采样
        → 插值到规则网格（griddata）
        → pcolormesh 渲染
        → cartopy 底图叠加海岸线/边界/网格线

    Parameters
    ----------
    subsample      : 坐标转换降采样步长（越大越快，精度越低）
    interp_nx/ny   : 插值目标网格分辨率（越大越精细）
    interp_method  : 'linear'/'nearest'/'cubic'
    extent         : [lon_min, lon_max, lat_min, lat_max]；None=自动全盘
    ocean_color    : 'none' 让卫星数据覆盖海洋；'lightblue' 先填海色
    map_resolution : '10m'（精细）/ '50m'（默认）/ '110m'（粗略，快）
    """
    data, attrs = read_fdi_channel(fdi_path, channel, calibrate)
    wl   = _decode(attrs.get('center_wavelength', ''))
    unit = _decode(attrs.get('units', ''))

    print(f"[info] 坐标转换中（subsample={subsample}）...")
    lon_pts, lat_pts, val_pts = _data_to_scatter(data, resolution, subsample)

    if extent is None:
        extent = [float(np.nanmin(lon_pts)), float(np.nanmax(lon_pts)),
                  float(np.nanmin(lat_pts)), float(np.nanmax(lat_pts))]
    lon_pts, lat_pts, val_pts = _clip_to_extent(lon_pts, lat_pts, val_pts, extent)

    print(f"[info] 插值到 {interp_nx}×{interp_ny} 网格（method={interp_method}）...")
    lon2d, lat2d, val2d = _scatter_to_grid(
        lon_pts, lat_pts, val_pts,
        (extent[0], extent[1]), (extent[2], extent[3]),
        nx=interp_nx, ny=interp_ny, method=interp_method
    )

    lo = vmin if vmin is not None else np.nanpercentile(val_pts, 2)
    hi = vmax if vmax is not None else np.nanpercentile(val_pts, 98)

    fig = plt.figure(figsize=(12, 9), dpi=110)
    ax  = _make_cartopy_ax(fig, 111,
                           extent=extent,
                           land_color=land_color,
                           ocean_color=ocean_color,
                           map_resolution=map_resolution)

    # 卫星数据层（zorder=1，在底图陆地/海洋之上，在海岸线之下）
    pcm = ax.pcolormesh(lon2d, lat2d, val2d,
                        cmap=cmap, vmin=lo, vmax=hi,
                        transform=_PC_CRS,
                        shading='auto', zorder=1, alpha=0.92)

    cb = fig.colorbar(pcm, ax=ax, fraction=0.03, pad=0.06, shrink=0.85)
    cb.set_label(f"{'Calibrated' if calibrate else 'DN'}  [{unit}]", fontsize=11)
    ax.set_title(title or f"FY-4A AGRI  Ch{channel:02d} ({wl})  —  Map View",
                 fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    return fig


def plot_rgb_on_map(fdi_path: str,
                    r_ch: int = 2,
                    g_ch: int = 3,
                    b_ch: int = 1,
                    resolution: int = 4000,
                    subsample: int = 4,
                    interp_nx: int = 600,
                    interp_ny: int = 600,
                    gamma: float = 1.5,
                    percentile: Tuple[float, float] = (2, 98),
                    extent: Optional[List[float]] = None,
                    land_color: str = 'none',
                    ocean_color: str = 'none',
                    map_resolution: str = '50m',
                    title: Optional[str] = None,
                    save_path: Optional[str] = None,
                    show: bool = True) -> plt.Figure:
    """
    RGB 合成真彩色/假彩色图像叠加到 cartopy 地图底图。

    Parameters
    ----------
    r_ch, g_ch, b_ch : R/G/B 对应的通道号（默认 2/3/1 = 0.65/0.83/0.47μm）
    gamma            : gamma 校正系数（>1 增亮）
    percentile       : 拉伸百分位 (lo, hi)
    """
    def _load_scatter(ch):
        d, _ = read_fdi_channel(fdi_path, ch, calibrate=True)
        lo, hi = np.nanpercentile(d[np.isfinite(d)], percentile)
        d = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1) ** (1 / gamma)
        lon, lat, val = _data_to_scatter(d, resolution, subsample)
        print(f'Ch{ch:02d} — lon min/max:', np.nanmin(lon), np.nanmax(lon))
        print(f'Ch{ch:02d} — lat min/max:', np.nanmin(lat), np.nanmax(lat))
        return _data_to_scatter(d, resolution, subsample)

    print("[info] 读取并坐标转换 R/G/B 三通道...")
    lon_r, lat_r, val_r = _load_scatter(r_ch)
    lon_g, lat_g, val_g = _load_scatter(g_ch)
    lon_b, lat_b, val_b = _load_scatter(b_ch)

    if extent is None:
        extent = [float(np.nanmin(lon_r)), float(np.nanmax(lon_r)),
                  float(np.nanmin(lat_r)), float(np.nanmax(lat_r))]
    lon_min, lon_max, lat_min, lat_max = extent

    def _proc(lon_pts, lat_pts, val_pts):
        lon_pts, lat_pts, val_pts = _clip_to_extent(
            lon_pts, lat_pts, val_pts, extent)
        _, _, v2d = _scatter_to_grid(
            lon_pts, lat_pts, val_pts,
            (lon_min, lon_max), (lat_min, lat_max),
            nx=interp_nx, ny=interp_ny, method='linear'
        )
        return np.nan_to_num(v2d, nan=0.0).clip(0, 1)

    print(f"[info] 插值到 {interp_nx}×{interp_ny} 网格...")
    R = _proc(lon_r, lat_r, val_r)
    G = _proc(lon_g, lat_g, val_g)
    B = _proc(lon_b, lat_b, val_b)

    rgb = np.stack([R, G, B], axis=-1)   # (ny, nx, 3)

    fig = plt.figure(figsize=(12, 9), dpi=110)
    ax  = _make_cartopy_ax(fig, 111,
                           extent=extent,
                           land_color=land_color,
                           ocean_color=ocean_color,
                           map_resolution=map_resolution)

    ax.imshow(rgb,
              origin='lower',
              extent=[lon_min, lon_max, lat_min, lat_max],
              transform=_PC_CRS,
              interpolation='bilinear',
              zorder=1, alpha=0.95)

    ax.set_title(
        title or f"FY-4A AGRI  RGB  R=Ch{r_ch:02d}  G=Ch{g_ch:02d}  B=Ch{b_ch:02d}",
        fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    return fig


def plot_geo_field_on_map(geo_path: str,
                          dataset: str = 'NOMSunZenith',
                          resolution: int = 4000,
                          subsample: int = 4,
                          interp_nx: int = 600,
                          interp_ny: int = 600,
                          extent: Optional[List[float]] = None,
                          cmap: str = 'RdYlBu_r',
                          vmin: Optional[float] = None,
                          vmax: Optional[float] = None,
                          land_color: str = 'lightgray',
                          ocean_color: str = 'none',
                          map_resolution: str = '50m',
                          title: Optional[str] = None,
                          save_path: Optional[str] = None,
                          show: bool = True) -> plt.Figure:
    """
    将 GEO 文件中的角度/几何数据叠加到 cartopy 地图底图。
    """
    data, attrs = read_geo_dataset(geo_path, dataset)
    unit = _decode(attrs.get('units', '°'))

    print(f"[info] 坐标转换中（subsample={subsample}）...")
    lon_pts, lat_pts, val_pts = _data_to_scatter(data, resolution, subsample)

    if extent is None:
        extent = [float(np.nanmin(lon_pts)), float(np.nanmax(lon_pts)),
                  float(np.nanmin(lat_pts)), float(np.nanmax(lat_pts))]
    lon_pts, lat_pts, val_pts = _clip_to_extent(lon_pts, lat_pts, val_pts, extent)

    print(f"[info] 插值到 {interp_nx}×{interp_ny} 网格...")
    lon2d, lat2d, val2d = _scatter_to_grid(
        lon_pts, lat_pts, val_pts,
        (extent[0], extent[1]), (extent[2], extent[3]),
        nx=interp_nx, ny=interp_ny
    )

    lo = vmin if vmin is not None else np.nanpercentile(val_pts, 2)
    hi = vmax if vmax is not None else np.nanpercentile(val_pts, 98)

    fig = plt.figure(figsize=(12, 9), dpi=110)
    ax  = _make_cartopy_ax(fig, 111,
                           extent=extent,
                           land_color=land_color,
                           ocean_color=ocean_color,
                           map_resolution=map_resolution)

    pcm = ax.pcolormesh(lon2d, lat2d, val2d,
                        cmap=cmap, vmin=lo, vmax=hi,
                        transform=_PC_CRS,
                        shading='auto', zorder=1, alpha=0.9)
    cb = fig.colorbar(pcm, ax=ax, fraction=0.03, pad=0.06, shrink=0.85)
    cb.set_label(f"{dataset}  [{unit}]", fontsize=11)

    ax.set_title(title or f"FY-4A AGRI  {dataset}  —  Map View",
                 fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    return fig


def plot_all_channels(fdi_path: str,
                      calibrate: bool = True,
                      ncols: int = 4,
                      cmap: str = 'gray',
                      save_path: Optional[str] = None,
                      show: bool = True) -> plt.Figure:
    """
    14 通道缩略图总览（行列号坐标，速度快，适合快速检查）。
    """
    nrows = -(-14 // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 4), dpi=100)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ch = i + 1
        if ch > 14:
            ax.axis('off')
            continue
        try:
            data, attrs = read_fdi_channel(fdi_path, ch, calibrate)
            wl = _decode(attrs.get('center_wavelength', ''))
            lo, hi = np.nanpercentile(data, [2, 98])
            ax.imshow(data, origin='upper', cmap=cmap,
                      vmin=lo, vmax=hi, aspect='equal', interpolation='nearest')
            ax.set_title(f"Ch{ch:02d} ({wl})", fontsize=9)
        except Exception as e:
            ax.set_title(f"Ch{ch:02d}  ERR", fontsize=9, color='red')
            print(f"[warn] Ch{ch:02d}: {e}")
        ax.axis('off')

    fig.suptitle("FY-4A AGRI — All Channels Overview",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    return fig


def plot_fdi_geo_comparison_on_map(
        fdi_path: str,
        geo_path: str,
        channel: int = 12,
        geo_dataset: str = 'NOMSunZenith',
        resolution: int = 4000,
        subsample: int = 4,
        interp_nx: int = 500,
        interp_ny: int = 500,
        extent: Optional[List[float]] = None,
        map_resolution: str = '50m',
        save_path: Optional[str] = None,
        show: bool = True) -> plt.Figure:
    """
    左：FDI 通道  右：GEO 数据集，均叠加在 cartopy 地图底图上。
    """
    data_fdi, attrs_fdi = read_fdi_channel(fdi_path, channel, calibrate=True)
    data_geo, attrs_geo = read_geo_dataset(geo_path, geo_dataset)

    wl       = _decode(attrs_fdi.get('center_wavelength', ''))
    unit_fdi = _decode(attrs_fdi.get('units', ''))
    unit_geo = _decode(attrs_geo.get('units', '°'))

    print("[info] 坐标转换中...")
    lon_f, lat_f, val_f = _data_to_scatter(data_fdi, resolution, subsample)
    lon_g, lat_g, val_g = _data_to_scatter(data_geo, resolution, subsample)

    if extent is None:
        extent = [float(np.nanmin(lon_f)), float(np.nanmax(lon_f)),
                  float(np.nanmin(lat_f)), float(np.nanmax(lat_f))]

    lon_f, lat_f, val_f = _clip_to_extent(lon_f, lat_f, val_f, extent)
    lon_g, lat_g, val_g = _clip_to_extent(lon_g, lat_g, val_g, extent)
    lon_range = (extent[0], extent[1])
    lat_range = (extent[2], extent[3])

    print("[info] 插值中...")
    lon2d, lat2d, val2d_fdi = _scatter_to_grid(
        lon_f, lat_f, val_f, lon_range, lat_range, nx=interp_nx, ny=interp_ny)
    _,     _,     val2d_geo = _scatter_to_grid(
        lon_g, lat_g, val_g, lon_range, lat_range, nx=interp_nx, ny=interp_ny)

    kw_ax = dict(extent=extent, land_color='lightgray', ocean_color='none',
                 map_resolution=map_resolution)

    fig = plt.figure(figsize=(20, 8), dpi=110)

    # ── 左图：FDI ──
    ax1 = _make_cartopy_ax(fig, 121, **kw_ax)
    lo1, hi1 = np.nanpercentile(val_f, [2, 98])
    pcm1 = ax1.pcolormesh(lon2d, lat2d, val2d_fdi,
                           cmap='gray', vmin=lo1, vmax=hi1,
                           transform=_PC_CRS, shading='auto',
                           zorder=1, alpha=0.92)
    fig.colorbar(pcm1, ax=ax1, fraction=0.046, pad=0.05,
                 label=f"[{unit_fdi}]")
    ax1.set_title(f"Ch{channel:02d} ({wl})", fontsize=12, fontweight='bold')

    # ── 右图：GEO ──
    ax2 = _make_cartopy_ax(fig, 122, **kw_ax)
    lo2, hi2 = np.nanpercentile(val_g, [2, 98])
    pcm2 = ax2.pcolormesh(lon2d, lat2d, val2d_geo,
                           cmap='RdYlBu_r', vmin=lo2, vmax=hi2,
                           transform=_PC_CRS, shading='auto',
                           zorder=1, alpha=0.92)
    fig.colorbar(pcm2, ax=ax2, fraction=0.046, pad=0.05,
                 label=f"{geo_dataset}  [{unit_geo}]")
    ax2.set_title(geo_dataset, fontsize=12, fontweight='bold')

    fig.suptitle("FY-4A AGRI — FDI vs GEO  (Map View)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    FDI_PATH = ("/data/Data_yuq/testdata/fy4a/"
                "FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_"
                "20190105000000_20190105001459_4000M_V0001.HDF")
    GEO_PATH = ("/data/Data_yuq/testdata/fy4a/"
                "FY4A-_AGRI--_N_DISK_1047E_L1-_GEO-_MULT_NOM_"
                "20190105000000_20190105001459_4000M_V0001.HDF")

    # 1. 文件摘要
    print_file_summary(FDI_PATH)
    print_file_summary(GEO_PATH)

    # 2. 通道列表
    for c in list_channels(FDI_PATH):
        print(f"  Ch{c['channel']:02d}  {c['wavelength']:8s}  {c['shape']}")

    # 3. 原始行列号图（无底图，速度最快）
    plot_channel(FDI_PATH, channel=12, cmap='gray',
                 save_path='ch12_raw.png', show=False)

    # 4. 所有通道总览
    plot_all_channels(FDI_PATH, save_path='all_channels.png', show=False)

    # 5. 单通道 + 地图底图（全盘）
    plot_channel_on_map(
        FDI_PATH, channel=12,
        resolution=4000, subsample=4,
        interp_nx=600, interp_ny=600,
        cmap='gray', ocean_color='lightblue',
        save_path='ch12_map_fulldisk.png', show=False
    )

    # 6. 单通道 + 地图底图（中国区域裁剪，更精细）
    plot_channel_on_map(
        FDI_PATH, channel=12,
        resolution=4000, subsample=2,
        interp_nx=800, interp_ny=800,
        extent=[70, 140, 10, 55],      # [lon_min, lon_max, lat_min, lat_max]
        cmap='gray', ocean_color='lightblue',
        map_resolution='10m',          # 中国区域用高精度底图
        save_path='ch12_map_china.png', show=False
    )

    # 7. RGB 合成 + 地图底图
    plot_rgb_on_map(
        FDI_PATH, r_ch=2, g_ch=3, b_ch=1,
        resolution=4000, subsample=4,
        save_path='rgb_map.png', show=False
    )

    # 8. GEO 太阳天顶角 + 地图底图
    plot_geo_field_on_map(
        GEO_PATH, dataset='NOMSunZenith',
        resolution=4000, subsample=4,
        cmap='RdYlBu_r',
        save_path='sunzenith_map.png', show=False
    )

    # 9. FDI vs GEO 对比（双图，均有底图）
    plot_fdi_geo_comparison_on_map(
        FDI_PATH, GEO_PATH,
        channel=12, geo_dataset='NOMSunZenith',
        resolution=4000, subsample=4,
        save_path='fdi_geo_map_compare.png', show=False
    )

    # 10. 坐标转换示例
    l, c = lonlat_to_linecolumn(lon=116.4, lat=39.9, resolution=4000)
    print(f"\n北京 (116.4°E, 39.9°N) → 行={l:.1f}, 列={c:.1f}")
    lon_back, lat_back = linecolumn_to_lonlat(l, c, resolution=4000)
    print(f"逆变换验证 → lon={lon_back:.4f}°, lat={lat_back:.4f}°")
