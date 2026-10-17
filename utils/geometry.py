# utils/geometry.py
"""几何与坐标转换工具。

约定：所有角度参数为弧度（函数名带 lat/lon 的除外，它们接受度）；
距离单位与输入的地球半径一致（本项目统一为米）。

自原项目迁移时的改动：删除了未被任何调用方使用的
`calculate_uav_sat_distance`、`euclidean_distance`、注释掉的"纪念版"
函数，以及文件中间重复的 `import numpy`。
"""

import numpy as np
from typing import Tuple

from config.parameters import EARTH_RADIUS


# ------------------------------------------------------------------
# 卫星轨道球坐标（ECI） helpers —— 供 environment/satellite.py 使用
# ------------------------------------------------------------------

def theta_sat(inclination_rad: float, nu_rad: float) -> float:
    """卫星在 ECI 坐标系中的极角 θ（天顶角，弧度）。

    cosθ = sin(i)·sin(ν)，i 为轨道倾角，ν 为真近点角。
    """
    cos_theta = np.sin(inclination_rad) * np.sin(nu_rad)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def phi_sat(Omega_rad: float, i_rad: float, nu_rad: float) -> float:
    """卫星在 ECI 坐标系中的方位角 φ（弧度）。

    φ = Ω + arctan2(sinν·cos i, cosν)
    """
    y_proj = np.sin(nu_rad) * np.cos(i_rad)
    x_proj = np.cos(nu_rad)
    return Omega_rad + np.arctan2(y_proj, x_proj)


def spherical_to_cartesian(
    r: float, theta_rad: float, phi_rad: float
) -> Tuple[float, float, float]:
    """球坐标 -> 直角坐标。

    x = r·sinθ·cosφ, y = r·sinθ·sinφ, z = r·cosθ
    输出单位与 r 相同。
    """
    x = r * np.sin(theta_rad) * np.cos(phi_rad)
    y = r * np.sin(theta_rad) * np.sin(phi_rad)
    z = r * np.cos(theta_rad)
    return (x, y, z)


# ------------------------------------------------------------------
# 大地坐标（lat/lon） <-> ECEF 笛卡尔坐标 —— 供 UAV/波束/信道使用
# ------------------------------------------------------------------

def latlon_to_cartesian(lat, lon, alt, R_earth=EARTH_RADIUS):
    """经纬度（度）+ 海拔（m）-> ECEF 直角坐标 (x, y, z) (m)。"""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    x = (R_earth + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (R_earth + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (R_earth + alt) * np.sin(lat_rad)

    return (x, y, z)


def cartesian_to_latlon(x, y, z, R_earth=EARTH_RADIUS):
    """ECEF 直角坐标 -> (纬度, 经度, 海拔)（度, 度, m）。"""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lat_rad = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    lon_rad = np.arctan2(y, x)
    alt = r - R_earth
    return (np.degrees(lat_rad), np.degrees(lon_rad), alt)


def geo_to_cartesian(lat, lon, R_earth=EARTH_RADIUS):
    """地表经纬度（度）-> 3D 笛卡尔坐标（海拔 0）。"""
    return latlon_to_cartesian(lat, lon, 0, R_earth)


# ------------------------------------------------------------------
# 距离计算
# ------------------------------------------------------------------

def calculate_surface_distance(point1, point2, radius=EARTH_RADIUS):
    """地心角 × 半径：地球表面两点的大圆距离 (m)。输入为 3D 坐标。"""
    dot_product = np.dot(point1, point2)
    norm_product = np.linalg.norm(point1) * np.linalg.norm(point2)
    cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
    return radius * np.arccos(cos_theta)


def great_circle_distance(point1, point2, R_earth=EARTH_RADIUS):
    """两点间大圆距离 (m)，输入为 ECEF 坐标（带零向量保护）。"""
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)

    norm1 = np.linalg.norm(point1)
    norm2 = np.linalg.norm(point2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    p1_norm = point1 / norm1
    p2_norm = point2 / norm2
    dot_product = np.clip(np.dot(p1_norm, p2_norm), -1.0, 1.0)
    return np.arccos(dot_product) * R_earth


def calculate_distance_between_points(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    R_earth: float = EARTH_RADIUS,
) -> float:
    """两个地理坐标 (纬度, 经度)（度）之间的地表大圆距离 (m)。"""
    cartesian1 = geo_to_cartesian(point1[0], point1[1], R_earth)
    cartesian2 = geo_to_cartesian(point2[0], point2[1], R_earth)
    return calculate_surface_distance(cartesian1, cartesian2, R_earth)


def calculate_line_of_sight_distance(point1, point2):
    """两点直线距离（不考虑地球曲率）。"""
    return np.linalg.norm(np.asarray(point1) - np.asarray(point2))


def normalize_vector(vector):
    """归一化向量（零向量原样返回）。"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
