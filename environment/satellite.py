# environment/satellite.py
"""卫星实体：圆轨道运动学模型（开普勒周期）。

相对原项目的瘦身：剥离了 `create_orbit_actor` 及其对 VTK 可视化层
的 import 依赖（原实现导致 `import environment.satellite` 会连带加载
vtk）。3D 轨道线绘制移至可选组件 viz/ 中。
"""

import numpy as np
from typing import Tuple

from utils.geometry import spherical_to_cartesian, theta_sat, phi_sat
from config.parameters import EARTH_RADIUS, EARTH_MU


class Satellite:
    """圆轨道卫星（ECI 坐标系，位置单位为米）。"""

    def __init__(self, altitude: float, inclination: float, raan: float):
        """
        参数:
            altitude: 轨道高度 (m)
            inclination: 轨道倾角 (deg)
            raan: 升交点赤经 (deg)
        """
        self.inclination_rad = np.radians(inclination)
        self.raan_rad = np.radians(raan)
        self.altitude = float(altitude)

        # 轨道参数：半长轴与开普勒周期
        self.semi_major_axis = EARTH_RADIUS + self.altitude
        self.period = 2 * np.pi * np.sqrt(self.semi_major_axis ** 3 / EARTH_MU)
        self.mean_motion = 2 * np.pi / self.period

        self.true_anomaly = 0.0
        self.position = self.calculate_position(0.0)

    def calculate_position(self, t: float) -> Tuple[float, float, float]:
        """t 时刻的 ECI 位置 (m)。圆轨道假设：真近点角 = 平均运动 × t。"""
        self.true_anomaly = self.mean_motion * t
        theta = theta_sat(self.inclination_rad, self.true_anomaly)
        phi = phi_sat(self.raan_rad, self.inclination_rad, self.true_anomaly)
        return spherical_to_cartesian(self.semi_major_axis, theta, phi)

    def update_position(self, t: float):
        """推进到 t 时刻并缓存位置。"""
        self.position = self.calculate_position(t)

    def get_position(self) -> Tuple[float, float, float]:
        """当前 ECI 位置 (m)。"""
        return self.position

    def get_orbital_parameters(self) -> dict:
        """当前轨道参数（弧度制）。"""
        return {
            'inclination': self.inclination_rad,
            'raan': self.raan_rad,
            'true_anomaly': self.true_anomaly,
            'altitude': self.altitude,
        }
