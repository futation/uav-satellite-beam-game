# channel/wireless_channel.py
"""UAV -> LEO 卫星波束下行链路信道模型。

完整计算链（与原项目 utils/channel.py 的论文公式一致）：

    波束几何 (公式3) -> 卫星天线增益 (公式2) -> 雨衰 -> 路径损耗 (公式7)
    -> 信道增益 (公式1/8) -> SNR (公式9) -> 带宽均分香农速率 (公式10)

相对原实现的修正：
1. 随机数发生器可注入（`rng` 参数）—— 原实现直接用 `np.random` 全局态，
   且雨衰在每次实例化时重新抽样，导致同一负载下的速率不可复现地抖动；
2. 雨衰只在外部显式调用 `update_rain_attenuation()` 时更新，博弈迭代
   期间保持不变（同一信道实现下比较效用）；
3. 日志改用标准 logging 模块，import 时不再创建日志文件。
"""

import logging

import numpy as np

from config import parameters as params
from utils import geometry

logger = logging.getLogger(__name__)


class WirelessChannel:
    """雨衰固定的卫星波束信道（确定性，可复现）。"""

    def __init__(self, rng: np.random.Generator | None = None):
        """
        参数:
            rng: numpy 随机数发生器。传入场景级共享的 Generator 即可保证
                 与 UAV 初始位置等随机量同种子可复现；None 则使用系统熵源。
        """
        comm = params.COMMUNICATION
        self.fc = comm['CARRIER_FREQ']            # 载波频率 (Hz)
        self.B = comm['BANDWIDTH']                # 波束总带宽 (Hz)
        self.tx_power = comm['TX_POWER']          # 发射功率 (W)
        self.N0 = comm['NOISE_DENSITY']           # 噪声功率谱密度 (W/Hz)
        self.alpha = comm['PATH_LOSS_EXP']        # 路损指数
        self.eta = comm['ANTENNA_EFF']            # 天线口径效率
        self.G_R = 10 ** (comm['RX_GAIN_DB'] / 10)  # 接收增益（线性）
        self.rain_mean = comm['RAIN_ATTEN_MEAN']  # 雨衰均值 (dB)
        self.rain_var = comm['RAIN_ATTEN_VAR']    # 雨衰方差 (dB^2)

        self.rng = rng if rng is not None else np.random.default_rng()

        # 雨衰在整个博弈过程中固定（同一信道实现），初始即抽样一次
        self.current_rain_attenuation = self._draw_rain_attenuation()

        # 卫星位置缓存 {satellite_id: np.ndarray}
        self.satellite_positions = {}

        logger.info("Wireless channel model initialized (rain=%.2f dB)",
                    -10 * np.log10(self.current_rain_attenuation))

    # ------------------------------------------------------------------
    # 雨衰
    # ------------------------------------------------------------------
    def _draw_rain_attenuation(self) -> float:
        """从高斯分布 N(mean, var) 抽样雨衰 dB 值并转为线性系数。"""
        L_0_db = self.rng.normal(self.rain_mean, np.sqrt(self.rain_var))
        return 10 ** (-L_0_db / 10)

    def update_rain_attenuation(self):
        """显式更新雨衰（例如按时隙变化时由场景层调用；博弈迭代内不调用）。"""
        self.current_rain_attenuation = self._draw_rain_attenuation()

    def get_rain_attenuation(self) -> float:
        """当前雨衰线性系数。"""
        return self.current_rain_attenuation

    # ------------------------------------------------------------------
    # 链路计算
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_beam_geometry(satellite_pos, beam_center, beam_radius):
        """计算波束几何参数（公式3）。

        参数:
            satellite_pos: 卫星位置（地心笛卡尔坐标, m）
            beam_center: 波束中心 (纬度, 经度) (度)
            beam_radius: 波束覆盖半径 (m)

        返回:
            theta_m: 波束宽度 (rad)
            d_m_s: 卫星到波束边缘距离 (m)
            d_m: 卫星到波束中心距离 (m)
        """
        earth_radius = params.EARTH_RADIUS
        satellite_pos = np.asarray(satellite_pos)

        # 卫星高度（到地表距离）
        sat_norm = np.linalg.norm(satellite_pos)
        h_s = max(0.0, sat_norm - earth_radius)

        # 波束中心地表笛卡尔坐标 与 星-波束中心直线距离
        beam_cart = np.asarray(geometry.geo_to_cartesian(beam_center[0], beam_center[1]))
        d_m = np.linalg.norm(satellite_pos - beam_cart)

        # 星下点 与 波束中心到星下点的地表距离
        if sat_norm < 1e-6:
            sub_satellite_point = np.array([0.0, 0.0, earth_radius])
        else:
            sub_satellite_point = (satellite_pos / sat_norm) * earth_radius
        d_m_o = geometry.great_circle_distance(beam_cart, sub_satellite_point)

        # 卫星到波束边缘的最短距离
        d_m_s = np.sqrt(h_s ** 2 + (d_m_o - beam_radius) ** 2)

        # 波束宽度：远距离用小角度近似 θ_m ≈ 2r/d，否则用余弦定理
        numerator = d_m_s ** 2 + d_m ** 2 - beam_radius ** 2
        denominator = 2 * d_m_s * d_m

        if d_m > 1e6:
            theta_m = 2 * beam_radius / d_m
        elif h_s > 1e-3:
            if abs(denominator) < 1e-6:
                theta_m = np.pi / 2
            else:
                cos_theta = numerator / denominator
                if cos_theta > 1.0 or abs(cos_theta - 1.0) < 1e-10:
                    theta_m = 2 * beam_radius / d_m
                else:
                    theta_m = 2 * np.arccos(np.clip(cos_theta, -1.0, 1.0))
        else:
            theta_m = np.pi / 2

        return theta_m, d_m_s, d_m

    def calculate_transmitter_gain(self, theta_m) -> float:
        """抛物面天线增益 G_m = η·(70π/θ_m)²（公式2）。"""
        if theta_m < 1e-10:
            return 0.0
        return self.eta * (70 * np.pi / theta_m) ** 2

    def calculate_path_loss(self, d_m_s) -> float:
        """自由空间路损 L_m = (c/4πf_c)²·d^(-α)（公式7）。"""
        c = params.LIGHT_SPEED
        return (c / (4 * np.pi * self.fc)) ** 2 * d_m_s ** (-self.alpha)

    def calculate_channel_gain(self, satellite_pos, beam_center, beam_radius) -> float:
        """综合信道增益 H_m = G_m·G_R·L_0·L_m（公式1/8）。"""
        theta_m, d_m_s, _ = self.calculate_beam_geometry(
            satellite_pos, beam_center, beam_radius)
        G_m = self.calculate_transmitter_gain(theta_m)
        L_0 = self.get_rain_attenuation()
        L_m = self.calculate_path_loss(d_m_s)
        return G_m * self.G_R * L_0 * L_m

    def calculate_snr(self, H_m) -> float:
        """SNR = P_tx·H_m / (N0·B)（公式9）。"""
        return self.tx_power * H_m / (self.N0 * self.B)

    def calculate_data_rate(self, snr, num_users_in_beam) -> float:
        """带宽均分香农速率 R = (B/K)·log2(1+SNR)（公式10）。"""
        effective_bandwidth = self.B / max(1, num_users_in_beam)
        return effective_bandwidth * np.log2(1 + snr)

    # ------------------------------------------------------------------
    # 便捷接口
    # ------------------------------------------------------------------
    def update_satellite_position(self, satellite_id, position):
        """缓存卫星位置，供 get_data_rate_for_uav 使用。"""
        self.satellite_positions[satellite_id] = position

    def get_data_rate_for_uav(self, satellite_id, beam_center, beam_radius,
                              num_users_in_beam, satellite_pos=None):
        """完整链路：几何 -> 增益 -> SNR -> 速率。

        satellite_pos 未提供时从缓存取（未知 id 抛 ValueError）。
        """
        if satellite_pos is None:
            if satellite_id not in self.satellite_positions:
                raise ValueError(f"Satellite position for {satellite_id} not available")
            satellite_pos = self.satellite_positions[satellite_id]

        H_m = self.calculate_channel_gain(satellite_pos, beam_center, beam_radius)
        snr = self.calculate_snr(H_m)
        return self.calculate_data_rate(snr, num_users_in_beam)
