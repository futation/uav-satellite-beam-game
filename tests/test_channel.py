# tests/test_channel.py
"""信道模型测试（改造自原项目 Channel_test.py，增加可复现性断言）。"""

import numpy as np
import pytest

from channel import WirelessChannel
from config import parameters as params
from utils import geometry


@pytest.fixture()
def satellite_pos():
    """位于 (20°, 10°) 上空 500km 的卫星。"""
    return np.asarray(geometry.latlon_to_cartesian(
        20.0, 10.0, params.EARTH_RADIUS + 500e3))


def make_channel(seed=7):
    return WirelessChannel(rng=np.random.default_rng(seed))


class TestRainAttenuation:
    def test_seeded_channels_share_rain(self):
        """同种子的两个信道实例雨衰一致（可复现性核心）。"""
        c1, c2 = make_channel(7), make_channel(7)
        assert c1.get_rain_attenuation() == c2.get_rain_attenuation()

    def test_rain_update_changes_value(self):
        c = make_channel(7)
        old = c.get_rain_attenuation()
        c.update_rain_attenuation()
        assert c.get_rain_attenuation() != old

    def test_rain_in_reasonable_range(self):
        """雨衰 dB 值应落在 mean ± 4σ 内。"""
        comm = params.COMMUNICATION
        mean, std = comm['RAIN_ATTEN_MEAN'], np.sqrt(comm['RAIN_ATTEN_VAR'])
        c = make_channel(1)
        for _ in range(50):
            c.update_rain_attenuation()
            db = -10 * np.log10(c.get_rain_attenuation())
            assert mean - 4 * std <= db <= mean + 4 * std


class TestDataRate:
    def test_positive_for_extreme_snrs(self):
        c = make_channel(7)
        assert c.calculate_data_rate(1000.0, 1) > 0
        assert c.calculate_data_rate(0.1, 1) > 0

    def test_bandwidth_shared_exactly(self):
        """速率应精确等于 (B/K)·log2(1+SNR)。"""
        c = make_channel(7)
        snr = 500.0
        for k in (1, 3, 10):
            expected = c.B / k * np.log2(1 + snr)
            assert c.calculate_data_rate(snr, k) == pytest.approx(expected)

    def test_bandwidth_split_conservation(self):
        """带宽均分下 K 个用户总速率恒等于满带宽单用户速率。"""
        c = make_channel(7)
        snr = 1000.0
        single = c.calculate_data_rate(snr, 1)
        total = c.calculate_data_rate(snr, 10) * 10
        assert total == pytest.approx(single)


class TestLinkChain:
    def test_full_chain_positive(self, satellite_pos):
        """几何 -> 增益 -> SNR 全链路各量均为正。"""
        c = make_channel(7)
        theta_m, d_m_s, d_m = c.calculate_beam_geometry(
            satellite_pos, (20.0, 10.0), 3000.0)
        assert 0 < theta_m <= np.pi
        assert d_m_s > 0 and d_m > 0

        gain = c.calculate_transmitter_gain(theta_m)
        assert gain > 0

        H = c.calculate_channel_gain(satellite_pos, (20.0, 10.0), 3000.0)
        assert H > 0

        snr = c.calculate_snr(H)
        assert snr > 0

        rate = c.calculate_data_rate(snr, 1)
        assert rate > 0

    def test_seeded_rates_reproducible(self, satellite_pos):
        """同种子两次构建的信道，相同负载下速率完全一致。"""
        c1, c2 = make_channel(7), make_channel(7)
        r1 = c1.get_data_rate_for_uav("SAT", (20.0, 10.0), 3000.0, 5,
                                      satellite_pos=satellite_pos)
        r2 = c2.get_data_rate_for_uav("SAT", (20.0, 10.0), 3000.0, 5,
                                      satellite_pos=satellite_pos)
        assert r1 == r2

    def test_unknown_satellite_raises(self, satellite_pos):
        c = make_channel(7)
        with pytest.raises(ValueError):
            c.get_data_rate_for_uav("UNKNOWN", (20.0, 10.0), 3000.0, 1)


class TestGeometry:
    def test_latlon_roundtrip(self):
        lat, lon, alt = 20.0, 10.0, 100.0
        x, y, z = geometry.latlon_to_cartesian(lat, lon, alt)
        lat2, lon2, alt2 = geometry.cartesian_to_latlon(x, y, z)
        assert lat2 == pytest.approx(lat, abs=1e-9)
        assert lon2 == pytest.approx(lon, abs=1e-9)
        assert alt2 == pytest.approx(alt, abs=1e-6)

    def test_surface_distance_1deg_latitude(self):
        """纬度差 1° 的大圆距离约 111.19 km。"""
        d = geometry.calculate_distance_between_points((0.0, 0.0), (1.0, 0.0))
        assert d == pytest.approx(111194.9, rel=1e-4)
