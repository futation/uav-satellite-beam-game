# tests/test_uav_physics.py
"""UAV 物理模型测试（关键断言提炼自原项目 1193 行手工验证脚本 uav_test.py）。"""

import numpy as np
import pytest

import config.parameters as params
from environment import UAV


@pytest.fixture()
def uav():
    return UAV(0, (20.0, 10.0), initial_velocity=15.0)


class TestLocalModel:
    def test_local_delay_is_cycles_over_freq(self, uav):
        expected = uav.task_cpu_cycles / uav.local_comp_freq
        assert uav.calculate_local_computation_delay() == pytest.approx(expected)

    def test_local_energy_is_kappa_c_f2(self, uav):
        expected = (uav.energy_eff_coeff * uav.task_cpu_cycles
                    * uav.local_comp_freq ** 2)
        assert uav.calculate_local_computation_energy() == pytest.approx(expected)

    def test_data_size_decoupled_from_cycles(self, uav):
        """关键修正：任务数据量必须独立于 CPU 周期数。"""
        assert uav.task_data_size == params.TASK_PARAMS['DATA_SIZE_BITS']
        assert uav.task_data_size != uav.task_cpu_cycles


class TestPropulsion:
    def test_hover_power_in_plausible_band(self, uav):
        """3kg 四旋翼悬停功率应在 50-500W 之间。"""
        p_hover = uav.calculate_vertical_power(0.0)
        assert 50.0 < p_hover < 500.0

    def test_power_grows_with_speed(self, uav):
        """推进功率随速度单调不减（诱导项在高速端近似抛物线增长）。"""
        p_low = uav.calculate_total_propulsion_power(5.0)
        p_high = uav.calculate_total_propulsion_power(20.0)
        assert p_high >= p_low

    def test_hover_energy_is_power_times_time(self, uav):
        p_hover = uav.calculate_vertical_power(0.0)
        assert uav.calculate_hover_energy(10.0) == pytest.approx(p_hover * 10.0)

    def test_local_total_energy_dominates_by_hover(self, uav):
        """本地总能耗 = 悬停能耗 + 计算能耗，且悬停项占主导。"""
        e_total = uav.calculate_local_total_energy()
        e_comp = uav.calculate_local_computation_energy()
        assert e_total == pytest.approx(uav.calculate_hover_energy(1.0) + e_comp)
        assert e_total > e_comp


class TestOffloadModel:
    def test_transmission_delay_uses_data_size(self, uav):
        """传输时延 = data_size / rate（而不是周期数 / rate）。"""
        uav.calculate_offload_total_delay(
            beam_center=(20.0, 10.0), beam_radius=300.0, data_rate=2e7,
            sat_comp_freq=1e10, num_users_in_beam=1, sat_distance=3e5)
        assert uav.last_delay_components['transmission'] \
            == pytest.approx(uav.task_data_size / 2e7)

    def test_zero_flight_delay_inside_beam(self, uav):
        """位于波束中心的 UAV 飞行时延为 0。"""
        uav_center = UAV(1, (20.0, 10.0), initial_velocity=15.0)
        delay = uav_center.calculate_flight_delay((20.0, 10.0), 300.0)
        assert delay == 0.0

    def test_satellite_delay_splits_by_users(self, uav):
        """卫星计算时延与波束用户数成反比（f_m 均分）。"""
        uav.calculate_offload_total_delay(
            beam_center=(20.0, 10.0), beam_radius=300.0, data_rate=2e7,
            sat_comp_freq=1e10, num_users_in_beam=1, sat_distance=3e5)
        d1 = uav.last_delay_components['computation']
        uav.calculate_offload_total_delay(
            beam_center=(20.0, 10.0), beam_radius=300.0, data_rate=2e7,
            sat_comp_freq=1e10, num_users_in_beam=5, sat_distance=3e5)
        d5 = uav.last_delay_components['computation']
        assert d5 == pytest.approx(d1 * 5)


class TestUtility:
    def test_local_utility_formula(self, uav):
        """本地效用 = -(α·T/T_ref + β·E/E_ref)。"""
        base = params.NORMALIZATION_BASELINE
        w = params.UTILITY_WEIGHTS
        expected = -(w['ALPHA'] * uav.calculate_local_computation_delay() / base['DELAY_REF']
                     + w['BETA'] * uav.calculate_local_total_energy() / base['ENERGY_REF'])
        assert uav.calculate_utility('local') == pytest.approx(expected)

    def test_normalization_makes_local_about_minus_one(self, uav):
        """基准取本地基线后，本地效用应接近 -(α+β) ≈ -0.8。"""
        assert -1.5 < uav.calculate_utility('local') <= -0.5

    def test_offload_requires_params(self, uav):
        with pytest.raises(ValueError):
            uav.calculate_utility('offload')

    def test_invalid_strategy_raises(self, uav):
        with pytest.raises(ValueError):
            uav.calculate_utility('fly')

    def test_utility_degrades_with_load(self, uav, satellite_pos=None):
        """卸载效用随波束负载增加而下降（拥塞外部性）。"""
        from utils import geometry
        sat_pos = geometry.latlon_to_cartesian(20.0, 10.0, params.SATELLITE_PARAMS['altitude'])
        from channel import WirelessChannel
        channel = WirelessChannel(rng=np.random.default_rng(0))
        H = channel.calculate_channel_gain(sat_pos, (20.0, 10.0), 300.0)
        snr = channel.calculate_snr(H)

        def util(k):
            return uav.calculate_utility(
                'offload', beam_center=(20.0, 10.0), beam_radius=300.0,
                data_rate=channel.calculate_data_rate(snr, k),
                sat_comp_freq=1e10, num_users_in_beam=k, sat_distance=3e5)

        u1, u8 = util(1), util(8)
        assert u1 > u8
