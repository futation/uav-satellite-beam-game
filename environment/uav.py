# environment/uav.py
"""无人机（UAV）实体：物理模型 + 归一化效用计算。

职责边界（相对原项目的瘦身）：
- 只保留物理建模与效用计算：四旋翼推进功率（公式19-24）、本地计算
  时延/能耗（公式12/13）、卸载时延/能耗（公式37/38）、归一化多目标
  效用（公式32 + 基准归一化）；
- 删除了原类中 3 套重复效用实现（原始/对数）与 4 个 make_*_decision
  贪心决策函数——策略更新是博弈层（game/engine.py）的职责，UAV 只
  是"被询问效用"的物理实体；
- 任务数据量 DATA_SIZE_BITS 与计算周期数 CPU_CYCLES 解耦（原实现把
  周期数当比特数传输，导致传输时延爆炸、卸载恒劣于本地）；
- 删除从未被读取的 battery 属性。
"""

import numpy as np

import config.parameters as params
from utils.geometry import (
    latlon_to_cartesian,
    cartesian_to_latlon,
    calculate_distance_between_points,
)
from config.parameters import EARTH_RADIUS


class UAV:
    """固定位置的四旋翼无人机，携带一个可本地计算或卸载的计算任务。"""

    def __init__(self, uav_id, initial_position, initial_velocity=None,
                 task_cpu_cycles=None, task_data_size=None):
        """
        参数:
            uav_id: 唯一标识 (int)
            initial_position: 初始位置 (纬度, 经度) (度)
            initial_velocity: 前飞速度 (m/s)，默认取参数配置
            task_cpu_cycles: 任务计算量 (cycles)，默认取参数配置
            task_data_size: 任务数据量 (bits)，默认取参数配置
        """
        self.id = uav_id

        # ---- 博弈状态 ----
        self.current_mode = "local"    # "local" 或 "offload"
        self.selected_beam = None      # 卸载模式下选中的波束 dict（本地模式为 None）
        self.last_utility_calculation = {'utility': 0.0, 'strategy': 'uninitialized'}

        # ---- 位置 ----
        initial_lat, initial_lon = initial_position
        self.position = latlon_to_cartesian(
            initial_lat, initial_lon, params.UAV_PARAMS['ALTITUDE'])
        self._validate_position()

        # ---- 机体参数 ----
        up = params.UAV_PARAMS
        self.velocity = initial_velocity or (up['VELOCITY_MIN'] + up['VELOCITY_MAX']) / 2
        self.mass = up['MASS']                        # kg
        self.rotor_radius = up['ROTOR_RADIUS']        # m
        self.drag_coeff = up['DRAG_COEFF']            # 叶片阻力系数
        self.ref_area = up['REF_AREA']                # m^2
        self.drag_coeff_uav = up['DRAG_COEFF_UAV']    # 机身阻力系数
        self.rotor_angular_vel = up['ROTOR_ANGULAR_VEL']  # rad/s
        self.tip_speed = up['TIP_SPEED']              # m/s
        self.num_rotors = up['NUM_ROTORS']
        self.rotor_blade_count = up['BLADE_COUNT']
        self.rotor_blade_chord = up['BLADE_CHORD']    # m

        # ---- 任务参数 ----
        tp = params.TASK_PARAMS
        self.task_cpu_cycles = task_cpu_cycles or tp['CPU_CYCLES']  # cycles
        self.task_data_size = task_data_size or tp['DATA_SIZE_BITS']  # bits
        self.local_comp_freq = tp['LOCAL_COMP_FREQ']  # Hz
        self.energy_eff_coeff = tp['ENERGY_EFF_COEFF']  # J/(cycle·Hz^2)

        # ---- 通信参数 ----
        self.tx_power = params.COMMUNICATION['TX_POWER']  # W

        # 调试缓存：最近一次卸载时延/能耗的分量
        self.last_delay_components = None
        self.last_energy_components = None

    # ------------------------------------------------------------------
    # 状态与位置
    # ------------------------------------------------------------------
    def _validate_position(self):
        """校验高度在允许范围内。"""
        height = np.linalg.norm(self.position) - EARTH_RADIUS
        if height < params.UAV_PARAMS['MIN_ALTITUDE']:
            raise ValueError(f"UAV {self.id} altitude too low: {height:.1f} m")
        if height > params.UAV_PARAMS['MAX_ALTITUDE']:
            raise ValueError(f"UAV {self.id} altitude too high: {height:.1f} m")

    @property
    def latitude(self):
        lat, _, _ = cartesian_to_latlon(*self.position)
        return lat

    @property
    def longitude(self):
        _, lon, _ = cartesian_to_latlon(*self.position)
        return lon

    @property
    def height_above_ground(self):
        return np.linalg.norm(self.position) - EARTH_RADIUS

    def __repr__(self):
        height = self.height_above_ground
        return (f"UAV(id={self.id}, lat={self.latitude:.6f}, lon={self.longitude:.6f}, "
                f"alt={height:.1f}m, mode={self.current_mode})")

    # ------------------------------------------------------------------
    # 本地计算模型
    # ------------------------------------------------------------------
    def calculate_local_computation_delay(self) -> float:
        """本地计算时延（公式12）: T_local = C / f_local。"""
        if self.task_cpu_cycles <= 0:
            return 0.0
        return self.task_cpu_cycles / self.local_comp_freq

    def calculate_local_computation_energy(self) -> float:
        """本地计算能耗（公式13）: E_local = κ·C·f_local²。"""
        if self.task_cpu_cycles <= 0:
            return 0.0
        return self.energy_eff_coeff * self.task_cpu_cycles * self.local_comp_freq ** 2

    # ------------------------------------------------------------------
    # 旋翼推进功率模型（标准四旋翼模型）
    # ------------------------------------------------------------------
    def calculate_vertical_power(self, velocity=None) -> float:
        """垂直功率分量 P_vertical（公式19-22）。

        公式19: P0 = (1/8)·ρ·Cd0·N·c·(ΩR)³          （型阻功率）
        公式20: v0 = sqrt(m·g / (2ρ·s_rotor))        （悬停诱导速度）
        公式22: P_v = P0·(1+3V²/U_tip²)
                    + m·g·v0·sqrt(1 + V⁴/(4v0⁴) − V²/(2v0²)) （诱导功率）
        """
        v = velocity if velocity is not None else self.velocity
        rho = params.AIR_DENSITY
        g = params.GRAVITY

        # 所有旋翼的总盘面积
        s_rotor = self.num_rotors * np.pi * self.rotor_radius ** 2

        # 公式19: 型阻功率基准值（桨叶总面积 = 桨叶数×弦长×旋翼半径）
        total_blade_area = self.rotor_blade_count * self.rotor_blade_chord * self.rotor_radius
        p0 = (1 / 8) * rho * self.drag_coeff * total_blade_area \
            * self.rotor_angular_vel ** 3 * self.rotor_radius ** 3

        # 公式20: 悬停诱导速度
        v0 = np.sqrt(self.mass * g / (2 * rho * s_rotor))

        # 诱导功率基准值
        p1 = self.mass * g * v0

        # 公式22
        term1 = p0 * (1 + 3 * v ** 2 / self.tip_speed ** 2)
        term2 = p1 * np.sqrt(1 + v ** 4 / (4 * v0 ** 4) - v ** 2 / (2 * v0 ** 2))
        return term1 + term2

    def calculate_horizontal_power(self, velocity=None) -> float:
        """机身阻力功率 P_horizontal = (d0/2)·ρ·S_A·V³（公式23）。"""
        v = velocity if velocity is not None else self.velocity
        rho = params.AIR_DENSITY
        return self.drag_coeff_uav / 2 * rho * self.ref_area * v ** 3

    def calculate_total_propulsion_power(self, velocity=None) -> float:
        """总推进功率 P_total = P_vertical + P_horizontal（公式24）。"""
        v = velocity if velocity is not None else self.velocity
        return self.calculate_vertical_power(v) + self.calculate_horizontal_power(v)

    def calculate_hover_energy(self, hover_time) -> float:
        """悬停能耗（公式29/30）: E_hover = P_vertical(V=0)·t_hover。"""
        return self.calculate_vertical_power(0.0) * hover_time

    def calculate_local_total_energy(self) -> float:
        """本地策略总能耗 = 本地计算期间悬停能耗 + 计算能耗。"""
        local_delay = self.calculate_local_computation_delay()
        return self.calculate_hover_energy(local_delay) \
            + self.calculate_local_computation_energy()

    # ------------------------------------------------------------------
    # 卸载模型
    # ------------------------------------------------------------------
    def _calculate_distance_to_beam(self, beam_center) -> float:
        """UAV 到波束中心的地表距离 (m)。"""
        return calculate_distance_between_points(
            (self.latitude, self.longitude), beam_center)

    def calculate_flight_delay(self, beam_center, beam_radius, velocity=None) -> float:
        """飞行到波束覆盖边缘的时延（公式14）: T_fly = d/V，波束内则为 0。"""
        v = velocity if velocity is not None else self.velocity
        distance_to_center = self._calculate_distance_to_beam(beam_center)
        distance_to_edge = max(0.0, distance_to_center - beam_radius)
        return distance_to_edge / v

    def calculate_offload_total_delay(self, beam_center, beam_radius, data_rate,
                                      sat_comp_freq, num_users_in_beam, sat_distance):
        """卸载策略总时延（公式37）。

        T = T_fly + D/R + d_sat/c + C/(f_m/K)
          = 飞行 + 传输 + 信号传播 + 卫星计算（计算频率按波束内用户均分）
        """
        flight_delay = self.calculate_flight_delay(beam_center, beam_radius)

        # 传输时延：数据量（bits）/ 速率（bps）
        transmission_delay = self.task_data_size / data_rate

        # 传播时延
        propagation_delay = sat_distance / params.LIGHT_SPEED

        # 卫星计算时延（公式16/17: f_m 均分给 K 个用户）
        effective_comp_freq = sat_comp_freq / max(1, num_users_in_beam)
        computation_delay = self.task_cpu_cycles / effective_comp_freq

        total_delay = flight_delay + transmission_delay \
            + propagation_delay + computation_delay

        self.last_delay_components = {
            'flight': flight_delay,
            'transmission': transmission_delay,
            'propagation': propagation_delay,
            'computation': computation_delay,
        }
        return total_delay

    def calculate_offload_total_energy(self, beam_center, beam_radius, data_rate,
                                       sat_comp_freq, num_users_in_beam, sat_distance):
        """卸载策略总能耗（公式38）。

        E = E_trans + E_sat + E_fly + E_hover
          = P_tx·T_trans + P_sat·T_comp + P_total·T_fly + P_hover·(T_trans+T_prop+T_comp)

        建模说明：卫星计算能耗按卫星功率×计算时延计入 UAV 总成本
        （激励 UAV 考虑占用卫星计算资源的时间成本），见 README 的
        Modeling notes 一节。
        """
        flight_delay = self.calculate_flight_delay(beam_center, beam_radius)
        transmission_delay = self.task_data_size / data_rate
        propagation_delay = sat_distance / params.LIGHT_SPEED
        effective_comp_freq = sat_comp_freq / max(1, num_users_in_beam)
        computation_delay = self.task_cpu_cycles / effective_comp_freq

        transmission_energy = self.tx_power * transmission_delay
        sat_power = params.SATELLITE_PARAMS['COMP_POWER']
        sat_computation_energy = sat_power * computation_delay
        flight_energy = self.calculate_total_propulsion_power() * flight_delay
        hover_time = transmission_delay + propagation_delay + computation_delay
        hover_energy = self.calculate_hover_energy(hover_time)

        total_energy = (transmission_energy + sat_computation_energy
                        + flight_energy + hover_energy)

        self.last_energy_components = {
            'transmission': transmission_energy,
            'sat_computation': sat_computation_energy,
            'flight': flight_energy,
            'hover': hover_energy,
        }
        return total_energy

    # ------------------------------------------------------------------
    # 归一化效用（博弈层唯一使用的效用函数）
    # ------------------------------------------------------------------
    def calculate_utility(self, strategy, **kwargs) -> float:
        """归一化多目标效用（公式32 + 基准归一化）。

        U = -( ALPHA·T/T_ref + BETA·E/E_ref + GAMMA·C_res/C_ref )

        基准值取"本地计算基线"（见 config/parameters.py 注释），因此
        U 是相对本地计算的相对成本：U ≈ -0.8 与本地相当，越接近 0 越优。

        参数:
            strategy: 'local' 或 'offload'
            kwargs（卸载策略必须提供）:
                beam_center: 波束中心 (纬度, 经度)
                beam_radius: 波束半径 (m)
                data_rate: 数据速率 (bps)
                sat_comp_freq: 卫星计算频率 (Hz)
                num_users_in_beam: 波束内用户数
                sat_distance: 卫星距离 (m)
        """
        if strategy == 'local':
            delay = self.calculate_local_computation_delay()
            energy = self.calculate_local_total_energy()
            resource = 0.0
        elif strategy == 'offload':
            required = ['beam_center', 'beam_radius', 'data_rate',
                        'sat_comp_freq', 'num_users_in_beam', 'sat_distance']
            missing = [p for p in required if p not in kwargs]
            if missing:
                raise ValueError(f"Offload utility requires params: {missing}")

            delay = self.calculate_offload_total_delay(
                kwargs['beam_center'], kwargs['beam_radius'], kwargs['data_rate'],
                kwargs['sat_comp_freq'], kwargs['num_users_in_beam'],
                kwargs['sat_distance'])
            energy = self.calculate_offload_total_energy(
                kwargs['beam_center'], kwargs['beam_radius'], kwargs['data_rate'],
                kwargs['sat_comp_freq'], kwargs['num_users_in_beam'],
                kwargs['sat_distance'])
            # 资源成本 ∝ 任务周期数 × 波束用户数
            resource = self.task_cpu_cycles * kwargs['num_users_in_beam'] \
                * params.SATELLITE_PARAMS['RESOURCE_COST_FACTOR']
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        base = params.NORMALIZATION_BASELINE
        norm_delay = delay / max(base['DELAY_REF'], 1e-6)
        norm_energy = energy / max(base['ENERGY_REF'], 1e-6)
        norm_resource = resource / max(base['RESOURCE_REF'], 1e-6)

        w = params.UTILITY_WEIGHTS
        return -(w['ALPHA'] * norm_delay
                 + w['BETA'] * norm_energy
                 + w['GAMMA'] * norm_resource)
