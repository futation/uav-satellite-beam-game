# game/scenario.py
"""场景构建：卫星位置、波束布局、UAV 群体的初始化。

合并自原项目 game_main.py / game_test.py 中两份重复的
create_test_uavs / create_test_beams，并修正：

1. 全部随机量（UAV 位置/速度、初始策略、雨衰）由同一个种子驱动的
   Generator 产生（原实现无种子，实验不可复现）；
2. 卫星位置由"星下点 = 大波束中心 + 配置高度"推导（原实现散落着与
   几何矛盾的随机 sat_distance / 硬编码 ECI 坐标）；
3. 波束 SNR 由共享信道模型计算后缓存 —— 雨衰固定，博弈迭代中速率
   只随负载变化。
"""

import numpy as np

import config.parameters as params
from channel import WirelessChannel
from environment import UAV
from utils.geometry import latlon_to_cartesian, geo_to_cartesian


def build_scenario(num_uavs=None, seed=None, num_beams=None,
                   force_offload=False, initial_offload_ratio=0.5):
    """构建一套自洽的仿真场景。

    参数:
        num_uavs: UAV 数量（默认取 SIMULATION['NUM_UAVS']）
        seed: 随机种子（默认取 RANDOM_SEED；None 表示参数文件也未设置时用系统熵源）
        num_beams: 使用的小波束数量（默认全部；测试可调小以加速）
        force_offload: True 则所有 UAV 初始即为卸载模式（纯波束选择博弈）
        initial_offload_ratio: 混合博弈下的初始卸载比例

    返回:
        (uavs, beams, channel, satellite_pos)

    波束 dict 字段:
        center/radius: 几何参数
        sat_distance: 卫星到波束中心距离 (m)（由几何推导）
        sat_comp_freq: 该波束可分得的计算频率 (Hz)
        snr: 缓存的信噪比（雨衰固定后不随负载变化）
        data_rate: 当前负载数下的单用户速率 (bps) = (B/K)·log2(1+snr)
        num_users: 当前波束内卸载用户数
    """
    rng = np.random.default_rng(
        seed if seed is not None else params.RANDOM_SEED)

    # ---- 卫星位置：星下点取大波束中心（场景自洽）----
    big_center = params.BEAM_PARAMS['BIG_BEAM_CENTER']
    big_radius = params.BEAM_PARAMS['BIG_BEAM_RADIUS']
    altitude = params.SATELLITE_PARAMS['altitude']
    satellite_pos = np.asarray(latlon_to_cartesian(big_center[0], big_center[1], altitude))

    # ---- 信道：雨衰抽样一次后固定 ----
    channel = WirelessChannel(rng=rng)

    # ---- 波束布局 ----
    centers = params.BEAM_PARAMS['SMALL_BEAM_CENTERS']
    if num_beams is not None:
        centers = centers[:num_beams]
    small_radius = params.BEAM_PARAMS['SMALL_BEAM_RADIUS']
    comp_freq = params.SATELLITE_PARAMS['comp_freq_per_beam']

    beams = []
    for i, center in enumerate(centers):
        beam_cart = np.asarray(geo_to_cartesian(center[0], center[1]))
        sat_distance = float(np.linalg.norm(satellite_pos - beam_cart))
        H_m = channel.calculate_channel_gain(satellite_pos, center, small_radius)
        snr = channel.calculate_snr(H_m)
        beams.append({
            'id': i,
            'center': center,
            'radius': small_radius,
            'sat_distance': sat_distance,
            'sat_comp_freq': comp_freq,
            'snr': snr,
            'data_rate': channel.calculate_data_rate(snr, 1),  # K=1 占位，引擎会刷新
            'num_users': 0,
        })

    # ---- UAV 群体：在撒点圆域内均匀分布（半径默认匹配波束阵列范围）----
    num_uavs = num_uavs or params.SIMULATION['NUM_UAVS']
    spawn_radius = params.BEAM_PARAMS['SPAWN_RADIUS']
    angles = rng.uniform(0, 2 * np.pi, num_uavs)
    radii = np.sqrt(rng.uniform(0, 1, num_uavs)) * spawn_radius
    lat_ref, lon_ref = big_center

    uavs = []
    for i in range(num_uavs):
        # 圆域均匀采样 -> 经纬度偏移（1 deg 纬度约 111 km，经度除以 cos(lat)）
        lat = lat_ref + radii[i] * np.sin(angles[i]) / 111000.0
        lon = lon_ref + radii[i] * np.cos(angles[i]) \
            / (111000.0 * np.cos(np.radians(lat_ref)))
        uav = UAV(i, (lat, lon),
                  initial_velocity=float(rng.uniform(
                      params.UAV_PARAMS['VELOCITY_MIN'],
                      params.UAV_PARAMS['VELOCITY_MAX'])))

        # 初始策略
        if force_offload:
            uav.current_mode = "offload"
            uav.selected_beam = beams[int(rng.integers(len(beams)))]
        elif rng.random() < initial_offload_ratio:
            uav.current_mode = "offload"
            uav.selected_beam = beams[int(rng.integers(len(beams)))]
        else:
            uav.current_mode = "local"
            uav.selected_beam = None

        uavs.append(uav)

    return uavs, beams, channel, satellite_pos
