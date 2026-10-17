# game/equilibrium.py
"""纳什均衡验证（纯函数，无副作用）。

保留自原项目的核心思想 —— 反事实验证：对每个 UAV 逐一模拟"单方面
偏离到其他波束/本地计算"后的效用，若任何偏离能带来严格正的效用增益，
则当前策略组合不是纳什均衡。

相对原实现的修正：
1. 不再 deepcopy 全部 UAV 与波束（原实现每次验证 O(N) 次 deepcopy，
   其中只有被测 UAV 的标量参数真正被用到）—— 效用是纯算术函数，
   直接用移动后的负载数值计算即可；
2. 复用共享信道的缓存 SNR（原实现每次验证新建 WirelessChannel，
   雨衰重新抽样，均衡判定被随机噪声污染，且与引擎所用的信道实现
   不一致）；
3. 通用 NE 验证 `check_general_ne` 重写：统一使用归一化效用（原实现
   用对数效用候选值与线性效用现值比较，量纲不一致），且不再篡改
   UAV 状态（原实现调用的 make_log_decision 会改写 current_mode）。
"""

import numpy as np


def _beam_counts(uavs, beams):
    """从 UAV 状态重新统计各波束用户数（不信任缓存值）。"""
    counts = [0] * len(beams)
    for uav in uavs:
        if uav.current_mode == "offload" and uav.selected_beam is not None:
            for j, beam in enumerate(beams):
                if beam['center'] == uav.selected_beam['center']:
                    counts[j] += 1
                    break
    return counts


def _offload_utility(uav, beam, k, channel):
    """UAV 卸载到 beam、该波束共 k 个用户时的效用。"""
    return uav.calculate_utility(
        'offload',
        beam_center=beam['center'],
        beam_radius=beam['radius'],
        data_rate=channel.calculate_data_rate(beam['snr'], k),
        sat_comp_freq=beam['sat_comp_freq'],
        num_users_in_beam=k,
        sat_distance=beam['sat_distance'],
    )


def check_offload_only_ne(uavs, beams, channel):
    """强制卸载（纯波束选择拥塞博弈）下的纳什均衡验证。

    对每个卸载 UAV 检查：单方面移动到任一其他波束（自身离开使原波束
    负载 -1、目标波束 +1，速率随负载重算）能否获得严格正的效用增益。

    返回:
        (is_equilibrium, explanation)
    """
    counts = _beam_counts(uavs, beams)
    deviating = []

    for uav in uavs:
        if uav.current_mode != "offload" or uav.selected_beam is None:
            continue

        current_idx = next(
            (j for j, b in enumerate(beams)
             if b['center'] == uav.selected_beam['center']), None)
        if current_idx is None:
            continue

        current_utility = _offload_utility(
            uav, beams[current_idx], counts[current_idx], channel)

        for j, beam in enumerate(beams):
            if j == current_idx:
                continue
            new_utility = _offload_utility(uav, beam, counts[j] + 1, channel)
            if new_utility > current_utility:
                deviating.append({
                    "uav_id": uav.id,
                    "current_beam": uav.selected_beam['center'],
                    "proposed_beam": beam['center'],
                    "utility_gain": new_utility - current_utility,
                })
                break  # 找到一个有利偏离即可

    is_equilibrium = len(deviating) == 0
    if is_equilibrium:
        explanation = ("No UAV can improve its utility by unilaterally "
                       "switching to another beam: Nash equilibrium.")
    else:
        lines = [
            f"{len(deviating)} UAV(s) can improve by switching beams:"
        ]
        for dev in deviating:
            lines.append(
                f"  UAV {dev['uav_id']}: {dev['current_beam']} -> "
                f"{dev['proposed_beam']} (gain {dev['utility_gain']:.4f})")
        explanation = "\n".join(lines)

    return is_equilibrium, explanation


def check_general_ne(uavs, beams, channel):
    """混合博弈（local / offload）下的纳什均衡验证。

    对每个 UAV 检查全部单方面偏离：
    - 卸载中的 UAV -> 本地计算，或 -> 任一其他波束（K_j+1）；
    - 本地的 UAV -> 任一波束（K_j+1）。

    全程使用归一化效用、只读不写，无任何状态副作用。

    返回:
        (is_equilibrium, explanation)
    """
    counts = _beam_counts(uavs, beams)
    deviating = []

    for uav in uavs:
        if uav.current_mode == "offload" and uav.selected_beam is not None:
            current_idx = next(
                (j for j, b in enumerate(beams)
                 if b['center'] == uav.selected_beam['center']), None)
            if current_idx is None:
                continue
            current_utility = _offload_utility(
                uav, beams[current_idx], counts[current_idx], channel)

            # 偏离 1：转本地计算
            local_utility = uav.calculate_utility('local')
            if local_utility > current_utility:
                deviating.append({
                    "uav_id": uav.id,
                    "current": f"beam {uav.selected_beam['center']}",
                    "proposed": "local",
                    "utility_gain": local_utility - current_utility,
                })
                continue

            # 偏离 2：切换到其他波束
            found = False
            for j, beam in enumerate(beams):
                if j == current_idx:
                    continue
                new_utility = _offload_utility(uav, beam, counts[j] + 1, channel)
                if new_utility > current_utility:
                    deviating.append({
                        "uav_id": uav.id,
                        "current": f"beam {uav.selected_beam['center']}",
                        "proposed": f"beam {beam['center']}",
                        "utility_gain": new_utility - current_utility,
                    })
                    found = True
                    break
            if found:
                continue
        else:
            # 本地 UAV：尝试卸载到任一波束
            current_utility = uav.calculate_utility('local')
            for j, beam in enumerate(beams):
                new_utility = _offload_utility(uav, beam, counts[j] + 1, channel)
                if new_utility > current_utility:
                    deviating.append({
                        "uav_id": uav.id,
                        "current": "local",
                        "proposed": f"beam {beam['center']}",
                        "utility_gain": new_utility - current_utility,
                    })
                    break

    is_equilibrium = len(deviating) == 0
    if is_equilibrium:
        explanation = ("No UAV can improve its utility by unilaterally "
                       "changing its strategy: Nash equilibrium.")
    else:
        lines = [f"{len(deviating)} UAV(s) can improve by deviating:"]
        for dev in deviating:
            lines.append(
                f"  UAV {dev['uav_id']}: {dev['current']} -> {dev['proposed']} "
                f"(gain {dev['utility_gain']:.4f})")
        explanation = "\n".join(lines)

    return is_equilibrium, explanation


def price_of_anarchy(uavs, beams, channel, n_samples=64, rng=None):
    """近似帕累托/最优社会效用对比（可选分析工具）。

    通过随机采样若干可行的策略组合（含当前状态），估计最优社会福利，
    再计算 PoA = 最优社会福利 / 当前社会福利（>= 1，越接近 1 越好）。
    仅在波束选择维度上采样（混合模式下保持当前卸载集合不变）。

    返回:
        (poa, best_welfare, current_welfare)
    """
    utilities_now = [u.last_utility_calculation['utility'] for u in uavs]
    current_welfare = float(np.sum(utilities_now))
    best_welfare = current_welfare

    rng = rng or np.random.default_rng(0)
    n_uavs = len(uavs)
    offloading = [u for u in uavs if u.current_mode == "offload"]

    for _ in range(n_samples):
        # 随机把每个卸载 UAV 分配到某个波束，统计该组合的社会福利
        counts = [0] * len(beams)
        assignment = []
        for u in offloading:
            j = int(rng.integers(len(beams)))
            assignment.append(j)
            counts[j] += 1

        welfare = 0.0
        for u, j in zip(offloading, assignment):
            welfare += _offload_utility(u, beams[j], counts[j], channel)
        for u in uavs:
            if u.current_mode != "offload" or u.selected_beam is None:
                welfare += u.calculate_utility('local')
        best_welfare = max(best_welfare, welfare)

    if current_welfare == 0:
        return float('inf'), best_welfare, current_welfare
    return best_welfare / current_welfare, best_welfare, current_welfare
