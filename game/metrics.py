# game/metrics.py
"""结果统计指标（合并自原项目 equilibrium.py 的
calculate_equilibrium_metrics 与 offload_policy.py 的
analyze_group_behavior / calculate_social_welfare —— 三者原本大量重复）。
"""

import numpy as np


def social_welfare(uavs) -> float:
    """社会福利：所有 UAV 效用之和。"""
    return float(sum(u.last_utility_calculation['utility'] for u in uavs))


def summarize(uavs, beams) -> dict:
    """汇总一轮博弈结束后的系统指标。

    返回 dict:
        offload_ratio: 卸载比例
        avg/min/max_utility, utility_variance: 效用统计
        avg_local_utility / avg_offload_utility: 分模式平均效用
        beam_loads: {波束中心: 用户数}
        beam_avg_utilities: {波束中心: 该波束用户平均效用}
        social_welfare: 社会福利
    """
    utilities = [u.last_utility_calculation['utility'] for u in uavs]
    local_utils = [u.last_utility_calculation['utility'] for u in uavs
                   if u.current_mode == "local"]
    offload_utils = [u.last_utility_calculation['utility'] for u in uavs
                     if u.current_mode == "offload"]

    # 各波束负载与波束内平均效用
    beam_loads = {beam['center']: 0 for beam in beams}
    beam_utils = {beam['center']: [] for beam in beams}
    for uav in uavs:
        if uav.current_mode == "offload" and uav.selected_beam is not None:
            center = uav.selected_beam['center']
            beam_loads[center] += 1
            beam_utils[center].append(uav.last_utility_calculation['utility'])

    return {
        "offload_ratio": len(offload_utils) / len(uavs),
        "avg_utility": float(np.mean(utilities)),
        "min_utility": float(np.min(utilities)),
        "max_utility": float(np.max(utilities)),
        "utility_variance": float(np.var(utilities)),
        "avg_local_utility": float(np.mean(local_utils)) if local_utils else 0.0,
        "avg_offload_utility": float(np.mean(offload_utils)) if offload_utils else 0.0,
        "beam_loads": beam_loads,
        "beam_avg_utilities": {
            center: float(np.mean(v)) if v else 0.0
            for center, v in beam_utils.items()},
        "social_welfare": social_welfare(uavs),
    }
