# game/engine.py
"""波束选择博弈引擎（重写版）。

两种可切换的策略更新规则（`update_rule` 参数）：

- `"best_response"` —— 局部学习规则（保留自原项目）：效用低于群体平均
  的 UAV 才考虑变更，在候选动作中取最优；仅当相对改进量超过阈值且以
  LEARNING_RATE 的概率（惯性）执行切换。概念上接近 log-linear learning
  的确定性部分 + 随机惯性。
- `"replicator"` —— 成对比例模仿（pairwise proportional imitation，
  Helbing 1992 / Traulsen et al. 2006）：随机抽取一个同伴，若同伴效用
  更高，则以 P = ΔU/(ΔU+scale) 的概率模仿其策略。这是复制者动态
  dx/dt = x·(π̄_strategy − π) 的标准个体级随机实现。

相对原实现修复的问题：
1. 【致命】混合博弈（local/offload）下重算效用时不分支处理 local UAV，
   对 selected_beam=None 解引用直接 TypeError —— 混合入口从未能运行；
2. `force_offload` 原本是死参数（强制卸载只靠入口脚本"恰好"全初始为
   offload 实现），现在真正在引擎内限制候选动作空间；
3. 循环内手工增减 beam['num_users'] 的无效记账（马上被全量重算覆盖）
   已删除，负载只由 `_refresh_loads_and_rates()` 一处维护；
4. 每轮新建 WirelessChannel（雨衰每轮重抽样，速率无谓抖动）→ 复用
   场景级共享实例，SNR 只算一次，速率只随负载变化；
5. 效用改进阈值由绝对值 3.0（依赖效用量纲）改为相对比例；
6. 收敛判据恢复启用（策略变化率 + 效用方差窗口），并修复进度条计数。
"""

import logging

import numpy as np
from tqdm import tqdm

import config.parameters as params
from game.equilibrium import check_offload_only_ne, check_general_ne

logger = logging.getLogger(__name__)

UPDATE_RULES = ("best_response", "replicator")


class BeamSelectionGame:
    """UAV 波束选择博弈（强制卸载模式 = 拥塞博弈；混合模式 = local/卸载二部决策）。"""

    def __init__(self, uavs, beams, channel, satellite_pos,
                 update_rule="best_response", force_offload=False,
                 max_iterations=None, check_equilibrium_each_iter=True):
        """
        参数:
            uavs: UAV 列表（selected_beam 必须引用 beams 中的元素或为 None）
            beams: 波束 dict 列表（由 game.scenario.build_scenario 构建）
            channel: 共享 WirelessChannel 实例（雨衰已固定）
            satellite_pos: 卫星 ECI 位置
            update_rule: "best_response" 或 "replicator"
            force_offload: True 则动作空间限制为"选择哪个波束"（纯拥塞博弈）
            max_iterations: 最大迭代轮数（默认取参数配置）
            check_equilibrium_each_iter: 每轮结束即检查纳什均衡（默认开）
        """
        if update_rule not in UPDATE_RULES:
            raise ValueError(f"update_rule must be one of {UPDATE_RULES}, got {update_rule!r}")

        self.uavs = uavs
        self.beams = beams
        self.channel = channel
        self.satellite_pos = satellite_pos
        self.update_rule = update_rule
        self.force_offload = force_offload
        self.max_iterations = max_iterations or params.SIMULATION['MAX_ITERATIONS']
        self.check_equilibrium_each_iter = check_equilibrium_each_iter

        # 所有随机决策复用场景级 rng（种子可复现）
        self.rng = channel.rng
        self.history = []

        self._refresh_loads_and_rates()
        self._refresh_all_utilities()

    # ------------------------------------------------------------------
    # 状态维护（唯一的负载/效用写入口）
    # ------------------------------------------------------------------
    def _beam_counts(self):
        """按 UAV 当前策略统计各波束用户数。"""
        counts = [0] * len(self.beams)
        for uav in self.uavs:
            if uav.current_mode == "offload" and uav.selected_beam is not None:
                for j, beam in enumerate(self.beams):
                    if beam['center'] == uav.selected_beam['center']:
                        counts[j] += 1
                        break
        return counts

    def _refresh_loads_and_rates(self):
        """重计波束负载并按 (B/K)·log2(1+SNR) 刷新速率（SNR 已缓存，雨衰固定）。"""
        counts = self._beam_counts()
        for beam, k in zip(self.beams, counts):
            beam['num_users'] = k
            beam['data_rate'] = self.channel.calculate_data_rate(beam['snr'], k)

    def _utility(self, uav, mode, beam=None) -> float:
        """按模式计算效用（单一出口，local 分支不再崩溃）。"""
        if mode == 'local':
            return uav.calculate_utility('local')
        return uav.calculate_utility(
            'offload',
            beam_center=beam['center'],
            beam_radius=beam['radius'],
            data_rate=beam['data_rate'],
            sat_comp_freq=beam['sat_comp_freq'],
            num_users_in_beam=beam['num_users'],
            sat_distance=beam['sat_distance'],
        )

    def _refresh_all_utilities(self):
        """按 UAV 各自的当前模式重算效用（修复原实现无条件按 offload 重算的 bug）。"""
        for uav in self.uavs:
            utility = self._utility(uav, uav.current_mode, uav.selected_beam)
            uav.last_utility_calculation = {
                'utility': utility, 'strategy': uav.current_mode}

    def _candidate_best(self, uav):
        """枚举 uav 的全部候选动作，返回 (best_utility, best_action)。

        动作: ("local", None) 或 ("offload", beam)。
        候选波束效用按"自己加入后的负载"（K_j+1）评估，与均衡验证的反
        事实口径一致；当前波束评估时不重复加 1。
        """
        counts = self._beam_counts()

        best_u, best_action = -np.inf, None
        if not self.force_offload:
            best_u = self._utility(uav, 'local')
            best_action = ("local", None)

        current_center = uav.selected_beam['center'] \
            if uav.current_mode == "offload" and uav.selected_beam else None

        for j, beam in enumerate(self.beams):
            k = counts[j] + (0 if beam['center'] == current_center else 1)
            utility = uav.calculate_utility(
                'offload',
                beam_center=beam['center'],
                beam_radius=beam['radius'],
                data_rate=self.channel.calculate_data_rate(beam['snr'], k),
                sat_comp_freq=beam['sat_comp_freq'],
                num_users_in_beam=k,
                sat_distance=beam['sat_distance'],
            )
            if utility > best_u:
                best_u = utility
                best_action = ("offload", beam)

        return best_u, best_action

    # ------------------------------------------------------------------
    # 两种更新规则
    # ------------------------------------------------------------------
    def _sample_uavs(self):
        """按 UPDATE_RATIO 随机抽取本轮参与更新的 UAV（打破同步决策）。"""
        ratio = params.GAME['UPDATE_RATIO']
        n = max(1, int(len(self.uavs) * ratio))
        idx = self.rng.permutation(len(self.uavs))[:n]
        return [self.uavs[i] for i in idx]

    def _update_best_response(self):
        """异步最好响应 + 惯性：效用有显著改进时以 LEARNING_RATE 概率切换。

        相对原实现去掉了一道额外门槛（"仅当自身效用低于群体平均才考虑
        变更"）——该门槛会系统性阻止有利可图的切换，使强制卸载模式无法
        收敛到纳什均衡（原项目仅靠"恰好全员初始卸载 + 只在波束间切换"
        掩盖了这一点）。标准的异步最好响应（inertia 型）在拥塞博弈中
        以概率 1 收敛到纯纳什均衡。
        """
        changes = 0
        min_improvement_ratio = params.GAME['MIN_IMPROVEMENT_RATIO']
        learning_rate = params.GAME['LEARNING_RATE']

        for uav in self._sample_uavs():
            current = uav.last_utility_calculation['utility']
            best_u, best_action = self._candidate_best(uav)

            improvement = best_u - current
            # 相对改进阈值（不依赖效用绝对量纲）
            if improvement <= min_improvement_ratio * max(abs(current), 1e-12):
                continue
            if self.rng.random() >= learning_rate:
                continue  # 惯性：以概率保持现状

            mode, beam = best_action
            if mode == "local":
                uav.current_mode = "local"
                uav.selected_beam = None
            else:
                uav.current_mode = "offload"
                uav.selected_beam = beam
            changes += 1

        return changes / len(self.uavs)

    def _update_replicator(self):
        """成对比例模仿：以正比于效用差的概率模仿更优同伴的策略。"""
        changes = 0
        scale = params.GAME['IMITATION_SCALE']
        n = len(self.uavs)

        for uav in self._sample_uavs():
            peer = self.uavs[int(self.rng.integers(n))]
            if peer is uav:
                continue
            delta = peer.last_utility_calculation['utility'] \
                - uav.last_utility_calculation['utility']
            if delta <= 0:
                continue
            if self.rng.random() >= delta / (delta + scale):
                continue

            # 模仿同伴的完整策略（模式 + 波束）
            uav.current_mode = peer.current_mode
            uav.selected_beam = peer.selected_beam
            changes += 1

        return changes / len(self.uavs)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def _check_ne(self):
        if self.force_offload:
            return check_offload_only_ne(self.uavs, self.beams, self.channel)
        return check_general_ne(self.uavs, self.beams, self.channel)

    def _check_stable(self, change_rate):
        """收敛判据（恢复启用）：策略变化率足够小且近期效用方差足够小。"""
        conv = params.CONVERGENCE
        if change_rate >= conv['STRATEGY_CHANGE_THRESHOLD']:
            return False
        window = conv['VARIANCE_WINDOW']
        if len(self.history) < window:
            return False
        recent_vars = [np.var(s['utilities']) for s in self.history[-window:]]
        return max(recent_vars) < conv['UTILITY_VAR_THRESHOLD']

    def run(self):
        """运行博弈至纳什均衡 / 稳定收敛 / 最大轮数。

        返回 dict:
            converged: 是否提前终止（NE 达成或状态稳定）
            stop_reason: "nash_equilibrium" | "stable" | "max_iterations"
            iterations / final_offload_ratio / is_equilibrium /
            equilibrium_explanation / history
        """
        iteration = 0
        stop_reason = "max_iterations"

        self.record_state(0)

        pbar = tqdm(total=self.max_iterations,
                    desc=f"Game({self.update_rule}{',forced' if self.force_offload else ''})")
        while iteration < self.max_iterations:
            iteration += 1
            pbar.update(1)

            # 1. 策略更新
            if self.update_rule == "best_response":
                change_rate = self._update_best_response()
            else:
                change_rate = self._update_replicator()

            # 2. 负载 -> 速率 -> 效用 依次刷新
            self._refresh_loads_and_rates()
            self._refresh_all_utilities()

            # 3. 记录状态
            self.record_state(iteration)
            logger.info("iter %d: change_rate=%.4f offload_ratio=%.2f",
                        iteration, change_rate, self.current_offload_ratio())

            # 4. 终止判定：NE 优先，其次稳定性
            if self.check_equilibrium_each_iter:
                is_eq, explanation = self._check_ne()
                if is_eq:
                    stop_reason = "nash_equilibrium"
                    logger.info("Nash equilibrium reached at iteration %d", iteration)
                    break

            if self._check_stable(change_rate):
                stop_reason = "stable"
                break

            pbar.set_postfix({
                "changes": f"{change_rate:.3f}",
                "offload": f"{self.current_offload_ratio():.2f}",
            })
        pbar.close()

        # 最终均衡判定（无论为何终止）
        is_equilibrium, explanation = self._check_ne()
        if is_equilibrium and stop_reason == "max_iterations":
            stop_reason = "nash_equilibrium"

        return {
            "converged": stop_reason != "max_iterations",
            "stop_reason": stop_reason,
            "iterations": iteration,
            "final_offload_ratio": self.current_offload_ratio(),
            "is_equilibrium": is_equilibrium,
            "equilibrium_explanation": explanation,
            "history": self.history,
        }

    # ------------------------------------------------------------------
    # 状态记录与查询
    # ------------------------------------------------------------------
    def current_offload_ratio(self) -> float:
        """当前卸载比例。"""
        return sum(1 for u in self.uavs if u.current_mode == "offload") / len(self.uavs)

    def beam_utilizations(self) -> dict:
        """{波束中心: 用户数}。"""
        return {beam['center']: beam['num_users'] for beam in self.beams}

    def record_state(self, iteration):
        """记录当前博弈状态（策略、效用、各波束负载）。"""
        self.history.append({
            "iteration": iteration,
            "offload_ratio": self.current_offload_ratio(),
            "strategies": [u.current_mode for u in self.uavs],
            "utilities": [u.last_utility_calculation['utility'] for u in self.uavs],
            "beam_loads": [beam['num_users'] for beam in self.beams],
        })
