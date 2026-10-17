# tests/test_engine.py
"""博弈引擎冒烟测试：收敛性、纳什均衡验证、混合模式、种子可复现。"""

import numpy as np
import pytest

from game.scenario import build_scenario
from game.engine import BeamSelectionGame
from game.equilibrium import check_offload_only_ne, check_general_ne
from game.metrics import summarize


@pytest.fixture(scope="module")
def small_scenario():
    """小场景（12 UAV / 4 波束）加速测试。"""
    return build_scenario(num_uavs=12, seed=123, num_beams=4, force_offload=True)


class TestOffloadOnly:
    def test_best_response_reaches_nash_equilibrium(self, small_scenario):
        uavs, beams, channel, sat_pos = small_scenario
        game = BeamSelectionGame(uavs, beams, channel, sat_pos,
                                 update_rule="best_response",
                                 force_offload=True, max_iterations=100)
        result = game.run()
        assert result["is_equilibrium"], result["equilibrium_explanation"]
        assert result["stop_reason"] == "nash_equilibrium"

    def test_offload_ratio_is_one_when_forced(self, small_scenario):
        uavs, beams, channel, sat_pos = small_scenario
        game = BeamSelectionGame(uavs, beams, channel, sat_pos,
                                 update_rule="best_response",
                                 force_offload=True, max_iterations=100)
        result = game.run()
        assert result["final_offload_ratio"] == pytest.approx(1.0)

    def test_history_records_every_iteration(self, small_scenario):
        uavs, beams, channel, sat_pos = small_scenario
        game = BeamSelectionGame(uavs, beams, channel, sat_pos,
                                 update_rule="best_response",
                                 force_offload=True, max_iterations=5)
        result = game.run()
        # 迭代 0（初始态）+ 每轮一条
        assert len(result["history"]) == result["iterations"] + 1

    def test_loads_consistent_with_strategies(self, small_scenario):
        """波束负载与 UAV 策略一一对应（原项目记账 bug 的回归测试）。"""
        uavs, beams, channel, sat_pos = small_scenario
        game = BeamSelectionGame(uavs, beams, channel, sat_pos,
                                 update_rule="best_response",
                                 force_offload=True, max_iterations=10)
        game.run()
        counts = [0] * len(beams)
        for uav in uavs:
            counts[next(j for j, b in enumerate(beams)
                        if b['center'] == uav.selected_beam['center'])] += 1
        assert counts == [b['num_users'] for b in beams]


class TestMixed:
    def test_mixed_game_runs_and_valid(self):
        """混合模式可运行且终止状态自洽（原项目在此模式下直接崩溃）。"""
        uavs, beams, channel, sat_pos = build_scenario(
            num_uavs=12, seed=5, num_beams=4, force_offload=False)
        game = BeamSelectionGame(uavs, beams, channel, sat_pos,
                                 update_rule="best_response",
                                 force_offload=False, max_iterations=50)
        result = game.run()
        assert 0.0 <= result["final_offload_ratio"] <= 1.0
        # 所有 UAV 状态自洽
        for uav in uavs:
            if uav.current_mode == "offload":
                assert uav.selected_beam is not None
            else:
                assert uav.selected_beam is None

    def test_ne_verifier_matches_engine_state(self):
        """均衡验证器与引擎最终状态一致（且为纯函数、无副作用）。"""
        uavs, beams, channel, sat_pos = build_scenario(
            num_uavs=12, seed=5, num_beams=4, force_offload=False)
        game = BeamSelectionGame(uavs, beams, channel, sat_pos,
                                 update_rule="best_response",
                                 force_offload=False, max_iterations=50)
        result = game.run()
        is_eq, _ = check_general_ne(uavs, beams, channel)
        assert is_eq == result["is_equilibrium"]
        # 纯函数性：重复调用结果不变
        is_eq2, _ = check_general_ne(uavs, beams, channel)
        assert is_eq == is_eq2


class TestReproducibility:
    def test_same_seed_same_history(self):
        """同种子两次运行的历史应逐位一致（雨衰/位置/策略全部可复现）。"""
        def run_once():
            uavs, beams, channel, sat_pos = build_scenario(
                num_uavs=10, seed=99, num_beams=4, force_offload=True)
            game = BeamSelectionGame(uavs, beams, channel, sat_pos,
                                     update_rule="best_response",
                                     force_offload=True, max_iterations=30)
            return game.run()["history"]

        h1, h2 = run_once(), run_once()
        assert len(h1) == len(h2)
        for s1, s2 in zip(h1, h2):
            assert s1["strategies"] == s2["strategies"]
            assert s1["beam_loads"] == s2["beam_loads"]
            assert s1["utilities"] == pytest.approx(s2["utilities"])

    def test_different_seeds_differ(self):
        """不同种子应产生不同场景（防止假种子注入）。"""
        s1 = build_scenario(num_uavs=8, seed=1, num_beams=3, force_offload=True)
        s2 = build_scenario(num_uavs=8, seed=2, num_beams=3, force_offload=True)
        pos1 = [u.position for u in s1[0]]
        pos2 = [u.position for u in s2[0]]
        assert any(not np.allclose(a, b) for a, b in zip(pos1, pos2))


class TestInvalidInput:
    def test_invalid_update_rule_rejected(self, small_scenario):
        uavs, beams, channel, sat_pos = small_scenario
        with pytest.raises(ValueError):
            BeamSelectionGame(uavs, beams, channel, sat_pos,
                              update_rule="genetic")


class TestSummary:
    def test_summarize_keys(self, small_scenario):
        uavs, beams, channel, sat_pos = small_scenario
        game = BeamSelectionGame(uavs, beams, channel, sat_pos,
                                 update_rule="best_response",
                                 force_offload=True, max_iterations=20)
        game.run()
        m = summarize(uavs, beams)
        for key in ("offload_ratio", "avg_utility", "beam_loads",
                    "beam_avg_utilities", "social_welfare"):
            assert key in m
        assert sum(m["beam_loads"].values()) == len(uavs)
