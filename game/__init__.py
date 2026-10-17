# game/__init__.py
"""博弈决策层：场景构建、博弈引擎、均衡验证、指标统计。"""
from game.engine import BeamSelectionGame
from game.scenario import build_scenario

__all__ = ["BeamSelectionGame", "build_scenario"]
