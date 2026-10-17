#!/usr/bin/env python
"""实验 2：混合博弈（本地计算 vs 卸载 + 波束选择）。

UAV 可在"本地计算"与"卸载到某个波束"之间自由选择，运行一种更新规则
至纳什均衡/稳定/最大轮数。此入口在原项目中因引擎 bug 从未能运行，
是本次重写修复的核心验证场景。

用法:
    python experiments/run_mixed.py [--uavs 60] [--seed 42]
        [--rule best_response|replicator] [--iterations 100] [--show]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config.parameters as params        # noqa: E402
from game.scenario import build_scenario          # noqa: E402
from game.engine import BeamSelectionGame         # noqa: E402
from game.metrics import summarize                # noqa: E402
from utils import plots                           # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--uavs", type=int, default=None, help="number of UAVs")
    parser.add_argument("--beams", type=int, default=None, help="number of beams")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--rule", choices=["best_response", "replicator"],
                        default="best_response", help="strategy update rule")
    parser.add_argument("--iterations", type=int, default=None, help="max iterations")
    parser.add_argument("--out", type=str, default="results", help="output directory")
    parser.add_argument("--show", action="store_true", help="also display figures")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 场景与博弈（初始策略 50/50 local/offload）----
    uavs, beams, channel, sat_pos = build_scenario(
        num_uavs=args.uavs, num_beams=args.beams, seed=(args.seed if args.seed is not None else params.RANDOM_SEED),
        force_offload=False, initial_offload_ratio=0.5)
    game = BeamSelectionGame(
        uavs, beams, channel, sat_pos,
        update_rule=args.rule, force_offload=False,
        max_iterations=args.iterations)

    result = game.run()
    metrics = summarize(uavs, beams)

    # ---- 文本结果 ----
    print("=" * 60)
    print("Mixed local/offload beam selection game")
    print("=" * 60)
    print(f"  update rule       : {args.rule}")
    print(f"  UAVs / beams      : {len(uavs)} / {len(beams)}")
    print(f"  converged         : {result['converged']} ({result['stop_reason']})")
    print(f"  iterations        : {result['iterations']}")
    print(f"  Nash equilibrium  : {result['is_equilibrium']}")
    print(f"  offloading ratio  : {metrics['offload_ratio']:.2f}")
    print(f"  mean utility      : {metrics['avg_utility']:.4f}")
    print(f"  local vs offload  : {metrics['avg_local_utility']:.4f} / "
          f"{metrics['avg_offload_utility']:.4f}")
    print(f"  social welfare    : {metrics['social_welfare']:.2f}")
    loads = sorted(metrics['beam_loads'].values(), reverse=True)
    print(f"  beam loads        : {loads}")
    print(f"\n{result['equilibrium_explanation']}")

    # ---- 图表 ----
    plots.plot_convergence(
        result['history'],
        title=f"Mixed game ({args.rule}, {len(uavs)} UAVs, {len(beams)} beams)",
        save_path=out_dir / "mixed_convergence.png", show=args.show)
    plots.plot_beam_utilization(
        beams, metrics['beam_loads'],
        title=f"Final beam load ({args.rule})",
        save_path=out_dir / "mixed_beam_loads.png", show=args.show)
    plots.plot_beam_load_evolution(
        result['history'],
        title=f"Beam load evolution ({args.rule})",
        save_path=out_dir / "mixed_beam_load_evolution.png", show=args.show)
    print(f"\nFigures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
