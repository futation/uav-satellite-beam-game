#!/usr/bin/env python
"""实验 1：强制卸载（纯波束选择拥塞博弈）。

所有 UAV 必须卸载，策略空间是"选择哪个卫星波束"。运行一种更新规则
直至纳什均衡/稳定/最大轮数，输出收敛曲线、波束负载与波束平均效用图。

用法:
    python experiments/run_offload_only.py [--uavs 60] [--seed 42]
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
from game.equilibrium import price_of_anarchy     # noqa: E402
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

    # ---- 场景与博弈 ----
    uavs, beams, channel, sat_pos = build_scenario(
        num_uavs=args.uavs, num_beams=args.beams, seed=(args.seed if args.seed is not None else params.RANDOM_SEED),
        force_offload=True)
    game = BeamSelectionGame(
        uavs, beams, channel, sat_pos,
        update_rule=args.rule, force_offload=True,
        max_iterations=args.iterations)

    result = game.run()

    # ---- 文本结果 ----
    metrics = summarize(uavs, beams)
    poa, best_welfare, welfare = price_of_anarchy(uavs, beams, channel)

    print("=" * 60)
    print("Offload-only beam selection game (congestion game)")
    print("=" * 60)
    print(f"  update rule      : {args.rule}")
    print(f"  UAVs / beams     : {len(uavs)} / {len(beams)}")
    print(f"  converged        : {result['converged']} ({result['stop_reason']})")
    print(f"  iterations       : {result['iterations']}")
    print(f"  Nash equilibrium : {result['is_equilibrium']}")
    print(f"  offloading ratio : {metrics['offload_ratio']:.2f}")
    print(f"  mean utility     : {metrics['avg_utility']:.4f}")
    print(f"  social welfare   : {welfare:.2f}  (sampled best: {best_welfare:.2f}, PoA ~ {poa:.2f})")
    loads = sorted(metrics['beam_loads'].values(), reverse=True)
    print(f"  beam loads       : {loads}")
    print(f"\n{result['equilibrium_explanation']}")

    # ---- 图表 ----
    plots.plot_convergence(
        result['history'],
        title=f"Offload-only game ({args.rule}, {len(uavs)} UAVs, {len(beams)} beams)",
        save_path=out_dir / "offload_only_convergence.png", show=args.show)
    plots.plot_beam_utilization(
        beams, metrics['beam_loads'],
        title=f"Final beam load ({args.rule})",
        save_path=out_dir / "offload_only_beam_loads.png", show=args.show)
    plots.plot_beam_avg_utility(
        beams, metrics['beam_avg_utilities'],
        title=f"Average utility per beam ({args.rule})",
        save_path=out_dir / "offload_only_beam_utilities.png", show=args.show)
    print(f"\nFigures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
