#!/usr/bin/env python
"""实验 3：两种更新规则的收敛对比。

在同一初始场景（相同种子）上分别运行
- best_response（异步最好响应 + 惯性）
- replicator（成对比例模仿，复制者动态的个体级实现）

并绘制效用/卸载比例曲线对比。典型观察（见 README "What to expect"）：
- 拥塞博弈中最好响应快速收敛到纯纳什均衡；
- 有限种群中的模仿动态会被"同质策略"吸收态困住（例如全体本地计算），
  即使该状态并不是纳什均衡 —— 复制者动态的均衡集严格大于 NE 集。

用法:
    python experiments/compare_rules.py [--uavs 60] [--seed 42]
        [--mode offload_only|mixed] [--iterations 200] [--show]
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

RULES = ("best_response", "replicator")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--uavs", type=int, default=None, help="number of UAVs")
    parser.add_argument("--beams", type=int, default=None, help="number of beams")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--mode", choices=["offload_only", "mixed"],
                        default="offload_only", help="game mode")
    parser.add_argument("--iterations", type=int, default=200, help="max iterations per rule")
    parser.add_argument("--out", type=str, default="results", help="output directory")
    parser.add_argument("--show", action="store_true", help="also display figures")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else params.RANDOM_SEED
    force_offload = args.mode == "offload_only"
    histories, summaries, results = {}, {}, {}

    for rule in RULES:
        # 相同种子 -> 两种规则的初始场景完全一致（公平对比）
        uavs, beams, channel, sat_pos = build_scenario(
            num_uavs=args.uavs, num_beams=args.beams, seed=seed,
            force_offload=force_offload)
        game = BeamSelectionGame(
            uavs, beams, channel, sat_pos,
            update_rule=rule, force_offload=force_offload,
            max_iterations=args.iterations)
        result = game.run()
        histories[rule] = result['history']
        summaries[rule] = summarize(uavs, beams)
        results[rule] = result

    # ---- 对比表 ----
    print("=" * 72)
    print(f"Update rule comparison ({args.mode}, seed={seed})")
    print("=" * 72)
    header = (f"{'rule':<15}{'iters':>6}{'stop':>18}{'NE':>6}"
              f"{'offload':>9}{'meanU':>10}{'welfare':>10}")
    print(header)
    for rule in RULES:
        s = summaries[rule]
        r = results[rule]
        print(f"{rule:<15}{r['iterations']:>6}{r['stop_reason']:>18}"
              f"{str(r['is_equilibrium']):>6}{s['offload_ratio']:>9.2f}"
              f"{s['avg_utility']:>10.3f}{s['social_welfare']:>10.2f}")

    plots.plot_rule_comparison(
        histories,
        title=f"Best response vs replicator ({args.mode}, seed={seed})",
        save_path=out_dir / "rule_comparison.png", show=args.show)
    print(f"\nFigure saved to: {(out_dir / 'rule_comparison.png').resolve()}")


if __name__ == "__main__":
    main()
