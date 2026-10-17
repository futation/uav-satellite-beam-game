# utils/plots.py
"""结果绘图（合并自原项目 game_main.py / game_test.py 中两份重复实现）。

全部使用英文标签，不依赖任何中文字体；默认只保存文件不弹窗，
由调用方决定是否 `plt.show()`。
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(history, title="Game convergence", save_path=None, show=False):
    """卸载比例 + 效用曲线随迭代的变化。"""
    iterations = [s['iteration'] for s in history]
    offload_ratios = [s['offload_ratio'] for s in history]
    avg_u = [np.mean(s['utilities']) for s in history]
    min_u = [np.min(s['utilities']) for s in history]
    max_u = [np.max(s['utilities']) for s in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(iterations, offload_ratios, 'b-', linewidth=2)
    ax1.set_ylabel("Offloading ratio")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True)
    ax1.set_title(title)

    ax2.plot(iterations, avg_u, 'g-', label="mean", linewidth=2)
    ax2.plot(iterations, max_u, 'b--', label="max", linewidth=1)
    ax2.plot(iterations, min_u, 'r--', label="min", linewidth=1)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Utility")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_beam_utilization(beams, beam_loads, title="Final beam utilization",
                          save_path=None, show=False):
    """各波束最终用户数柱状图。beam_loads: {center: count}。"""
    labels = [f"B{beam['id']}\n({beam['center'][0]:.4f}, {beam['center'][1]:.4f})"
              for beam in beams]
    counts = [beam_loads.get(beam['center'], 0) for beam in beams]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts, color='skyblue', edgecolor='steelblue')
    ax.bar_label(bars, padding=2)
    ax.set_title(title)
    ax.set_xlabel("Beam center (lat, lon)")
    ax.set_ylabel("Number of offloading UAVs")
    ax.grid(axis='y')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_beam_avg_utility(beams, beam_avg_utilities,
                          title="Average utility per beam",
                          save_path=None, show=False):
    """各波束用户平均效用柱状图。"""
    labels = [f"B{beam['id']}" for beam in beams]
    values = [beam_avg_utilities.get(beam['center'], 0.0) for beam in beams]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color='lightgreen', edgecolor='seagreen')
    ax.bar_label(bars, fmt="%.3f", padding=2)
    ax.set_title(title)
    ax.set_xlabel("Beam")
    ax.set_ylabel("Average utility of its users")
    ax.grid(axis='y')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_rule_comparison(histories, title="Update rule comparison",
                         save_path=None, show=False):
    """多规则收敛对比。histories: {规则名: history list}。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for label, history in histories.items():
        iterations = [s['iteration'] for s in history]
        avg_u = [np.mean(s['utilities']) for s in history]
        offload = [s['offload_ratio'] for s in history]
        ax1.plot(iterations, avg_u, label=label, linewidth=2)
        ax2.plot(iterations, offload, label=label, linewidth=2)

    ax1.set_ylabel("Mean utility")
    ax1.grid(True)
    ax1.legend()
    ax1.set_title(title)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Offloading ratio")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_beam_load_evolution(history, title="Beam load evolution",
                             save_path=None, show=False):
    """各波束负载随迭代的热力演化（线图）。"""
    iterations = [s['iteration'] for s in history]
    loads = np.array([s['beam_loads'] for s in history])  # (T, M)

    fig, ax = plt.subplots(figsize=(9, 5))
    for j in range(loads.shape[1]):
        ax.plot(iterations, loads[:, j], label=f"Beam {j}", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Users per beam")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
