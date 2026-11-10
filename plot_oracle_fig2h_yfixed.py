#!/usr/bin/env python3
"""Fig.2h-style Oracle Place Agent trajectory comparison with corrected goal marker.

Corrects ONLY visualization of the target Y coordinate from legacy
ORACLE.GOAL.XY. Does not change/checkpoint/model/training data or reward events.

Run from ~/grid-cells-torch:
    python plot_oracle_fig2h_yfixed.py

Optional:
    python plot_oracle_fig2h_yfixed.py --episodes 5 25 78 74 --returns 4
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_trajectory(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty {path}")
    required = {"frame", "x_m", "y_m", "goal_hit", "goal_x_m_debug", "goal_y_m_debug"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Missing columns {required.difference(rows[0])} in {path}")
    frames = [int(float(row["frame"])) for row in rows]
    x = [float(row["x_m"]) for row in rows]
    y = [float(row["y_m"]) for row in rows]
    hits = [i for i, row in enumerate(rows) if float(row["goal_hit"]) > 0]
    goal_x = float(rows[0]["goal_x_m_debug"])
    # Legacy Lua observation is vertically mirrored relative to DEBUG.POS.TRANS.
    goal_y = -float(rows[0]["goal_y_m_debug"])
    return frames, x, y, hits, goal_x, goal_y


def plot_episode(ax, path: Path, mode: str, max_returns: int, episode: int):
    frames, x, y, hits, gx, gy = read_trajectory(path)
    color = "#1666A5" if mode == "Normal" else "#D57B22"

    # Original search: dashed gray, from episode start until the first reward.
    end = (hits[0] + 1) if hits else len(x)
    ax.plot(x[:end], y[:end], color="#999999", linewidth=1.7,
            linestyle=(0, (3.5, 2.5)), alpha=0.86, zorder=2)
    ax.scatter([x[0]], [y[0]], s=31, marker="s", color="#555555", zorder=5)

    # A reward occurs *after* the position on its CSV row was recorded.
    # Therefore the first row AFTER each hit is the teleport destination.
    returns_shown = 0
    for k in range(min(max_returns, max(0, len(hits) - 1))):
        a = hits[k] + 1
        b = hits[k + 1] + 1
        if b <= a:
            continue
        ax.plot(x[a:b], y[a:b], linewidth=1.65, color=color,
                alpha=0.82, zorder=3)
        ax.scatter([x[a]], [y[a]], marker="s", s=32,
                   facecolor="white", edgecolor=color, linewidth=1.3, zorder=6)
        ax.scatter([x[b-1]], [y[b-1]], marker="o", s=17,
                   color=color, zorder=6)
        returns_shown += 1

    # Goal marker is the physical goal, not the mirrored debug value.
    ax.scatter([gx], [gy], s=170, marker="*", color="#D3424A",
               edgecolor="white", linewidth=0.45, zorder=9)
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.28, 1.28)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(labelsize=8)
    ax.grid(color="#d6d6d6", linewidth=0.55, alpha=0.55)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    if len(hits) >= 2:
        first_return_s = (frames[hits[1]] - frames[hits[0]]) / 60.0
        details = f"{len(hits)} hits  |  first return {first_return_s:.1f} s"
    elif hits:
        details = "1 hit  |  no return"
    else:
        details = "0 hits"
    ax.set_title(f"Ep {episode:02d} | {mode} | {details}", fontsize=9.4, pad=8)
    return returns_shown


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--normal-dir", type=Path, default=Path("results/oracle_500m_normal"))
    p.add_argument("--zero-dir", type=Path, default=Path("results/oracle_500m_zero"))
    p.add_argument("--episodes", nargs="+", type=int, default=[5, 25, 78, 74])
    p.add_argument("--returns", type=int, default=4,
                   help="maximum subsequent completed return trajectories to draw")
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/oracle_500m_compare/goal_y_corrected"))
    args = p.parse_args()
    if not args.episodes or args.returns < 1:
        p.error("Provide one or more episodes and --returns >= 1")

    n = len(args.episodes)
    fig, axes = plt.subplots(n, 2, figsize=(9.3, 3.5*n + 0.9), squeeze=False)
    for row_idx, episode in enumerate(args.episodes):
        for col_idx, (mode, folder) in enumerate((("Normal", args.normal_dir),
                                                     ("Zero", args.zero_dir))):
            path = folder / f"trajectory_ep{episode:03d}.csv"
            nret = plot_episode(axes[row_idx][col_idx], path, mode, args.returns, episode)
            print(f"Episode {episode:03d} {mode}: displayed {nret} completed revisits")

    fig.suptitle("Oracle Place Agent (500M): First Search and Goal Revisits",
                 fontsize=14, y=0.994)
    fig.text(0.5, 0.028,
             "Gray dashed: first search   |   Solid: subsequent completed returns   |   "
             "Squares: starts   |   Red star: physical goal",
             ha="center", fontsize=9)
    fig.text(0.5, 0.012,
             "Legacy debug goal Y corrected for plotting only (y -> -y). "
             "Reward locations are logged pre-action; goal-code inputs were NOT changed.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.055, 1, 0.971), h_pad=2.3, w_pad=2.4)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / "fig2h_normal_vs_zero_corrected.png"
    pdf = args.out_dir / "fig2h_normal_vs_zero_corrected.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", png)
    print("Saved:", pdf)
    print("NOTE: plotting fix only. Existing checkpoints and model goal codes are unchanged.")


if __name__ == "__main__":
    main()
