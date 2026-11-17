#!/usr/bin/env python3
"""
Compare 500M Oracle Place Agent trajectories:
- normal goal memory
- zero goal memory

Usage:
python plot_oracle_normal_zero_compare.py ^
  --normal-dir results/oracle_500m_normal ^
  --zero-dir results/oracle_500m_zero ^
  --panels 4 ^
  --out results/oracle_500m_compare/fig_normal_vs_zero.png
"""

import argparse
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_episodes(eval_dir: Path):
    path = eval_dir / "episodes.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def load_traj(eval_dir: Path, episode_idx: int):
    path = eval_dir / f"trajectory_ep{episode_idx:03d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def split_segments(df, jump_threshold=0.6):
    """
    Split trajectory when a large position jump occurs.
    This helps avoid drawing a straight line across teleportation.
    """
    if len(df) <= 1:
        return [df]

    xs = df["x_m"].values
    ys = df["y_m"].values

    segments = []
    start = 0

    for i in range(1, len(df)):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > jump_threshold:
            seg = df.iloc[start:i].copy()
            if len(seg) > 0:
                segments.append(seg)
            start = i

    seg = df.iloc[start:].copy()
    if len(seg) > 0:
        segments.append(seg)

    return segments


def get_plot_limits(df1, df2):
    xs = list(df1["x_m"].values) + list(df2["x_m"].values)
    ys = list(df1["y_m"].values) + list(df2["y_m"].values)

    if "goal_x_m_debug" in df1.columns:
        xs.append(float(df1["goal_x_m_debug"].iloc[0]))
        ys.append(float(df1["goal_y_m_debug"].iloc[0]))
    if "goal_x_m_debug" in df2.columns:
        xs.append(float(df2["goal_x_m_debug"].iloc[0]))
        ys.append(float(df2["goal_y_m_debug"].iloc[0]))

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    # same padding on all sides
    pad = 0.15
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad

    # keep square axes
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    half = max(xmax - xmin, ymax - ymin) * 0.5

    return xmid - half, xmid + half, ymid - half, ymid + half


def plot_panel(ax, traj_df, first_hit_frame, title):
    # split into "before first goal" and "after first goal"
    if first_hit_frame >= 0:
        before = traj_df[traj_df["frame"] <= first_hit_frame].copy()
        after = traj_df[traj_df["frame"] > first_hit_frame].copy()
    else:
        before = traj_df.copy()
        after = traj_df.iloc[0:0].copy()

    # first trajectory segment
    if len(before) > 1:
        ax.plot(
            before["x_m"], before["y_m"],
            linewidth=2.0,
            color="black",
            alpha=0.85,
            label="First search"
        )

    # after first hit
    after_segments = split_segments(after, jump_threshold=0.6)
    for i, seg in enumerate(after_segments):
        if len(seg) > 1:
            if i == 0:
                ax.plot(
                    seg["x_m"], seg["y_m"],
                    linewidth=1.5,
                    alpha=0.75,
                    color="#6BAED6",
                    label="After teleport"
                )
            else:
                ax.plot(
                    seg["x_m"], seg["y_m"],
                    linewidth=1.5,
                    alpha=0.75,
                    color="#6BAED6"
                )

    # start point
    ax.scatter(
        float(traj_df["x_m"].iloc[0]),
        float(traj_df["y_m"].iloc[0]),
        marker="s",
        s=45,
        color="black",
        label="First start",
        zorder=5
    )

    # goal point
    gx = float(traj_df["goal_x_m_debug"].iloc[0])
    gy = float(traj_df["goal_y_m_debug"].iloc[0])
    ax.scatter(
        gx, gy,
        marker="*",
        s=120,
        color="#D95F5F",
        label="Goal",
        zorder=6
    )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-dir", type=Path, required=True)
    parser.add_argument("--zero-dir", type=Path, required=True)
    parser.add_argument("--panels", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    normal_ep = load_episodes(args.normal_dir)
    zero_ep = load_episodes(args.zero_dir)

    merged = normal_ep.merge(
        zero_ep,
        on="episode",
        suffixes=("_normal", "_zero")
    )

    # keep only episodes where both trajectory files exist
    valid_rows = []
    for _, row in merged.iterrows():
        ep = int(row["episode"])
        normal_path = args.normal_dir / f"trajectory_ep{ep:03d}.csv"
        zero_path = args.zero_dir / f"trajectory_ep{ep:03d}.csv"
        if normal_path.exists() and zero_path.exists():
            valid_rows.append(row)

    if len(valid_rows) == 0:
        raise RuntimeError("No matching trajectory files found.")

    merged = pd.DataFrame(valid_rows)

    # choose episodes where normal clearly outperforms zero
    merged["delta_goals"] = merged["goals_normal"] - merged["goals_zero"]

    merged = merged.sort_values(
        by=["delta_goals", "goals_normal", "goals_zero"],
        ascending=[False, False, True]
    )

    selected = merged.head(args.panels)

    nrows = len(selected)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(10, 4 * nrows))

    if nrows == 1:
        axes = [axes]

    mean_normal = merged["goals_normal"].mean()
    mean_zero = merged["goals_zero"].mean()

    for i, (_, row) in enumerate(selected.iterrows()):
        ep = int(row["episode"])

        ntraj = load_traj(args.normal_dir, ep)
        ztraj = load_traj(args.zero_dir, ep)

        if nrows == 1:
            ax_left, ax_right = axes[0], axes[1]
        else:
            ax_left, ax_right = axes[i]

        xmin, xmax, ymin, ymax = get_plot_limits(ntraj, ztraj)

        plot_panel(
            ax_left,
            ntraj,
            int(row["first_hit_frame_normal"]),
            f"Episode {ep} · normal · {int(row['goals_normal'])} hits"
        )

        plot_panel(
            ax_right,
            ztraj,
            int(row["first_hit_frame_zero"]),
            f"Episode {ep} · zero · {int(row['goals_zero'])} hits"
        )

        ax_left.set_xlim(xmin, xmax)
        ax_left.set_ylim(ymin, ymax)
        ax_right.set_xlim(xmin, xmax)
        ax_right.set_ylim(ymin, ymax)

        if i == 0:
            ax_left.legend(loc="upper right", fontsize=8)
            ax_right.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Oracle Place Agent Trajectories (500M)\n"
        f"Normal mean goals = {mean_normal:.2f}, Zero mean goals = {mean_zero:.2f}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()