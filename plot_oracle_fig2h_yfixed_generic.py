#!/usr/bin/env python3
"""Make Banino-style Fig.2h comparison for Oracle Place Agent eval results.

Reads two evaluation folders (e.g. normal vs zero), corrects the legacy debug goal
marker for plotting only by flipping the sign of goal_y_m_debug, and draws:
- gray dashed: first search (before first goal hit)
- solid: subsequent completed returns (teleport start -> next goal hit)
- open squares: start points of completed returns
- red star: physical goal (goal_x_m_debug, -goal_y_m_debug)

Example:
python plot_oracle_fig2h_yfixed_generic.py \
  --normal-dir results/oracle_50m_normal \
  --zero-dir results/oracle_50m_zero \
  --panels 4 \
  --out-dir results/oracle_50m_compare/goal_y_corrected
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_episodes(eval_dir: Path) -> pd.DataFrame:
    path = eval_dir / 'episodes.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_traj(eval_dir: Path, ep: int) -> pd.DataFrame:
    path = eval_dir / f'trajectory_ep{ep:03d}.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def split_into_search_and_completed_returns(df: pd.DataFrame):
    hits = np.flatnonzero(df['goal_hit'].to_numpy() > 0)
    if len(df) == 0:
        return df.iloc[0:0].copy(), []
    if len(hits) == 0:
        return df.copy(), []

    first_search = df.iloc[: hits[0] + 1].copy()
    returns = []
    # completed returns: after one hit, from next sample (teleport start) until next hit
    for i in range(len(hits) - 1):
        start = hits[i] + 1
        end = hits[i + 1] + 1
        seg = df.iloc[start:end].copy()
        if len(seg) >= 2:
            returns.append(seg)
    return first_search, returns


def best_episode_rows(normal_df: pd.DataFrame, zero_df: pd.DataFrame, normal_dir: Path, zero_dir: Path, n: int) -> pd.DataFrame:
    merged = normal_df.merge(zero_df, on='episode', suffixes=('_normal', '_zero'))
    rows = []
    for _, row in merged.iterrows():
        ep = int(row['episode'])
        if (normal_dir / f'trajectory_ep{ep:03d}.csv').exists() and (zero_dir / f'trajectory_ep{ep:03d}.csv').exists():
            rows.append(row)
    if not rows:
        raise RuntimeError('No matching trajectory files were found in both folders.')
    merged = pd.DataFrame(rows)
    merged['delta_goals'] = merged['goals_normal'] - merged['goals_zero']
    merged = merged.sort_values(['delta_goals', 'goals_normal', 'goals_zero'], ascending=[False, False, True])
    return merged.head(n)


def corrected_goal_xy(df: pd.DataFrame):
    gx = float(df['goal_x_m_debug'].iloc[0])
    gy = -float(df['goal_y_m_debug'].iloc[0])  # key correction
    return gx, gy


def completed_return_time_seconds(seg: pd.DataFrame) -> float:
    if len(seg) == 0:
        return float('nan')
    frames = int(seg['frame'].iloc[-1] - seg['frame'].iloc[0])
    return frames / 60.0


def plot_panel(ax, traj: pd.DataFrame, color: str, mode: str, ep: int, goals: int):
    first_search, returns = split_into_search_and_completed_returns(traj)
    gx, gy = corrected_goal_xy(traj)

    # first search (legacy coordinates stay as recorded agent positions)
    if len(first_search) >= 2:
        ax.plot(first_search['x_m'], first_search['y_m'], linestyle=(0, (3, 3)), color='0.65', linewidth=1.6)

    # completed returns
    starts_x, starts_y = [], []
    first_return_s = np.nan
    for idx, seg in enumerate(returns):
        ax.plot(seg['x_m'], seg['y_m'], color=color, linewidth=1.7, alpha=0.95)
        starts_x.append(float(seg['x_m'].iloc[0]))
        starts_y.append(float(seg['y_m'].iloc[0]))
        if idx == 0:
            first_return_s = completed_return_time_seconds(seg)

    if starts_x:
        ax.scatter(starts_x, starts_y, marker='s', s=28, facecolors='white', edgecolors=color, linewidths=1.1, zorder=4)

    # reward points (where goal_hit==1, logged pre-action) for visual sanity
    hit_df = traj[traj['goal_hit'] > 0]
    if len(hit_df):
        ax.scatter(hit_df['x_m'], hit_df['y_m'], s=14, color=color, zorder=3, alpha=0.9)

    # initial spawn of episode
    ax.scatter(float(traj['x_m'].iloc[0]), float(traj['y_m'].iloc[0]), marker='s', s=26, color='0.35', zorder=5)

    ax.scatter([gx], [gy], marker='*', s=90, color='#d9534f', zorder=6)

    title = f"Ep {ep:02d} | {mode} | {goals} hits"
    if np.isfinite(first_return_s):
        title += f"  |  first return {first_return_s:.1f} s"
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(True, alpha=0.18)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--normal-dir', type=Path, required=True)
    p.add_argument('--zero-dir', type=Path, required=True)
    p.add_argument('--panels', type=int, default=4)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()

    normal_ep = load_episodes(args.normal_dir)
    zero_ep = load_episodes(args.zero_dir)
    selected = best_episode_rows(normal_ep, zero_ep, args.normal_dir, args.zero_dir, args.panels)

    nrows = len(selected)
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 4 * nrows), squeeze=False)

    for r, (_, row) in enumerate(selected.iterrows()):
        ep = int(row['episode'])
        ntraj = load_traj(args.normal_dir, ep)
        ztraj = load_traj(args.zero_dir, ep)
        plot_panel(axes[r, 0], ntraj, '#2b73b7', 'Normal', ep, int(row['goals_normal']))
        plot_panel(axes[r, 1], ztraj, '#d98932', 'Zero', ep, int(row['goals_zero']))

    stem = args.normal_dir.name.replace('oracle_', '').replace('_normal', '')
    fig.suptitle(f'Oracle Place Agent ({stem.upper()}): First Search and Goal Revisits', fontsize=18, y=0.995)
    fig.text(0.5, 0.033,
             'Gray dashed: first search   |   Solid: subsequent completed returns   |   Squares: starts   |   Red star: physical goal',
             ha='center', fontsize=10)
    fig.text(0.5, 0.013,
             'Legacy debug goal Y corrected for plotting only (y -> -y). Reward locations are logged pre-action; goal-code inputs were NOT changed.',
             ha='center', fontsize=9, color='0.35')
    plt.tight_layout(rect=[0, 0.05, 1, 0.965])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / 'fig2h_normal_vs_zero_corrected.png'
    pdf = args.out_dir / 'fig2h_normal_vs_zero_corrected.pdf'
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    print('Saved:', png)
    print('Saved:', pdf)


if __name__ == '__main__':
    main()
