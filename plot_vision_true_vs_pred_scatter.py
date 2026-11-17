#!/usr/bin/env python3
"""Plot actual vs predicted positions for the vision module validation set.

Reads a CSV with columns:
  true_x_m, true_y_m, pred_x_m, pred_y_m, position_error_cm

Example
-------
python plot_vision_true_vs_pred_scatter.py \
  --csv vision_val_position_errors.csv \
  --outdir results/vision_figures \
  --sample-lines 300
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=Path, default=Path('vision_val_position_errors.csv'))
    p.add_argument('--outdir', type=Path, default=Path('results/vision_figures'))
    p.add_argument('--sample-lines', type=int, default=300,
                   help='Number of true→pred pairs to connect with faint line segments.')
    p.add_argument('--seed', type=int, default=20261110)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    required = ['true_x_m', 'true_y_m', 'pred_x_m', 'pred_y_m', 'position_error_cm']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    true_x = df['true_x_m'].to_numpy()
    true_y = df['true_y_m'].to_numpy()
    pred_x = df['pred_x_m'].to_numpy()
    pred_y = df['pred_y_m'].to_numpy()
    err_cm = df['position_error_cm'].to_numpy()

    n = len(df)
    mean_err = float(np.mean(err_cm))
    median_err = float(np.median(err_cm))

    all_x = np.concatenate([true_x, pred_x])
    all_y = np.concatenate([true_y, pred_y])
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    pad = 0.08
    x_min -= pad; x_max += pad; y_min -= pad; y_max += pad
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * max(x_max - x_min, y_max - y_min)
    x_min, x_max = cx - half, cx + half
    y_min, y_max = cy - half, cy + half

    rng = np.random.default_rng(args.seed)
    sample_n = min(args.sample_lines, n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    # Faint connection lines for a readable subset
    for i in sample_idx:
        ax.plot([true_x[i], pred_x[i]], [true_y[i], pred_y[i]],
                linewidth=0.6, alpha=0.18)

    # Actual positions
    ax.scatter(true_x, true_y, s=10, alpha=0.28, label='Actual position')
    # Predicted positions
    ax.scatter(pred_x, pred_y, s=10, alpha=0.28, label='Predicted position')

    # Mean centers
    ax.scatter([np.mean(true_x)], [np.mean(true_y)], s=90, marker='x', linewidths=2,
               label='Mean actual')
    ax.scatter([np.mean(pred_x)], [np.mean(pred_y)], s=90, marker='x', linewidths=2,
               label='Mean predicted')

    ax.set_title(
        f'Vision module: actual vs predicted positions (N={n:,})\n'
        f'Mean error = {mean_err:.1f} cm, Median error = {median_err:.1f} cm',
        fontsize=18
    )
    ax.set_xlabel('x position (m)', fontsize=14)
    ax.set_ylabel('y position (m)', fontsize=14)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.22)
    ax.legend(loc='upper right', fontsize=12)

    args.outdir.mkdir(parents=True, exist_ok=True)
    png_path = args.outdir / 'vision_true_vs_pred_scatter.png'
    pdf_path = args.outdir / 'vision_true_vs_pred_scatter.pdf'
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    print(f'Saved: {png_path}')
    print(f'Saved: {pdf_path}')


if __name__ == '__main__':
    main()
