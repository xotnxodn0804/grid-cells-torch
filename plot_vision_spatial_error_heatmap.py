#!/usr/bin/env python3
"""Vision module: spatial mean position-error heatmap (16 x 16 grid).

Input: validation CSV created by plot_vision_position_histogram.py.
Required columns: true_x_m, true_y_m, position_error_cm.

Usage from ~/grid-cells-torch:
  python plot_vision_spatial_error_heatmap.py
  python plot_vision_spatial_error_heatmap.py --csv results/vision_position_errors/vision_val_position_errors.csv
  python plot_vision_spatial_error_heatmap.py --csv <path> --outdir results/vision_figures

No model loading or training. Produces PNG + PDF + CSV.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def resolve_csv(path):
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"CSV file not found: {p}")
        return p
    guesses = [
        Path('results/vision_position_errors/vision_val_position_errors.csv'),
        Path('results/vision_figures/vision_val_position_errors.csv'),
        Path('vision_val_position_errors.csv'),
        Path('results/vision_position_error/vision_val_position_errors.csv'),
    ]
    for p in guesses:
        if p.is_file():
            return p
    for p in sorted(Path('results').glob('**/vision_val_position_errors.csv')) if Path('results').exists() else []:
        return p
    raise FileNotFoundError('vision_val_position_errors.csv not found. Use --csv PATH.\n'
                            'Try: find . -name "vision_val_position_errors.csv"')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default=None, help='Validation predictions CSV')
    parser.add_argument('--outdir', default='results/vision_figures')
    parser.add_argument('--bins', type=int, default=16)
    parser.add_argument('--arena-half', type=float, default=1.25,
                        help='Half of the 2.5 m square arena width')
    args = parser.parse_args()

    path = resolve_csv(args.csv)
    rows = []
    with path.open('r', newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        required = ('true_x_m', 'true_y_m', 'position_error_cm')
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f'CSV must include {required}; got {reader.fieldnames}')
        for r in reader:
            rows.append([float(r[k]) for k in required])
    a = np.asarray(rows, dtype=float)
    if a.size == 0:
        raise ValueError(f'No rows in {path}')
    x, y, error = a.T
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(error)
    x, y, error = x[valid], y[valid], error[valid]

    n = args.bins
    lo, hi = -args.arena_half, args.arena_half
    edges = np.linspace(lo, hi, n + 1)
    # histogram2d gives [x, y]; transpose for imshow [row_y, column_x]
    counts_xy, _, _ = np.histogram2d(x, y, bins=(edges, edges))
    sum_xy, _, _ = np.histogram2d(x, y, bins=(edges, edges), weights=error)
    count = counts_xy.T
    sums = sum_xy.T
    mean_error = np.divide(sums, count, out=np.full_like(sums, np.nan), where=count > 0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 7.5))
    im = ax.imshow(mean_error, origin='lower', extent=(lo, hi, lo, hi),
                   interpolation='nearest', cmap='viridis', aspect='equal')
    cb = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.04)
    cb.set_label('Mean position error (cm)', fontsize=12)
    ax.set_xlabel('True x position (m)', fontsize=12)
    ax.set_ylabel('True y position (m)', fontsize=12)
    ax.set_title(f'Vision module: spatial position error (N={len(x):,})\n'
                 f'{n}×{n} bins | Mean error = {error.mean():.1f} cm', fontsize=15)
    ax.set_xticks(np.linspace(lo, hi, 6))
    ax.set_yticks(np.linspace(lo, hi, 6))
    ax.grid(False)
    fig.tight_layout()
    path_png = outdir / 'vision_spatial_error_heatmap.png'
    path_pdf = outdir / 'vision_spatial_error_heatmap.pdf'
    fig.savefig(path_png, dpi=230)
    fig.savefig(path_pdf)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8.8, 7.5))
    image = ax2.imshow(count, origin='lower', extent=(lo, hi, lo, hi),
                       interpolation='nearest', cmap='viridis', aspect='equal')
    cb2 = fig2.colorbar(image, ax=ax2, fraction=0.047, pad=0.04)
    cb2.set_label('Number of validation images', fontsize=12)
    ax2.set_title(f'Validation sample coverage ({n}×{n} bins)', fontsize=15)
    ax2.set_xlabel('True x position (m)', fontsize=12)
    ax2.set_ylabel('True y position (m)', fontsize=12)
    fig2.tight_layout()
    count_png = outdir / 'vision_spatial_sample_counts.png'
    fig2.savefig(count_png, dpi=230)
    plt.close(fig2)

    bin_csv = outdir / 'vision_spatial_error_bins.csv'
    with bin_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['x_bin','y_bin','x_center_m','y_center_m','count','mean_error_cm'])
        for j in range(n):
            for i in range(n):
                writer.writerow([i, j, (edges[i]+edges[i+1])/2,
                                 (edges[j]+edges[j+1])/2, int(count[j, i]),
                                 '' if not np.isfinite(mean_error[j, i]) else f'{mean_error[j, i]:.6f}'])

    print(f'Input: {path} | valid samples: {len(x):,}')
    print(f'Mean error: {error.mean():.2f} cm | Median error: {np.median(error):.2f} cm')
    print('Saved:', path_png)
    print('Saved:', path_pdf)
    print('Saved:', count_png)
    print('Saved:', bin_csv)

if __name__ == '__main__':
    main()
