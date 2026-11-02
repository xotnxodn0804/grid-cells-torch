"""Reproduce the original 4-neighbor Grid Spacing BIC selection and make journal-like plots.

Requirements: numpy, pandas, matplotlib, scikit-learn.

Usage in grid-cells-torch (from project root):
    python plot_grid_scale_bic_paper.py
or:
    python plot_grid_scale_bic_paper.py --csv "results/sonnet_deepmind20_mean/grid_spacing_108_4neighbor/grid_spacing_108_units.csv"

The original GMM implementation used: random_state=0, n_init=5, reg_covar=1e-3,
and fitted observations in their original CSV order, measured in cm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def find_csv(explicit: str | None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"CSV does not exist: {path.resolve()}")
        return path

    roots = (Path('results'), Path('.'))
    candidates = []
    for root in roots:
        if root.exists():
            candidates += [p for p in root.rglob('grid_spacing_108_units.csv')
                           if '4neighbor' in str(p).lower()]
    unique = sorted(set(p.resolve() for p in candidates))
    if len(unique) != 1:
        raise FileNotFoundError(
            'Expected exactly one 4-neighbor CSV, found: '\
            + ', '.join(map(str, unique))\
            + '\nSupply --csv FULL_OR_RELATIVE_PATH explicitly.'
        )
    return unique[0]


def fit_bic_models(values_cm: np.ndarray):
    """Fit GMMs k=1..8 in CENTIMETERS to preserve historical BIC values."""
    vals = np.asarray(values_cm, dtype=float).reshape(-1, 1)
    fits = []
    for k in range(1, 9):
        model = GaussianMixture(n_components=k,
                                random_state=0,
                                n_init=5,
                                reg_covar=1e-3)
        model.fit(vals)
        fits.append({'k': k, 'bic': float(model.bic(vals)),
                     'aic': float(model.aic(vals)), 'model': model})
    best = min(fits, key=lambda item: item['bic'])
    return best, fits


def plot_panel(ax, values_cm, best, title, letter, xlim, ymax):
    xs_m = np.asarray(values_cm, dtype=float) / 100.
    m = best['model']
    xgrid = np.linspace(*xlim, 1200)

    # Density in m^-1: fitted model is trained with inputs in cm.
    mixture_density_per_m = np.exp(m.score_samples((xgrid * 100.).reshape(-1, 1))) * 100.
    # Same histogram boundaries in both panels to ensure fair comparison.
    edges = np.linspace(*xlim, 26)
    ax.hist(xs_m, bins=edges, density=True, color='#a6bfdc',
            edgecolor='#ffffff', linewidth=0.7, alpha=0.70)
    ax.plot(xgrid, mixture_density_per_m, color='#174a9c', linewidth=2.2,
            label=f'{best["k"]}-component GMM density')

    means_m = np.sort(m.means_.ravel()) / 100.
    for v in means_m:
        ax.axvline(v, 0, 0.88, color='#174a9c', linestyle=(0, (3, 3)), linewidth=1.25)
        ax.text(v, ymax * 0.95, f'{v:.3f}', ha='center', va='bottom',
                fontsize=10.4, color='#1c2a37', fontweight='semibold')

    arrow_y = ymax * 0.865
    for i, (left, right) in enumerate(zip(means_m[:-1], means_m[1:])):
        ax.annotate('', xy=(left, arrow_y), xytext=(right, arrow_y),
                    arrowprops=dict(arrowstyle='<->', color='#174a9c',
                                    linewidth=1.5, shrinkA=2, shrinkB=2))
        ax.text((left + right) / 2, arrow_y + ymax * 0.026,
                f'{right/left:.2f}', ha='center', va='bottom',
                color='#174a9c', fontsize=10.2, fontweight='bold')

    ax.set_title(f'{title}  |  BIC-selected K={best["k"]}', fontsize=12, pad=34)
    ax.set_xlabel('Grid scale (m)', fontsize=11)
    ax.set_ylabel('Probability density (m$^{-1}$)', fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, ymax)
    ax.text(0.005, 1.165, letter, transform=ax.transAxes,
            fontsize=23, fontweight='bold', ha='left', va='center')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.text(0.985, 0.77,
            f'BIC = {best["bic"]:.2f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9.2, color='#3a4858')
    return means_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default=None,
                    help='108-row 4-neighbor grid_spacing_108_units.csv')
    ap.add_argument('--out-dir', type=Path,
                    default=Path('results/grid_scale_paper_style_comparison'))
    args = ap.parse_args()
    csv_path = find_csv(args.csv)
    df = pd.read_csv(csv_path)
    if 'grid_spacing_cm' not in df.columns:
        raise ValueError('CSV must contain grid_spacing_cm column')
    if 'unit' not in df.columns:
        raise ValueError('CSV must contain unit column')
    df = df[np.isfinite(pd.to_numeric(df['grid_spacing_cm'], errors='coerce'))].copy()
    original_cm = df['grid_spacing_cm'].to_numpy(dtype=float)  # never SORT the data before fitting!

    if len(original_cm) < 10:
        raise ValueError('Not enough finite observations for GMM/BIC')
    # Remove ONE minimum and ONE maximum only, preserving the original row order.
    # If several units tie for the minimum, remove only one: the lowest unit ID.
    # For the supplied 108-unit dataset, that is unit #127 (not #158 or #455).
    minimum_positions = np.flatnonzero(original_cm == np.min(original_cm))
    min_position = min(minimum_positions, key=lambda pos: int(df.iloc[pos]['unit']))
    remove_positions = {int(min_position), int(np.argmax(original_cm))}
    trimmed_cm = original_cm[[i not in remove_positions for i in range(len(original_cm))]]

    best_original, models_original = fit_bic_models(original_cm)
    best_trimmed, models_trimmed = fit_bic_models(trimmed_cm)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    bic = pd.DataFrame({
        'k': [z['k'] for z in models_original],
        'bic_original': [z['bic'] for z in models_original],
        'bic_trimmed': [z['bic'] for z in models_trimmed],
    })
    bic.to_csv(out / 'bic_comparison.csv', index=False)

    xlim = (min(0.66, original_cm.min()/100.-0.05),
            max(2.38, original_cm.max()/100.+0.05))
    ymax = 3.45
    titles = [f'Original · n={len(original_cm)}',
              f'Trimmed · n={len(trimmed_cm)}']
    data = [(original_cm, best_original), (trimmed_cm, best_trimmed)]

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6), sharex=True, sharey=True)
    for ax, (values, best), title, letter in zip(axes, data, titles, 'ab'):
        plot_panel(ax, values, best, title, letter, xlim, ymax)
    fig.subplots_adjust(left=.07, right=.986, bottom=.14, top=.81, wspace=.17)
    fig.savefig(out / 'grid_scale_BIC_108_vs_106.png', dpi=260, facecolor='white')
    fig.savefig(out / 'grid_scale_BIC_108_vs_106.svg', facecolor='white')
    plt.close(fig)

    for (values, best), title, letter, suffix in zip(data, titles, 'ab', ('108', '106')):
        fig, ax = plt.subplots(figsize=(8.3, 5.6))
        plot_panel(ax, values, best, title, letter, xlim, ymax)
        fig.subplots_adjust(left=.10, right=.97, bottom=.13, top=.82)
        fig.savefig(out / f'grid_scale_BIC_{suffix}.png', dpi=260, facecolor='white')
        plt.close(fig)

    print('CSV:', csv_path)
    print('Removed units:', df.iloc[sorted(remove_positions)][['unit','grid_spacing_cm']].to_dict('records'))
    for title, (values, best) in zip(titles, data):
        means = np.sort(best['model'].means_.ravel())
        print(f'{title}: K={best["k"]}, BIC={best["bic"]:.4f}, '
              f'means_cm={np.round(means, 4).tolist()}, '
              f'ratios={np.round(means[1:]/means[:-1], 4).tolist()}')
    print('Saved:', out.resolve())


if __name__ == '__main__':
    main()
