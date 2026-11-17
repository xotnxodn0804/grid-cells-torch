#!/usr/bin/env python3
"""Plot all 512 spatial autocorrelograms (SACs) from B-Online evaluation.

This is READ-ONLY: it loads an existing ratemaps_and_scores.npz and
writes a PDF. It does not rerun DeepMind Lab or change checkpoints.

Example:
    python plot_all_512_sacs_B.py \
        --input results/grid1350_B_mid_78k_100ep/ratemaps_and_scores.npz \
        --out-dir results/grid1350_B_mid_78k_100ep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path,
                    default=Path('results/grid1350_B_mid_78k_100ep/ratemaps_and_scores.npz'),
                    help='NPZ produced by eval_grid1350_B_online_mid_gridness.py')
    ap.add_argument('--out-dir', type=Path, default=None,
                    help='Output directory (defaults to same directory as --input)')
    ap.add_argument('--sort', choices=['gridness', 'unit'], default='gridness',
                    help='Unit display order. Default matches all_512_ratemaps.pdf')
    ap.add_argument('--per-page', type=int, choices=[32, 64], default=64,
                    help='32 SAC/page gives 16 pages, 64 SAC/page gives 8 pages')
    args = ap.parse_args()

    path = args.input.expanduser()
    if not path.is_file():
        ap.error(f'Input NPZ not found: {path}\nRun the 100-episode evaluator first, or check --input.')
    out = args.out_dir.expanduser() if args.out_dir is not None else path.parent
    out.mkdir(parents=True, exist_ok=True)

    with np.load(path, allow_pickle=False) as source:
        required = ('sacs', 'scores_mean', 'scores_minmax')
        missing = [k for k in required if k not in source.files]
        if missing:
            ap.error(f'{path} missing keys: {missing}; available: {source.files}')
        sacs = np.array(source['sacs'], copy=True)
        mean = np.array(source['scores_mean'], copy=True)
        minmax = np.array(source['scores_minmax'], copy=True)
        split = np.array(source['split_half_stability'], copy=True) \
            if 'split_half_stability' in source.files else None

    if sacs.shape != (512, 39, 39):
        ap.error(f'Expected 512 SACs with shape (39,39); got {sacs.shape}')
    if mean.shape != (512,) or minmax.shape != (512,):
        ap.error(f'Expected 512 gridness scores, got {mean.shape}, {minmax.shape}')
    if split is not None and split.shape != (512,):
        ap.error(f'Expected 512 split-half r values, got {split.shape}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if args.sort == 'gridness':
        order = np.argsort(np.nan_to_num(mean, nan=-1e10))[::-1]
        suffix = 'all_512_sacs.pdf'
    else:
        order = np.arange(512)
        suffix = 'all_512_sacs_by_unit.pdf'
    output = out / suffix

    summary_path = path.parent / 'summary.json'
    caption = 'B-Online 78K / 100 validation episodes'
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            updates = summary.get('grid_updates')
            n_episodes = summary.get('validation_episodes')
            if updates is not None and n_episodes is not None:
                caption = f'B-Online Grid {updates:,} updates / {n_episodes} validation episodes'
        except (ValueError, OSError):
            pass

    ncol, nrow = (8, 8) if args.per_page == 64 else (8, 4)
    with PdfPages(output) as pdf:
        for start in range(0, 512, args.per_page):
            units = order[start:start + args.per_page]
            fig, axs = plt.subplots(nrow, ncol,
                                    figsize=(17, 17) if args.per_page == 64 else (17, 9))
            fig.suptitle(f'{caption} | All 512 SACs | {args.sort} order | '
                         f'units {start + 1}–{start + len(units)}', fontsize=13)
            for ax, uid in zip(axs.flat, units):
                # Match the original evaluator's SAC orientation and color scaling.
                ax.imshow(sacs[uid].T, origin='lower', cmap='RdBu_r',
                          vmin=-1, vmax=1, interpolation='nearest')
                annotation = f'#{uid} G={mean[uid]:+.2f} | MM={minmax[uid]:+.2f}'
                if split is not None:
                    annotation += f' | r={split[uid]:+.2f}'
                ax.set_title(annotation, fontsize=6.5)
                ax.set_xticks([])
                ax.set_yticks([])
            for ax in list(axs.flat)[len(units):]:
                ax.axis('off')
            fig.tight_layout(rect=(0, 0, 1, .965), pad=.6)
            pdf.savefig(fig)
            plt.close(fig)
            print(f'SAVED PAGE | {start // args.per_page + 1}/{512 // args.per_page}', flush=True)

    print(f'SAVED | {output.resolve()}', flush=True)
    print('COMPLETE | 512 SACs; checkpoint and NPZ unchanged', flush=True)


if __name__ == '__main__':
    main()
