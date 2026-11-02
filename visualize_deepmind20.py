"""Render 20x20 rate maps and SACs from an existing supervised checkpoint.

Does NOT train, shuffle, or modify model weights. Uses exactly the 10-annulus
Mean formula from run_field_shuffle_deepmind20.py.

Run from grid-cells-torch repository root (with those earlier 20x20 analysis files):
  python visualize_deepmind20.py --checkpoint data/checkpoints/sonnet_v127_lstm_512/checkpoint_epoch299.pt --shard-dir data/datasets/gpu1m/shards --analysis-dir results/sonnet_deepmind20_mean
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from run_field_shuffle_deepmind20 import make_scorer, masks_for_20
from field_shuffle_nosmoothing import compute_ratemaps


def load_existing_scores(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    with path.open(newline='', encoding='utf-8') as f:
        return {int(r['unit']): r for r in csv.DictReader(f)}


def draw_image(ax, a, *, cmap='turbo', kind='ratemap'):
    a = np.asarray(a)
    if kind == 'sac':
        lim = float(np.nanmax(np.abs(a))) if np.isfinite(a).any() else 1.0
        lim = max(lim, 1e-8)
        ax.imshow(a.T, cmap='RdBu_r', interpolation='nearest', vmin=-lim, vmax=lim,
                  origin='lower')
    else:
        ax.imshow(a.T, cmap=cmap, interpolation='nearest', origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])


def save_overview(pdf: Path, maps: np.ndarray, order: np.ndarray, scores: np.ndarray,
                  units_per_page=64):
    """8x8 image grid per page, ordered by raw Gridness score descending."""
    with PdfPages(pdf) as out:
        for start in range(0, len(order), units_per_page):
            sub = order[start:start+units_per_page]
            fig, axes = plt.subplots(8, 8, figsize=(17, 18))
            fig.suptitle('20x20 rate maps | DeepMind 10 annuli | Mean Gridness | '
                         f'units {start+1}-{start+len(sub)} of {len(order)}', fontsize=16)
            for ax in axes.flat:
                ax.axis('off')
            for ax, u in zip(axes.flat, sub):
                draw_image(ax, maps[u])
                ax.axis('on')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'#{u}  G={scores[u]:.3f}', fontsize=8)
            fig.subplots_adjust(left=.03, right=.97, bottom=.03, top=.95,
                                wspace=.25, hspace=.36)
            out.savefig(fig)
            plt.close(fig)
            print(f'  overview saved page {start//units_per_page+1}', flush=True)


def save_top_detail(pdf: Path, maps: np.ndarray, sacs: list[np.ndarray],
                    order: np.ndarray, scores: np.ndarray, top_n=32):
    with PdfPages(pdf) as out:
        for start in range(0, min(top_n, len(order)), 8):
            sub = order[start:start+8]
            fig, axes = plt.subplots(4, 4, figsize=(15, 13))
            fig.suptitle('Top Gridness units | Left: rate map (20x20)  Right: SAC (39x39)\n'
                         'Scores use 10-annulus DeepMind Mean method, NOT a significance test',
                         fontsize=14)
            for ax in axes.flat:
                ax.axis('off')
            for k, u in enumerate(sub):
                a, b = axes.flat[2*k], axes.flat[2*k+1]
                draw_image(a, maps[u])
                draw_image(b, sacs[u], kind='sac')
                a.axis('on'); b.axis('on')
                a.set_xticks([]);a.set_yticks([]);b.set_xticks([]);b.set_yticks([])
                a.set_title(f'#{u}  G={scores[u]:.3f}  Rate map', fontsize=10)
                b.set_title(f'#{u}  Spatial AC', fontsize=10)
            fig.subplots_adjust(left=.04,right=.96,bottom=.03,top=.91,
                                wspace=.18,hspace=.36)
            out.savefig(fig)
            plt.close(fig)


def save_results_csv(path: Path, scores: np.ndarray, annuli: np.ndarray,
                     prev: dict[int, dict]):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['unit','gridness_mean_10annuli','best_inner_radius_bin',
                         'best_outer_radius_bin','historical_reference_over_0.37',
                         'shuffled_95_if_available','above_own_shuffled_95_if_available'])
        for u in np.argsort(-scores,kind='stable'):
            old = prev.get(int(u), {})
            cutoff = old.get('shuffle_95','')
            valid = bool(cutoff) and np.isfinite(float(cutoff))
            writer.writerow([u,round(float(scores[u]),9),float(annuli[u,0]),
                             float(annuli[u,1]),int(scores[u]>.37),
                             cutoff if valid else '', int(scores[u]>float(cutoff)) if valid else ''])


def smoke_test():
    assert len(masks_for_20()) == 10
    assert masks_for_20()[0] == (4.0,8.0)
    rng = np.random.default_rng(4)
    rm = rng.random((20,20))
    scorer = make_scorer(masks_for_20(), 1.1, 'mean')
    score,mask,sac=scorer.get_scores(rm)
    assert sac.shape==(39,39) and np.isfinite(score)
    out=Path('results/_visualize20_test.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    save_top_detail(out,rm[None], [sac],np.array([0]),np.array([score]),top_n=1)
    print('PASS: 20x20 maps, 39x39 SAC, mean score, PDF render ->',out)


def main():
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--checkpoint')
    ap.add_argument('--shard-dir')
    ap.add_argument('--analysis-dir',default='results/sonnet_deepmind20_mean',
                    help='Existing field shuffle results; read CSV only, NEVER rerun shuffling.')
    ap.add_argument('--out-dir',default=None,
                    help='Default: --analysis-dir (put visualizations next to shuffle output)')
    ap.add_argument('--n-trajectories',type=int,default=4000)
    ap.add_argument('--bottleneck-units',type=int,default=512)
    ap.add_argument('--no-open',action='store_true',help='Do not automatically open the top-unit PDF on Windows')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        smoke_test()
        return
    if not args.checkpoint or not args.shard_dir:
        ap.error('--checkpoint and --shard-dir are required')
    if args.n_trajectories<=0 or args.bottleneck_units<=0:
        ap.error('n-trajectories and bottleneck-units must be positive')

    import torch
    args.device='cuda' if torch.cuda.is_available() else 'cpu'
    args.nbins=20
    out=Path(args.out_dir or args.analysis_dir)
    out.mkdir(parents=True, exist_ok=True)
    print('Reloading fixed trained weights and 20x20 activity maps (NO training, NO shuffling) ...',flush=True)
    maps,epoch,cfg=compute_ratemaps(args)
    assert maps.shape==(args.bottleneck_units,20,20)
    scorer=make_scorer(masks_for_20(),cfg.task.env_size/2,'mean')
    scores=np.empty(len(maps),dtype=float)
    annuli=np.empty((len(maps),2),dtype=float)
    sacs=[]
    for i,rm in enumerate(maps):
        score,mask,sac=scorer.get_scores(rm)
        scores[i]=score
        annuli[i]=mask
        sacs.append(sac)
        if (i+1)%128==0: print(f'  scored {i+1}/{len(maps)}',flush=True)

    prev=load_existing_scores(Path(args.analysis_dir)/'unit_statistics.csv')
    if prev:
        diffs=[abs(scores[u]-float(p['observed_gridness'])) for u,p in prev.items()
               if 0<=u<len(scores)]
        if diffs and max(diffs)>1e-3:
            print('WARNING: scores differ from existing field-shuffle CSV (max delta '
                  f'{max(diffs):.5g}). Check checkpoint, shard selection, or trajectories. '
                  'The PDFs show re-evaluated scores.',flush=True)
        else:
            print('Cross-checked against previous field-shuffle score CSV: match',flush=True)

    order=np.argsort(-scores,kind='stable')
    save_results_csv(out/'gridness_scores_sorted.csv',scores,annuli,prev)
    save_overview(out/'all_512_ratemaps_20x20.pdf',maps,order,scores)
    save_top_detail(out/'top_32_ratemaps_and_sacs.pdf',maps,sacs,order,scores)
    print('\n=== DONE ===')
    print('Epoch:',epoch)
    print('Formula: Mean, DeepMind public code; Ratemap: 20x20; 10 annuli; no shuffling')
    print('Above historical 0.37 (NOT recalibrated for this method):',
          int((scores>.37).sum()),'/',len(scores))
    print('Files:')
    for name in ('gridness_scores_sorted.csv','all_512_ratemaps_20x20.pdf',
                 'top_32_ratemaps_and_sacs.pdf'):
        print(' ',(out/name).resolve())
    if os.name=='nt' and not args.no_open:
        try:
            os.startfile(str((out/'top_32_ratemaps_and_sacs.pdf').resolve()))
        except OSError as e:
            print('Auto-open failed:',e,'; open the PDF manually.')

if __name__=='__main__':
    main()
