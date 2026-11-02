"""Compare FIELD-SEGMENTATION smoothing only (0, 0.9, 1.5 bins).

For soluslan/grid-cells-torch and the companion field_shuffle_nosmoothing.py.

- Original 32x32 ratemap, stored activation values, SAC, and gridness are NEVER smoothed.
- Gaussian smoothing is applied ONLY to a disposable copy used to find
  watershed markers and region boundaries.
- The same source ratemap and scored annulus are used for every sigma.
- Explicitly checks whether field relocation breaks source-field adjacency.
- A small number of shuffles is for ALGORITHM DIAGNOSTICS, not significance.

Example (run at grid-cells-torch root):
    python compare_field_smoothing.py --self-test
    python compare_field_smoothing.py \
      --checkpoint data/checkpoints/gpu1m_512/checkpoint_epoch299.pt \
      --shard-dir data/datasets/gpu_eval4k/shards \
      --unit-id 415 --n-shuffles 10 \
      --out-dir results/sigma_comparison_415
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

import field_shuffle_nosmoothing as base


def segment_for_sigma(raw: np.ndarray, sigma: float):
    """Segment an OPTIONAL smoothed COPY; the raw map never changes."""
    if sigma < 0:
        raise ValueError('sigma must be nonnegative')
    raw = np.asarray(raw, dtype=np.float64)
    valid = np.isfinite(raw)
    if sigma == 0:
        guide = raw.copy()
    else:
        # Weighted Gaussian smooth prevents NaN contamination at unvisited bins.
        numerator = ndi.gaussian_filter(np.where(valid, raw, 0.0), sigma=sigma,
                                        mode='nearest')
        denominator = ndi.gaussian_filter(valid.astype(np.float64), sigma=sigma,
                                          mode='nearest')
        guide = np.divide(numerator, denominator,
                          out=np.full_like(raw, np.nan), where=denominator > 1e-12)
        guide[~valid] = np.nan
    labels, peaks = base.segment_raw_fields(guide)
    return guide, (labels, peaks)


def relocated_with_metrics(raw: np.ndarray, seg, rng: np.random.Generator):
    """Reference script's Manhattan / round-robin shuffle with diagnostics.

    Keeps raw values, visited-bin mask, and relative offsets, up to collisions.
    Values are never smoothed, even when segmentation guide is smoothed.
    """
    raw = np.asarray(raw, dtype=np.float64)
    n = raw.shape[0]
    labels, peaks = seg
    valid = np.isfinite(raw)
    valid_flat = np.flatnonzero(valid.ravel())
    n_fields = len(peaks)
    anchors_idx = rng.choice(valid_flat, size=n_fields, replace=False)
    anchors_xy = np.array(np.unravel_index(anchors_idx, raw.shape)).T

    coordinates = np.indices(raw.shape).reshape(2, -1).T
    label_flat = labels.ravel()
    raw_flat = raw.ravel()
    per_field = []
    for f in range(n_fields):
        src = np.flatnonzero(label_flat == f + 1)
        if not len(src):
            raise RuntimeError('Empty watershed field')
        offset = coordinates[src] - peaks[f]
        dist_sq = (offset.astype(np.int32)**2).sum(axis=1)
        order = src[np.lexsort((src, dist_sq))]
        if order[0] != np.ravel_multi_index(peaks[f], raw.shape):
            raise RuntimeError('Field peak not first in ordering')
        per_field.append(order)

    out = np.full(raw.size, np.nan, dtype=np.float64)
    free = valid.ravel().copy()
    source_to_target = np.full(raw.size, -1, dtype=np.int32)
    collisions = 0
    boundary_clipped = 0
    displacements = []

    for f, anchor in enumerate(anchors_idx):
        source_idx = per_field[f][0]
        out[anchor] = raw_flat[source_idx]
        source_to_target[source_idx] = anchor
        free[anchor] = False

    nearest = base.nearest_available_order(n)
    order = rng.permutation(n_fields)
    q = deque((int(f), 1) for f in order if len(per_field[f]) > 1)
    while q:
        f, pos = q.popleft()
        src = int(per_field[f][pos])
        offset = coordinates[src] - peaks[f]
        unbounded = anchors_xy[f] + offset
        target_xy = np.clip(unbounded, 0, n-1)
        if np.any(target_xy != unbounded):
            boundary_clipped += 1
        idx = int(target_xy[0] * n + target_xy[1])
        if free[idx]:
            dst = idx
        else:
            collisions += 1
            dst = next((int(k) for k in nearest[idx] if free[k]), None)
            if dst is None:
                raise RuntimeError('No unoccupied visited bin')
        out[dst] = raw_flat[src]
        source_to_target[src] = dst
        free[dst] = False
        displacements.append(int(np.abs(coordinates[dst] - target_xy).sum()))
        if pos + 1 < len(per_field[f]):
            q.append((f, pos + 1))

    shuffled = out.reshape(raw.shape)
    if np.any(free):
        raise RuntimeError('Some valid bins were left empty')
    if not np.array_equal(np.isfinite(shuffled), valid):
        raise RuntimeError('Visited-bin mask changed')
    if not np.array_equal(np.sort(shuffled[valid]), np.sort(raw[valid])):
        raise RuntimeError('Raw activation values not preserved')

    # Among all horizontal and vertical neighboring source-bin pairs belonging
    # to the SAME field, how many remain direct neighbors after relocation?
    idx_grid = np.arange(n * n).reshape(n, n)
    pairs = []
    for a, b in ((idx_grid[:-1,:], idx_grid[1:,:]),
                 (idx_grid[:,:-1], idx_grid[:,1:])):
        same = (labels.ravel()[a.ravel()] == labels.ravel()[b.ravel()]) & \
               (labels.ravel()[a.ravel()] > 0)
        pairs.extend(zip(a.ravel()[same].tolist(), b.ravel()[same].tolist()))
    kept = 0
    for a, b in pairs:
        pa = coordinates[source_to_target[a]]
        pb = coordinates[source_to_target[b]]
        kept += int(np.abs(pa-pb).sum() == 1)

    nonpeaks = max(1, len(valid_flat)-n_fields)
    metrics = {
        'n_fields': int(n_fields),
        'n_visited_bins': int(len(valid_flat)),
        'collision_rate': collisions / nonpeaks,
        'clipped_fraction': boundary_clipped / nonpeaks,
        'mean_collision_displacement_bins': float(np.mean(displacements)) if displacements else 0.,
        'within_field_adjacency_retained': kept / len(pairs) if pairs else float('nan'),
        'n_within_field_links': len(pairs),
    }
    return shuffled, metrics


def run_comparison(rm, sigmas, n_shuffles, seed, scorer, unit_id):
    from math import isfinite
    obs = float(scorer.get_scores(rm)[0])
    results = []
    for sigma in sigmas:
        guide, segmented = segment_for_sigma(rm, sigma)
        # Identical RNG seed stream across sigma conditions; the number of
        # markers may differ, so generated placements cannot be identical.
        rng = np.random.default_rng(np.random.SeedSequence([seed, unit_id]))
        null = []
        metrics = []
        previews = []
        for i in range(n_shuffles):
            shuffled, m = relocated_with_metrics(rm, segmented, rng)
            if i < 2:
                previews.append(shuffled)
            metrics.append(m)
            null.append(float(scorer.get_scores(shuffled)[0]))
        vals = np.asarray(null)
        results.append(dict(
            sigma=float(sigma),
            guide=guide,
            segmented=segmented,
            previews=previews,
            observed_gridness=obs,
            shuffle_scores=vals,
            shuffle_mean_gridness=float(np.mean(vals)),
            shuffle_95_exploratory=float(np.quantile(vals, .95)),
            n_fields=len(segmented[1]),
            collision_rate=float(np.mean([m['collision_rate'] for m in metrics])),
            clipped_fraction=float(np.mean([m['clipped_fraction'] for m in metrics])),
            mean_collision_displacement_bins=float(np.mean([m['mean_collision_displacement_bins'] for m in metrics])),
            within_field_adjacency_retained=float(np.mean([m['within_field_adjacency_retained'] for m in metrics])),
        ))
    return results


def save_visual(path, rm, results, unit_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lo, hi = np.nanpercentile(rm, [1,99])
    fig, axes = plt.subplots(len(results), 5, figsize=(16,3.65 * len(results)),
                             squeeze=False)
    for i,r in enumerate(results):
        arrays = [rm, r['guide'], r['segmented'][0], *r['previews']]
        titles = ["Original RAW", f"Segmentation guide (σ={r['sigma']:g})",
                  f"Watershed: {r['n_fields']} fields",
                  "Shuffle #1 (raw values)", "Shuffle #2 (raw values)"]
        for j, (data,title) in enumerate(zip(arrays, titles)):
            ax=axes[i,j]
            if j==2:
                ax.imshow(np.ma.masked_equal(data,0), cmap='tab20', interpolation='nearest')
            else:
                ax.imshow(data, cmap='viridis', interpolation='nearest',vmin=lo,vmax=hi)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        axes[i,0].set_ylabel(f"σ={r['sigma']:g}", rotation=90, fontsize=12)
    fig.suptitle(f"Unit {unit_id}: sigma affects ONLY field identification\n"
                 f"Original and shuffled gridness use raw, unsmoothed values",
                 fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(path,dpi=160,bbox_inches='tight')
    plt.close(fig)


def save_diagnostics(path, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[f"σ={r['sigma']:g}" for r in results]
    figures=[('n_fields', 'Number of watershed fields'),
             ('collision_rate', 'Occupied-target fraction'),
             ('within_field_adjacency_retained', 'Original field-neighbor links retained')]
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ax,(key,title) in zip(axs,figures):
        ax.bar(names,[r[key] for r in results])
        ax.set_title(title,fontsize=10)
        if key!='n_fields':
            ax.set_ylim(0,1)
    fig.suptitle('Shuffle diagnostics (before choosing smoothing)',fontsize=12)
    fig.tight_layout()
    fig.savefig(path,dpi=160,bbox_inches='tight')
    plt.close(fig)


def self_test():
    rng=np.random.default_rng(10)
    n=32
    x,y=np.meshgrid(np.arange(n),np.arange(n),indexing='ij')
    rm=sum(np.exp(-((x-(6+8*c+3*(r%2)))**2+(y-(6+9*r))**2)/14.)
           for r in range(3) for c in range(3)) - 1.3
    rm[:2, :3]=np.nan
    original=rm.copy()
    for sigma in (0,0.9,1.5):
        guide,seg=segment_for_sigma(rm,sigma)
        sh,met=relocated_with_metrics(rm,seg,rng)
        assert np.array_equal(np.isnan(sh), np.isnan(rm))
        assert np.array_equal(np.sort(sh[np.isfinite(sh)]),np.sort(rm[np.isfinite(rm)]))
        assert 0<=met['collision_rate']<=1
        assert 0<=met['within_field_adjacency_retained']<=1
        assert np.all(seg[0][np.isfinite(rm)]>0)
        assert np.array_equal(np.isnan(guide),np.isnan(rm))
        print(f"  σ={sigma:g} | {met['n_fields']} fields | collisions {100*met['collision_rate']:.1f}% | "
              f"same-field neighbors retained {100*met['within_field_adjacency_retained']:.1f}%")
    assert np.allclose(rm,original,equal_nan=True)
    print('PASS: raw values never smoothed, valid mask preserved, relocation and metrics work.')


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--shard-dir')
    parser.add_argument('--unit-id',type=int,default=415)
    parser.add_argument('--sigmas',default='0,0.9,1.5')
    parser.add_argument('--n-shuffles',type=int,default=10)
    parser.add_argument('--n-trajectories',type=int,default=4000)
    parser.add_argument('--nbins',type=int,default=32)
    parser.add_argument('--bottleneck-units',type=int,default=512)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--out-dir',default='results/sigma_comparison_415')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if not args.checkpoint or not args.shard_dir:
        parser.error('--checkpoint and --shard-dir required unless --self-test')
    if args.n_shuffles<2:
        parser.error('--n-shuffles must be >=2')
    if not 0<=args.unit_id<args.bottleneck_units:
        parser.error('Invalid --unit-id')
    sigmas=[float(v.strip()) for v in args.sigmas.split(',')]
    if len(sigmas)<2 or len(set(sigmas))!=len(sigmas) or any(s<0 for s in sigmas):
        parser.error('--sigmas need at least two distinct nonnegative values')
    import torch
    from grid_cells import evaluate
    args.device='cuda' if torch.cuda.is_available() else 'cpu'
    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    print(f'Loading {args.n_trajectories} trajectories on {args.device} ...',flush=True)
    maps,epoch,cfg=base.compute_ratemaps(args)
    rm=maps[args.unit_id]
    scorer=evaluate.build_scorer(cfg,args.nbins)
    radii=[[float(lo),float(hi)] for _,(lo,hi) in scorer._masks]
    print('Annulus in existing scorer:',radii,flush=True)
    print('RAW Gridness (must stay fixed across sigmas):',scorer.get_scores(rm)[0],flush=True)
    results=run_comparison(rm,sigmas,args.n_shuffles,args.seed,scorer,args.unit_id)
    for r in results:
        print(f"σ={r['sigma']:g}: fields={r['n_fields']}, "
              f"collision={r['collision_rate']:.1%}, "
              f"adjacency kept={r['within_field_adjacency_retained']:.1%}, "
              f"exploratory null 95%={r['shuffle_95_exploratory']:.3f}",flush=True)
    save_visual(out_dir/f'unit_{args.unit_id}_sigma_comparison.png',rm,results,args.unit_id)
    save_diagnostics(out_dir/f'unit_{args.unit_id}_diagnostics.png',results)
    fields=['sigma','n_fields','observed_gridness','shuffle_mean_gridness',
            'shuffle_95_exploratory','collision_rate','clipped_fraction',
            'mean_collision_displacement_bins','within_field_adjacency_retained']
    with (out_dir/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in results:
            writer.writerow({key:r[key] for key in fields})
    metadata={'unit_id':args.unit_id,'epoch':epoch,'n_trajectories':args.n_trajectories,
              'n_shuffles':args.n_shuffles,'sigmas':sigmas,'annulus':radii,
              'no_smoothing_original_maps':True,
              'no_smoothing_gridness_score':True,
              'segmentation_only_gaussian':True,
              'algorithm_is_exact_original_banino':False,
              'note':'Small-shuffle thresholds are EXPLORATORY only, not significance tests.'}
    (out_dir/'experiment_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved:',out_dir/'comparison.csv',out_dir/f'unit_{args.unit_id}_sigma_comparison.png',
          out_dir/f'unit_{args.unit_id}_diagnostics.png',sep='\n  ')
    print('NOTE: Do not interpret 10-shuffle 95th percentiles as statistical significance.')


if __name__=='__main__':
    main()
