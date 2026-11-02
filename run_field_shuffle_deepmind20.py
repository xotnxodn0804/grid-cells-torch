"""Evaluate a trained bottleneck using 20x20 rate maps and the released
DeepMind grid-cells 10-annulus settings, with optional watershed field-shuffle.

This is a *separate evaluation script*: no training or checkpoint changes.

DeepMind released train.py: 20 bins, starts=[0.2]*10,
ends=linspace(0.4,1.0,10); released GridScorer default uses MEAN-based
60-degree gridness. --formula minmax retains the paper-style min/max formula.
The field shuffle approximates (rather than exactly reconstructs) the
unpublished placement details of the Banino et al. procedure.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import field_shuffle_nosmoothing as base
import compare_field_smoothing as smooth


def masks_for_20():
    n = 20
    return [(0.2 * n, float(end * n)) for end in np.linspace(0.4, 1.0, 10)]


def make_scorer(masks, half: float, formula: str):
    from grid_cells.scores import GridScorer

    if formula == 'minmax':
        return GridScorer(20, ((-half, half), (-half, half)), masks)

    # The published repository's GridScorer(..., min_max=False) default.
    class ReleasedMeanScorer(GridScorer):
        def grid_score_60(self, corr):
            return (corr[60] + corr[120]) / 2.0 - (corr[30] + corr[90] + corr[150]) / 3.0

    return ReleasedMeanScorer(20, ((-half, half), (-half, half)), masks)


def process_unit(task):
    unit, rm, sigma, n_shuffles, seed, masks, half, formula = task
    scorer = make_scorer(masks, half, formula)
    observed = float(scorer.get_scores(rm)[0])
    try:
        _guide, segmented = smooth.segment_for_sigma(rm, sigma)
    except (ValueError, RuntimeError) as exc:
        return (unit, observed, float('nan'), float('nan'), 0,
                np.full(n_shuffles, np.nan), str(exc))

    rng = np.random.default_rng(np.random.SeedSequence([seed, unit]))
    null = np.empty(n_shuffles, dtype=np.float64)
    for j in range(n_shuffles):
        shuffled = base.shuffle_fields(rm, segmented, rng)
        null[j] = float(scorer.get_scores(shuffled)[0])

    if not np.all(np.isfinite(null)):
        return (unit, observed, float('nan'), float('nan'), len(segmented[1]),
                null, 'Non-finite null scores')
    cutoff = float(np.quantile(null, .95))
    p = float((1 + np.count_nonzero(null >= observed)) / (n_shuffles + 1))
    return (unit, observed, cutoff, p, len(segmented[1]), null, '')


def self_test():
    masks = masks_for_20()
    assert len(masks) == 10
    assert all(abs(lo-4.0) < 1e-10 for lo, _ in masks)
    assert abs(masks[0][1]-8.0) < 1e-10
    assert abs(masks[-1][1]-20.0) < 1e-10
    assert all(a[1] < b[1] for a, b in zip(masks, masks[1:]))
    print('PASS: 20x20 ratemap, 39x39 SAC, 10 DeepMind released-code annuli')
    print('inner: 4.0 bins; outer:', ', '.join(f'{hi:.4f}' for _, hi in masks))

    # Check relocation changes only arrangement, not values, including negatives.
    rng = np.random.default_rng(41)
    xx, yy = np.indices((20, 20))
    rm = np.exp(-((xx-6)**2+(yy-7)**2)/15.0) + .8*np.exp(-((xx-14)**2+(yy-13)**2)/15.0) - .6
    guide, segmented = smooth.segment_for_sigma(rm, .9)
    shuffled = base.shuffle_fields(rm, segmented, rng)
    assert np.array_equal(np.sort(rm.ravel()), np.sort(shuffled.ravel()))
    assert len(segmented[1]) > 0
    print('PASS: 20x20 watershed/shuffle preserves raw activity values')
    # Check the scoring formula numerically, independently of project files.
    corr = {30: .2, 60: .8, 90: .1, 120: .7, 150: .2}
    mean = (corr[60]+corr[120])/2 - (corr[30]+corr[90]+corr[150])/3
    minmax = min(corr[60],corr[120])-max(corr[30],corr[90],corr[150])
    assert np.isclose(mean, .75-1/6)
    assert np.isclose(minmax, .5)
    print('PASS: mean and minmax gridness are distinct formulas')


def result_summary(rows, args, out_dir: Path, epoch, masks):
    rows = sorted(rows, key=lambda r: r[0])
    good = [r for r in rows if np.isfinite(r[2])]
    observed = np.array([r[1] for r in rows], dtype=float)
    null95 = np.array([r[2] for r in good], dtype=float)
    mean95 = float(np.mean(null95)) if len(null95) else float('nan')
    own95 = sum(int(r[1] > r[2]) for r in good)
    p05 = sum(int(r[3] <= .05) for r in good)

    with (out_dir / 'unit_statistics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['unit', 'observed_gridness', 'shuffle_95', 'empirical_p',
                    'n_fields', 'above_own_shuffle_95', 'above_historical_0_37', 'warning'])
        for unit, score, cutoff, p, nfields, null, warning in rows:
            w.writerow([unit, score, cutoff, p, nfields,
                        int(score > cutoff) if np.isfinite(cutoff) else '',
                        int(score > .37), warning])

    np.savez_compressed(out_dir / 'null_distributions.npz',
                        units=np.array([r[0] for r in rows], dtype=int),
                        null_scores=np.stack([r[5] for r in rows]))

    summary = {
        'method': '20x20; DeepMind public GitHub 10 annuli; 0.2 inner, 0.4..1.0 outer',
        'formula': args.formula,
        'formula_note': 'mean=released GitHub default; minmax=paper-style',
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'epoch': epoch, 'n_trajectories': args.n_trajectories,
        'n_shuffles': args.n_shuffles, 'sigma_watershed_only_bins': args.sigma,
        'n_units': len(rows), 'n_valid': len(good),
        'annulus_pixels': masks,
        'mean_observed_gridness': float(np.mean(observed)),
        'mean_shuffle95_cutoff': mean95,
        'n_above_own_shuffle95': own95,
        'n_empirical_p_le_0_05': p05,
        'n_above_historical_0_37': int(np.sum(observed > .37)),
        'caveats': [
            '0.37 was calibrated under a different scoring protocol; historical reference ONLY.',
            'The watershed field relocation approximates details not fully released by Banino et al.',
            'The watershed smoothing sigma=0.9 bins is an exploratory setting, not a verified original.',
            'Do not interpret changed counts as changes in trained weights; this is evaluation only.',
            'Null95 for each unit and the mean over unit thresholds are not identical rules.'
        ],
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(observed, bins=32)
    axes[0].axvline(.37, color='gray', linestyle='--', label='0.37 historical')
    axes[0].set(title=f'20x20 observed ({args.formula})', xlabel='Gridness', ylabel='Units')
    axes[0].legend()
    if len(null95):
        axes[1].hist(null95, bins=32)
        axes[1].axvline(mean95, color='black', linestyle='--', label=f'Mean 95th={mean95:.3f}')
        axes[1].legend()
    axes[1].set(title='Per-unit shuffled 95th percentile', xlabel='Gridness threshold', ylabel='Units')
    fig.tight_layout()
    fig.savefig(out_dir / 'gridness_null95.png', dpi=150)
    plt.close(fig)

    print('\n=== 20x20 GRIDNESS + FIELD SHUFFLE ===')
    print('Formula:', args.formula)
    print(f'Observed mean:              {summary["mean_observed_gridness"]:.4f}')
    print(f'Shuffled threshold mean:    {mean95:.4f}')
    print(f'Above own shuffled 95th:   {own95}/{len(rows)}')
    print(f'Empirical p <= .05:        {p05}/{len(rows)}')
    print(f'Above historical 0.37:     {summary["n_above_historical_0_37"]}/{len(rows)} (NOT calibrated here)')
    print('Results:', out_dir.resolve())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--checkpoint')
    ap.add_argument('--shard-dir')
    ap.add_argument('--out-dir', default='results/sonnet_deepmind20_gridness')
    ap.add_argument('--n-trajectories', type=int, default=4000)
    ap.add_argument('--n-shuffles', type=int, default=100)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--bottleneck-units', type=int, default=512)
    ap.add_argument('--unit-ids', default=None, help='Comma-separated unit IDs; optional pilot')
    ap.add_argument('--sigma', type=float, default=.9, help='Watershed guide ONLY, bins; exploratory')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--formula', choices=('mean', 'minmax'), default='mean',
                    help='mean = DeepMind released default; minmax = paper Methods')
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.checkpoint or not args.shard_dir:
        ap.error('--checkpoint and --shard-dir required')
    if args.n_shuffles < 2 or args.workers < 1 or args.sigma < 0:
        ap.error('n_shuffles >= 2, workers >= 1, sigma >= 0 required')

    import torch
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.nbins = 20  # Explicit: never borrow a possibly patched global evaluator.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    maps, epoch, cfg = base.compute_ratemaps(args)
    masks = masks_for_20()
    print(f'epoch={epoch}, ratemap={args.nbins}x{args.nbins}, SAC=39x39, annuli=10')
    print('inner=4 bins; outer:', ', '.join(f'{v:.4f}' for _,v in masks))
    print('gridness formula:', args.formula)
    if args.formula == 'mean':
        print('WARNING: fixed 0.37 threshold is NOT calibrated for released mean formula')

    units = (list(range(args.bottleneck_units)) if args.unit_ids is None
             else list(dict.fromkeys(int(x.strip()) for x in args.unit_ids.split(',') if x.strip())))
    if not units or min(units)<0 or max(units)>=args.bottleneck_units:
        ap.error('Invalid --unit-ids')

    fingerprint = dict(
        version=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        deps=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (base.__file__, smooth.__file__)],
        checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_size=Path(args.checkpoint).stat().st_size,
        checkpoint_mtime=Path(args.checkpoint).stat().st_mtime_ns,
        shards=str(Path(args.shard_dir).resolve()),
        n_trajectories=args.n_trajectories,
        n_shuffles=args.n_shuffles, sigma=args.sigma,
        formula=args.formula, seed=args.seed, masks=masks,
    )
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    cache = out_dir / 'cache' / key
    cache.mkdir(parents=True, exist_ok=True)
    tasks, rows = [], []
    for unit in units:
        file = cache / f'unit_{unit:03d}.npz'
        if file.is_file():
            with np.load(file, allow_pickle=False) as z:
                rows.append((unit, float(z['observed']), float(z['cutoff']), float(z['p']),
                             int(z['nfields']), z['null'].copy(), str(z['warning'].item())))
        else:
            tasks.append((unit, maps[unit], args.sigma, args.n_shuffles,
                          args.seed, masks, cfg.task.env_size/2, args.formula))
    print(f'units={len(units)}; cached={len(rows)}; to compute={len(tasks)}', flush=True)

    def save(row):
        unit, observed, cutoff, p, nfields, null, warning = row
        np.savez_compressed(cache/f'unit_{unit:03d}.npz',
                            observed=observed, cutoff=cutoff, p=p,
                            nfields=nfields, null=null, warning=warning)
        rows.append(row)
        if len(rows)%8 == 0 or len(rows) == len(units):
            print(f'completed {len(rows)}/{len(units)}', flush=True)

    if args.workers == 1:
        for task in tasks:
            save(process_unit(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_unit, task) for task in tasks]
            for future in as_completed(futures):
                save(future.result())
    result_summary(rows, args, out_dir, epoch, masks)


if __name__ == '__main__':
    main()
