#!/usr/bin/env python3
"""Exploratory 20x20 Gridness for 512-unit noisy-odometry Grid LSTM.

Run in the user's grid-cells-torch WSL checkout next to
train_grid_vision_512_noisy_motion.py. No weights, datasets, or checkpoints are modified.

Uses every non-overlapping --seq-len window from held-out validation episodes,
restarting the LSTM at zero at each window, as in the training setup. The learned
vision module is frozen and is seen on ~5% of time steps. The grid bottleneck is
recorded BEFORE dropout/nonlinearity. Rate maps are occupancy-normalized, not
smoothed. Gridness uses DeepMind's public 20-bin, 10-annulus, MEAN formula.

IMPORTANT: score > 0.37 is a descriptive historical comparator, NOT a significance
test for this dataset. A future field-shuffle null is needed for statistical claims.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def accumulate(sums, counts, xy, activations, half, nbins):
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    activations = np.asarray(activations, dtype=np.float32).reshape(-1, sums.shape[1])
    valid = (np.isfinite(xy).all(axis=1) & (np.abs(xy) <= half + 1e-5).all(axis=1)
             & np.isfinite(activations).all(axis=1))
    if not np.any(valid):
        return 0
    bins_xy = np.floor((xy[valid] + half) * nbins / (2 * half)).astype(np.int32)
    bins_xy = np.clip(bins_xy, 0, nbins - 1)
    ids = bins_xy[:, 0] * nbins + bins_xy[:, 1]
    counts += np.bincount(ids, minlength=nbins * nbins)
    a = activations[valid]
    for unit in range(sums.shape[1]):
        sums[:, unit] += np.bincount(ids, weights=a[:, unit], minlength=nbins*nbins)
    return int(valid.sum())


def rates(sums, counts, n, units):
    out = np.full_like(sums, np.nan, dtype=np.float64)
    good = counts > 0
    out[good] = sums[good] / counts[good, None]
    return out.T.reshape(units, n, n)


def split_corr(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 50:
        return float('nan')
    x, y = a[mask], b[mask]
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def groups_for_val(source, length):
    """Yield individual windows with their actual validation episode index."""
    episode_global = 0
    for entry in source.entries:
        steps = entry['steps']
        if steps < length:
            raise ValueError(f'Episode length {steps} < requested window {length}')
        windows = steps // length
        arrays = entry['arrays']
        for ep in range(entry['episodes']):
            for w in range(windows):
                start = ep * steps + w * length
                s = slice(start, start + length)
                if not np.all(arrays['valid_motion'][s] == 1):
                    continue  # training excluded teleport-containing windows
                yield (episode_global,
                       np.asarray(arrays['images'][s]),
                       np.asarray(arrays['pos'][s]),
                       np.asarray(arrays['vel_trans'][s]),
                       np.asarray(arrays['ego_vel'][s]))
            episode_global += 1


def plot_outputs(out, rmap, sacs, scores, stability, counts, threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    order = np.argsort(np.nan_to_num(scores, nan=-1e8))[::-1]

    fig, ax = plt.subplots(figsize=(7.3, 4.4))
    valid = scores[np.isfinite(scores)]
    if len(valid):
        ax.hist(valid, bins=45)
    ax.axvline(threshold, linestyle='--', color='black',
               label=f'Historical threshold {threshold:g} (not p-value)')
    ax.set_xlabel('Mean 60-degree Gridness score')
    ax.set_ylabel('Bottleneck units')
    ax.set_title('Gridness distribution (20x20, DeepMind mean formula)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'gridness_histogram.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    p = ax.imshow(counts.reshape(20, 20).T, origin='lower', cmap='viridis')
    ax.set_title('Held-out validation occupancy (20x20)')
    ax.set_xlabel('X bin')
    ax.set_ylabel('Y bin')
    fig.colorbar(p, ax=ax, label='Visits')
    fig.tight_layout()
    fig.savefig(out / 'occupancy.png', dpi=140)
    plt.close(fig)

    with PdfPages(out / 'all_512_ratemaps.pdf') as pdf:
        for offset in range(0, len(order), 64):
            fig, axs = plt.subplots(8, 8, figsize=(16, 17))
            subset = order[offset:offset+64]
            fig.suptitle(f'Bottleneck activation rate maps (before dropout) | '
                         f'units {offset+1}-{offset+len(subset)} / {len(order)}\n'
                         'S = mean 10-annulus Gridness, no significance test', fontsize=13)
            for ax in axs.flat:
                ax.axis('off')
            for ax, u in zip(axs.flat, subset):
                rm = rmap[u]
                ax.imshow(rm.T, origin='lower', cmap='viridis', interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                ax.axis('on')
                ax.set_title(f'#{u}  G={scores[u]:.2f}', fontsize=8)
            fig.subplots_adjust(left=.035, right=.98, bottom=.025, top=.91,
                                wspace=.26, hspace=.52)
            pdf.savefig(fig)
            plt.close(fig)
            print(f'  saved rate map page {offset//64+1}/{(len(order)+63)//64}', flush=True)

    top = order[:min(32, len(order))]
    with PdfPages(out / 'top_32_ratemaps_sacs.pdf') as pdf:
        for offset in range(0, len(top), 8):
            fig, axs = plt.subplots(4, 4, figsize=(13, 13))
            fig.suptitle('Top-Gridness bottleneck units: rate map (left) / SAC (right)\n'
                         'Historical 0.37 threshold is descriptive only', fontsize=13)
            for j, u in enumerate(top[offset:offset+8]):
                left, right = axs.flat[2*j:2*j+2]
                left.imshow(rmap[u].T, origin='lower', cmap='viridis', interpolation='nearest')
                sac = sacs[u]
                limit = max(0.01, float(np.nanmax(np.abs(sac))))
                right.imshow(sac.T, origin='lower', cmap='RdBu_r',
                             vmin=-limit, vmax=limit, interpolation='nearest')
                left.set_title(f'Unit {u} | G={scores[u]:.3f}', fontsize=10)
                right.set_title(f'SAC | split-half r={stability[u]:.2f}', fontsize=10)
                for a in (left, right):
                    a.set_xticks([]); a.set_yticks([])
            fig.tight_layout(rect=(0,.01,1,.92))
            pdf.savefig(fig)
            plt.close(fig)


def self_test():
    n = 20
    sums = np.zeros((n*n, 3), dtype=np.float64)
    counts = np.zeros(n*n, dtype=np.int64)
    xy = np.array([[-1.0,-1.0],[-1.0,-1.0],[0.2,0.3]],dtype=float)
    acts = np.array([[2.,3.,4.],[4.,5.,6.],[8.,9.,10.]],dtype=float)
    assert accumulate(sums, counts, xy, acts, 1.25, 20)==3
    m = rates(sums, counts,20,3)
    x = int(np.floor((-.1e1+1.25)*20/2.5))
    assert abs(m[0,x,x]-3.)<1e-6
    assert counts.sum()==3 and np.isfinite(m).sum()==6
    assert abs(split_corr(np.arange(100),np.arange(100))-1)<1e-9
    print('SELF-TEST PASS: occupancy aggregation, signed means, split correlations')


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data-dirs', nargs='+', type=Path)
    p.add_argument('--vision-checkpoint', type=Path)
    p.add_argument('--grid-checkpoint', type=Path)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=100)
    p.add_argument('--max-sequences', type=int, default=0,
                   help='0 = all validation windows (default); >0 for a quick preview')
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--out-dir', type=Path, default=Path('results/grid_vision_gridness_512'))
    p.add_argument('--threshold', type=float, default=.37,
                   help='Historical descriptive threshold, NOT a p-value')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if not args.data_dirs or args.vision_checkpoint is None or args.grid_checkpoint is None:
        p.error('--data-dirs, --vision-checkpoint and --grid-checkpoint are required')
    if args.batch_size < 1 or args.seq_len < 1 or args.max_sequences < 0:
        p.error('batch-size/seq-len must be positive, max-sequences must be >=0')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this WSL environment')

    from config import Config
    from grid_cells.ensembles import build_rl_ensembles
    from grid_cells.scores import GridScorer, deepmind_mask_parameters
    from train_grid_vision_512_noisy_motion import (
        Episodes, VisionCorrectedGrid, file_hash, set_seed, noisy_motion,
    )

    ckpt_path = args.grid_checkpoint.expanduser().resolve()
    vision_path = args.vision_checkpoint.expanduser().resolve()
    dirs = [d.expanduser().resolve() for d in args.data_dirs]
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if checkpoint.get('grid_input') != 272 or checkpoint.get('grid_bottleneck') != 512:
        raise ValueError('This script expects the 272-input / 512-bottleneck checkpoint')
    fp = checkpoint['fingerprint']
    if fp.get('motion_source') != 'collector_measured_yaw_v2_valid_only_plus_gaussian_noise':
        raise ValueError('Expected a noisy measured-yaw checkpoint')
    for name in ('speed_noise_std_m_per_s', 'angular_noise_std_rad_per_s', 'angular_encoding'):
        if name not in fp:
            raise ValueError(f'Missing noise metadata {name}')
    if fp['data_dirs'] != [str(d) for d in dirs]:
        raise ValueError('Validation folders differ from the training checkpoint')
    if fp['vision_checkpoint_sha256'] != file_hash(vision_path):
        raise ValueError('Frozen vision checkpoint differs from the training checkpoint')
    if args.seq_len != int(fp['seq_len']):
        raise ValueError(f'Training used seq_len={fp["seq_len"]}; use the same length')
    if abs(float(fp['vision_prob']) - .05) > 1e-10:
        raise ValueError('Expected a Grid model trained with 5% visual cues')

    cfg = Config()
    set_seed(int(fp['seed']))
    pces, hdes = build_rl_ensembles(cfg, device)
    pc, hd = pces[0], hdes[0]
    model = VisionCorrectedGrid(pc, hd, vision_path, device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.grid.eval(); model.vision.eval()

    source = Episodes(dirs, 'val')
    total_windows = sum(1 for _ in groups_for_val(source, args.seq_len))
    budget = min(args.max_sequences, total_windows) if args.max_sequences else total_windows
    nbins, units, half = 20, 512, cfg.rl.env.env_size_m/2
    sums = np.zeros((nbins*nbins, units), dtype=np.float64)
    counts = np.zeros(nbins*nbins, dtype=np.int64)
    split_sums = np.zeros((2, nbins*nbins, units), dtype=np.float64)
    split_counts = np.zeros((2, nbins*nbins), dtype=np.int64)
    gen = torch.Generator(device=device).manual_seed(int(fp['seed']) + 939911)
    noise_rng = np.random.default_rng(int(fp['seed']) + 909013)
    scale = cfg.rl.env.env_size_m / (cfg.rl.env.wall_world_units[1] - cfg.rl.env.wall_world_units[0])
    dt = cfg.rl.env.action_repeat / 60.0
    n_seen = n_valid = n_cues = 0
    stash = []
    print(f'CHECKPOINT | {ckpt_path} | update {checkpoint["step"]:,}', flush=True)
    print(f'VALIDATION | {source.n_episodes} held-out episodes; '
          f'{budget} non-overlapping sequences × {args.seq_len} steps; '
          f'vision cues 5%; device={device}', flush=True)
    if args.max_sequences:
        print('WARNING: limited preview may have poor spatial coverage', flush=True)

    def process_batch(batch):
        nonlocal n_seen, n_valid, n_cues
        ids, images, positions, vt, ego = zip(*batch)
        img = np.stack(images)
        position_np = np.stack(positions)
        b = len(batch); t = args.seq_len
        movement_np = noisy_motion(np.stack(vt), np.stack(ego), noise_rng, scale, dt,
                                   float(fp['speed_noise_std_m_per_s']),
                                   float(fp['angular_noise_std_rad_per_s']),
                                   str(fp['angular_encoding']))
        movement = torch.from_numpy(movement_np).to(device)
        _, _, g, seen = model(img, movement, vision_keep_prob=.05, generator=gen)
        activ = g.detach().cpu().numpy()
        n_cues += seen
        n_seen += b*t
        n_valid += accumulate(sums, counts, position_np, activ, half, nbins)
        for k, ep_idx in enumerate(ids):
            parity = ep_idx % 2
            accumulate(split_sums[parity], split_counts[parity],
                       position_np[k], activ[k], half, nbins)

    with torch.inference_mode():
        for item in groups_for_val(source, args.seq_len):
            if (n_seen // args.seq_len) + len(stash) >= budget:
                break
            stash.append(item)
            if len(stash) >= args.batch_size:
                process_batch(stash)
                stash.clear()
                if n_seen % (200 * args.seq_len) == 0 or n_seen == budget*args.seq_len:
                    print(f'  processed {n_seen//args.seq_len}/{budget} sequences', flush=True)
        if stash:
            process_batch(stash)
    if n_seen != budget*args.seq_len:
        raise RuntimeError(f'Expected {budget*args.seq_len} samples, collected {n_seen}')
    if not np.any(counts):
        raise RuntimeError('No valid position samples in the arena')
    rmaps = rates(sums, counts, nbins, units)
    half_maps = [rates(split_sums[i], split_counts[i], nbins, units) for i in range(2)]
    stability = np.array([split_corr(half_maps[0][u],half_maps[1][u])
                          for u in range(units)])

    class MeanScorer(GridScorer):
        def grid_score_60(self, correlations):
            return ((correlations[60] + correlations[120]) / 2.0
                    - (correlations[30]+correlations[90]+correlations[150]) / 3.0)

    scorer = MeanScorer(nbins, ((-half,half),(-half,half)), deepmind_mask_parameters(nbins))
    scores = np.full(units,np.nan)
    masks = np.zeros((units,2))
    sacs = []
    for u in range(units):
        rm = rmaps[u]
        if np.isfinite(rm).sum()<50 or np.nanstd(rm)<1e-8:
            sacs.append(np.zeros((2*nbins-1,2*nbins-1)))
            continue
        score, mask, sac = scorer.get_scores(rm)
        scores[u] = float(score)
        masks[u] = mask
        sacs.append(sac)
        if (u+1)%128==0:
            print(f'  scored {u+1}/{units} bottleneck units', flush=True)

    args.out_dir.mkdir(parents=True,exist_ok=True)
    dest = args.out_dir
    order = np.argsort(np.nan_to_num(scores,nan=-1e8))[::-1]
    with (dest / 'gridness_scores_sorted.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['unit','gridness_mean_10annuli','annulus_inner_bins',
                         'annulus_outer_bins','rate_map_split_half_pearson_r',
                         'over_historical_0_37_NOT_SIGNIFICANCE'])
        for u in order:
            writer.writerow([int(u), f'{scores[u]:.6f}' if np.isfinite(scores[u]) else '',
                             f'{masks[u,0]:.3f}', f'{masks[u,1]:.3f}',
                             f'{stability[u]:.6f}' if np.isfinite(stability[u]) else '',
                             int(scores[u]>args.threshold)])
    np.savez_compressed(dest/'ratemaps_and_scores.npz',
                        ratemaps=rmaps.astype(np.float32),
                        gridness=scores, occupancy=counts.reshape(nbins,nbins),
                        split_half_stability=stability, selected_annuli=masks)
    plot_outputs(dest,rmaps,sacs,scores,stability,counts,args.threshold)
    passing = np.flatnonzero(scores>args.threshold)
    report = {
        'experiment':'frozen_vision_grid_supervised_272_512_noisy_motion_v1',
        'motion_source': fp['motion_source'],
        'speed_noise_std_m_per_s': fp['speed_noise_std_m_per_s'],
        'angular_noise_std_rad_per_s': fp['angular_noise_std_rad_per_s'],
        'angular_encoding': fp['angular_encoding'],
        'checkpoint':str(ckpt_path),'update':int(checkpoint['step']),
        'data_split':'held-out validation only',
        'validation_episodes':source.n_episodes,
        'n_sequences':budget,'seq_len':args.seq_len,
        'n_samples':n_seen,'valid_position_samples':n_valid,
        'cues_transmitted':n_cues,'vision_fraction_observed':n_cues/n_seen,
        'grid_bottleneck_units':units,'arena_m':cfg.rl.env.env_size_m,
        'bins':nbins,'bins_visited':int(np.count_nonzero(counts)),
        'bin_min_visit_count':int(counts.min()),
        'gridness_method':'20x20, 10 DeepMind annuli, mean 60 vs mean 30/90/150, no smoothing',
        'above_historical_0_37':int(len(passing)),
        'top_10':[{'unit':int(u),'gridness':float(scores[u])} for u in order[:10]
                  if np.isfinite(scores[u])],
        'note':'Threshold >0.37 is NOT a significant-grid-cell count. No field-shuffle null. All evaluated windows have valid motion (no teleport), and LSTM restarts at zero.'
    }
    (dest/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n=== GRIDNESS RESULTS (EXPLORATORY) ===',flush=True)
    print(f'Checkpoint update: {report["update"]:,} | bins visited: {report["bins_visited"]}/400 '
          f'| min visits per bin: {report["bin_min_visit_count"]}',flush=True)
    print(f'Cues: {n_cues}/{n_seen} = {n_cues/n_seen:.2%}',flush=True)
    print(f'Historical score > {args.threshold:g}: {len(passing)}/{units} units '
          '(NOT a statistical significance test)',flush=True)
    print('Top 10: '+', '.join(f'#{u}:{scores[u]:.3f}' for u in order[:10]),flush=True)
    print('Outputs:',dest.resolve(),flush=True)
    for name in ['gridness_scores_sorted.csv','gridness_histogram.png',
                 'all_512_ratemaps.pdf','top_32_ratemaps_sacs.pdf','occupancy.png',
                 'ratemaps_and_scores.npz','summary.json']:
        print(' ',name,flush=True)


if __name__=='__main__':
    main()
