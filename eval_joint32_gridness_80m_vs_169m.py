#!/usr/bin/env python3
"""READ-ONLY matched-path Gridness of 80M vs 169M Joint-32 checkpoints.

Run from ~/grid-cells-torch (WSL repo root), even if this file is saved in
C:\\Users\\xotnd\\GitHubProjects\\grid-cells-torch on Windows.

Uses the 169M Actor solely to generate SHARED behavioral trajectories. Both
Grid/Vision pairs see identical observations, noisy motion and visual cue masks.
Analysis: signed 512D Grid bottleneck activations (not PC logits), 20x20 raw
occupancy-normalized spatial maps, 10 DeepMind annuli, mean and min-max 60deg
Gridness, and odd/even-episode split-half spatial consistency. 0.37 is a
DESCRIPTIVE threshold, NOT statistical significance or grid-cell proof.

Does not modify model checkpoints, replay buffers or training metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import torch
import torch.nn.functional as F

BINS = 20
UNITS = 512
CUTOFF = 0.37
LABELS = ('80M', '169M')


def allocate():
    return {'sums': np.zeros((BINS * BINS, UNITS), np.float64),
            'halves': np.zeros((2, BINS * BINS, UNITS), np.float64)}


def spatial_bin(pos):
    p = np.asarray(pos, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(p)) or np.any(np.abs(p) > 1.25001):
        return None
    ij = np.clip(np.floor((p + 1.25) * BINS / 2.5).astype(int), 0, BINS - 1)
    return int(ij[0] * BINS + ij[1])


def rate_maps(sums, counts):
    r = np.full((BINS * BINS, UNITS), np.nan, dtype=np.float32)
    good = counts > 0
    r[good] = (sums[good] / counts[good, None]).astype(np.float32)
    return r.T.reshape(UNITS, BINS, BINS)


def split_corr(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 50:
        return float('nan')
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def grid_scores(maps):
    from grid_cells.scores import GridScorer, deepmind_mask_parameters

    class MeanGridScorer(GridScorer):
        def grid_score_60(self, corr):
            return (corr[60] + corr[120]) / 2 - (corr[30] + corr[90] + corr[150]) / 3

    kw = dict(nbins=BINS, coords_range=((-1.25, 1.25), (-1.25, 1.25)),
              mask_parameters=deepmind_mask_parameters(BINS))
    scorer_mean = MeanGridScorer(**kw)
    scorer_minmax = GridScorer(**kw)
    mean = np.full(UNITS, np.nan, np.float64)
    minmax = np.full(UNITS, np.nan, np.float64)
    sac = np.full((UNITS, 2 * BINS - 1, 2 * BINS - 1), np.nan, np.float32)
    for u, rm in enumerate(maps):
        if np.isfinite(rm).sum() < 50 or np.nanstd(rm) < 1e-9:
            continue
        mean[u], _, ac = scorer_mean.get_scores(rm)
        minmax[u], _, _ = scorer_minmax.get_scores(rm)
        sac[u] = ac
        if (u + 1) % 128 == 0:
            print(f'SCORED | {u + 1}/{UNITS} units', flush=True)
    return mean, minmax, sac


def plotting(outdir, models, occ):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    img = ax.imshow(occ.reshape(BINS, BINS).T, origin='lower', cmap='viridis')
    fig.colorbar(img, ax=ax, label='Observations per spatial bin')
    ax.set(xlabel='X bin', ylabel='Y bin', title='Shared 169M-policy occupancy (20x20)')
    fig.tight_layout()
    fig.savefig(outdir / 'occupancy.png', dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for label, color in [('80M', 'steelblue'), ('169M', 'darkorange')]:
        st = models[label]
        for ax, field, name in zip(axes, ('mean', 'minmax'), ('Mean-correlation', 'Min-max')):
            vals = st[field][np.isfinite(st[field])]
            if len(vals):
                ax.hist(vals, bins=35, alpha=.5, color=color, label=label)
            ax.axvline(CUTOFF, color='black', lw=1, ls='--', label='0.37 descriptive')
            ax.set(xlabel='Gridness score', ylabel='Number of units',
                   title=f'{name} Gridness (10 annuli)')
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / 'gridness_comparison.png', dpi=190)
    plt.close(fig)

    for label in LABELS:
        st = models[label]
        ranked = np.argsort(np.nan_to_num(st['mean'], nan=-1e9))[::-1]
        for field, fname, cmap in [('maps', 'ratemaps', 'viridis'),
                                   ('sac', 'autocorrelograms', 'RdBu_r')]:
            fig, axs = plt.subplots(4, 8, figsize=(18, 9.6))
            for ax, u in zip(axs.flat, ranked[:32]):
                kwargs = dict(origin='lower', interpolation='nearest', cmap=cmap)
                if field == 'sac':
                    kwargs.update(vmin=-1, vmax=1)
                ax.imshow(st[field][u].T, **kwargs)
                ax.set_title(f'unit {u} | G={st["mean"][u]:.2f} | r={st["stability"][u]:.2f}',
                             fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f'{label} | top-32 signed Grid bottleneck {fname}', fontsize=14)
            fig.tight_layout()
            fig.savefig(outdir / f'{label}_top32_{fname}.png', dpi=170)
            plt.close(fig)


def run(args):
    from train_joint_nav_online import make_models, GridCodeActorCritic
    from train_joint_nav_32actors import seed_all, choose_device, ALGORITHM, digest
    from train_joint_nav_32actors import batched_motions, batched_grid_step, grid_hidden_init, actor_hidden
    from eval_joint32_80m_vs_169m import make_runner, model_observation

    device = choose_device(args.device)
    paths = {'80M': args.checkpoint_80m, '169M': args.checkpoint_169m}
    if not args.initial_checkpoint.is_file():
        raise FileNotFoundError(f'Initial B checkpoint missing: {args.initial_checkpoint}')
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'{label} checkpoint missing: {path}\n'
                                    'NEVER substitute latest.pt if ongoing 269M training might overwrite it.')
    cks = {label: torch.load(path, map_location='cpu', weights_only=False)
           for label, path in paths.items()}
    for label in LABELS:
        ck = cks[label]
        if ck.get('algorithm') != ALGORITHM:
            raise RuntimeError(f'{label} checkpoint algorithm mismatch')
        print(f'CHECKPOINT {label} | frames={ck["frames"]:,} path={paths[label]}', flush=True)
    if int(cks['80M']['frames']) != 80_000_000:
        raise RuntimeError('Expected exactly 80M checkpoint')
    if not 160_000_000 <= int(cks['169M']['frames']) <= 175_000_000:
        raise RuntimeError('Expected 169M checkpoint, not a live newer latest.pt')
    if cks['80M']['initial_sha256'] != cks['169M']['initial_sha256']:
        raise RuntimeError('Pretrained B checkpoint hash mismatch')
    if digest(args.initial_checkpoint) != cks['80M']['initial_sha256']:
        raise RuntimeError('Supplied initial B checkpoint differs from checkpoint provenance')
    seed0 = int(cks['80M'].get('args', {}).get('seed', 42))
    if seed0 != int(cks['169M'].get('args', {}).get('seed', 42)):
        raise RuntimeError('Different model training seeds')
    seed_all(seed0)
    cfg, _, _, grid_initial, vision_initial = make_models(device, args.initial_checkpoint)
    actor_initial = GridCodeActorCritic().to(device)
    net = {}
    for label in LABELS:
        import copy
        grid = copy.deepcopy(grid_initial)
        vision = copy.deepcopy(vision_initial)
        actor = copy.deepcopy(actor_initial)
        grid.load_state_dict(cks[label]['grid'], strict=True)
        vision.load_state_dict(cks[label]['vision'], strict=True)
        actor.load_state_dict(cks[label]['actor'], strict=True)
        grid.eval(); vision.eval(); actor.eval()
        net[label] = (grid, vision, actor)
    print('CHECK PASS | model 512D bottlenecks and both checkpoints loaded read-only', flush=True)
    if args.mode == 'check':
        return
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f'Output directory not empty: {args.out_dir}')
    if args.episodes < 2:
        raise ValueError('For split-half consistency require >= 2 episodes')
    if not 0 <= args.vision_rate <= 1:
        raise ValueError('Vision cue rate must be in [0,1]')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    occ = np.zeros(BINS * BINS, np.int64)
    half_occ = np.zeros((2, BINS * BINS), np.int64)
    accum = {label: allocate() for label in LABELS}
    total_seen = 0; n_hits = 0
    visits_per_episode = []
    runner = make_runner(cfg, args.renderer, args.env_seed_base)
    try:
        obs = runner.reset()
        with torch.inference_mode():
            for ep in range(args.episodes):
                gh = {label: grid_hidden_init(1, device) for label in LABELS}
                ah = actor_hidden(1, device)
                pa = torch.zeros(1, dtype=torch.long, device=device)
                pr = torch.zeros(1, 1, device=device)
                goal_mem = torch.zeros(1, UNITS, device=device)
                known_goal = False
                last_obs = None
                force_reset = True
                rng = np.random.default_rng(args.motion_seed_base + ep)
                action_rng = torch.Generator(device=device)
                action_rng.manual_seed(args.action_seed_base + ep)
                episode_bins = set()
                hits = 0
                done = False
                for t in range(1, 1351):
                    obs_b = model_observation(obs)
                    last_b = None if last_obs is None else model_observation(last_obs)
                    reset = np.array([force_reset], dtype=np.bool_)
                    if last_obs is not None and np.linalg.norm(obs['pos'] - last_obs['pos']) > .35:
                        reset[0] = True
                    motion = batched_motions(obs_b, last_b, reset, rng)
                    seen = bool(reset[0] or rng.random(1)[0] < args.vision_rate)
                    total_seen += int(seen)
                    visual_mask = np.array([seen], dtype=np.bool_)
                    code = {}
                    for label in LABELS:
                        grid, vision, _ = net[label]
                        code[label], gh[label] = batched_grid_step(
                            grid, vision, obs_b, motion, visual_mask,
                            gh[label], reset, device)
                    loc = spatial_bin(obs['pos'])
                    if loc is not None:
                        occ[loc] += 1
                        half_occ[ep % 2, loc] += 1
                        episode_bins.add(loc)
                        for label in LABELS:
                            v = code[label][0].detach().cpu().numpy()
                            accum[label]['sums'][loc] += v
                            accum[label]['halves'][ep % 2, loc] += v
                    # Identical behavior from 169M actor, whose own Grid Code is used.
                    logits, _, ah = net['169M'][2].step(code['169M'], goal_mem, pa, pr, ah)
                    prob = F.softmax(logits, dim=-1)
                    if not bool(torch.isfinite(prob).all()):
                        raise RuntimeError('Non-finite action probabilities')
                    action = int(torch.multinomial(prob[0], 1, generator=action_rng).item())
                    nxt, reward, done = runner.step(action)
                    if reward > 0:
                        hits += 1; n_hits += 1
                        if not known_goal:
                            known_goal = True
                            goal_mem = code['169M'].detach().clone()
                    pa = torch.tensor([action], dtype=torch.long, device=device)
                    pr = torch.tensor([[reward]], dtype=torch.float32, device=device)
                    force_reset = bool(reward > 0 or done)
                    last_obs, obs = obs, nxt
                    if done:
                        break
                if not done:
                    raise RuntimeError(f'Episode {ep} did not end within 1350 actions')
                visits_per_episode.append(len(episode_bins))
                if ep == 0 or (ep + 1) % 5 == 0:
                    print(f'COLLECT | episode={ep+1}/{args.episodes} hits={hits} '
                          f'unique_bins={len(episode_bins)}/{BINS*BINS}', flush=True)
    finally:
        runner.close()
    n_visited = int(np.sum(occ > 0))
    print(f'OCCUPANCY | {n_visited}/400 bins visited | {int(occ.sum()):,} samples | '
          f'vision cues={total_seen/int(occ.sum()):.2%} | goals={n_hits}', flush=True)
    if n_visited < 360:
        print('WARNING | Less than 90% occupancy; gridness may be unreliable.', flush=True)
    stats = {}
    for label in LABELS:
        m = accum[label]
        maps = rate_maps(m['sums'], occ)
        halves = [rate_maps(m['halves'][k], half_occ[k]) for k in range(2)]
        stability = np.array([split_corr(halves[0][u], halves[1][u])
                              for u in range(UNITS)])
        print(f'SCORE | {label} | 512 Grid Code dimensions', flush=True)
        mean, minmax, sac = grid_scores(maps)
        stats[label] = dict(maps=maps, sac=sac, mean=mean,
                            minmax=minmax, stability=stability)
        n_mean = int(np.sum(mean > CUTOFF))
        n_minmax = int(np.sum(minmax > CUTOFF))
        print(f'GRIDNESS | {label}: mean G>{CUTOFF}: {n_mean}/512; '
              f'minmax G>{CUTOFF}: {n_minmax}/512 (DESCRIPTIVE ONLY)', flush=True)
    summary = {
        'models': {},
        'checkpoint_frames': {k: int(cks[k]['frames']) for k in LABELS},
        'checkpoint_paths': {k: str(paths[k]) for k in LABELS},
        'episodes': args.episodes,
        'samples': int(occ.sum()),
        'occupied_bins': n_visited,
        'bins': BINS,
        'mean_bins_visited_per_episode': float(np.mean(visits_per_episode)),
        'vision_rate': args.vision_rate,
        'actual_vision_fraction': total_seen / int(occ.sum()),
        'behavior_policy': '169M trained actor, shared trajectories, both Grid/Vision pairs',
        'grid_definition': 'signed 512D linear Grid bottleneck; NOT place outputs',
        'scoring': 'raw occupancy-normalized 20x20 maps, 39x39 autocorrelation, 10 DeepMind annuli',
        'threshold_caution': '>0.37 is descriptive. No shuffle significance, FDR, or biological proof.',
        'sampling_caution': 'Trained goal navigation may oversample rewarded locations.',
    }
    for label in LABELS:
        st = stats[label]
        order = np.argsort(np.nan_to_num(st['mean'], nan=-1e9))[::-1]
        np.savez_compressed(args.out_dir / f'{label}_gridness_data.npz',
                            maps=st['maps'], autocorrelograms=st['sac'],
                            scores_mean=st['mean'], scores_minmax=st['minmax'],
                            split_half_r=st['stability'], occupancy=occ.reshape(BINS,BINS))
        with (args.out_dir / f'{label}_gridness_sorted.csv').open('w',
                                                                 newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['unit', 'gridness_mean', 'gridness_minmax',
                             'split_half_r', 'mean_gt_037_descriptive',
                             'minmax_gt_037_descriptive'])
            for u in order:
                writer.writerow([int(u), float(st['mean'][u]), float(st['minmax'][u]),
                                 float(st['stability'][u]), int(st['mean'][u] > CUTOFF),
                                 int(st['minmax'][u] > CUTOFF)])
        summary['models'][label] = {
            'mean_gt_037': int(np.sum(st['mean'] > CUTOFF)),
            'minmax_gt_037': int(np.sum(st['minmax'] > CUTOFF)),
            'valid_mean': int(np.isfinite(st['mean']).sum()),
            'top10': [{'unit': int(u), 'gridness_mean': float(st['mean'][u]),
                       'gridness_minmax': float(st['minmax'][u]),
                       'split_half_r': float(st['stability'][u])}
                      for u in order[:10]],
        }
    (args.out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    plotting(args.out_dir, stats, occ)
    print('FINISHED |', args.out_dir.resolve(), flush=True)
    print('READ ONLY | Training/checkpoint/replay data untouched', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--mode', choices=('self-test', 'check', 'eval'), default='check')
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='cuda')
    p.add_argument('--renderer', choices=('software', 'hardware'), default='software')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--vision-rate', type=float, default=.05)
    p.add_argument('--env-seed-base', type=int, default=70000)
    p.add_argument('--motion-seed-base', type=int, default=80000)
    p.add_argument('--action-seed-base', type=int, default=90000)
    p.add_argument('--initial-checkpoint', type=Path, default=Path(
        'data/checkpoints/grid1350_online_B_seq_run1/grid_78k_analysis.pt'))
    p.add_argument('--checkpoint-80m', type=Path, default=Path(
        'data/checkpoints/joint_nav_32actors_200m/latest_backup_80m.pt'))
    p.add_argument('--checkpoint-169m', type=Path, default=Path(
        'data/checkpoints/joint_nav_32actors_200m/latest_backup_169m.pt'))
    p.add_argument('--out-dir', type=Path, default=Path(
        'results/joint32_gridness_80m_vs_169m_100ep'))
    args = p.parse_args()
    if args.mode == 'self-test':
        assert spatial_bin([0., 0.]) == 10 * BINS + 10
        assert spatial_bin([-5., 0.]) is None
        sums = np.zeros((BINS * BINS, UNITS), np.float64)
        count = np.zeros(BINS * BINS, np.int64)
        count[10 * BINS + 10] = 2
        sums[10 * BINS + 10, 0] = 3
        m = rate_maps(sums, count)
        assert m.shape == (512, 20, 20)
        assert abs(m[0, 10, 10] - 1.5) < 1e-8
        assert np.isnan(m[0, 0, 0])
        assert abs(split_corr(np.arange(100.), np.arange(100.)) - 1.) < 1e-8
        print('SELF TEST PASS | rate map binning, count normalization and stability', flush=True)
        return
    run(args)


if __name__ == '__main__':
    main()
