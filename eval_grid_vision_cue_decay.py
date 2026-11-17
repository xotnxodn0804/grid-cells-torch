#!/usr/bin/env python3
"""Analyse cue correction and the error trajectory after a visual observation.

Standalone evaluation for train_grid_vision_supervised.py (272-input, frozen CNN).
Does NOT train or alter any model checkpoint.

Three complementary analyses on the SAME held-out validation snippets:
  A. Natural 5% visual cues, grouped by steps since the most recent cue.
  B. Controlled single visual cue at t=20 followed by motion-only input,
     paired against an identical motion-only run on the same trajectories.
  C. Remove a SINGLE naturally occurring 5% cue, leaving all other cues intact.
     Compare against the original sequence until the next cue (paired intervention).

The second test deliberately changes the frequency of cues relative to training;
its results are a diagnostic, not a matched-distribution validation score.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from config import Config
from grid_cells.ensembles import build_rl_ensembles
from train_grid_vision_supervised import (
    Episodes, VisionCorrectedGrid, file_hash, make_batch, set_seed,
)


def err_arrays(p_logits, hd_logits, pos, heading, pc, hd):
    """(B,T) position error (cm), heading error (degrees)."""
    pp = p_logits.softmax(-1)
    hp = hd_logits.softmax(-1)
    xy = pp @ pc.means
    yaw = torch.atan2(hp @ torch.sin(hd.means), hp @ torch.cos(hd.means))
    pos_err = torch.linalg.vector_norm(xy - pos, dim=-1) * 100.0
    angle = torch.atan2(torch.sin(yaw - heading), torch.cos(yaw - heading))
    return pos_err.detach().cpu().numpy(), torch.rad2deg(angle.abs()).detach().cpu().numpy()


def visual_features(model, images: np.ndarray, keep: torch.Tensor, device):
    """Build [B,T,268] PC/HD probability vectors only at selected cue times."""
    b, t = keep.shape
    out = torch.zeros((b * t, 268), dtype=torch.float32, device=device)
    indices = torch.nonzero(keep.reshape(-1), as_tuple=False).flatten()
    if indices.numel() == 0:
        return out.reshape(b, t, 268)
    rgb_flat = images.reshape(b * t, 3, 64, 64)
    # Work in chunks (both host and device) to bound RAM/VRAM usage.
    for sub_indices in indices.split(128):
        ids = sub_indices.cpu().numpy()
        rgb = torch.from_numpy(np.ascontiguousarray(rgb_flat[ids])).to(device)
        rgb = rgb.float().div_(127.5).sub_(1.0)
        z = model.vision.cnn(rgb)
        v = torch.cat((model.vision.place_head(z).softmax(-1),
                       model.vision.hd_head(z).softmax(-1)), dim=-1)
        out.index_copy_(0, sub_indices, v)
    return out.reshape(b, t, 268)


def bucket(age: int) -> str:
    if age < 0:
        return 'before-first-cue'
    if age == 0:
        return '0 (cue step)'
    if age == 1:
        return '1'
    if age == 2:
        return '2'
    if age <= 5:
        return '3-5'
    if age <= 10:
        return '6-10'
    if age <= 20:
        return '11-20'
    if age <= 40:
        return '21-40'
    return '41+'


BUCKETS = ['before-first-cue', '0 (cue step)', '1', '2', '3-5',
           '6-10', '11-20', '21-40', '41+']


def add_bucket(stats, key, a, b, c, d):
    obj = stats.setdefault(key, np.zeros(5, dtype=np.float64))
    obj += np.array((1.0, float(a), float(b), float(c), float(d)))


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data-dirs', nargs='+', type=Path, required=True)
    p.add_argument('--vision-checkpoint', type=Path, required=True)
    p.add_argument('--grid-checkpoint', type=Path, required=True)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=100)
    p.add_argument('--val-batches', type=int, default=32)
    p.add_argument('--cue-step', type=int, default=20,
                   help='Step for the controlled single-cue experiment (0-based).')
    p.add_argument('--out-dir', type=Path, default=Path('results/grid_vision_cue_decay_100k'))
    args = p.parse_args()
    if args.batch_size < 1 or args.seq_len < 2 or args.val_batches < 1:
        p.error('Positive batch-size/val-batches and seq-len >= 2 required')
    if not (0 <= args.cue_step < args.seq_len - 1):
        p.error('--cue-step must leave at least one subsequent step')

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; check WSL PyTorch environment')
    grid_path = args.grid_checkpoint.resolve()
    vision_path = args.vision_checkpoint.resolve()
    dirs = [d.resolve() for d in args.data_dirs]
    checkpoint = torch.load(grid_path, map_location='cpu', weights_only=False)
    if checkpoint.get('grid_input') != 272 or checkpoint.get('grid_bottleneck') != 512:
        raise ValueError('Expected checkpoint with input 272 / bottleneck 512')
    fp = checkpoint['fingerprint']
    if fp.get('seq_len') != args.seq_len or fp.get('batch_size') != args.batch_size:
        raise ValueError('Checkpoint training batch-size/seq-len differ; use original settings')
    if [str(d) for d in dirs] != fp['data_dirs']:
        raise ValueError('Validation dataset paths do not match checkpoint')
    if file_hash(vision_path) != fp['vision_checkpoint_sha256']:
        raise ValueError('Frozen vision checkpoint differs from one used in training')
    vision_prob = float(fp['vision_prob'])
    if not (0.0 < vision_prob <= 1.0):
        raise ValueError('Invalid trained visual-cue probability')
    seed = int(fp['seed'])
    set_seed(seed)
    cfg = Config()
    pc_list, hd_list = build_rl_ensembles(cfg, device)
    pc, hd = pc_list[0], hd_list[0]
    model = VisionCorrectedGrid(pc, hd, vision_path, device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.grid.eval()
    model.vision.eval()
    val = Episodes(dirs, 'val')

    rng_data = np.random.default_rng(seed + 909012)  # same as standard validation
    rng_cues = torch.Generator(device=device)
    rng_cues.manual_seed(seed + 939911)
    stats_age = {}
    stats_one_cue_removed = {}
    eligible_cues_removed = 0
    t = args.seq_len
    single_pos = np.zeros((2, t), dtype=np.float64)
    single_hd = np.zeros((2, t), dtype=np.float64)
    summary_sum = np.zeros((2, 2), dtype=np.float64)  # normal/zero by pos/hd
    natural_cues = 0
    total = 0
    print(f'CHECKPOINT | {grid_path} | updates={checkpoint["step"]:,} | '
          f'vision={vision_prob*100:.1f}%')
    print(f'VALIDATION | {args.val_batches} batches × {args.batch_size} trajectories × '
          f'{t} steps = {args.val_batches*args.batch_size*t:,} points')
    print('MODE A: natural visual-cue intervals (5% schedule)')
    print(f'MODE B: one cue at t={args.cue_step}; all later inputs are motion-only')

    with torch.inference_mode():
        for i in range(args.val_batches):
            images, movement, _, _, pos, heading = make_batch(
                val, rng_data, args.batch_size, t, pc, hd, cfg, device)
            b = args.batch_size
            cue_mask = torch.rand((b, t), device=device, generator=rng_cues) < vision_prob
            v_nat = visual_features(model, images, cue_mask, device)
            zero = torch.zeros_like(v_nat)
            p_normal, h_normal, _ = model.grid(torch.cat((movement, v_nat), -1))
            p_zero, h_zero, _ = model.grid(torch.cat((movement, zero), -1))
            err_nat_pos, err_nat_hd = err_arrays(p_normal, h_normal, pos, heading, pc, hd)
            err_zero_pos, err_zero_hd = err_arrays(p_zero, h_zero, pos, heading, pc, hd)
            summary_sum[0] += (err_nat_pos.sum(), err_nat_hd.sum())
            summary_sum[1] += (err_zero_pos.sum(), err_zero_hd.sum())
            total += b * t
            mask = cue_mask.cpu().numpy()
            natural_cues += int(mask.sum())
            for j in range(b):
                last = -1
                for k in range(t):
                    if mask[j, k]:
                        last = k
                    age = (k - last) if last >= 0 else -1
                    add_bucket(stats_age, bucket(age), err_nat_pos[j, k],
                               err_nat_hd[j, k], err_zero_pos[j, k], err_zero_hd[j, k])

            # Causal paired test within the familiar 5% cue schedule:
            # remove exactly one naturally occurring cue from each eligible sequence;
            # leave every OTHER cue, motion input and LSTM initialization unchanged.
            # Count effects only until the NEXT visual cue, if there is one.
            for j in range(b):
                cue_indices = np.flatnonzero(mask[j])
                eligible = cue_indices[cue_indices >= min(20, t-2)]
                if not len(eligible):
                    continue
                cue_t = int(eligible[0])
                later = cue_indices[cue_indices > cue_t]
                end_t = int(later[0]) if len(later) else t
                no_one = v_nat[j:j+1].clone()
                no_one[:, cue_t, :] = 0.0
                p_remove, h_remove, _ = model.grid(
                    torch.cat((movement[j:j+1], no_one), dim=-1))
                pe_remove, he_remove = err_arrays(
                    p_remove, h_remove, pos[j:j+1], heading[j:j+1], pc, hd)
                eligible_cues_removed += 1
                for k in range(cue_t, end_t):
                    add_bucket(stats_one_cue_removed, bucket(k-cue_t),
                               err_nat_pos[j,k], err_nat_hd[j,k],
                               pe_remove[0,k], he_remove[0,k])

            # Matched intervention: ONLY one cue at same fixed time for each trajectory.
            one_mask = torch.zeros((b, t), dtype=torch.bool, device=device)
            one_mask[:, args.cue_step] = True
            v_one = visual_features(model, images, one_mask, device)
            p_single, h_single, _ = model.grid(torch.cat((movement, v_one), -1))
            err_one_pos, err_one_hd = err_arrays(p_single, h_single, pos, heading, pc, hd)
            single_pos[0] += err_one_pos.sum(axis=0)
            single_pos[1] += err_zero_pos.sum(axis=0)
            single_hd[0] += err_one_hd.sum(axis=0)
            single_hd[1] += err_zero_hd.sum(axis=0)
            if (i + 1) % max(1, args.val_batches // 4) == 0 or i + 1 == args.val_batches:
                print(f'  evaluated {i+1}/{args.val_batches} batches', flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_a = []
    for key in BUCKETS:
        count, pnat, hnat, pzero, hzero = stats_age.get(key, np.zeros(5))
        if count < 1:
            continue
        rows_a.append({
            'cue_age_steps': key, 'points': int(count),
            'normal_position_cm': round(pnat/count, 4),
            'normal_hd_deg': round(hnat/count, 4),
            'no_vision_position_cm': round(pzero/count, 4),
            'no_vision_hd_deg': round(hzero/count, 4),
            'normal_minus_no_vision_position_cm': round((pnat-pzero)/count, 4),
            'normal_minus_no_vision_hd_deg': round((hnat-hzero)/count, 4),
        })
    columns_a = list(rows_a[0])
    write_csv(args.out_dir/'natural_cue_age_bins.csv', rows_a, columns_a)
    rows_causal = []
    for key in BUCKETS[1:]:
        count, pfull, hfull, pdrop, hdrop = stats_one_cue_removed.get(key, np.zeros(5))
        if count < 1:
            continue
        rows_causal.append({
            'steps_since_cue': key, 'paired_points': int(count),
            'normal_position_cm': round(pfull/count,4),
            'remove_one_cue_position_cm': round(pdrop/count,4),
            'benefit_position_cm': round((pdrop-pfull)/count,4),
            'normal_hd_deg': round(hfull/count,4),
            'remove_one_cue_hd_deg': round(hdrop/count,4),
            'benefit_hd_deg': round((hdrop-hfull)/count,4),
        })
    if rows_causal:
        write_csv(args.out_dir/'single_natural_cue_removed.csv', rows_causal,
                  list(rows_causal[0]))

    n_traj = args.val_batches * args.batch_size
    single_pos /= n_traj
    single_hd /= n_traj
    rows_b = []
    for k in range(t):
        rows_b.append({
            'step': k, 'steps_since_forced_cue': k-args.cue_step,
            'single_cue_position_cm': round(float(single_pos[0, k]), 4),
            'no_vision_position_cm': round(float(single_pos[1, k]), 4),
            'cue_benefit_position_cm': round(float(single_pos[1, k]-single_pos[0, k]), 4),
            'single_cue_hd_deg': round(float(single_hd[0, k]), 4),
            'no_vision_hd_deg': round(float(single_hd[1, k]), 4),
            'cue_benefit_hd_deg': round(float(single_hd[1, k]-single_hd[0, k]), 4),
        })
    write_csv(args.out_dir/'single_cue_by_step.csv', rows_b, list(rows_b[0]))
    # Summarize single-cue effects by the same age bins (post-cue only).
    post = {key: [] for key in BUCKETS}
    for row in rows_b[args.cue_step:]:
        post[bucket(row['steps_since_forced_cue'])].append(row)
    rows_c = []
    for key in BUCKETS[1:]:
        vals = post[key]
        if not vals:
            continue
        r = {'steps_since_cue': key, 'points': len(vals)*n_traj}
        for field in ('single_cue_position_cm', 'no_vision_position_cm',
                      'cue_benefit_position_cm', 'single_cue_hd_deg',
                      'no_vision_hd_deg', 'cue_benefit_hd_deg'):
            r[field] = round(float(np.mean([v[field] for v in vals])), 3)
        rows_c.append(r)
    write_csv(args.out_dir/'single_cue_age_bins.csv', rows_c, list(rows_c[0]))
    print('\nA: NATURAL 5% CUE SCHEDULE (same trajectories/motion)')
    print(f'Visual cues: {natural_cues:,}/{total:,} ({100*natural_cues/total:.2f}%)')
    print(f'Overall: normal pos={summary_sum[0,0]/total:.2f}cm, '
          f'HD={summary_sum[0,1]/total:.2f}deg | '
          f'0%-cue pos={summary_sum[1,0]/total:.2f}cm, '
          f'HD={summary_sum[1,1]/total:.2f}deg')
    print(f'{"Age since last cue":<23} {"N":>6} {"Pos cm":>10} {"HD deg":>10}')
    for row in rows_a:
        print(f'{row["cue_age_steps"]:<23} {row["points"]:>6} '
              f'{row["normal_position_cm"]:>10.2f} {row["normal_hd_deg"]:>10.2f}')
    print('\nC: REMOVING ONE NATURAL 5% CUE (paired intervention)')
    print(f'Paired cue events tested: {eligible_cues_removed:,} / {n_traj:,} sequences')
    print('Benefit = error with ONE cue removed - error with normal cue (positive means cue helped)')
    print(f'{"Steps since cue":<18} {"N":>6} {"Pos benefit":>13} {"HD benefit":>12}')
    for r in rows_causal:
        print(f'{r["steps_since_cue"]:<18} {r["paired_points"]:>6} '
              f'{r["benefit_position_cm"]:>13.2f} {r["benefit_hd_deg"]:>12.2f}')
    print('\nB: SINGLE CUE, THEN MOTION ONLY vs NEVER SEEING A CUE')
    print(f'{"Steps since cue":<18} {"Pos cue":>9} {"Pos zero":>9} {"HD cue":>9} {"HD zero":>9}')
    for r in rows_c:
        print(f'{r["steps_since_cue"]:<18} {r["single_cue_position_cm"]:>9.2f} '
              f'{r["no_vision_position_cm"]:>9.2f} '
              f'{r["single_cue_hd_deg"]:>9.2f} {r["no_vision_hd_deg"]:>9.2f}')
    meta = {
        'checkpoint': str(grid_path), 'trained_updates': int(checkpoint['step']),
        'vision_checkpoint': str(vision_path), 'val_episodes': int(val.n_episodes),
        'val_trajectories': int(n_traj), 'val_steps_per_traj': int(t),
        'natural_vision_probability': vision_prob,
        'natural_total_vision_cues': natural_cues, 'total_points': total,
        'single_cue_step': args.cue_step,
        'paired_natural_cue_events': eligible_cues_removed,
        'training_grid_zero_initialized': True,
        'interpretation': 'Cue-age association and paired single-cue intervention on sampled '
                          'DMLab trajectories. Not a direct measurement of intrinsic PI drift; '
                          'controlled one-cue schedule differs from training.'
    }
    (args.out_dir/'analysis_metadata.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        x = np.arange(t - args.cue_step)
        for series, label, filename, unit in (
            (single_pos, 'Position error', 'single_cue_position.png', 'cm'),
            (single_hd, 'Head-direction error', 'single_cue_hd.png', 'deg'),
        ):
            fig, ax = plt.subplots(figsize=(7.5, 4.4))
            ax.plot(x, series[0, args.cue_step:], label='Visual cue at t='+str(args.cue_step))
            ax.plot(x, series[1, args.cue_step:], label='Motion only (no vision)')
            ax.set(xlabel='Steps since the single visual cue', ylabel=f'{label} ({unit})',
                   title='Held-out validation: one cue followed by motion only')
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(args.out_dir/filename, dpi=165)
            plt.close(fig)
        print('Plots saved: single_cue_position.png, single_cue_hd.png')
    except ImportError:
        print('matplotlib unavailable: CSV results saved; plots skipped')
    print(f'OUTPUT | {args.out_dir.resolve()}')
    print('INTERPRETATION CAUTION: Cue-age bins may mix different starting states. '
          'Single-cue test gives a matched no-cue control, but a 1-cue schedule is '
          'different from the 5%-cue training distribution. Neither metric alone '
          'proves pure path-integration drift.')


if __name__ == '__main__':
    main()
