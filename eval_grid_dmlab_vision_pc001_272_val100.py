#!/usr/bin/env python3
"""Evaluate a FROZEN, already trained 272-input Grid Network on ALL 100 DMLab validation episodes.

Uses the SAME dataset loader, 4-D noisy odometry, frozen Vision Module, 5% visual
mask, deterministic validation seeds, TBPTT-style forward segmentation and metrics
as train_grid_dmlab_vision_pc001_272_noisy.py. NO optimizer and NO model training.

Optionally reproduces the original fixed 16-episode subset first, to check that
this evaluation reproduces the training log before reporting the full 100.
The trained checkpoint and both original data folders are READ ONLY.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import train_grid_dmlab_vision_pc001_272_noisy as baseline


def check_float(label: str, saved: object, requested: float) -> None:
    if not np.isclose(float(saved), float(requested), atol=1e-12, rtol=1e-9):
        raise ValueError(f'{label}: checkpoint={saved!r}, requested={requested!r}. '
                         'Use the EXACT training-time validation settings.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data-dirs', nargs='+', required=True, type=Path)
    p.add_argument('--grid-checkpoint', required=True, type=Path,
                   help='The final trained Grid latest.pt at rollout 1000')
    p.add_argument('--vision-checkpoint', required=True, type=Path,
                   help='The sigma=0.01 Vision vision_step300000.pt')
    p.add_argument('--out-json', type=Path, default=Path(
        'data/evaluations/grid_dmlab_vision_pc001_512_run1_val100.json'))
    p.add_argument('--val-batch-size', type=int, default=8)
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    p.add_argument('--compare-original-16', action=argparse.BooleanOptionalAction,
                   default=True, help='Re-evaluate the original 16 episodes first')
    a = p.parse_args()
    if a.val_batch_size < 1:
        p.error('--val-batch-size must be positive')
    paths = [path.expanduser().resolve() for path in a.data_dirs]
    if len(set(paths)) != len(paths):
        p.error('Duplicate data folders')
    if not a.grid_checkpoint.is_file() or not a.vision_checkpoint.is_file():
        raise FileNotFoundError('Grid and Vision checkpoints must exist')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device('cuda' if (a.device == 'auto' and torch.cuda.is_available())
                          else 'cpu' if a.device == 'auto' else a.device)

    # This is an evaluation program. It never constructs an optimizer or calls backward().
    ckpt = torch.load(a.grid_checkpoint, map_location='cpu', weights_only=False)
    fp = ckpt.get('fingerprint')
    if not isinstance(fp, dict):
        raise ValueError('Not a compatible Grid Network checkpoint: missing fingerprint')
    if fp.get('script_version') != 'grid_vision_dmlab_20261112_v2_272_noisy':
        raise ValueError(f'Unexpected Grid training script version: {fp.get("script_version")}')
    if fp.get('motion_source') != 'VEL.TRANS_XY_plus_measured_yaw_delta_noisy':
        raise ValueError('Wrong motion encoding: expected noisy 4-D DMLab motion')
    if fp.get('data_dirs') != [str(path) for path in paths]:
        raise ValueError('The dataset paths differ from the original training checkpoint')
    if fp.get('episode_steps') != baseline.STEPS or fp.get('pc_sigma') != baseline.SIGMA:
        raise ValueError('Episode length / Place Cell sigma mismatch')
    check_float('dt', fp['dt'], baseline.DT)
    vision_sha = baseline.file_digest(a.vision_checkpoint)
    if fp.get('vision_sha') != vision_sha:
        raise ValueError('The supplied Vision checkpoint differs from the one used for training')

    train = baseline.EpisodeStore(paths, 'train')
    val = baseline.EpisodeStore(paths, 'val')
    if len(train) != int(fp['train_episodes']) or len(val) != int(fp['val_episodes']):
        raise ValueError('Dataset episode count differs from training checkpoint')
    if train.seeds & val.seeds:
        raise ValueError('Train and validation episode seeds overlap')
    if len(val) != 100:
        raise ValueError(f'Expected 100 validation episodes, found {len(val)}')
    if fp.get('tbptt') is None or int(fp['tbptt']) < 1:
        raise ValueError('Missing valid TBPTT length')

    seed = int(fp['seed'])
    baseline.seed_everything(seed)
    grid, vision, pc, hd, v_update = baseline.make_ensembles_and_models(
        device, a.vision_checkpoint)
    if grid.rl_lstm.input_size != 272 or tuple(grid.rl_lstm.weight_ih_l0.shape) != (512, 272):
        raise ValueError('Grid model architecture is not the required 272-input variant')
    # Strict loading catches accidentally using an old 3-D motion model.
    grid.load_state_dict(ckpt['grid'], strict=True)
    if not torch.equal(ckpt['pc_centers'].cpu(), pc.means.cpu()):
        raise ValueError('Grid checkpoint PC center positions/order mismatch')
    if not torch.equal(ckpt['hd_centers'].cpu(), hd.means.cpu()):
        raise ValueError('Grid checkpoint HD center positions/order mismatch')
    grid.eval()
    vision.eval()
    vision.requires_grad_(False)
    grid.requires_grad_(False)

    # Same per-episode deterministic visual mask, noise and validation logic used
    # during training. For this evaluator, only the number of episodes changes.
    settings = SimpleNamespace(
        val_batch_size=a.val_batch_size,
        vision_rate=float(fp['vision_rate']),
        force_first_vision=bool(fp['force_first']),
        speed_noise_std=float(fp['speed_noise_std']),
        angular_noise_std=float(fp['angular_noise_std']),
        tbptt=int(fp['tbptt']), seed=seed)
    print(f'DATA OK | train={len(train)}, val={len(val)} episodes; '
          f'{baseline.STEPS} steps/ep, dt={baseline.DT:.6f}s', flush=True)
    print(f'GRID LOADED | rollout={ckpt["rollout"]:,}, '
          f'optimizer_updates={ckpt.get("optimizer_updates", "unknown")}, '
          f'W_ih={tuple(grid.rl_lstm.weight_ih_l0.shape)}', flush=True)
    print(f'VISION LOADED | step={v_update:,}, sigma={baseline.SIGMA}m, '
          f'sha256={vision_sha[:14]}', flush=True)
    print(f'VALIDATION SETTINGS | vision_rate={settings.vision_rate}, '
          f'force_first_vision={settings.force_first_vision}, '
          f'speed_noise_std={settings.speed_noise_std}m/s, '
          f'angular_noise_std={settings.angular_noise_std}rad/s, '
          f'tbptt={settings.tbptt}, seed={seed}', flush=True)

    # Original fixed 16 chosen exactly as in the training program (not the first 16).
    original_ids = np.sort(np.random.default_rng(seed + 129731).choice(
        len(val), size=16, replace=False))
    if fp.get('val_ids') != original_ids.tolist():
        raise ValueError('Original 16 validation IDs do not match stored checkpoint')

    old_report = None
    if a.compare_original_16:
        print('CHECK | re-evaluating original fixed 16 validation episodes...', flush=True)
        old_report = baseline.validate(grid, vision, pc, hd, val, original_ids, device, settings)
        print(f'ORIGINAL 16 | CE={old_report["ce"]:.4f} | '
              f'pos(top3)={old_report["pos_top3_cm"]:.2f}cm | '
              f'final={old_report["final_pos_top3_cm"]:.2f}cm | '
              f'HD={old_report["hd_circular_deg"]:.2f}deg', flush=True)

    full_ids = np.arange(len(val), dtype=np.int64)  # ALL 100; no random subset.
    print('EVALUATING ALL 100 VALIDATION EPISODES (130,000 time steps)...', flush=True)
    full_report = baseline.validate(grid, vision, pc, hd, val, full_ids, device, settings)
    print('=== VALIDATION: ALL 100 EPISODES ===', flush=True)
    for label, key, unit in (
        ('Cross entropy', 'ce', ''),
        ('Position Top-3', 'pos_top3_cm', ' cm'),
        ('Position weighted mean', 'pos_weighted_cm', ' cm'),
        ('Position argmax', 'pos_argmax_cm', ' cm'),
        ('Head direction (circular)', 'hd_circular_deg', ' deg'),
        ('Final-step position Top-3', 'final_pos_top3_cm', ' cm'),
        ('Final-step head direction', 'final_hd_deg', ' deg')):
        print(f'  {label}: {full_report[key]:.4f}{unit}', flush=True)

    result = {
        'grid_checkpoint': str(a.grid_checkpoint.expanduser().resolve()),
        'grid_checkpoint_sha256': baseline.file_digest(a.grid_checkpoint),
        'rollout': int(ckpt['rollout']),
        'optimizer_updates': int(ckpt['optimizer_updates']),
        'vision_checkpoint': str(a.vision_checkpoint.expanduser().resolve()),
        'vision_checkpoint_sha256': vision_sha,
        'vision_update': int(v_update),
        'data_dirs': [str(path) for path in paths],
        'settings': vars(settings),
        'original_16': {'episode_ids': original_ids.tolist(), 'metrics': old_report}
                        if old_report is not None else None,
        'full_100': {'episode_ids': full_ids.tolist(), 'episodes': len(full_ids),
                     'time_steps': len(full_ids)*baseline.STEPS, 'metrics': full_report},
    }
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out_json.with_name(a.out_json.name + '.tmp')
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(a.out_json)
    print(f'REPORT SAVED | {a.out_json.resolve()}', flush=True)
    print('READ-ONLY EVALUATION COMPLETE. Grid and Vision weights were NOT updated.', flush=True)


if __name__ == '__main__':
    main()
