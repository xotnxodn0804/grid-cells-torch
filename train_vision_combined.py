#!/usr/bin/env python3
"""Train IMAGE-ONLY vision module from multiple completed DMLab dataset folders.

Designed for collect_vision_rh_1000_v2.py outputs. Dataset files are memory-mapped,
not copied or concatenated. Training uses RGB ONLY; DMLab measured position/heading
are used exclusively for PC/HDC soft-label targets and evaluation.

Example:
  python train_vision_combined.py --check-only \
    --data-dirs data/vision_rh_pilot_100 data/vision_rh_extra_900
  python train_vision_combined.py \
    --data-dirs data/vision_rh_pilot_100 data/vision_rh_extra_900 \
    --out-dir data/checkpoints/vision_rh_combined_1000 \
    --device cuda --updates 10000

The script does NOT change config.py, grid_cells/vision.py, previous datasets,
or previous checkpoints. Its in-file truncated-normal initializer has the same
implementation as the pilot vision_300k.py, avoiding a dependency on that pilot.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


class MultiSplit:
    """Read-only, weighted concatenation of many complete .npy datasets."""

    def __init__(self, folders: list[Path], split: str):
        self.split = split
        self.entries: list[dict[str, Any]] = []
        counts = []
        episode_seed_ids: set[int] = set()
        for folder in folders:
            marker = folder / f'{split}_complete.json'
            settings = folder / 'dataset_settings.json'
            if not marker.is_file():
                raise FileNotFoundError(f'Incomplete or missing {split} split: {marker}')
            if not settings.is_file():
                raise FileNotFoundError(f'Missing dataset settings: {settings}')
            meta, setup = load_json(marker), load_json(settings)
            if setup.get('collector_version') != 2:
                raise ValueError(f'Unsupported collector version in {folder}')
            for key in ('images', 'pos', 'hd', 'episode_id'):
                if not (folder / f'{split}_{key}.npy').is_file():
                    raise FileNotFoundError(folder / f'{split}_{key}.npy')
            arrays = {key: np.load(folder / f'{split}_{key}.npy', mmap_mode='r')
                      for key in ('images', 'pos', 'hd', 'episode_id')}
            images, poses, headings, ids = (arrays[k] for k in ('images','pos','hd','episode_id'))
            n = len(images)
            expected = int(meta['episodes']) * int(meta['steps_per_episode'])
            if (meta.get('frames') != n or n != expected or
                poses.shape != (n, 2) or headings.shape != (n,) or
                ids.shape != (n,) or images.shape != (n, 3, 64, 64) or
                images.dtype != np.uint8 or poses.dtype != np.float32 or
                headings.dtype != np.float32):
                raise ValueError(f'Split {folder}/{split} has incorrect dimensions or types')
            if meta.get('env_size_m') != 2.5:
                raise ValueError(f'Unexpected arena size in {folder}: {meta.get("env_size_m")}')
            if meta.get('steps_per_episode') != setup.get('steps_per_episode'):
                raise ValueError(f'Steps mismatch in {folder}')
            if (split == 'train' and meta['episodes'] != setup['episodes'] - setup['val_episodes']
                or split == 'val' and meta['episodes'] != setup['val_episodes']):
                raise ValueError(f'Episode count mismatch in {folder}/{split}')
            # Prevent accidentally combining identical DMLab episode seeds.
            seeds = {int(meta['seed']) + i for i in range(int(meta['episodes']))}
            repeated = episode_seed_ids.intersection(seeds)
            if repeated:
                raise ValueError(f'Duplicate {split} environment seeds across directories: {list(repeated)[:5]}')
            episode_seed_ids.update(seeds)
            # Check episode indexing with small, representative reads (without full scan).
            steps = int(meta['steps_per_episode'])
            sample_eps = sorted({0, int(meta['episodes'])//2, int(meta['episodes'])-1})
            for ep in sample_eps:
                if int(ids[ep*steps]) != ep or int(ids[(ep+1)*steps-1]) != ep:
                    raise ValueError(f'Episode ID mismatch in {folder} at episode {ep}')
            counts.append(n)
            self.entries.append({'folder': str(folder.resolve()), 'images': images,
                                 'pos': poses, 'hd': headings, 'frames': n,
                                 'episodes': int(meta['episodes']), 'seed': int(meta['seed'])})

        self.offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
        self.total = int(self.offsets[-1])
        self.episodes = sum(e['episodes'] for e in self.entries)

    def get_batch(self, global_indices: np.ndarray, device: torch.device):
        idx = np.asarray(global_indices, dtype=np.int64)
        if idx.ndim != 1 or not len(idx) or np.any(idx < 0) or np.any(idx >= self.total):
            raise ValueError('Invalid batch global indices')
        out_img = np.empty((len(idx), 3, 64, 64), dtype=np.uint8)
        out_pos = np.empty((len(idx), 2), dtype=np.float32)
        out_hd = np.empty((len(idx),), dtype=np.float32)
        source_ids = np.searchsorted(self.offsets[1:], idx, side='right')
        for s in np.unique(source_ids):
            which = np.flatnonzero(source_ids == s)
            local = idx[which] - self.offsets[s]
            e = self.entries[int(s)]
            out_img[which] = e['images'][local]
            out_pos[which] = e['pos'][local]
            out_hd[which] = e['hd'][local]
        image = torch.from_numpy(out_img).to(device).float().div_(127.5).sub_(1.0)
        pos = torch.from_numpy(out_pos).to(device)
        heading = torch.from_numpy(out_hd).to(device)
        return image, pos, heading

    def describe(self):
        return ', '.join(f"{Path(e['folder']).name}: {e['episodes']} eps / {e['frames']:,} imgs"
                         for e in self.entries)



def init_truncated_normal_vision(model: nn.Module) -> None:
    """Same Sonnet-style initialization used by the standalone pilot.

    Conv2d/Linear: fan-in inverse square-root standard deviation,
    truncated to +/-2 standard deviations; biases set to zero.
    This initializer does not change grid_cells/vision.py (shared by RL).
    """
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                fan_in = m.weight[0].numel()
                std = fan_in ** -0.5
                nn.init.trunc_normal_(m.weight, mean=0.0, std=std,
                                      a=-2 * std, b=2 * std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(model, pc, hdc, source: MultiSplit, global_ids: np.ndarray,
             device: torch.device, batch_size: int):
    model.eval()
    ce_sum = pos_err_sum = hd_err_sum = 0.0
    sin_hd = torch.sin(hdc.means)
    cos_hd = torch.cos(hdc.means)
    for left in range(0, len(global_ids), batch_size):
        ids = global_ids[left:left+batch_size]
        images, positions, headings = source.get_batch(ids, device)
        target_pc = pc.posterior(positions.unsqueeze(1)).squeeze(1)
        target_hdc = hdc.posterior(headings.unsqueeze(1)).squeeze(1)
        e = model.cnn(images)
        place_logits = model.place_head(e)
        head_logits = model.hd_head(e)
        loss = (-(target_pc * F.log_softmax(place_logits, -1)).sum(-1)
                -(target_hdc * F.log_softmax(head_logits, -1)).sum(-1))
        pprob = F.softmax(place_logits, dim=-1)
        hprob = F.softmax(head_logits, dim=-1)
        pred_xy = pprob @ pc.means
        pred_hd = torch.atan2(hprob @ sin_hd, hprob @ cos_hd)
        angular_error = torch.atan2(torch.sin(pred_hd - headings),
                                    torch.cos(pred_hd - headings)).abs()
        ce_sum += loss.sum().item()
        pos_err_sum += (pred_xy - positions).norm(dim=-1).sum().item() * 100.0
        hd_err_sum += torch.rad2deg(angular_error).sum().item()
    n = len(global_ids)
    model.train()
    return (ce_sum/n, pos_err_sum/n, hd_err_sum/n)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data-dirs', nargs='+', required=True, type=Path,
                   help='Multiple completed collector v2 folders, e.g. pilot_100 extra_900')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--out-dir', default='data/checkpoints/vision_rh_combined_1000')
    p.add_argument('--updates', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--eval-samples', type=int, default=10000,
                   help='Fixed random images per split used for comparable train/val metrics')
    p.add_argument('--eval-batch-size', type=int, default=256)
    p.add_argument('--save-every', type=int, default=5000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('auto','cuda','cpu'), default='auto')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    for k in ('updates','batch_size','eval_every','eval_samples','eval_batch_size','save_every'):
        if getattr(a,k) < 1: p.error(f'--{k.replace("_","-")} must be >=1')
    if not (0 < a.lr < 1): p.error('--lr out of range')
    folders = [x.expanduser().resolve() for x in a.data_dirs]
    if len(set(folders)) != len(folders):
        p.error('Duplicate dataset directory')
    train = MultiSplit(folders, 'train')
    val = MultiSplit(folders, 'val')
    # Cross-split seed collision check (val seeds should not occur in train).
    train_seeds = {e['seed']+i for e in train.entries for i in range(e['episodes'])}
    val_seeds = {e['seed']+i for e in val.entries for i in range(e['episodes'])}
    if train_seeds & val_seeds:
        raise ValueError('Training/validation episode seeds overlap')
    print(f'DATA READY | train: {train.episodes} episodes, {train.total:,} images; '
          f'val: {val.episodes} episodes, {val.total:,} images', flush=True)
    print(f'  train = {train.describe()}')
    print(f'  val   = {val.describe()}')
    print('  RGB images are the ONLY model inputs; DMLab actual pos/heading make targets.')
    if a.check_only:
        print('CHECK OK. No training was started.')
        return

    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if a.device == 'auto' else a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; activate your WSL PyTorch CUDA environment')
    print('DEVICE:', device, ('| ' + torch.cuda.get_device_name(0)) if device.type == 'cuda' else '')
    set_seed(a.seed)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    # Same model / PC-HD ensembles / optimizer / truncated-normal init as the pilot.
    from config import Config
    from grid_cells.ensembles import build_rl_ensembles
    from grid_cells.vision import VisionModule

    cfg = Config()
    pc_e, hd_e = build_rl_ensembles(cfg, device)
    if len(pc_e) != 1 or len(hd_e) != 1:
        raise ValueError('Expected one PC and one HD ensemble')
    pc, hd = pc_e[0], hd_e[0]
    model = VisionModule(pc, hd, embed_dim=256, mask_prob=0.95,
                         mask_mode='whole').to(device)
    init_truncated_normal_vision(model)
    optim = torch.optim.RMSprop(model.parameters(), lr=a.lr, alpha=.99, eps=1e-8)
    rng = np.random.default_rng(a.seed)
    # Frozen evaluation sets, chosen independently of batch sampling RNG.
    eval_rng = np.random.default_rng(a.seed + 129731)
    train_eval_ids = np.sort(eval_rng.choice(train.total,
        size=min(a.eval_samples, train.total), replace=False))
    val_eval_ids = np.sort(eval_rng.choice(val.total,
        size=min(a.eval_samples, val.total), replace=False))

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    last = out / 'latest.pt'
    csv_file = out / 'metrics.csv'
    fingerprint = {'data_dirs': [str(x) for x in folders],
                   'train_images': train.total, 'val_images': val.total,
                   'train_episodes': train.episodes, 'val_episodes': val.episodes,
                   'seed': a.seed, 'lr': a.lr, 'batch_size': a.batch_size,
                   'eval_samples': a.eval_samples}
    start = 0
    if a.resume:
        if not last.is_file():
            raise FileNotFoundError(f'Cannot resume: missing {last}')
        checkpoint = torch.load(last, map_location=device, weights_only=False)
        if checkpoint.get('dataset_fingerprint') != fingerprint:
            raise ValueError('Checkpoint was made with different data/train settings')
        model.load_state_dict(checkpoint['vision_module'])
        optim.load_state_dict(checkpoint['optimizer'])
        start = int(checkpoint['update'])
        rng.bit_generator.state = checkpoint['numpy_rng_state']
        torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
        if device.type == 'cuda' and 'cuda_rng_state' in checkpoint:
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu(), device)
        print(f'RESUMED at step {start:,}', flush=True)
    elif last.exists() or csv_file.exists():
        raise FileExistsError(f'Existing run in {out}; use --resume or a new --out-dir')
    if start >= a.updates:
        print(f'Already at {start:,} updates; requested {a.updates:,}. Nothing to train.')
        return
    columns = ['update','minibatch_ce','train_ce','val_ce','train_pos_cm',
               'val_pos_cm','train_hd_deg','val_hd_deg']

    def save_checkpoint(step: int):
        obj = {
            'vision_module': model.state_dict(), 'optimizer': optim.state_dict(),
            'update': step, 'dataset_fingerprint': fingerprint,
            'train_frames': train.total, 'val_frames': val.total,
            'pc_centers': pc.means.detach().cpu(), 'hd_centers': hd.means.detach().cpu(),
            'pc_stdev': cfg.rl.pc_scale, 'hd_concentration': cfg.task.hdc_concentration,
            'env_size_m': cfg.rl.env.env_size_m,
            'seed': a.seed, 'init': 'truncated_normal_fan_in', 'vision_mask_mode': 'whole',
            'numpy_rng_state': rng.bit_generator.state,
            'torch_rng_state': torch.get_rng_state().cpu(),
        }
        if device.type == 'cuda':
            obj['cuda_rng_state'] = torch.cuda.get_rng_state(device).cpu()
        tmp = out / 'latest.tmp.pt'
        torch.save(obj, tmp)
        os.replace(tmp, last)
        if step % a.save_every == 0 or step == a.updates:
            torch.save(obj, out / f'vision_step{step:06d}.pt')

    with open(csv_file, 'a' if a.resume else 'w', newline='', buffering=1) as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        if not a.resume:
            writer.writeheader()
        model.train()
        step = start
        try:
            for step in range(start+1, a.updates+1):
                idx = rng.integers(0, train.total, size=a.batch_size)
                imgs, positions, headings = train.get_batch(idx, device)
                pc_target = pc.posterior(positions.unsqueeze(1)).squeeze(1)
                hd_target = hd.posterior(headings.unsqueeze(1)).squeeze(1)
                loss = model.loss(imgs, pc_target, hd_target)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                if step % a.eval_every == 0 or step == a.updates:
                    tr = evaluate(model, pc, hd, train, train_eval_ids, device, a.eval_batch_size)
                    va = evaluate(model, pc, hd, val, val_eval_ids, device, a.eval_batch_size)
                    writer.writerow({'update':step, 'minibatch_ce':float(loss.item()),
                                    'train_ce':tr[0], 'val_ce':va[0],
                                    'train_pos_cm':tr[1], 'val_pos_cm':va[1],
                                    'train_hd_deg':tr[2], 'val_hd_deg':va[2]})
                    print(f'step {step:,}/{a.updates:,} | train/val CE {tr[0]:.4f}/{va[0]:.4f} '
                          f'| pos {tr[1]:.1f}/{va[1]:.1f}cm '
                          f'| HD {tr[2]:.1f}/{va[2]:.1f}deg', flush=True)
                if step % a.save_every == 0 or step == a.updates:
                    save_checkpoint(step)
        except KeyboardInterrupt:
            print(f'Interrupted at step {step}; saving resumable checkpoint.', flush=True)
            save_checkpoint(step)
    print('RESULTS:', out.resolve())


if __name__ == '__main__':
    main()
