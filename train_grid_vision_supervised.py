#!/usr/bin/env python3
"""Train a vision-corrected Grid Network on existing DMLab trajectories.

Built for the xotnxodn0804/grid-cells-torch repository (2026-11).

Pipeline (per DMLab action block):
 RGB --frozen pretrained VisionModule--> PC256 + HD12 predictions
 motion                                  --> [u,v,sin(delta_yaw),cos(delta_y)]
 vision predictions are made available with P=0.05 (otherwise 268 zeros).
 4+268=272 inputs --> LSTM128 --> linear bottleneck512 (dropout 0.5)
 --> PC256 + HD12 logits --> supervised soft cross-entropy to DMLab GT.

No ground-truth positions/headings are fed to the model; they are ONLY labels.
Grid LSTM starts with ZERO cell and hidden states, not oracle initial states.
All NEW grid trainable weights are N(0, fan_in^-1), truncated at +/-2 sigma;
all grid biases are 0 except forget gate bias +1. Pretrained CNN is NOT reset.

The collector v2 stores whole sequential episodes; this script samples slices
contained within one episode, never across an episode boundary. Measured
VEL.TRANS/VEL.ROT are the motion observations as provided by DMLab; their
instantaneous nature means this is a first experimental approximation.

Example (from original ~/grid-cells-torch, in .venv-dmlab):
 python train_grid_vision_supervised.py --check-only \
   --data-dirs data/vision_rh_pilot_100 data/vision_rh_extra_900 \
   --vision-checkpoint data/checkpoints/vision_rh_combined_1000/latest.pt

 python train_grid_vision_supervised.py \
   --data-dirs data/vision_rh_pilot_100 data/vision_rh_extra_900 \
   --vision-checkpoint data/checkpoints/vision_rh_combined_1000/latest.pt \
   --updates 100 --batch-size 4 --eval-every 25 --save-every 50 --device cuda \
   --out-dir data/checkpoints/grid_vision_272_smoke

Long run: --updates 10000 --batch-size 8 --eval-every 500 --save-every 1000
          --out-dir data/checkpoints/grid_vision_272_run1

Run --resume with the same directory and same data/model/train settings to
continue to a larger --updates value. Training and validation episode seeds
are disjoint by data collection. Frozen vision only computes ~5% of the RGB
images per batch, rather than doing unnecessary CNN work on the 95% masked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from grid_cells.ensembles import build_rl_ensembles, soft_cross_entropy
from grid_cells.rl_common import ego_velocity
from grid_cells.vision import VisionModule


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Episodes:
    """Read-only memory-mapped DMLab trajectories from collector version 2."""
    KEYS = ('images', 'pos', 'hd', 'vel_trans', 'vel_rot', 'episode_id')

    def __init__(self, dirs: list[Path], split: str):
        self.split = split
        self.entries = []
        self.episode_lookup = []
        all_seeds = set()
        for folder in dirs:
            setup = json.loads((folder / 'dataset_settings.json').read_text())
            meta = json.loads((folder / f'{split}_complete.json').read_text())
            if int(setup.get('collector_version', -1)) != 2:
                raise ValueError(f'{folder}: expected collect_vision_rh_1000_v2.py data')
            n_ep = int(meta['episodes'])
            steps = int(meta['steps_per_episode'])
            n = n_ep * steps
            if n < 1 or int(meta['frames']) != n or steps < 100:
                raise ValueError(f'{folder}/{split}: invalid episode lengths')
            if int(setup['steps_per_episode']) != steps or int(setup['action_repeat']) != 4:
                raise ValueError(f'{folder}/{split}: expected action_repeat=4, matched steps')
            if float(meta['env_size_m']) != 2.5:
                raise ValueError(f'{folder}/{split}: expected 2.5m arena')
            if n_ep != (int(setup['episodes'])-int(setup['val_episodes']) if split == 'train'
                        else int(setup['val_episodes'])):
                raise ValueError(f'{folder}/{split}: episode count mismatch')
            arrays = {k: np.load(folder / f'{split}_{k}.npy', mmap_mode='r')
                      for k in self.KEYS}
            shapes = {'images': (n, 3, 64, 64), 'pos': (n, 2),
                      'hd': (n,), 'vel_trans': (n, 3),
                      'vel_rot': (n, 3), 'episode_id': (n,)}
            for k, shape in shapes.items():
                if arrays[k].shape != shape:
                    raise ValueError(f'{folder}/{split}_{k} has shape '
                                     f'{arrays[k].shape}, expected {shape}')
            if arrays['images'].dtype != np.uint8:
                raise ValueError(f'{folder}: images must be uint8')
            for ep in (0, n_ep // 2, n_ep - 1):
                if int(arrays['episode_id'][ep * steps]) != ep or \
                   int(arrays['episode_id'][(ep + 1) * steps - 1]) != ep:
                    raise ValueError(f'{folder}: episode_id alignment error')
            episode_seeds = {int(meta['seed']) + i for i in range(n_ep)}
            if episode_seeds & all_seeds:
                raise ValueError('Overlapping episode seeds between datasets')
            all_seeds.update(episode_seeds)
            i = len(self.entries)
            self.entries.append({'folder': str(folder), 'steps': steps,
                                 'episodes': n_ep, 'arrays': arrays,
                                 'seed': int(meta['seed'])})
            self.episode_lookup.extend((i, ep) for ep in range(n_ep))
        self.seeds = all_seeds
        self.n_episodes = len(self.episode_lookup)
        self.n_frames = sum(e['episodes'] * e['steps'] for e in self.entries)

    def sample(self, rng: np.random.Generator, batch: int, length: int):
        # Slice starts are independently sampled INSIDE each single episode.
        images = np.empty((batch, length, 3, 64, 64), dtype=np.uint8)
        pos = np.empty((batch, length, 2), dtype=np.float32)
        hd = np.empty((batch, length), dtype=np.float32)
        vt = np.empty((batch, length, 3), dtype=np.float32)
        vr = np.empty((batch, length, 3), dtype=np.float32)
        for j in range(batch):
            eidx, ep = self.episode_lookup[int(rng.integers(self.n_episodes))]
            src = self.entries[eidx]
            n = src['steps']
            if length > n:
                raise ValueError(f'--seq-len={length} exceeds episode steps={n}')
            t0 = int(rng.integers(n - length + 1))
            s = slice(ep * n + t0, ep * n + t0 + length)
            a = src['arrays']
            images[j] = a['images'][s]
            pos[j] = a['pos'][s]
            hd[j] = a['hd'][s]
            vt[j] = a['vel_trans'][s]
            vr[j] = a['vel_rot'][s]
        return images, pos, hd, vt, vr


class GridLSTM(nn.Module):
    def __init__(self, hidden: int = 128, bottleneck: int = 512):
        super().__init__()
        self.hidden_size = hidden
        self.lstm = nn.LSTM(input_size=272, hidden_size=hidden, batch_first=True)
        self.bottleneck = nn.Linear(hidden, bottleneck, bias=True)
        self.pc_head = nn.Linear(bottleneck, 256)
        self.hd_head = nn.Linear(bottleneck, 12)
        self.apply(init_grid_weights)

    def forward(self, movement_and_visual: torch.Tensor):
        b = movement_and_visual.size(0)
        # PyTorch accepts (hidden, cell); notation: m_t = hidden, l_t = cell.
        m0 = movement_and_visual.new_zeros(1, b, self.hidden_size)
        l0 = movement_and_visual.new_zeros(1, b, self.hidden_size)
        out, _ = self.lstm(movement_and_visual, (m0, l0))
        g = self.bottleneck(out)  # NO intermediate nonlinear activation.
        z = F.dropout(g, p=0.5, training=self.training)
        return self.pc_head(z), self.hd_head(z), g


def init_grid_weights(m: nn.Module) -> None:
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            fan_in = m.weight.shape[1]
            sigma = fan_in ** -0.5
            nn.init.trunc_normal_(m.weight, mean=0., std=sigma,
                                  a=-2 * sigma, b=2 * sigma)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, p in m.named_parameters(recurse=False):
                if name.startswith('weight_'):
                    sigma = p.shape[1] ** -0.5
                    nn.init.trunc_normal_(p, mean=0., std=sigma,
                                          a=-2 * sigma, b=2 * sigma)
                elif name.startswith('bias_'):
                    nn.init.zeros_(p)
            # Forget gate ordering in PyTorch: [input, forget, candidate, output].
            h = m.hidden_size
            m.bias_hh_l0[h:2 * h].fill_(1.)


class VisionCorrectedGrid(nn.Module):
    def __init__(self, pc, hd, frozen_vision_path: Path, device: torch.device):
        super().__init__()
        self.vision = VisionModule(pc, hd, embed_dim=256,
                                   mask_prob=.95, mask_mode='whole')
        ck = torch.load(frozen_vision_path, map_location='cpu', weights_only=False)
        if 'vision_module' not in ck:
            raise ValueError('Not a vision module checkpoint')
        if not np.isclose(float(ck['env_size_m']), 2.5):
            raise ValueError('Vision checkpoint environment is not 2.5m')
        if 'pc_centers' not in ck or 'hd_centers' not in ck:
            raise ValueError('Vision checkpoint lacks cell centers; cannot ensure matching labels')
        if not torch.allclose(ck['pc_centers'].cpu(), pc.means.cpu(), atol=1e-6):
            raise ValueError('PC centers do NOT match pretrained vision module')
        if not torch.allclose(ck['hd_centers'].cpu(), hd.means.cpu(), atol=1e-6):
            raise ValueError('HD centers do NOT match pretrained vision module')
        if not np.allclose(ck['pc_stdev'], [0.4], atol=1e-8):
            raise ValueError('Place-cell scale mismatch: expected RL PC stdev 0.4m')
        self.vision.load_state_dict(ck['vision_module'], strict=True)
        self.grid = GridLSTM(hidden=128, bottleneck=512)
        self.to(device)
        for p in self.vision.parameters():
            p.requires_grad_(False)
        self.vision.eval()

    def forward(self, images: np.ndarray, vel: torch.Tensor,
                *, vision_keep_prob: float, generator: torch.Generator | None = None):
        b, t = images.shape[:2]
        device = vel.device
        keep = (torch.rand((b, t), device=device, generator=generator)
                < vision_keep_prob).reshape(-1)
        visuals = vel.new_zeros((b * t, 268))
        idx = torch.nonzero(keep, as_tuple=False).flatten()
        # Use CNN only where vision is transmitted to the Grid LSTM.
        if idx.numel():
            idx_np = idx.cpu().numpy()
            rgb = np.ascontiguousarray(images.reshape(-1, 3, 64, 64)[idx_np])
            self.vision.eval()
            with torch.no_grad():
                # Chunk to avoid transient CUDA spikes even with large batches.
                predictions = []
                for start in range(0, len(rgb), 128):
                    x = torch.from_numpy(rgb[start:start+128]).to(device)
                    x = x.float().div_(127.5).sub_(1.)
                    z = self.vision.cnn(x)
                    predictions.append(torch.cat((self.vision.place_head(z).softmax(-1),
                                                   self.vision.hd_head(z).softmax(-1)), dim=-1))
            visuals.index_copy_(0, idx, torch.cat(predictions, dim=0))
        inp = torch.cat((vel, visuals.reshape(b, t, 268)), dim=-1)
        if inp.shape[-1] != 272:
            raise RuntimeError('Input must have 272 channels')
        p, h, g = self.grid(inp)
        return p, h, g, int(keep.sum().item())


def make_batch(source: Episodes, rng: np.random.Generator, batch: int,
               seq_len: int, pc, hd, cfg, device: torch.device):
    images, pos_np, head_np, vt_np, vr_np = source.sample(rng, batch, seq_len)
    pos = torch.from_numpy(pos_np).to(device)
    head = torch.from_numpy(head_np).to(device)
    with torch.no_grad():
        target_pc = pc.posterior(pos)
        target_hd = hd.posterior(head.unsqueeze(-1))  # (B,T,1) -> HD12
        vt = torch.from_numpy(vt_np).to(device).reshape(-1, 3)
        vr = torch.from_numpy(vr_np).to(device).reshape(-1, 3)
        movement = ego_velocity(vt, vr, cfg.rl.env).reshape(batch, seq_len, 4)
    return images, movement, target_pc, target_hd, pos, head


def calculate_loss(p, h, target_pc, target_hd):
    return (soft_cross_entropy(p, target_pc) + soft_cross_entropy(h, target_hd)).mean()


@torch.no_grad()
def evaluate(model, source, pc, hd, cfg, device, args):
    model.grid.eval()
    rng = np.random.default_rng(args.seed + 909012)
    rand_gen = torch.Generator(device=device)
    rand_gen.manual_seed(args.seed + 939911)
    total_loss = 0.
    pos_err = 0.
    hd_err = 0.
    n = 0
    for _ in range(args.val_batches):
        images, vels, pc_t, hd_t, pos, heading = make_batch(
            source, rng, args.batch_size, args.seq_len, pc, hd, cfg, device)
        p, h, _, n_seen = model(images, vels, vision_keep_prob=args.vision_prob,
                                generator=rand_gen)
        loss = calculate_loss(p, h, pc_t, hd_t)
        pp, hp = p.softmax(-1), h.softmax(-1)
        estimated_pos = pp @ pc.means
        estimated_dir = torch.atan2(hp @ torch.sin(hd.means),
                                    hp @ torch.cos(hd.means))
        a_err = torch.atan2(torch.sin(estimated_dir - heading),
                            torch.cos(estimated_dir - heading)).abs()
        k = args.batch_size * args.seq_len
        total_loss += float(loss) * k
        pos_err += float((estimated_pos - pos).norm(dim=-1).sum()) * 100.
        hd_err += float(torch.rad2deg(a_err).sum())
        n += k
    model.grid.train()
    return total_loss / n, pos_err / n, hd_err / n


def atomic_save(obj, path: Path):
    tmp = path.with_suffix('.tmp.pt')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data-dirs', nargs='+', type=Path, required=True)
    ap.add_argument('--vision-checkpoint', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path,
                    default=Path('data/checkpoints/grid_vision_272_run1'))
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--updates', type=int, default=10000)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--seq-len', type=int, default=100)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--val-batches', type=int, default=8)
    ap.add_argument('--save-every', type=int, default=1000)
    ap.add_argument('--vision-prob', type=float, default=0.05)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = ap.parse_args()
    for name in ('updates', 'batch_size', 'seq_len', 'eval_every', 'val_batches', 'save_every'):
        if getattr(args, name) <= 0:
            ap.error(f'{name} must be >0')
    if not 0. < args.vision_prob <= 1.:
        ap.error('--vision-prob must be in (0,1]')
    if args.lr <= 0 or args.weight_decay < 0:
        ap.error('Bad optimizer settings')
    dirs = [p.expanduser().resolve() for p in args.data_dirs]
    if len(set(dirs)) != len(dirs):
        ap.error('Duplicate --data-dirs')
    vision_ckpt = args.vision_checkpoint.expanduser().resolve()
    if not vision_ckpt.is_file():
        raise FileNotFoundError(vision_ckpt)
    train = Episodes(dirs, 'train')
    val = Episodes(dirs, 'val')
    if train.seeds & val.seeds:
        raise ValueError('Train/val DMLab episode seeds overlap')
    if args.seq_len > min(x['steps'] for x in train.entries + val.entries):
        ap.error('seq length exceeds at least one dataset episode')
    print(f'EPISODES | train={train.n_episodes:,}, val={val.n_episodes:,} | '
          f'frames={train.n_frames + val.n_frames:,} | seq={args.seq_len}', flush=True)
    dev = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                       if args.device == 'auto' else args.device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    cfg = Config()
    if cfg.rl.env.action_repeat != 4 or cfg.rl.env.env_size_m != 2.5:
        raise ValueError('Config RL environment no longer matches collected trajectories')
    set_seed(args.seed)
    pcs, hds = build_rl_ensembles(cfg, dev)
    if len(pcs) != 1 or len(hds) != 1 or pcs[0].n_cells != 256 or hds[0].n_cells != 12:
        raise ValueError('Expected PC256 and HD12 for 272-D input')
    pc, hd = pcs[0], hds[0]
    model = VisionCorrectedGrid(pc, hd, vision_ckpt, dev)
    if args.check_only:
        print('CHECK OK: aligned DMLab episodes and pretrained Vision Module; '
              'PC/HD centers match. No weights changed.', flush=True)
        print('Grid: [u, v, sin(dyaw), cos(dyaw)] + PC256+HD12 = 272 -> '
              'LSTM128 -> bottleneck512 -> soft PC/HD CE')
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        'data_dirs': [str(x) for x in dirs], 'train_frames': train.n_frames,
        'val_frames': val.n_frames, 'vision_checkpoint_sha256': file_hash(vision_ckpt),
        'seed': args.seed, 'batch_size': args.batch_size, 'seq_len': args.seq_len,
        'lr': args.lr, 'weight_decay': args.weight_decay, 'vision_prob': args.vision_prob,
    }
    decay = [model.grid.pc_head.weight, model.grid.hd_head.weight]
    decay_ids = {id(p) for p in decay}
    others = [p for p in model.grid.parameters() if id(p) not in decay_ids]
    opt = torch.optim.RMSprop([{'params': others},
                               {'params': decay, 'weight_decay': args.weight_decay}],
                              lr=args.lr, alpha=.99, eps=1e-8, momentum=0.0)
    latest = args.out_dir / 'latest.pt'
    log = args.out_dir / 'metrics.csv'
    rng = np.random.default_rng(args.seed + 77)
    step = 0
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError(f'Resume requires {latest}')
        checkpoint = torch.load(latest, map_location=dev, weights_only=False)
        if checkpoint['fingerprint'] != fingerprint:
            raise ValueError('Resume data / hyperparameters / pretrained vision differ')
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer'])
        step = int(checkpoint['step'])
        rng.bit_generator.state = checkpoint['numpy_rng_state']
        torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
        if dev.type == 'cuda' and 'cuda_rng_states' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_states'])
        print(f'RESUMED at update {step:,}', flush=True)
    elif latest.exists() or log.exists():
        raise FileExistsError(f'{args.out_dir} already contains training. '
                              'Use --resume or a new --out-dir')

    def save(k: int):
        obj = {
            'experiment_version': 'frozen_vision_grid_supervised_272_v1',
            'step': k, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
            'fingerprint': fingerprint, 'pc_centers': pc.means.detach().cpu(),
            'hd_centers': hd.means.detach().cpu(),
            'grid_hidden': 128, 'grid_bottleneck': 512, 'grid_input': 272,
            'init': 'truncated_normal_fan_in_2sigma', 'vision_keep_prob': args.vision_prob,
            'numpy_rng_state': rng.bit_generator.state,
            'torch_rng_state': torch.get_rng_state().cpu(),
        }
        if dev.type == 'cuda':
            obj['cuda_rng_states'] = torch.cuda.get_rng_state_all()
        atomic_save(obj, latest)
        if k % args.save_every == 0 or k == args.updates:
            atomic_save(obj, args.out_dir / f'grid_step{k:07d}.pt')

    fields = ['update', 'train_ce', 'val_ce', 'val_pos_cm', 'val_hd_deg',
              'vision_seen', 'vision_total']
    print(f'TRAIN | device={dev} updates={args.updates:,} | '
          f'vision={100*args.vision_prob:.1f}% | fresh grid init, CNN frozen', flush=True)
    with log.open('a' if args.resume else 'w', newline='', buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not args.resume:
            writer.writeheader()
        model.grid.train()
        if step >= args.updates:
            print(f'Already at step {step:,}; nothing to train')
            return
        try:
            for step in range(step + 1, args.updates + 1):
                images, movement, pc_t, hd_t, _, _ = make_batch(
                    train, rng, args.batch_size, args.seq_len, pc, hd, cfg, dev)
                p, h, _, seen = model(images, movement,
                                      vision_keep_prob=args.vision_prob)
                loss = calculate_loss(p, h, pc_t, hd_t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss')
                opt.step()
                if step % args.eval_every == 0 or step == args.updates:
                    va_loss, va_pos, va_hd = evaluate(model, val, pc, hd, cfg, dev, args)
                    writer.writerow({'update': step, 'train_ce': float(loss.detach()),
                                     'val_ce': va_loss, 'val_pos_cm': va_pos,
                                     'val_hd_deg': va_hd, 'vision_seen': seen,
                                     'vision_total': args.batch_size * args.seq_len})
                    print(f'update {step:,}/{args.updates:,} | '
                          f'train/val CE {float(loss.detach()):.4f}/{va_loss:.4f} '
                          f'| val pos {va_pos:.1f} cm HD {va_hd:.1f} deg '
                          f'| visual cues {seen}/{args.batch_size*args.seq_len}', flush=True)
                if step % args.save_every == 0 or step == args.updates:
                    save(step)
        except KeyboardInterrupt:
            print(f'Keyboard interrupt; saving at update {step:,}', flush=True)
            save(step)
    print(f'FINISHED | checkpoint: {latest}', flush=True)


if __name__ == '__main__':
    main()
