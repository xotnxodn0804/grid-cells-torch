#!/usr/bin/env python3
"""Train a 256-bottleneck vision-corrected Grid LSTM with noisy velocity inputs.

Independent experiment from measured_yaw 100k. Original dataset is unchanged.
Training and evaluation both add independent N(0,0.01**2) perturbations to
DMLab VEL.TRANS x/y (in m/s) and to measured angular velocity (in rad/s).
Angular rate is derived from stored ground-truth successive headings, NOT
DMLab VEL.ROT. Hence this is *noisy oracle odometry*, NOT an exact replication
of Banino et al. 2018 despite matching the requested Gaussian parameters.

Default --angular-encoding increment preserves the previous measured_yaw
model: [u,v,sin(delta_heading),cos(delta_heading)], with noise added to
angular velocity in rad/s *before* multiplying by action interval.
Optional --angular-encoding rate produces [u,v,sin(omega),cos(omega)]
with omega in rad/s, closer to the literal angular-velocity formula in
the paper but also changes the encoding (a separate experimental factor).

Fresh checkpoint directory required: cannot --resume from measured_yaw 100k,
because input definitions differ. Frozen vision checkpoint remains reusable.
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
    KEYS = ('images', 'pos', 'hd', 'vel_trans', 'vel_rot', 'ego_vel', 'valid_motion', 'episode_id')

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
                      'vel_rot': (n, 3), 'ego_vel': (n, 3), 'valid_motion': (n,), 'episode_id': (n,)}
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
        measured_ego = np.empty((batch, length, 3), dtype=np.float32)
        for j in range(batch):
            # Do not bridge a teleport/reset using continuous-motion odometry.
            for attempt in range(200):
                eidx, ep = self.episode_lookup[int(rng.integers(self.n_episodes))]
                src = self.entries[eidx]
                n = src['steps']
                if length > n:
                    raise ValueError(f'--seq-len={length} exceeds episode steps={n}')
                t0 = int(rng.integers(n - length + 1))
                s = slice(ep * n + t0, ep * n + t0 + length)
                a = src['arrays']
                if np.all(a['valid_motion'][s] == 1):
                    break
            else:
                raise RuntimeError('Unable to sample a motion-valid sequence; inspect valid_motion')
            images[j] = a['images'][s]
            pos[j] = a['pos'][s]
            hd[j] = a['hd'][s]
            vt[j] = a['vel_trans'][s]
            vr[j] = a['vel_rot'][s]
            measured_ego[j] = a['ego_vel'][s]
        return images, pos, hd, vt, vr, measured_ego


class GridLSTM(nn.Module):
    def __init__(self, hidden: int = 128, bottleneck: int = 256):
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
        self.grid = GridLSTM(hidden=128, bottleneck=256)
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


def noisy_motion(vt_np: np.ndarray, measured_ego_np: np.ndarray, rng: np.random.Generator,
                 scale: float, dt: float, speed_noise_std: float, angular_noise_std: float,
                 angular_encoding: str) -> np.ndarray:
    """Return [u,v,sin/cos(omega or omega*dt)] with noises in *physical units*.

    VEL.TRANS columns 0/1 are in DMLab world units/s; scale converts to m/s.
    Stored ego_vel = [speed, sin(delta_yaw), cos(delta_y)] from the collector.
    At ~4 frames/action, delta_yaw / dt estimates the average angular rate.
    Uses numpy RNG for train/checkpoint/validation reproducibility.
    """
    if dt <= 0 or scale <= 0 or speed_noise_std < 0 or angular_noise_std < 0:
        raise ValueError('Invalid scale/dt/noise std')
    if angular_encoding not in ('rate', 'increment'):
        raise ValueError('angular_encoding must be rate or increment')
    if vt_np.shape[:-1] != measured_ego_np.shape[:-1] or vt_np.shape[-1] != 3 \
            or measured_ego_np.shape[-1] != 3:
        raise ValueError('Unexpected velocity shapes')
    # Translational velocity components in meters/second.
    uv = (vt_np[..., :2].astype(np.float64) * scale)
    uv += rng.normal(0.0, speed_noise_std, size=uv.shape)
    # Wrapped true yaw increment and its action-block average rate (rad/s).
    dyaw = np.arctan2(measured_ego_np[..., 1], measured_ego_np[..., 2])
    omega = dyaw.astype(np.float64) / dt
    omega += rng.normal(0.0, angular_noise_std, size=omega.shape)
    arg = omega if angular_encoding == 'rate' else omega * dt
    movement = np.concatenate((uv, np.sin(arg)[..., None], np.cos(arg)[..., None]), axis=-1)
    if not np.isfinite(movement).all():
        raise ValueError('Non-finite noisy input')
    return movement.astype(np.float32)


def make_batch(source: Episodes, rng: np.random.Generator, batch: int,
               seq_len: int, pc, hd, cfg, device: torch.device, *,
               noise_rng: np.random.Generator, speed_noise_std: float, angular_noise_std: float,
               angular_encoding: str):
    images, pos_np, head_np, vt_np, vr_np, measured_ego_np = source.sample(
        rng, batch, seq_len)
    pos = torch.from_numpy(pos_np).to(device)
    head = torch.from_numpy(head_np).to(device)
    with torch.no_grad():
        target_pc = pc.posterior(pos)
        target_hd = hd.posterior(head.unsqueeze(-1))
        scale = cfg.rl.env.env_size_m / (
            cfg.rl.env.wall_world_units[1] - cfg.rl.env.wall_world_units[0])
        dt = cfg.rl.env.action_repeat / 60.0
        movement_np = noisy_motion(vt_np, measured_ego_np, noise_rng, scale, dt,
                                   speed_noise_std, angular_noise_std, angular_encoding)
        movement = torch.from_numpy(movement_np).to(device)
        if not torch.allclose((movement[..., 2:]**2).sum(-1),
                              torch.ones_like(movement[..., 2]), atol=1e-4):
            raise ValueError('Invalid noisy sin/cos angular components')
    return images, movement, target_pc, target_hd, pos, head


def calculate_loss(p, h, target_pc, target_hd):
    return (soft_cross_entropy(p, target_pc) + soft_cross_entropy(h, target_hd)).mean()


@torch.no_grad()
def evaluate(model, source, pc, hd, cfg, device, args):
    model.grid.eval()
    rng = np.random.default_rng(args.seed + 909012)
    noise_rng = np.random.default_rng(args.seed + 909013)
    rand_gen = torch.Generator(device=device)
    rand_gen.manual_seed(args.seed + 939911)
    total_loss = 0.
    pos_err = 0.
    hd_err = 0.
    n = 0
    for _ in range(args.val_batches):
        images, vels, pc_t, hd_t, pos, heading = make_batch(
            source, rng, args.batch_size, args.seq_len, pc, hd, cfg, device,
            noise_rng=noise_rng, speed_noise_std=args.speed_noise_std,
            angular_noise_std=args.angular_noise_std, angular_encoding=args.angular_encoding)
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
                    default=Path('data/checkpoints/grid_vision_256_noisy_motion_run1'))
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
    ap.add_argument('--speed-noise-std', type=float, default=0.01,
                    help='Per-component Gaussian sigma in m/s, applied to DMLab VEL.TRANS xy')
    ap.add_argument('--angular-noise-std', type=float, default=0.01,
                    help='Gaussian sigma in rad/s, applied to measured yaw rate')
    ap.add_argument('--angular-encoding', choices=['rate', 'increment'], default='increment',
                    help='increment: preserve prior model and add noise only; rate: sin/cos(omega in rad/s)')
    ap.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = ap.parse_args()
    for name in ('updates', 'batch_size', 'seq_len', 'eval_every', 'val_batches', 'save_every'):
        if getattr(args, name) <= 0:
            ap.error(f'{name} must be >0')
    if not 0. < args.vision_prob <= 1.:
        ap.error('--vision-prob must be in (0,1]')
    if not np.isfinite(args.speed_noise_std) or not np.isfinite(args.angular_noise_std) \
            or args.speed_noise_std < 0 or args.angular_noise_std < 0:
        ap.error('Noise std must be finite and >= 0')
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
        trial_rng = np.random.default_rng(args.seed + 12345)
        mock_vt = np.zeros((2, 10, 3), np.float32)
        mock_ego = np.zeros((2, 10, 3), np.float32)
        mock_ego[..., 2] = 1.0
        probe = noisy_motion(mock_vt, mock_ego, trial_rng, .0025, 4 / 60,
                             args.speed_noise_std, args.angular_noise_std,
                             args.angular_encoding)
        assert probe.shape == (2, 10, 4)
        assert np.allclose(probe[..., 2]**2 + probe[..., 3]**2, 1, atol=1e-6)
        print(f'MOTION OK: [u,v] m/s + noise sigma={args.speed_noise_std} m/s; '
              f'{args.angular_encoding} angular encoding with omega noise sigma='
              f'{args.angular_noise_std} rad/s. No weights changed.')
        print('WARNING: yaw source is measured GT displacement; noisy oracle odometry, not DMLab VEL.ROT')
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        'data_dirs': [str(x) for x in dirs], 'train_frames': train.n_frames,
        'val_frames': val.n_frames, 'vision_checkpoint_sha256': file_hash(vision_ckpt),
        'seed': args.seed, 'batch_size': args.batch_size, 'seq_len': args.seq_len,
        'grid_bottleneck': 256,
        'motion_source': 'collector_measured_yaw_v2_valid_only_plus_gaussian_noise',
        'speed_noise_std_m_per_s': args.speed_noise_std,
        'angular_noise_std_rad_per_s': args.angular_noise_std,
        'angular_encoding': args.angular_encoding,
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
    noise_rng = np.random.default_rng(args.seed + 78)
    step = 0
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError(f'Resume requires {latest}')
        checkpoint = torch.load(latest, map_location=dev, weights_only=False)
        if checkpoint['fingerprint'] != fingerprint:
            raise ValueError('Resume data / hyperparameters / pretrained vision differ')
        if checkpoint.get('grid_bottleneck') != 256:
            raise ValueError('Cannot resume a model with a different bottleneck size')
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer'])
        step = int(checkpoint['step'])
        rng.bit_generator.state = checkpoint['numpy_rng_state']
        noise_rng.bit_generator.state = checkpoint['noise_rng_state']
        torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
        if dev.type == 'cuda' and 'cuda_rng_states' in checkpoint:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng_states']])
        print(f'RESUMED at update {step:,}', flush=True)
    elif latest.exists() or log.exists():
        raise FileExistsError(f'{args.out_dir} already contains training. '
                              'Use --resume or a new --out-dir')

    def save(k: int):
        obj = {
            'experiment_version': 'frozen_vision_grid_supervised_272_256_noisy_motion_v1',
            'step': k, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
            'fingerprint': fingerprint, 'pc_centers': pc.means.detach().cpu(),
            'hd_centers': hd.means.detach().cpu(),
            'grid_hidden': 128, 'grid_bottleneck': 256, 'grid_input': 272,
            'motion_source': 'collector_measured_yaw_v2_valid_only_plus_gaussian_noise',
            'speed_noise_std_m_per_s': args.speed_noise_std,
            'angular_noise_std_rad_per_s': args.angular_noise_std,
            'angular_encoding': args.angular_encoding,
            'init': 'truncated_normal_fan_in_2sigma', 'vision_keep_prob': args.vision_prob,
            'numpy_rng_state': rng.bit_generator.state,
            'noise_rng_state': noise_rng.bit_generator.state,
            'torch_rng_state': torch.get_rng_state().cpu(),
        }
        if dev.type == 'cuda':
            obj['cuda_rng_states'] = torch.cuda.get_rng_state_all()
        atomic_save(obj, latest)
        if k % args.save_every == 0 or k == args.updates:
            atomic_save(obj, args.out_dir / f'grid_step{k:07d}.pt')

    fields = ['update', 'train_ce', 'val_ce', 'val_pos_cm', 'val_hd_deg',
              'vision_seen', 'vision_total']
    print(f'MOTION | measured yaw rate (noisy oracle); sigma(speed)={args.speed_noise_std} m/s, '
          f'sigma(angular)={args.angular_noise_std} rad/s, encoding={args.angular_encoding}; '
          'exclude teleport windows', flush=True)
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
                    train, rng, args.batch_size, args.seq_len, pc, hd, cfg, dev,
                    noise_rng=noise_rng, speed_noise_std=args.speed_noise_std,
                    angular_noise_std=args.angular_noise_std, angular_encoding=args.angular_encoding)
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
