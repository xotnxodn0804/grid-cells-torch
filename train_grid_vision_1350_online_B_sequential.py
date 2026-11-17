#!/usr/bin/env python3
"""B-Online SEQUENTIAL + full-history raw replay (user-requested 1 Vision path).

One NEW DMLab episode (1350 steps) is collected at a time. For each successive
10-step observation window in that episode: generate sparse 268-d Grid visual
inputs with the CURRENT Vision model, add only those 10 observed frames to an
UNBOUNDED disk replay, then train Vision once from 32 uniformly sampled frames
among ALL observations seen so far. Vision weights and replay persist across
episodes. Future episodes and future frames are never Vision training candidates.
After completing ten sequential episodes, train Grid with 10 path sequences in
14 TBPTT updates (13x100 and final 50 weighted by 0.5). The stored Grid visual
inputs correspond to the Vision weights at their historical observation times.

Vision stops at 300,000 updates. Past replay is kept on disk until *the user*
deletes it. Grid continues to 300,006 updates (21,429 rollouts). RGB raw uint8
is stored in one append-only .bin file (34.33 GiB for 3M frames) with a separate
float32 x/y/heading .bin file, NOT millions of PNGs. Each episode's RGB is held
in RAM temporarily; Grid stores ONLY motion, labels, masks, and Vision outputs.

This is chronologically causal *supervised* training, not online RL navigation.
The episode is simulated to completion before its 10-step windows are admitted
to Vision; code never trains Vision on future data before the appropriate window.
Relies on train_grid_vision_1350_online_A.py and existing local project modules.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from grid_cells.ensembles import build_rl_ensembles
from grid_cells.model import GridCellsRNN
from grid_cells.vision import VisionModule
from grid_cells.rl_common import make_dmlab_env
from sonnet_init_override import initialize_sonnet_lstm_

import train_grid_vision_1350_online_A as online_a
import train_grid_vision_1350_causal_ab as old_b
import train_grid_dmlab_vision_pc001_272_noisy as base
import collect_vision_rh_1000_v2 as collector
import collect_rh_1350_v3 as rh1350

STEPS = online_a.STEPS
BATCH = online_a.BATCH
TBPTT = online_a.TBPTT
SIGMA = online_a.SIGMA
VISION_DIMS = online_a.VISION_DIMS
MOTION_DIMS = online_a.MOTION_DIMS
ACTION_REPEAT = online_a.ACTION_REPEAT
FPS = online_a.FPS
VAL_SEED_OFFSET = online_a.VAL_SEED_OFFSET
TRAIN_SEED_DEFAULT = online_a.TRAIN_SEED_DEFAULT
assert STEPS == 1350 and TBPTT == 100 and BATCH == 10
base.STEPS = STEPS


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class DiskHistoryReplay:
    """All-history causal raw RGB replay, exact equal-probability sampling.

    images.bin is a concatenation of uint8 (3,64,64) frames, fixed stride 12288.
    labels.bin is a concatenation of little-endian float32 (x,y,heading), 12B.
    Both can be truncated exactly to the last atomically saved checkpoint.
    No eviction, no lossy compression, no per-image file overhead.
    """
    FRAME_BYTES = 3 * 64 * 64
    INDEX_STRUCT = struct.Struct('<fff')

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.image_path = root / 'images.bin'
        self.index_path = root / 'labels.bin'
        self._image_f = self.image_path.open('r+b' if self.image_path.exists() else 'w+b')
        self._index_f = self.index_path.open('r+b' if self.index_path.exists() else 'w+b')
        self.total_seen = self.size = self.image_bytes = 0

    def add_window(self, store: online_a.InMemoryEpisodes, begin: int, end: int):
        if not (0 <= begin < end <= STEPS):
            raise ValueError('Noncausal or empty temporal window')
        arrays = store._arrays
        # For sequential Vision, exactly ONE active episode is supplied.
        if arrays['images'].shape[0] != 1:
            raise AssertionError('Vision must collect only ONE episode at a time')
        imgs = np.ascontiguousarray(arrays['images'][0, begin:end], dtype=np.uint8)
        positions = arrays['pos'][0, begin:end]
        headings = arrays['hd'][0, begin:end]
        if imgs.shape != (end-begin, 3, 64, 64):
            raise ValueError(f'Unexpected RGB shape: {imgs.shape}')
        labels = np.empty((end-begin, 3), dtype='<f4')
        labels[:, :2] = positions
        labels[:, 2] = headings
        self._image_f.seek(self.image_bytes)
        self._image_f.write(imgs.tobytes(order='C'))
        self._index_f.seek(self.total_seen * self.INDEX_STRUCT.size)
        self._index_f.write(labels.tobytes(order='C'))
        self.total_seen += len(imgs)
        self.size = self.total_seen
        self.image_bytes += len(imgs) * self.FRAME_BYTES
        # Flush Python's buffered writes before any subsequent random reads.
        self._image_f.flush()
        self._index_f.flush()

    def sample(self, count: int, rng: np.random.Generator):
        if not self.total_seen:
            raise RuntimeError('Replay is empty')
        ids = rng.integers(0, self.total_seen, size=count, dtype=np.int64)
        rgb = np.empty((count, 3, 64, 64), dtype=np.uint8)
        pos = np.empty((count, 2), dtype=np.float32)
        hd = np.empty((count,), dtype=np.float32)
        for row, idx in enumerate(ids):
            self._image_f.seek(int(idx) * self.FRAME_BYTES)
            payload = self._image_f.read(self.FRAME_BYTES)
            if len(payload) != self.FRAME_BYTES:
                raise IOError(f'Missing RGB frame index {idx}')
            rgb[row] = np.frombuffer(payload, dtype=np.uint8).reshape(3,64,64)
            self._index_f.seek(int(idx) * self.INDEX_STRUCT.size)
            meta = self._index_f.read(self.INDEX_STRUCT.size)
            if len(meta) != self.INDEX_STRUCT.size:
                raise IOError(f'Missing position/heading for frame index {idx}')
            x, y, theta = self.INDEX_STRUCT.unpack(meta)
            pos[row] = (x, y)
            hd[row] = theta
        return rgb, pos, hd

    def sync(self):
        for f in (self._image_f, self._index_f):
            f.flush()
            os.fsync(f.fileno())

    def state_dict(self):
        return {'format': 'all_history_raw_rgb_v1', 'total_seen': self.total_seen,
                'image_bytes': self.image_bytes}

    def load_state_dict(self, state):
        if state.get('format') != 'all_history_raw_rgb_v1':
            raise ValueError('Raw replay format mismatch')
        frames, image_bytes = int(state['total_seen']), int(state['image_bytes'])
        if frames < 0 or image_bytes != frames * self.FRAME_BYTES:
            raise ValueError('Corrupted replay checkpoint counters')
        self._image_f.seek(0, 2)
        self._index_f.seek(0, 2)
        if (self._image_f.tell() < image_bytes
                or self._index_f.tell() < frames * self.INDEX_STRUCT.size):
            raise IOError('Replay files are shorter than checkpoint. Cannot resume.')
        self._image_f.truncate(image_bytes)
        self._index_f.truncate(frames * self.INDEX_STRUCT.size)
        self.total_seen = self.size = frames
        self.image_bytes = image_bytes
        self.sync()

    def close(self):
        self._image_f.close()
        self._index_f.close()


def free_space_gib(path: Path) -> float:
    """Guard against both WSL virtual size and actual Windows C: storage."""
    candidates = [shutil.disk_usage(path).free]
    if Path('/mnt/c').is_dir():
        candidates.append(shutil.disk_usage('/mnt/c').free)
    return min(candidates) / 2**30


def vision_update(vision, optimizer, pc, hdc, replay: DiskHistoryReplay,
                  rng: np.random.Generator, device: torch.device) -> float:
    rgb, pos, hd = replay.sample(32, rng)
    img, positions, headings = old_b.to_gpu(rgb, pos, hd, device)
    with torch.no_grad():
        targets_pc = pc.posterior(positions.unsqueeze(1)).squeeze(1)
        targets_hd = hdc.posterior(headings.unsqueeze(1)).squeeze(1)
    vision.train()
    optimizer.zero_grad(set_to_none=True)
    loss = vision.loss(img, targets_pc, targets_hd)
    if not torch.isfinite(loss):
        raise FloatingPointError('Vision loss became nonfinite')
    loss.backward()
    optimizer.step()
    vision.eval()
    return float(loss.detach())


@torch.no_grad()
def validate_b(grid, vision, pc, hdc, env, cfg, a, yaw_gain,
               nominal_speed, device: torch.device):
    """Held-out fixed-seed episodes, regenerated only at evaluation time.

    Return Grid errors and Vision metrics from the SAME held-out episodes.
    """
    grid.eval()
    vision.eval()
    sums = np.zeros(5, np.float64)
    seen = 0
    vsum = np.zeros(3, np.float64)
    vn = 0
    rng = np.random.default_rng(a.seed + 129731)
    sample_eids = rng.integers(0, a.val_episodes, size=256, dtype=np.int64)
    sample_ts = rng.integers(0, STEPS, size=256, dtype=np.int64)
    val_seed_base = a.train_seed + VAL_SEED_OFFSET
    try:
        for first in range(0, a.val_episodes, a.val_batch_size):
            ids = np.arange(first, min(first + a.val_batch_size, a.val_episodes), dtype=np.int64)
            store, _ = online_a.generate_episodes(
                env, cfg, ids, val_seed_base, yaw_gain, nominal_speed)
            idx = (sample_eids >= first) & (sample_eids < first + len(ids))
            if np.any(idx):
                stats = old_b.vision_eval(
                    vision, pc, hdc, store, sample_eids[idx], sample_ts[idx], device)
                count = int(np.count_nonzero(idx))
                vsum += np.asarray(stats) * count
                vn += count
            roll = base.build_rollout(
                store, ids, vision, device,
                np.random.default_rng(0), np.random.default_rng(0),
                a.vision_rate, a.force_first_vision,
                a.speed_noise_std, a.angular_noise_std,
                deterministic_val=True, val_seed=a.seed + 83817)
            b = len(ids)
            hidden = (torch.zeros(1, b, 128, device=device),
                      torch.zeros(1, b, 128, device=device))
            for start in range(0, STEPS, TBPTT):
                end = min(start + TBPTT, STEPS)
                lp, lh, hidden = base.step_segment(
                    grid, roll.vel[:, start:end], roll.vis[:, start:end],
                    roll.reset[:, start:end], hidden, training=False)
                tpc, thd = base.targets(pc, hdc, roll.pos[:, start:end], roll.hd[:, start:end])
                ce = (-(tpc * F.log_softmax(lp, -1)).sum(-1)
                      -(thd * F.log_softmax(lh, -1)).sum(-1))
                err = base.prediction_errors(
                    pc, hdc, lp, lh, roll.pos[:, start:end], roll.hd[:, start:end])
                for j, values in enumerate(err):
                    sums[j] += values.sum().item()
                sums[4] += ce.sum().item()
                seen += b * (end - start)
            del store, roll
    finally:
        grid.train()
    if not vn or not seen:
        raise RuntimeError('Validation produced no observations')
    return dict(grid_ce=sums[4]/seen, grid_pos_weighted_cm=sums[1]/seen,
                grid_hd_deg=sums[3]/seen,
                vision_ce=vsum[0]/vn, vision_pos_cm=vsum[1]/vn,
                vision_hd_deg=vsum[2]/vn)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out-dir', type=Path,
                   default=Path('data/checkpoints/grid1350_online_B'))
    p.add_argument('--rollouts', type=int, default=21429)
    p.add_argument('--train-seed', type=int, default=TRAIN_SEED_DEFAULT)
    p.add_argument('--val-episodes', type=int, default=100)
    p.add_argument('--val-batch-size', type=int, default=8)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=25)
    p.add_argument('--vision-update-every', type=int, default=10)
    p.add_argument('--max-vision-updates', type=int, default=300000)
    p.add_argument('--min-free-gib', type=float, default=20.0,
                   help='Stop if host C: or WSL disk has less than this GiB free (plus one rollout)')
    p.add_argument('--vision-lr', type=float, default=1e-4)
    p.add_argument('--grid-lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--vision-rate', type=float, default=0.05)
    p.add_argument('--speed-noise-std', type=float, default=0.01)
    p.add_argument('--angular-noise-std', type=float, default=0.01)
    p.add_argument('--force-first-vision', action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument('--renderer', choices=('hardware', 'software'), default='hardware')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    for name in ('rollouts', 'val_episodes', 'val_batch_size', 'eval_every',
                 'save_every', 'vision_update_every', 'max_vision_updates'):
        if getattr(a, name) <= 0:
            p.error(f'{name} must be positive')
    if a.vision_update_every > TBPTT or TBPTT % a.vision_update_every or 50 % a.vision_update_every:
        p.error('--vision-update-every must divide both 100 and 50')
    if a.min_free_gib < 0:
        p.error('Invalid free-space threshold')
    if a.vision_lr <= 0 or a.grid_lr <= 0 or a.weight_decay < 0:
        p.error('Invalid optimizer settings')
    if not 0 < a.vision_rate < 1 or min(a.speed_noise_std, a.angular_noise_std) < 0:
        p.error('Invalid vision rate or odometry noise')
    if (a.train_seed < 0 or a.train_seed + a.rollouts * BATCH >= a.train_seed + VAL_SEED_OFFSET
            or a.train_seed + VAL_SEED_OFFSET + a.val_episodes >= 1_000_000_000):
        p.error('Train and validation seed ranges overlap or overflow')
    return a


def check_replay_causality():
    """Synthetic tests: one-episode ordering, no future, full RGB, resume."""
    class FakeStore:
        def __init__(self, episode: int):
            images = np.zeros((1, 30, 3, 64, 64), dtype=np.uint8)
            for t in range(30):
                images[0, t, 0, 0, 0] = episode*100 + t
                images[0, t, 1, 5, 7] = episode*50 + t
            self._arrays = {
                'images': images,
                'pos': np.tile(np.arange(30, dtype=np.float32)[None, :, None],(1,1,2)),
                'hd': np.zeros((1,30), np.float32)}
    with tempfile.TemporaryDirectory() as folder:
        b = DiskHistoryReplay(Path(folder))
        ep0 = FakeStore(0)
        b.add_window(ep0, 0, 10)
        assert b.total_seen == 10
        rgb, pos, _ = b.sample(32, np.random.default_rng(1))
        assert rgb[:,0,0,0].max() <= 9 and rgb[:,1,5,7].max() <= 9
        assert pos.max() <= 9
        b.add_window(ep0, 10, 30)
        assert b.total_seen == 30
        ep1 = FakeStore(1)
        b.add_window(ep1, 0, 10)
        assert b.total_seen == 40
        assert b.image_bytes == 40*b.FRAME_BYTES
        saved = b.state_dict()
        b.add_window(ep1, 10, 30)
        assert b.total_seen == 60
        b.close()
        resumed = DiskHistoryReplay(Path(folder))
        resumed.load_state_dict(saved)
        assert resumed.total_seen == 40
        assert resumed.image_path.stat().st_size == 40*resumed.FRAME_BYTES
        assert resumed.index_path.stat().st_size == 40*resumed.INDEX_STRUCT.size
        ids = np.random.default_rng(1).integers(0,40,size=100)
        rgb, pos, _ = resumed.sample(100, np.random.default_rng(1))
        expect = np.where(ids < 30, ids, 100+ids-30)
        assert np.array_equal(rgb[:,0,0,0], expect.astype(np.uint8))
        assert np.array_equal(rgb[:,1,5,7], np.where(ids < 30, ids, 50+ids-30).astype(np.uint8))
        assert np.array_equal(pos[:,0], np.where(ids < 30, ids, ids-30))
        resumed.close()
    print('CHECK OK | sequential single-path causal history, raw RGB, uniform index, rollback',
          flush=True)

def main():
    a = parse_args()
    device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if a.device == 'auto' else a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    if a.check_only:
        check_replay_causality()
        return
    cfg = Config()
    if (cfg.rl.env.env_size_m != 2.5 or cfg.rl.env.action_repeat != ACTION_REPEAT
            or (cfg.rl.env.width, cfg.rl.env.height) != (64, 64)
            or tuple(cfg.rl.env.wall_world_units) != (100., 1100.)):
        raise ValueError('Arena/time/RGB settings differ from original A/B')
    if not np.allclose(
        (cfg.data.speed_rayleigh_scale, cfg.data.rot_vel_std_deg,
         cfg.data.sim_dt, cfg.data.perimeter_m, cfg.data.perimeter_speed_reduction),
        (0.13, 330., .02, .03, .25)):
        raise ValueError('RH motion parameters differ from original collector')
    cfg.rl.pc_scale = (SIGMA,)
    cfg.rl.env.level = rh1350.LEVEL_NAME
    cfg.rl.env.goal_visible = False
    level_path = rh1350.ensure_level(Path('levels').resolve(), write=True)
    level_sha = sha256_file(level_path)
    base.seed_everything(a.seed)
    pcs, hds = build_rl_ensembles(cfg, device)
    pc, hdc = pcs[0], hds[0]
    base.seed_everything(a.seed)
    grid = GridCellsRNN(
        target_ensembles=[pc, hdc], nh_lstm=128, nh_bottleneck=512,
        dropout_rates=(.5,), bottleneck_has_bias=True,
        ego_vel_dim=MOTION_DIMS, vision_dim=VISION_DIMS).to(device)
    initialize_sonnet_lstm_(grid.rl_lstm)
    if grid.rl_lstm.input_size != MOTION_DIMS + VISION_DIMS:
        raise RuntimeError('Grid LSTM must receive 4 motion + 268 Vision dimensions')
    base.seed_everything(a.seed + 100)
    vision = VisionModule(pc, hdc, embed_dim=256,
                          mask_prob=.95, mask_mode='whole').to(device)
    old_b.init_truncated_normal_vision(vision)
    vision.eval()
    base.seed_everything(a.seed + 999)
    grid_opt = torch.optim.RMSprop(
        grid.parameters(), lr=a.grid_lr, alpha=.9, momentum=.9,
        eps=1e-10, weight_decay=a.weight_decay)
    vision_opt = torch.optim.RMSprop(
        vision.parameters(), lr=a.vision_lr, alpha=.99, eps=1e-8)
    replay = None
    sample_rng = np.random.default_rng(a.seed + 89001)
    out = a.out_dir.expanduser().resolve()
    latest = out / 'latest.pt'
    metrics = out / 'metrics.csv'
    fingerprint = dict(
        version='online_B_sequential_all_history_raw_v1', train_seed=a.train_seed,
        val_seed=a.train_seed + VAL_SEED_OFFSET,
        val_episodes=a.val_episodes, val_batch_size=a.val_batch_size,
        eval_every=a.eval_every, save_every=a.save_every,
        vision_update_every=a.vision_update_every,
        max_vision_updates=a.max_vision_updates,
        replay_format='raw_rgb64_uint8',
        min_free_gib=a.min_free_gib,
        vision_lr=a.vision_lr, grid_lr=a.grid_lr, weight_decay=a.weight_decay,
        vision_rate=a.vision_rate, force_first_vision=a.force_first_vision,
        speed_noise_std=a.speed_noise_std, angular_noise_std=a.angular_noise_std,
        renderer=a.renderer, seed=a.seed, steps=STEPS, batch=BATCH,
        tbptt=TBPTT, level_sha=level_sha,
        script_sha=sha256_file(Path(__file__).resolve()),
        online_a_sha=sha256_file(Path(online_a.__file__).resolve()),
        old_b_sha=sha256_file(Path(old_b.__file__).resolve()),
        original_base_sha=sha256_file(Path(base.__file__).resolve()),
        collector_sha=sha256_file(Path(collector.__file__).resolve()),
        rh1350_sha=sha256_file(Path(rh1350.__file__).resolve()))
    start = 0
    grid_updates = 0
    vision_updates = 0
    replay_seen_total = 0
    if a.resume:
        if not latest.is_file():
            raise FileNotFoundError(f'Resume checkpoint missing: {latest}')
        ck = torch.load(latest, map_location='cpu', weights_only=False)
        if ck.get('fingerprint') != fingerprint:
            raise RuntimeError('Run settings/source changed: cannot safely resume')
        grid.load_state_dict(ck['grid'])
        grid_opt.load_state_dict(ck['grid_optimizer'])
        vision.load_state_dict(ck['vision'])
        vision_opt.load_state_dict(ck['vision_optimizer'])
        start = int(ck['rollout'])
        grid_updates = int(ck['grid_updates'])
        vision_updates = int(ck['vision_updates'])
        replay_seen_total = int(ck.get('replay_seen_total', 0))
        if grid_updates != start * 14:
            raise RuntimeError('Checkpoint Grid update count mismatch')
        if vision_updates < a.max_vision_updates:
            if ck.get('replay') is None:
                raise RuntimeError('Vision is still learning; resume replay is missing')
            replay = DiskHistoryReplay(out/'replay')
            replay.load_state_dict(ck['replay'])
            if replay_seen_total != replay.total_seen:
                raise RuntimeError('Replay observation count mismatch')
        else:
            replay = None
        sample_rng.bit_generator.state = ck['sampling_rng_state']
        torch.set_rng_state(ck['torch_rng_state'].cpu())
        if device.type == 'cuda' and 'cuda_rng_state' in ck:
            torch.cuda.set_rng_state(ck['cuda_rng_state'].cpu(), device)
        print(f'RESUMED | rollout={start}, Grid={grid_updates}, Vision={vision_updates}, '
              f'replay={replay.size if replay is not None else 0}', flush=True)
    elif latest.exists() or metrics.exists():
        raise FileExistsError(f'Existing output {out}; choose new --out-dir or --resume')
    if start >= a.rollouts:
        print('Already reached target rollouts', flush=True)
        return
    # Fresh training must not silently overwrite an orphaned replay after an interrupted run.
    if not a.resume and (out / 'replay').exists():
        raise FileExistsError(f'Existing replay files in {out}; choose a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    if not a.resume:
        replay = DiskHistoryReplay(out/'replay')

    print(f'EXPERIMENT | B-Online SEQUENTIAL 1 Vision path, 10 Grid paths | device={device}', flush=True)
    print(f'ENV | {STEPS} steps x {ACTION_REPEAT}/{FPS} sec = 90 s; '
          f'{BATCH} NEW episodes/rollout, ONE at a time; 14 Grid updates/rollout', flush=True)
    print(f'VISION | SINGLE EPISODE sequential; batch32 update every '
          f'{a.vision_update_every} new frames; cap={a.max_vision_updates:,}', flush=True)
    print('VISION->GRID | predictions stored at observation-time weights; no Grid gradient to Vision', flush=True)
    print(f'REPLAY | ALL history, RAW uint8 RGB (no eviction), disk={out / "replay"}', flush=True)
    print(f'DISK GUARD | minimum physical free space {a.min_free_gib:.1f}GiB + one rollout', flush=True)
    print(f'SEEDS | train [{a.train_seed}, {a.train_seed + a.rollouts*BATCH-1}] '
          f'val [{a.train_seed+VAL_SEED_OFFSET}, '
          f'{a.train_seed+VAL_SEED_OFFSET+a.val_episodes-1}]', flush=True)
    print('RGB | one current episode in RAM; all past frames RAW on disk until Vision freezes', flush=True)
    print(f'REPLAY DISK PROJECTION | at Vision300K: {a.max_vision_updates*a.vision_update_every*12288/2**30:.2f} GiB RGB '
          f'+ {a.max_vision_updates*a.vision_update_every*12/2**30:.3f} GiB labels; '
          f'host_or_wsl_free={free_space_gib(out):.1f} GiB', flush=True)

    def save(rollout_id: int):
        if replay is not None:
            replay.sync()
        checkpoint = dict(
            grid=grid.state_dict(), grid_optimizer=grid_opt.state_dict(),
            vision=vision.state_dict(), vision_module=vision.state_dict(),
            vision_optimizer=vision_opt.state_dict(),
            pc_stdev=(SIGMA,), env_size_m=2.5,
            hd_concentration=Config().task.hdc_concentration,
            pc_centers=pc.means.detach().cpu(), hd_centers=hdc.means.detach().cpu(),
            vision_mask_mode='whole', update=vision_updates,
            rollout=rollout_id, grid_updates=grid_updates,
            vision_updates=vision_updates,
            replay=replay.state_dict() if vision_updates < a.max_vision_updates else None,
            replay_seen_total=replay_seen_total,
            sampling_rng_state=sample_rng.bit_generator.state,
            fingerprint=fingerprint, torch_rng_state=torch.get_rng_state().cpu())
        if device.type == 'cuda':
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state(device).cpu()
        base.atomic_torch_save(checkpoint, latest)
        print(f'SAVED | rollout={rollout_id} | Grid={grid_updates} '
              f'Vision={vision_updates} | {latest}', flush=True)

    fields = ['rollout', 'fresh_train_episodes', 'grid_updates', 'vision_updates',
              'replay_seen_frames', 'replay_size', 'train_grid_ce',
              'recent_vision_loss', 'val_grid_ce', 'val_grid_pos_weighted_cm',
              'val_grid_hd_deg', 'val_vision_ce', 'val_vision_pos_cm',
              'val_vision_hd_deg', 'vision_input_ratio', 'teleport_steps']
    env = make_dmlab_env(cfg.rl.env, renderer=a.renderer)
    try:
        yaw_gain, nominal_speed = collector.calibrate_controls(
            env, cfg.rl.env, a.train_seed + 90_000_000, ACTION_REPEAT)
        with metrics.open('a' if a.resume else 'w', newline='', buffering=1) as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if not a.resume:
                writer.writeheader()
            for rollout_id in range(start + 1, a.rollouts + 1):
                # Check actual Windows C: free space, NOT just the WSL /dev/sdX
                # virtual 1-TB capacity, before writing each 10-episode group.
                if replay is not None and free_space_gib(out) < a.min_free_gib + 0.16:
                    save(rollout_id - 1)
                    raise RuntimeError('Not enough physical disk free space for ALL RGB frames (20GiB reserve). '
                                       'Existing files/checkpoint are retained.')
                ids = np.arange((rollout_id - 1) * BATCH,
                                rollout_id * BATCH, dtype=np.int64)
                # These arrays hold the 10 trajectories' *small* model inputs,
                # NOT their RGB. Only ONE episode of RGB exists in RAM at a time.
                positions = np.empty((BATCH, STEPS, 2), dtype=np.float32)
                headings = np.empty((BATCH, STEPS), dtype=np.float32)
                motion = np.empty((BATCH, STEPS, MOTION_DIMS), dtype=np.float32)
                resets = np.empty((BATCH, STEPS), dtype=np.bool_)
                visual = np.zeros((BATCH, STEPS, VISION_DIMS), dtype=np.float32)
                count_visible = 0
                teleports = 0
                last_vision_loss = float('nan')

                for row, eid in enumerate(ids):
                    one_id = np.asarray([eid], dtype=np.int64)
                    # Sequential episodes: ep 2 doesn't even get generated before
                    # Vision finishes processing ep 1, etc.
                    store, ep_teleports = online_a.generate_episodes(
                        env, cfg.rl.env, one_id, a.train_seed,
                        yaw_gain, nominal_speed)
                    teleports += ep_teleports
                    p, h, v, r, visible = old_b.make_motion(
                        store, one_id, a.seed, rollout_id, a.vision_rate,
                        a.force_first_vision, a.speed_noise_std, a.angular_noise_std)
                    positions[row] = p[0]
                    headings[row] = h[0]
                    motion[row] = v[0]
                    resets[row] = r[0]
                    count_visible += int(visible.sum())
                    # Historical visual input: computed with the Vision network
                    # available AT THAT MOMENT, before learning on window images.
                    vis_one = torch.zeros((1, STEPS, VISION_DIMS),
                                          dtype=torch.float32, device=device)
                    for t in range(0, STEPS, a.vision_update_every):
                        t_end = min(t + a.vision_update_every, STEPS)
                        old_b.infer_vision_window(
                            vision, store, one_id, visible, t, t_end,
                            vis_one, device)
                        if vision_updates < a.max_vision_updates:
                            replay.add_window(store, t, t_end)
                            replay_seen_total = replay.total_seen
                            last_vision_loss = vision_update(
                                vision, vision_opt, pc, hdc,
                                replay, sample_rng, device)
                            vision_updates += 1
                            if vision_updates == a.max_vision_updates:
                                print(f'VISION FROZEN | update={vision_updates:,} '
                                      f'after {replay_seen_total:,} frames, '
                                      f'rollout={rollout_id}, episode={int(eid)}, '
                                      f'step={t_end}', flush=True)
                                # Preserve replay data on disk; no new RGB storage.
                                replay.close()
                                replay = None
                    visual[row] = vis_one.cpu().numpy()[0]
                    del store, p, h, v, r, visible, vis_one
                    if rollout_id == 1 or (row == BATCH - 1 and rollout_id % 10 == 0):
                        print(f'EPISODE | rollout={rollout_id} path={row+1}/{BATCH} '
                              f'vision_updates={vision_updates:,} '
                              f'replay_frames={replay_seen_total:,}', flush=True)

                # Grid receives 10 fresh paths, with visual predictions from the
                # Vision versions available when each image was observed.
                pos = torch.from_numpy(positions).to(device)
                hd = torch.from_numpy(headings).to(device)
                vel = torch.from_numpy(motion).to(device)
                reset = torch.from_numpy(resets).to(device)
                vis = torch.from_numpy(visual).to(device)
                hidden = (torch.zeros(1, BATCH, 128, device=device),
                          torch.zeros(1, BATCH, 128, device=device))
                total_grid_ce = 0.0
                grid.train()
                for start_t in range(0, STEPS, TBPTT):
                    end_t = min(start_t + TBPTT, STEPS)
                    hidden = (hidden[0].detach(), hidden[1].detach())
                    lp, lh, hidden = base.step_segment(
                        grid, vel[:, start_t:end_t], vis[:, start_t:end_t],
                        reset[:, start_t:end_t], hidden, training=True)
                    tpc, thd = base.targets(
                        pc, hdc, pos[:, start_t:end_t], hd[:, start_t:end_t])
                    ce = (-(tpc * F.log_softmax(lp, -1)).sum(-1)
                          -(thd * F.log_softmax(lh, -1)).sum(-1)).mean()
                    if not torch.isfinite(ce):
                        raise FloatingPointError('Nonfinite Grid loss')
                    grid_opt.zero_grad(set_to_none=True)
                    (ce * ((end_t - start_t) / TBPTT)).backward()
                    torch.nn.utils.clip_grad_norm_(grid.parameters(), max_norm=1.0)
                    grid_opt.step()
                    grid_updates += 1
                    total_grid_ce += float(ce.detach()) * (end_t - start_t)
                train_ce = total_grid_ce / STEPS
                ratio = count_visible / (BATCH*STEPS)
                del positions, headings, motion, resets, visual
                del pos, hd, vel, reset, vis, hidden
                if rollout_id % a.eval_every == 0 or rollout_id == a.rollouts:
                    report = validate_b(grid, vision, pc, hdc, env, cfg.rl.env,
                                        a, yaw_gain, nominal_speed, device)
                    writer.writerow(dict(
                        rollout=rollout_id, fresh_train_episodes=rollout_id * BATCH,
                        grid_updates=grid_updates, vision_updates=vision_updates,
                        replay_seen_frames=replay_seen_total,
                        replay_size=replay.size if replay is not None else 0,
                        train_grid_ce=train_ce, recent_vision_loss=last_vision_loss,
                        val_grid_ce=report['grid_ce'],
                        val_grid_pos_weighted_cm=report['grid_pos_weighted_cm'],
                        val_grid_hd_deg=report['grid_hd_deg'],
                        val_vision_ce=report['vision_ce'],
                        val_vision_pos_cm=report['vision_pos_cm'],
                        val_vision_hd_deg=report['vision_hd_deg'],
                        vision_input_ratio=ratio, teleport_steps=teleports))
                    print(f'rollout={rollout_id}/{a.rollouts} | '
                          f'new_ep={rollout_id*BATCH:,} Grid={grid_updates:,} '
                          f'Vision={vision_updates:,} | '
                          f'Grid train/val CE={train_ce:.4f}/{report["grid_ce"]:.4f} '
                          f'pos={report["grid_pos_weighted_cm"]:.1f}cm '
                          f'HD={report["grid_hd_deg"]:.1f}deg | '
                          f'Vision val pos={report["vision_pos_cm"]:.1f}cm '
                          f'HD={report["vision_hd_deg"]:.1f}deg | '
                          f'vis={ratio:.1%} teleports={teleports}', flush=True)
                elif rollout_id % 10 == 0 or rollout_id == 1:
                    print(f'rollout={rollout_id}/{a.rollouts} | '
                          f'new_ep={rollout_id*BATCH:,} Grid={grid_updates:,} '
                          f'Vision={vision_updates:,} | train CE={train_ce:.4f} '
                          f'replay={replay.size if replay is not None else 0} '
                          f'vis={ratio:.1%}', flush=True)
                # Raw replay bytes give an exact disk capacity forecast.
                if replay is not None and (rollout_id == 1 or rollout_id % 10 == 0):
                    avg = replay.image_bytes / max(1, replay.total_seen)
                    expected = (a.max_vision_updates * a.vision_update_every)
                    forecast_gib = avg * expected / 2**30
                    on_disk_gib = (replay.image_bytes + replay.total_seen * replay.INDEX_STRUCT.size) / 2**30
                    free_gib = free_space_gib(out)
                    print(f'REPLAY DISK | stored={replay.total_seen:,} frames, '
                          f'avg_payload={avg:.0f} B/image, actual={on_disk_gib:.3f} GiB, '
                          f'estimate_for_Vision300K={forecast_gib:.1f} GiB '
                          f'(+ labels ~0.034 GiB), host_or_wsl_free={free_gib:.1f} GiB', flush=True)
                if rollout_id % a.save_every == 0 or rollout_id == a.rollouts:
                    save(rollout_id)
    finally:
        env.close()
        if replay is not None:
            replay.close()
    print(f'COMPLETE | {grid_updates:,} Grid updates, {vision_updates:,} '
          f'Vision updates, {a.rollouts*BATCH:,} fresh train episodes', flush=True)


if __name__ == '__main__':
    main()
