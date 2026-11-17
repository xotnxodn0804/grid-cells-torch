#!/usr/bin/env python3
"""B-Online: fresh DMLab trajectories + CAUSAL Vision training with bounded replay.

At every rollout, generate 10 never-reused DMLab episodes x 1350 steps in RAM.
At each temporal window [t, t+10), predict Grid's Vision inputs using the
CURRENT Vision weights *before* the Vision update, add ONLY the already-
observed RGB/position/heading samples to a bounded uniform reservoir, and
train Vision once with batch 32 from that reservoir. Then run a 100-step
Grid TBPTT update (final 50 steps weighted by 0.5). Past episodes are not
revisited as paths. The replay reservoir can retain SOME prior observed RGB
images for Vision training, and is persisted in the latest checkpoint while
Vision is still learning. No multi-gigabyte image dataset is written.

This is *causally ordered supervised learning from newly generated paths*;
it is NOT online actor-critic, and the whole 10-episode rollout is collected
into RAM before any model update. Future frames are nevertheless inaccessible
to the Vision learner until their timestamp has been reached.

Uses existing local project modules, including train_grid_vision_1350_online_A.py
and train_grid_vision_1350_causal_ab.py. Does not overwrite A or old B runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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


class CausalReservoir:
    """Bounded, exactly uniform reservoir of ALL RGB frames observed so far.

    Memory is approximately capacity * (3*64*64+12) bytes. The sampling
    algorithm is equivalent to sequential Algorithm R, vectorized over each
    small temporal window for efficiency. It never reads unobserved frames.
    """
    def __init__(self, capacity: int, seed: int):
        if capacity < 32:
            raise ValueError('Replay capacity must be at least 32')
        self.capacity = int(capacity)
        self.images = np.empty((capacity, 3, 64, 64), np.uint8)
        self.pos = np.empty((capacity, 2), np.float32)
        self.hd = np.empty((capacity,), np.float32)
        self.size = 0
        self.total_seen = 0
        self.rng = np.random.default_rng(seed)

    def add_window(self, store: online_a.InMemoryEpisodes, begin: int, end: int):
        """Add only timestamps begin <= t < end from CURRENT rollout."""
        if not 0 <= begin < end <= STEPS:
            raise ValueError('Noncausal or empty temporal window')
        arrays = store._arrays
        # Time-major ordering: frames at t first, then all 10 agents at t.
        imgs = arrays['images'][:, begin:end].transpose(1, 0, 2, 3, 4).reshape(-1, 3, 64, 64)
        positions = arrays['pos'][:, begin:end].transpose(1, 0, 2).reshape(-1, 2)
        headings = arrays['hd'][:, begin:end].transpose(1, 0).reshape(-1)
        n = len(imgs)
        if n <= 0:
            return
        n_init = min(n, self.capacity - self.size)
        if n_init:
            target = slice(self.size, self.size + n_init)
            self.images[target] = imgs[:n_init]
            self.pos[target] = positions[:n_init]
            self.hd[target] = headings[:n_init]
            self.size += n_init
            self.total_seen += n_init
        remainder = n - n_init
        if remainder:
            # Seen counts before each incoming frame are total_seen + i.
            highs = self.total_seen + np.arange(1, remainder + 1, dtype=np.int64)
            slots = np.floor(self.rng.random(remainder) * highs).astype(np.int64)
            accepted = np.flatnonzero(slots < self.capacity)
            if len(accepted):
                target_slots = slots[accepted]
                # Multiple replacements can target one slot in this window;
                # only the LAST accepted replacement survives.
                _, last_from_reversed = np.unique(target_slots[::-1], return_index=True)
                which = accepted[len(accepted) - 1 - last_from_reversed]
                slots_final = slots[which]
                self.images[slots_final] = imgs[n_init + which]
                self.pos[slots_final] = positions[n_init + which]
                self.hd[slots_final] = headings[n_init + which]
            self.total_seen += remainder
        if self.size != min(self.capacity, self.total_seen):
            raise AssertionError('Replay reservoir size mismatch')

    def sample(self, size: int, rng: np.random.Generator):
        if not self.size:
            raise RuntimeError('Cannot sample empty replay buffer')
        which = rng.integers(0, self.size, size=size)
        return self.images[which], self.pos[which], self.hd[which]

    def state_dict(self):
        return {'capacity': self.capacity, 'size': self.size,
                'total_seen': self.total_seen,
                'images': self.images[:self.size].copy(),
                'pos': self.pos[:self.size].copy(),
                'hd': self.hd[:self.size].copy(),
                'reservoir_rng_state': self.rng.bit_generator.state}

    def load_state_dict(self, state):
        if state['capacity'] != self.capacity:
            raise ValueError('Replay capacity changed while resuming')
        self.size = int(state['size'])
        self.total_seen = int(state['total_seen'])
        if not 0 <= self.size <= self.capacity or self.size != min(self.capacity, self.total_seen):
            raise ValueError('Damaged replay state')
        for name in ('images', 'pos', 'hd'):
            source = state[name]
            if source.shape != getattr(self, name)[:self.size].shape:
                raise ValueError(f'Wrong replay {name} shape')
            getattr(self, name)[:self.size] = source
        self.rng.bit_generator.state = state['reservoir_rng_state']


def vision_update(vision, optimizer, pc, hdc, replay: CausalReservoir,
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
    """Held-out 100 fixed-seed episodes, regenerated only at evaluation time.

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
    p.add_argument('--replay-capacity', type=int, default=8192,
                   help='Uniform reservoir of previous RGB frames, in RAM (8192 ~= 96 MiB)')
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
                 'save_every', 'vision_update_every', 'max_vision_updates',
                 'replay_capacity'):
        if getattr(a, name) <= 0:
            p.error(f'{name} must be positive')
    if a.vision_update_every > TBPTT or TBPTT % a.vision_update_every or 50 % a.vision_update_every:
        p.error('--vision-update-every must divide both 100 and 50')
    if a.replay_capacity < 32:
        p.error('--replay-capacity must be >= 32')
    if a.vision_lr <= 0 or a.grid_lr <= 0 or a.weight_decay < 0:
        p.error('Invalid optimizer settings')
    if not 0 < a.vision_rate < 1 or min(a.speed_noise_std, a.angular_noise_std) < 0:
        p.error('Invalid vision rate or odometry noise')
    if (a.train_seed < 0 or a.train_seed + a.rollouts * BATCH >= a.train_seed + VAL_SEED_OFFSET
            or a.train_seed + VAL_SEED_OFFSET + a.val_episodes >= 1_000_000_000):
        p.error('Train and validation seed ranges overlap or overflow')
    return a


def check_replay_causality():
    class FakeStore:
        def __init__(self):
            images = np.zeros((10, 30, 3, 64, 64), dtype=np.uint8)
            for t in range(30):
                images[:, t, 0, 0, 0] = t % 251
            self._arrays = {
                'images': images,
                'pos': np.zeros((10, STEPS, 2), np.float32),
                'hd': np.zeros((10, STEPS), np.float32)}
    store = FakeStore()
    b = CausalReservoir(64, seed=1)
    b.add_window(store, 0, 10)
    assert b.total_seen == 100 and b.size == 64
    assert np.max(b.images[:b.size, 0, 0, 0]) <= 9
    b.add_window(store, 10, 30)
    assert b.total_seen == 300
    assert np.max(b.images[:b.size, 0, 0, 0]) <= 29
    saved = b.state_dict()
    restored = CausalReservoir(64, seed=0)
    restored.load_state_dict(saved)
    assert np.array_equal(restored.images[:restored.size], b.images[:b.size])
    assert restored.total_seen == b.total_seen
    print('CHECK OK | t=10 only <=10; t=30 only <=30; reservoir checkpoint restored', flush=True)


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
    replay = CausalReservoir(a.replay_capacity, a.seed + 24680)
    sample_rng = np.random.default_rng(a.seed + 89001)
    out = a.out_dir.expanduser().resolve()
    latest = out / 'latest.pt'
    metrics = out / 'metrics.csv'
    fingerprint = dict(
        version='online_B_reservoir_v1', train_seed=a.train_seed,
        val_seed=a.train_seed + VAL_SEED_OFFSET,
        val_episodes=a.val_episodes, val_batch_size=a.val_batch_size,
        eval_every=a.eval_every, save_every=a.save_every,
        vision_update_every=a.vision_update_every,
        max_vision_updates=a.max_vision_updates,
        replay_capacity=a.replay_capacity,
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
    out.mkdir(parents=True, exist_ok=True)

    print(f'EXPERIMENT | B-Online fresh DMLab episodes | device={device}', flush=True)
    print(f'ENV | {STEPS} steps x {ACTION_REPEAT}/{FPS} sec = 90 s; '
          f'{BATCH} NEW episodes/rollout; 14 Grid updates/rollout', flush=True)
    print(f'VISION | fresh weights; one 32-frame replay update every '
          f'{a.vision_update_every} steps; cap={a.max_vision_updates:,}', flush=True)
    print(f'REPLAY | reservoir={a.replay_capacity} RGB frames '
          f'(~{a.replay_capacity * 12288 / 2**20:.1f} MiB), '
          'saved with checkpoint while Vision trains', flush=True)
    print(f'SEEDS | train [{a.train_seed}, {a.train_seed + a.rollouts*BATCH-1}] '
          f'val [{a.train_seed+VAL_SEED_OFFSET}, '
          f'{a.train_seed+VAL_SEED_OFFSET+a.val_episodes-1}]', flush=True)
    print('RGB | current rollout in RAM; no full RGB dataset saved', flush=True)

    def save(rollout_id: int):
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
                ids = np.arange((rollout_id - 1) * BATCH,
                                rollout_id * BATCH, dtype=np.int64)
                store, teleports = online_a.generate_episodes(
                    env, cfg.rl.env, ids, a.train_seed, yaw_gain, nominal_speed)
                positions, headings, motion, resets, visible = old_b.make_motion(
                    store, ids, a.seed, rollout_id, a.vision_rate,
                    a.force_first_vision, a.speed_noise_std, a.angular_noise_std)
                pos = torch.from_numpy(positions).to(device)
                hd = torch.from_numpy(headings).to(device)
                vel = torch.from_numpy(motion).to(device)
                reset = torch.from_numpy(resets).to(device)
                visual = torch.zeros((BATCH, STEPS, VISION_DIMS),
                                     device=device, dtype=torch.float32)
                hidden = (torch.zeros(1, BATCH, 128, device=device),
                          torch.zeros(1, BATCH, 128, device=device))
                total_grid_ce = 0.0
                last_vision_loss = float('nan')
                grid.train()
                for start_t in range(0, STEPS, TBPTT):
                    end_t = min(start_t + TBPTT, STEPS)
                    for t in range(start_t, end_t, a.vision_update_every):
                        t_end = min(t + a.vision_update_every, end_t)
                        # Predict Grid's Vision input using Vision weights BEFORE learning this window.
                        old_b.infer_vision_window(
                            vision, store, ids, visible, t, t_end, visual, device)
                        if vision_updates < a.max_vision_updates:
                            # The only new frames admitted are <= t_end, never beyond it.
                            replay.add_window(store, t, t_end)
                            replay_seen_total = replay.total_seen
                            last_vision_loss = vision_update(
                                vision, vision_opt, pc, hdc, replay, sample_rng, device)
                            vision_updates += 1
                            if vision_updates == a.max_vision_updates:
                                print(f'VISION FROZEN | update={vision_updates:,} '
                                      f'at rollout={rollout_id}, step={t_end}', flush=True)
                                # Older replay frames are unnecessary after freeze.
                                replay = None
                    hidden = (hidden[0].detach(), hidden[1].detach())
                    lp, lh, hidden = base.step_segment(
                        grid, vel[:, start_t:end_t], visual[:, start_t:end_t],
                        reset[:, start_t:end_t], hidden, training=True)
                    target_pc, target_hd = base.targets(
                        pc, hdc, pos[:, start_t:end_t], hd[:, start_t:end_t])
                    ce = (-(target_pc * F.log_softmax(lp, -1)).sum(-1)
                          -(target_hd * F.log_softmax(lh, -1)).sum(-1)).mean()
                    if not torch.isfinite(ce):
                        raise FloatingPointError('Nonfinite Grid loss')
                    grid_opt.zero_grad(set_to_none=True)
                    (ce * ((end_t - start_t) / TBPTT)).backward()
                    torch.nn.utils.clip_grad_norm_(grid.parameters(), max_norm=1.0)
                    grid_opt.step()
                    grid_updates += 1
                    total_grid_ce += float(ce.detach()) * (end_t - start_t)
                train_ce = total_grid_ce / STEPS
                ratio = float(visible.mean())
                del store, positions, headings, motion, resets, visible
                del pos, hd, vel, reset, visual, hidden
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
                if rollout_id % a.save_every == 0 or rollout_id == a.rollouts:
                    save(rollout_id)
    finally:
        env.close()
    print(f'COMPLETE | {grid_updates:,} Grid updates, {vision_updates:,} '
          f'Vision updates, {a.rollouts*BATCH:,} fresh train episodes', flush=True)


if __name__ == '__main__':
    main()
