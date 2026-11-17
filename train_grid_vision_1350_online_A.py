#!/usr/bin/env python3
"""A-Online: fresh 1350-step DMLab episodes for EVERY Grid rollout (no RGB dataset saved).

Preserves offline-A configuration: frozen Vision PC256/HD12 (sigma=0.01), 5% whole
vision with first observation and all detected teleports forced visible; 4-D noisy
odometry, LSTM128, bottleneck512, 50% dropout, TBPTT100 (+ final weighted 50),
10 independent episodes/rollout, RMSprop and 300006 Grid updates over 21429 rollouts.

Each episode uses a *unique* DMLab reset seed. RGB is held briefly in RAM for one
10-episode rollout (or one validation batch), then discarded. It is NEVER written
as a .npy dataset. Validation uses separate FIXED seeds, regenerated on demand.
The original offline-A checkpoint and all existing result/figure files are untouched.

Requires existing project files:
    collect_vision_rh_1000_v2.py, collect_rh_1350_v3.py,
    train_grid_vision_1350_causal_ab.py,
    train_grid_dmlab_vision_pc001_272_noisy.py, config.py, grid_cells/*

This is supervised path integration using live-generated DMLab trajectories,
NOT online actor-critic / policy learning. It is currently single-environment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
import collect_vision_rh_1000_v2 as collector
import collect_rh_1350_v3 as rh1350
import train_grid_vision_1350_causal_ab as offline
import train_grid_dmlab_vision_pc001_272_noisy as base

STEPS = 1350
BATCH = 10
TBPTT = 100
SIGMA = 0.01
MOTION_DIMS = 4
VISION_DIMS = 268
FPS = 60
ACTION_REPEAT = 4
TRAIN_SEED_DEFAULT = 75_000_000  # Match the original 1350-step collection seeds
VAL_SEED_OFFSET = 10_000_000    # Match the original 1350-step validation seeds
# The imported original training helper otherwise assumes 1300 steps.
base.STEPS = STEPS
assert offline.STEPS == STEPS and rh1350.STEPS == STEPS


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for buf in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


class InMemoryEpisodes:
    """Temporary compatible EpisodeStore for the existing original training helpers.

    Episode IDs can be absolute within a split (validation) or local to the
    currently sampled batch (training). All backing arrays are NumPy RAM arrays.
    """
    def __init__(self, ids: np.ndarray):
        self.ids = [int(v) for v in ids]
        if len(set(self.ids)) != len(self.ids) or not self.ids:
            raise ValueError('Episode IDs must be unique and nonempty')
        n = len(ids)
        self._arrays = {
            'images': np.empty((n, STEPS, 3, 64, 64), np.uint8),
            'pos': np.empty((n, STEPS, 2), np.float32),
            'hd': np.empty((n, STEPS), np.float32),
            'vel_trans': np.empty((n, STEPS, 3), np.float32),
            'ego_vel': np.empty((n, STEPS, 3), np.float32),
            'valid_motion': np.empty((n, STEPS), np.bool_),
        }
        # offline.read_samples uses store.lookup[eid] and store.entries[..]['arrays']
        self.lookup = {eid: (0, row) for row, eid in enumerate(self.ids)}
        self.entries = [{'arrays': {name: array.reshape((n * STEPS, *array.shape[2:]))
                                    for name, array in self._arrays.items()}}]

    def __len__(self):
        # offline.read_samples validates absolute IDs against len(store).
        return max(self.ids) + 1

    def episode(self, eid: int):
        row = self.lookup[int(eid)][1]
        a = self._arrays
        return (a['pos'][row], a['hd'][row], a['vel_trans'][row],
                a['ego_vel'][row], a['valid_motion'][row])

    def images_at(self, eid: int, indices: np.ndarray):
        row = self.lookup[int(eid)][1]
        return self._arrays['images'][row, indices]


def generate_episodes(env, cfg, ids: np.ndarray, episode_seed_base: int,
                      yaw_gain: float, nominal_speed: float) -> tuple[InMemoryEpisodes, int]:
    """Exact v2 collector movement policy, without memmaps or image files."""
    store = InMemoryEpisodes(ids)
    a = store._arrays
    interval = ACTION_REPEAT / FPS
    teleports = 0
    for row, eid in enumerate(store.ids):
        ep_seed = episode_seed_base + eid
        rng = np.random.default_rng(episode_seed_base + eid * 7919)
        env.reset(seed=ep_seed)
        prev = collector.read_obs(env, cfg)
        motor = collector.RHTargetMotion(arena_m=cfg.env_size_m)
        for t in range(STEPS):
            if not env.is_running():
                raise RuntimeError(f'DMLab stopped before step {t}, seed={ep_seed}')
            target_speed, target_turn = motor.sample(
                rng, prev['pos'], float(prev['hd'][0]), interval)
            look = int(np.clip(round(math.degrees(target_turn) / yaw_gain), -512, 512))
            forward = bool(rng.random() < min(1.0, target_speed / nominal_speed))
            env.step(collector.action(look, forward), num_steps=ACTION_REPEAT)
            if not env.is_running():
                raise RuntimeError(f'DMLab stopped at step {t+1}/{STEPS}, seed={ep_seed}; '
                                   'verify that 91-second arena level is active')
            obs = collector.read_obs(env, cfg)
            displacement = float(np.linalg.norm(obs['pos'] - prev['pos']))
            dtheta = collector.wrap_angle(float(obs['hd'][0] - prev['hd'][0]))
            valid = displacement <= 0.35  # Match the old collector teleport threshold.
            if not valid:
                dtheta = 0.0
                motor.last_target_speed = 0.0
                teleports += 1
            a['images'][row, t] = obs['image']
            a['pos'][row, t] = obs['pos']
            a['hd'][row, t] = float(obs['hd'][0])
            a['vel_trans'][row, t] = obs['vel_trans']
            a['ego_vel'][row, t] = (displacement / interval if valid else 0.0,
                                    math.sin(dtheta), math.cos(dtheta))
            a['valid_motion'][row, t] = valid
            prev = obs
    return store, teleports


def validate_live(grid, vision, pc, hd_cell, env, cfg, args, yaw_gain, nominal_speed,
                  val_base: int, device) -> dict:
    """Same calculations/masks as base.validate; regenerated fixed validation seeds.

    Processes at most val_batch_size episodes of raw RGB in RAM at any time.
    """
    grid.eval()
    sums = np.zeros(5, np.float64)
    last_sum = np.zeros(2, np.float64)
    seen = 0
    try:
        with torch.no_grad():
            for first in range(0, args.val_episodes, args.val_batch_size):
                ids = np.arange(first, min(first+args.val_batch_size, args.val_episodes),
                                dtype=np.int64)
                store, _ = generate_episodes(env, cfg, ids, val_base, yaw_gain, nominal_speed)
                roll = base.build_rollout(
                    store, ids, vision, device, np.random.default_rng(0),
                    np.random.default_rng(0), args.vision_rate,
                    args.force_first_vision, args.speed_noise_std,
                    args.angular_noise_std, deterministic_val=True,
                    val_seed=args.seed+83817)
                batch_n = len(ids)
                hidden = (torch.zeros(1, batch_n, 128, device=device),
                          torch.zeros(1, batch_n, 128, device=device))
                for s in range(0, STEPS, TBPTT):
                    e = min(s+TBPTT, STEPS)
                    logits_pc, logits_hd, hidden = base.step_segment(
                        grid, roll.vel[:, s:e], roll.vis[:, s:e],
                        roll.reset[:, s:e], hidden, training=False)
                    target_pc, target_hd = base.targets(
                        pc, hd_cell, roll.pos[:, s:e], roll.hd[:, s:e])
                    ce = (-(target_pc*F.log_softmax(logits_pc,-1)).sum(-1)
                          -(target_hd*F.log_softmax(logits_hd,-1)).sum(-1))
                    errors = base.prediction_errors(
                        pc, hd_cell, logits_pc, logits_hd,
                        roll.pos[:, s:e], roll.hd[:, s:e])
                    for j, err in enumerate(errors):
                        sums[j] += err.sum().item()
                    sums[4] += ce.sum().item()
                    if e == STEPS:
                        last_sum[0] += errors[0][:,-1].sum().item()
                        last_sum[1] += errors[3][:,-1].sum().item()
                    seen += batch_n * (e-s)
                del store, roll
    finally:
        grid.train()
    return dict(ce=sums[4]/seen, pos_weighted_cm=sums[1]/seen,
                pos_top3_cm=sums[0]/seen, pos_argmax_cm=sums[2]/seen,
                hd_circular_deg=sums[3]/seen,
                final_pos_top3_cm=last_sum[0]/args.val_episodes,
                final_hd_deg=last_sum[1]/args.val_episodes)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--vision-checkpoint', type=Path,
                   default=Path('data/checkpoints/vision1350_offline_A/latest.pt'))
    p.add_argument('--out-dir', type=Path,
                   default=Path('data/checkpoints/grid1350_online_A'))
    p.add_argument('--rollouts', type=int, default=21429)
    p.add_argument('--train-seed', type=int, default=TRAIN_SEED_DEFAULT)
    p.add_argument('--val-episodes', type=int, default=100)
    p.add_argument('--val-batch-size', type=int, default=8)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--vision-rate', type=float, default=0.05)
    p.add_argument('--speed-noise-std', type=float, default=0.01)
    p.add_argument('--angular-noise-std', type=float, default=0.01)
    p.add_argument('--force-first-vision', action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument('--grid-lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--renderer', choices=('hardware','software'), default='hardware')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('auto','cuda','cpu'), default='auto')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if any(getattr(a, field) < 1 for field in
           ('rollouts','val_episodes','val_batch_size','eval_every','save_every')):
        p.error('rollouts, val-episodes, val-batch-size, eval-every, save-every must be >= 1')
    if not (0 < a.vision_rate < 1):
        p.error('vision-rate must be between 0 and 1')
    if min(a.speed_noise_std, a.angular_noise_std, a.grid_lr, a.weight_decay) < 0:
        p.error('noise/learning-rate/weight-decay values must be nonnegative')
    if a.grid_lr <= 0:
        p.error('grid-lr must be positive')
    if (a.train_seed < 0 or a.train_seed + a.rollouts*BATCH >= a.train_seed+VAL_SEED_OFFSET
            or a.train_seed+VAL_SEED_OFFSET+a.val_episodes >= 1_000_000_000):
        p.error('Train/val DMLab seed regions would overlap or overflow')
    return a


def main():
    a = parse_args()
    device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if a.device == 'auto' else a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    cfg = Config()
    if (cfg.rl.env.env_size_m != 2.5 or cfg.rl.env.action_repeat != ACTION_REPEAT
            or (cfg.rl.env.width, cfg.rl.env.height) != (64, 64)
            or tuple(cfg.rl.env.wall_world_units) != (100., 1100.)):
        raise ValueError('Arena, RGB shape or action timing differs from original A')
    if not np.allclose(
        (cfg.data.speed_rayleigh_scale, cfg.data.rot_vel_std_deg,
         cfg.data.sim_dt, cfg.data.perimeter_m, cfg.data.perimeter_speed_reduction),
        (0.13, 330.0, 0.02, 0.03, 0.25)):
        raise ValueError('RH collector motion settings differ from old A')
    cfg.rl.pc_scale = (SIGMA,)
    cfg.rl.env.level = rh1350.LEVEL_NAME
    cfg.rl.env.goal_visible = False
    # The old collector creates a verified 91s clone, never changes square_arena.lua.
    level_file = rh1350.ensure_level(Path('levels').resolve(), write=True)
    level_hash = digest(level_file)
    vis_path = a.vision_checkpoint.expanduser().resolve()
    if not vis_path.is_file():
        raise FileNotFoundError(f'Pretrained Vision checkpoint is missing: {vis_path}')
    pretrained_sha = digest(vis_path)

    base.seed_everything(a.seed)
    pcs, hds = build_rl_ensembles(cfg, device)
    pc, hdc = pcs[0], hds[0]
    base.seed_everything(a.seed)
    grid = GridCellsRNN(
        target_ensembles=[pc, hdc], nh_lstm=128, nh_bottleneck=512,
        dropout_rates=(.5,), bottleneck_has_bias=True,
        ego_vel_dim=MOTION_DIMS, vision_dim=VISION_DIMS).to(device)
    initialize_sonnet_lstm_(grid.rl_lstm)
    base.seed_everything(a.seed+100)
    vision = VisionModule(pc, hdc, embed_dim=256,
                          mask_prob=.95, mask_mode='whole').to(device)
    saved_vision = torch.load(vis_path, map_location='cpu', weights_only=False)
    if (abs(float(saved_vision['pc_stdev'][0])-SIGMA) > 1e-9
            or float(saved_vision['env_size_m']) != 2.5
            or not torch.allclose(saved_vision['pc_centers'].cpu(), pc.means.cpu(),
                                   atol=1e-7, rtol=0)
            or not torch.allclose(saved_vision['hd_centers'].cpu(), hdc.means.cpu(),
                                   atol=1e-7, rtol=0)):
        raise ValueError('Vision checkpoint target centers/sigma are incompatible')
    vision.load_state_dict(saved_vision['vision_module'], strict=True)
    vision.eval().requires_grad_(False)
    base.seed_everything(a.seed+999)
    grid_opt = torch.optim.RMSprop(grid.parameters(), lr=a.grid_lr,
                                   alpha=.9, momentum=.9, eps=1e-10,
                                   weight_decay=a.weight_decay)
    out = a.out_dir.expanduser().resolve()
    latest = out / 'latest.pt'
    metrics = out / 'metrics.csv'
    fingerprint = dict(version='online_A_v1', train_seed=a.train_seed,
                       val_seed=a.train_seed+VAL_SEED_OFFSET, val_episodes=a.val_episodes,
                       val_batch_size=a.val_batch_size, eval_every=a.eval_every,
                       save_every=a.save_every, vision_rate=a.vision_rate,
                       speed_noise_std=a.speed_noise_std,
                       angular_noise_std=a.angular_noise_std,
                       force_first_vision=a.force_first_vision,
                       grid_lr=a.grid_lr, weight_decay=a.weight_decay,
                       renderer=a.renderer, seed=a.seed,
                       steps=STEPS, batch=BATCH, tbptt=TBPTT,
                       pretrained_sha=pretrained_sha, level_sha=level_hash,
                       script_sha=digest(Path(__file__).resolve()),
                       collector_sha=digest(Path(collector.__file__).resolve()),
                       rh1350_sha=digest(Path(rh1350.__file__).resolve()),
                       original_train_sha=digest(Path(offline.__file__).resolve()),
                       original_base_sha=digest(Path(base.__file__).resolve()))
    start = 0
    grid_updates = 0
    if a.resume:
        if not latest.is_file():
            raise FileNotFoundError(f'Cannot resume: {latest} is missing')
        ck = torch.load(latest, map_location=device, weights_only=False)
        if ck.get('fingerprint') != fingerprint:
            raise RuntimeError('Settings differ from saved run; keep flags identical')
        grid.load_state_dict(ck['grid'])
        grid_opt.load_state_dict(ck['grid_optimizer'])
        start = int(ck['rollout'])
        grid_updates = int(ck['grid_updates'])
        if grid_updates != start*14:
            raise RuntimeError('Checkpoint Grid updates count is inconsistent')
        torch.set_rng_state(ck['torch_rng_state'].cpu())
        if device.type == 'cuda' and 'cuda_rng_state' in ck:
            torch.cuda.set_rng_state(ck['cuda_rng_state'].cpu(), device)
        print(f'RESUMED | rollout={start}, Grid updates={grid_updates}', flush=True)
    elif latest.exists() or metrics.exists():
        raise FileExistsError(f'Output exists: {out}; choose new out-dir or --resume')
    if start >= a.rollouts:
        print('Already reached target rollouts',flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)

    print(f'EXPERIMENT | A-Online fresh DMLab episodes | device={device}',flush=True)
    print(f'ENV | {STEPS} steps x {ACTION_REPEAT}/{FPS} sec = 90 s; '
          '10 episodes/rollout, 14 Grid updates/rollout',flush=True)
    print(f'SEEDS | train [{a.train_seed}, {a.train_seed+a.rollouts*BATCH-1}] '
          f'validation [{a.train_seed+VAL_SEED_OFFSET}, '
          f'{a.train_seed+VAL_SEED_OFFSET+a.val_episodes-1}]',flush=True)
    print(f'FROZEN VISION | sha256={pretrained_sha[:12]}, update={saved_vision.get("update")}',flush=True)
    print('RGB | kept in RAM for only the current rollout; no raw RGB files written',flush=True)
    print('WARNING | DMLab 10x1350 steps per rollout is expensive; test 1 rollout first',flush=True)

    def save(rollout_id: int):
        obj = dict(grid=grid.state_dict(), grid_optimizer=grid_opt.state_dict(),
                   vision=vision.state_dict(),vision_module=vision.state_dict(),
                   pc_stdev=(SIGMA,),env_size_m=2.5,
                   hd_concentration=Config().task.hdc_concentration,
                   pc_centers=pc.means.detach().cpu(),
                   hd_centers=hdc.means.detach().cpu(),
                   vision_mask_mode='whole',
                   rollout=rollout_id,grid_updates=grid_updates,vision_updates=0,
                   fingerprint=fingerprint,torch_rng_state=torch.get_rng_state().cpu())
        if device.type == 'cuda':
            obj['cuda_rng_state'] = torch.cuda.get_rng_state(device).cpu()
        base.atomic_torch_save(obj, latest)
        print(f'SAVED | rollout={rollout_id} | {latest}',flush=True)

    fields = ['rollout','fresh_train_episodes','grid_updates','train_grid_ce',
              'val_grid_ce','val_grid_pos_weighted_cm','val_grid_hd_deg',
              'vision_input_ratio','teleport_steps']
    env = make_dmlab_env(cfg.rl.env, renderer=a.renderer)
    try:
        yaw_gain, nominal_speed = collector.calibrate_controls(
            env, cfg.rl.env, a.train_seed+90_000_000, ACTION_REPEAT)
        with metrics.open('a' if a.resume else 'w', newline='',buffering=1) as fh:
            writer=csv.DictWriter(fh,fieldnames=fields)
            if not a.resume:
                writer.writeheader()
            for rollout_id in range(start+1, a.rollouts+1):
                # IDs are global unique train IDs even though only 10 are in RAM.
                ids = np.arange((rollout_id-1)*BATCH, rollout_id*BATCH,
                                dtype=np.int64)
                store, teleport_count = generate_episodes(
                    env, cfg.rl.env, ids, a.train_seed, yaw_gain, nominal_speed)
                positions, headings, motion, resets, visible = offline.make_motion(
                    store, ids, a.seed, rollout_id, a.vision_rate,
                    a.force_first_vision, a.speed_noise_std, a.angular_noise_std)
                pos = torch.from_numpy(positions).to(device)
                hd = torch.from_numpy(headings).to(device)
                vel = torch.from_numpy(motion).to(device)
                reset = torch.from_numpy(resets).to(device)
                visual = torch.zeros((BATCH,STEPS,VISION_DIMS),
                                     device=device,dtype=torch.float32)
                hidden = (torch.zeros(1,BATCH,128,device=device),
                          torch.zeros(1,BATCH,128,device=device))
                grid.train()
                total_ce=0.
                for s in range(0, STEPS, TBPTT):
                    e = min(s+TBPTT, STEPS)
                    offline.infer_vision_window(vision,store,ids,visible,s,e,visual,device)
                    hidden = (hidden[0].detach(), hidden[1].detach())
                    lp, lh, hidden = base.step_segment(
                        grid,vel[:,s:e],visual[:,s:e],
                        reset[:,s:e],hidden,training=True)
                    target_p,target_h=base.targets(pc,hdc,pos[:,s:e],hd[:,s:e])
                    ce = (-(target_p*F.log_softmax(lp,-1)).sum(-1)
                          -(target_h*F.log_softmax(lh,-1)).sum(-1)).mean()
                    if not torch.isfinite(ce):
                        raise FloatingPointError('Nonfinite loss in Grid training')
                    grid_opt.zero_grad(set_to_none=True)
                    (ce * ((e-s)/TBPTT)).backward()
                    torch.nn.utils.clip_grad_norm_(grid.parameters(),max_norm=1.0)
                    grid_opt.step()
                    grid_updates += 1
                    total_ce += float(ce.detach())*(e-s)
                train_ce = total_ce/STEPS
                ratio = float(visible.mean())
                del store, positions, headings, motion, resets, visible
                del pos, hd, vel, reset, visual, hidden
                if rollout_id % a.eval_every == 0 or rollout_id == a.rollouts:
                    report = validate_live(grid,vision,pc,hdc,env,cfg.rl.env,a,
                                           yaw_gain,nominal_speed,
                                           a.train_seed+VAL_SEED_OFFSET,device)
                    writer.writerow(dict(
                        rollout=rollout_id,fresh_train_episodes=rollout_id*BATCH,
                        grid_updates=grid_updates,train_grid_ce=train_ce,
                        val_grid_ce=report['ce'],
                        val_grid_pos_weighted_cm=report['pos_weighted_cm'],
                        val_grid_hd_deg=report['hd_circular_deg'],
                        vision_input_ratio=ratio,teleport_steps=teleport_count))
                    print(f'rollout={rollout_id}/{a.rollouts} | '
                          f'fresh_episodes={rollout_id*BATCH:,} | gridupd={grid_updates} | '
                          f'train/val CE={train_ce:.4f}/{report["ce"]:.4f} | '
                          f'val pos={report["pos_weighted_cm"]:.1f} cm, '
                          f'HD={report["hd_circular_deg"]:.1f} deg | '
                          f'vision={ratio:.1%} | teleports={teleport_count}',flush=True)
                elif rollout_id % 10 == 0 or rollout_id == 1:
                    print(f'rollout={rollout_id}/{a.rollouts} | '
                          f'fresh_episodes={rollout_id*BATCH:,} | gridupd={grid_updates} | '
                          f'train CE={train_ce:.4f} | vision={ratio:.1%}',flush=True)
                if rollout_id % a.save_every == 0 or rollout_id == a.rollouts:
                    save(rollout_id)
    finally:
        env.close()
    print(f'COMPLETE | {grid_updates:,} Grid updates; '
          f'{a.rollouts*BATCH:,} newly generated train episodes',flush=True)


if __name__ == '__main__':
    main()
