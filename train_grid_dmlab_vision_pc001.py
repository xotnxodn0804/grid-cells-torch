#!/usr/bin/env python3
"""Train Grid Network on the exact DMLab episodes used by the sigma=0.01 Vision Module.

Vision: frozen PC256+HD12 probabilities at randomly selected 5% of time points.
Motion: measured [speed_m_per_s, sin(delta_heading), cos(delta_heading)] every step.
Grid: new LSTM128 -> bottleneck512 -> PC256/HD12, supervised labels sigma=0.01.
Data: 2.5m square, 1300 samples/episode @ 4/60s, 900 train/100 val episodes.
Optimization: 100-step truncated BPTT; hidden state persists through 1300-step
rollout, and is reset only at episode boundaries and detected teleports.

Use --check-only and --rollouts 1 for an integration smoke test before long runs.
A rollout means one batch of full 1300-step episodes and 13 optimizer updates.
Neither the vision checkpoint nor any existing Grid/RL checkpoint is modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from grid_cells.ensembles import build_rl_ensembles
from grid_cells.model import GridCellsRNN
from grid_cells.vision import VisionModule
from sonnet_init_override import initialize_sonnet_lstm_

DT = 4.0 / 60.0
STEPS = 1300
SIGMA = 0.01
VISION_DIMS = 256 + 12
MOTION_DIMS = 3


def file_digest(p: Path):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EpisodeStore:
    """Memory-map previously collected DMLab data; never rewrite raw dataset."""
    FIELDS = ('images', 'pos', 'hd', 'ego_vel', 'valid_motion', 'episode_id')

    def __init__(self, dirs: list[Path], split: str):
        self.split = split
        self.entries = []
        self.lookup = []
        seed_ids = set()
        for root in dirs:
            marker = root / f'{split}_complete.json'
            settings = root / 'dataset_settings.json'
            if not marker.is_file() or not settings.is_file():
                raise FileNotFoundError(f'Missing completed dataset at {root} ({split})')
            meta = json.loads(marker.read_text(encoding='utf-8'))
            sett = json.loads(settings.read_text(encoding='utf-8'))
            n_ep = int(meta['episodes'])
            n_step = int(meta['steps_per_episode'])
            if n_step != STEPS or float(meta['env_size_m']) != 2.5 or int(sett['action_repeat']) != 4:
                raise ValueError(f'Data geometry/timing mismatch in {root}: {meta}')
            if (n_ep != int(sett['episodes']) - int(sett['val_episodes']) if split == 'train'
                    else n_ep != int(sett['val_episodes'])):
                raise ValueError(f'Wrong split count at {root}')
            arrays = {}
            for field in self.FIELDS:
                path = root / f'{split}_{field}.npy'
                if not path.is_file():
                    raise FileNotFoundError(path)
                arrays[field] = np.load(path, mmap_mode='r', allow_pickle=False)
                if len(arrays[field]) != STEPS * n_ep:
                    raise ValueError(f'Incorrect length: {path}')
            if arrays['images'].shape != (STEPS*n_ep, 3, 64, 64):
                raise ValueError(f'RGB shape mismatch in {root}')
            if arrays['pos'].shape != (STEPS*n_ep, 2) or arrays['hd'].shape != (STEPS*n_ep,):
                raise ValueError(f'Position/heading shape mismatch in {root}')
            if arrays['ego_vel'].shape != (STEPS*n_ep, 3) or arrays['valid_motion'].shape != (STEPS*n_ep,):
                raise ValueError(f'Motion shape mismatch in {root}')
            seeds = {int(meta['seed']) + i for i in range(n_ep)}
            if seeds & seed_ids:
                raise ValueError(f'Overlapping DMLab episode seeds in {root} / {split}')
            seed_ids.update(seeds)
            entry_id = len(self.entries)
            self.entries.append({'root': str(root), 'arrays': arrays, 'n_ep': n_ep,
                                 'first_seed': int(meta['seed'])})
            for ep in range(n_ep):
                first, last = ep*STEPS, (ep+1)*STEPS-1
                if int(arrays['episode_id'][first]) != ep or int(arrays['episode_id'][last]) != ep:
                    raise ValueError(f'episode_id mismatch: {root} {split} episode {ep}')
                self.lookup.append((entry_id, ep))
        self.seeds = seed_ids

    def __len__(self):
        return len(self.lookup)

    def episode(self, eid: int):
        entry_id, ep = self.lookup[eid]
        a = self.entries[entry_id]['arrays']
        offset = ep * STEPS
        sl = slice(offset, offset + STEPS)
        return (np.array(a['pos'][sl]), np.array(a['hd'][sl]),
                np.array(a['ego_vel'][sl]), np.array(a['valid_motion'][sl]).astype(bool))

    def images_at(self, eid: int, step_indices: np.ndarray):
        entry_id, ep = self.lookup[eid]
        a = self.entries[entry_id]['arrays']['images']
        return np.asarray(a[ep*STEPS + step_indices], dtype=np.uint8)


def make_ensembles_and_models(device, vision_ckpt: Path):
    cfg = Config()
    cfg.rl.pc_scale = (SIGMA,)  # Do not alter config.py or existing 0.4m model.
    if cfg.rl.env.env_size_m != 2.5 or cfg.task.n_pc != (256,) or cfg.task.n_hdc != (12,):
        raise ValueError('Unexpected model geometry; expected 2.5m / PC256 / HD12')
    pc_list, hd_list = build_rl_ensembles(cfg, device)
    pc, hd = pc_list[0], hd_list[0]
    obj = torch.load(vision_ckpt, map_location='cpu', weights_only=False)
    scale = obj.get('pc_stdev')
    if scale is None or len(scale) != 1 or abs(float(scale[0]) - SIGMA) > 1e-9:
        raise ValueError(f'WRONG VISION SIGMA: {scale}. Need sigma=0.01 checkpoint, not 0.4.')
    if float(obj.get('env_size_m', -1)) != 2.5:
        raise ValueError('Vision checkpoint arena is not 2.5m')
    if not torch.allclose(obj['pc_centers'].cpu(), pc.means.cpu(), atol=1e-7, rtol=0):
        raise ValueError('Vision PC center coordinates/order differ from Grid PC centers')
    if not torch.allclose(obj['hd_centers'].cpu(), hd.means.cpu(), atol=1e-7, rtol=0):
        raise ValueError('Vision HD center coordinates/order differ from Grid HD centers')
    vision = VisionModule(pc, hd, embed_dim=256, mask_prob=.95, mask_mode='whole').to(device)
    vision.load_state_dict(obj['vision_module'], strict=True)
    vision.eval()
    vision.requires_grad_(False)
    grid = GridCellsRNN(target_ensembles=[pc, hd], nh_lstm=128, nh_bottleneck=512,
                        dropout_rates=(.5,), bottleneck_has_bias=True,
                        ego_vel_dim=MOTION_DIMS, vision_dim=VISION_DIMS).to(device)
    initialize_sonnet_lstm_(grid.rl_lstm)  # match the original supervised baseline initialization style
    return grid, vision, pc, hd, int(obj.get('update', -1))


@dataclass
class Rollout:
    vel: torch.Tensor
    pos: torch.Tensor
    hd: torch.Tensor
    vis: torch.Tensor
    reset: torch.Tensor
    n_vis: int
    n_invalid: int


def build_rollout(store: EpisodeStore, episode_ids: np.ndarray, vision: VisionModule,
                  device: torch.device, rng: np.random.Generator, vision_rate: float,
                  force_first: bool, deterministic_val: bool = False, val_seed: int = 0) -> Rollout:
    b = len(episode_ids)
    positions = np.empty((b, STEPS, 2), np.float32)
    headings = np.empty((b, STEPS), np.float32)
    motion = np.empty((b, STEPS, 3), np.float32)
    invalid = np.zeros((b, STEPS), dtype=np.bool_)
    visible = np.zeros((b, STEPS), dtype=np.bool_)
    for i, eid in enumerate(episode_ids):
        pos, hd, vel, valid = store.episode(int(eid))
        positions[i], headings[i], motion[i] = pos, hd, vel
        invalid[i] = ~valid
        mrng = (np.random.default_rng(np.random.SeedSequence([val_seed, int(eid), 9173]))
                if deterministic_val else rng)
        visible[i] = mrng.random(STEPS) < vision_rate
        if force_first:
            visible[i, 0] = True
        # A teleport changes actual position without faithful odometry. Reset & relocalize.
        visible[i, invalid[i]] = True
    n_visual = int(visible.sum())
    vision_inputs = torch.zeros((b, STEPS, VISION_DIMS), device=device, dtype=torch.float32)
    with torch.no_grad():
        for i, eid in enumerate(episode_ids):
            steps = np.flatnonzero(visible[i]).astype(np.int64)
            for start in range(0, len(steps), 128):
                s = steps[start:start+128]
                img = np.array(store.images_at(int(eid), s), copy=True)
                timg = torch.from_numpy(img).to(device).float().div_(127.5).sub_(1.0)
                emb = vision.cnn(timg)
                out = torch.cat([F.softmax(vision.place_head(emb), -1),
                                 F.softmax(vision.hd_head(emb), -1)], dim=-1)
                vision_inputs[i, s] = out
    return Rollout(
        torch.from_numpy(motion).to(device),
        torch.from_numpy(positions).to(device),
        torch.from_numpy(headings).to(device),
        vision_inputs,
        torch.from_numpy(invalid).to(device), n_visual, int(invalid.sum()))


def step_segment(grid: GridCellsRNN, vel, vis, reset, hidden, training: bool):
    """Forward chunk; preserve RNN state except when motion discontinuity is detected."""
    x = torch.cat([vel, vis], dim=-1)
    t = x.shape[1]
    breakpoints = torch.nonzero(reset.any(dim=0), as_tuple=False).flatten().tolist()
    pieces = []
    start = 0
    for k in breakpoints:
        if k > start:
            out, hidden = grid.rl_lstm(x[:, start:k], hidden)
            pieces.append(out)
        keep = (~reset[:, k]).to(x.dtype).view(1, -1, 1)
        hidden = (hidden[0] * keep, hidden[1] * keep)
        out, hidden = grid.rl_lstm(x[:, k:k+1], hidden)
        pieces.append(out)
        start = k+1
    if start < t:
        out, hidden = grid.rl_lstm(x[:, start:], hidden)
        pieces.append(out)
    h = torch.cat(pieces, dim=1)
    bottleneck = grid.bottleneck(h)
    if training:
        bottleneck = F.dropout(bottleneck, p=.5, training=True)
    return grid.output_heads[0](bottleneck), grid.output_heads[1](bottleneck), hidden


def targets(pc, hdc, pos, hd):
    return pc.posterior(pos), hdc.posterior(hd.unsqueeze(-1))


def circular_error_deg(pred, target):
    return torch.rad2deg(torch.atan2(torch.sin(pred-target), torch.cos(pred-target)).abs())


def prediction_errors(pc, hdc, pc_logits, hd_logits, pos, hd):
    p = F.softmax(pc_logits, -1)
    top3 = pc.decode_position(p, 'top3')
    mean = pc.decode_position(p, 'weighted_mean')
    hard = pc.decode_position(p, 'argmax')
    q = F.softmax(hd_logits, -1)
    pred_hd = torch.atan2(q @ torch.sin(hdc.means), q @ torch.cos(hdc.means))
    return (100 * torch.linalg.vector_norm(top3-pos, dim=-1),
            100 * torch.linalg.vector_norm(mean-pos, dim=-1),
            100 * torch.linalg.vector_norm(hard-pos, dim=-1),
            circular_error_deg(pred_hd, hd))


@torch.no_grad()
def validate(grid, vision, pc, hdc, store, val_ids, device, args):
    grid.eval()
    sums = np.zeros(5, np.float64)
    final = np.zeros(2, np.float64)
    steps_seen = 0
    for left in range(0, len(val_ids), args.val_batch_size):
        ids = val_ids[left:left+args.val_batch_size]
        rollout = build_rollout(store, ids, vision, device, np.random.default_rng(0),
                                args.vision_rate, args.force_first_vision,
                                deterministic_val=True, val_seed=args.seed+83817)
        b = len(ids)
        hidden = (torch.zeros(1, b, 128, device=device),
                  torch.zeros(1, b, 128, device=device))
        for s in range(0, STEPS, args.tbptt):
            e = min(STEPS, s+args.tbptt)
            a, d, hidden = step_segment(grid, rollout.vel[:,s:e], rollout.vis[:,s:e],
                                         rollout.reset[:,s:e], hidden, training=False)
            pc_t, hd_t = targets(pc, hdc, rollout.pos[:,s:e], rollout.hd[:,s:e])
            ce = (-(pc_t*F.log_softmax(a, -1)).sum(-1)
                  -(hd_t*F.log_softmax(d, -1)).sum(-1))
            errors = prediction_errors(pc, hdc, a, d,
                                       rollout.pos[:,s:e], rollout.hd[:,s:e])
            for k, err in enumerate(errors):
                sums[k] += err.sum().item()
            sums[4] += ce.sum().item()
            if e == STEPS:
                final[0] += errors[0][:,-1].sum().item()
                final[1] += errors[3][:,-1].sum().item()
            steps_seen += b*(e-s)
    grid.train()
    n = len(val_ids)
    return dict(pos_top3_cm=sums[0]/steps_seen, pos_weighted_cm=sums[1]/steps_seen,
                pos_argmax_cm=sums[2]/steps_seen, hd_circular_deg=sums[3]/steps_seen,
                ce=sums[4]/steps_seen, final_pos_top3_cm=final[0]/n,
                final_hd_deg=final[1]/n)


def atomic_torch_save(obj, path):
    temp = path.with_name(path.name + '.tmp')
    torch.save(obj, temp)
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data-dirs', nargs='+', type=Path, required=True)
    p.add_argument('--vision-checkpoint', type=Path,
                   default=Path('data/checkpoints/vision_rh_pc001_1000/latest.pt'))
    p.add_argument('--out-dir', type=Path,
                   default=Path('data/checkpoints/grid_dmlab_vision_pc001_512_run1'))
    p.add_argument('--rollouts', type=int, default=1000,
                   help='Target FULL 1300-step episode batches; each entails 13 TBPTT optimizer updates')
    p.add_argument('--batch-size', type=int, default=8,
                   help='Number of full episodes per training rollout')
    p.add_argument('--tbptt', type=int, default=100)
    p.add_argument('--vision-rate', type=float, default=.05)
    p.add_argument('--force-first-vision', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--val-episodes', type=int, default=16)
    p.add_argument('--val-batch-size', type=int, default=8)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--save-every', type=int, default=25)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if args.rollouts < 1 or args.batch_size < 1 or args.tbptt < 1 or args.val_batch_size < 1:
        p.error('rollouts, batch-size, tbptt, val-batch-size must be positive')
    if args.eval_every < 1 or args.save_every < 1 or args.val_episodes < 1:
        p.error('eval-every, save-every, val-episodes must be positive')
    if not (0 < args.vision_rate < 1) or not (0 < args.lr < 1):
        p.error('vision-rate and lr must be in (0,1)')
    if args.tbptt > STEPS:
        p.error('tbptt cannot exceed episode steps (1300)')
    folders = [x.expanduser().resolve() for x in args.data_dirs]
    if len(set(folders)) != len(folders):
        p.error('Duplicate dataset folders')
    if not args.vision_checkpoint.is_file():
        raise FileNotFoundError(f'Missing vision checkpoint: {args.vision_checkpoint}')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available())
                          else 'cpu' if args.device == 'auto' else args.device)
    train = EpisodeStore(folders, 'train')
    val = EpisodeStore(folders, 'val')
    if train.seeds & val.seeds:
        raise ValueError('Train and validation DMLab seeds overlap')
    if args.val_episodes > len(val):
        raise ValueError(f'val-episodes={args.val_episodes} but only {len(val)} available')
    print(f'DATA OK | train={len(train)} eps, val={len(val)} eps; 1300 steps/ep, dt={DT:.6f}s, 2.5m arena', flush=True)
    print('MOVEMENT = saved actual DMLab ego_vel; not fresh Rayleigh samples', flush=True)
    print('VISION = frozen PC256+HD12 predictions, whole-time-step masking, 5% + first/teleport anchors', flush=True)
    seed_everything(args.seed)
    grid, vision, pc, hd, vision_update = make_ensembles_and_models(device, args.vision_checkpoint)
    vision_sha = file_digest(args.vision_checkpoint)
    print(f'VISION OK | sigma={SIGMA}m PC centers and HD centers EXACTLY match; update={vision_update:,}, sha256={vision_sha[:14]}', flush=True)
    print(f'GRID OK | input={MOTION_DIMS}+{VISION_DIMS}={MOTION_DIMS+VISION_DIMS}, LSTM128 Sonnet-init, bottleneck512', flush=True)
    if args.check_only:
        ids = np.array([0], np.int64)
        small = build_rollout(train, ids, vision, device, np.random.default_rng(args.seed),
                              args.vision_rate, args.force_first_vision)
        with torch.no_grad():
            hidden = (torch.zeros(1, 1, 128, device=device), torch.zeros(1, 1, 128, device=device))
            aa, dd, _ = step_segment(grid, small.vel[:,:4], small.vis[:,:4],
                                     small.reset[:,:4], hidden, training=False)
            if not (torch.isfinite(aa).all() and torch.isfinite(dd).all()):
                raise ValueError('Non-finite model outputs')
        print('CHECK OK | real RGB -> frozen vision -> masked PC/HD -> Grid forward -> finite logits', flush=True)
        return
    optimizer = torch.optim.RMSprop(grid.parameters(), lr=args.lr, alpha=.9, momentum=.9,
                                    eps=1e-10, weight_decay=args.weight_decay)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    latest = out/'latest.pt'
    csv_path = out/'metrics.csv'
    val_rng = np.random.default_rng(args.seed + 129731)
    val_ids = np.sort(val_rng.choice(len(val), size=args.val_episodes, replace=False))
    fingerprint = dict(script_version='grid_vision_dmlab_20261112_v1',
                       data_dirs=[str(x) for x in folders], train_episodes=len(train),
                       val_episodes=len(val), vision_sha=vision_sha,
                       pc_sigma=SIGMA, dt=DT, episode_steps=STEPS,
                       batch_size=args.batch_size, tbptt=args.tbptt,
                       vision_rate=args.vision_rate, force_first=args.force_first_vision,
                       lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
                       val_ids=val_ids.tolist())
    train_rng = np.random.default_rng(args.seed+481)
    start = 0
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError(f'Cannot resume: {latest}')
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint['fingerprint'] != fingerprint:
            raise ValueError('Different model/data/vision/settings. Use a new out-dir.')
        grid.load_state_dict(checkpoint['grid'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = int(checkpoint['rollout'])
        train_rng.bit_generator.state = checkpoint['train_rng_state']
        torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
        if device.type=='cuda' and 'cuda_rng_state' in checkpoint:
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu(), device)
        print(f'RESUMED at rollout={start}, optimizer_updates={start * ((STEPS+args.tbptt-1)//args.tbptt)}', flush=True)
    elif latest.exists() or csv_path.exists():
        raise FileExistsError(f'Existing training experiment in {out}; use --resume or a new out-dir')
    if start >= args.rollouts:
        print(f'Already reached rollout {start}; requested {args.rollouts}.')
        return
    fields=['rollout','optimizer_updates','train_ce','val_ce','val_pos_top3_cm',
            'val_pos_weighted_cm','val_pos_argmax_cm','val_hd_deg',
            'val_final_pos_top3_cm','val_final_hd_deg','train_vision_rate','teleports']

    def save(rollout, filename='latest.pt'):
        obj = dict(grid=grid.state_dict(), optimizer=optimizer.state_dict(),
                   rollout=rollout, optimizer_updates=rollout*((STEPS+args.tbptt-1)//args.tbptt),
                   fingerprint=fingerprint, pc_centers=pc.means.detach().cpu(),
                   hd_centers=hd.means.detach().cpu(), vision_checkpoint=str(args.vision_checkpoint),
                   vision_update=vision_update, train_rng_state=train_rng.bit_generator.state,
                   torch_rng_state=torch.get_rng_state().cpu())
        if device.type == 'cuda':
            obj['cuda_rng_state'] = torch.cuda.get_rng_state(device).cpu()
        atomic_torch_save(obj, out/filename)

    with open(csv_path, 'a' if args.resume else 'w', newline='', buffering=1) as file:
        writer=csv.DictWriter(file, fieldnames=fields)
        if not args.resume:
            writer.writeheader()
        try:
            for rollout_id in range(start+1, args.rollouts+1):
                grid.train()
                episodes = train_rng.integers(0, len(train), size=args.batch_size)
                batch = build_rollout(train, episodes, vision, device, train_rng,
                                      args.vision_rate, args.force_first_vision)
                b=args.batch_size
                hidden=(torch.zeros(1,b,128, device=device), torch.zeros(1,b,128, device=device))
                train_ce_sum=0.0
                for s in range(0, STEPS, args.tbptt):
                    e=min(STEPS, s+args.tbptt)
                    # Truncate between windows, preserving state values, not backprop graph.
                    hidden=(hidden[0].detach(),hidden[1].detach())
                    a,d,hidden=step_segment(grid,batch.vel[:,s:e],batch.vis[:,s:e],
                                             batch.reset[:,s:e],hidden,training=True)
                    pc_t, hd_t = targets(pc,hd,batch.pos[:,s:e],batch.hd[:,s:e])
                    loss = (-(pc_t*F.log_softmax(a,-1)).sum(-1)
                            -(hd_t*F.log_softmax(d,-1)).sum(-1)).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(grid.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_ce_sum += float(loss.detach())*(e-s)
                train_ce=train_ce_sum/STEPS
                if rollout_id % args.eval_every == 0 or rollout_id == args.rollouts:
                    report=validate(grid,vision,pc,hd,val,val_ids,device,args)
                    row=dict(rollout=rollout_id,
                             optimizer_updates=rollout_id*((STEPS+args.tbptt-1)//args.tbptt),
                             train_ce=train_ce, val_ce=report['ce'],
                             val_pos_top3_cm=report['pos_top3_cm'],
                             val_pos_weighted_cm=report['pos_weighted_cm'],
                             val_pos_argmax_cm=report['pos_argmax_cm'],
                             val_hd_deg=report['hd_circular_deg'],
                             val_final_pos_top3_cm=report['final_pos_top3_cm'],
                             val_final_hd_deg=report['final_hd_deg'],
                             train_vision_rate=batch.n_vis/(b*STEPS),teleports=batch.n_invalid)
                    writer.writerow(row)
                    print(f'rollout {rollout_id:,}/{args.rollouts:,} | optim {row["optimizer_updates"]:,} | '
                          f'train/val CE {train_ce:.4f}/{report["ce"]:.4f} | '
                          f'val pos(top3) {report["pos_top3_cm"]:.1f}cm '
                          f'final {report["final_pos_top3_cm"]:.1f}cm | '
                          f'HD {report["hd_circular_deg"]:.1f}deg | '
                          f'vis {row["train_vision_rate"]:.1%} | jumps {batch.n_invalid}',flush=True)
                if rollout_id % args.save_every == 0 or rollout_id == args.rollouts:
                    save(rollout_id)
                    print(f'SAVED rollout {rollout_id}: {latest}',flush=True)
        except KeyboardInterrupt:
            print('INTERRUPTED. Resume from the last COMPLETED saved rollout with --resume.', flush=True)
            raise
    print('RESULTS:',out.resolve(),flush=True)


if __name__ == '__main__':
    main()
