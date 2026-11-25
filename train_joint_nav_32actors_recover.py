#!/usr/bin/env python3
"""EXPERIMENTAL / UNVERIFIED: 32-actor Joint Vision + Grid + synchronous A2C.

Separate, fresh-training implementation. It NEVER loads a 1-actor Joint checkpoint.
Paper-inspired A2C (NOT the original paper's asynchronous A3C). Only an initial
supervised B-Grid/Vision checkpoint, if explicitly supplied, may be used.

Frame accounting: 1 action = 4 DMLab frames; 1 update = n_envs * n_step actions.
Each actor has its own environment, episode, recurrent states, goal memory and
chronological replay ring; model weights and optimizer are shared.

Caution: training is unverified on user's GPU/DeepMind Lab installation.
Run --mode self-test, --mode check and a NEW --mode train smoke run first.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import os
from pathlib import Path
import random
import signal
import time
import sys

# Run from WSL repo even if this script is kept in the Windows project folder.
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import torch
import torch.nn.functional as F

GRID_DIM, VIS_DIM, N_ACTIONS = 512, 268, 6
ALGORITHM = 'joint_vision_grid_sync_a2c_32actors_v1_fresh'


def digest(path: Path | None):
    if path is None:
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name):
    dev = torch.device('cuda' if name == 'auto' and torch.cuda.is_available()
                       else ('cpu' if name == 'auto' else name))
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; try --device cpu')
    return dev


def assert_vector_shapes(obs, nenv):
    required = {'image': (nenv, 3, 64, 64), 'pos': (nenv, 2)}
    for field, shape in required.items():
        if tuple(obs[field].shape) != shape:
            raise RuntimeError(f'{field}: expected {shape}, got {obs[field].shape}')
    if 'oracle_goal_xy' in obs:
        raise RuntimeError('Privileged ORACLE.GOAL.XY must not enter training observations')


def drop_oracle(obs):
    # Existing Vec32 worker returns ORACLE.GOAL.XY for the old Oracle experiment.
    # Remove it immediately so no network path can accidentally see the target.
    if 'oracle_goal_xy' in obs:
        obs = {k: v for k, v in obs.items() if k != 'oracle_goal_xy'}
    return obs


class IndependentReplay:
    """One disk ring per actor: sequences can NEVER bridge actor trajectories."""
    def __init__(self, base: Path, nenv: int, capacity: int, *, create: bool):
        from joint_nav_replay import DiskRingReplay
        self.nenv = nenv
        self.capacity = capacity
        self.rings = [DiskRingReplay(base / f'actor_{i:02d}', capacity, create=create)
                      for i in range(nenv)]

    def load_count(self, actions_each: int):
        for ring in self.rings:
            ring.load_count(actions_each)

    def add(self, actor: int, **record):
        self.rings[actor].add(**record)

    def sample_vision(self, batch: int, rng: np.random.Generator):
        eligible = [i for i, r in enumerate(self.rings) if r.size >= 1]
        if not eligible:
            raise RuntimeError('No Vision examples in replay')
        chosen = rng.choice(eligible, size=batch, replace=len(eligible) < batch)
        pieces = []
        for i in chosen:
            ring = self.rings[int(i)]
            k = int(rng.integers(ring.total - ring.size, ring.total))
            pieces.append(ring.read(np.array([k], dtype=np.int64)))
        return {key: np.concatenate([p[key] for p in pieces], axis=0)
                for key in pieces[0]}

    def sample_grid(self, paths: int, length: int, rng: np.random.Generator):
        eligible = [i for i, r in enumerate(self.rings) if r.size >= length]
        if not eligible:
            raise RuntimeError(f'No actor with >= {length} consecutive records')
        chosen = rng.choice(eligible, size=paths, replace=len(eligible) < paths)
        pieces = []
        for i in chosen:
            ring = self.rings[int(i)]
            start = int(rng.integers(ring.total - ring.size, ring.total - length + 1))
            pieces.append(ring.read(np.arange(start, start + length, dtype=np.int64)))
        return {key: np.stack([p[key] for p in pieces], axis=0)
                for key in pieces[0]}

    def flush(self):
        for ring in self.rings:
            ring.flush()


def _sigint_safe_dmlab_worker(pipe, cfg, renderer, rank, nenv, seed):
    """Worker ignores terminal Ctrl+C; parent saves after the full rollout."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from train_oracle_place_agent import worker
    return worker(pipe, cfg, renderer, rank, nenv, seed)


class SafeVec32:
    """Identical Vec32 IPC, with SIGINT ignored inside DMLab children."""
    def __init__(self, cfg, renderer, nenv, seed):
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        self.nenv = nenv
        self.pipes, self.procs = [], []
        for rank in range(nenv):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_sigint_safe_dmlab_worker,
                               args=(child, cfg, renderer, rank, nenv, seed), daemon=True)
            proc.start()
            child.close()
            self.pipes.append(parent)
            self.procs.append(proc)

    def reset(self):
        from train_oracle_place_agent import stack
        for p in self.pipes:
            p.send(('reset', None))
        return stack([p.recv() for p in self.pipes])

    def step(self, actions):
        from train_oracle_place_agent import stack
        for p, action in zip(self.pipes, actions):
            p.send(('step', int(action)))
        out = [p.recv() for p in self.pipes]
        obs = stack([r[0] for r in out])
        rewards = np.asarray([r[1] for r in out], dtype=np.float32)
        done = np.asarray([r[2] for r in out], dtype=np.bool_)
        return obs, rewards, done

    def close(self):
        for p in self.pipes:
            try: p.send(('close', None))
            except (BrokenPipeError, EOFError, OSError): pass
        for p in self.pipes:
            try: p.recv()
            except (BrokenPipeError, EOFError, OSError): pass
        for p in self.procs:
            p.join(timeout=3)
            if p.is_alive(): p.terminate()


def reset_hidden(hidden, mask):
    """Reset only selected actors, leaving other LSTM states untouched."""
    keep = (~mask).to(hidden[0].device).float()[None, :, None]
    return tuple(v * keep for v in hidden)


def batched_grid_step(grid, vision, obs, motion, visual_mask, hidden, resets, device):
    """Detach spatial module: A2C never changes Vision/Grid via policy loss."""
    from train_joint_nav_online import vision_cues
    n = len(resets)
    hidden = reset_hidden(hidden, torch.as_tensor(resets, device=device, dtype=torch.bool))
    cue = torch.zeros(n, VIS_DIM, device=device)
    seen = np.flatnonzero(visual_mask)
    with torch.no_grad():
        if len(seen):
            im = torch.from_numpy(np.ascontiguousarray(obs['image'][seen])).to(device)
            cue[seen] = vision_cues(vision, im, device)
        code, hidden = grid.step(torch.from_numpy(np.asarray(motion, dtype=np.float32)).to(device),
                                 cue, hidden)
    if code.shape != (n, GRID_DIM) or not bool(torch.isfinite(code).all()):
        raise RuntimeError('Invalid batch Grid code')
    return code.detach(), tuple(x.detach() for x in hidden)


def batched_motions(obs, last_obs, resets, rng):
    from train_joint_nav_online import noisy_motion
    n = len(resets)
    vectors = []
    for i in range(n):
        one = {k: obs[k][i] for k in ('pos', 'hd', 'vel_trans', 'vel_rot')}
        prev = None if last_obs is None else {
            k: last_obs[k][i] for k in ('pos', 'hd', 'vel_trans', 'vel_rot')}
        vectors.append(noisy_motion(one, prev, bool(resets[i]), rng))
    return np.stack(vectors, axis=0)


def actor_hidden(n, device):
    zero = torch.zeros(1, n, 256, device=device)
    return zero, zero.clone()


def grid_hidden_init(n, device):
    zero = torch.zeros(1, n, 128, device=device)
    return zero, zero.clone()


def atomic_save(state, path):
    tmp = path.with_suffix('.tmp.pt')
    torch.save(state, tmp)
    os.replace(tmp, path)


def self_test():
    # 32 separate recurrent states; forcing reset must not touch other actors.
    n = 32
    states = (torch.arange(n, dtype=torch.float32).reshape(1, n, 1).repeat(1, 1, 4),
              torch.ones(1, n, 4))
    mask = torch.zeros(n, dtype=torch.bool)
    mask[3], mask[17] = True, True
    out = reset_hidden(states, mask)
    assert out[0][0, 3].sum().item() == 0
    assert out[0][0, 17].sum().item() == 0
    assert torch.equal(out[0][0, 2], states[0][0, 2])
    assert torch.equal(out[0][0, 18], states[0][0, 18])
    # Batch returns match hand calculation and must be independent per actor.
    from train_joint_nav_online import compute_returns
    rewards = [torch.ones(n), torch.zeros(n)]
    done = [torch.zeros(n), torch.zeros(n)]
    done[0][5] = 1
    vals = compute_returns(rewards, done, torch.ones(n), .5)
    assert vals.shape == (2, n)
    assert abs(vals[0, 5].item() - 1) < 1e-7
    assert abs(vals[0, 4].item() - 1.25) < 1e-7
    assert 32 * 100 * 4 == 12_800
    # An episode's privileged oracle coordinates must be dropped.
    assert 'oracle_goal_xy' not in drop_oracle({'pos': 1, 'oracle_goal_xy': 2})
    print('SELF-TEST PASS | 32 actors: reset separation, return masks, oracle isolation, frame accounting')


def check(args, device):
    from train_joint_nav_online import GridCodeActorCritic, make_models
    if args.resume:
        raise RuntimeError('check cannot resume')
    cfg, pc, hd, grid, vision = make_models(device, args.init_checkpoint)
    policy = GridCodeActorCritic().to(device)
    n = args.num_envs
    with torch.no_grad():
        code = torch.randn(n, GRID_DIM, device=device)
        goal = torch.zeros_like(code)
        pa = torch.zeros(n, dtype=torch.long, device=device)
        pr = torch.zeros(n, 1, device=device)
        logits, value, h = policy.step(code, goal, pa, pr, actor_hidden(n, device))
        assert logits.shape == (n, N_ACTIONS) and value.shape == (n,)
    print(f'PREFLIGHT PASS | {n} independent actors; Vision 268D, Grid 512D, A2C 6 actions')
    print(f'INITIALIZATION | Grid/Vision={"supervised B only" if args.init_checkpoint else "random"}; Actor=random')
    print('NO DMLab launched; NO existing training checkpoints touched')
    self_test()


def train(args, device):
    from train_joint_nav_online import (make_models, GridCodeActorCritic,
                                        train_vision, train_grid, finish_actor_rollout)
    Vec32 = SafeVec32  # keep all children alive through parent Ctrl+C

    cfg, pc, hd, grid, vision = make_models(device, args.init_checkpoint)
    env_cfg = cfg.rl.env
    env_cfg.level = 'square_arena_oracle'
    env_cfg.goal_visible = False
    env_cfg.action_repeat = 4
    if (env_cfg.env_size_m, env_cfg.width, env_cfg.height) != (2.5, 64, 64):
        raise RuntimeError('Environment must be 2.5m square with 64x64 images')
    if not Path(env_cfg.level_directory, 'square_arena_oracle.lua').is_file():
        raise RuntimeError('Missing levels/square_arena_oracle.lua')

    n = args.num_envs
    per_batch = n * args.n_step * env_cfg.action_repeat
    if args.total_env_frames % per_batch:
        raise ValueError(f'--total-env-frames must be a multiple of {per_batch} '
                         f'({n} actors x {args.n_step} actions x 4 frames)')
    if args.warmup_per_actor % args.n_step:
        raise ValueError('warmup-per-actor must be a multiple of n-step')
    if args.total_env_frames <= n * args.warmup_per_actor * 4:
        raise ValueError('Need at least one actor rollout after warmup')

    outdir = args.out_dir.expanduser().resolve()
    if outdir == Path('data/checkpoints/joint_nav_50m_from21m').resolve():
        raise RuntimeError('Refusing to overwrite the existing 28.53M model')
    if args.resume:
        if not (outdir / 'latest.pt').is_file():
            raise FileNotFoundError('No 32-actor checkpoint in --out-dir')
    else:
        if outdir.exists() and any(outdir.iterdir()):
            raise FileExistsError(f'Output directory is not empty: {outdir}')
        outdir.mkdir(parents=True, exist_ok=True)

    grid.eval(); vision.eval()
    policy = GridCodeActorCritic().to(device)
    grid_opt = torch.optim.Adam(grid.parameters(), lr=args.grid_lr, weight_decay=1e-5)
    vision_opt = torch.optim.Adam(vision.parameters(), lr=args.vision_lr)
    actor_opt = torch.optim.RMSprop(policy.parameters(), lr=args.actor_lr,
                                    alpha=.99, eps=1e-8, momentum=0.)
    rng = np.random.default_rng(args.seed + 10022)
    init_hash = digest(args.init_checkpoint)
    decisions, actor_updates, vision_updates, grid_updates, episodes = 0, 0, 0, 0, 0
    ckpt_path = outdir / 'latest.pt'

    if args.resume:
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if ck.get('algorithm') != ALGORITHM:
            raise RuntimeError('Not a fresh 32-actor Joint checkpoint; refusing resume')
        locked = ['num_envs', 'n_step', 'warmup_per_actor', 'replay_per_actor',
                  'vision_updates_per_rollout', 'grid_updates_per_rollout', 'vision_rate',
                  'seed', 'actor_lr', 'grid_lr', 'vision_lr', 'gamma', 'value_cost', 'entropy_cost']
        for key in locked:
            if ck['args'][key] != vars(args)[key]:
                raise RuntimeError(f'Resume mismatch: {key}; preserve old settings')
        if ck['initial_sha256'] != init_hash:
            raise RuntimeError('Initial supervised checkpoint SHA256 mismatch')
        grid.load_state_dict(ck['grid']); vision.load_state_dict(ck['vision'])
        policy.load_state_dict(ck['actor'])
        grid_opt.load_state_dict(ck['grid_opt']); vision_opt.load_state_dict(ck['vision_opt'])
        actor_opt.load_state_dict(ck['actor_opt'])
        decisions, actor_updates = int(ck['decisions_per_actor']), int(ck['actor_updates'])
        if args.total_env_frames <= decisions*n*4:
            raise RuntimeError('Resume target must be greater than completed frames')
        vision_updates, grid_updates, episodes = (int(ck['vision_updates']),
                                                 int(ck['grid_updates']), int(ck['episodes']))
        rng.bit_generator.state = ck['sample_rng']
        torch.set_rng_state(ck['torch_rng_state'].cpu())
        if device.type == 'cuda' and ck.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state(ck['cuda_rng_state'].cpu(), device=device)
        replay_subdir = ck.get('replay_subdir', 'replay')
        replay_count = int(ck.get('replay_decisions_per_actor', decisions))
        metrics_name = ck.get('metrics_name', 'metrics_32actors.csv')
        if args.reset_replay_on_resume:
            # The 80M checkpoint is older than 83.47M of replay writes.
            # Reconstruct optimizer/model state at 80M but never relabel stale samples.
            if replay_subdir != 'replay':
                raise RuntimeError('This checkpoint already uses a recovered replay. '
                                   'Run --resume WITHOUT --reset-replay-on-resume.')
            replay_subdir = f'replay_recovered_from_{decisions*n*4:012d}'
            if (outdir / replay_subdir).exists():
                raise FileExistsError('Recovery replay directory already exists; preserve it '
                                      'and inspect latest.pt before retrying.')
            replay = IndependentReplay(outdir/replay_subdir, n, args.replay_per_actor, create=True)
            metrics_name = f'metrics_recovered_from_{decisions*n*4:012d}.csv'
            print(f'RECOVERY | model/optimizers at {decisions*n*4:,} frames; '
                  f'NEW replay={replay_subdir} (0 records) | previous replay untouched',flush=True)
            print(f'RECOVERY | new metrics={metrics_name}; historical logs untouched',flush=True)
        else:
            replay = IndependentReplay(outdir/replay_subdir, n, args.replay_per_actor, create=False)
            replay.load_count(replay_count)
            print(f'REPLAY RESUME | {replay_subdir} verified at {replay_count:,} records/actor',flush=True)
        print('RESUMED | actor weights and optimizers; ALL environments restart fresh episodes', flush=True)
    else:
        replay_subdir = 'replay'
        metrics_name = 'metrics_32actors.csv'
        replay = IndependentReplay(outdir/replay_subdir, n, args.replay_per_actor, create=True)

    def save():
        replay.flush()
        checkpoint = {
            'algorithm': ALGORITHM, 'kind': '32_actor_shared_weights',
            'frames': decisions*n*4, 'actions': decisions*n,
            'decisions_per_actor': decisions, 'num_envs': n,
            'actor_updates': actor_updates, 'vision_updates': vision_updates,
            'grid_updates': grid_updates, 'episodes': episodes,
            'replay_subdir': replay_subdir,
            'replay_decisions_per_actor': replay.rings[0].total,
            'metrics_name': metrics_name,
            'initial_sha256': init_hash, 'args': vars(args),
            'grid': grid.state_dict(), 'vision': vision.state_dict(), 'actor': policy.state_dict(),
            'grid_opt': grid_opt.state_dict(), 'vision_opt': vision_opt.state_dict(),
            'actor_opt': actor_opt.state_dict(), 'sample_rng': rng.bit_generator.state,
            'torch_rng_state': torch.get_rng_state().cpu(),
            'cuda_rng_state': torch.cuda.get_rng_state(device).cpu() if device.type=='cuda' else None,
        }
        atomic_save(checkpoint, ckpt_path)
        print(f'SAVED | frames={decisions*n*4:,} actors={n} actor_updates={actor_updates:,}',flush=True)

    def snapshot():
        # Evaluation-only snapshot. Replay+optimizers are intentionally NOT snapshotted.
        fname = outdir / f'weights_only_{decisions*n*4:012d}_frames.pt'
        atomic_save({'algorithm': ALGORITHM, 'frames': decisions*n*4,
                     'num_envs': n, 'grid': grid.state_dict(),
                     'vision': vision.state_dict(), 'actor': policy.state_dict()}, fname)
        print(f'WEIGHTS SNAPSHOT | {fname}', flush=True)

    # Commit recovery metadata at 80M immediately, before launching DMLab.
    # A separate user-made copy of the original latest.pt is recommended.
    if args.resume and args.reset_replay_on_resume:
        save()
        print('RECOVERY COMMITTED | latest.pt now points to NEW empty replay', flush=True)

    # Ctrl-C requests a SAFE stop after the current complete n-step batch.
    # This prevents replay from being newer than latest.pt after a partial batch.
    stopping = False
    def request_stop(signum, _frame):
        nonlocal stopping
        stopping = True
        print(f'SIGNAL {signum} | completing current {args.n_step}-step rollout before save', flush=True)
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    vec = None
    try:
        # Vec32 is the project's existing tested 32-process DMLab launcher.
        vec = Vec32(env_cfg, args.renderer, n, args.seed + 1000 + episodes*n)
        obs = drop_oracle(vec.reset())
        assert_vector_shapes(obs, n)
        last_obs = None
        force_reset = np.ones(n, dtype=np.bool_)
        grid_h = grid_hidden_init(n, device)
        actor_h = actor_hidden(n, device)
        prev_a = torch.zeros(n, dtype=torch.long, device=device)
        prev_r = torch.zeros(n, 1, device=device)
        goal_mem = torch.zeros(n, GRID_DIM, device=device)
        goal_known = torch.zeros(n, dtype=torch.bool, device=device)
        episode_index = np.zeros(n, dtype=np.int64)
        episode_hits = np.zeros(n, dtype=np.int64)
        total_goals_since_log = 0
        last_vision_loss = last_grid_loss = last_actor_loss = float('nan')
        metrics = outdir / metrics_name
        with metrics.open('a' if args.resume and not args.reset_replay_on_resume else 'w', newline='', buffering=1) as logf:
            csvout = csv.writer(logf)
            if not args.resume or args.reset_replay_on_resume:
                csvout.writerow(['frames','total_actions','num_actors','actor_updates',
                                 'vision_updates','grid_updates','vision_ce','grid_ce',
                                 'a2c_loss','goal_hits_since_last_log','elapsed_sec','frames_per_sec'])
            start_t = time.monotonic()
            since_log = 0
            next_save = (decisions*n*4 // args.save_every_frames + 1)*args.save_every_frames
            next_snapshot = (decisions*n*4 // args.snapshot_every_frames + 1)*args.snapshot_every_frames
            print(f'START | actors={n} warmup={args.warmup_per_actor} actions/actor '
                  f'A2C={args.n_step} actions/actor/update, frames/update={per_batch:,}',flush=True)
            print(f'REPLAY | capacity={args.replay_per_actor:,} per actor x {n} actors '
                  f'~{n*args.replay_per_actor*12288/2**30:.2f} GiB images',flush=True)
            if args.resume:
                print('INIT | Actor/Grid/Vision restored from latest.pt',flush=True)
            else:
                print(f'INIT | Actor=fresh; Grid/Vision={"B supervised checkpoint" if args.init_checkpoint else "fresh"}',flush=True)
            print('NOT VALIDATED in 32 DMLab environments. Run short smoke test first.',flush=True)
            while decisions*n*4 < args.total_env_frames:
                policy_mode = decisions >= args.warmup_per_actor
                rewards, dones, logps, values, entropies = [], [], [], [], []
                goals_this_rollout = 0
                for _ in range(args.n_step):
                    resets = force_reset.copy()
                    if last_obs is not None:
                        jump = np.linalg.norm(obs['pos'] - last_obs['pos'], axis=1) > .35
                        resets |= jump
                    motion = batched_motions(obs, last_obs, resets, rng)
                    visual_mask = resets | (rng.random(n) < args.vision_rate)
                    code, grid_h = batched_grid_step(grid, vision, obs, motion,
                                                     visual_mask, grid_h, resets, device)
                    if policy_mode:
                        logits, value, actor_next_h = policy.step(
                            code, goal_mem, prev_a, prev_r, actor_h)
                        dist = torch.distributions.Categorical(logits=logits)
                        action_t = dist.sample()
                    else:
                        action_t = torch.as_tensor(rng.integers(0, N_ACTIONS, size=n),
                                                   device=device, dtype=torch.long)
                    actions = action_t.detach().cpu().numpy().astype(np.int64)
                    next_obs, reward_np, done_np = vec.step(actions)
                    next_obs = drop_oracle(next_obs)
                    assert_vector_shapes(next_obs, n)
                    hit = reward_np > 0
                    goals_this_rollout += int(hit.sum())
                    for i in range(n):
                        replay.add(i, image=obs['image'][i], pos=obs['pos'][i],
                                   hd=float(obs['hd'][i][0]), motion=motion[i],
                                   reset=bool(resets[i]), vision_seen=bool(visual_mask[i]),
                                   episode=int(i*1_000_000 + episode_index[i]),
                                   action=int(actions[i]), reward=float(reward_np[i]))
                    if policy_mode:
                        # Note: goal touches teleport but do NOT terminate the episode.
                        logps.append(dist.log_prob(action_t))
                        values.append(value)
                        entropies.append(dist.entropy())
                        rewards.append(torch.as_tensor(reward_np, device=device))
                        dones.append(torch.as_tensor(done_np, device=device, dtype=torch.float32))
                        actor_h = reset_hidden(actor_next_h, torch.as_tensor(done_np, device=device))
                    # Store code at FIRST hit only (the PILOT's pre-hit approximation).
                    new_goal = torch.as_tensor(hit, device=device, dtype=torch.bool) & ~goal_known
                    if bool(new_goal.any()):
                        goal_mem = torch.where(new_goal[:, None], code.detach(), goal_mem)
                        goal_known |= new_goal
                    done_t = torch.as_tensor(done_np, device=device, dtype=torch.bool)
                    if bool(done_t.any()):
                        goal_mem = torch.where(done_t[:, None], 0., goal_mem)
                        goal_known = goal_known & ~done_t
                    prev_a = torch.where(done_t, torch.zeros_like(action_t), action_t.detach())
                    prev_r = torch.where(done_t[:, None], 0.,
                                         torch.as_tensor(reward_np[:, None], device=device))
                    force_reset = hit | done_np
                    last_obs, obs = obs, next_obs
                    decisions += 1
                    episode_hits += hit.astype(np.int64)
                    if bool(done_np.any()):
                        episode_index[done_np] += 1
                        episodes += int(done_np.sum())
                        episode_hits[done_np] = 0
                if policy_mode:
                    # Bootstrap without advancing the environment or consuming RNG.
                    with torch.no_grad():
                        saved_rng = copy.deepcopy(rng.bit_generator.state)
                        resets = force_reset.copy()
                        resets |= np.linalg.norm(obs['pos']-last_obs['pos'],axis=1) > .35
                        mot = batched_motions(obs,last_obs,resets,rng)
                        seen = resets | (rng.random(n)<args.vision_rate)
                        code_end,_ = batched_grid_step(grid,vision,obs,mot,seen,grid_h,resets,device)
                        _,bootstrap,_ = policy.step(code_end,goal_mem,prev_a,prev_r,actor_h)
                        rng.bit_generator.state = saved_rng
                    stats = finish_actor_rollout(policy,actor_opt,rewards,dones,logps,
                                                  values,entropies,bootstrap,args.gamma,
                                                  args.value_cost,args.entropy_cost)
                    actor_updates += 1
                    last_actor_loss = stats['a2c']
                    actor_h = tuple(x.detach() for x in actor_h)
                else:
                    if decisions == args.warmup_per_actor:
                        actor_h = actor_hidden(n,device)
                        print(f'WARMUP COMPLETE | frames={decisions*n*4:,} '
                              f'Vision/Grid trained; Actor begins next rollout',flush=True)
                # Auxiliary updates use replay but never alter the on-policy A2C graph.
                for _ in range(args.vision_updates_per_rollout):
                    last_vision_loss = train_vision(vision,pc,hd,replay,rng,vision_opt,device)
                    vision_updates += 1
                for _ in range(args.grid_updates_per_rollout):
                    last_grid_loss = train_grid(grid,vision,pc,hd,replay,rng,grid_opt,device)
                    grid_updates += 1
                total_goals_since_log += goals_this_rollout
                frames = decisions*n*4
                since_log += 1
                if actor_updates and actor_updates % args.log_every_updates == 0 or frames == args.total_env_frames:
                    elapsed = max(time.monotonic()-start_t, 1e-6)
                    fps = frames / elapsed if not args.resume else n*args.n_step*4*since_log/elapsed
                    csvout.writerow([frames,decisions*n,n,actor_updates,vision_updates,
                                     grid_updates,last_vision_loss,last_grid_loss,last_actor_loss,
                                     total_goals_since_log,round(elapsed,2),round(fps,2)])
                    print(f'LOG | frames={frames:,}/{args.total_env_frames:,} '
                          f'Actor={actor_updates:,} Vision={vision_updates:,} Grid={grid_updates:,} '
                          f'goals={total_goals_since_log} fps={fps:.1f}',flush=True)
                    total_goals_since_log=0
                if frames >= next_save or frames == args.total_env_frames:
                    save()
                    next_save=(frames//args.save_every_frames+1)*args.save_every_frames
                if frames >= next_snapshot or frames == args.total_env_frames:
                    snapshot()
                    next_snapshot=(frames//args.snapshot_every_frames+1)*args.snapshot_every_frames
                if stopping:
                    save()
                    print('SAFE STOP | saved after a complete on-policy A2C rollout', flush=True)
                    break
    except KeyboardInterrupt:
        if vec is not None:
            # Incomplete environment rollouts are not checkpointed; current design
            # only advances decisions at complete n-step boundaries.
            if decisions % args.n_step == 0:
                save()
            print('INTERRUPTED | no existing single-actor checkpoint modified',flush=True)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if vec is not None:
            vec.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode',choices=['self-test','check','train'],default='check')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--renderer',choices=['hardware','software'],default='hardware')
    p.add_argument('--num-envs',type=int,default=32)
    p.add_argument('--n-step',type=int,default=100)
    p.add_argument('--warmup-per-actor',type=int,default=1000)
    p.add_argument('--total-env-frames',type=int,default=200_000_000,
                   help='Environment frames summed across ALL actors; not actions')
    p.add_argument('--replay-per-actor',type=int,default=1000)
    p.add_argument('--init-checkpoint',type=Path,default=None,
                   help='Supervised B Grid/Vision ONLY; never use a 28.53M joint checkpoint')
    p.add_argument('--out-dir',type=Path,default=Path('data/checkpoints/joint_nav_32actors_NEW'))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--reset-replay-on-resume', action='store_true',
                   help='ONE-TIME crash recovery: keep weights/optimizers at latest.pt, start NEW replay and new metrics; old files untouched')
    p.add_argument('--vision-updates-per-rollout',type=int,default=100,
                   help='100 matches old one-update-per-32-action cadence (32x100/32)')
    p.add_argument('--grid-updates-per-rollout',type=int,default=32,
                   help='32 matches old one-update-per-100-action cadence (32x100/100)')
    p.add_argument('--vision-rate',type=float,default=.05)
    p.add_argument('--vision-lr',type=float,default=1e-4)
    p.add_argument('--grid-lr',type=float,default=1e-4)
    p.add_argument('--actor-lr',type=float,default=.0001005)
    p.add_argument('--gamma',type=float,default=.99)
    p.add_argument('--value-cost',type=float,default=.5)
    p.add_argument('--entropy-cost',type=float,default=.00008)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save-every-frames',type=int,default=10_000_000)
    p.add_argument('--snapshot-every-frames',type=int,default=10_000_000)
    p.add_argument('--log-every-updates',type=int,default=10)
    a=p.parse_args()
    for key in ('num_envs','n_step','total_env_frames','warmup_per_actor','replay_per_actor',
                'save_every_frames','snapshot_every_frames','log_every_updates'):
        if getattr(a,key)<=0:
            p.error(f'{key} must be positive')
    if a.replay_per_actor < 1000:
        p.error('replay-per-actor must be at least 1000')
    if not 0<=a.vision_rate<=1:
        p.error('vision-rate must be in [0,1]')
    if a.vision_updates_per_rollout < 0 or a.grid_updates_per_rollout < 0:
        p.error('auxiliary update counts must be nonnegative')
    if a.init_checkpoint and not a.init_checkpoint.is_file():
        p.error(f'No supervised B checkpoint at {a.init_checkpoint}')
    if a.reset_replay_on_resume and not a.resume:
        p.error('--reset-replay-on-resume requires --resume')
    if a.resume and a.mode!='train':
        p.error('--resume works only with --mode train')
    return a


def main():
    a=parse_args()
    if a.mode=='self-test':
        self_test()
        return
    seed_all(a.seed)
    d=choose_device(a.device)
    if a.mode=='check':
        check(a,d)
    elif a.mode=='train':
        train(a,d)


if __name__=='__main__':
    main()
