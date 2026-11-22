#!/usr/bin/env python3
"""Joint Vision + Grid + A2C online-navigation PILOT (one DMLab agent).

Contract (all steps are *agent actions*, each repeated for 4 DMLab frames):
  steps 1..1000: 6 actions uniformly random, NOT policy-generated.
  after >=32 replay samples: Vision update every 32 new actions (32 RGB).
  at action 1000, then every 100 actions: Grid update using 10 x 100
      non-overlapping, chronologically ordered replay sequences.
  actions 1001+: sample Actor's six-way Softmax distribution; A2C update
      every 100 on-policy actions, with its OWN short rollout (not replay).
  disk replay: most recent 200000 action records, oldest overwritten.

Vision is an auxiliary supervised CNN; Grid is trained with PC/HD targets;
Actor-Critic receives only live Grid code, remembered first-hit Grid code,
previous action, previous reward. Actor loss never enters Vision/Grid.
This is a controlled A2C research prototype, NOT a verified reproduction of
Banino et al. (2018) A3C. DEBUG position/heading are ONLY used for auxiliary
labels, teleport detection, and pilot yaw odometry (privileged signal).
Reward > 0 stores PRE-hit Grid code as goal memory; it is an approximation.

Needed in the SAME directory:
  joint_nav_replay.py
And the existing repo's config.py, grid_cells/, and levels/square_arena_oracle.lua.

Examples:
  python train_joint_nav_online.py --mode check --device cuda --init-checkpoint \\
    data/checkpoints/grid1350_online_B_seq_run1/grid_78k_analysis.pt
  python -u train_joint_nav_online.py --mode train --device cuda \\
    --init-checkpoint data/checkpoints/grid1350_online_B_seq_run1/grid_78k_analysis.pt \\
    --steps 1200 --replay-capacity 2000 \\
    --out-dir data/checkpoints/joint_nav_smoke
  python -u train_joint_nav_online.py --mode train --device cuda \\
    --init-checkpoint data/checkpoints/grid1350_online_B_seq_run1/grid_78k_analysis.pt \\
    --steps 100000 --replay-capacity 200000 \\
    --out-dir data/checkpoints/joint_nav_run1
  python -u train_joint_nav_online.py --mode train --resume \\
    --out-dir data/checkpoints/joint_nav_run1 --steps 150000
"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from joint_nav_replay import DiskRingReplay, self_test as replay_self_test

ALG = 'joint_navigation_ring_vision_grid_a2c_v1'
GRID_DIM, VIS_DIM, N_ACTIONS = 512, 268, 6
WARMUP = 1_000
VISION_BATCH, GRID_PATHS, GRID_LEN = 32, 10, 100
WORLD_TO_METERS = 2.5 / 1000.0
DT = 4.0 / 60.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for data in iter(lambda: fh.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def choose_device(s):
    d = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if s == 'auto' else s)
    if d.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return d


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GridCodeActorCritic(nn.Module):
    """Grid-only policy: intentionally excludes a second direct RGB shortcut."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2 * GRID_DIM + N_ACTIONS + 1, 256, batch_first=True)
        self.actor = nn.Linear(256, N_ACTIONS)
        self.critic = nn.Linear(256, 1)
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    std = m.weight.shape[1] ** -.5
                    nn.init.trunc_normal_(m.weight, std=std, a=-2*std, b=2*std)
                    nn.init.zeros_(m.bias)
            for name, p in self.lstm.named_parameters():
                if name.startswith('weight'):
                    std = p.shape[1] ** -.5
                    nn.init.trunc_normal_(p, std=std, a=-2*std, b=2*std)
                else:
                    nn.init.zeros_(p)
            self.lstm.bias_hh_l0[256:512].fill_(1.)

    def init_hidden(self, device):
        a = torch.zeros(1, 1, 256, device=device)
        return a, a.clone()

    def step(self, grid_code, goal_code, previous_action, previous_reward, hidden):
        onehot = F.one_hot(previous_action, N_ACTIONS).float()
        x = torch.cat([grid_code, goal_code, onehot, previous_reward], -1)
        z, hidden = self.lstm(x[:, None], hidden)
        z = z[:, 0]
        return self.actor(z), self.critic(z).squeeze(-1), hidden


def make_models(device, initial_checkpoint):
    from config import Config
    from grid_cells.ensembles import build_rl_ensembles
    from grid_cells.model import GridCellsRNN
    from grid_cells.vision import VisionModule
    cfg = Config()
    cfg.rl.pc_scale = (.01,)
    (pc,), (hd,) = build_rl_ensembles(cfg, device)
    assert pc.n_cells == 256 and hd.n_cells == 12
    grid = GridCellsRNN(target_ensembles=[pc, hd], nh_lstm=128,
                        nh_bottleneck=GRID_DIM, dropout_rates=(.5,),
                        bottleneck_has_bias=True, ego_vel_dim=4,
                        vision_dim=VIS_DIM).to(device)
    vision = VisionModule(pc, hd, embed_dim=256, mask_prob=.95,
                          mask_mode='whole').to(device)
    if initial_checkpoint:
        ck = torch.load(initial_checkpoint, map_location='cpu', weights_only=False)
        if not {'grid', 'vision', 'pc_centers', 'hd_centers'} <= ck.keys():
            raise RuntimeError('Initial checkpoint does not contain B Grid/Vision/PC/HD')
        if not torch.allclose(ck['pc_centers'].cpu(), pc.means.cpu(), atol=1e-7, rtol=0):
            raise RuntimeError('PC ensemble mismatch in initialization checkpoint')
        if not torch.allclose(ck['hd_centers'].cpu(), hd.means.cpu(), atol=1e-7, rtol=0):
            raise RuntimeError('HD ensemble mismatch in initialization checkpoint')
        grid.load_state_dict(ck['grid'], strict=True)
        vision.load_state_dict(ck['vision'], strict=True)
        print(f'INITIALIZED | B Grid={ck.get("grid_updates", "?")}, '
              f'Vision={ck.get("vision_updates", "?")} (new optimizers)', flush=True)
    else:
        print('INITIALIZED | Grid and Vision from scratch; warmup data may be inadequate', flush=True)
    return cfg, pc, hd, grid, vision


def noisy_motion(obs, prev_obs, reset, rng):
    if reset:
        return np.array([0., 0., 0., 1.], np.float32)
    uv = np.asarray(obs['vel_trans'][:2], dtype=np.float64) * WORLD_TO_METERS
    uv += rng.normal(0., .01, 2)
    yaw_now = float(obs['hd'][0])
    yaw_prev = float(prev_obs['hd'][0])
    dyaw = math.atan2(math.sin(yaw_now-yaw_prev), math.cos(yaw_now-yaw_prev))
    dyaw += float(rng.normal(0., .01 * DT))
    return np.asarray([uv[0], uv[1], math.sin(dyaw), math.cos(dyaw)], np.float32)


def vision_cues(vision, images_uint8, device):
    """RGB uint8 [N,3,64,64] -> soft PC+HD [N,268], no dropout mask."""
    from grid_cells.rl_common import image_to_input
    e = vision.cnn(image_to_input(images_uint8.to(device)))
    return torch.cat([F.softmax(vision.place_head(e), -1),
                      F.softmax(vision.hd_head(e), -1)], -1)


@torch.no_grad()
def live_code(grid, vision, obs, motion, vision_seen, hidden, device, reset):
    if reset:
        hidden = tuple(torch.zeros_like(z) for z in hidden)
    cue = torch.zeros(1, VIS_DIM, device=device)
    if vision_seen:
        image = torch.from_numpy(np.array(obs['image'], copy=True))[None].to(device)
        cue = vision_cues(vision, image, device)
    grid.eval(); vision.eval()
    code, hidden = grid.step(torch.as_tensor(motion, device=device)[None], cue, hidden)
    if code.shape != (1, GRID_DIM) or not torch.isfinite(code).all():
        raise RuntimeError('Invalid online Grid code')
    return code.detach(), tuple(h.detach() for h in hidden)


def step_grid_segment(grid, vel, vis, reset):
    """Same reset-aware LSTM segment semantics as B supervised Grid trainer."""
    batch, seq = vel.shape[:2]
    x = torch.cat([vel, vis], -1)
    h = (torch.zeros(1, batch, 128, device=vel.device),
         torch.zeros(1, batch, 128, device=vel.device))
    pieces = []
    start = 0
    for k in torch.nonzero(reset.any(0), as_tuple=False).flatten().tolist():
        if k > start:
            y, h = grid.rl_lstm(x[:, start:k], h)
            pieces.append(y)
        keep = (~reset[:, k]).float().view(1, batch, 1)
        h = tuple(v * keep for v in h)
        y, h = grid.rl_lstm(x[:, k:k+1], h)
        pieces.append(y)
        start = k+1
    if start < seq:
        y, h = grid.rl_lstm(x[:, start:], h)
        pieces.append(y)
    out = torch.cat(pieces, dim=1)
    b = grid.bottleneck(out)
    b = F.dropout(b, .5, training=True)
    return grid.output_heads[0](b), grid.output_heads[1](b)


def train_vision(vision, pc, hd, replay, rng, optimizer, device):
    batch = replay.sample_vision(VISION_BATCH, rng)
    from grid_cells.rl_common import image_to_input
    image = image_to_input(torch.from_numpy(batch['image']).to(device))
    pos = torch.from_numpy(batch['pos']).to(device)
    heading = torch.from_numpy(batch['hd']).to(device)
    with torch.no_grad():
        label_pc = pc.posterior(pos[:, None]).squeeze(1)
        label_hd = hd.posterior(heading[:, None]).squeeze(1)
    vision.train()
    optimizer.zero_grad(set_to_none=True)
    loss = vision.loss(image, label_pc, label_hd)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite Vision loss')
    loss.backward()
    nn.utils.clip_grad_norm_(vision.parameters(), 1.)
    optimizer.step()
    vision.eval()
    return float(loss.detach())


def train_grid(grid, vision, pc, hd, replay, rng, optimizer, device):
    batch = replay.sample_grid(GRID_PATHS, GRID_LEN, rng)
    motion = torch.from_numpy(batch['motion']).to(device)
    reset = torch.from_numpy(batch['reset'].astype(np.bool_)).to(device)
    pos = torch.from_numpy(batch['pos']).to(device)
    heading = torch.from_numpy(batch['hd']).to(device)
    visual = torch.zeros(GRID_PATHS, GRID_LEN, VIS_DIM, device=device)
    visible = np.argwhere(batch['vision_seen'] != 0)
    # Aux Vision provides Grid cues; Grid CE does NOT backprop into Vision.
    vision.eval()
    with torch.no_grad():
        for start in range(0, len(visible), 128):
            take = visible[start:start+128]
            image = torch.from_numpy(batch['image'][take[:, 0], take[:, 1]]).to(device)
            visual[take[:, 0], take[:, 1]] = vision_cues(vision, image, device)
        label_pc = pc.posterior(pos)
        label_hd = hd.posterior(heading[..., None])
    grid.train()
    optimizer.zero_grad(set_to_none=True)
    lp, lh = step_grid_segment(grid, motion, visual, reset)
    loss = (-(label_pc * F.log_softmax(lp, -1)).sum(-1)
            -(label_hd * F.log_softmax(lh, -1)).sum(-1)).mean()
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite Grid loss')
    loss.backward()
    nn.utils.clip_grad_norm_(grid.parameters(), 1.)
    optimizer.step()
    grid.eval()
    return float(loss.detach())


def compute_returns(rewards, dones, bootstrap, gamma):
    running = bootstrap.detach()
    out = []
    for r, done in zip(reversed(rewards), reversed(dones)):
        running = r + gamma * running * (1. - done)
        out.append(running)
    return torch.stack(out[::-1])


def finish_actor_rollout(policy, optimizer, rewards, dones, logps,
                         values, entropies, bootstrap, gamma, value_coef, entropy_coef):
    if not rewards:
        return None
    returns = compute_returns(rewards, dones, bootstrap, gamma)
    logps_t, value_t, ent_t = torch.stack(logps), torch.stack(values), torch.stack(entropies)
    advantage = returns - value_t.detach()
    policy_loss = -(logps_t * advantage).mean()
    value_loss = F.mse_loss(value_t, returns)
    loss = policy_loss + value_coef * value_loss - entropy_coef * ent_t.mean()
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite A2C loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.)
    optimizer.step()
    return dict(a2c=float(loss.detach()), policy=float(policy_loss.detach()),
                value=float(value_loss.detach()), entropy=float(ent_t.mean().detach()))


class LabRunner:
    """A single 2.5m x 2.5m environment. Debug goal coordinates not requested."""
    def __init__(self, env_cfg, renderer, seed):
        from grid_cells.rl_common import make_dmlab_env
        self.cfg = env_cfg
        self.seed = seed
        self.episode = 0
        self.env = make_dmlab_env(env_cfg, renderer=renderer, level='square_arena_oracle')
        self.env.reset(seed=self.seed)

    def observe(self):
        from grid_cells.rl_common import convert_observation
        return convert_observation(self.env.observations(), self.cfg)

    def step(self, action):
        from grid_cells.actor_critic import DISCRETE_ACTIONS
        reward = float(self.env.step(DISCRETE_ACTIONS[action], num_steps=self.cfg.action_repeat))
        done = not self.env.is_running()
        if done:
            self.episode += 1
            self.env.reset(seed=self.seed + self.episode)
        return self.observe(), reward, done

    def close(self):
        self.env.close()


def atomic_save(state, path):
    tmp = path.with_suffix('.tmp.pt')
    torch.save(state, tmp)
    os.replace(tmp, path)


def check(args, device):
    cfg, pc, hd, grid, vision = make_models(device, args.init_checkpoint)
    grid.eval(); vision.eval()
    policy = GridCodeActorCritic().to(device)
    fake = {'image': np.zeros((3, 64, 64), dtype=np.uint8),
            'pos': np.zeros((2,), dtype=np.float32),
            'hd': np.zeros((1,), dtype=np.float32),
            'vel_trans': np.zeros((3,), dtype=np.float32)}
    motion = np.array([0., 0., 0., 1.], dtype=np.float32)
    code, _ = live_code(grid, vision, fake, motion, True,
                        (torch.zeros(1,1,128,device=device),)*2, device, True)
    z, value, _ = policy.step(code, torch.zeros_like(code),
                              torch.zeros(1,dtype=torch.long,device=device),
                              torch.zeros(1,1,device=device), policy.init_hidden(device))
    assert z.shape == (1,6) and value.shape == (1,)
    with tempfile.TemporaryDirectory() as td:
        replay_self_test(Path(td)/'ring')
    print('PREFLIGHT PASS | 2.5m arena config, 268D Vision -> 512D Grid -> 6-way Actor', flush=True)
    print('CHECK MODE: no DMLab launched and no checkpoint modified', flush=True)


def train(args, device):
    cfg, pc, hd, grid, vision = make_models(device, args.init_checkpoint)
    env_cfg = cfg.rl.env
    env_cfg.level = 'square_arena_oracle'
    env_cfg.goal_visible = False
    env_cfg.action_repeat = 4
    if (env_cfg.env_size_m, env_cfg.width, env_cfg.height) != (2.5, 64, 64):
        raise RuntimeError('Expected 2.5m x 2.5m, 64x64 RGB')
    if not Path(env_cfg.level_directory, 'square_arena_oracle.lua').is_file():
        raise FileNotFoundError('Missing levels/square_arena_oracle.lua (run your original Oracle level setup)')
    grid.eval(); vision.eval()
    actor = GridCodeActorCritic().to(device)
    grid_opt = torch.optim.Adam(grid.parameters(), lr=args.grid_lr, weight_decay=1e-5)
    vision_opt = torch.optim.Adam(vision.parameters(), lr=args.vision_lr)
    actor_opt = torch.optim.RMSprop(actor.parameters(), lr=args.actor_lr,
                                    alpha=.99, eps=1e-8, momentum=0.)
    out = args.out_dir.expanduser().resolve()
    checkpoint_path = out/'latest.pt'
    rng = np.random.default_rng(args.seed + 10022)
    global_actions, vision_updates, grid_updates, actor_updates, episodes = 0, 0, 0, 0, 0
    init_hash = sha256(args.init_checkpoint) if args.init_checkpoint else None

    if args.resume:
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if ck.get('algorithm') != ALG:
            raise RuntimeError('Not a joint-navigation checkpoint')
        if ck['replay_capacity'] != args.replay_capacity:
            raise RuntimeError('Resume requires the original --replay-capacity')
        if ck['initial_sha256'] != init_hash:
            raise RuntimeError('Resume requires exactly the original --init-checkpoint')
        grid.load_state_dict(ck['grid']); vision.load_state_dict(ck['vision'])
        actor.load_state_dict(ck['actor'])
        grid_opt.load_state_dict(ck['grid_opt']); vision_opt.load_state_dict(ck['vision_opt'])
        actor_opt.load_state_dict(ck['actor_opt'])
        global_actions = int(ck['actions'])
        vision_updates = int(ck['vision_updates'])
        grid_updates = int(ck['grid_updates'])
        actor_updates = int(ck['actor_updates'])
        episodes = int(ck['episodes'])
        rng.bit_generator.state = ck['sample_rng']
        torch.set_rng_state(ck['torch_rng_state'])
        if device.type == 'cuda' and ck.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state(ck['cuda_rng_state'].cpu(), device)
        replay = DiskRingReplay(out/'replay', args.replay_capacity, create=False)
        replay.load_count(global_actions)
        print(f'RESUME | actions={global_actions:,}; new DMLab episode, recurrent states reset', flush=True)
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'Nonempty output directory: {out}; use --resume or a NEW path')
        out.mkdir(parents=True, exist_ok=True)
        replay = DiskRingReplay(out/'replay', args.replay_capacity, create=True)

    last_snapshot_bucket = (actor_updates // args.snapshot_every_updates
                            if args.snapshot_every_updates else 0)

    def save():
        nonlocal last_snapshot_bucket
        replay.flush()
        state = {'algorithm':ALG, 'actions':global_actions, 'episodes':episodes,
                 'replay_capacity':args.replay_capacity, 'initial_sha256':init_hash,
                 'vision_updates':vision_updates, 'grid_updates':grid_updates,
                 'actor_updates':actor_updates,
                 'grid':grid.state_dict(), 'vision':vision.state_dict(),
                 'actor':actor.state_dict(), 'grid_opt':grid_opt.state_dict(),
                 'vision_opt':vision_opt.state_dict(), 'actor_opt':actor_opt.state_dict(),
                 'sample_rng':rng.bit_generator.state, 'torch_rng_state':torch.get_rng_state().cpu(),
                 'cuda_rng_state':torch.cuda.get_rng_state(device).cpu() if device.type=='cuda' else None,
                 'args':vars(args)}
        atomic_save(state, checkpoint_path)
        if args.snapshot_every_updates:
            bucket = actor_updates // args.snapshot_every_updates
            if bucket > last_snapshot_bucket:
                milestone = bucket * args.snapshot_every_updates
                snap = out / f'snapshot_actor_{milestone:07d}.pt'
                if snap.exists():
                    raise FileExistsError(f'Snapshot already exists: {snap}')
                atomic_save(state, snap)
                last_snapshot_bucket = bucket
                print(f'SNAPSHOT | actions={global_actions:,} '
                      f'actor_updates={actor_updates:,} path={snap}', flush=True)
        print(f'SAVED | actions={global_actions:,}, Vision={vision_updates}, '
              f'Grid={grid_updates}, Actor={actor_updates}', flush=True)

    runner = LabRunner(env_cfg, args.renderer, args.seed + 1000 + episodes)
    obs = runner.observe()
    previous_obs = None
    force_reset = True
    grid_hidden = (torch.zeros(1,1,128,device=device), torch.zeros(1,1,128,device=device))
    policy_hidden = actor.init_hidden(device)
    previous_action = torch.zeros(1, dtype=torch.long,device=device)
    previous_reward = torch.zeros(1, 1,device=device)
    goal_memory = torch.zeros(1,GRID_DIM,device=device)
    goal_known = False
    running_episode_reward = 0.
    running_episode_goals = 0
    logps, values, entropies, rewards, dones = [], [], [], [], []
    last_vis_loss, last_grid_loss, last_actor_loss = float('nan'),float('nan'),float('nan')
    action_stats = np.zeros(6, dtype=np.int64)
    grid_seen = 0
    visited = np.zeros((10, 10), dtype=bool)
    log_mode = 'a' if args.resume else 'w'
    with (out/'metrics.csv').open(log_mode, newline='', buffering=1) as fh:
        writer = csv.writer(fh)
        if not args.resume:
            writer.writerow(['actions','buffer_size','vision_updates','grid_updates','actor_updates',
                             'vision_CE','grid_CE','a2c_loss','goals_episode','visited_bins',
                             'uniform_action_counts','last_reward'])
        print(f'START | single agent; uniform actions 1..{WARMUP:,}; '
              f'Vision>=32 every {args.vision_every}; Grid>=1000 every {args.grid_every}; '
              f'Actor after {WARMUP}, A2C n={args.n_step}', flush=True)
        print(f'REPLAY | capacity={args.replay_capacity:,}, disk storage~'
              f'{args.replay_capacity * 12288 / 2**30:.2f}GiB plus metadata',flush=True)
        if args.resume and global_actions % args.n_step != 0:
            print('NOTE: A2C restarts a fresh on-policy rollout after resume', flush=True)
        try:
            while global_actions < args.steps:
                # Observation at t represents exactly ONE decision/action.
                # force_reset is set at episode start and *after* a reward teleport.
                reset = (force_reset or previous_obs is None or
                         np.linalg.norm(obs['pos'] - previous_obs['pos']) > .35)
                motion = noisy_motion(obs, previous_obs, reset, rng)
                vis_seen = bool(reset or rng.random() < args.vision_rate)
                code, grid_hidden = live_code(grid, vision, obs, motion, vis_seen,
                                               grid_hidden, device, reset)
                # Position is a LABEL ONLY. Store pre-action observation now,
                # action/reward as soon as environment executes the action.
                bin_xy = np.clip(((obs['pos'] + 1.25) / 2.5 * 10).astype(int), 0, 9)
                visited[int(bin_xy[0]), int(bin_xy[1])] = True

                policy_step = global_actions >= WARMUP
                if policy_step:
                    logits, value, next_policy_hidden = actor.step(
                        code, goal_memory, previous_action, previous_reward,
                        policy_hidden)
                    dist = torch.distributions.Categorical(logits=logits)
                    action_t = dist.sample()
                    action = int(action_t.item())
                else:
                    action = int(rng.integers(0, N_ACTIONS))
                next_obs, reward, done = runner.step(action)
                replay.add(image=obs['image'], pos=obs['pos'], hd=obs['hd'][0],
                           motion=motion, reset=reset, vision_seen=vis_seen,
                           episode=episodes, action=action, reward=reward)
                action_stats[action] += (not policy_step)
                global_actions += 1
                running_episode_reward += reward
                running_episode_goals += int(reward > 0)
                if reward > 0 and not goal_known:
                    goal_memory = code.detach().clone()  # pre-hit, no oracle goal XY
                    goal_known = True
                if policy_step:
                    logps.append(dist.log_prob(action_t))
                    values.append(value[0])
                    entropies.append(dist.entropy()[0])
                    rewards.append(torch.tensor(reward, device=device))
                    dones.append(torch.tensor(float(done), device=device))
                    policy_hidden = next_policy_hidden
                previous_action = torch.tensor([action],device=device)
                previous_reward = torch.tensor([[reward]],device=device)
                previous_obs, obs = obs, next_obs
                force_reset = bool(done or reward > 0)

                if done:
                    print(f'EPISODE | actions={global_actions:,} reward={running_episode_reward:.2f} '
                          f'goals={running_episode_goals}', flush=True)
                    episodes += 1
                    policy_hidden = actor.init_hidden(device)
                    previous_action.zero_(); previous_reward.zero_()
                    goal_memory.zero_(); goal_known = False
                    previous_obs = None
                    running_episode_reward = 0.
                    running_episode_goals = 0

                # Auxiliary learning can proceed while policy is still uniform.
                # Keep these losses OUT of the actor's computation graph.
                if global_actions >= VISION_BATCH and global_actions % args.vision_every == 0:
                    last_vis_loss = train_vision(vision, pc, hd, replay, rng, vision_opt, device)
                    vision_updates += 1

                # Close the A2C rollout BEFORE updating Grid weights.
                if len(rewards) == args.n_step or (global_actions == args.steps and rewards):
                    with torch.no_grad():
                        peek_reset = (force_reset or previous_obs is None or
                                      np.linalg.norm(obs['pos']-previous_obs['pos']) > .35)
                        # Bootstrap must not consume randomness for the real
                        # next observation (motion noise OR sparse Vision cue).
                        saved_rng = copy.deepcopy(rng.bit_generator.state)
                        peek_motion = noisy_motion(obs, previous_obs, peek_reset, rng)
                        peek_seen = bool(peek_reset or rng.random() < args.vision_rate)
                        bootstrap_code, _ = live_code(grid, vision, obs, peek_motion,
                                                       peek_seen, grid_hidden, device, peek_reset)
                        _, bootstrap_value, _ = actor.step(
                            bootstrap_code, goal_memory, previous_action,
                            previous_reward, policy_hidden)
                        rng.bit_generator.state = saved_rng
                    stats = finish_actor_rollout(actor, actor_opt, rewards, dones,
                                                logps, values, entropies,
                                                bootstrap_value[0], args.gamma,
                                                args.value_cost, args.entropy_cost)
                    actor_updates += 1
                    last_actor_loss = stats['a2c']
                    logps, values, entropies, rewards, dones = [], [], [], [], []
                    policy_hidden = tuple(z.detach() for z in policy_hidden)
                    print(f'A2C | action={global_actions:,} updates={actor_updates} '
                          f'loss={last_actor_loss:.4f} entropy={stats["entropy"]:.3f}',flush=True)

                if global_actions >= WARMUP and (global_actions == WARMUP or
                                                   global_actions % args.grid_every == 0):
                    last_grid_loss = train_grid(grid, vision, pc, hd, replay, rng,
                                                grid_opt, device)
                    grid_updates += 1
                    print(f'GRID | action={global_actions:,} updates={grid_updates} '
                          f'CE={last_grid_loss:.4f}',flush=True)

                if global_actions == WARMUP:
                    # First policy action is the NEXT decision (1001).
                    policy_hidden = actor.init_hidden(device)
                    print(f'WARMUP COMPLETE | 1,000 uniform actions; '
                          f'counts={action_stats.tolist()}, '
                          f'visited_bins={int(visited.sum())}/100', flush=True)

                if global_actions % args.log_every == 0 or global_actions == args.steps:
                    writer.writerow([global_actions, replay.size, vision_updates,
                                     grid_updates, actor_updates,
                                     last_vis_loss,last_grid_loss,last_actor_loss,
                                     running_episode_goals,int(visited.sum()),
                                     str(action_stats.tolist()), reward])
                    print(f'LOG | actions={global_actions:,} replay={replay.size:,} '
                          f'Vision={vision_updates} Grid={grid_updates} Actor={actor_updates} '
                          f'goals={running_episode_goals}',flush=True)
                if global_actions % args.save_every == 0 or global_actions == args.steps:
                    save()
        except KeyboardInterrupt:
            # The incomplete A2C segment is discarded (no off-policy replay).
            # Save trained weights plus the exact number of completed actions.
            policy_hidden = tuple(z.detach() for z in policy_hidden)
            logps.clear(); values.clear(); entropies.clear(); rewards.clear(); dones.clear()
            save()
            print('INTERRUPTED | saved auxiliary/Actor weights and replay count',flush=True)
        finally:
            runner.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode',choices=['check','train'],default='check')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--renderer',choices=['hardware','software'],default='hardware')
    p.add_argument('--init-checkpoint',type=Path,default=None,
                   help='Optional B Grid+Vision checkpoint; never modified')
    p.add_argument('--out-dir',type=Path,default=Path('data/checkpoints/joint_nav_smoke'))
    p.add_argument('--steps',type=int,default=1200,
                   help='Total agent actions (NOT engine frames)')
    p.add_argument('--replay-capacity',type=int,default=200_000)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--vision-every',type=int,default=32)
    p.add_argument('--grid-every',type=int,default=100)
    p.add_argument('--n-step',type=int,default=100)
    p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--snapshot-every-updates',type=int,default=0,
                   help='Keep separate checkpoint per N Actor updates; 0 disables')
    p.add_argument('--log-every',type=int,default=100)
    p.add_argument('--vision-rate',type=float,default=.05)
    p.add_argument('--vision-lr',type=float,default=1e-4)
    p.add_argument('--grid-lr',type=float,default=1e-4)
    p.add_argument('--actor-lr',type=float,default=.0001005)
    p.add_argument('--gamma',type=float,default=.99)
    p.add_argument('--value-cost',type=float,default=.5)
    p.add_argument('--entropy-cost',type=float,default=.00008)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    for name in ['steps','replay_capacity','vision_every','grid_every','n_step','save_every','log_every']:
        if getattr(a,name) < 1:
            p.error(f'{name} must be positive')
    if a.snapshot_every_updates < 0:
        p.error('--snapshot-every-updates must be >= 0')
    if a.replay_capacity < GRID_LEN*GRID_PATHS:
        p.error('replay-capacity must be >=1000')
    if a.vision_every < VISION_BATCH:
        p.error('vision-every must be >=32 (first Vision update must not precede 32 images)')
    if not (0 <= a.vision_rate <= 1):
        p.error('vision-rate must be between 0 and 1')
    if a.resume and a.mode!='train':
        p.error('--resume is only valid with --mode train')
    if a.init_checkpoint and not a.init_checkpoint.is_file():
        p.error('initial checkpoint does not exist: '+str(a.init_checkpoint))
    return a


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    if args.mode == 'check':
        check(args, device)
    else:
        train(args, device)


if __name__ == '__main__':
    main()
