#!/usr/bin/env python3
"""Read-only, paired 1M vs 21M joint-navigation evaluation (100 episodes each).

Replicates online inference from train_joint_nav_online.py, but performs NO
training, NO optimizer steps, NO checkpoint writes, and NO replay reads/writes.
Environment seed per evaluation episode is the same for each checkpoint.
A positive reward is counted as a goal hit; goal memory stores the pre-hit
512-D Grid code, just as in training.

Examples:
  python -u eval_joint_nav_1m_vs_21m.py --mode check --device cuda
  python -u eval_joint_nav_1m_vs_21m.py --mode eval --device cuda --episodes 100
  python -u eval_joint_nav_1m_vs_21m.py --mode eval --device cuda --episodes 10 --goal-memory off --out-dir data/evaluations/joint_nav_ablation_10
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch
import torch.nn.functional as F

import train_joint_nav_online as joint

DEFAULT_MODELS = {
    '1m': Path('data/checkpoints/joint_nav_pilot_1m_backup/latest.pt'),
    '21m': Path('data/checkpoints/joint_nav_pilot_21m_backup/latest.pt'),
}


def load_models(device, initial_path, checkpoint_path):
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if ck.get('algorithm') != joint.ALG:
        raise RuntimeError(f'Invalid joint checkpoint: {checkpoint_path}')
    if ck.get('initial_sha256') != joint.sha256(initial_path):
        raise RuntimeError(f'Original initialization hash mismatch: {checkpoint_path}')
    cfg, pc, hd, grid, vision = joint.make_models(device, initial_path)
    actor = joint.GridCodeActorCritic().to(device)
    grid.load_state_dict(ck['grid'], strict=True)
    vision.load_state_dict(ck['vision'], strict=True)
    actor.load_state_dict(ck['actor'], strict=True)
    grid.eval(); vision.eval(); actor.eval()

    # Check all tensors before launching DMLab. No optimizer or replay is touched.
    for name, module in [('vision', vision), ('grid', grid), ('actor', actor)]:
        for n, p in module.named_parameters():
            if not bool(torch.isfinite(p).all()):
                raise RuntimeError(f'Non-finite parameter: {name}.{n}')
    return cfg, grid, vision, actor, ck


def make_env_cfg(cfg):
    env_cfg = cfg.rl.env
    env_cfg.level = 'square_arena_oracle'
    env_cfg.goal_visible = False
    env_cfg.action_repeat = 4
    if (env_cfg.env_size_m, env_cfg.width, env_cfg.height) != (2.5, 64, 64):
        raise RuntimeError('Expected the same 2.5m, 64x64 environment as training')
    if not Path(env_cfg.level_directory, 'square_arena_oracle.lua').is_file():
        raise FileNotFoundError('Missing levels/square_arena_oracle.lua')
    return env_cfg


@torch.inference_mode()
def evaluate_one(name, checkpoint, initial_path, args, device, writer):
    cfg, grid, vision, actor, ck = load_models(device, initial_path, checkpoint)
    env_cfg = make_env_cfg(cfg)
    runner = joint.LabRunner(env_cfg, args.renderer, args.env_seed_base)
    results = []
    try:
        obs = runner.observe()
        for episode_idx in range(args.episodes):
            # Fresh state each episode, as in train_joint_nav_online.py.
            grid_hidden = (torch.zeros(1, 1, 128, device=device),
                           torch.zeros(1, 1, 128, device=device))
            policy_hidden = actor.init_hidden(device)
            prev_action = torch.zeros(1, dtype=torch.long, device=device)
            prev_reward = torch.zeros(1, 1, device=device)
            goal_memory = torch.zeros(1, joint.GRID_DIM, device=device)
            goal_known = False
            prev_obs = None
            force_reset = True
            rng = np.random.default_rng(args.motion_seed_base + episode_idx)
            action_rng = torch.Generator(device=device)
            action_rng.manual_seed(args.policy_seed_base + episode_idx)
            hits = []
            total_reward = 0.0
            entropy_sum = 0.0
            action_counts = np.zeros(joint.N_ACTIONS, dtype=np.int64)
            visited = np.zeros((10, 10), dtype=np.bool_)
            done = False

            for t in range(1, args.max_steps_per_episode + 1):
                reset = (force_reset or prev_obs is None or
                         np.linalg.norm(obs['pos'] - prev_obs['pos']) > .35)
                motion = joint.noisy_motion(obs, prev_obs, reset, rng)
                vis_seen = bool(reset or rng.random() < args.vision_rate)
                code, grid_hidden = joint.live_code(
                    grid, vision, obs, motion, vis_seen, grid_hidden, device, reset)

                b = np.clip(((obs['pos'] + 1.25) / 2.5 * 10).astype(int), 0, 9)
                visited[int(b[0]), int(b[1])] = True

                logits, value, policy_hidden = actor.step(
                    code, goal_memory, prev_action, prev_reward, policy_hidden)
                probs = F.softmax(logits, dim=-1)
                if probs.shape != (1, joint.N_ACTIONS) or not bool(torch.isfinite(probs).all()):
                    raise RuntimeError(f'Invalid policy at {name} ep={episode_idx} step={t}')
                action = int(torch.multinomial(probs[0], 1, generator=action_rng).item())
                entropy_sum += float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
                action_counts[action] += 1
                next_obs, reward, done = runner.step(action)
                total_reward += reward
                if reward > 0:
                    hits.append(t)
                    if not goal_known and args.goal_memory == 'on':
                        goal_memory = code.detach().clone()  # matching training (pre-hit)
                        goal_known = True
                prev_action = torch.tensor([action], device=device, dtype=torch.long)
                prev_reward = torch.tensor([[reward]], device=device)
                prev_obs, obs = obs, next_obs
                force_reset = bool(done or reward > 0)
                if done:
                    break
            if not done:
                raise RuntimeError(f'Episode {episode_idx} exceeded {args.max_steps_per_episode} steps; '
                                   'check environment or increase --max-steps-per-episode')

            intervals = np.diff(hits).tolist() if len(hits) >= 2 else []
            rec = {
                'model': name,
                'checkpoint_actions': int(ck['actions']),
                'episode': episode_idx,
                'env_seed': args.env_seed_base + episode_idx,
                'steps': t,
                'goals': len(hits),
                'reward': round(total_reward, 5),
                'first_hit_step': hits[0] if hits else '',
                'revisit_intervals': json.dumps(intervals),
                'goal_hit_steps': json.dumps(hits),
                'revisit_count': len(intervals),
                'mean_revisit_steps': float(np.mean(intervals)) if intervals else '',
                'mean_action_entropy': entropy_sum / t,
                'visited_bins': int(visited.sum()),
                'action_counts': json.dumps(action_counts.tolist()),
                'goal_memory': args.goal_memory,
            }
            writer.writerow(rec)
            results.append(rec)
            print(f'[{name}] EP {episode_idx+1:3d}/{args.episodes} | '
                  f'goals={len(hits):2d} first={hits[0] if hits else "-":>4} '
                  f'revisits={len(intervals)} entropy={entropy_sum/t:.3f} '
                  f'steps={t}', flush=True)
    finally:
        runner.close()
    return results, ck


def summarize(records):
    n = len(records)
    goals = [r['goals'] for r in records]
    firsts = [r['first_hit_step'] for r in records if r['first_hit_step'] != '']
    revisit_episodes = [r for r in records if r['revisit_count'] > 0]
    pooled_revisits = [d for r in records for d in json.loads(r['revisit_intervals'])]
    counts = np.array([json.loads(r['action_counts']) for r in records], dtype=np.int64)
    return {
        'episodes': n,
        'mean_goals_per_episode': float(np.mean(goals)) if goals else None,
        'total_goals': int(sum(goals)),
        'episodes_with_goal': int(sum(x > 0 for x in goals)),
        'success_rate_first_goal': float(np.mean(np.array(goals) > 0)) if goals else None,
        'episodes_with_revisit': len(revisit_episodes),
        'total_revisits': int(sum(r['revisit_count'] for r in records)),
        'mean_first_hit_steps_successes_only': float(np.mean(firsts)) if firsts else None,
        'mean_revisit_interval_pooled': float(np.mean(pooled_revisits)) if pooled_revisits else None,
        'mean_reward_per_episode': float(np.mean([r['reward'] for r in records])) if n else None,
        'mean_action_entropy': float(np.mean([r['mean_action_entropy'] for r in records])) if n else None,
        'action_counts': counts.sum(axis=0).tolist(),
        'action_fractions': (counts.sum(axis=0) / counts.sum()).tolist() if counts.sum() else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--mode', choices=['check', 'eval'], default='check')
    ap.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='auto')
    ap.add_argument('--renderer', choices=['hardware', 'software'], default='hardware')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--max-steps-per-episode', type=int, default=2000)
    ap.add_argument('--vision-rate', type=float, default=.05)
    ap.add_argument('--env-seed-base', type=int, default=70000)
    ap.add_argument('--motion-seed-base', type=int, default=80000)
    ap.add_argument('--policy-seed-base', type=int, default=90000)
    ap.add_argument('--goal-memory', choices=['on', 'off'], default='on')
    ap.add_argument('--initial-checkpoint', type=Path, default=Path(
        'data/checkpoints/grid1350_online_B_seq_run1/grid_78k_analysis.pt'))
    ap.add_argument('--checkpoint-1m', type=Path, default=DEFAULT_MODELS['1m'])
    ap.add_argument('--checkpoint-21m', type=Path, default=DEFAULT_MODELS['21m'])
    ap.add_argument('--out-dir', type=Path, default=Path('data/evaluations/joint_nav_1m_vs_21m_100ep'))
    args = ap.parse_args()
    if args.episodes <= 0 or args.max_steps_per_episode < 1:
        ap.error('--episodes and --max-steps-per-episode must be positive')
    if not 0 <= args.vision_rate <= 1:
        ap.error('--vision-rate must be in [0,1]')
    for x in [args.initial_checkpoint, args.checkpoint_1m, args.checkpoint_21m]:
        if not x.is_file():
            ap.error(f'Missing file: {x}')
    device = joint.choose_device(args.device)
    model_map = [('1m', args.checkpoint_1m), ('21m', args.checkpoint_21m)]
    print(f'JOINT NAV READ-ONLY EVAL | device={device} mode={args.mode} '
          f'episodes/model={args.episodes} goal_memory={args.goal_memory}', flush=True)
    for label, cp in model_map:
        ck = torch.load(cp, map_location='cpu', weights_only=False)
        print(f'CHECKPOINT {label}: actions={ck["actions"]:,} '
              f'grid_updates={ck["grid_updates"]:,} '
              f'actor_updates={ck["actor_updates"]:,} path={cp}', flush=True)

    if args.mode == 'check':
        for label, cp in model_map:
            _, grid, vision, actor, _ = load_models(device, args.initial_checkpoint, cp)
            n = sum(p.numel() for module in (grid, vision, actor) for p in module.parameters())
            print(f'CHECK PASS | {label} weights matched, finite; params={n:,}', flush=True)
            del grid, vision, actor
        print('CHECK MODE: no DMLab, no evaluation output, no training changes', flush=True)
        return

    # Refuse to overwrite evaluations; no checkpoint/replay state is ever modified.
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f'Evaluation destination is not empty: {args.out_dir}; '
                              'choose a fresh --out-dir')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / 'episodes.csv'
    names = ['model', 'checkpoint_actions', 'episode', 'env_seed', 'steps', 'goals',
             'reward', 'first_hit_step', 'revisit_intervals', 'goal_hit_steps',
             'revisit_count', 'mean_revisit_steps', 'mean_action_entropy',
             'visited_bins', 'action_counts', 'goal_memory']
    all_results = {}
    with output_path.open('w', newline='', buffering=1) as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for label, cp in model_map:
            rows, ck = evaluate_one(label, cp, args.initial_checkpoint, args, device, writer)
            all_results[label] = rows
            fh.flush()
            print(f'SUMMARY {label}: {json.dumps(summarize(rows), ensure_ascii=False)}', flush=True)
    summary = {label: summarize(rows) for label, rows in all_results.items()}
    one = all_results['1m']; twentyone = all_results['21m']
    wins_1m = sum(a['goals'] > b['goals'] for a, b in zip(one, twentyone))
    wins_21m = sum(b['goals'] > a['goals'] for a, b in zip(one, twentyone))
    ties = sum(a['goals'] == b['goals'] for a, b in zip(one, twentyone))
    summary['paired'] = {
        'env_seed_base': args.env_seed_base,
        'goal_memory': args.goal_memory,
        'per_episode_goals_21m_minus_1m': [b['goals'] - a['goals'] for a, b in zip(one, twentyone)],
        'mean_goals_difference_21m_minus_1m': float(np.mean([b['goals'] - a['goals'] for a, b in zip(one, twentyone)])),
        'episodes_1m_more_goals': wins_1m,
        'episodes_21m_more_goals': wins_21m,
        'ties': ties,
        'note': 'Same episode environment seeds, separate policy/motion RNGs with matching per-episode seeds. '
                'Start layouts are paired, but trajectories differ because policies differ.'
    }
    with (args.out_dir / 'summary.json').open('w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print('\n' + '='*74)
    for label in ['1m', '21m']:
        s = summary[label]
        print(f'{label.upper()}: mean goals/ep={s["mean_goals_per_episode"]:.3f}, '
              f'success={s["success_rate_first_goal"]:.1%}, '
              f'episodes with revisit={s["episodes_with_revisit"]}/{s["episodes"]}, '
              f'avg entropy={s["mean_action_entropy"]:.4f}', flush=True)
    print(f'PAIRED 21M - 1M: {summary["paired"]["mean_goals_difference_21m_minus_1m"]:+.3f} goals/ep; '
          f'21M wins={wins_21m}, 1M wins={wins_1m}, ties={ties}', flush=True)
    print(f'OUTPUT: {args.out_dir / "episodes.csv"}, {args.out_dir / "summary.json"}', flush=True)
    print('DONE | Training checkpoints and replay were NEVER modified.', flush=True)


if __name__ == '__main__':
    main()
