#!/usr/bin/env python3
"""Independent A2C sanity-check of the Policy LSTM using oracle spatial codes.

Run from the repository root with its DeepMind Lab virtualenv. This intentionally
bypasses the Grid Network and never changes the original project's source.

Requires levels/square_arena_oracle.lua (made by make_oracle_square_level.py).
The extra ORACLE.GOAL.XY debug observation is *never* fed to the policy before
its first successful goal arrival. This is an oracle-place-code control, NOT
an exact reconstruction of Banino et al.'s Grid Cell Agent or A3C learner.

Example pilot:
python train_oracle_place_agent.py --num-envs 2 --total-env-frames 25600 \
  --vision-checkpoint data/checkpoints/vision_rh_combined_1000/latest.pt \
  --device cpu --out-dir data/checkpoints/oracle_place_smoke

Main:
python train_oracle_place_agent.py --num-envs 32 --total-env-frames 1000000000 \
  --vision-checkpoint data/checkpoints/vision_rh_combined_1000/latest.pt \
  --device cuda --out-dir data/checkpoints/oracle_place_a2c
"""
from __future__ import annotations
import argparse
import csv
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from grid_cells.actor_critic import ActorCriticLSTM, DISCRETE_ACTIONS
from grid_cells.ensembles import build_rl_ensembles
from grid_cells.rl_common import convert_observation, image_to_input, make_dmlab_env


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class OracleRunner:
    def __init__(self, env_cfg, renderer, rank, nenv, seed):
        self.env_cfg = env_cfg
        import deepmind_lab
        from grid_cells.rl_common import OBSERVATIONS
        self.env = deepmind_lab.Lab(
            'square_arena_oracle',
            OBSERVATIONS + ['ORACLE.GOAL.XY'],
            config={
                'width': str(env_cfg.width),
                'height': str(env_cfg.height),
                'levelDirectory': os.path.abspath(env_cfg.level_directory),
                'goalVisible': 'true' if env_cfg.goal_visible else 'false',
            },
            renderer=renderer,
        )
        names = {x['name'] for x in self.env.observation_spec()}
        if 'ORACLE.GOAL.XY' not in names:
            raise RuntimeError('ORACLE.GOAL.XY not defined. Create levels/square_arena_oracle.lua first')
        self.rank, self.nenv, self.seed = rank, nenv, seed
        self.episode = 0

    def obs(self):
        raw = self.env.observations()
        o = convert_observation(raw, self.env_cfg)
        o['oracle_goal_xy'] = np.asarray(raw['ORACLE.GOAL.XY'], dtype=np.float32)
        return o

    def reset(self):
        self.env.reset(seed=self.seed + self.rank + self.nenv * self.episode)
        return self.obs()

    def step(self, action):
        reward = float(self.env.step(DISCRETE_ACTIONS[int(action)],
                                     num_steps=self.env_cfg.action_repeat))
        done = not self.env.is_running()
        if done:
            self.episode += 1
            self.env.reset(seed=self.seed + self.rank + self.nenv * self.episode)
        return self.obs(), reward, done

    def close(self):
        self.env.close()


def worker(pipe, cfg, renderer, rank, nenv, seed):
    torch.set_num_threads(1)
    runner = OracleRunner(cfg, renderer, rank, nenv, seed)
    try:
        while True:
            command, args = pipe.recv()
            if command == 'reset':
                pipe.send(runner.reset())
            elif command == 'step':
                pipe.send(runner.step(args))
            elif command == 'close':
                runner.close()
                pipe.send(None)
                return
    except (EOFError, KeyboardInterrupt):
        runner.close()


def stack(observations):
    return {k: np.stack([o[k] for o in observations]) for k in observations[0]}


class Vec32:
    def __init__(self, cfg, renderer, nenv, seed):
        ctx = mp.get_context('spawn')
        self.nenv = nenv
        self.pipes, self.procs = [], []
        for rank in range(nenv):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=worker, args=(child, cfg, renderer, rank, nenv, seed), daemon=True)
            proc.start()
            child.close()
            self.pipes.append(parent)
            self.procs.append(proc)

    def reset(self):
        for p in self.pipes:
            p.send(('reset', None))
        return stack([p.recv() for p in self.pipes])

    def step(self, actions):
        for pipe, action in zip(self.pipes, actions):
            pipe.send(('step', int(action)))
        out = [p.recv() for p in self.pipes]
        obs = stack([x[0] for x in out])
        rewards = np.asarray([x[1] for x in out], dtype=np.float32)
        done = np.asarray([x[2] for x in out], dtype=np.bool_)
        return obs, rewards, done

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(('close', None))
            except (BrokenPipeError, EOFError):
                pass
        for pipe in self.pipes:
            try:
                pipe.recv()
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()


def trunc_init(model):
    """Grid-project style: all new weights ~ N(0, fan_in^-1), truncated +/-2 sigma."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                fan_in = m.weight[0].numel()
                std = fan_in ** -.5
                nn.init.trunc_normal_(m.weight, std=std, a=-2*std, b=2*std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if name.startswith('weight_'):
                        std = param.shape[1] ** -.5
                        nn.init.trunc_normal_(param, std=std, a=-2*std, b=2*std)
                    elif name.startswith('bias_'):
                        nn.init.zeros_(param)
                h = m.hidden_size
                m.bias_hh_l0[h:2*h].fill_(1.0)  # Grid-model forget-gate offset.


def load_pretrained_cnn(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    weights = ckpt['vision_module']
    stem = 'cnn.'
    cnn_state = {k[len(stem):]: v for k,v in weights.items() if k.startswith(stem)}
    model.cnn.load_state_dict(cnn_state, strict=True)
    print(f'CNN: loaded pretrained vision backbone ({len(cnn_state)} tensors) from {checkpoint_path}', flush=True)


def oracle_codes(obs, pc, hd, device):
    xy = torch.as_tensor(obs['pos'], dtype=torch.float32, device=device)
    angle = torch.as_tensor(obs['hd'], dtype=torch.float32, device=device).reshape(-1)
    return torch.cat([pc.posterior(xy.unsqueeze(1)).squeeze(1),
                      hd.posterior(angle.unsqueeze(1)).squeeze(1)], dim=-1)


def goal_codes(obs, hit_angles, pc, hd, device, goal_hd):
    xy = torch.as_tensor(obs['oracle_goal_xy'], dtype=torch.float32, device=device)
    p = pc.posterior(xy.unsqueeze(1)).squeeze(1)
    if goal_hd == 'arrival':
        h = hd.posterior(hit_angles.unsqueeze(1)).squeeze(1)
    else:
        h = torch.zeros(xy.shape[0], hd.n_cells, device=device)
    return torch.cat([p, h], -1)


def reset_hidden(hidden, done):
    keep = (~done).to(dtype=hidden[0].dtype).view(1, -1, 1)
    return tuple(h * keep for h in hidden)


def compute_returns(rewards, done, bootstrap, gamma):
    # Goal rewards do not terminate an episode; only the 5400-frame timeout does.
    running = bootstrap
    result = []
    for r, d in zip(reversed(rewards), reversed(done)):
        running = r + gamma * running * (~d).float()
        result.append(running)
    return torch.stack(result[::-1])


def save_checkpoint(out, frames, updates, model, opt, args):
    state = {'frames': frames, 'updates': updates,
             'actor_critic': model.state_dict(), 'optimizer': opt.state_dict(),
             'args': vars(args), 'algorithm': 'oracle_place_A2C',
             'torch_rng_state': torch.get_rng_state(),
             'numpy_rng_state': np.random.get_state(),
             'python_rng_state': random.getstate()}
    if torch.cuda.is_available():
        state['cuda_rng_states'] = torch.cuda.get_rng_state_all()
    tmp = out/'latest.tmp.pt'
    torch.save(state, tmp)
    os.replace(tmp, out/'latest.pt')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--num-envs', type=int, default=32)
    ap.add_argument('--total-env-frames', type=int, default=1_000_000_000)
    ap.add_argument('--device', choices=('cuda','cpu','auto'), default='cuda')
    ap.add_argument('--renderer', choices=('hardware','software'), default='hardware')
    ap.add_argument('--vision-checkpoint', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, default=Path('data/checkpoints/oracle_place_a2c'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-step', type=int, default=100)
    ap.add_argument('--action-repeat', type=int, default=4)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--lr', type=float, default=0.0001005)  # Midpoint Table 2.
    ap.add_argument('--value-cost', type=float, default=0.5)
    ap.add_argument('--entropy-cost', type=float, default=0.00008)
    ap.add_argument('--rmsprop-momentum', type=float, default=0.0,
                    help='0 = stable independent repo A2C baseline; 0.99 = literal Table 2')
    ap.add_argument('--finetune-cnn', action='store_true', help='default freeze pretrained CNN')
    ap.add_argument('--goal-hd', choices=('arrival','zero'), default='arrival',
                    help='arrival = pre-hit observed heading (approx. at contact), zero = no goal HD')
    ap.add_argument('--log-every', type=int, default=20)
    ap.add_argument('--save-every-frames', type=int, default=10_000_000)
    ap.add_argument('--resume', type=Path, default=None,
                    help='restart from checkpoint; DMLab episodes restart, not bit-identical')
    args = ap.parse_args()
    if args.num_envs < 1 or args.n_step < 1 or args.action_repeat < 1:
        ap.error('num-envs, n-step, action-repeat must be positive')
    if args.total_env_frames < 1:
        ap.error('total-env-frames must be positive')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    cfg = Config()
    cfg.rl.env.goal_visible = False
    cfg.rl.env.action_repeat = args.action_repeat
    cfg.rl.env.level = 'square_arena_oracle'
    assert cfg.rl.env.env_size_m==2.5, 'Use 2.5m square environment'
    (pc,), (hd,) = build_rl_ensembles(cfg, device)
    dim = pc.n_cells + hd.n_cells
    model = ActorCriticLSTM(grid_code_dim=dim, embed_dim=256, lstm_units=256).to(device)
    trunc_init(model)
    load_pretrained_cnn(model, args.vision_checkpoint)
    if not args.finetune_cnn:
        for p in model.cnn.parameters():
            p.requires_grad_(False)
    model.train()
    opt = torch.optim.RMSprop((p for p in model.parameters() if p.requires_grad),
                              lr=args.lr, alpha=0.99, eps=1e-8, momentum=args.rmsprop_momentum)
    frame_count, update_count = 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint['algorithm'] != 'oracle_place_A2C':
            raise ValueError('Incompatible checkpoint')
        model.load_state_dict(checkpoint['actor_critic'])
        opt.load_state_dict(checkpoint['optimizer'])
        frame_count,update_count = int(checkpoint['frames']),int(checkpoint['updates'])
        print('RESUMED weights and optimizer; DMLab environments restart new episodes.',flush=True)
    print(f'Oracle Place Agent: {args.num_envs} workers, A2C, {args.n_step} agent actions / update, '
          f'{args.action_repeat} frames/action. {dim}-dim current and goal codes.', flush=True)
    print('Goal image hidden; oracle goal xy held out of policy until first reward.', flush=True)
    print('No gradient clipping. PC/HD ground truth; pretrained policy CNN weights copied, not shared.',flush=True)
    vec=Vec32(cfg.rl.env, args.renderer, args.num_envs, args.seed+1000)
    last_saved = frame_count
    try:
        obs=vec.reset()
        n=args.num_envs
        hidden=model.init_hidden(n,device)
        prev_a=torch.zeros(n,dtype=torch.long,device=device)
        prev_r=torch.zeros(n,1,device=device)
        remembered_goal=torch.zeros(n,dim,device=device)
        ep_frames=np.zeros(n,dtype=np.int64)
        ep_goals=np.zeros(n,dtype=np.int64)
        ep_first=np.full(n,-1,dtype=np.int64)
        ep_reward=np.zeros(n,dtype=np.float32)
        logfile=args.out_dir/'updates.csv'
        episode_file=args.out_dir/'episodes.csv'
        with logfile.open('a',newline='',buffering=1) as logf, episode_file.open('a',newline='',buffering=1) as epf:
            logs=csv.writer(logf)
            episodes=csv.writer(epf)
            if logfile.stat().st_size==0:
                logs.writerow(['total_env_frames','optimizer_updates','loss','policy_loss','value_loss',
                               'entropy','mean_reward','grad_norm','fps'])
            if episode_file.stat().st_size==0:
                episodes.writerow(['total_env_frames','env_id','episode_reward','goals','first_goal_frame'])
            start_wall=time.monotonic()
            while frame_count < args.total_env_frames:
                logps,vals,ents,rews,dones=[],[],[],[],[]
                observed_goals=0
                for _ in range(args.n_step):
                    # No gradient through ground-truth spatial targets.
                    with torch.no_grad():
                        spatial = oracle_codes(obs, pc, hd, device)
                    image = image_to_input(torch.as_tensor(obs['image'],device=device))
                    output = model.step(image, prev_r, prev_a, spatial, remembered_goal, hidden)
                    dist=torch.distributions.Categorical(logits=output.action_logits)
                    action=dist.sample()
                    next_obs,reward_np,done_np=vec.step(action.detach().cpu().numpy())
                    reward=torch.as_tensor(reward_np,device=device)
                    done=torch.as_tensor(done_np,device=device,dtype=torch.bool)
                    goal=reward>0
                    observed_goals += int(goal.sum().item())
                    logps.append(dist.log_prob(action))
                    vals.append(output.value)
                    ents.append(dist.entropy())
                    rews.append(reward)
                    dones.append(done)
                    # The goal coordinate is a debug label. Never feed it to policy
                    # until after successful contact, and then only via the spatial code.
                    if goal.any():
                        with torch.no_grad():
                            pre_heading=torch.as_tensor(obs['hd'],device=device).reshape(-1)
                            candidate=goal_codes(next_obs,pre_heading,pc,hd,device,args.goal_hd)
                            remembered_goal=torch.where(goal[:,None],candidate,remembered_goal)
                    hidden=reset_hidden(output.hidden,done)
                    remembered_goal=torch.where(done[:,None],0.,remembered_goal)
                    prev_a=torch.where(done,torch.zeros_like(action),action)
                    prev_r=torch.where(done[:,None],0.,reward[:,None])
                    ep_frames += args.action_repeat
                    ep_reward += reward_np
                    ep_goals += (reward_np>0)
                    new_first=(reward_np>0) & (ep_first<0)
                    ep_first[new_first]=ep_frames[new_first]
                    for k in np.flatnonzero(done_np):
                        episodes.writerow([frame_count+n*args.action_repeat,int(k),
                                          float(ep_reward[k]),int(ep_goals[k]),int(ep_first[k])])
                        ep_frames[k]=0
                        ep_reward[k]=0
                        ep_goals[k]=0
                        ep_first[k]=-1
                    frame_count += n*args.action_repeat
                    obs=next_obs
                # BPTT=100 action steps, not 100 environment frames.
                with torch.no_grad():
                    obs_code=oracle_codes(obs,pc,hd,device)
                    img=image_to_input(torch.as_tensor(obs['image'],device=device))
                    bootstrap=model.step(img,prev_r,prev_a,obs_code,remembered_goal,hidden).value
                    returns=compute_returns(rews,dones,bootstrap,args.gamma)
                logp=torch.stack(logps)
                value=torch.stack(vals)
                entropy=torch.stack(ents)
                advantage=returns-value.detach()
                policy_loss=-(logp*advantage).mean()
                value_loss=F.mse_loss(value,returns)
                loss=policy_loss+args.value_cost*value_loss-args.entropy_cost*entropy.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # Diagnostics only: compute norm, do not clip.
                with torch.no_grad():
                    grad_norm=math.sqrt(sum(float((p.grad.float()**2).sum().item())
                                            for p in model.parameters() if p.grad is not None))
                if not math.isfinite(grad_norm):
                    raise RuntimeError('Non-finite gradient; no gradient clipping requested')
                opt.step()
                hidden=tuple(x.detach() for x in hidden)
                remembered_goal=remembered_goal.detach()
                update_count += 1
                elapsed=max(time.monotonic()-start_wall,1e-6)
                fps=(frame_count-last_saved)/elapsed
                logs.writerow([frame_count,update_count,float(loss.item()),float(policy_loss.item()),
                               float(value_loss.item()),float(entropy.mean().item()),
                               float(torch.stack(rews).mean().item()),grad_norm,round(fps,1)])
                if update_count % args.log_every==0:
                    print(f'frame {frame_count:,} update {update_count:,} | '
                          f'loss {loss.item():.4f} H {entropy.mean().item():.4f} '
                          f'goals/rollout {observed_goals} grad {grad_norm:.3f}',flush=True)
                if frame_count-last_saved>=args.save_every_frames:
                    save_checkpoint(args.out_dir,frame_count,update_count,model,opt,args)
                    last_saved=frame_count
                    start_wall=time.monotonic()
            save_checkpoint(args.out_dir,frame_count,update_count,model,opt,args)
    except KeyboardInterrupt:
        save_checkpoint(args.out_dir,frame_count,update_count,model,opt,args)
        print(f'Interrupted: saved at frame {frame_count:,}',flush=True)
    finally:
        vec.close()


if __name__=='__main__':
    mp.freeze_support()
    main()
