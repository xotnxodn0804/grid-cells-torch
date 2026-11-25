#!/usr/bin/env python3
"""Read-only, matched-seed 80M vs ~169M Joint-32 navigation evaluation.

Save into the Windows repository's results directory and run from WSL repo root.
No checkpoint, optimizer, or replay data is modified.

Modes:
  python -u /mnt/c/Users/xotnd/GitHubProjects/grid-cells-torch/results/eval_joint32_80m_vs_169m.py --mode check
  python -u /mnt/c/Users/xotnd/GitHubProjects/grid-cells-torch/results/eval_joint32_80m_vs_169m.py --mode eval --episodes 100

Required in ~/grid-cells-torch: train_joint_nav_online.py, train_joint_nav_32actors.py,
train_oracle_place_agent.py, pretrained B checkpoint and both 32-actor checkpoints.

Evaluation-only goal coordinates are used for metrics/plots, NEVER as model inputs.
One Episode = 1350 action decisions = 5400 DMLab frames = 90 simulated seconds.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# When script is stored on Windows, allow imports from the WSL working directory.
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import torch
import torch.nn.functional as F

MODELS = ('80M', '169M')
RAW_FIELDS = ['model','episode','action_step','x','y','yaw_rad','debug_goal_x',
              'debug_goal_y','action','reward','prior_hits','since_teleport_actions',
              'next_x','next_y']
EP_FIELDS = ['model','episode','goals','first_hit_action','revisits']
ATTEMPT_FIELDS = ['model','episode','attempt','start_step','elapsed_actions',
                  'success','censored','distance_m','distance_bin','actions_per_m',
                  'spawn_x','spawn_y','travel_distance_m','path_efficiency']
DIST_EDGES = [0, .5, 1, 1.5, float('inf')]
DIST_LABELS = ['0-0.5m', '0.5-1.0m', '1.0-1.5m', '>=1.5m']


def read_ckpt(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {path}')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    required = {'algorithm','frames','grid','vision','actor','initial_sha256'}
    missing = required - set(ck)
    if missing:
        raise RuntimeError(f'{path} missing keys: {missing}')
    return ck


def setup(args):
    from train_joint_nav_online import make_models, GridCodeActorCritic
    from train_joint_nav_32actors import seed_all, choose_device, digest, ALGORITHM
    if not args.initial_checkpoint.is_file():
        raise FileNotFoundError(f'Missing initial supervised checkpoint: {args.initial_checkpoint}')
    ck = {'80M':read_ckpt(args.old_checkpoint), '169M':read_ckpt(args.new_checkpoint)}
    for name, c in ck.items():
        if c['algorithm'] != ALGORITHM:
            raise RuntimeError(f'{name}: wrong algorithm {c["algorithm"]!r}')
        if int(c.get('num_envs',32)) != 32:
            raise RuntimeError(f'{name}: expected 32 actors')
    if ck['80M']['initial_sha256'] != ck['169M']['initial_sha256']:
        raise RuntimeError('Model family mismatch: different pretrained initial weights')
    if digest(args.initial_checkpoint) != ck['80M']['initial_sha256']:
        raise RuntimeError('Initial supervised checkpoint SHA256 differs. STOP.')
    seed0 = int(ck['80M'].get('args',{}).get('seed',42))
    seed1 = int(ck['169M'].get('args',{}).get('seed',42))
    if seed0 != seed1:
        raise RuntimeError(f'Incompatible training seeds: {seed0}, {seed1}')
    f0, f1 = int(ck['80M']['frames']), int(ck['169M']['frames'])
    if f0 != 80_000_000 or not (160_000_000 <= f1 <= 175_000_000):
        raise RuntimeError(f'Unexpected checkpoints: old={f0:,}, new={f1:,}; '
                           'check paths to avoid mislabeled results')
    if f1 <= f0:
        raise RuntimeError('New checkpoint is not newer than old one')
    print(f'CHECKPOINT 80M  | {args.old_checkpoint}  frames={f0:,}', flush=True)
    print(f'CHECKPOINT 169M | {args.new_checkpoint}  frames={f1:,}', flush=True)
    print('VERIFY | Same model family, training seed, B-model SHA256; no checkpoint writes', flush=True)
    from train_joint_nav_32actors import choose_device
    device = choose_device(args.device)
    seed_all(seed0)
    cfg, _, _, grid0, vision0 = make_models(device, args.initial_checkpoint)
    actor0 = GridCodeActorCritic().to(device)
    networks = {}
    for name in MODELS:
        g = copy.deepcopy(grid0)
        v = copy.deepcopy(vision0)
        a = copy.deepcopy(actor0)
        g.load_state_dict(ck[name]['grid'], strict=True)
        v.load_state_dict(ck[name]['vision'], strict=True)
        a.load_state_dict(ck[name]['actor'], strict=True)
        g.eval(); v.eval(); a.eval()
        networks[name] = (g,v,a)
    print(f'CHECK PASS | loaded both networks on {device}; '
          f'100 episodes would be {2 * args.episodes * 1350:,} action decisions',flush=True)
    return cfg, networks, device, f0, f1


def make_runner(cfg, renderer, seed):
    from train_oracle_place_agent import OracleRunner
    ecfg=cfg.rl.env
    ecfg.level='square_arena_oracle'
    ecfg.goal_visible=False
    ecfg.action_repeat=4
    if (ecfg.env_size_m,ecfg.width,ecfg.height)!=(2.5,64,64):
        raise ValueError('Environment mismatch: expected 2.5m, 64x64')
    if not (Path(ecfg.level_directory)/'square_arena_oracle.lua').is_file():
        raise FileNotFoundError('Missing levels/square_arena_oracle.lua')
    return OracleRunner(ecfg,renderer,0,1,seed)


def model_observation(obs):
    from train_joint_nav_32actors import drop_oracle
    return {k:np.expand_dims(v,axis=0) for k,v in drop_oracle(obs).items()}


@torch.inference_mode()
def evaluate_one(name, networks, cfg, args, device, sw, ew):
    from train_joint_nav_32actors import (batched_motions,batched_grid_step,
                                          actor_hidden,grid_hidden_init)
    grid,vision,actor=networks
    runner=make_runner(cfg,args.renderer,args.env_seed_base)
    episodes=[]
    try:
        obs=runner.reset()
        for ep in range(args.episodes):
            rng=np.random.default_rng(args.motion_seed_base+ep)
            action_rng=torch.Generator(device=device)
            action_rng.manual_seed(args.action_seed_base+ep)
            gh=grid_hidden_init(1,device)
            ah=actor_hidden(1,device)
            prev_a=torch.zeros(1,dtype=torch.long,device=device)
            prev_r=torch.zeros(1,1,device=device)
            goal_mem=torch.zeros(1,512,device=device)
            first_known=False
            last_obs=None
            force_reset=True
            hits=0
            first_hit=None
            since_tp=0
            done=False
            for t in range(1,1351):
                curr_b=model_observation(obs)
                prev_b=None if last_obs is None else model_observation(last_obs)
                reset=np.array([force_reset],dtype=np.bool_)
                if last_obs is not None and np.linalg.norm(obs['pos']-last_obs['pos'])>.35:
                    reset[0]=True
                motion=batched_motions(curr_b,prev_b,reset,rng)
                visual_mask=reset | (rng.random(1)<args.vision_rate)
                code,gh=batched_grid_step(grid,vision,curr_b,motion,visual_mask,
                                          gh,reset,device)
                logits,_,ah=actor.step(code,goal_mem,prev_a,prev_r,ah)
                probs=F.softmax(logits,dim=-1)
                action=int(torch.multinomial(probs[0],1,generator=action_rng).item())
                nxt,reward,done=runner.step(action)
                if 'oracle_goal_xy' not in obs:
                    raise RuntimeError('Evaluation-only goal coordinates unavailable')
                gx,gy=np.asarray(obs['oracle_goal_xy'],dtype=np.float64).reshape(2)
                record={
                    'model':name,'episode':ep,'action_step':t,
                    'x':float(obs['pos'][0]),'y':float(obs['pos'][1]),
                    'yaw_rad':float(obs['hd'][0]),
                    'debug_goal_x':float(gx),'debug_goal_y':float(gy),
                    'action':action,'reward':float(reward),
                    'prior_hits':hits,'since_teleport_actions':since_tp,
                    'next_x':float(nxt['pos'][0]),'next_y':float(nxt['pos'][1]),
                }
                sw.writerow(record)
                if reward>0:
                    hits+=1
                    if first_hit is None:
                        first_hit=t
                    if not first_known and args.goal_memory=='on':
                        goal_mem=code.detach().clone()  # pre-hit grid code; training semantics
                        first_known=True
                prev_a=torch.tensor([action],dtype=torch.long,device=device)
                prev_r=torch.tensor([[reward]],dtype=torch.float32,device=device)
                force_reset=bool(reward>0 or done)
                since_tp=0 if reward>0 else since_tp+1
                last_obs,obs=obs,nxt
                if done:
                    break
            if not done:
                raise RuntimeError(f'Episode {ep} did not terminate by action 1350')
            out={'model':name,'episode':ep,'goals':hits,
                 'first_hit_action':first_hit if first_hit is not None else '',
                 'revisits':max(0,hits-1)}
            episodes.append(out)
            ew.writerow(out)
            if (ep+1)%5==0 or ep==0:
                print(f'EVAL | {name} {ep+1}/{args.episodes} | goals={hits} '
                      f'first-hit={first_hit or "none"}',flush=True)
    finally:
        runner.close()
    return episodes


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def goal_y_sign(rows):
    hits=[r for r in rows if float(r['reward'])>0]
    if not hits:
        raise RuntimeError('No successful goal hits; cannot calibrate goal coordinates')
    scores={s:float(np.median([math.hypot(float(r['x'])-float(r['debug_goal_x']),
                       float(r['y'])-s*float(r['debug_goal_y'])) for r in hits]))
            for s in (1,-1)}
    sign=min(scores,key=scores.get)
    print(f'GOAL AXIS AUDIT | sign={sign}, median hit distance={scores[sign]:.3f}m '
          f'(raw={scores[1]:.3f}, flip={scores[-1]:.3f})')
    if scores[sign]>.40:
        raise RuntimeError('Neither goal coordinate convention matches hit positions')
    return sign,scores


def heading_calibration(rows):
    observations=[]
    for r in rows:
        if int(r['action']) not in (2,3) or float(r['reward'])>0:continue
        dx=float(r['next_x'])-float(r['x'])
        dy=float(r['next_y'])-float(r['y'])
        d=math.hypot(dx,dy)
        if d<.003 or d>.08:continue
        if int(r['action'])==3:dx,dy=-dx,-dy
        observations.append((float(r['yaw_rad']),math.atan2(dy,dx)))
    if len(observations)<20:
        raise RuntimeError('Insufficient unblocked forward/backward movement samples')
    possibilities=[]
    for s in (1,-1):
        for deg in (0,90,180,270):
            offset=math.radians(deg)
            score=float(np.mean([math.cos(s*yaw+offset-actual) for yaw,actual in observations]))
            possibilities.append((score,s,deg))
    score,sign,off=max(possibilities)
    if score<.30:
        raise RuntimeError(f'Head-direction axis calibration failed (score {score:.3f})')
    print(f'HEADING AXIS AUDIT | sign={sign} offset={off}deg score={score:.3f} '
          f'(n={len(observations):,})')
    return {'sign':sign,'offset_deg':off,'mean_forward_cosine':score,'samples':len(observations)}


def heading_error(r,goal_sign,heading):
    angle=math.atan2(goal_sign*float(r['debug_goal_y'])-float(r['y']),
                     float(r['debug_goal_x'])-float(r['x']))
    front=heading['sign']*float(r['yaw_rad'])+math.radians(heading['offset_deg'])
    return abs(math.degrees(math.atan2(math.sin(angle-front),math.cos(angle-front))))


def distance_bin(d):
    for lo,hi,label in zip(DIST_EDGES[:-1],DIST_EDGES[1:],DIST_LABELS):
        if lo<=d<hi:return label
    raise ValueError(f'Invalid distance: {d}')


def finalize_attempt(attempt, last_t, success, previous_segment):
    out={k:v for k,v in attempt.items() if k!='goal_y_sign'}
    out['elapsed_actions']=last_t-attempt['start_step']
    out['success']=bool(success)
    out['censored']=not bool(success)
    # Start at the true POST-hit teleport position, not the old pre-hit point.
    # End at the goal center for successful attempts (approximate final movement).
    pts=[(attempt['spawn_x'],attempt['spawn_y'])]
    pts.extend((float(r['x']),float(r['y'])) for r in previous_segment)
    if success and previous_segment:
        last=previous_segment[-1]
        pts.append((float(last['debug_goal_x']),
                    attempt['goal_y_sign']*float(last['debug_goal_y'])))
    travel=sum(math.dist(p,q) for p,q in zip(pts,pts[1:]))
    out['travel_distance_m']=travel
    out['actions_per_m']=(out['elapsed_actions']/out['distance_m'] if
                          success and out['distance_m']>=.15 else '')
    out['path_efficiency']=(out['distance_m']/travel if
                            success and travel>0.005 else '')
    return out


def extract_returns(raw,sign):
    """First hit is discovery, later hits are returns; final incomplete attempt is censored.

    Return start coordinates are POST-goal-hit teleport; not pre-hit position.
    The path-efficiency ratio is only a rough measurement from pre-action positions.
    """
    grouped=defaultdict(list)
    for r in raw:grouped[(r['model'],int(r['episode']))].append(r)
    attempts=[]
    for (name,ep),steps in grouped.items():
        active=None
        segment=[]
        ix=0
        for r in steps:
            t=int(r['action_step'])
            reward=float(r['reward'])
            if active is not None:segment.append(r)
            if reward>0:
                if active is not None:
                    attempts.append(finalize_attempt(active,t,True,segment))
                ix+=1
                active=None
                segment=[]
                if t<1350:
                    gx=float(r['debug_goal_x']); gy=sign*float(r['debug_goal_y'])
                    spawn=(float(r['next_x']),float(r['next_y']))
                    d=math.dist(spawn,(gx,gy))
                    active={'model':name,'episode':ep,'attempt':ix,'start_step':t,
                            'distance_m':d,'distance_bin':distance_bin(d),
                            'spawn_x':spawn[0],'spawn_y':spawn[1],
                            'goal_y_sign':sign}
        if active is not None:
            attempts.append(finalize_attempt(active,int(steps[-1]['action_step']),False,segment))
    return attempts


def horizons(attempts, h):
    successes=sum(bool(a['success']) and a['elapsed_actions']<=h for a in attempts)
    failures=sum(a['elapsed_actions']>=h and
                 not (bool(a['success']) and a['elapsed_actions']<=h) for a in attempts)
    excluded=len(attempts)-successes-failures
    return {'actions':h,'seconds':round(h*4/60,3),'successes':successes,
            'observable_attempts':successes+failures,'censored_early':excluded,
            'rate':successes/(successes+failures) if successes+failures else None}


def episode_summary(items):
    goals=np.array([int(x['goals']) for x in items])
    first=[int(x['first_hit_action']) for x in items if x['first_hit_action']!='']
    return {'episodes':len(items),'mean_goals_per_episode':float(goals.mean()),
            'median_goals_per_episode':float(np.median(goals)),
            'total_goals':int(goals.sum()),
            'first_hit_success_rate':float(np.mean(goals>0)),
            'mean_first_hit_actions_success_only':float(np.mean(first)) if first else None,
            'mean_first_hit_seconds_success_only':float(np.mean(first)*4/60) if first else None,
            'total_revisits':int(np.maximum(goals-1,0).sum()),
            'no_goal_episodes':int(np.sum(goals==0))}


def bootstrap_delta(ep_rows, n_boot=5000, seed=321):
    a={int(r['episode']):int(r['goals']) for r in ep_rows if r['model']=='80M'}
    b={int(r['episode']):int(r['goals']) for r in ep_rows if r['model']=='169M'}
    if a.keys()!=b.keys():
        raise RuntimeError('Episode IDs differ between models')
    diffs=np.array([b[i]-a[i] for i in sorted(a)],dtype=np.float64)
    rng=np.random.default_rng(seed)
    means=np.empty(n_boot,dtype=np.float64)
    for k in range(n_boot):
        means[k]=float(np.mean(rng.choice(diffs,len(diffs),replace=True)))
    return {'paired_mean_delta_goals_per_ep':float(diffs.mean()),
            'paired_bootstrap_95pct_CI':[float(x) for x in np.quantile(means,[.025,.975])],
            'episodes_169m_better':int((diffs>0).sum()),
            'episodes_80m_better':int((diffs<0).sum()),
            'episodes_tied':int((diffs==0).sum()),
            'interpretation':'Descriptive paired-bootstrap CI; same initial seeds but '
                            'independent stochastic policies, NOT a causal mechanism test.'}


def heading_stats(raw,goal_sign,heading):
    data=defaultdict(lambda:defaultdict(list))
    for r in raw:
        if int(r['prior_hits'])<1:continue
        age=int(r['since_teleport_actions'])
        if age<=30:
            data[r['model']][age].append(heading_error(r,goal_sign,heading))
    return {m:{str(i):{'n':len(data[m][i]),
                        'mean_error_deg':float(np.mean(data[m][i])) if data[m][i] else None,
                        'fraction_30deg':float(np.mean(np.array(data[m][i])<=30))
                                         if data[m][i] else None}
               for i in range(31)} for m in MODELS}


def read_training_csv(path):
    if not path.is_file():return []
    rows=read_csv(path)
    data=[]
    prev=None
    for r in rows:
        try:
            f=int(r['frames']); hits=int(r['goal_hits_since_last_log'])
        except (KeyError,ValueError):continue
        # Only ten-rollout (= 128k) intervals are directly comparable.
        if prev is not None and f-prev==128_000:
            data.append((f/1e6,hits/(128_000/5400)))
        prev=f
    return data


def make_figures(out,raw,ep_rows,attempts,summary,args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors={'80M':'#718096','169M':'#277db8'}
    plt.rcParams.update({'figure.facecolor':'white','savefig.facecolor':'white',
                         'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})

    # 1. Whole episode metrics: means, episode hit count hist and first hit rate.
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    keys=['80M','169M']
    ax=axes[0]
    ax.bar(keys,[summary['behavior'][m]['mean_goals_per_episode'] for m in keys],
           color=[colors[m] for m in keys],width=.58)
    ax.set_title('Goals / 90-s episode');ax.set_ylabel('Mean goal hits');ax.set_ylim(bottom=0)
    for i,m in enumerate(keys):
        ax.text(i,summary['behavior'][m]['mean_goals_per_episode']+.1,
                f'{summary["behavior"][m]["mean_goals_per_episode"]:.2f}',ha='center')
    ax=axes[1]
    d=[int(r['goals']) for r in ep_rows]
    top=max(5,max(d))
    for m in keys:
        data=[int(r['goals']) for r in ep_rows if r['model']==m]
        ax.hist(data,bins=np.arange(-.5,top+1.5,1),alpha=.56,label=m,color=colors[m])
    ax.set(title='Episode goal-hit distribution',xlabel='Goals / episode',ylabel='Episodes')
    ax.legend(frameon=False)
    ax=axes[2]
    rate=[summary['behavior'][m]['first_hit_success_rate']*100 for m in keys]
    ax.bar(keys,rate,width=.58,color=[colors[m] for m in keys]);ax.set_ylim(0,110)
    ax.set(title='First goal found',ylabel='Episodes with >=1 hit (%)')
    for i,v in enumerate(rate):ax.text(i,v+2,f'{v:.0f}%',ha='center')
    fig.suptitle('Joint-32 | Same evaluation episode seeds',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(out/'behavior_comparison.png',dpi=220);fig.savefig(out/'behavior_comparison.pdf')
    plt.close(fig)

    # 2. Return attempt times; avoid treating right-censored attempts as failures at H.
    fig,axes=plt.subplots(1,2,figsize=(10.3,3.9))
    for m in keys:
        xs=[a['elapsed_actions']*4/60 for a in attempts if a['model']==m and a['success']]
        if xs:
            axes[0].hist(xs,bins=np.linspace(0,90,31),alpha=.55,label=f'{m} (n={len(xs)})',color=colors[m])
    axes[0].set(xlabel='Return time (simulated seconds)',ylabel='Successful attempts',
                title='Successful returns (descriptive)');axes[0].legend(frameon=False)
    horizons_sec=[10,20,40]
    xs=np.arange(len(horizons_sec))
    for i,m in enumerate(keys):
        vals=[]
        for h in [150,300,600]:
            st=summary['return_horizons'][m][str(h)]
            vals.append(100*st['rate'] if st['rate'] is not None else np.nan)
        axes[1].bar(xs+(-.18 if i==0 else .18),vals,.36,label=m,color=colors[m])
    axes[1].set_xticks(xs,[f'{s}s' for s in horizons_sec])
    axes[1].set(xlabel='Return horizon',ylabel='Successful returns / observable attempts (%)',
                ylim=(0,100),title='Return probability by horizon*')
    axes[1].legend(frameon=False)
    fig.text(.5,.004,'* Attempts censored before the horizon are excluded; unequal attempts and censoring may bias comparisons.',
             ha='center',fontsize=8,color='#677384')
    fig.tight_layout(rect=(0,.055,1,1))
    fig.savefig(out/'return_analysis.png',dpi=220);fig.savefig(out/'return_analysis.pdf')
    plt.close(fig)

    # 3. Trajectories: top 3 per model, selected separately; split after each teleport.
    met={(r['model'],int(r['episode'])):r for r in ep_rows}
    by_ep=defaultdict(list)
    for r in raw:by_ep[(r['model'],int(r['episode']))].append(r)
    fig,axs=plt.subplots(2,3,figsize=(11.4,7.9),sharex=True,sharey=True)
    selections={}
    for ri,m in enumerate(keys):
        candidates=sorted((r for r in ep_rows if r['model']==m),
                          key=lambda r:(-int(r['goals']),int(r['first_hit_action'] or 10**9),int(r['episode'])))
        selections[m]=[int(r['episode']) for r in candidates[:3]]
        for ci,ep in enumerate(selections[m]):
            ax=axs[ri,ci];records=by_ep[(m,ep)]
            gx=float(records[0]['debug_goal_x']);gy=summary['goal_y_sign']*float(records[0]['debug_goal_y'])
            explored=[r for r in records if int(r['prior_hits'])==0]
            if explored:
                ax.plot([float(r['x']) for r in explored],[float(r['y']) for r in explored],
                        ls='--',lw=1.25,color='#90949b',alpha=.72)
            for k in sorted(set(int(r['prior_hits']) for r in records if int(r['prior_hits'])>=1)):
                seg=[r for r in records if int(r['prior_hits'])==k]
                if not any(float(r['reward'])>0 for r in seg):continue
                seg=seg[:next(i for i,r in enumerate(seg) if float(r['reward'])>0)+1]
                if k>5:break
                ax.plot([float(r['x']) for r in seg],[float(r['y']) for r in seg],
                        color='#277db8',alpha=max(.45,1-k*.08),lw=1.3)
                ax.scatter([float(seg[0]['x'])],[float(seg[0]['y'])],marker='s',
                           s=16,color='#277db8',zorder=4)
            ax.scatter([gx],[gy],s=110,marker='*',color='#d04b51',zorder=6)
            ax.set(xlim=(-1.3,1.3),ylim=(-1.3,1.3),aspect='equal',xlabel='x (m)')
            ax.set_title(f'{m} | Episode {ep} | {met[(m,ep)]["goals"]} hits',fontsize=10)
            ax.grid(alpha=.13)
    for ax in axs[:,0]:ax.set_ylabel('y (m)')
    fig.legend([Line2D([0],[0],ls='--',color='#90949b'),
                Line2D([0],[0],color='#277db8'),
                Line2D([0],[0],marker='s',ls='',color='#277db8'),
                Line2D([0],[0],marker='*',ls='',color='#d04b51',markersize=12)],
               ['First search','Up to 5 returns','Teleport spawn','Goal'],
               loc='lower center',ncol=4,frameon=False)
    fig.suptitle('Top 3 per model: illustrative episodes, NOT average behavior',fontsize=13)
    fig.tight_layout(rect=(0,.065,1,.96))
    fig.savefig(out/'top3_trajectories.png',dpi=230);fig.savefig(out/'top3_trajectories.pdf')
    plt.close(fig)
    summary['top3_episode_ids']=selections

    # 4. Heading is a secondary analysis, not proof of vector navigation.
    fig,ax=plt.subplots(figsize=(7.4,4.1))
    for m in keys:
        hs=summary['heading_alignment'][m]
        xs=[i for i in range(31) if hs[str(i)]['n']>0]
        ys=[hs[str(i)]['mean_error_deg'] for i in xs]
        ax.plot(xs,ys,label=m,lw=2,color=colors[m])
    ax.axhline(90,color='gray',ls=':',lw=1,label='Random heading mean: 90 deg')
    ax.set(xlabel='Actions since teleport (15 actions ~1 s)',
           ylabel='Mean absolute heading-to-goal error (deg)',
           ylim=(0,180),title='Head orientation after teleport | known goal')
    ax.grid(alpha=.14);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(out/'heading_after_teleport.png',dpi=220)
    plt.close(fig)

    # 5. Recombine independent original/recovered training logs without bridging resume gap.
    chunks=[]
    for name,loc in [('original',args.old_training_metrics),('recovered',args.new_training_metrics)]:
        sub=read_training_csv(loc)
        if sub:
            chunks.append((name,sub))
    if chunks:
        fig,ax=plt.subplots(figsize=(9.2,4.5))
        for name,chunk in chunks:
            xs=np.asarray([p[0] for p in chunk]);ys=np.asarray([p[1] for p in chunk])
            smooth=np.asarray([np.mean(ys[max(0,i-38):i+1]) for i in range(len(ys))])
            ax.plot(xs,ys,color='#90b4d2',alpha=.19,lw=.6)
            ax.plot(xs,smooth,lw=2,label=f'{name} (rolling ~5M frames)')
        ax.axvline(80,color='gray',ls='--',lw=1,label='80M replay reset')
        ax.axvline(summary['checkpoint_frames']['169M']/1e6,color='#333',ls=':',lw=1,
                   label='Final evaluated checkpoint')
        ax.set(xlabel='Environment frames summed across all actors (millions)',
               ylabel='Training goals / 90-s episode equivalent',
               title='32-actor learning curve (separate original and recovered CSVs)')
        ax.set_ylim(bottom=0);ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=9)
        fig.tight_layout();fig.savefig(out/'training_curve.png',dpi=220)
        plt.close(fig)
    else:
        print('NOTICE | training CSV missing; skipping learning curve')


def summary_text(out,s):
    old=s['behavior']['80M'];new=s['behavior']['169M'];paired=s['paired_difference']
    txt=[
        'JOINT-32: 80M vs 169M | MATCHED-SEED EVALUATION',
        f"Frames: 80M={s['checkpoint_frames']['80M']:,}, 169M={s['checkpoint_frames']['169M']:,}",
        f"Episodes/model: {old['episodes']}, action decisions/episode: 1350, goal memory: {s['goal_memory']}",
        '',
        'Metric                                      80M            169M',
        f"Goals/episode (mean)                      {old['mean_goals_per_episode']:8.3f}       {new['mean_goals_per_episode']:8.3f}",
        f"First hit success (%)                     {old['first_hit_success_rate']*100:8.1f}       {new['first_hit_success_rate']*100:8.1f}",
        f"First hit actions (successful only)       {old['mean_first_hit_actions_success_only'] if old['mean_first_hit_actions_success_only'] is not None else float('nan'):8.1f}       {new['mean_first_hit_actions_success_only'] if new['mean_first_hit_actions_success_only'] is not None else float('nan'):8.1f}",
        f"Total returns (100 episodes)              {old['total_revisits']:8d}       {new['total_revisits']:8d}",
        '',
        f"Paired goals/ep delta 169M - 80M: {paired['paired_mean_delta_goals_per_ep']:+.3f}",
        f"Episode bootstrap 95% CI: [{paired['paired_bootstrap_95pct_CI'][0]:.3f}, {paired['paired_bootstrap_95pct_CI'][1]:.3f}]",
        f"Better/tie/worse episodes: {paired['episodes_169m_better']}/{paired['episodes_tied']}/{paired['episodes_80m_better']}",
        '',
        'Return horizons (censored-before-horizon attempts excluded):',
    ]
    for h in ('150','300','600'):
        x=s['return_horizons']['80M'][h];y=s['return_horizons']['169M'][h]
        def fmt(z):return f"{100*z['rate']:.1f}% ({z['successes']}/{z['observable_attempts']}, excluded={z['censored_early']})" if z['rate'] is not None else 'N/A'
        txt.append(f" {x['seconds']:g}s: 80M {fmt(x)}  |  169M {fmt(y)}")
    txt += ['',
            'CAUTION: first-hit mean conditions on success; evaluate success rate together.',
            'CAUTION: return-attempt rows are correlated within episodes, not independent trials.',
            'CAUTION: fewer heading errors do NOT establish vector-based navigation.',
            'CAUTION: Gridness requires a separate, properly sampled rate-map evaluation.',
            'CAUTION: 80M->169M included replay reset/restarted environments at 80M.',
            ]
    dest=out/'report.txt';dest.write_text('\n'.join(txt)+'\n',encoding='utf-8')
    print('\n'.join(txt),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--mode',choices=('check','eval'),default='check')
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--device',choices=('auto','cuda','cpu'),default='cuda')
    p.add_argument('--renderer',choices=('hardware','software'),default='software')
    p.add_argument('--goal-memory',choices=('on','off'),default='on')
    p.add_argument('--vision-rate',type=float,default=.05)
    p.add_argument('--initial-checkpoint',type=Path,default=Path('data/checkpoints/grid1350_online_B_seq_run1/grid_78k_analysis.pt'))
    p.add_argument('--old-checkpoint',type=Path,default=Path('data/checkpoints/joint_nav_32actors_200m/latest_backup_80m.pt'))
    p.add_argument('--new-checkpoint',type=Path,default=Path('data/checkpoints/joint_nav_32actors_200m/latest.pt'))
    p.add_argument('--old-training-metrics',type=Path,default=Path('data/checkpoints/joint_nav_32actors_200m/metrics_32actors.csv'))
    p.add_argument('--new-training-metrics',type=Path,default=Path('data/checkpoints/joint_nav_32actors_200m/metrics_recovered_from_000080000000.csv'))
    p.add_argument('--out-dir',type=Path,default=Path('results/joint32_80m_vs_169m_100ep'))
    p.add_argument('--env-seed-base',type=int,default=70000)
    p.add_argument('--motion-seed-base',type=int,default=80000)
    p.add_argument('--action-seed-base',type=int,default=90000)
    args=p.parse_args()
    if args.episodes<1:p.error('--episodes must be >=1')
    if not 0<=args.vision_rate<=1:p.error('--vision-rate must be in [0,1]')
    cfg,networks,device,frames80,frames169=setup(args)
    if args.mode=='check':
        print('READ-ONLY PREFLIGHT PASS | run --mode eval to collect trajectories')
        return
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError('Output folder already contains files; choose a NEW --out-dir '
                              'to avoid silently overwriting earlier evaluations')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    print(f'OUTPUT | {args.out_dir.resolve()}',flush=True)
    ep_rows=[]
    with (args.out_dir/'step_samples.csv').open('w',newline='',encoding='utf-8') as sf,\
         (args.out_dir/'episode_metrics.csv').open('w',newline='',encoding='utf-8') as ef:
        sw=csv.DictWriter(sf,fieldnames=RAW_FIELDS);sw.writeheader()
        ew=csv.DictWriter(ef,fieldnames=EP_FIELDS);ew.writeheader()
        for m in MODELS:
            ep_rows.extend(evaluate_one(m,networks[m],cfg,args,device,sw,ew))
            sf.flush();ef.flush()
    print('POSTPROCESS | reading trajectories and calibrating goal/heading axes',flush=True)
    raw=read_csv(args.out_dir/'step_samples.csv')
    sign,scores=goal_y_sign(raw)
    heading=heading_calibration(raw)
    attempts=extract_returns(raw,sign)
    with (args.out_dir/'return_attempts.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=ATTEMPT_FIELDS);w.writeheader();w.writerows(attempts)
    horizons_by={}
    for m in MODELS:
        sub=[a for a in attempts if a['model']==m]
        horizons_by[m]={str(h):horizons(sub,h) for h in (150,300,600)}
    s={'checkpoint_frames':{'80M':frames80,'169M':frames169},
       'old_checkpoint':str(args.old_checkpoint),'new_checkpoint':str(args.new_checkpoint),
       'episodes_per_model':args.episodes,'goal_memory':args.goal_memory,
       'vision_rate':args.vision_rate,
       'env_seed_base':args.env_seed_base,'motion_seed_base':args.motion_seed_base,
       'action_seed_base':args.action_seed_base,
       'goal_y_sign':sign,'goal_axis_median_hit_distance_m':scores,
       'heading_axis_calibration':heading,'heading_alignment':heading_stats(raw,sign,heading),
       'behavior':{m:episode_summary([r for r in ep_rows if r['model']==m]) for m in MODELS},
       'return_horizons':horizons_by,
       'return_attempt_count':{m:sum(a['model']==m for a in attempts) for m in MODELS},
       'paired_difference':bootstrap_delta(ep_rows)}
    print('PLOT | generating comparison charts',flush=True)
    make_figures(args.out_dir,raw,ep_rows,attempts,s,args)
    with (args.out_dir/'summary.json').open('w',encoding='utf-8') as f:
        json.dump(s,f,ensure_ascii=False,indent=2)
    summary_text(args.out_dir,s)
    print(f'FINISHED | {args.out_dir.resolve()}',flush=True)


if __name__=='__main__':
    main()
