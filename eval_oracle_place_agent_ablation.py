#!/usr/bin/env python3
"""Evaluate saved oracle-place A2C agent over full 90-second square-arena episodes.

Writes per-episode goal count, first/subsequent goal arrival frames, and raw (x,y)
trajectories for making Banino-style Fig. 2h route visualizations. Do not mistake
oracle-place control for a learned grid-code agent.
"""
import argparse
import csv
import random
from pathlib import Path
import numpy as np
import torch
from config import Config
from grid_cells.actor_critic import ActorCriticLSTM
from grid_cells.ensembles import build_rl_ensembles
from grid_cells.rl_common import image_to_input
from train_oracle_place_agent import OracleRunner, oracle_codes, goal_codes, stack


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--seed',type=int,default=10023)
    p.add_argument('--device',choices=('cuda','cpu','auto'),default='cuda')
    p.add_argument('--renderer',choices=('hardware','software'),default='hardware')
    p.add_argument('--out-dir',type=Path,default=Path('results/oracle_place_eval'))
    p.add_argument('--save-trajectories',type=int,default=100)
    p.add_argument('--goal-mode',choices=('normal','zero'),default='normal',
                   help='normal: remembered goal code; zero: mask goal code throughout evaluation')
    p.add_argument('--argmax',action='store_true',help='deterministic test (default stochastic sample)')
    args=p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    cfg=Config()
    cfg.rl.env.level='square_arena_oracle'
    cfg.rl.env.goal_visible=False
    (pc,), (hd,) = build_rl_ensembles(cfg,device)
    dim=pc.n_cells+hd.n_cells
    model=ActorCriticLSTM(grid_code_dim=dim,embed_dim=256,lstm_units=256).to(device)
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    if ckpt.get('algorithm') != 'oracle_place_A2C':
        raise ValueError('Not an oracle-place A2C checkpoint')
    model.load_state_dict(ckpt['actor_critic'])
    model.eval()
    goal_hd=ckpt.get('args',{}).get('goal_hd','arrival')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    env=OracleRunner(cfg.rl.env,args.renderer,0,1,args.seed)
    results=[]
    try:
        obs=env.reset()
        with (args.out_dir/'episodes.csv').open('w',newline='') as logfile:
            writer=csv.writer(logfile)
            writer.writerow(['episode','goals','reward','first_hit_frame', 'later_mean_frames'])
            for ep in range(args.episodes):
                h=model.init_hidden(1,device)
                prev_a=torch.zeros(1,dtype=torch.long,device=device)
                prev_r=torch.zeros(1,1,device=device)
                memory=torch.zeros(1,dim,device=device)
                xys=[]
                arrivals=[]
                reward_total=0.
                last_hit=0
                done=False
                for action_step in range(1350):
                    batched=stack([obs])
                    with torch.no_grad():
                        spatial=oracle_codes(batched,pc,hd,device)
                        image=image_to_input(torch.as_tensor(batched['image'],device=device))
                        goal_input = memory if args.goal_mode == 'normal' else torch.zeros_like(memory)
                        out=model.step(image,prev_r,prev_a,spatial,goal_input,h)
                        if args.argmax:
                            action=out.action_logits.argmax(-1)
                        else:
                            action=torch.distributions.Categorical(logits=out.action_logits).sample()
                    frame=(action_step+1)*cfg.rl.env.action_repeat
                    xys.append((frame,float(obs['pos'][0]),float(obs['pos'][1]),
                                float(obs['oracle_goal_xy'][0]),float(obs['oracle_goal_xy'][1])))
                    next_obs,r,done=env.step(int(action.item()))
                    reward_total+=r
                    xys[-1] = (*xys[-1], int(r>0)) # hit occurs AFTER this logged position
                    if r>0:
                        arrivals.append(frame-last_hit)
                        last_hit=frame
                        with torch.no_grad():
                            hitangle=torch.as_tensor(obs['hd'],device=device).reshape(-1)
                            memory=goal_codes(stack([next_obs]),hitangle,pc,hd,device,goal_hd)
                    h=out.hidden
                    prev_a=action
                    prev_r=torch.tensor([[r]],device=device)
                    obs=next_obs
                    if done:
                        break
                first=arrivals[0] if arrivals else -1
                later=float(np.mean(arrivals[1:])) if len(arrivals)>1 else float('nan')
                writer.writerow([ep,len(arrivals),round(reward_total,3),first,later])
                results.append({'goals':len(arrivals),'first':first,'later':later})
                if ep<args.save_trajectories:
                    with (args.out_dir/f'trajectory_ep{ep:03d}.csv').open('w',newline='') as fh:
                        tw=csv.writer(fh)
                        tw.writerow(['frame','x_m','y_m','goal_x_m_debug','goal_y_m_debug','goal_hit'])
                        tw.writerows(xys)
                if not done:
                    raise RuntimeError('Episode did not end by 5400 frames; check DeepMind Lab timeout')
                if (ep+1)%10==0:
                    print(f'{ep+1}/{args.episodes}: mean goals={np.mean([r["goals"] for r in results]):.2f}',flush=True)
    finally:
        env.close()
    hits=[r['first'] for r in results if r['first']>=0]
    later=[r['later'] for r in results if np.isfinite(r['later'])]
    print('Goal mode',args.goal_mode)
    print('RESULTS',args.out_dir)
    print('Mean goals/episode',float(np.mean([r['goals'] for r in results])))
    print('First goal found',len(hits),'/',len(results),'episodes')
    print('Total revisits',sum(max(r['goals']-1,0) for r in results))
    print('Mean first hit (among found, frames)',float(np.mean(hits)) if hits else 'n/a')
    print('Mean later return latency (frames)',float(np.mean(later)) if later else 'n/a')


if __name__=='__main__':
    main()
