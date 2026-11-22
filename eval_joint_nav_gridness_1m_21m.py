#!/usr/bin/env python3
"""READ-ONLY, matched-path Grid-like bottleneck analysis for 1M/21M Joint Navigation.

Run from the existing grid-cells-torch directory. Uses the 21M Actor solely as a
behavior policy, feeding the EXACT SAME RGB observations, noisy ego-motion and
5%-vision cue flags into both separately trained Grid/Vision pairs. The policy
is never optimized. No checkpoint, optimizer, replay or training log is changed.

Measure 512 signed linear bottleneck units (the actual Grid code consumed by
Actor-Critic), not place-cell logits or the Actor LSTM. Score 20x20 unsmoothed
occupancy-normalized rate maps with 10 DeepMind-style SAC annuli and BOTH
mean-correlation and stringent min-max gridness definitions. Report 0.37 as a
DESCRIPTIVE cutoff, not a shuffle test or proof of biological grid cells.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train_joint_nav_online as joint
import eval_joint_nav_1m_vs_21m as eva

NBINS = 20
NUNITS = 512
CUTOFF = 0.37
LABELS = ('1m', '21m')


def allocate():
    return {
        'sums': np.zeros((NBINS*NBINS, NUNITS), dtype=np.float64),
        'half_sums': np.zeros((2, NBINS*NBINS, NUNITS), dtype=np.float64),
    }


def bin_xy(pos):
    a = np.asarray(pos, dtype=float).reshape(2)
    if not np.isfinite(a).all() or (np.abs(a) > 1.25001).any():
        return None
    ij = np.clip(np.floor((a + 1.25)*NBINS/2.5).astype(int), 0, NBINS-1)
    return int(ij[0]*NBINS + ij[1])


def to_ratemap(sums, occ):
    out = np.full_like(sums, np.nan, dtype=np.float32)
    valid = occ > 0
    out[valid] = (sums[valid] / occ[valid, None]).astype(np.float32)
    return out.T.reshape(NUNITS, NBINS, NBINS)


def split_corr(a, b):
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 50:
        return float('nan')
    x = a[good].astype(np.float64)
    y = b[good].astype(np.float64)
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float('nan')
    return float(np.corrcoef(x, y)[0,1])


def score_all(maps):
    from grid_cells.scores import GridScorer, deepmind_mask_parameters
    class MeanScorer(GridScorer):
        def grid_score_60(self, c):
            return (c[60]+c[120])/2 - (c[30]+c[90]+c[150])/3
    args = dict(nbins=NBINS,
                coords_range=((-1.25, 1.25),(-1.25, 1.25)),
                mask_parameters=deepmind_mask_parameters(NBINS))
    mean_scorer = MeanScorer(**args)
    minmax_scorer = GridScorer(**args)
    scores_a = np.full(NUNITS, np.nan, np.float64)
    scores_b = np.full(NUNITS, np.nan, np.float64)
    sacs = np.zeros((NUNITS, 2*NBINS-1, 2*NBINS-1), np.float32)
    for u, rm in enumerate(maps):
        if np.isfinite(rm).sum() < 50 or np.nanstd(rm) < 1e-9:
            continue
        scores_a[u], _, sac = mean_scorer.get_scores(rm)
        scores_b[u], _, _ = minmax_scorer.get_scores(rm)
        sacs[u] = sac
        if (u+1) % 128 == 0:
            print(f'  SCORED {u+1}/{NUNITS}', flush=True)
    return scores_a, scores_b, sacs


def render(outdir, models, occ):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    fig, ax = plt.subplots(figsize=(6,5))
    v = ax.imshow(occ.reshape(NBINS,NBINS).T, origin='lower', cmap='viridis')
    fig.colorbar(v, ax=ax, label='observations / spatial bin')
    ax.set_title('Matched 21M-policy position occupancy, 20x20')
    ax.set_xlabel('X bin'); ax.set_ylabel('Y bin')
    fig.tight_layout(); fig.savefig(outdir/'occupancy.png',dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(13,4))
    for label, color in [('1m','steelblue'),('21m','darkorange')]:
        model = models[label]
        for ax, arr, metric in zip(axes, [model['scores_mean'],model['scores_minmax']],
                                   ['mean','minmax']):
            finite = arr[np.isfinite(arr)]
            if len(finite):
                ax.hist(finite,bins=36,alpha=.50,label=label,color=color)
            ax.axvline(CUTOFF,color='black',ls='--',lw=1)
            ax.set_title(f'{metric} gridness, 10 SAC annuli')
            ax.set_xlabel('Gridness score'); ax.set_ylabel('Units')
            ax.legend()
    fig.tight_layout();fig.savefig(outdir/'gridness_comparison.png',dpi=160);plt.close(fig)

    # PDF is visual output of this Python script; no external PDF is being analyzed.
    for label in LABELS:
        model = models[label]
        order = np.argsort(np.nan_to_num(model['scores_mean'],nan=-1e9))[::-1]
        with PdfPages(outdir/f'{label}_all_512_ratemaps.pdf') as pdf:
            for start in range(0,NUNITS,64):
                fig, axs = plt.subplots(8,8,figsize=(16,17))
                fig.suptitle(f'{label.upper()} | Grid bottleneck 512 | raw 20x20 rate maps | units ranked {start+1}-{start+64}',fontsize=13)
                for ax, uid in zip(axs.flat,order[start:start+64]):
                    ax.imshow(model['maps'][uid].T,origin='lower',cmap='viridis',interpolation='nearest')
                    ax.set_title(f'#{uid} G={model["scores_mean"][uid]:.2f}',fontsize=8)
                    ax.set_xticks([]);ax.set_yticks([])
                fig.subplots_adjust(left=.03,right=.98,bottom=.025,top=.91,wspace=.20,hspace=.45)
                pdf.savefig(fig);plt.close(fig)
        fig,axs=plt.subplots(4,8,figsize=(17,9))
        for ax,uid in zip(axs.flat,order[:32]):
            ax.imshow(model['maps'][uid].T,origin='lower',cmap='viridis',interpolation='nearest')
            ax.set_title(f'#{uid} mean={model["scores_mean"][uid]:.2f}  split={model["stability"][uid]:.2f}',fontsize=8)
            ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f'{label.upper()} | top 32 of 512 gridness rate maps')
        fig.tight_layout();fig.savefig(outdir/f'{label}_top32_ratemaps.png',dpi=155);plt.close(fig)
        fig,axs=plt.subplots(4,8,figsize=(17,9))
        for ax,uid in zip(axs.flat,order[:32]):
            ax.imshow(model['sacs'][uid].T,origin='lower',cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest')
            ax.set_title(f'#{uid} mean={model["scores_mean"][uid]:.2f}',fontsize=8)
            ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f'{label.upper()} | top 32 spatial autocorrelograms (SAC)')
        fig.tight_layout();fig.savefig(outdir/f'{label}_top32_sac.png',dpi=155);plt.close(fig)


def run(args):
    device=joint.choose_device(args.device)
    print(f'JOINT GRIDNESS READ ONLY | device={device} mode={args.mode} episodes={args.episodes}',flush=True)
    paths={'1m':args.checkpoint_1m,'21m':args.checkpoint_21m}
    for label,cp in paths.items():
        if not cp.is_file():
            raise FileNotFoundError(str(cp))
    modules={}
    for label in LABELS:
        cfg,grid,vision,actor,ck=eva.load_models(device,args.initial_checkpoint,paths[label])
        modules[label]=(grid,vision,actor)
        print(f'MODEL {label} | actions={ck["actions"]:,} grid_updates={ck["grid_updates"]:,} '
              f'actor_updates={ck["actor_updates"]:,} | code={joint.GRID_DIM}',flush=True)
    if args.mode=='check':
        print('CHECK PASS | both models finite; Grid code shape matches Actor input; no DMLab',flush=True)
        return
    if args.episodes<1: raise ValueError('episodes must be positive')
    if not 0 <= args.vision_rate <= 1: raise ValueError('vision-rate must be 0..1')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f'Output directory not empty: {args.out_dir}')
    envcfg=eva.make_env_cfg(cfg)
    acc={name:allocate() for name in LABELS}
    occ=np.zeros(NBINS*NBINS,np.int64)
    halves=np.zeros((2,NBINS*NBINS),np.int64)
    n_invalid=0; n_vision=0; n_rewards=0; visited=[]
    runner=joint.LabRunner(envcfg,args.renderer,args.env_seed_base)
    try:
        obs=runner.observe()
        with torch.inference_mode():
            for epi in range(args.episodes):
                states={name:(torch.zeros(1,1,128,device=device),torch.zeros(1,1,128,device=device))
                        for name in LABELS}
                actor=modules['21m'][2]
                ah=actor.init_hidden(device)
                prev_action=torch.zeros(1,dtype=torch.long,device=device)
                prev_reward=torch.zeros(1,1,device=device)
                goal_code=torch.zeros(1,joint.GRID_DIM,device=device)
                goal_known=False
                prev_obs=None
                force_reset=True
                rng=np.random.default_rng(args.motion_seed_base+epi)
                action_rng=torch.Generator(device=device)
                action_rng.manual_seed(args.policy_seed_base+epi)
                hits=0
                binset=set()
                done=False
                for t in range(1,args.max_steps_per_episode+1):
                    reset=(force_reset or prev_obs is None or
                           np.linalg.norm(obs['pos']-prev_obs['pos'])>.35)
                    motion=joint.noisy_motion(obs,prev_obs,reset,rng)
                    seen=bool(reset or rng.random()<args.vision_rate)
                    n_vision+=int(seen)
                    codes={}
                    for name in LABELS:
                        g,v,_=modules[name]
                        codes[name],states[name]=joint.live_code(g,v,obs,motion,seen,
                                                                 states[name],device,reset)
                    loc=bin_xy(obs['pos'])
                    if loc is not None:
                        occ[loc]+=1;halves[epi%2,loc]+=1; binset.add(loc)
                        for name in LABELS:
                            activation=codes[name][0].cpu().numpy()
                            acc[name]['sums'][loc]+=activation
                            acc[name]['half_sums'][epi%2,loc]+=activation
                    logits,value,ah=actor.step(codes['21m'],goal_code,prev_action,prev_reward,ah)
                    prob=F.softmax(logits,-1)
                    if not bool(torch.isfinite(prob).all()):
                        raise RuntimeError('Nonfinite action probabilities')
                    action=int(torch.multinomial(prob[0],1,generator=action_rng).item())
                    next_obs,reward,done=runner.step(action)
                    if reward>0:
                        hits+=1;n_rewards+=1
                        if not goal_known:
                            goal_code=codes['21m'].detach().clone();goal_known=True
                    prev_action=torch.tensor([action],device=device,dtype=torch.long)
                    prev_reward=torch.tensor([[reward]],device=device)
                    prev_obs,obs=obs,next_obs
                    force_reset=bool(done or reward>0)
                    n_invalid+=int(force_reset)
                    if done:break
                if not done:
                    raise RuntimeError(f'Episode {epi} did not finish by {args.max_steps_per_episode} actions')
                visited.append(len(binset))
                print(f'COLLECTED | {epi+1}/{args.episodes} episodes | '
                      f'steps={t} goals={hits} visited_bins20={len(binset)}',flush=True)
    finally:
        runner.close()
    if occ.sum()<=0: raise RuntimeError('No positions collected')
    covered=int((occ>0).sum())
    print(f'POSITION COVERAGE | {covered}/{NBINS*NBINS} bins; samples={occ.sum():,}; '
          f'vision_rate={n_vision/occ.sum():.2%}; goals={n_rewards}',flush=True)
    if covered<.90*NBINS*NBINS:
        print('WARNING | Low spatial coverage; gridness scores may be misleading. '
              'Increase --episodes or use broader exploration.',flush=True)
    stats={}
    for name in LABELS:
        a=acc[name]
        rm=to_ratemap(a['sums'],occ)
        hm=[to_ratemap(a['half_sums'][i],halves[i]) for i in range(2)]
        stability=np.array([split_corr(hm[0][j],hm[1][j]) for j in range(NUNITS)])
        print(f'SCORING | {name} 512 units ...',flush=True)
        mn,mm,sac=score_all(rm)
        stats[name]={'maps':rm,'scores_mean':mn,'scores_minmax':mm,
                     'stability':stability,'sacs':sac}
        for typ,arr in [('mean',mn),('minmax',mm)]:
            print(f'GRIDNESS | {name} {typ} >{CUTOFF}: {int(np.sum(arr>CUTOFF))}/{NUNITS} '
                  f'(DESCRIPTIVE; not significance)',flush=True)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for name in LABELS:
        s=stats[name]
        order=np.argsort(np.nan_to_num(s['scores_mean'],nan=-1e9))[::-1]
        np.savez_compressed(args.out_dir/f'{name}_ratemaps_scores.npz',
                            ratemaps=s['maps'],sacs=s['sacs'],scores_mean=s['scores_mean'],
                            scores_minmax=s['scores_minmax'],stability=s['stability'],
                            occupancy=occ.reshape(NBINS,NBINS))
        with (args.out_dir/f'{name}_gridness_sorted.csv').open('w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['unit','mean_gridness','minmax_gridness','split_half_r',
                             'mean_above_0.37_descriptive','minmax_above_0.37_descriptive'])
            for u in order:
                writer.writerow([int(u),float(s['scores_mean'][u]),float(s['scores_minmax'][u]),
                                 float(s['stability'][u]),int(s['scores_mean'][u]>CUTOFF),
                                 int(s['scores_minmax'][u]>CUTOFF)])
    summary={
        'checkpoint_paths':{k:str(v) for k,v in paths.items()},
        'behavior_policy':'21m trained Actor on identical trajectories for both Grid/Vision pairs',
        'episodes':args.episodes,'samples':int(occ.sum()),'occupied_bins':covered,
        'space_bins':NBINS,'vision_cue_rate':n_vision/int(occ.sum()),
        'goal_hits_while_collecting':n_rewards,
        'mean_bins_visited_per_episode':float(np.mean(visited)),
        'scoring':'unsmoothed 20x20 signed bottleneck maps, 39x39 SAC, 10 DeepMind annuli, both mean and minmax formulas',
        'warning':'Score >0.37 is descriptive only. No field shuffling / p-values; inspect SAC, stability and occupancy.',
        'models':{},
    }
    for name in LABELS:
        s=stats[name]
        order=np.argsort(np.nan_to_num(s['scores_mean'],nan=-1e9))[::-1]
        summary['models'][name]={
            'mean_gt_0.37':int(np.sum(s['scores_mean']>CUTOFF)),
            'minmax_gt_0.37':int(np.sum(s['scores_minmax']>CUTOFF)),
            'valid_mean_scores':int(np.isfinite(s['scores_mean']).sum()),
            'top10':[{'unit':int(u),'mean_gridness':float(s['scores_mean'][u]),
                      'minmax_gridness':float(s['scores_minmax'][u]),
                      'split_half_r':float(s['stability'][u])} for u in order[:10]]}
    (args.out_dir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    render(args.out_dir,stats,occ)
    print('RESULTS |',args.out_dir.resolve(),flush=True)
    print('READ-ONLY COMPLETE | No training checkpoint or replay was modified.',flush=True)


def self_test():
    x=allocate()
    loc=bin_xy(np.array([0,0]))
    assert loc==10*20+10
    occ=np.zeros(NBINS*NBINS,np.int64)
    occ[loc]=2
    x['sums'][loc]=3.0
    m=to_ratemap(x['sums'],occ)
    assert m.shape==(NUNITS,NBINS,NBINS)
    assert np.isclose(m[0,10,10],1.5)
    assert np.isnan(m[0,0,0])
    assert np.isclose(split_corr(np.arange(100,dtype=float),np.arange(100,dtype=float)),1)
    print('SELF TEST PASS | binning, occupancy normalization, split-half correlation',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--mode',choices=['self-test','check','eval'],default='check')
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--renderer',choices=['software','hardware'],default='hardware')
    ap.add_argument('--episodes',type=int,default=100)
    ap.add_argument('--max-steps-per-episode',type=int,default=2000)
    ap.add_argument('--vision-rate',type=float,default=.05)
    ap.add_argument('--env-seed-base',type=int,default=70000)
    ap.add_argument('--motion-seed-base',type=int,default=80000)
    ap.add_argument('--policy-seed-base',type=int,default=90000)
    ap.add_argument('--initial-checkpoint',type=Path,default=Path(
        'data/checkpoints/grid1350_online_B_seq_run1/grid_78k_analysis.pt'))
    ap.add_argument('--checkpoint-1m',type=Path,default=Path('data/checkpoints/joint_nav_pilot_1m_backup/latest.pt'))
    ap.add_argument('--checkpoint-21m',type=Path,default=Path('data/checkpoints/joint_nav_pilot_21m_backup/latest.pt'))
    ap.add_argument('--out-dir',type=Path,default=Path('data/evaluations/joint_gridness_1m_vs_21m_100ep'))
    args=ap.parse_args()
    if args.mode=='self-test': return self_test()
    return run(args)


if __name__=='__main__':
    main()
