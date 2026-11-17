#!/usr/bin/env python3
"""Read-only, variable-binning 1350-step held-out gridness for causal/offline Grid+Vision checkpoints.

Requires prior analysis helper analyze_grid_dmlab_vision_pc001_272_gridness.py
and the 1350 trainer. Computes configurable 20x20 or 32x32, 10 annuli mean/minmax scores
for 512 bottleneck units. Historical threshold 0.37 is descriptive only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import analyze_grid_dmlab_vision_pc001_272_gridness as scoring
import train_grid_dmlab_vision_pc001_272_noisy as base
from config import Config
from grid_cells.ensembles import build_rl_ensembles
from grid_cells.model import GridCellsRNN
from grid_cells.vision import VisionModule

STEPS=1350
base.STEPS=STEPS


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data-dirs',type=Path,nargs='+',required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--max-episodes',type=int,default=0,
                   help='0 means all held-out episodes; subset is PREVIEW only')
    p.add_argument('--val-batch-size',type=int,default=8)
    p.add_argument('--spatial-bins',type=int,choices=(20,32),default=32,
                   help='Spatial rate-map bins (default: 32, DeepMind public code: 20)')
    p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        scoring.self_test();return
    dirs=[d.expanduser().resolve() for d in a.data_dirs]
    path=a.checkpoint.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if a.val_batch_size<1 or a.max_episodes<0:
        p.error('Bad evaluation batch or episode count')
    ck=torch.load(path,map_location='cpu',weights_only=False)
    fp=ck.get('fingerprint',{})
    if (fp.get('script')!='causal_grid_vision_1350_v1' or fp.get('steps')!=STEPS or
            fp.get('data_dirs') != [str(d) for d in dirs]):
        raise ValueError('Not a matching 1350 A/B training checkpoint or data dirs')
    device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else
                        'cpu' if a.device=='auto' else a.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    cfg=Config();cfg.rl.pc_scale=(.01,)
    pc_l,hd_l=build_rl_ensembles(cfg,device);pc,hd=pc_l[0],hd_l[0]
    grid=GridCellsRNN(target_ensembles=[pc,hd],nh_lstm=128,
                      nh_bottleneck=512,dropout_rates=(.5,),
                      bottleneck_has_bias=True,ego_vel_dim=4,vision_dim=268).to(device)
    vision=VisionModule(pc,hd,embed_dim=256,mask_prob=.95,mask_mode='whole').to(device)
    grid.load_state_dict(ck['grid'],strict=True)
    vision.load_state_dict(ck['vision'],strict=True)
    if not torch.allclose(ck['pc_centers'].cpu(),pc.means.cpu(),rtol=0,atol=1e-7):
        raise ValueError('PC centers mismatch')
    if not torch.allclose(ck['hd_centers'].cpu(),hd.means.cpu(),rtol=0,atol=1e-7):
        raise ValueError('HD centers mismatch')
    grid.eval();vision.eval();grid.requires_grad_(False);vision.requires_grad_(False)
    val=base.EpisodeStore(dirs,'val')
    ids=np.arange(len(val) if a.max_episodes==0 else min(a.max_episodes,len(val)))
    if not len(ids):
        raise ValueError('No validation episodes')
    nbins=a.spatial_bins
    scoring.NBINS=nbins  # score_ratemaps() constructs scorers from this module constant
    sums=np.zeros((nbins*nbins,512),np.float64);counts=np.zeros(nbins*nbins,np.int64)
    split_sums=np.zeros((2,nbins*nbins,512),np.float64)
    split_counts=np.zeros((2,nbins*nbins),np.int64)
    error_sums=np.zeros(3,np.float64)
    n_steps=0
    cfg_eval=dict(vision_rate=float(fp['vision_rate']),
                  force_first=bool(fp['force_first_vision']),
                  speed_noise=float(fp['speed_noise_std']),
                  angular_noise=float(fp['angular_noise_std']),
                  seed=int(fp['seed']))
    captures=[]
    hook=grid.bottleneck.register_forward_hook(
        lambda _m,_x,out:captures.append(out.detach()))
    try:
        with torch.inference_mode():
            for left in range(0,len(ids),a.val_batch_size):
                chosen=ids[left:left+a.val_batch_size]
                # Same deterministic held-out forward process in both experiment arms.
                rollout=base.build_rollout(val,chosen,vision,device,
                    np.random.default_rng(0),np.random.default_rng(0),
                    cfg_eval['vision_rate'],cfg_eval['force_first'],
                    cfg_eval['speed_noise'],cfg_eval['angular_noise'],
                    deterministic_val=True,val_seed=cfg_eval['seed']+83817)
                b=len(chosen)
                hidden=(torch.zeros(1,b,128,device=device),torch.zeros(1,b,128,device=device))
                for start in range(0,STEPS,100):
                    end=min(STEPS,start+100)
                    lp,lh,hidden=base.step_segment(
                        grid,rollout.vel[:,start:end],rollout.vis[:,start:end],
                        rollout.reset[:,start:end],hidden,training=False)
                    if len(captures)!=1:
                        raise RuntimeError(f'Bottleneck hook returned {len(captures)} tensors')
                    bott=captures.pop().cpu().numpy()
                    positions=rollout.pos[:,start:end]
                    xy=positions.cpu().numpy()
                    scoring.accumulate(sums,counts,xy,bott,n=nbins)
                    for row,eid in enumerate(chosen):
                        parity=int(eid)%2
                        scoring.accumulate(split_sums[parity],split_counts[parity],xy[row],bott[row],n=nbins)
                    pt,ht=base.targets(pc,hd,positions,rollout.hd[:,start:end])
                    ce=(-(pt*F.log_softmax(lp,-1)).sum(-1)
                        -(ht*F.log_softmax(lh,-1)).sum(-1))
                    err=base.prediction_errors(pc,hd,lp,lh,positions,rollout.hd[:,start:end])
                    error_sums+=(float(ce.sum()),float(err[1].sum()),float(err[3].sum()))
                    n_steps+=b*(end-start)
                print(f'EVALUATED | {left+len(chosen)}/{len(ids)} held-out episodes',flush=True)
    finally:
        hook.remove()
    if n_steps!=len(ids)*STEPS:
        raise RuntimeError('Incomplete validation frame count')
    ratemaps=scoring.make_ratemaps(sums,counts,n=nbins)
    halves=[scoring.make_ratemaps(split_sums[i],split_counts[i],n=nbins) for i in (0,1)]
    stability=np.asarray([scoring.split_half_pearson(halves[0][i],halves[1][i]) for i in range(512)])
    mean_scores,minmax_scores,ring_mean,ring_minmax,sacs=scoring.score_ratemaps(ratemaps)
    rank=np.argsort(np.nan_to_num(mean_scores,nan=-1e9))[::-1]
    out=a.out_dir.expanduser().resolve()
    if (out/'ratemaps_and_scores.npz').exists():
        raise FileExistsError(f'Output NPZ already exists, use a fresh --out-dir: {out}')
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'ratemaps_and_scores.npz',
                        ratemaps=ratemaps,sacs=sacs,occupancy=counts.reshape(nbins,nbins),
                        scores_mean=mean_scores,scores_minmax=minmax_scores,
                        split_half_stability=stability,
                        annuli_mean=ring_mean,annuli_minmax=ring_minmax)
    with (out/'gridness_scores_sorted.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        w.writerow(['unit','mean_gridness','minmax_gridness','split_half_r',
                    'mean_above_historical_0.37','minmax_above_historical_0.37'])
        for k in rank:
            w.writerow([int(k),float(mean_scores[k]),float(minmax_scores[k]),
                        float(stability[k]),int(mean_scores[k]>.37),
                        int(minmax_scores[k]>.37)])
    report=dict(mode=fp['mode'],rollout=int(ck['rollout']),
                grid_updates=int(ck['grid_updates']),vision_updates=int(ck['vision_updates']),
                validation_episodes=len(ids),validation_steps=n_steps,
                ratemap_bins=nbins,sac_bins=nbins*2-1,bin_size_cm=250.0/nbins,
                occupied_bins=int((counts>0).sum()),
                eval_ce=float(error_sums[0]/n_steps),
                eval_weighted_pos_cm=float(error_sums[1]/n_steps),
                eval_hd_deg=float(error_sums[2]/n_steps),
                mean_above_0_37=int(np.sum(mean_scores>.37)),
                minmax_above_0_37=int(np.sum(minmax_scores>.37)),
                top10=[dict(unit=int(k),score=float(mean_scores[k])) for k in rank[:10]],
                note='Historical 0.37 is DESCRIPTIVE, not shuffled p-value')
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,8,figsize=(16,9))
    for ax,k in zip(axes.flat,rank[:32]):
        ax.imshow(ratemaps[k],origin='lower',cmap='viridis')
        ax.set_title(f'#{k}  G={mean_scores[k]:.2f}',fontsize=9)
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(f'{fp["mode"].upper()} | {nbins}x{nbins} bins | 1350 steps x {len(ids)} held-out episodes | bottleneck512')
    fig.tight_layout()
    fig.savefig(out/'top32_ratemaps.png',dpi=150)
    plt.close(fig)
    print(f'MEAN formula, historical score >0.37: {report["mean_above_0_37"]}/512 (DESCRIPTIVE ONLY)',flush=True)
    print(f'MINMAX formula, historical score >0.37: {report["minmax_above_0_37"]}/512 (DESCRIPTIVE ONLY)',flush=True)
    print(f'SAVED TO | {out}',flush=True)
    print('READ-ONLY COMPLETE | no optimizer or checkpoint was modified',flush=True)

if __name__=='__main__':
    main()
