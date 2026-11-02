"""Analyze all 20x20 Mean-Gridness >0.37 bottleneck units using the user's
adaptive central-peak exclusion and six nearest strict 4-neighbor SAC maxima.

Run in soluslan/grid-cells-torch root:
  python analyze_108_grid_spacing_4neighbor.py --self-test
  python analyze_108_grid_spacing_4neighbor.py

Does NOT retrain, rescore shuffle nulls, or modify model weights.
The obtained grid scale clustering is descriptive/exploratory, NOT forced to
match the published ~47, 70, 106 cm spacing peaks or ratios 1.49 and 1.51.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle

NBINS = 20
GRID_CUTOFF = .37
PAPER_MODES_CM = [47., 70., 106.]


def get_scorer(env_size_m):
    # Port's GridScorer with the DeepMind public *training* settings:
    # 20x20 ratemap, 10 annuli, inner=20*.2=4, outer linspace(8,20,10)
    from grid_cells.scores import GridScorer
    class MeanScorer(GridScorer):
        def grid_score_60(self, corrs):
            return (corrs[60] + corrs[120])/2. - (corrs[30] + corrs[90] + corrs[150])/3.
    half = env_size_m/2.
    return MeanScorer(NBINS, ((-half,half),(-half,half)),
                      [(4.,float(r)) for r in np.linspace(8.,20.,10)])


def compute_ratemaps(args):
    """Compute all 512 maps in ONE inference pass, streaming spatial averages."""
    import torch
    from config import Config
    from grid_cells import evaluate
    from grid_cells.dataset import build_dataloader
    from grid_cells.ensembles import encode_initial_conditions
    cfg = Config()
    cfg.model.nh_bottleneck = args.bottleneck_units
    dev = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f'Inference device: {dev} (NO TRAINING)',flush=True)
    model, _, epoch, place, hd = evaluate.load_models(cfg, args.checkpoint, dev)
    model.eval()
    loader = build_dataloader(args.shard_dir, shard_indices=None,
                              batch_size=cfg.train.minibatch_size,
                              shuffle=False,num_workers=0)
    n = NBINS
    sums = np.zeros((n*n,args.bottleneck_units),dtype=np.float64)
    counts = np.zeros(n*n,dtype=np.int64)
    half = cfg.task.env_size / 2.
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= args.n_trajectories: break
            bs=len(batch['init_pos'])
            if seen+bs > args.n_trajectories:
                batch={k:v[:args.n_trajectories-seen] for k,v in batch.items()}
                bs=len(batch['init_pos'])
            init = encode_initial_conditions(batch['init_pos'].to(dev),
                     batch['init_hd'].to(dev),place,hd)
            out=model(init,batch['ego_vel'].to(dev))
            act=out.bottleneck.detach().cpu().numpy().reshape(-1,args.bottleneck_units)
            xy=batch['target_pos'].numpy().reshape(-1,2)
            keep=((xy[:,0]>=-half)&(xy[:,0]<=half)&(xy[:,1]>=-half)&(xy[:,1]<=half))
            ix=np.floor((xy[keep,0]+half)/(2*half)*n).astype(int).clip(0,n-1)
            iy=np.floor((xy[keep,1]+half)/(2*half)*n).astype(int).clip(0,n-1)
            ibin=ix*n+iy
            counts+=np.bincount(ibin,minlength=n*n)
            for u in range(args.bottleneck_units):
                sums[:,u]+=np.bincount(ibin,weights=act[keep,u],minlength=n*n)
            seen+=bs
            if seen%1000==0 or seen==args.n_trajectories:
                print(f'  collected {seen}/{args.n_trajectories} trajectories',flush=True)
    if seen<args.n_trajectories:
        raise RuntimeError(f'Dataset has only {seen}, expected {args.n_trajectories} trajectories')
    maps=np.full_like(sums,np.nan)
    good=counts>0
    maps[good]=sums[good]/counts[good,None]
    return maps.T.reshape(args.bottleneck_units,n,n),int(epoch),float(cfg.task.env_size)


def adaptive_central_radius(sac,threshold=.5):
    """Follow threshold-connected central SAC peak along only 4 axial rays.
    Exact same definition as the previous Unit 212 diagnostic (8-connectivity).
    """
    s=np.asarray(sac,dtype=float)
    assert s.shape==(39,39)
    center=(19,19)
    if not np.isfinite(s).all() or not s[center]>threshold:
        raise ValueError('central SAC not finite/above threshold')
    labels,_=ndimage.label(s>threshold,structure=np.ones((3,3),dtype=bool))
    region=(labels==labels[center])
    directions={'right (+x)':(1,0),'left (-x)':(-1,0),
                'up (+y)':(0,1),'down (-y)':(0,-1)}
    reach={}
    for key,(dx,dy) in directions.items():
        extent=0
        for d in range(1,20):
            if not region[19+d*dx,19+d*dy]: break
            extent=d
        reach[key]=extent
    return center,region,reach,float(max(reach.values()))


def nearest_six_peaks(sac,center,radius):
    """Strict 4-neighbor peaks ranked solely by distance, as requested.
    Boundary peaks and low-overlap peaks are retained but flagged in CSV.
    """
    s=np.asarray(sac,dtype=float)
    # Four-neighbor rule: compare ONLY up, down, left, right.
    # Center is not a neighbor; diagonals are deliberately excluded.
    neighbors=np.array([[False,True,False],
                        [True,False,True],
                        [False,True,False]],dtype=bool)
    outer=ndimage.maximum_filter(s,footprint=neighbors,
                                   mode='constant',cval=-np.inf)
    xy=np.argwhere(s>outer)
    rel=xy-np.asarray(center)
    d=np.hypot(rel[:,0],rel[:,1])
    keep=d>radius
    xy,rel,d=xy[keep],rel[keep],d[keep]
    angle=(np.degrees(np.arctan2(rel[:,1],rel[:,0]))+360)%360
    heights=s[xy[:,0],xy[:,1]]
    # nearest distance; ties highest SAC; then coordinates
    ix=np.lexsort((xy[:,1],xy[:,0],-heights,d))
    return [{'x':int(x),'y':int(y),'dx':int(dx),'dy':int(dy),
             'distance_bins':float(dist),'angle_deg':float(a),
             'sac_value':float(h),
             'overlap_bins':int((NBINS-abs(dx))*(NBINS-abs(dy)))}
            for (x,y),(dx,dy),dist,a,h in zip(xy[ix],rel[ix],d[ix],angle[ix],heights[ix])][:6], int(len(xy))


def angle_gaps(peaks):
    if len(peaks)!=6: return []
    a=np.sort([p['angle_deg'] for p in peaks])
    return np.diff(np.r_[a,a[0]+360]).tolist()


def characterize(sac, threshold, cm_per_bin):
    center,central,reach,radius=adaptive_central_radius(sac,threshold)
    peaks,n_candidates=nearest_six_peaks(sac,center,radius)
    gaps=angle_gaps(peaks)
    if len(peaks)<6: return dict(center=center,central=central,reach=reach,radius=radius,
                                  peaks=peaks,n_candidates=n_candidates,
                                  spacing_bins=None,spacing_cm=None,gaps=gaps,
                                  min_overlap=None,overlap_lt100=None,
                                  mean_abs_gap_error_deg=None)
    med=float(np.median([p['distance_bins'] for p in peaks]))
    return dict(center=center,central=central,reach=reach,radius=radius,
                peaks=peaks,n_candidates=n_candidates,
                spacing_bins=med,spacing_cm=med*cm_per_bin,gaps=gaps,
                min_overlap=min(p['overlap_bins'] for p in peaks),
                overlap_lt100=sum(p['overlap_bins']<100 for p in peaks),
                mean_abs_gap_error_deg=float(np.mean(np.abs(np.array(gaps)-60.))))


def plot_sac(ax,sac,data,annotate=False):
    lm=max(1e-9,float(np.nanmax(np.abs(sac))))
    ax.imshow(sac.T,origin='lower',interpolation='nearest',cmap='RdBu_r',vmin=-lm,vmax=lm)
    ax.add_patch(Circle((19,19),data['radius'],fill=False,edgecolor='#ffe900',lw=1.2,ls='--'))
    for k,p in enumerate(data['peaks'],1):
        ax.plot([19,p['x']],[19,p['y']],c='#00e8e8',lw=.65)
        ax.plot(p['x'],p['y'],'o',markersize=3.5,markeredgecolor='#00ffff',markerfacecolor='none')
        if annotate:ax.annotate(str(k),(p['x'],p['y']),xytext=(2,2),textcoords='offset points',
                                fontsize=6,color='#00ffff',weight='bold')
    ax.set_xlim(-.5,38.5);ax.set_ylim(-.5,38.5);ax.set_aspect('equal')
    ax.set_xticks([]);ax.set_yticks([])


def plot_map(ax,ratemap):
    ax.imshow(ratemap.T,origin='lower',interpolation='nearest',cmap='turbo')
    ax.set_xticks([]);ax.set_yticks([])


def mosaic_pdf(path, rows, maps, sacs, derived, per_row=12):
    # Exactly 108 entries if original 108 threshold is reproduced.
    nrow=int(np.ceil(len(rows)/per_row))
    with PdfPages(path) as pdf:
        fig=plt.figure(figsize=(per_row*2.15,nrow*2.20+0.6))
        outer=fig.add_gridspec(nrow,per_row,left=.012,right=.988,bottom=.012,
                               top=.965,wspace=.20,hspace=.42)
        fig.suptitle('4-neighbor peaks | Mean Gridness > 0.37 | 20x20 rate map + 39x39 SAC',fontsize=17)
        for j,unit in enumerate(rows):
            gs=outer[j//per_row,j%per_row].subgridspec(1,2,wspace=.02)
            a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1])
            plot_map(a,maps[unit]);plot_sac(b,sacs[unit],derived[unit])
            record=derived[unit]
            s='NA' if record['spacing_cm'] is None else f'{record["spacing_cm"]:.0f}cm'
            a.set_title(f'#{unit} G={record["gridness"]:.2f} / {s}',fontsize=7.2,pad=4)
        pdf.savefig(fig)
        plt.close(fig)


def detailed_pdf(path, rows, maps, sacs, derived):
    with PdfPages(path) as pdf:
        for start in range(0,len(rows),9):
            fig=plt.figure(figsize=(15,14))
            outer=fig.add_gridspec(3,3,left=.035,right=.97,bottom=.035,top=.94,wspace=.20,hspace=.34)
            fig.suptitle(f'4-neighbor peaks: nearest 6 outside central region | {start+1}-{min(start+9,len(rows))} of {len(rows)}',fontsize=14)
            for j,u in enumerate(rows[start:start+9]):
                gs=outer[j//3,j%3].subgridspec(1,2,wspace=.02)
                a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1])
                plot_map(a,maps[u]);plot_sac(b,sacs[u],derived[u],annotate=True)
                rec=derived[u]
                sc='N/A' if rec['spacing_cm'] is None else f'{rec["spacing_cm"]:.1f}cm'
                flag=' !low overlap' if rec['overlap_lt100'] else ''
                a.set_title(f'Unit {u}, G={rec["gridness"]:.3f}',fontsize=10)
                b.set_title(f'Spacing={sc}, exc={rec["radius"]:.0f}bin{flag}',fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
        print('  detailed SAC PDF pages:',int(np.ceil(len(rows)/9)),flush=True)


def gmm_find(scales, max_k=8):
    """GMM/BIC used only to report if 3 peaks arise, not to force k=3."""
    from sklearn.mixture import GaussianMixture
    a=np.asarray(scales,dtype=float)
    a=a[np.isfinite(a)].reshape(-1,1)
    if len(a)<max(10,max_k): return None
    candidates=[]
    for k in range(1,min(max_k,len(a)-1)+1):
        fit=GaussianMixture(n_components=k,random_state=0,n_init=5,reg_covar=1e-3).fit(a)
        candidates.append((k,float(fit.bic(a)),float(fit.aic(a)),fit))
    best=min(candidates,key=lambda z:z[1])
    means=np.sort(best[3].means_.ravel())
    ratios=(means[1:]/means[:-1]).tolist() if len(means)>1 else []
    # A three-component fit is reported separately for comparison with the
    # paper, even if model selection does NOT prefer three components.
    three=next((item for item in candidates if item[0]==3),None)
    if three is not None:
        modes3=np.sort(three[3].means_.ravel())
        three_result=dict(means_cm=modes3.tolist(),
                          adjacent_ratios=(modes3[1:]/modes3[:-1]).tolist(),
                          bic=three[1],delta_bic_vs_best=three[1]-best[1])
    else:
        three_result=None
    return dict(best_k=best[0],means_cm=means.tolist(),ratios=ratios,
                three_component_fit_for_comparison_only=three_result,
                bic=[{'k':k,'bic':bic,'aic':aic} for k,bic,aic,_ in candidates])


def scale_distribution(path, scales, fit):
    fig,ax=plt.subplots(figsize=(10,5.5))
    scales=np.asarray(scales,float)
    scales=scales[np.isfinite(scales)]
    if len(scales):
        ax.hist(scales,bins=max(10,min(25,int(np.sqrt(len(scales))*2))),color='#739fc1',
                edgecolor='#315574',alpha=.85,label=f'{len(scales)} measurable units')
    for i,v in enumerate(PAPER_MODES_CM):
        ax.axvline(v,c='#bb6a2c',ls='--',lw=1.2,
                   label='Paper modes (47, 70, 106 cm)' if i==0 else None)
    if fit is not None:
        for i,v in enumerate(fit['means_cm']):
            ax.axvline(v,c='#1a816c',ls=':',lw=1.5,
                       label=f'BIC-best GMM (k={fit["best_k"]})' if i==0 else None)
    ax.set_title('4-neighbor grid spacing | Mean Gridness > 0.37 | nearest 6 maxima')
    ax.set_xlabel('Grid spacing (cm)');ax.set_ylabel('Number of units')
    ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)


def save_csvs(folder, order, derived, epoch, cm_per_bin):
    summary=folder/'grid_spacing_108_units.csv'
    with summary.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        w.writerow(['unit','gridness_mean','center_exclusion_radius_bins','reach_right_bins',
                    'reach_left_bins','reach_up_bins','reach_down_bins','n_local_maxima',
                    'selected_peak_count','grid_spacing_bins','grid_spacing_cm',
                    'gap1_deg','gap2_deg','gap3_deg','gap4_deg','gap5_deg','gap6_deg',
                    'angular_gap_mean_abs_error_60_deg','minimum_peak_overlap_pixels',
                    'n_peaks_overlap_lt100','warning'])
        for u in order:
            x=derived[u]
            note=[]
            if x['spacing_cm'] is None:note.append('fewer than six peaks')
            if x['overlap_lt100']:note.append('low-overlap edge peaks')
            if x['spacing_cm'] is not None and x['spacing_cm']>220:note.append('spacing > arena size')
            w.writerow([u,x['gridness'],x['radius'],x['reach']['right (+x)'],
                        x['reach']['left (-x)'],x['reach']['up (+y)'],x['reach']['down (-y)'],
                        x['n_candidates'],len(x['peaks']),x['spacing_bins'],x['spacing_cm'],
                        *(x['gaps']+['']*(6-len(x['gaps']))),x['mean_abs_gap_error_deg'],
                        x['min_overlap'],x['overlap_lt100'],'; '.join(note)])
    with (folder/'six_peaks_per_unit.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        w.writerow(['unit','peak_rank_by_distance','sac_x','sac_y','dx_bins','dy_bins',
                    'distance_bins','distance_cm','angle_deg','sac_value','overlap_pixels'])
        for u in order:
            for rank,p in enumerate(derived[u]['peaks'],1):
                w.writerow([u,rank,p['x'],p['y'],p['dx'],p['dy'],p['distance_bins'],
                            p['distance_bins']*cm_per_bin,p['angle_deg'],p['sac_value'],p['overlap_bins']])
    return summary


def compare_with_eight_neighbor(previous_csv, output_folder, order, derived):
    """Optional 4-neighbor vs already-computed 8-neighbor comparison.

    Uses the old CSV only as a reference. It does not change the 4-neighbor
    peak selection, model weights, ratemaps, SACs, or Gridness values.
    """
    old_file=Path(previous_csv)
    if not old_file.exists():
        print(f'No saved 8-neighbor CSV at {old_file}; skipping comparison.')
        return None
    with old_file.open(newline='',encoding='utf-8-sig') as f:
        old={int(row['unit']):row for row in csv.DictReader(f)}
    combined=[]; old_vals=[]; new_vals=[]
    for u in order:
        if u not in old:continue
        old_txt=old[u].get('grid_spacing_cm','')
        new=derived[u]['spacing_cm']
        try:prior=float(old_txt)
        except (TypeError,ValueError):prior=None
        diff=None if (prior is None or new is None) else new-prior
        combined.append({'unit':u,'gridness':derived[u]['gridness'],
                         'grid_spacing_8neighbor_cm':prior,
                         'grid_spacing_4neighbor_cm':new,
                         'delta_4_minus_8_cm':diff,
                         'local_maxima_8neighbor':old[u].get('n_local_maxima',''),
                         'local_maxima_4neighbor':derived[u]['n_candidates'],
                         'min_selected_overlap_4neighbor':derived[u]['min_overlap']})
        if diff is not None:
            old_vals.append(prior);new_vals.append(new)
    if not combined:
        print('8-neighbor CSV found, but no matching units; comparison skipped.')
        return None
    out=output_folder/'comparison_4_vs_8_neighbor.csv'
    with out.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)
    if old_vals:
        old_vals=np.array(old_vals);new_vals=np.array(new_vals)
        delta=new_vals-old_vals
        changed=int((np.abs(delta)>1e-6).sum())
        f,axs=plt.subplots(1,2,figsize=(12,4.9))
        lo=float(min(old_vals.min(),new_vals.min()));hi=float(max(old_vals.max(),new_vals.max()))
        axs[0].plot([lo,hi],[lo,hi],'--',color='gray',linewidth=1,label='unchanged')
        axs[0].scatter(old_vals,new_vals,s=18,alpha=.72)
        axs[0].set(xlabel='8-neighbor spacing (cm)',ylabel='4-neighbor spacing (cm)',
                   title=f'Per-unit comparison ({len(delta)} matched)')
        axs[0].set_xlim(lo-4,hi+4);axs[0].set_ylim(lo-4,hi+4)
        axs[0].set_aspect('equal',adjustable='box')
        bins=np.linspace(lo,hi,24)
        axs[1].hist(old_vals,bins=bins,alpha=.48,label='8-neighbor')
        axs[1].hist(new_vals,bins=bins,alpha=.48,label='4-neighbor')
        axs[1].set(xlabel='Grid spacing (cm)',ylabel='Units',title='Spacing distribution')
        axs[1].legend()
        f.suptitle('Same network + same ratemaps + same SACs | only local-max neighborhood changed')
        f.tight_layout()
        f.savefig(output_folder/'comparison_4_vs_8_neighbor.png',dpi=170)
        plt.close(f)
        print(f'4-vs-8 comparison: {changed}/{len(delta)} measured units changed spacing; '
              f'median(4-8)={float(np.median(delta)):.2f} cm',flush=True)
        return {'matched_units':len(combined),'both_measurable':len(delta),
                'changed_spacing_count':changed,
                'median_delta_cm':float(np.median(delta)),
                'mean_delta_cm':float(np.mean(delta))}
    print('8-neighbor comparison CSV created, but no pairs with both spacings.')
    return {'matched_units':len(combined),'both_measurable':0}


def test_self():
    # Test a 39x39 synthetic hex pattern with 6 local peaks at ~8 bins.
    ix,iy=np.indices((39,39));s=np.exp(-((ix-19)**2+(iy-19)**2)/(2*2.5**2))
    expected={(27,19),(23,26),(15,26),(11,19),(15,12),(23,12)}
    for x,y in expected:s+=.85*np.exp(-((ix-x)**2+(iy-y)**2)/(2*.7**2))
    c,reg,reach,r=adaptive_central_radius(s,.5)
    peaks,n=nearest_six_peaks(s,c,r)
    assert r==2. and len(peaks)==6, (r,peaks)
    assert {(p['x'],p['y']) for p in peaks}==expected
    assert abs(sum(angle_gaps(peaks))-360)<1e-9
    assert np.isclose(np.median([p['distance_bins'] for p in peaks]),np.sqrt(65))
    assert all(p['overlap_bins']==(20-abs(p['dx']))*(20-abs(p['dy'])) for p in peaks)
    # A diagonal higher neighbor must NOT suppress a 4-neighbor maximum.
    # (6,6) is lower than diagonal (7,7) but higher than all four cardinal neighbors.
    diagonal_case=np.zeros((39,39),dtype=float)
    diagonal_case[6,6]=.8
    diagonal_case[7,7]=1.0
    four, n4=nearest_six_peaks(diagonal_case,(19,19),4.)
    assert n4==2 and {(x['x'],x['y']) for x in four}=={(6,6),(7,7)},four
    # One complete overview rendering from six synthetic entries
    fake={0:dict(**characterize(s,.5,11.),gridness=.91)}
    fake_map=np.zeros((1,20,20));fake_sac={0:s}
    out=Path('results/_grid108_4neighbor_synthetic_test.pdf');out.parent.mkdir(parents=True,exist_ok=True)
    mosaic_pdf(out,[0],fake_map,fake_sac,fake,per_row=1)
    print('PASS: adaptive center, 4-neighbor peaks, median, 6 angles, overlaps, PDF')
    print('Synthetic PDF:',out)


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--checkpoint',default='data/checkpoints/sonnet_v127_lstm_512/checkpoint_epoch299.pt')
    p.add_argument('--shard-dir',default='data/datasets/gpu1m/shards')
    p.add_argument('--n-trajectories',type=int,default=4000)
    p.add_argument('--bottleneck-units',type=int,default=512)
    p.add_argument('--center-threshold',type=float,default=.5)
    p.add_argument('--gridness-threshold',type=float,default=.37)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--out-dir',default='results/sonnet_deepmind20_mean/grid_spacing_108_4neighbor')
    p.add_argument('--previous-csv',default='results/sonnet_deepmind20_mean/unit_statistics.csv')
    p.add_argument('--previous-eight-csv',default='results/sonnet_deepmind20_mean/grid_spacing_108/grid_spacing_108_units.csv',help='Existing 8-neighbor results; optional comparison')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--no-open',action='store_true')
    args=p.parse_args()
    if args.self_test:return test_self()
    if args.n_trajectories<1 or args.bottleneck_units<1:
        p.error('trajectories and bottleneck units must be positive')
    if not 0 < args.center_threshold < 1:
        p.error('center-threshold must be within (0, 1)')
    folder=Path(args.out_dir);folder.mkdir(parents=True,exist_ok=True)
    maps,epoch,env_m=compute_ratemaps(args)
    cm_per_bin=env_m*100/NBINS
    scorer=get_scorer(env_m)
    gridness=np.empty(args.bottleneck_units,float)
    sacs={}; masks={}
    print(f'Scoring {args.bottleneck_units} units (Mean, 20x20, 10 annuli)',flush=True)
    for u,rm in enumerate(maps):
        g,mask,sac=scorer.get_scores(rm)
        gridness[u]=g;masks[u]=tuple(map(float,mask));sacs[u]=sac
    units=np.flatnonzero(gridness>args.gridness_threshold).tolist()
    units.sort(key=lambda u:-gridness[u])
    print(f'Units passing Mean Gridness >{args.gridness_threshold}: {len(units)}/{len(gridness)}',flush=True)
    if not units:raise RuntimeError('No units pass threshold')
    if len(units)!=108:
        print('WARNING: expected 108 from previous run; mismatch indicates different checkpoint/data/scoring.',flush=True)
    # Reference scores cross-check; does not control unit selection.
    if Path(args.previous_csv).is_file():
        with open(args.previous_csv,newline='',encoding='utf-8-sig') as f:
            table={int(row['unit']):float(row['observed_gridness']) for row in csv.DictReader(f)}
        deviations=[abs(g-table[u]) for u,g in enumerate(gridness) if u in table]
        if deviations:
            print('Previous observed score CSV max difference:',max(deviations),flush=True)
            if max(deviations)>1e-3:print('WARNING: differences >.001; check evaluation inputs!',flush=True)
    derived={}
    for u in units:
        try:
            result=characterize(sacs[u],args.center_threshold,cm_per_bin)
        except (ValueError,AssertionError) as e:
            result=dict(center=(19,19),central=None,reach={k:0 for k in
                       ('right (+x)','left (-x)','up (+y)','down (-y)')},radius=0.,
                       peaks=[],n_candidates=0,spacing_bins=None,spacing_cm=None,gaps=[],
                       min_overlap=None,overlap_lt100=None,mean_abs_gap_error_deg=None,
                       error=str(e))
        result['gridness']=float(gridness[u]);result['annulus']=masks[u]
        derived[u]=result
    summary_csv=save_csvs(folder,units,derived,epoch,cm_per_bin)
    valid=[derived[u]['spacing_cm'] for u in units if derived[u]['spacing_cm'] is not None]
    suspicious=sum(bool(derived[u]['overlap_lt100']) for u in units)
    fit=gmm_find(valid)
    if fit is not None:
        with (folder/'gmm_bic.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=['k','bic','aic']);w.writeheader();w.writerows(fit['bic'])
    scale_distribution(folder/'spacing_histogram_gmm.png',valid,fit)
    print('Rendering a single-page overview PDF...',flush=True)
    mosaic_pdf(folder/'all_selected_108_one_page.pdf',units,maps,sacs,derived)
    print('Rendering annotated 9-units-per-page PDF...',flush=True)
    detailed_pdf(folder/'all_selected_peak_details.pdf',units,maps,sacs,derived)
    comparison=compare_with_eight_neighbor(args.previous_eight_csv,folder,units,derived)
    report={'method':'20x20 Mean 10-annuli, G>0.37 selection, SAC center threshold>0.5, cardinal rays, strict 4-neighbor 6 nearest maxima',
            'checkpoint':str(Path(args.checkpoint).resolve()),'epoch':epoch,
            'n_trajectories':args.n_trajectories,'n_selected_units':len(units),
            'n_scales_measurable':len(valid),'n_units_any_peak_overlap_lt100':suspicious,
            'min_scales_cm':float(np.min(valid)) if valid else None,
            'median_scale_cm':float(np.median(valid)) if valid else None,
            'max_scales_cm':float(np.max(valid)) if valid else None,
            'gmm_bic_exploratory':fit,
            'four_vs_eight_comparison':comparison,
            'paper_comparison_peaks_cm':PAPER_MODES_CM,
            'paper_adjacent_ratios':[70/47,106/70],
            'warning':'Grid spacing with 6 nearest strict local maxima can select SAC-edge artifacts; no overlap filter was applied. GMM clusters need not match paper and are descriptive only.',
            'cm_per_bin':cm_per_bin}
    (folder/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n========== 108 UNIT GRID SPACING ==========')
    print(f'Gridness threshold: >{args.gridness_threshold} => {len(units)} units')
    print(f'Central threshold: >{args.center_threshold}, auto axial exclusion')
    print(f'SAC bin size: {cm_per_bin:g} cm')
    print(f'Six peaks measured: {len(valid)}/{len(units)} units')
    print(f'Units with >=1 peak overlap <100 pixels: {suspicious}/{len(units)}')
    if valid:print(f'Median grid spacing across units: {np.median(valid):.2f} cm (min {np.min(valid):.2f}, max {np.max(valid):.2f})')
    if fit is not None:
        print('BIC-best GMM number of clusters:',fit['best_k'])
        print('BIC-best component centers (cm):',[round(x,2) for x in fit['means_cm']])
        print('BIC-best adjacent ratios:',[round(x,3) for x in fit['ratios']])
        if fit['three_component_fit_for_comparison_only']:
            a=fit['three_component_fit_for_comparison_only']
            print('3-cluster comparison only: centers (cm)',[round(x,2) for x in a['means_cm']])
            print('3-cluster comparison only: ratios',[round(x,3) for x in a['adjacent_ratios']])
            print('3-cluster BIC penalty vs best:',round(a['delta_bic_vs_best'],2))
    print('Paper reference modes ~47,70,106cm; adjacent ratios ~1.49,1.51 (comparison, NOT forced).')
    for name in ['grid_spacing_108_units.csv','six_peaks_per_unit.csv','all_selected_108_one_page.pdf',
                 'all_selected_peak_details.pdf','spacing_histogram_gmm.png','comparison_4_vs_8_neighbor.csv','comparison_4_vs_8_neighbor.png','summary.json']:
        if (folder/name).exists():
            print(' ',(folder/name).resolve())
    if os.name=='nt' and not args.no_open:
        try:os.startfile(str((folder/'all_selected_108_one_page.pdf').resolve()))
        except OSError: pass


if __name__=='__main__':main()
