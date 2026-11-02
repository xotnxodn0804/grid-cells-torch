"""No-smoothing field-shuffle gridness analysis for soluslan/grid-cells-torch.

This script is a TRANSPARENT APPROXIMATION of the spatial-field shuffle
reported by Banino et al. (2018), following Barry & Burgess (2017).

- No Gaussian smoothing anywhere, including watershed segmentation.
- Same PyTorch gridness scorer/annulus as the currently patched evaluate.py.
- Raw activity values are preserved exactly while relocating fields.
- Each source unit receives its own shuffled null distribution.

From grid-cells-torch project root:
    python field_shuffle_nosmoothing.py --self-test
    python field_shuffle_nosmoothing.py --checkpoint data/checkpoints/gpu1m_512/checkpoint_epoch299.pt --shard-dir data/datasets/gpu_eval4k/shards --unit-ids 235,415,74,367 --n-shuffles 10 --out-dir results/shuffle_pilot
    python field_shuffle_nosmoothing.py --checkpoint data/checkpoints/gpu1m_512/checkpoint_epoch299.pt --shard-dir data/datasets/gpu_eval4k/shards --n-shuffles 100 --workers 4 --out-dir results/shuffle_full

Scientific caveats:
- Banino does not disclose every watershed/local-max/placement detail.
- Using ALL raw local maxima is a prespecified, no-smoothing choice which
  can cause over-segmentation. Report marker/field counts and inspect plots.
- Do not tune field segmentation/annuli to match the paper's 129/512.
- The 0.37 criterion is only a historical comparator, not this null cutoff.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


@lru_cache(maxsize=4)
def nearest_available_order(n: int) -> np.ndarray:
    """Nearest destination cells by city-block distance, then Euclidean/tie index."""
    xy = np.indices((n, n), dtype=np.int16).reshape(2, -1).T
    result = np.empty((n*n, n*n), dtype=np.int32)
    for i, pt in enumerate(xy):
        delta = np.abs(xy - pt)
        manhattan = delta.sum(axis=1)
        squared = (delta * delta).sum(axis=1)
        result[i] = np.lexsort((np.arange(n*n), squared, manhattan))
    return result


def segment_raw_fields(rm: np.ndarray):
    """Watershed from *raw*, unsmoothed 8-neighbour regional maxima.

    Each flat local-maximum plateau has a single representative marker.
    Labels mark every visited cell, with 0 for missing/never-visited cells.
    No Gaussian kernel, no arbitrary intensity threshold, and no fixed cap
    on marker number. A nearly constant map is not testable.
    """
    rm = np.asarray(rm, dtype=np.float64)
    if rm.ndim != 2 or rm.shape[0] != rm.shape[1]:
        raise ValueError("Need square rate map")
    valid = np.isfinite(rm)
    if valid.sum() < 8:
        raise ValueError("Too few observed spatial bins")
    lo, hi = float(rm[valid].min()), float(rm[valid].max())
    if hi - lo < 1e-10:
        raise ValueError("Near-constant spatial activity; cannot identify fields")

    # No smoothing: local maxima in the exact raw spatial activity map.
    work = np.where(valid, rm, -np.inf)
    local_max = (work >= ndi.maximum_filter(work, size=3, mode="constant", cval=-np.inf)) & valid
    plateau_labels, n_plateaus = ndi.label(local_max, structure=np.ones((3, 3), dtype=np.uint8))
    peaks = []
    for p in range(1, n_plateaus + 1):
        cells = np.argwhere(plateau_labels == p)
        best = cells[np.argmax(rm[tuple(cells.T)])]
        peaks.append(best)
    if not peaks:
        raise ValueError("No local maxima identified")
    peaks = np.array(peaks, dtype=np.int32)

    markers = np.zeros(rm.shape, dtype=np.int32)
    for i, (x, y) in enumerate(peaks, 1):
        markers[x, y] = i
    # Watershed inverse intensity: peaks are minima. Quantization is
    # required by scipy.ndimage.watershed_ift (uint8/uint16 input).
    gradient = np.full(rm.shape, 65535, dtype=np.uint16)
    gradient[valid] = np.rint((hi - rm[valid]) / (hi - lo) * 65535).astype(np.uint16)
    labels = ndi.watershed_ift(gradient, markers)
    labels[~valid] = 0
    if np.any(labels[valid] == 0):
        raise RuntimeError("Watershed left valid bins unlabelled")
    if np.any(labels[tuple(peaks.T)] != np.arange(1, len(peaks)+1)):
        raise RuntimeError("Watershed moved a field peak's label")
    return labels, peaks


def shuffle_fields(rm: np.ndarray, segmented, rng: np.random.Generator) -> np.ndarray:
    """Relocate unsmoothed fields; round-robin placement with collision repair.

    Peaks are placed first at distinct random visited bins. Remaining bins
    are relocated relative to their peaks. An occupied target is resolved by
    the nearest unused VISITED bin (Manhattan distance). The missing-bin
    mask and the multiset of original activity values remain unchanged.
    """
    rm = np.asarray(rm, dtype=np.float64)
    n = rm.shape[0]
    labels, peaks = segmented
    valid = np.isfinite(rm)
    valid_flat = np.flatnonzero(valid.ravel())
    n_fields = len(peaks)
    if n_fields > len(valid_flat):
        raise ValueError("More markers than observed bins")
    anchors_idx = rng.choice(valid_flat, size=n_fields, replace=False)
    anchors_xy = np.array(np.unravel_index(anchors_idx, rm.shape)).T

    coordinates = np.indices(rm.shape).reshape(2, -1).T
    labelled = labels.ravel()
    value_flat = rm.ravel()
    per_field = []
    for f in range(n_fields):
        src = np.flatnonzero(labelled == f+1)
        if not len(src):
            raise RuntimeError("Empty watershed field")
        offset = coordinates[src] - peaks[f]
        dist2 = (offset.astype(np.int32)**2).sum(axis=1)
        # First must be peak (offset=[0,0]) -- initially reserve all peaks.
        order = src[np.lexsort((src, dist2))]
        if order[0] != np.ravel_multi_index(peaks[f], rm.shape):
            raise RuntimeError("Field peak was not the first bin")
        per_field.append(order)

    out = np.full(rm.size, np.nan, dtype=np.float64)
    free = valid.ravel().copy()
    for f, anchor in enumerate(anchors_idx):
        out[anchor] = value_flat[per_field[f][0]]
        free[anchor] = False

    nearest = nearest_available_order(n)
    order = rng.permutation(n_fields)
    q = deque((int(f), 1) for f in order if len(per_field[f]) > 1)
    while q:
        f, pos = q.popleft()
        source_idx = int(per_field[f][pos])
        offset = coordinates[source_idx] - peaks[f]
        target_xy = anchors_xy[f] + offset
        # Out-of-arena targets are resolved toward the nearest valid edge.
        target_xy = np.clip(target_xy, 0, n-1)
        target_idx = int(target_xy[0]*n + target_xy[1])
        if free[target_idx]:
            dst_idx = target_idx
        else:
            # Nearby cells are normally free; break at the first free one.
            dst_idx = next((int(candidate) for candidate in nearest[target_idx]
                            if free[candidate]), None)
            if dst_idx is None:
                raise RuntimeError("No free destination for a visited cell")
        out[dst_idx] = value_flat[source_idx]
        free[dst_idx] = False
        if pos + 1 < len(per_field[f]):
            q.append((f, pos+1))

    if np.any(free):
        raise RuntimeError("Field shuffle did not fill all observed bins")
    return out.reshape(rm.shape)


def _score_task(task):
    unit_id, rm, n_shuffles, seed, mask_parameters, half = task
    from grid_cells.scores import GridScorer
    scorer = GridScorer(rm.shape[0], ((-half,half),(-half,half)), mask_parameters)
    observed = float(scorer.get_scores(rm)[0])
    try:
        seg = segment_raw_fields(rm)
    except ValueError as exc:
        return unit_id, observed, np.nan, np.nan, 0, np.full(n_shuffles, np.nan), str(exc)
    rng = np.random.default_rng(np.random.SeedSequence([seed, unit_id]))
    null = np.empty(n_shuffles, dtype=np.float64)
    for j in range(n_shuffles):
        shuffled = shuffle_fields(rm, seg, rng)
        null[j] = float(scorer.get_scores(shuffled)[0])
    t95 = float(np.quantile(null, 0.95))
    p = float((1 + np.count_nonzero(null >= observed))/(n_shuffles+1))
    return unit_id, observed, t95, p, len(seg[1]), null, ""


def compute_ratemaps(args):
    """Stream activations; don't retain >1GB tensor of 4000*100*512 activations."""
    import torch
    from config import Config
    from grid_cells import evaluate
    from grid_cells.dataset import build_dataloader
    from grid_cells.ensembles import encode_initial_conditions

    cfg = Config()
    cfg.model.nh_bottleneck = args.bottleneck_units
    device = torch.device(args.device)
    model, _, epoch, place, hd = evaluate.load_models(cfg, args.checkpoint, device)
    model.eval()
    loader = build_dataloader(args.shard_dir, shard_indices=None,
                              batch_size=cfg.train.minibatch_size,
                              shuffle=False, num_workers=0)
    nbins = args.nbins
    sums = np.zeros((nbins*nbins, args.bottleneck_units), dtype=np.float64)
    counts = np.zeros(nbins*nbins, dtype=np.int64)
    seen = 0
    half = cfg.task.env_size/2
    with torch.no_grad():
        for batch in loader:
            bsize = len(batch['init_pos'])
            if seen >= args.n_trajectories:
                break
            if seen+bsize > args.n_trajectories:
                batch = {k:v[:args.n_trajectories-seen] for k,v in batch.items()}
                bsize = len(batch['init_pos'])
            init_conds = encode_initial_conditions(batch['init_pos'].to(device),
                             batch['init_hd'].to(device), place, hd)
            out = model(init_conds, batch['ego_vel'].to(device))
            a = out.bottleneck.detach().cpu().numpy().reshape(-1, args.bottleneck_units)
            xy = batch['target_pos'].numpy().reshape(-1,2)
            valid = (xy[:,0]>=-half)&(xy[:,0]<=half)&(xy[:,1]>=-half)&(xy[:,1]<=half)
            ix = np.floor((xy[valid,0]+half)/(2*half)*nbins).astype(int).clip(0,nbins-1)
            iy = np.floor((xy[valid,1]+half)/(2*half)*nbins).astype(int).clip(0,nbins-1)
            idx = ix*nbins+iy
            av = a[valid]
            counts += np.bincount(idx, minlength=nbins*nbins)
            for j in range(args.bottleneck_units):
                sums[:,j] += np.bincount(idx, weights=av[:,j], minlength=nbins*nbins)
            seen += bsize
            if seen % 1000 == 0 or seen == args.n_trajectories:
                print(f'  collected {seen}/{args.n_trajectories} trajectories',flush=True)
    if seen < args.n_trajectories:
        raise RuntimeError(f'Only {seen} trajectories available; wanted {args.n_trajectories}')
    maps = np.full_like(sums, np.nan)
    maps[counts>0] = sums[counts>0]/counts[counts>0,None]
    return maps.T.reshape(args.bottleneck_units,nbins,nbins), int(epoch), cfg


def self_test():
    rng = np.random.default_rng(15)
    n = 32
    x,y = np.meshgrid(np.arange(n),np.arange(n),indexing='ij')
    hexmap = sum(np.exp(-((x-(7+9*c+4*(r%2)))**2+(y-(6+9*r))**2)/13.0)
                 for c in range(3) for r in range(3))
    tests = [('hexagon',hexmap),('negative',hexmap-1.8),
             ('random',rng.standard_normal((n,n)))]
    hole = hexmap.copy(); hole[:2,:3] = np.nan
    tests.append(('missing-bin',hole))
    for name,rm in tests:
        labels, peaks=segment_raw_fields(rm)
        out=shuffle_fields(rm,(labels,peaks),rng)
        assert np.array_equal(np.isfinite(rm), np.isfinite(out))
        assert np.array_equal(np.sort(rm[np.isfinite(rm)]),np.sort(out[np.isfinite(out)]))
        assert np.all(labels[np.isfinite(rm)]>0)
        print(f'  {name}: fields={len(peaks)}, bin values and visit mask preserved')
    print('PASS: no-smoothing watershed, city-block collision handling, and field shuffle invariants')


def save_preview(path: Path, rm, seed: int, unit: int):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seg = segment_raw_fields(rm)
    rng=np.random.default_rng(np.random.SeedSequence([seed,unit]))
    fig,axes=plt.subplots(1,5,figsize=(14,3.3))
    lo,hi=np.nanpercentile(rm,[1,99])
    displays=[rm,seg[0].astype(float)]
    displays.extend([shuffle_fields(rm,seg,rng) for _ in range(3)])
    titles=['Original (raw)','Watershed labels','Shuffle 1','Shuffle 2','Shuffle 3']
    for a,d,t in zip(axes,displays,titles):
        if t=='Watershed labels':
            a.imshow(d,cmap='tab20',interpolation='nearest')
        else:
            a.imshow(d,cmap='viridis',interpolation='nearest',vmin=lo,vmax=hi)
        a.set_title(t,fontsize=9);a.axis('off')
    fig.suptitle(f'Unit {unit} | no smoothing | {len(seg[1])} fields',fontsize=11)
    fig.tight_layout()
    fig.savefig(path,dpi=140)
    plt.close(fig)


def write_results(out_dir,results,args,epoch,mask_params):
    results=sorted(results,key=lambda x:x[0])
    with (out_dir/'unit_statistics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['unit','gridness','shuffle_95','empirical_p','n_fields',
                         'above_0.37','above_mean_shuffle_95','above_own_shuffle_95','warning'])
        valid=[r for r in results if np.isfinite(r[2])]
        mean_t95=float(np.mean([r[2] for r in valid])) if valid else np.nan
        for u,g,t,p,k,null,warn in results:
            writer.writerow([u,g,t,p,k,int(g>0.37),int(g>mean_t95) if np.isfinite(mean_t95) else '',
                             int(g>t) if np.isfinite(t) else '',warn])
    np.savez_compressed(out_dir/'null_distributions.npz',
                        units=np.array([r[0] for r in results]),
                        null_scores=np.stack([r[5] for r in results]))
    count=lambda test: int(sum(bool(test(r)) for r in results))
    summary=dict(epoch=epoch,source_checkpoint=str(Path(args.checkpoint).resolve()),
                 n_trajectories=args.n_trajectories,n_units=len(results),
                 n_shuffles=args.n_shuffles,annulus=mask_params,
                 n_testable=len(valid),mean_shuffle_95=mean_t95,
                 above_0_37=count(lambda r:r[1]>0.37),
                 above_common_shuffle_threshold=count(lambda r:r[1]>mean_t95),
                 above_own_shuffle_95=count(lambda r:np.isfinite(r[2]) and r[1]>r[2]),
                 individual_empirical_p_at_05=count(lambda r:np.isfinite(r[3]) and r[3]<=0.05),
                 procedure='unsmoothed_raw_maxima + watershed_ift + round_robin_field_relocation',
                 exact_original_banino_implementation=False,
                 caveats=['Raw maps and null maps not smoothed',
                          'All 8-neighbour regional maxima used: risk of oversegmentation',
                          'Additional segmentation details not given in Banino 2018',
                          '0.37 shown for comparison only; not recalibrated to this annulus',
                          'Individual 5% tests are not multiple-testing corrected'])
    (out_dir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4))
        ax.hist([r[2] for r in valid],bins=20)
        ax.axvline(0.37,c='k',ls='--',label='Paper reference 0.37')
        if np.isfinite(mean_t95):
            ax.axvline(mean_t95,c='tab:orange',label=f'Mean shuffled 95% = {mean_t95:.3f}')
        ax.set(xlabel='Per-unit shuffled gridness 95th percentile',ylabel='Units')
        ax.legend();fig.tight_layout()
        fig.savefig(out_dir/'shuffle_thresholds.png',dpi=140)
        plt.close(fig)
    except ImportError:
        pass
    print('\n=== RESULTS (exploratory, literature-guided approximation) ===')
    print(f'  mean shuffled 95% cutoff: {mean_t95:.3f}')
    print(f"  observed gridness > 0.37: {summary['above_0_37']}/{len(results)}")
    print(f"  observed gridness > mean shuffled cutoff: {summary['above_common_shuffle_threshold']}/{len(results)}")
    print(f"  observed gridness > own shuffled 95% cutoff: {summary['above_own_shuffle_95']}/{len(results)}")
    print(f"  empirical one-sided p<=.05: {summary['individual_empirical_p_at_05']}/{len(results)}")
    print('  NOTE: Small pilot (e.g. 10 shuffles) cannot resolve p<=.05.')
    print('  NOTE: Field counts should be checked for raw-map over-segmentation.')
    print('  saved',out_dir/'unit_statistics.csv',out_dir/'summary.json',out_dir/'shuffle_thresholds.png',sep='\n  ')


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--self-test',action='store_true',help='Check shuffle invariants without checkpoint')
    parser.add_argument('--checkpoint')
    parser.add_argument('--shard-dir')
    parser.add_argument('--out-dir',default='results/field_shuffle_nosmoothing')
    parser.add_argument('--n-trajectories',type=int,default=4000)
    parser.add_argument('--n-shuffles',type=int,default=100)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--nbins',type=int,default=32)
    parser.add_argument('--bottleneck-units',type=int,default=512)
    parser.add_argument('--unit-ids',help='Comma separated unit IDs for a small pilot, e.g. 235,415,74,367')
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if not args.checkpoint or not args.shard_dir:
        parser.error('--checkpoint and --shard-dir are required unless --self-test')
    if args.workers<1 or args.n_shuffles<2 or args.nbins<8 or args.n_trajectories<1:
        parser.error('workers>=1, n-shuffles>=2, nbins>=8, n-trajectories>=1')
    import torch
    args.device='cuda' if torch.cuda.is_available() else 'cpu'
    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    print(f'Loading trained bottleneck Ratemaps: {args.n_trajectories} trajectories, device={args.device}',flush=True)
    maps,epoch,cfg=compute_ratemaps(args)
    from grid_cells import evaluate
    scorer=evaluate.build_scorer(cfg,args.nbins)
    mask_params=[[float(lo),float(hi)] for _,(lo,hi) in scorer._masks]
    print('Using annuli from installed PyTorch scorer:',mask_params,flush=True)
    if args.unit_ids:
        unit_ids=list(dict.fromkeys(int(x.strip()) for x in args.unit_ids.split(',')))
    else:
        unit_ids=list(range(args.bottleneck_units))
    if not unit_ids or min(unit_ids)<0 or max(unit_ids)>=args.bottleneck_units:
        parser.error('invalid --unit-ids')
    try:
        save_preview(out_dir/f'unit_{unit_ids[0]}_shuffle_preview.png',
                     maps[unit_ids[0]],args.seed,unit_ids[0])
    except ValueError as ex:
        print('Skipping preview:',ex)
    fingerprint={'algorithm_version':2,
                 'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'checkpoint':str(Path(args.checkpoint).resolve()),
                 'checkpoint_stat':(Path(args.checkpoint).stat().st_size,Path(args.checkpoint).stat().st_mtime_ns),
                 'shard_dir':str(Path(args.shard_dir).resolve()),
                 'n_trajectories':args.n_trajectories,
                 'mask_params':mask_params,'seed':args.seed,'n_shuffles':args.n_shuffles}
    key=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()[:16]
    cache=out_dir/'by_unit'/key
    cache.mkdir(parents=True,exist_ok=True)
    tasks=[];results=[]
    for u in unit_ids:
        path=cache/f'unit_{u:03d}.npz'
        if path.exists():
            with np.load(path) as q:
                results.append((u,float(q['observed']),float(q['threshold']),float(q['p']),
                                int(q['n_fields']),q['null'].copy(),str(q['warning'].item())))
        else:
            tasks.append((u,maps[u],args.n_shuffles,args.seed,mask_params,cfg.task.env_size/2))
    print(f'Checkpoint epoch {epoch} | {len(unit_ids)} selected | {len(results)} cached | {len(tasks)} to analyze',flush=True)
    if args.workers==1:
        iterator=map(_score_task,tasks)
        for result in iterator:
            _handle_result(result,cache,results,len(unit_ids))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(_score_task,task) for task in tasks]
            for fut in as_completed(jobs):
                _handle_result(fut.result(),cache,results,len(unit_ids))
    write_results(out_dir,results,args,epoch,mask_params)


def _handle_result(result,cache,results,total):
    u,g,t,p,k,null,warn=result
    np.savez_compressed(cache/f'unit_{u:03d}.npz',observed=g,threshold=t,p=p,
                        n_fields=k,null=null,warning=warn)
    results.append(result)
    if len(results)%4==0 or len(results)==total:
        print(f'  finished {len(results)}/{total} | last unit {u} | {k} fields',flush=True)


if __name__=='__main__':
    main()
