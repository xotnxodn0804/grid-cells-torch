#!/usr/bin/env python3
"""READ-ONLY Bottleneck-512 rate maps and gridness for the 300K DMLab 272-D Grid Network.

Uses the *exact* deterministic held-out inputs and forward pass used by
train_grid_dmlab_vision_pc001_272_noisy.validate(): 100 episodes x 1300 steps,
noisy 4-D motion, 5% frozen-vision cues, resets at teleports, LSTM state carried
across 100-step segments. Grid and Vision checkpoints are never modified.

Gridness: 20x20 occupancy-normalized UNSMOOTHED rate maps; 39x39 SAC;
DeepMind public 10 annuli (inner 4 bins, 10 outer radii 8..20 bins);
calculates both (a) public-code MEAN formula and (b) Methods MINMAX formula.
The historical score > 0.37 is descriptive only, not field-shuffle significance.

Requires this script in the grid-cells-torch repository next to the trainer.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

NBINS = 20
UNITS = 512
HALF_ARENA_M = 1.25
HISTORICAL_THRESHOLD = 0.37


def accumulate(sums, counts, xy, acts, n=NBINS, half=HALF_ARENA_M):
    """Accumulate the *signed* pre-dropout bottleneck activation by X,Y bin."""
    xy = np.asarray(xy).reshape(-1, 2)
    acts = np.asarray(acts).reshape(-1, sums.shape[-1])
    good = np.isfinite(xy).all(axis=-1) & np.isfinite(acts).all(axis=-1)
    good &= (np.abs(xy) <= half + 1e-5).all(axis=-1)
    xy, acts = xy[good], acts[good]
    if len(xy) == 0:
        return 0
    indices = np.floor((xy + half) * n / (2 * half)).astype(np.int32)
    indices = np.clip(indices, 0, n - 1)
    flat_bin = indices[:, 0] * n + indices[:, 1]
    counts += np.bincount(flat_bin, minlength=n*n)
    # Vectorized accumulation; shared rate-map bins, one column per bottleneck unit.
    np.add.at(sums, flat_bin, acts)
    return len(flat_bin)


def make_ratemaps(sums, counts, n=NBINS):
    out = np.full(sums.shape, np.nan, dtype=np.float32)
    good = counts > 0
    out[good] = (sums[good] / counts[good, None]).astype(np.float32)
    return out.T.reshape(sums.shape[-1], n, n)


def split_half_pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return float('nan')
    x, y = a[m].astype(np.float64), b[m].astype(np.float64)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def self_test():
    n, units = 20, 3
    sums = np.zeros((n*n, units), np.float64)
    counts = np.zeros(n*n, np.int64)
    xy = np.array([[-1.0, -1.0],[-1.0,-1.0],[.2,.3]], dtype=np.float32)
    acts = np.array([[2,3,4], [4,5,6], [8,9,10]], dtype=np.float32)
    assert accumulate(sums, counts, xy, acts) == 3
    r = make_ratemaps(sums, counts)
    b = int(np.floor((-.1e1+1.25)*n / 2.5))
    assert abs(r[0,b,b] - 3) < 1e-6
    assert int(counts.sum()) == 3
    assert np.isfinite(r).sum() == 6
    assert abs(split_half_pearson(np.arange(100), np.arange(100))-1) < 1e-9
    print('SELF TEST OK | occupancy, activation averages, split-half correlation')


def score_ratemaps(ratemaps):
    from grid_cells.scores import GridScorer, deepmind_mask_parameters

    class MeanScorer(GridScorer):
        def grid_score_60(self, c):
            return (c[60]+c[120])/2 - (c[30]+c[90]+c[150])/3

    kwargs = dict(nbins=NBINS,
                  coords_range=((-HALF_ARENA_M, HALF_ARENA_M),
                                (-HALF_ARENA_M, HALF_ARENA_M)),
                  mask_parameters=deepmind_mask_parameters(NBINS))
    mean_scorer = MeanScorer(**kwargs)
    minmax_scorer = GridScorer(**kwargs)

    scores_mean = np.full(UNITS, np.nan, np.float64)
    scores_minmax = np.full(UNITS, np.nan, np.float64)
    annuli_mean = np.full((UNITS, 2), np.nan, np.float64)
    annuli_minmax = np.full((UNITS, 2), np.nan, np.float64)
    sacs = np.empty((UNITS, NBINS*2-1, NBINS*2-1), np.float32)

    for u in range(UNITS):
        rm = ratemaps[u]
        if np.isfinite(rm).sum() < 50 or np.nanstd(rm) < 1e-8:
            sacs[u] = 0
            continue
        # SAC and rotations are identical for both formulae; only the ring-selection
        # score changes. The methods return the best-scoring of 10 candidate rings.
        sm, am, sac = mean_scorer.get_scores(rm)
        sx, ax, _ = minmax_scorer.get_scores(rm)
        scores_mean[u], scores_minmax[u] = float(sm), float(sx)
        annuli_mean[u], annuli_minmax[u] = am, ax
        sacs[u] = sac
        if (u+1) % 128 == 0:
            print(f'SCORED | {u+1}/512 bottleneck units', flush=True)
    return scores_mean, scores_minmax, annuli_mean, annuli_minmax, sacs


def plots(out_dir, ratemaps, sacs, occupancy, scores_mean, scores_minmax, stability):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    order = np.argsort(np.nan_to_num(scores_mean, nan=-1e9))[::-1]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, scores, title in zip(axs,
                                 [scores_mean, scores_minmax],
                                 ['DeepMind code: mean', 'Paper Methods: min-max']):
        valid = scores[np.isfinite(scores)]
        if len(valid):
            ax.hist(valid, bins=40, color='steelblue')
        ax.axvline(HISTORICAL_THRESHOLD, color='black', ls='--',
                   label='Historical 0.37 (NOT a p-value)')
        ax.set_title(title); ax.set_xlabel('Gridness score'); ax.set_ylabel('Units')
        ax.legend(fontsize=8)
    fig.suptitle('300K DMLab Grid Network | Bottleneck 512 | 100 held-out episodes')
    fig.tight_layout()
    fig.savefig(out_dir/'gridness_histograms.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8,5.2))
    im = ax.imshow(occupancy.T, origin='lower', cmap='viridis')
    ax.set_title('Validation position coverage, 20x20')
    ax.set_xlabel('X bin'); ax.set_ylabel('Y bin')
    fig.colorbar(im, ax=ax, label='Time steps')
    fig.tight_layout(); fig.savefig(out_dir/'occupancy.png', dpi=150)
    plt.close(fig)

    with PdfPages(out_dir/'all_512_ratemaps.pdf') as pdf:
        for offset in range(0, UNITS, 64):
            fig, axs = plt.subplots(8, 8, figsize=(16, 17))
            subset = order[offset:offset+64]
            fig.suptitle(f'Sorted by mean Gridness | units {offset+1}-{offset+len(subset)}/512\n'
                         '20x20 raw position rate maps, linear bottleneck, no smoothing',
                         fontsize=13)
            for ax, u in zip(axs.flat, subset):
                ax.imshow(ratemaps[u].T, origin='lower', cmap='viridis', interpolation='nearest')
                ax.set_title(f'#{u}  mean={scores_mean[u]:.2f}', fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            fig.subplots_adjust(left=.035,right=.98,bottom=.025,top=.91,
                                wspace=.22,hspace=.52)
            pdf.savefig(fig)
            plt.close(fig)
        print('SAVED | all_512_ratemaps.pdf (8 pages)', flush=True)

    with PdfPages(out_dir/'top_32_ratemaps_sacs.pdf') as pdf:
        for start in range(0, 32, 8):
            fig, axs = plt.subplots(4, 4, figsize=(13,13))
            fig.suptitle('Top mean-gridness units | rate map (left) and SAC (right)\n'
                         '0.37 is only a historical comparison, not shuffle significance',
                         fontsize=12)
            for i, u in enumerate(order[start:start+8]):
                ax1, ax2 = axs.flat[2*i:2*i+2]
                ax1.imshow(ratemaps[u].T, origin='lower', cmap='viridis', interpolation='nearest')
                ax2.imshow(sacs[u].T, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1,
                           interpolation='nearest')
                ax1.set_title(f'#{u}: {scores_mean[u]:.3f}', fontsize=10)
                ax2.set_title(f'SAC; split r={stability[u]:.2f}', fontsize=10)
                for ax in (ax1, ax2):
                    ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout(rect=(0,.01,1,.93))
            pdf.savefig(fig); plt.close(fig)
    print('SAVED | top_32_ratemaps_sacs.pdf (4 pages)', flush=True)


def check_reference(out_ref, measured):
    if not out_ref.is_file():
        print(f'REFERENCE CHECK SKIPPED | not found: {out_ref}', flush=True)
        return
    report = json.loads(out_ref.read_text(encoding='utf-8'))
    ref = report['full_100']['metrics']
    for k in ['ce', 'pos_weighted_cm', 'hd_circular_deg']:
        observed, expected = float(measured[k]), float(ref[k])
        tolerance = 0.004 if k == 'ce' else 0.04
        if abs(observed - expected) > tolerance:
            raise ValueError(f'VALIDATION MISMATCH for {k}: got {observed:.5f}, '
                             f'reference {expected:.5f}. Do not interpret Gridness '
                             'until forward/data mismatch is resolved.')
    print('REFERENCE CHECK PASSED | CE / weighted position / HD match original val100', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data-dirs', nargs='+', type=Path, required=False)
    p.add_argument('--grid-checkpoint', type=Path)
    p.add_argument('--vision-checkpoint', type=Path)
    p.add_argument('--val-batch-size', type=int, default=8)
    p.add_argument('--max-episodes', type=int, default=100,
                   help='100 for full analysis (default); smaller is only a preview')
    p.add_argument('--out-dir', type=Path, default=Path('results/grid_dmlab_300k_gridness'))
    p.add_argument('--reference-json', type=Path, default=Path('data/evaluations/grid_dmlab_300k_val100.json'))
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if not args.data_dirs or args.grid_checkpoint is None or args.vision_checkpoint is None:
        p.error('--data-dirs, --grid-checkpoint and --vision-checkpoint required')
    if args.val_batch_size < 1 or not 1 <= args.max_episodes <= 100:
        p.error('val-batch-size must be >=1 and max-episodes must be 1..100')

    import train_grid_dmlab_vision_pc001_272_noisy as baseline
    from eval_grid_dmlab_vision_pc001_272_val100 import check_float

    data_dirs = [d.expanduser().resolve() for d in args.data_dirs]
    ckpt_path = args.grid_checkpoint.expanduser().resolve()
    vision_path = args.vision_checkpoint.expanduser().resolve()
    if not ckpt_path.is_file() or not vision_path.is_file():
        raise FileNotFoundError('Missing model checkpoint')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA selected but not available')
    device = torch.device('cuda' if args.device=='auto' and torch.cuda.is_available()
                          else 'cpu' if args.device=='auto' else args.device)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    fp = ckpt.get('fingerprint', {})
    if fp.get('script_version') != 'grid_vision_dmlab_20261112_v2_272_noisy':
        raise ValueError('Wrong training checkpoint version')
    if fp.get('motion_source') != 'VEL.TRANS_XY_plus_measured_yaw_delta_noisy':
        raise ValueError('Wrong checkpoint motion source')
    if fp.get('data_dirs') != [str(d) for d in data_dirs]:
        raise ValueError('Dataset paths differ from training checkpoint')
    if fp.get('episode_steps') != baseline.STEPS or float(fp.get('pc_sigma',-1)) != baseline.SIGMA:
        raise ValueError('Episode steps or PC sigma mismatch')
    check_float('dt', fp['dt'], baseline.DT)
    if fp.get('vision_sha') != baseline.file_digest(vision_path):
        raise ValueError('Vision checkpoint SHA mismatch')

    baseline.seed_everything(int(fp['seed']))
    grid, vision, pc, hd, vision_update = baseline.make_ensembles_and_models(device, vision_path)
    grid.load_state_dict(ckpt['grid'], strict=True)
    if not torch.equal(ckpt['pc_centers'].cpu(), pc.means.cpu()):
        raise ValueError('Place Cell centers inconsistent with checkpoint')
    if not torch.equal(ckpt['hd_centers'].cpu(), hd.means.cpu()):
        raise ValueError('Head Direction centers inconsistent with checkpoint')
    if grid.rl_lstm.input_size != 272 or grid.bottleneck.out_features != UNITS:
        raise ValueError('Expected 272 input, 512 bottleneck grid network')
    grid.eval(); vision.eval(); grid.requires_grad_(False); vision.requires_grad_(False)

    val = baseline.EpisodeStore(data_dirs, 'val')
    if len(val) != 100:
        raise ValueError(f'Expected 100 held-out validation episodes, got {len(val)}')
    if fp.get('val_episodes') != len(val):
        raise ValueError('Validation episode count mismatch')
    settings = SimpleNamespace(val_batch_size=args.val_batch_size,
        vision_rate=float(fp['vision_rate']), force_first_vision=bool(fp['force_first']),
        speed_noise_std=float(fp['speed_noise_std']),
        angular_noise_std=float(fp['angular_noise_std']),
        tbptt=int(fp['tbptt']), seed=int(fp['seed']))

    nbin2 = NBINS*NBINS
    sums = np.zeros((nbin2, UNITS), np.float64)
    counts = np.zeros(nbin2, np.int64)
    split_sums = np.zeros((2, nbin2, UNITS), np.float64)
    split_counts = np.zeros((2, nbin2), np.int64)
    metric_sums = np.zeros(3, np.float64)  # CE, weighted pos cm, HD deg
    n_frames, n_vis, n_invalid, n_valid = 0, 0, 0, 0

    ids_all = np.arange(args.max_episodes, dtype=np.int64)
    print(f'CHECKPOINT | rollout={ckpt["rollout"]:,} updates={ckpt["optimizer_updates"]:,}; '
          f'input 272 -> bottleneck512; Vision step={vision_update:,}', flush=True)
    print(f'ANALYZING | {len(ids_all)} held-out episodes x 1300 steps; '
          f'20x20; DeepMind 10 annuli; deterministic validation vision/noise', flush=True)
    if args.max_episodes != 100:
        print('PREVIEW ONLY | subset results are not final counts', flush=True)

    # Forward hook ensures that the recorded units are the SAME bottleneck output
    # the evaluator uses; no modified forward path, no different state handling.
    captures = []
    def capture(_module, _inputs, output):
        captures.append(output.detach())
    handle = grid.bottleneck.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for left in range(0, len(ids_all), args.val_batch_size):
                ids = ids_all[left:left+args.val_batch_size]
                batch = baseline.build_rollout(
                    val, ids, vision, device, np.random.default_rng(0),
                    np.random.default_rng(0), settings.vision_rate,
                    settings.force_first_vision, settings.speed_noise_std,
                    settings.angular_noise_std, deterministic_val=True,
                    val_seed=settings.seed+83817)
                b = len(ids)
                n_vis += batch.n_vis; n_invalid += batch.n_invalid
                hidden = (torch.zeros(1,b,128,device=device),
                          torch.zeros(1,b,128,device=device))
                for s in range(0, baseline.STEPS, settings.tbptt):
                    e = min(baseline.STEPS,s+settings.tbptt)
                    if captures:
                        raise RuntimeError('Bottleneck hook still holds previous segment')
                    a, d, hidden = baseline.step_segment(
                        grid, batch.vel[:,s:e], batch.vis[:,s:e],
                        batch.reset[:,s:e], hidden, training=False)
                    if len(captures) != 1:
                        raise RuntimeError('Expected exactly one bottleneck activation per segment')
                    activ = captures.pop().cpu().numpy()
                    pos_t = batch.pos[:,s:e]
                    pos = pos_t.cpu().numpy()
                    if activ.shape != (b,e-s,UNITS):
                        raise RuntimeError(f'Activation mismatch: {activ.shape}')
                    n_valid += accumulate(sums, counts, pos, activ)
                    for i, eid in enumerate(ids):
                        parity = int(eid) % 2
                        accumulate(split_sums[parity],split_counts[parity],
                                   pos[i], activ[i])
                    pc_t, hd_t = baseline.targets(pc, hd, pos_t, batch.hd[:,s:e])
                    ce = (-(pc_t*F.log_softmax(a,-1)).sum(-1)
                          -(hd_t*F.log_softmax(d,-1)).sum(-1))
                    error = baseline.prediction_errors(pc, hd, a, d,
                                         pos_t, batch.hd[:,s:e])
                    metric_sums[0] += float(ce.sum())
                    metric_sums[1] += float(error[1].sum())
                    metric_sums[2] += float(error[3].sum())
                    n_frames += b*(e-s)
                print(f'PROGRESS | {left+len(ids)}/{len(ids_all)} episodes', flush=True)
    finally:
        handle.remove()
    if n_frames != len(ids_all)*baseline.STEPS or counts.sum()!=n_valid:
        raise RuntimeError('Mismatch in time steps/position samples')
    summary_metrics = dict(zip(['ce','pos_weighted_cm','hd_circular_deg'],
                               (metric_sums/n_frames).tolist()))
    print('REPRODUCED VALIDATION | CE={ce:.4f} | Weighted Mean={pos_weighted_cm:.4f}cm | '
          'HD={hd_circular_deg:.4f}deg'.format(**summary_metrics), flush=True)
    if args.max_episodes == 100:
        check_reference(args.reference_json, summary_metrics)
    print(f'POSITION COVERAGE | {int((counts>0).sum())}/400 bins, '
          f'valid samples={n_valid:,}/{n_frames:,}, vision={n_vis/n_frames:.2%}, '
          f'teleport resets={n_invalid}', flush=True)

    ratemaps = make_ratemaps(sums, counts)
    halfmaps = [make_ratemaps(split_sums[i],split_counts[i]) for i in range(2)]
    stability = np.array([split_half_pearson(halfmaps[0][u],halfmaps[1][u])
                          for u in range(UNITS)], dtype=np.float64)
    scores_mean, scores_minmax, rings_mean, rings_minmax, sacs = score_ratemaps(ratemaps)
    order = np.argsort(np.nan_to_num(scores_mean,nan=-1e9))[::-1]
    out = args.out_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    with (out/'gridness_scores_sorted.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['unit','mean_10annuli_score','minmax_10annuli_score',
                    'mean_inner_bins','mean_outer_bins','minmax_inner_bins','minmax_outer_bins',
                    'split_half_pearson','mean_above_0_37_DESCRIPTIVE_ONLY',
                    'minmax_above_0_37_DESCRIPTIVE_ONLY'])
        for u in order:
            w.writerow([int(u), scores_mean[u], scores_minmax[u],
                        *rings_mean[u],*rings_minmax[u],stability[u],
                        int(scores_mean[u]>HISTORICAL_THRESHOLD),
                        int(scores_minmax[u]>HISTORICAL_THRESHOLD)])
    np.savez_compressed(out/'ratemaps_and_scores.npz',
                        ratemaps=ratemaps, occupancy=counts.reshape(NBINS,NBINS),
                        scores_mean=scores_mean,scores_minmax=scores_minmax,
                        split_half_stability=stability,
                        annuli_mean=rings_mean,annuli_minmax=rings_minmax)
    plots(out, ratemaps, sacs, counts.reshape(NBINS,NBINS),
          scores_mean, scores_minmax, stability)
    report = {
        'experiment':'grid_dmlab_vision_pc001_272_noisy_300k_read_only_gridness',
        'checkpoint':str(ckpt_path),'checkpoint_sha256':baseline.file_digest(ckpt_path),
        'optimizer_updates':int(ckpt['optimizer_updates']),
        'vision_checkpoint':str(vision_path),'vision_update':vision_update,
        'validation_episodes':len(ids_all),'validation_steps':n_frames,
        'position_samples_inside_arena':n_valid,'occupied_bins':int((counts>0).sum()),
        'bins':NBINS,'units':UNITS,'vision_fraction':n_vis/n_frames,
        'teleport_resets':n_invalid,'evaluation_metrics':summary_metrics,
        'gridness_mean_formula_above_0_37':int(np.sum(scores_mean>HISTORICAL_THRESHOLD)),
        'gridness_minmax_formula_above_0_37':int(np.sum(scores_minmax>HISTORICAL_THRESHOLD)),
        'gridness_mean_score_mean':float(np.nanmean(scores_mean)),
        'gridness_minmax_score_mean':float(np.nanmean(scores_minmax)),
        'top_10_mean_formula':[{'unit':int(u),'gridness':float(scores_mean[u]),
                                'split_half_r':float(stability[u])} for u in order[:10]],
        'method':'20x20 unsmoothed signed bottleneck rate maps, 39x39 SAC, '
                 'DeepMind 10 annuli (inner=4, outer 8..20 bins), both mean and minmax formulae',
        'caution':'Historical 0.37 count is descriptive NOT statistically significant; '
                  'requires a matched field-shuffle null. Visual cues 5%, noisy ground-truth-assisted yaw.',
    }
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),
                                     encoding='utf-8')
    print('\n=== BOTTLENECK 512 GRIDNESS ===', flush=True)
    print(f'MEAN formula, historical score >0.37: '
          f'{report["gridness_mean_formula_above_0_37"]}/512 (DESCRIPTIVE ONLY)',flush=True)
    print(f'MINMAX formula, historical score >0.37: '
          f'{report["gridness_minmax_formula_above_0_37"]}/512 (DESCRIPTIVE ONLY)',flush=True)
    print('TOP10 mean: '+', '.join(f'#{u}: {scores_mean[u]:.3f}' for u in order[:10]),
          flush=True)
    print('SAVED TO | '+str(out.resolve()), flush=True)
    print('READ-ONLY COMPLETE | neither checkpoint was changed.',flush=True)


if __name__ == '__main__':
    main()
