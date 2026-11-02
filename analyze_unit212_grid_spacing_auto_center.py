"""Unit 212: adaptive central SAC peak, six nearest 8-neighbor maxima, spacing + angles.

User-selected exploratory algorithm (not claimed to be Banino's exact method):
  - compute 20x20 ratemap and 39x39 SAC from trained checkpoint
  - find 8-connected region containing SAC center where SAC > 0.5
  - walk from center along N/S/E/W on center row/column; take largest
    uninterrupted distance over these four directions as exclusion radius
  - find strict 8-neighbor maxima on unsmoothed SAC; remove distance <= radius
  - choose 6 nearest; distances, polar angles, six angular gaps
  - grid spacing is median of 6 distances, in bins and centimeters

No training, weight updates, or field shuffling.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy import ndimage


class MeanGridScorerMixin:
    def grid_score_60(self, corr):
        return (corr[60] + corr[120]) / 2 - (corr[30] + corr[90] + corr[150]) / 3


def deepmind20_scorer(env_size_m=2.2):
    from grid_cells.scores import GridScorer
    class MeanGridScorer(MeanGridScorerMixin, GridScorer):
        pass
    half = env_size_m / 2
    masks = [(4., float(radius)) for radius in np.linspace(8., 20., 10)]
    return MeanGridScorer(20, ((-half, half), (-half, half)), masks)


def adaptive_central_radius(sac: np.ndarray, threshold: float = 0.5):
    """Central superlevel component and four axial extents.

    Array axes represent x, y. The returned radius is integer SAC bins;
    strictly > threshold; zero if only the center survives. A center not
    above threshold is invalid (normally autocorrelation at center ~= 1).
    """
    arr = np.asarray(sac, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] % 2 != 1:
        raise ValueError('Require odd square SAC, e.g. 39x39')
    if not np.isfinite(arr).all():
        raise ValueError('SAC must contain only finite numbers')
    ci = arr.shape[0] // 2
    if not arr[ci, ci] > threshold:
        raise ValueError(f'SAC center is {arr[ci,ci]:.3f}, not above threshold {threshold}')

    above = arr > threshold
    labeled, _ = ndimage.label(above, structure=np.ones((3, 3), dtype=bool))
    center_component = labeled == labeled[ci, ci]

    directions = {'right (+x)': (1, 0), 'left (-x)': (-1, 0),
                  'up (+y)': (0, 1), 'down (-y)': (0, -1)}
    extents = {}
    for name, (sx, sy) in directions.items():
        distance = 0
        for step in range(1, ci + 1):
            ix, iy = ci + sx * step, ci + sy * step
            if not center_component[ix, iy]:
                break
            distance = step
        extents[name] = distance
    radius = float(max(extents.values()))
    return (ci, ci), center_component, extents, radius


def find_six_peaks(sac: np.ndarray, center, exclusion_radius: float,
                   n_peaks: int = 6, max_radius: float | None = None):
    arr = np.asarray(sac, dtype=float)
    neighbors = np.ones((3, 3), dtype=bool)
    neighbors[1, 1] = False
    other_max = ndimage.maximum_filter(arr, footprint=neighbors,
                                       mode='constant', cval=-np.inf)
    locations = np.argwhere(arr > other_max)  # strictly greater than each of 8 neighbors
    if locations.size == 0:
        return np.empty((0, 5)), np.empty((0, 5))
    origin = np.asarray(center, dtype=float)
    relative = locations.astype(float) - origin
    radii = np.hypot(relative[:, 0], relative[:, 1])
    angles = (np.degrees(np.arctan2(relative[:, 1], relative[:, 0])) + 360.) % 360.
    keep = radii > exclusion_radius
    if max_radius is not None:
        keep &= radii <= max_radius
    # columns: x-index, y-index, distance-bins, angle-deg, SAC-value
    points = np.column_stack([locations, radii, angles, arr[locations[:, 0], locations[:, 1]]])[keep]
    if len(points):
        # nearest first; ties broken by highest SAC value, then coordinates
        order = np.lexsort((points[:, 1], points[:, 0], -points[:, 4], points[:, 2]))
        points = points[order]
    return points, points[:n_peaks]


def angular_gaps(chosen):
    if len(chosen) != 6:
        return np.empty(0)
    sorted_angles = np.sort(chosen[:, 3])
    return np.diff(np.r_[sorted_angles, sorted_angles[0] + 360])


def ratemap_for_unit(checkpoint, shard_dir, unit_id=212, n_trajectories=4000,
                     bottleneck_units=512, device='cuda'):
    """Reconstruct a 20x20 ratemap from the SAME checkpoint and data trajectories."""
    import torch
    from config import Config
    from grid_cells import evaluate
    from grid_cells.dataset import build_dataloader
    from grid_cells.ensembles import encode_initial_conditions

    if not 0 <= unit_id < bottleneck_units:
        raise ValueError('Unit index outside bottleneck layer')
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        device = 'cpu'
    dev = torch.device(device)
    cfg = Config()
    cfg.model.nh_bottleneck = bottleneck_units
    model, _untrained, epoch, place, hd = evaluate.load_models(cfg, checkpoint, dev)
    model.eval()
    loader = build_dataloader(shard_dir, shard_indices=None,
                              batch_size=cfg.train.minibatch_size,
                              shuffle=False, num_workers=0)
    nbins = 20
    half = cfg.task.env_size / 2
    sums = np.zeros((nbins * nbins,), dtype=np.float64)
    counts = np.zeros((nbins * nbins,), dtype=np.int64)
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= n_trajectories:
                break
            bs = len(batch['init_pos'])
            if seen + bs > n_trajectories:
                bs = n_trajectories - seen
                batch = {key: val[:bs] for key, val in batch.items()}
            initial = encode_initial_conditions(batch['init_pos'].to(dev),
                                                batch['init_hd'].to(dev), place, hd)
            output = model(initial, batch['ego_vel'].to(dev))
            activations = output.bottleneck[..., unit_id].detach().cpu().numpy().reshape(-1)
            xy = batch['target_pos'].numpy().reshape(-1, 2)
            valid = ((xy[:, 0] >= -half) & (xy[:, 0] <= half) &
                     (xy[:, 1] >= -half) & (xy[:, 1] <= half))
            ix = np.floor((xy[valid, 0] + half) / (2*half)*nbins).astype(int).clip(0, nbins-1)
            iy = np.floor((xy[valid, 1] + half) / (2*half)*nbins).astype(int).clip(0, nbins-1)
            flat_bins = ix*nbins+iy
            counts += np.bincount(flat_bins, minlength=nbins*nbins)
            sums += np.bincount(flat_bins, weights=activations[valid], minlength=nbins*nbins)
            seen += bs
            if seen % 1000 == 0 or seen == n_trajectories:
                print(f'Processed {seen}/{n_trajectories} trajectories', flush=True)
    if seen < n_trajectories:
        raise RuntimeError(f'Only found {seen} trajectories')
    result = np.full(nbins*nbins, np.nan)
    visited = counts > 0
    result[visited] = sums[visited]/counts[visited]
    return result.reshape((nbins, nbins)), float(cfg.task.env_size), int(epoch)


def make_figure(rm, sac, center, center_component, extents, radius,
                all_peaks, selected, bin_cm, gridness, annulus,
                unit, threshold, output):
    fig = plt.figure(figsize=(16, 7.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.35],
                          left=.04, right=.97, bottom=.12, top=.87, wspace=.28)
    ax_rm, ax_sac, ax_txt = [fig.add_subplot(gs[0, k]) for k in range(3)]
    ax_rm.imshow(rm.T, origin='lower', interpolation='nearest', cmap='turbo')
    ax_rm.set_title('20 x 20 rate map')
    ax_rm.set_xlabel('x bin'); ax_rm.set_ylabel('y bin')

    limit = max(1.0, float(np.nanmax(np.abs(sac))))
    ax_sac.imshow(sac.T, origin='lower', interpolation='nearest',
                  cmap='RdBu_r', vmin=-limit, vmax=limit)
    # Display center-connected >0.5 zone in light yellow.
    center_overlay = np.ma.masked_where(~center_component.T,
                                         np.ones_like(sac.T))
    ax_sac.imshow(center_overlay, origin='lower', cmap='autumn',
                  alpha=.17, interpolation='nearest', vmin=0, vmax=1)
    ax_sac.add_patch(Circle(center, radius, fill=False,
                            edgecolor='#ffe900', lw=2.6, ls='--'))
    ax_sac.scatter([center[0]], [center[1]], c='#ffe900', marker='+', s=140, linewidths=2.6)
    chosen_set = {(int(p[0]), int(p[1])) for p in selected}
    remaining = [p for p in all_peaks if (int(p[0]), int(p[1])) not in chosen_set]
    if remaining:
        other = np.asarray(remaining)
        ax_sac.scatter(other[:, 0], other[:, 1], c='white', marker='.', s=24,
                       alpha=.85, label='other maxima')
    for k, (x, y, d, angle, value) in enumerate(selected, start=1):
        ax_sac.plot([center[0], x], [center[1], y], '-', color='#0ceeff', lw=1.5, alpha=.78)
        ax_sac.scatter([x], [y], facecolors='none', edgecolors='#00edff',
                       linewidths=2.1, s=130)
        ax_sac.annotate(str(k), (x,y), xytext=(7,5), textcoords='offset points',
                        fontsize=11, fontweight='bold', color='#00faff',
                        bbox={'facecolor':'black','edgecolor':'none','alpha':.72,'pad':1.8})
    ax_sac.set_xlim(-.5, 38.5); ax_sac.set_ylim(-.5, 38.5)
    ax_sac.set_aspect('equal')
    ax_sac.set_xlabel('SAC x index (center = 19)')
    ax_sac.set_ylabel('SAC y index (center = 19)')
    ax_sac.set_title(f'39 x 39 SAC: connected central peak > {threshold:g}')
    # No legend: peak numbers and center circle are self-explanatory.

    ax_txt.axis('off')
    lines = [f'Unit #{unit} | Mean Gridness = {gridness:.4f}',
             f'Annulus used by gridness: {annulus}',
             f'SAC center: {center} -> relative (0, 0)',
             f'Center SAC correlation = {sac[center]:.4f}',
             f'Central threshold: > {threshold:.2f}',
             ('Axial reach: +x={right}, -x={left}, +y={up}, -y={down} bins'
              .format(right=extents['right (+x)'], left=extents['left (-x)'],
                      up=extents['up (+y)'], down=extents['down (-y)'])),
             f'Exclusion radius: {radius:g} SAC bins',
             '', 'PEAK   dx   dy     bins      cm   angle']
    for i, (x,y,d,ang,val) in enumerate(selected, 1):
        lines.append(f'{i:>4}   {int(x-center[0]):+3d}  {int(y-center[1]):+3d}   {d:6.2f}  {d*bin_cm:7.1f}   {ang:6.1f} deg')
    lines.append('')
    if len(selected) == 6:
        median = np.median(selected[:,2]); gaps = angular_gaps(selected)
        lines.append(f'GRID SPACING: {median:.3f} bins = {median*bin_cm:.2f} cm')
        lines.append('Angle gaps (sorted, circular):')
        lines.append(' / '.join(f'{gap:.1f}' for gap in gaps) + ' deg')
    else:
        lines.append(f'Only {len(selected)}/6 peaks: spacing undefined')
    lines.append(f'1 SAC bin = {bin_cm:g} cm')
    lines.append('Angles measured CCW from positive x-axis.')
    ax_txt.text(.0, .99, '\n'.join(lines), va='top', ha='left',
                family='monospace', fontsize=10, linespacing=1.43)
    fig.suptitle(f'Unit {unit}: adaptive central SAC exclusion + nearest six peaks (8-neighbour)', fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def synthetic_test(output: Path | None = None):
    ix, iy = np.indices((39,39))
    center = (19,19)
    # Central Gaussian exceeds 0.5 out to approx 3 bins; 6 outer Gaussians.
    arr = np.exp(-((ix-19)**2+(iy-19)**2)/(2*2.5**2))
    peaks_xy = [(27,19), (23,26), (15,26), (11,19), (15,12), (23,12)]
    for x,y in peaks_xy:
        arr += .85*np.exp(-((ix-x)**2+(iy-y)**2)/(2*.7**2))
    center, central, lengths, radius = adaptive_central_radius(arr, .5)
    assert center == (19,19) and radius == 2., (center,radius,lengths)
    candidates, chosen = find_six_peaks(arr, center, radius)
    assert len(chosen) == 6
    assert {(int(p[0]),int(p[1])) for p in chosen} == set(peaks_xy), chosen
    assert all(chosen[:,2] > radius)
    gaps = angular_gaps(chosen)
    assert abs(gaps.sum()-360) < 1e-6
    assert np.isclose(np.median(chosen[:,2]), np.sqrt(65)), chosen
    if output:
        rm = np.zeros((20,20))
        make_figure(rm, arr, center, central, lengths, radius,
                    candidates, chosen, 11., np.nan, (4,8), 212, .5, output)
    print('PASS adaptive center threshold, cardinal extents, strict 8-neighbor maxima, six peaks, angles, median')


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--checkpoint', default='data/checkpoints/sonnet_v127_lstm_512/checkpoint_epoch299.pt')
    p.add_argument('--shard-dir', default='data/datasets/gpu1m/shards')
    p.add_argument('--unit', type=int, default=212)
    p.add_argument('--n-trajectories', type=int, default=4000)
    p.add_argument('--bottleneck-units', type=int, default=512)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--max-peak-radius', type=float, default=None)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--out-dir', default='results/sonnet_deepmind20_mean')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        synthetic_test()
        return

    rm, env_m, epoch = ratemap_for_unit(args.checkpoint, args.shard_dir,
                                       unit_id=args.unit, n_trajectories=args.n_trajectories,
                                       bottleneck_units=args.bottleneck_units, device=args.device)
    scorer = deepmind20_scorer(env_m)
    gridness, annulus, sac = scorer.get_scores(rm)
    center, central, extents, radius = adaptive_central_radius(sac, args.threshold)
    candidates, chosen = find_six_peaks(sac, center, radius, max_radius=args.max_peak_radius)
    cm_per_bin = env_m * 100 / 20
    gaps = angular_gaps(chosen)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'unit{args.unit}_adaptive_center_{args.threshold:g}_spacing'
    plot = output_dir / (stem + '.png')
    csv_path = output_dir / (stem + '.csv')
    json_path = output_dir / (stem + '.json')
    array_path = output_dir / (stem + '.npz')
    make_figure(rm, sac, center, central, extents, radius, candidates, chosen,
                cm_per_bin, float(gridness), tuple(annulus), args.unit, args.threshold, plot)
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['rank','sac_x_index','sac_y_index','dx_bins','dy_bins',
                         'distance_bins','distance_cm','angle_deg_from_positive_x_ccw','sac_value'])
        for rank, (x, y, d, angle, val) in enumerate(chosen, 1):
            writer.writerow([rank, int(x), int(y), int(x-center[0]), int(y-center[1]),
                             float(d), float(d*cm_per_bin), float(angle), float(val)])
    med = float(np.median(chosen[:,2])) if len(chosen)==6 else None
    record = {'unit':args.unit, 'epoch':epoch,'n_trajectories':args.n_trajectories,
              'ratemap_shape':list(rm.shape), 'sac_shape':list(sac.shape),
              'gridness_formula':'mean', 'gridness':float(gridness),
              'gridness_best_annulus_radii_bins':list(annulus),
              'center_index':list(center),'center_threshold_strictly_above':args.threshold,
              'center_axial_extents_bins':extents,'exclusion_radius_bins':radius,
              'num_local_maxima_outside_exclusion':len(candidates),
              'num_selected_peaks':len(chosen),'cm_per_sac_bin':cm_per_bin,
              'grid_spacing_bins':med, 'grid_spacing_cm':med*cm_per_bin if med is not None else None,
              'angular_gaps_sorted_deg':gaps.tolist()}
    with json_path.open('w',encoding='utf-8') as f:
        json.dump(record, f, indent=2)
    np.savez_compressed(array_path, ratemap=rm, sac=sac, center_component=central,
                        selected_peaks=chosen, candidates=candidates)

    print(f'\nUNIT {args.unit} (epoch {epoch}) | G_mean={gridness:.4f}')
    print(f'SAC center index={center}; threshold>{args.threshold}, axial spans={extents}')
    print(f'Exclusion circle radius={radius:g} bins')
    print('Peak   dx  dy   distance(bins)  distance(cm)   angle(+x CCW)')
    for rank, (x,y,d,angle,val) in enumerate(chosen,1):
        print(f'{rank:>4}  {int(x-center[0]):+3d} {int(y-center[1]):+3d}      {d:7.3f}         {d*cm_per_bin:7.2f}        {angle:7.2f} deg')
    if med is not None:
        print(f'GRID SPACING = {med:.3f} SAC bins x {cm_per_bin:.2f} cm/bin = {med*cm_per_bin:.2f} cm')
        print('Six angular gaps (deg):',np.round(gaps, 2).tolist())
    else:
        print('GRID SPACING cannot be computed: fewer than six outer local maxima')
    print('Saved figure:',plot.resolve())
    print('Saved CSV:',csv_path.resolve())
    print('Saved JSON:',json_path.resolve())
    print('Saved raw arrays:',array_path.resolve())
    if os.name == 'nt':
        try:
            os.startfile(str(plot.resolve()))
        except OSError:
            pass


if __name__=='__main__':
    main()
