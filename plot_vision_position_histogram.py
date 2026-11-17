#!/usr/bin/env python3
"""Plot the real validation position-error histogram for train_vision_combined.py.

Reads the *saved* vision checkpoint and validation RGB/position numpy files.
By default reproduces the exact 10k-image validation subset used by the training
script's eval() call. It does not run DMLab or update model weights.

Example:
  python plot_vision_position_histogram.py \
    --checkpoint data/checkpoints/vision_rh_combined_1000/latest.pt \
    --device cpu --out-dir results/vision_position_errors
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--checkpoint', type=Path,
                   default=Path('data/checkpoints/vision_rh_combined_1000/latest.pt'))
    p.add_argument('--data-dirs', nargs='+', type=Path, default=None,
                   help='Overrides dataset directories saved in checkpoint, if moved')
    p.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cpu')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--eval-samples', type=int, default=None,
                   help='Default uses checkpoint eval_samples (normally 10000)')
    p.add_argument('--all-val', action='store_true',
                   help='Use all validation images instead of original eval subset')
    p.add_argument('--out-dir', type=Path,
                   default=Path('results/vision_position_errors'))
    args = p.parse_args()

    if args.batch_size < 1:
        p.error('--batch-size must be positive')
    if args.eval_samples is not None and args.eval_samples < 1:
        p.error('--eval-samples must be positive')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'Vision checkpoint does not exist: {args.checkpoint}')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use --device cpu.')

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'vision_module' not in ckpt:
        raise ValueError('Checkpoint is not a train_vision_combined.py vision checkpoint')
    fingerprint = ckpt.get('dataset_fingerprint', {})
    dirs = args.data_dirs or [Path(d) for d in fingerprint.get('data_dirs', [])]
    if not dirs:
        raise RuntimeError('Dataset directories not saved in checkpoint; use --data-dirs')
    if any(not d.is_dir() for d in dirs):
        raise FileNotFoundError(
            'One or more saved dataset directories are missing:\n' +
            '\n'.join(f'  {d}: {"OK" if d.is_dir() else "MISSING"}' for d in dirs) +
            '\nPass --data-dirs with their current locations.')

    # Import local project classes after checking the dataset paths.
    from train_vision_combined import MultiSplit
    from grid_cells.vision import VisionModule
    from grid_cells.ensembles import build_rl_ensembles
    from config import Config

    train = MultiSplit(dirs, 'train')
    val = MultiSplit(dirs, 'val')
    cfg = Config()
    (pc,), (hd,) = build_rl_ensembles(cfg, device)
    model = VisionModule(pc, hd, embed_dim=256, mask_prob=0.95,
                         mask_mode='whole').to(device)
    model.load_state_dict(ckpt['vision_module'], strict=True)
    model.eval()

    # Use the exact place-cell centers saved at training time.
    pc_centers = ckpt.get('pc_centers')
    if pc_centers is None:
        print('WARNING: pc_centers not saved; using centers from current config.py')
        pc_centers = pc.means
    pc_centers = torch.as_tensor(pc_centers, device=device, dtype=torch.float32)
    if pc_centers.shape != pc.means.shape:
        raise ValueError(f'Saved place cell centers have wrong shape {pc_centers.shape}')

    if args.all_val:
        indices = np.arange(val.total, dtype=np.int64)
        note = f'all {val.total:,} validation images'
    else:
        n = int(args.eval_samples or fingerprint.get('eval_samples', 10000))
        seed = int(ckpt.get('seed', fingerprint.get('seed', 42)))
        eval_rng = np.random.default_rng(seed + 129731)
        # Training script samples train IDs first with the same eval_rng.
        _ = eval_rng.choice(train.total, size=min(n, train.total), replace=False)
        indices = np.sort(eval_rng.choice(val.total, size=min(n, val.total), replace=False))
        note = f'the {len(indices):,} deterministic validation images from training evaluation'

    print(f'Loaded: update {ckpt.get("update", "?")} / {args.checkpoint}', flush=True)
    print(f'Validation: {note}', flush=True)
    print('Data directories:', ', '.join(map(str, dirs)), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    errors = np.empty(len(indices), dtype=np.float32)
    gt = np.empty((len(indices), 2), dtype=np.float32)
    pred = np.empty((len(indices), 2), dtype=np.float32)

    with torch.inference_mode():
        for left in range(0, len(indices), args.batch_size):
            right = min(len(indices), left + args.batch_size)
            imgs, pos, _hd = val.get_batch(indices[left:right], device)
            e = model.cnn(imgs)
            logits = model.place_head(e)
            xy_hat = F.softmax(logits, dim=-1) @ pc_centers
            gt[left:right] = pos.cpu().numpy()
            pred[left:right] = xy_hat.cpu().numpy()
            errors[left:right] = ((xy_hat - pos).norm(dim=-1) * 100.0).cpu().numpy()
            if right % 2000 < args.batch_size or right == len(indices):
                print(f'{right:,}/{len(indices):,} images evaluated', flush=True)

    # Persist all values to make every plotted measurement auditable.
    csv_path = args.out_dir / 'vision_val_position_errors.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['validation_global_index', 'true_x_m', 'true_y_m',
                    'pred_x_m', 'pred_y_m', 'position_error_cm'])
        for i, x, y, ph_x, ph_y, err in zip(indices, gt[:, 0], gt[:, 1],
                                           pred[:, 0], pred[:, 1], errors):
            w.writerow([int(i), float(x), float(y), float(ph_x), float(ph_y), float(err)])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mean = float(np.mean(errors))
    median = float(np.median(errors))
    p90 = float(np.percentile(errors, 90))
    fig, ax = plt.subplots(figsize=(8, 4.9))
    ax.hist(errors, bins=50, weights=np.ones(len(errors)) * 100 / len(errors),
            color='#5785b5', edgecolor='white', linewidth=.5, alpha=.9)
    ax.axvline(mean, color='#d45046', lw=2, ls='--',
               label=f'Mean: {mean:.1f} cm')
    ax.axvline(median, color='#2a7d55', lw=1.8, ls=':',
               label=f'Median: {median:.1f} cm')
    ax.set_xlabel('Position error (cm)')
    ax.set_ylabel('Validation images per bin (%)')
    ax.set_title(f'Vision module: position error distribution (N={len(errors):,})')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    img_path = args.out_dir / 'vision_position_error_histogram.png'
    fig.savefig(img_path, dpi=220)
    plt.close(fig)

    print(f'Mean position error  : {mean:.2f} cm')
    print(f'Median position error: {median:.2f} cm')
    print(f'90th percentile      : {p90:.2f} cm')
    for threshold in (10, 20, 30, 50):
        print(f'Within {threshold:>2} cm          : {np.mean(errors <= threshold) * 100:.1f}%')
    print('Figure saved:', img_path)
    print('Data CSV saved:', csv_path)
    print('Note: Results should match checkpoint-era val_pos_cm only if data/checkpoint match.')


if __name__ == '__main__':
    main()
