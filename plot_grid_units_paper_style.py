"""Paper-style panels for the CURRENT Sonnet-1.27 512-unit supervised checkpoint.

Run from the grid-cells-torch repository root (PowerShell):
    python plot_grid_units_paper_style.py --self-test
    python plot_grid_units_paper_style.py
    python plot_grid_units_paper_style.py --all

NO model training and NO checkpoint modification. Reuses the repository's
GridScorer, rate-map conventions and plotting helpers. The third (polar) row
is HEAD-DIRECTION TUNING, not a rotation-correlation curve.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

NBINS = 20
N_HD_BINS = 20
THRESHOLD = 0.37  # Historical *exploratory* cutoff; not re-calibrated for mean scoring.
SCRIPT_VERSION = 1


def scorer_for_current_analysis(environment_m: float):
    """Same 20-bin, 10-annulus, mean gridness configuration as the existing analysis."""
    from grid_cells.scores import GridScorer, deepmind_mask_parameters

    class MeanScorer(GridScorer):
        def grid_score_60(self, corr):
            return (corr[60] + corr[120]) / 2.0 - (corr[30] + corr[90] + corr[150]) / 3.0

    half = environment_m / 2.0
    return MeanScorer(NBINS, ((-half, half), (-half, half)),
                      deepmind_mask_parameters(NBINS))


def _inputs(shard_dir: Path, trajectories: int):
    shards = sorted(shard_dir.glob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"No NPZ shards found in: {shard_dir}")
    # Shards are 10,000 trajectories in this project. Fingerprint all shards to be safe.
    return shards


def cache_metadata(args, shards):
    ckpt = args.checkpoint.resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Supervised checkpoint not found: {ckpt}")
    stat = ckpt.stat()
    return {
        "version": SCRIPT_VERSION,
        "checkpoint": str(ckpt),
        "checkpoint_bytes": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "shard_dir": str(args.shard_dir.resolve()),
        "shards": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in shards],
        "n_trajectories": args.n_trajectories,
        "nbins": NBINS,
        "hd_bins": N_HD_BINS,
    }


def _accumulate_binned_sums(sums, counts, indices, activations):
    """Accumulate average activation per spatial/directional bin without huge tensors."""
    indices = np.asarray(indices, dtype=np.int64)
    acts = np.asarray(activations)
    n_units = sums.shape[1]
    assert acts.ndim == 2 and acts.shape[0] == indices.size and acts.shape[1] == n_units
    counts += np.bincount(indices, minlength=len(counts))
    # Flatten (bin,unit) into a 1-D histogram; equivalent to per-unit np.bincount.
    keys = (indices[:, None] * n_units + np.arange(n_units, dtype=np.int64)).ravel()
    weighted = np.bincount(keys, weights=acts.ravel(), minlength=sums.size)
    sums += weighted.reshape(sums.shape)


def infer_maps_and_tuning(args, shards):
    """Only the first N sorted trajectories; never loads the entire 1M dataset into RAM."""
    import torch
    from config import Config
    from grid_cells.ensembles import build_ensembles, encode_initial_conditions
    from grid_cells.training import build_model

    cfg = Config()
    cfg.model.nh_bottleneck = 512
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    place, hd = build_ensembles(cfg, device)
    model = build_model(cfg, place + hd, device)
    # Loading weights_only=False is appropriate only for your own trusted local checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise ValueError("Expected a SUPERVISED checkpoint with key 'model', not an RL checkpoint")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()  # IMPORTANT: no bottleneck dropout; model weights are frozen.
    epoch = int(checkpoint.get("epoch", -1))
    n_units = model.nh_bottleneck
    half = cfg.task.env_size / 2.0
    sums = np.zeros((NBINS * NBINS, n_units), dtype=np.float64)
    counts = np.zeros(NBINS * NBINS, dtype=np.int64)
    hd_sums = np.zeros((N_HD_BINS, n_units), dtype=np.float64)
    hd_counts = np.zeros(N_HD_BINS, dtype=np.int64)
    u_idx = np.arange(n_units, dtype=np.int64)
    seen = 0
    print(f"Epoch {epoch} | device {device} | units {n_units} | inference only", flush=True)
    with torch.inference_mode():
        for shard_path in shards:
            if seen >= args.n_trajectories:
                break
            with np.load(shard_path, allow_pickle=False) as shard:
                available = int(shard["init_pos"].shape[0])
                taken = min(available, args.n_trajectories - seen)
                # Read each compressed field ONCE, not every batch. Otherwise npz decompression
                # would be repeated hundreds of times for the same shard.
                samples = {key: shard[key][:taken] for key in
                           ("init_pos", "init_hd", "ego_vel", "target_pos", "target_hd")}
                for start in range(0, taken, args.batch_size):
                    end = min(start + args.batch_size, taken)
                    init_pos = torch.from_numpy(samples["init_pos"][start:end]).to(device)
                    init_hd = torch.from_numpy(samples["init_hd"][start:end]).to(device)
                    ego_vel = torch.from_numpy(samples["ego_vel"][start:end]).to(device)
                    init = encode_initial_conditions(init_pos, init_hd, place, hd)
                    out = model(init, ego_vel)
                    a = out.bottleneck.cpu().numpy().reshape(-1, n_units)
                    xy = samples["target_pos"][start:end].reshape(-1, 2)
                    headings = samples["target_hd"][start:end].reshape(-1)
                    valid = ((xy[:, 0] >= -half) & (xy[:, 0] <= half) &
                             (xy[:, 1] >= -half) & (xy[:, 1] <= half))
                    scaled = np.floor((xy[valid] + half) / (2 * half) * NBINS)
                    scaled = np.clip(scaled.astype(np.int64), 0, NBINS - 1)
                    idx = scaled[:, 0] * NBINS + scaled[:, 1]
                    _accumulate_binned_sums(sums, counts, idx, a[valid])
                    hd_idx = np.clip(np.digitize(headings,
                                      np.linspace(-np.pi, np.pi, N_HD_BINS + 1)) - 1,
                                     0, N_HD_BINS - 1)
                    _accumulate_binned_sums(hd_sums, hd_counts, hd_idx, a)
                    del a, out
                seen += taken
                del samples
            print(f"  processed {seen}/{args.n_trajectories} trajectories", flush=True)
    if seen < args.n_trajectories:
        raise RuntimeError(f"Only {seen} trajectories available; requested {args.n_trajectories}")
    ratemaps = np.full_like(sums, np.nan)
    valid_bins = counts > 0
    ratemaps[valid_bins] = sums[valid_bins] / counts[valid_bins, None]
    hd_tuning = np.zeros_like(hd_sums)
    observed_hd = hd_counts > 0
    hd_tuning[observed_hd] = hd_sums[observed_hd] / hd_counts[observed_hd, None]
    if not np.all(observed_hd):
        print("WARNING: Some heading bins were not visited; HD tuning is incomplete")
    return ratemaps.T.reshape(n_units, NBINS, NBINS), hd_tuning.T, counts, epoch, cfg.task.env_size


def score_all(ratemaps, env_size_m):
    scorer = scorer_for_current_analysis(env_size_m)
    scores = np.empty(len(ratemaps), dtype=np.float64)
    sacs = np.empty((len(ratemaps), 2 * NBINS - 1, 2 * NBINS - 1), dtype=np.float64)
    annuli = np.empty((len(ratemaps), 2), dtype=np.float64)
    for unit, rate_map in enumerate(ratemaps):
        score, annulus, sac = scorer.get_scores(rate_map)
        scores[unit], sacs[unit], annuli[unit] = score, sac, annulus
    return scores, sacs, annuli


def correlations_for_best_annulus(scorer, sac, annulus):
    """Match grid_cells.scores.GridScorer.get_grid_scores_for_mask exactly."""
    mask = scorer._get_ring_mask(float(annulus[0]), float(annulus[1]))
    area = float(mask.sum())
    masked_sac = sac * mask
    mean = masked_sac.sum() / area
    centered = (masked_sac - mean) * mask
    variance = (centered ** 2).sum() / area + 1e-5
    corrs = {}
    for angle, rotated in zip(scorer._corr_angles,
                              scorer.rotated_sacs(sac, scorer._corr_angles)):
        centered_rotated = (rotated - mean) * mask
        corrs[angle] = float(np.sum(centered * centered_rotated) / area / variance)
    return corrs, mask


def save_csv(out_dir, scores, annuli, reference):
    path = out_dir / "visualization_unit_scores.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["unit", "mean_gridness", "annulus_inner_bins", "annulus_outer_bins",
                    "above_exploratory_0_37"])
        for unit, (s, a) in enumerate(zip(scores, annuli)):
            w.writerow([unit, float(s), float(a[0]), float(a[1]), int(s > THRESHOLD)])
    if reference.is_file():
        with reference.open(newline="", encoding="utf-8-sig") as f:
            table = {int(r["unit"]): float(r["observed_gridness"])
                     for r in csv.DictReader(f)}
        diffs = [abs(float(s) - table[i]) for i, s in enumerate(scores) if i in table]
        if diffs:
            print(f"Reference CSV comparison: max |delta gridness| = {max(diffs):.7f}")
            if max(diffs) > 1e-3:
                print("WARNING: Scores differ from reference; check trajectory subset, "
                      "checkpoint, and analysis settings before comparing counts!")
    print(f"Scores CSV: {path}")


def overview_page(ratemaps, sacs, annuli, scores, tuning, units, title, cols=4):
    from grid_cells.figures import plot_unit_grid
    return plot_unit_grid(ratemaps, sacs, annuli, scores, NBINS,
                          indices=units, cols=cols, threshold=THRESHOLD,
                          hd_tuning=tuning, panel_size=1.72, title=title)


def detailed_figure(unit, ratemaps, sacs, annuli, scores, tuning, env_size_m):
    from grid_cells.figures import plot_ratemap, plot_sac, plot_hd_tuning
    scorer = scorer_for_current_analysis(env_size_m)
    sac = sacs[unit]
    annulus = annuli[unit]
    corrs, mask = correlations_for_best_annulus(scorer, sac, annulus)
    reconstructed = scorer.grid_score_60(corrs)
    if not np.isclose(reconstructed, scores[unit], atol=1e-7, rtol=1e-7):
        raise AssertionError(f"Unit {unit}: corr/score mismatch: {reconstructed} vs {scores[unit]}")

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 0.9], hspace=.42, wspace=.35)
    a0 = fig.add_subplot(gs[0, 0]); plot_ratemap(ratemaps[unit], ax=a0, title="Rate map (20x20)")
    a1 = fig.add_subplot(gs[0, 1]); plot_sac(sac, NBINS, mask_params=annulus, ax=a1,
                                             title="Circular SAC (39x39)")
    a2 = fig.add_subplot(gs[0, 2])
    a2.imshow(np.where(mask > 0, sac, np.nan), interpolation="none", cmap="jet")
    a2.set_title(f"Selected annulus: {annulus[0]:.1f}-{annulus[1]:.1f} bins", fontsize=9)
    a2.axis("off")
    a3 = fig.add_subplot(gs[0, 3], projection="polar")
    plot_hd_tuning(tuning[unit], ax=a3, title="Head-direction tuning")

    ar = fig.add_subplot(gs[1, :2])
    angles = [30, 60, 90, 120, 150]
    values = [corrs[a] for a in angles]
    ar.plot(angles, values, "o-", lw=1.6)
    ar.set_xticks(angles); ar.set_xlabel("SAC rotation angle (degrees)")
    ar.set_ylabel("Correlation (selected annulus)")
    ar.set_title("Five rotations used in Mean Gridness", fontsize=10)
    ar.grid(alpha=.25)

    notes = fig.add_subplot(gs[1, 2:]); notes.axis("off")
    notes.text(.04, .91,
               f"Unit {unit}  |  G = {scores[unit]:.4f}\n"
               f"Annulus (SAC pixels): {annulus[0]:.2f} - {annulus[1]:.2f}\n"
               f"Mean(60,120) - Mean(30,90,150)\n"
               f"G > 0.37? {'yes (exploratory)' if scores[unit] > THRESHOLD else 'no'}\n\n"
               "Polar plot = head-direction tuning,\n"
               "NOT rotation correlation.\n\n"
               "0.37 not recalibrated for mean gridness.\n"
               "No field-shuffle threshold is used here.",
               va="top", ha="left", fontsize=11, linespacing=1.55)
    fig.suptitle(f"Sonnet v1.27  |  Supervised checkpoint  |  Bottleneck unit {unit}",
                 fontsize=14)
    return fig


def self_test():
    # Test fast bin sums and circular-mask correlations with synthetic (NOT experimental) data.
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 5))
    idx = rng.integers(0, 6, size=30)
    sums = np.zeros((6, 5), dtype=float); counts = np.zeros(6, dtype=np.int64)
    _accumulate_binned_sums(sums, counts, idx, a)
    for b in range(6):
        assert counts[b] == np.count_nonzero(idx == b)
        assert np.allclose(sums[b], a[idx == b].sum(axis=0))
    scorer = scorer_for_current_analysis(2.2)
    xx, yy = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
    rm = np.cos(2*np.pi*xx/5) + np.cos(2*np.pi*yy/6)
    score, annulus, sac = scorer.get_scores(rm)
    c, mask = correlations_for_best_annulus(scorer, sac, annulus)
    assert sac.shape == (39, 39) and mask.shape == (39, 39)
    assert len(scorer._masks) == 10 and annulus[0] == 4.0
    assert np.isclose(scorer.grid_score_60(c), score, rtol=1e-8, atol=1e-8)
    print("PASS: bin accumulation, 20x20 -> 39x39 SAC, mean formula, 10 annuli")
    print("Self-test uses synthetic values; it does not inspect the trained checkpoint.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, default=Path(
        "data/checkpoints/sonnet_v127_lstm_512/checkpoint_epoch299.pt"))
    ap.add_argument("--shard-dir", type=Path, default=Path("data/datasets/gpu1m/shards"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/paper_style_512"))
    ap.add_argument("--reference-csv", type=Path,
                    default=Path("results/sonnet_deepmind20_mean/unit_statistics.csv"))
    ap.add_argument("--n-trajectories", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--top", type=int, default=12,
                    help="Show this many highest-scoring units in the overview")
    ap.add_argument("--unit-ids", type=str,
                    help="Comma-separated unit IDs to show instead of the top-scoring units")
    ap.add_argument("--details", type=int, default=4,
                    help="Number of unit detail PNGs to save (from selected overview units)")
    ap.add_argument("--all", action="store_true", help="Also generate a multi-page PDF for ALL 512 units")
    ap.add_argument("--refresh", action="store_true", help="Recompute instead of using a matching cache")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.n_trajectories <= 0 or args.batch_size <= 0 or args.top <= 0 or args.details < 0:
        ap.error("n-trajectories, batch-size and top must be positive; details >= 0")
    if not Path("grid_cells/scores.py").is_file() or not Path("config.py").is_file():
        ap.error("Run from the grid-cells-torch repository root (where config.py lives)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = _inputs(args.shard_dir, args.n_trajectories)
    meta = cache_metadata(args, shards)
    cache_path = args.out_dir / "calculated_maps_sacs_hd.npz"
    use_cache = False
    if cache_path.is_file() and not args.refresh:
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache["meta"].item()) == json.dumps(meta, sort_keys=True):
                ratemaps = cache["ratemaps"].copy()
                hd_tuning = cache["hd_tuning"].copy()
                counts = cache["counts"].copy()
                scores = cache["scores"].copy()
                sacs = cache["sacs"].copy()
                annuli = cache["annuli"].copy()
                epoch, env_size_m = int(cache["epoch"]), float(cache["env_size_m"])
                use_cache = True
    if use_cache:
        print(f"Using validated numerical cache: {cache_path}")
    else:
        ratemaps, hd_tuning, counts, epoch, env_size_m = infer_maps_and_tuning(args, shards)
        scores, sacs, annuli = score_all(ratemaps, env_size_m)
        np.savez_compressed(cache_path,
                            meta=json.dumps(meta, sort_keys=True), ratemaps=ratemaps,
                            hd_tuning=hd_tuning, counts=counts, scores=scores,
                            sacs=sacs, annuli=annuli, epoch=epoch, env_size_m=env_size_m)
        print(f"Saved numerical cache: {cache_path}")
    n_units = len(scores)
    n_above = int(np.sum(scores > THRESHOLD))
    print(f"G > {THRESHOLD} (EXPLORATORY): {n_above}/{n_units} ({100*n_above/n_units:.1f}%)")
    print(f"Visited spatial bins: {(counts > 0).sum()}/{NBINS*NBINS}")
    save_csv(args.out_dir, scores, annuli, args.reference_csv)
    if args.unit_ids:
        try:
            selected = list(dict.fromkeys(int(s.strip()) for s in args.unit_ids.split(",") if s.strip()))
        except ValueError as e:
            ap.error(f"Invalid --unit-ids: {e}")
        if not selected or min(selected) < 0 or max(selected) >= n_units:
            ap.error(f"--unit-ids must be between 0 and {n_units - 1}")
    else:
        selected = np.argsort(-scores)[:min(args.top, n_units)].tolist()
    fig = overview_page(ratemaps, sacs, annuli, scores, hd_tuning, selected,
                        f"Sonnet v1.27 | epoch {epoch} | {args.n_trajectories} trajectories | "
                        "Mean Gridness (exploratory G>0.37)")
    overview_png = args.out_dir / "selected_units_paper_style.png"
    overview_pdf = args.out_dir / "selected_units_paper_style.pdf"
    fig.savefig(overview_png, dpi=170, bbox_inches="tight")
    fig.savefig(overview_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Overview: {overview_png}, {overview_pdf}")

    for unit in selected[:args.details]:
        fig = detailed_figure(unit, ratemaps, sacs, annuli, scores, hd_tuning, env_size_m)
        path = args.out_dir / f"unit_{unit:03d}_detail.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"  detail: {path}")
    if args.all:
        pdf_path = args.out_dir / "all_512_units_paper_style.pdf"
        with PdfPages(pdf_path) as pdf:
            for start in range(0, n_units, 16):
                page_units = list(range(start, min(start + 16, n_units)))
                fig = overview_page(ratemaps, sacs, annuli, scores, hd_tuning,
                                    page_units, f"Units {start}–{page_units[-1]} | "
                                    "Rate Map / SAC / Head-direction Tuning")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                print(f"  rendered {page_units[-1]+1}/{n_units} units", flush=True)
        print(f"ALL UNITS: {pdf_path}")
    print("Done. Check output figures; no model parameters were updated.")


if __name__ == "__main__":
    main()
