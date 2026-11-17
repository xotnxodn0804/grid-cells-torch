#!/usr/bin/env python3
"""Evaluate a trained standalone supervised GridCellsRNN on NEW R-H trajectories.

No training, no checkpoint changes, no vision module and no DeepMind Lab.
Uses the original repo's trajectory simulator and PC/HD ensembles.

Example:
 python -u eval_sonnet_512_position_hd.py \
   --checkpoint data/checkpoints/sonnet_v127_lstm_512/checkpoint_epoch299.pt \
   --n-trajectories 2000 --batch-size 256 --seed 20261112 \
   --out-dir results/sonnet_v127_lstm_512/position_hd_eval

Evaluation trajectories are freshly generated with torch CUDA RNG seed 20261112,
separate from original GPU training shard seeds 0..99. Each trajectory contains
100 samples covering simulation times t=0,0.15,...,14.85 s.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from config import Config
from generate_data_gpu import cubic_lookup, simulate_gpu
from grid_cells.data_generation import MotionParams
from grid_cells.ensembles import (
    build_ensembles,
    encode_initial_conditions,
    soft_cross_entropy,
)
from grid_cells.training import build_model


POSITION_MODES = ("argmax", "weighted_mean", "top3")
HD_MODES = ("circular", "argmax")


def circular_angle_difference_deg(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    diff = pred - truth
    return torch.rad2deg(torch.atan2(torch.sin(diff), torch.cos(diff)).abs())


def decode_hd(probs: torch.Tensor, preferred_angles: torch.Tensor, mode: str):
    if mode == "argmax":
        return preferred_angles[probs.argmax(dim=-1)]
    if mode == "circular":
        return torch.atan2(
            (probs * torch.sin(preferred_angles)).sum(dim=-1),
            (probs * torch.cos(preferred_angles)).sum(dim=-1),
        )
    raise ValueError(mode)


def describe(a: np.ndarray) -> dict:
    # a is [N, T], in cm for position or degrees for HD.
    final = a[:, -1]
    return {
        "first_mean": float(a[:, 0].mean()),
        "last_mean": float(final.mean()),
        "last_median": float(np.median(final)),
        "last_p90": float(np.quantile(final, 0.9)),
        "all_timesteps_mean": float(a.mean()),
    }


def make_figures(pos_err, hd_err, examples, timestamps, out_dir, half_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode in POSITION_MODES:
        ax.plot(timestamps, pos_err["model"][mode].mean(axis=0), label=f"model / {mode}")
    ax.plot(timestamps, pos_err["target_decoder"]["top3"].mean(axis=0),
            "--", label="true-PC decoding / top3")
    ax.set(xlabel="Elapsed time (s)", ylabel="Mean position error (cm)",
           title="Standalone Grid Network: position decoding error")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "position_error_by_time.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode in HD_MODES:
        ax.plot(timestamps, hd_err["model"][mode].mean(axis=0), label=f"model / {mode}")
    ax.plot(timestamps, hd_err["target_decoder"]["circular"].mean(axis=0),
            "--", label="true-HD decoding / circular")
    ax.set(xlabel="Elapsed time (s)", ylabel="Mean heading error (degrees)",
           title="Standalone Grid Network: heading decoding error")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "hd_error_by_time.png", dpi=170)
    plt.close(fig)

    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.2), squeeze=False)
    for i, (ax, e) in enumerate(zip(axes[0], examples)):
        ax.plot(e["truth"][:, 0], e["truth"][:, 1], color="black", label="actual")
        ax.plot(e["pred_top3"][:, 0], e["pred_top3"][:, 1],
                color="tab:blue", alpha=0.9, label="predicted (top3)")
        ax.scatter(*e["truth"][0], marker="o", s=30, color="green", zorder=5)
        ax.scatter(*e["truth"][-1], marker="x", s=40, color="red", zorder=5)
        ax.set(xlim=(-half_size, half_size), ylim=(-half_size, half_size),
               xlabel="x (m)", ylabel="y (m)", title=f"Independent trajectory {i + 1}")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "example_trajectories.png", dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, default=Path(
        "data/checkpoints/sonnet_v127_lstm_512/checkpoint_epoch299.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path(
        "results/sonnet_v127_lstm_512/position_hd_eval"))
    ap.add_argument("--n-trajectories", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--cubic-weights", type=int, default=32)
    ap.add_argument("--check-only", action="store_true",
                    help="Load checkpoint and verify 512-unit model without simulation")
    args = ap.parse_args()
    if args.n_trajectories <= 0 or args.batch_size <= 0 or args.cubic_weights <= 0:
        ap.error("--n-trajectories, --batch-size, and --cubic-weights must be > 0")
    if args.device != "cuda":
        ap.error("The repository's generate_data_gpu.py simulator requires CUDA")
    if not torch.cuda.is_available():
        ap.error("CUDA is unavailable; activate .venv-dmlab and verify torch.cuda.is_available()")
    if not args.checkpoint.is_file():
        ap.error(f"Checkpoint not found: {args.checkpoint}")

    cfg = Config()
    device = torch.device("cuda:0")
    place_list, hd_list = build_ensembles(cfg, device)
    if len(place_list) != 1 or len(hd_list) != 1:
        raise ValueError("This evaluator expects exactly one PC and one HD ensemble")
    place, hd = place_list[0], hd_list[0]
    model = build_model(cfg, place_list + hd_list, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()  # Essential: disables bottleneck dropout.
    assert cfg.model.nh_bottleneck == 512 and model.nh_bottleneck == 512
    assert model.output_heads[0].out_features == 256
    assert model.output_heads[1].out_features == 12
    assert torch.allclose(model.target_ensembles[0].means, place.means)
    assert torch.allclose(model.target_ensembles[1].means, hd.means)

    epoch = checkpoint.get("epoch", "unknown")
    print(f"MODEL OK | epoch={epoch}, LSTM={model.nh_lstm}, bottleneck={model.nh_bottleneck}, "
          f"PC={place.n_cells}, HD={hd.n_cells}", flush=True)
    print("CHECKPOINT:", args.checkpoint, flush=True)
    if args.check_only:
        print("CHECK OK | weights loaded strictly; NO training or simulation executed.")
        return

    # Independent simulated trajectories: not a split of the old training shards.
    p = MotionParams.from_config(cfg)
    if p.n_steps != 100:
        raise ValueError(f"Unexpected n_steps={p.n_steps}; expected 100")
    dt = p.duration_s / p.n_steps
    timestamps = np.arange(p.n_steps, dtype=np.float64) * dt
    lookup = cubic_lookup(p, args.cubic_weights, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pos_err: dict[str, dict[str, list[np.ndarray]]] = {
        c: {mode: [] for mode in POSITION_MODES}
        for c in ("model", "target_decoder")
    }
    hd_err: dict[str, dict[str, list[np.ndarray]]] = {
        c: {mode: [] for mode in HD_MODES}
        for c in ("model", "target_decoder")
    }
    examples: list[dict] = []
    place_ce_sum = hd_ce_sum = 0.0
    frames_evaluated = 0

    print(f"EVALUATION | fresh independent GPU trajectories={args.n_trajectories} | "
          f"seed={args.seed} | dt={dt:g}s | 100 time samples", flush=True)
    print("No gradient updates; no vision input; checkpoint is read-only.", flush=True)

    with torch.inference_mode():
        for offset in range(0, args.n_trajectories, args.batch_size):
            take = min(args.batch_size, args.n_trajectories - offset)
            arr = simulate_gpu(take, p, lookup, generator, device)
            init_pos = torch.from_numpy(arr["init_pos"]).to(device)
            init_hd = torch.from_numpy(arr["init_hd"]).to(device)
            ego_vel = torch.from_numpy(arr["ego_vel"]).to(device)
            true_pos = torch.from_numpy(arr["target_pos"]).to(device)
            true_hd = torch.from_numpy(arr["target_hd"]).to(device).squeeze(-1)
            assert torch.allclose(init_pos, true_pos[:, 0], atol=1e-5)
            init_conds = encode_initial_conditions(init_pos, init_hd, place_list, hd_list)
            out = model(init_conds, ego_vel)
            pred_pc = torch.softmax(out.logits[0], dim=-1)
            pred_hd = torch.softmax(out.logits[1], dim=-1)
            gt_pc = place.posterior(true_pos)
            gt_hd = hd.posterior(true_hd.unsqueeze(-1))

            place_ce_sum += float(soft_cross_entropy(out.logits[0], gt_pc).sum().item())
            hd_ce_sum += float(soft_cross_entropy(out.logits[1], gt_hd).sum().item())
            frames_evaluated += take * p.n_steps

            for cond, pc_probs, hd_probs in (
                ("model", pred_pc, pred_hd),
                ("target_decoder", gt_pc, gt_hd),
            ):
                for mode in POSITION_MODES:
                    pred_xy = place.decode_position(pc_probs, mode)
                    e = ((pred_xy - true_pos).norm(dim=-1) * 100.0).cpu().numpy()
                    pos_err[cond][mode].append(e)
                for mode in HD_MODES:
                    angle = decode_hd(hd_probs, hd.means, mode)
                    e = circular_angle_difference_deg(angle, true_hd).cpu().numpy()
                    hd_err[cond][mode].append(e)

            # Save three example trajectories for qualitative inspection.
            for i in range(min(take, 3 - len(examples))):
                examples.append({
                    "truth": true_pos[i].cpu().numpy().copy(),
                    "pred_top3": place.decode_position(pred_pc[i], "top3").cpu().numpy(),
                })
            done = offset + take
            print(f"  evaluated {done}/{args.n_trajectories} trajectories", flush=True)

    pos_err = {c: {mode: np.concatenate(xs, axis=0)
                   for mode, xs in modes.items()} for c, modes in pos_err.items()}
    hd_err = {c: {mode: np.concatenate(xs, axis=0)
                  for mode, xs in modes.items()} for c, modes in hd_err.items()}

    result = {
        "experiment": "standalone_supervised_sonnet_512_position_and_HD",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": epoch,
        "evaluation_seed": args.seed,
        "evaluation_data": "fresh_generated_Raudies_Hasselmo_simulated_trajectories",
        "evaluation_is_new_data_not_training_shards": True,
        "n_trajectories": args.n_trajectories,
        "n_steps": p.n_steps,
        "sample_interval_seconds": dt,
        "sample_times": "0, 0.15, ..., 14.85 seconds; trajectory sim horizon 15 sec",
        "mean_place_CE": place_ce_sum / frames_evaluated,
        "mean_HD_CE": hd_ce_sum / frames_evaluated,
        "mean_total_CE": (place_ce_sum + hd_ce_sum) / frames_evaluated,
        "position_error_cm": {cond: {m: describe(v) for m, v in modes.items()}
                              for cond, modes in pos_err.items()},
        "heading_error_deg": {cond: {m: describe(v) for m, v in modes.items()}
                              for cond, modes in hd_err.items()},
        "position_notes": "top3 averages three most probable place-cell centers; weighted_mean uses all; model trained on PC/HD soft CE, not coordinate MAE",
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    with (args.out_dir / "errors_by_timestep.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["step", "elapsed_sec", "position_argmax_cm", "position_weighted_mean_cm",
                     "position_top3_cm", "heading_circular_deg", "heading_argmax_deg",
                     "top3_true_PC_decoding_floor_cm", "circular_true_HD_decoding_floor_deg"])
        for t in range(p.n_steps):
            wr.writerow([t, round(float(timestamps[t]), 4),
                         *[float(pos_err["model"][mode][:, t].mean())
                           for mode in POSITION_MODES],
                         *[float(hd_err["model"][mode][:, t].mean())
                           for mode in HD_MODES],
                         float(pos_err["target_decoder"]["top3"][:, t].mean()),
                         float(hd_err["target_decoder"]["circular"][:, t].mean())])

    with (args.out_dir / "per_trajectory_final_errors.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["trajectory", "top3_final_cm", "weighted_mean_final_cm",
                     "argmax_final_cm", "HD_circular_final_deg", "HD_argmax_final_deg"])
        for i in range(args.n_trajectories):
            wr.writerow([i,
                         float(pos_err["model"]["top3"][i, -1]),
                         float(pos_err["model"]["weighted_mean"][i, -1]),
                         float(pos_err["model"]["argmax"][i, -1]),
                         float(hd_err["model"]["circular"][i, -1]),
                         float(hd_err["model"]["argmax"][i, -1])])

    make_figures(pos_err, hd_err, examples, timestamps, args.out_dir,
                 half_size=cfg.task.env_size / 2.0)

    print("\n=== POSITION DECODING (cm; lower is better) ===")
    for mode in POSITION_MODES:
        a = result["position_error_cm"]["model"][mode]
        floor = result["position_error_cm"]["target_decoder"][mode]
        print(f"  {mode:<14} first={a['first_mean']:6.2f}  final={a['last_mean']:6.2f} "
              f" median(final)={a['last_median']:6.2f}  overall={a['all_timesteps_mean']:6.2f} "
              f" | true-PC decoding floor(final)={floor['last_mean']:.2f}")
    print("=== HEAD DIRECTION DECODING (degrees) ===")
    for mode in HD_MODES:
        a = result["heading_error_deg"]["model"][mode]
        floor = result["heading_error_deg"]["target_decoder"][mode]
        print(f"  {mode:<14} first={a['first_mean']:6.2f}  final={a['last_mean']:6.2f} "
              f" median(final)={a['last_median']:6.2f}  overall={a['all_timesteps_mean']:6.2f} "
              f" | true-HD decoding floor(final)={floor['last_mean']:.2f}")
    print(f"=== SUPERVISED CROSS ENTROPY ===\n"
          f"  PC={result['mean_place_CE']:.4f}, HD={result['mean_HD_CE']:.4f}, "
          f"total={result['mean_total_CE']:.4f}")
    print("NOTE: Mean over new independent trajectories, not the old training loss 4.06.")
    print("NOTE: 100th sample corresponds to 14.85 s; simulation horizon is 15 s.")
    print("SAVED:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
