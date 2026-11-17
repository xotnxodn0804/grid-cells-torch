#!/usr/bin/env python3
"""Evaluate a trained 272-D Grid LSTM with its visual cue input enabled vs ablated.

Run from the same WSL project where train_grid_vision_supervised.py lives.
This never trains or overwrites any checkpoint.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from config import Config
from grid_cells.ensembles import build_rl_ensembles
from train_grid_vision_supervised import Episodes, VisionCorrectedGrid, evaluate, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--vision-checkpoint", type=Path, required=True)
    ap.add_argument("--grid-checkpoint", type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=100)
    ap.add_argument("--val-batches", type=int, default=32)
    args = ap.parse_args()
    if min(args.batch_size, args.seq_len, args.val_batches) < 1:
        ap.error("batch-size, seq-len and val-batches must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    ckpt = torch.load(args.grid_checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("grid_input") != 272:
        raise ValueError("Not a 272-input Grid checkpoint")
    fp = ckpt["fingerprint"]
    if fp.get("seq_len") != args.seq_len:
        raise ValueError(f"Checkpoint trained with seq_len={fp.get('seq_len')}")
    if fp.get("batch_size") != args.batch_size:
        raise ValueError(f"Checkpoint trained with batch_size={fp.get('batch_size')}")
    if [str(p.resolve()) for p in args.data_dirs] != fp["data_dirs"]:
        raise ValueError("Dataset paths differ from the training checkpoint")

    from train_grid_vision_supervised import file_hash
    if file_hash(args.vision_checkpoint.resolve()) != fp["vision_checkpoint_sha256"]:
        raise ValueError("Vision checkpoint differs from the training checkpoint")

    set_seed(int(fp["seed"]))
    cfg = Config()
    pc_list, hd_list = build_rl_ensembles(cfg, device)
    pc, hd = pc_list[0], hd_list[0]
    model = VisionCorrectedGrid(pc, hd, args.vision_checkpoint.resolve(), device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.grid.eval()
    model.vision.eval()

    val = Episodes([p.resolve() for p in args.data_dirs], "val")
    settings = SimpleNamespace(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        val_batches=args.val_batches,
        seed=int(fp["seed"]),
        vision_prob=0.05,
    )
    print(f"Checkpoint: {args.grid_checkpoint} | trained updates: {ckpt['step']:,}")
    print(f"Validation: {args.val_batches} batches × {args.batch_size} trajectories"
          f" × {args.seq_len} steps = {args.val_batches*args.batch_size*args.seq_len:,} points")
    print("Same validation trajectory sampling for every condition.")
    print(f"{'Cue availability':<22}{'Val CE':>12}{'Position (cm)':>18}{'HD (deg)':>13}")
    results = {}
    for cue_prob in (0.05, 0.0):
        settings.vision_prob = cue_prob
        with torch.inference_mode():
            ce, pos, hderr = evaluate(model, val, pc, hd, cfg, device, settings)
        results[cue_prob] = (ce, pos, hderr)
        label = "5% vision (normal)" if cue_prob == 0.05 else "0% vision (ablation)"
        print(f"{label:<22}{ce:>12.4f}{pos:>18.2f}{hderr:>13.2f}")

    normal = results[0.05]
    zero = results[0.0]
    print(f"\nRemoving vision changes position error by {zero[1]-normal[1]:+.2f} cm "
          f"and HD error by {zero[2]-normal[2]:+.2f} degrees.")
    print("This is an inference-time ablation; it does not measure how a "
          "separately trained motion-only model would perform.")


if __name__ == "__main__":
    main()
