#!/usr/bin/env python3
"""Collect real DeepMind Lab vision frames and train the VisionModule for 300k updates.

Place next to config.py inside the user's grid-cells-torch checkout.
This script is an optional standalone *pilot* experiment, not the original
Banino et al. online A3C training schedule. It does not modify Grid Network
weights or existing supervised checkpoints.

Example:
  python vision_300k.py collect --train-frames 100000 --val-frames 10000
  python vision_300k.py train --updates 300000 --batch-size 32 --device auto
  python vision_300k.py train --updates 300000 --resume --device auto
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_one_split(env, env_cfg, root: Path, split: str, n_samples: int, seed: int):
    from grid_cells.rl_common import convert_observation
    from grid_cells.actor_critic import DISCRETE_ACTIONS

    img_file = root / f"{split}_images.npy"
    pos_file = root / f"{split}_pos.npy"
    hd_file = root / f"{split}_hd.npy"
    done_file = root / f"{split}_complete.json"
    for path in (img_file, pos_file, hd_file, done_file):
        if path.exists():
            raise FileExistsError(f"Already exists: {path} (use a different --data-dir)")

    images = np.lib.format.open_memmap(img_file, mode="w+", dtype=np.uint8,
                                       shape=(n_samples, 3, env_cfg.height, env_cfg.width))
    positions = np.lib.format.open_memmap(pos_file, mode="w+", dtype=np.float32,
                                          shape=(n_samples, 2))
    headings = np.lib.format.open_memmap(hd_file, mode="w+", dtype=np.float32,
                                         shape=(n_samples,))
    rng = np.random.default_rng(seed)
    # Encourage walking forward, occasional turns, and some strafe/backtracking.
    action_probs = np.array([0.20, 0.20, 0.40, 0.06, 0.07, 0.07])
    episode_id = 0
    env.reset(seed=seed + episode_id)
    action = 2
    hold = 0

    for i in range(n_samples):
        if not env.is_running():
            episode_id += 1
            env.reset(seed=seed + episode_id)
            hold = 0
        if hold == 0:
            action = int(rng.choice(len(DISCRETE_ACTIONS), p=action_probs))
            hold = int(rng.integers(2, 10))
        hold -= 1
        env.step(DISCRETE_ACTIONS[action], num_steps=env_cfg.action_repeat)
        if not env.is_running():
            episode_id += 1
            env.reset(seed=seed + episode_id)
            hold = 0
        ob = convert_observation(env.observations(), env_cfg)
        images[i] = ob["image"]
        positions[i] = ob["pos"]
        headings[i] = ob["hd"].item()
        if (i + 1) % 5000 == 0 or i + 1 == n_samples:
            print(f"collect {split}: {i + 1:,}/{n_samples:,} frames, {episode_id + 1} episodes", flush=True)

    images.flush()
    positions.flush()
    headings.flush()
    del images, positions, headings
    done_file.write_text(json.dumps({"split": split, "frames": n_samples,
                                     "seed": seed, "env_size_m": env_cfg.env_size_m,
                                     "height": env_cfg.height, "width": env_cfg.width}, indent=2))


def collect_main(args):
    from config import Config
    from grid_cells.rl_common import make_dmlab_env
    cfg = Config()
    cfg.rl.env.goal_visible = args.visible_goal
    cfg.rl.env.level = "square_arena"
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.train_frames < 1 or args.val_frames < 1:
        raise ValueError("--train-frames and --val-frames must be positive")
    print("Collecting REAL DMLab observations; data are not bundled with this script.")
    print("Dataset folder:", root.resolve())
    env = make_dmlab_env(cfg.rl.env, renderer=args.renderer)
    try:
        collect_one_split(env, cfg.rl.env, root, "train", args.train_frames, args.seed)
        # Make validation episodes disjoint from training episodes.
        collect_one_split(env, cfg.rl.env, root, "val", args.val_frames, args.seed + 1_000_000)
    finally:
        env.close()
    print("COLLECTION COMPLETE")


def init_truncated_normal_vision(model: nn.Module) -> None:
    """Explicit Sonnet-Linear-like truncated normal for Conv2d and Linear.

    Fan-in is kernel_h*kernel_w*in_channels for convolution; for Linear,
    fan-in is in_features. Bias is zero. Applied at creation before training,
    even when using an unmodified (upstream) grid_cells/vision.py.
    """
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                fan_in = m.weight[0].numel()
                std = fan_in ** -0.5
                nn.init.trunc_normal_(m.weight, mean=0.0, std=std,
                                      a=-2 * std, b=2 * std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def batch_tensors(images, positions, headings, indices, device):
    # np.array(..., copy=True) makes read-only memmaps safely writable for torch.
    img = torch.from_numpy(np.array(images[indices], copy=True)).to(device)
    img = img.float().div_(127.5).sub_(1.0)
    pos = torch.from_numpy(np.array(positions[indices], copy=True)).to(device)
    hd = torch.from_numpy(np.array(headings[indices], copy=True)).to(device)
    return img, pos, hd


@torch.no_grad()
def evaluate(model, pc, hd_ens, images, poses, headings, device, batch_size=128):
    model.eval()
    ce_total, err_cm_total, err_deg_total, total = 0., 0., 0., 0
    cos_centers = torch.cos(hd_ens.means)
    sin_centers = torch.sin(hd_ens.means)
    for offset in range(0, len(images), batch_size):
        ids = np.arange(offset, min(offset + batch_size, len(images)))
        im, pos, hd = batch_tensors(images, poses, headings, ids, device)
        pc_targets = pc.posterior(pos.unsqueeze(1)).squeeze(1)
        hd_targets = hd_ens.posterior(hd.unsqueeze(1)).squeeze(1)
        embedding = model.cnn(im)
        pl = model.place_head(embedding)
        dl = model.hd_head(embedding)
        loss = (-(pc_targets * F.log_softmax(pl, -1)).sum(-1)
                -(hd_targets * F.log_softmax(dl, -1)).sum(-1)).mean()
        pprob = pl.softmax(-1)
        dprob = dl.softmax(-1)
        pred_xy = pprob @ pc.means
        pred_hd = torch.atan2(dprob @ sin_centers, dprob @ cos_centers)
        angle_err = torch.atan2(torch.sin(pred_hd - hd), torch.cos(pred_hd - hd)).abs()
        n = len(ids)
        ce_total += loss.item() * n
        err_cm_total += (pred_xy - pos).norm(dim=-1).sum().item() * 100
        err_deg_total += torch.rad2deg(angle_err).sum().item()
        total += n
    model.train()
    return {"val_loss": ce_total / total, "pos_error_cm": err_cm_total / total,
            "hd_error_deg": err_deg_total / total}


def train_main(args):
    from config import Config
    from grid_cells.ensembles import build_rl_ensembles
    from grid_cells.vision import VisionModule

    root = Path(args.data_dir)
    for split in ("train", "val"):
        if not (root / f"{split}_complete.json").exists():
            raise FileNotFoundError(f"Missing {split} split in {root}. First run: python vision_300k.py collect")
    train_images = np.load(root / "train_images.npy", mmap_mode="r")
    train_positions = np.load(root / "train_pos.npy", mmap_mode="r")
    train_headings = np.load(root / "train_hd.npy", mmap_mode="r")
    val_images = np.load(root / "val_images.npy", mmap_mode="r")
    val_positions = np.load(root / "val_pos.npy", mmap_mode="r")
    val_headings = np.load(root / "val_hd.npy", mmap_mode="r")
    for split, im, po, hd in (("train",train_images,train_positions,train_headings),
                              ("val",val_images,val_positions,val_headings)):
        if not (len(im) == len(po) == len(hd)) or im.shape[1:] != (3,64,64):
            raise ValueError(f"Inconsistent or unexpected {split} data shapes: {im.shape}, {po.shape}, {hd.shape}")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    _set_seed(args.seed)
    print(f"device={device} | train_frames={len(train_images):,} | val_frames={len(val_images):,}")
    if device.type == "cpu":
        print("WARNING: CPU training selected; check torch.cuda.is_available() for long runs.")
    cfg = Config()
    pc_ensembles, hd_ensembles = build_rl_ensembles(cfg, device)
    if len(pc_ensembles) != 1 or len(hd_ensembles) != 1:
        raise ValueError("This pilot expects one place and one head-direction ensemble")
    pc, hd_ens = pc_ensembles[0], hd_ensembles[0]
    model = VisionModule(pc, hd_ens, embed_dim=256, mask_prob=0.95, mask_mode="whole").to(device)
    init_truncated_normal_vision(model)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-8)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    latest_path = out / "latest.pt"
    csv_path = out / "metrics.csv"
    start = 0
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"Resume requested, checkpoint not found: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint.get("train_frames") != len(train_images):
            raise ValueError("Training frame count differs from checkpoint")
        model.load_state_dict(checkpoint["vision_module"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = int(checkpoint["update"])
        if "numpy_rng_state" in checkpoint:
            rng.bit_generator.state = checkpoint["numpy_rng_state"]
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), device)
        print(f"RESUMED at update {start:,}")
    elif latest_path.exists() or csv_path.exists():
        raise FileExistsError(f"Output already contains a run: {out}. Use --resume or another --out-dir")

    csvfile = open(csv_path, "a" if args.resume else "w", newline="", buffering=1)
    writer = csv.DictWriter(csvfile, fieldnames=["update", "train_loss", "val_loss", "pos_error_cm", "hd_error_deg"])
    if not args.resume:
        writer.writeheader()
    model.train()

    def save_checkpoint(step):
        obj = {
            "vision_module": model.state_dict(), "optimizer": optimizer.state_dict(),
            "update": step, "train_frames": len(train_images),
            "pc_centers": pc.means.detach().cpu(), "hd_centers": hd_ens.means.detach().cpu(),
            "pc_stdev": cfg.rl.pc_scale, "hd_concentration": cfg.task.hdc_concentration,
            "env_size_m": cfg.rl.env.env_size_m, "seed": args.seed,
            "init": "truncated_normal_fan_in", "vision_mask_mode": "whole",
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
        }
        if device.type == "cuda":
            obj["cuda_rng_state"] = torch.cuda.get_rng_state(device)
        tmp_path = out / "latest.tmp.pt"
        torch.save(obj, tmp_path)
        os.replace(tmp_path, latest_path)
        if step % args.save_every == 0 or step == args.updates:
            torch.save(obj, out / f"vision_step{step:06d}.pt")

    try:
        for step in range(start + 1, args.updates + 1):
            idx = rng.integers(0, len(train_images), size=args.batch_size)
            x, pos, hd = batch_tensors(train_images, train_positions, train_headings, idx, device)
            target_pc = pc.posterior(pos.unsqueeze(1)).squeeze(1)
            target_hd = hd_ens.posterior(hd.unsqueeze(1)).squeeze(1)
            loss = model.loss(x, target_pc, target_hd)  # No masking in supervised CE.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % args.eval_every == 0 or step == args.updates:
                result = evaluate(model, pc, hd_ens, val_images, val_positions,
                                  val_headings, device)
                writer.writerow({"update": step, "train_loss": float(loss.item()), **result})
                print(f"step {step:,}/{args.updates:,} | train CE {loss.item():.4f} | "
                      f"val CE {result['val_loss']:.4f} | pos {result['pos_error_cm']:.1f}cm | "
                      f"HD {result['hd_error_deg']:.1f}deg", flush=True)
            if step % args.save_every == 0 or step == args.updates:
                save_checkpoint(step)
    except KeyboardInterrupt:
        print("Interrupted: saving resumable latest checkpoint ...", flush=True)
        save_checkpoint(step)
    finally:
        csvfile.close()
    print(f"Results: {out.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = p.add_subparsers(dest="command", required=True)
    c = sp.add_parser("collect", help="Collect real DMLab images, position and heading")
    c.add_argument("--data-dir", default="data/vision_pilot_20261104")
    c.add_argument("--train-frames", type=int, default=100000)
    c.add_argument("--val-frames", type=int, default=10000)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--renderer", choices=("hardware", "software"), default="hardware")
    c.add_argument("--visible-goal", action="store_true", help="Show goal (default hidden to match paper)")
    c.set_defaults(func=collect_main)
    t = sp.add_parser("train", help="Supervised learning of vision PC/HD outputs")
    t.add_argument("--data-dir", default="data/vision_pilot_20261104")
    t.add_argument("--out-dir", default="data/checkpoints/vision_300k_pilot")
    t.add_argument("--updates", type=int, default=300000)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default="auto")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--eval-every", type=int, default=5000)
    t.add_argument("--save-every", type=int, default=25000)
    t.set_defaults(func=train_main)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
