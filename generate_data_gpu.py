"""GPU-batched Raudies/Hasselmo trajectory generator for soluslan/grid-cells-torch.

Run this script from the grid-cells-torch repository root. No changes to
original source files are required.

Important: This matches the motion model, resampling scheme, and .npz schema,
but uses torch CUDA random numbers rather than NumPy's original RNG stream.
It is a distributional reproduction, NOT bitwise reproduction of CPU shards.

Example (100,000 trajectories, 10 shards x 10,000):
    .\\.venv\\Scripts\\python.exe generate_data_gpu.py --num-shards 10 \\
        --trajectories-per-shard 10000 --batch-size 2048 \\
        --output-dir data/datasets/gpu100k
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d

from config import Config
from grid_cells.data_generation import MotionParams, shard_filename, trajectories_to_arrays
from grid_cells.dataset import FIELDS


def cubic_lookup(p: MotionParams, k: int, device: torch.device):
    """Precompute accurate sparse weights for SciPy's not-a-knot cubic interpolator.

    interp1d(x, identity, kind='cubic') returns the exact linear interpolation
    operator for this *fixed* time grid. The 32 largest-magnitude coefficients
    per output time sample reproduce that operator to ~1e-9 absolute weight
    mass (for the repository's default 750->100 steps).
    """
    fine_t = np.linspace(0.0, p.duration_s, p.n_fine_steps + 1)
    sample_t = np.linspace(0.0, p.duration_s, p.n_steps, endpoint=False)
    operator = interp1d(
        fine_t, np.eye(p.n_fine_steps + 1, dtype=np.float64),
        axis=0, kind="cubic",
    )(sample_t)
    k = min(k, p.n_fine_steps + 1)
    indices = np.argsort(np.abs(operator), axis=1)[:, -k:]
    weights = np.take_along_axis(operator, indices, axis=1)
    omitted_l1 = np.max(np.sum(np.abs(operator), axis=1)
                        - np.sum(np.abs(weights), axis=1))
    print(f"cubic interpolation: {k} coefficients / output point; "
          f"maximum omitted weight L1 = {omitted_l1:.3g}")
    if omitted_l1 > 1e-5:
        raise ValueError("Cubic stencil inaccurate; increase --cubic-weights")

    left = np.searchsorted(fine_t, sample_t, side="right") - 1
    left = np.clip(left, 0, len(fine_t) - 2)
    fraction = (sample_t - fine_t[left]) / (fine_t[left + 1] - fine_t[left])
    return (
        torch.as_tensor(indices, device=device, dtype=torch.long),
        torch.as_tensor(weights, device=device, dtype=torch.float32),
        torch.as_tensor(left, device=device, dtype=torch.long),
        torch.as_tensor(fraction, device=device, dtype=torch.float32),
    )


def cubic_resample(data: torch.Tensor, indices: torch.Tensor,
                   weights: torch.Tensor) -> torch.Tensor:
    """data: (B, fine_steps+1, channels) -> (B, 100, channels)."""
    selected = data[:, indices, :]  # B, 100, K, channels
    return (selected * weights[None, :, :, None]).sum(dim=2)


@torch.no_grad()
def simulate_gpu(batch_size: int, p: MotionParams,
                 lookup: tuple, generator: torch.Generator, device: torch.device):
    """Vectorize independent animals; time updates remain sequential."""
    n = p.n_fine_steps
    dt, L = p.sim_dt, p.env_size
    lo, hi = p.perimeter_m, L - p.perimeter_m
    halfpi, pi = np.pi / 2.0, np.pi
    sig = np.deg2rad(p.rot_vel_std_deg) * dt

    x = torch.rand(batch_size, device=device, generator=generator) * L
    y = torch.rand(batch_size, device=device, generator=generator) * L
    ang = (torch.rand(batch_size, device=device, generator=generator) * 2 - 1) * pi
    vel = torch.zeros_like(x)

    pos_fine = torch.empty((batch_size, n + 1, 2), device=device, dtype=torch.float32)
    ang_fine = torch.empty((batch_size, n + 1), device=device, dtype=torch.float32)
    vel_fine = torch.empty((batch_size, n + 1), device=device, dtype=torch.float32)
    pos_fine[:, 0, 0], pos_fine[:, 0, 1] = x, y
    ang_fine[:, 0], vel_fine[:, 0] = ang, vel

    # Pre-generate independent draws on GPU instead of launching an RNG kernel
    # 1,500 times inside the loop. Both streams are independent. In the CPU
    # implementation, speed samples are only drawn outside wall zones; unused
    # draws here change RNG sequences but do not change the distribution.
    uniform = torch.rand((batch_size, n), device=device, generator=generator)
    random_speed = p.speed_rayleigh_scale * torch.sqrt(-2 * torch.log1p(-uniform))
    noise = torch.randn((batch_size, n), device=device, generator=generator) * sig
    del uniform

    for i in range(1, n + 1):
        # Same branch order as the CPU wall_zone() and wall_redirect().
        a = (ang >= 0) & (ang <= halfpi)
        b = (ang >= halfpi) & (ang <= pi)
        c = (ang >= -pi) & (ang <= -halfpi)
        d = (ang >= -halfpi) & (ang <= 0)

        wa = (x > hi) | (y > hi)
        wb = (x < lo) | (y > hi)
        wc = (x < lo) | (y < lo)
        wd = (x > hi) | (y < lo)
        near_wall = torch.where(a, wa, torch.where(b, wb, torch.where(c, wc,
                                torch.where(d, wd, torch.zeros_like(a)))))

        ra = torch.where(y > hi, -halfpi, torch.where(x > hi, halfpi, 0.0))
        rb = torch.where(y > hi, halfpi, torch.where(x < lo, -halfpi, 0.0))
        rc = torch.where(y < lo, -halfpi, torch.where(x < lo, halfpi, 0.0))
        rd = torch.where(y < lo, halfpi, torch.where(x > hi, -halfpi, 0.0))
        redirect = torch.where(a, ra, torch.where(b, rb, torch.where(c, rc,
                               torch.where(d, rd, 0.0))))

        v_next = torch.where(near_wall, vel * (1 - p.perimeter_speed_reduction),
                             random_speed[:, i - 1])
        delta_angle = noise[:, i - 1] + torch.where(near_wall, redirect, 0.0)

        # Position uses pre-turn heading (exactly as CPU implementation).
        x_next = x + torch.cos(ang) * v_next * dt
        y_next = y + torch.sin(ang) * v_next * dt
        ang_next = ang + delta_angle
        # Preserve original one-turn wrap formula rather than modulo wrapping.
        ang_next = torch.where(torch.abs(ang_next) >= pi,
                               ang_next - (2 * pi) * torch.sign(ang_next), ang_next)
        x, y, ang, vel = x_next, y_next, ang_next, v_next
        pos_fine[:, i, 0], pos_fine[:, i, 1] = x, y
        ang_fine[:, i], vel_fine[:, i] = ang, vel

    indices, weights, left, fraction = lookup
    heading_fine = torch.stack((torch.cos(ang_fine), torch.sin(ang_fine)), dim=-1)
    velocity_fine = vel_fine[..., None] * heading_fine

    pos = cubic_resample(pos_fine, indices, weights)
    hd = cubic_resample(heading_fine, indices, weights)
    hd = hd / torch.linalg.vector_norm(hd, dim=-1, keepdim=True).clamp_min(1e-12)

    # SciPy interp1d(..., kind='linear') for allocentric velocity.
    a_vel = velocity_fine[:, left, :]
    b_vel = velocity_fine[:, left + 1, :]
    v = a_vel + fraction[None, :, None] * (b_vel - a_vel)

    # Reuse EXACT original repository's conversion of physical trajectories
    # into init_pos/init_hd/ego_vel/target_pos/target_hd arrays.
    arrays = trajectories_to_arrays(
        pos.cpu().numpy(), v.cpu().numpy(), hd.cpu().numpy(), L,
    )
    return arrays


def main():
    cfg = Config()
    p = MotionParams.from_config(cfg)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--num-shards", type=int, default=10,
                    help="number of shards to generate (default 10 = 100k trajectories)")
    ap.add_argument("--trajectories-per-shard", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="data/datasets/gpu100k")
    ap.add_argument("--cubic-weights", type=int, default=32)
    ap.add_argument("--resume", action="store_true",
                    help="skip already-existing .npz shards; never overwrite them")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA GPU not available. Run the CUDA check first.")
    for name in ("num_shards", "trajectories_per_shard", "batch_size", "cubic_weights"):
        if getattr(args, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_shards > cfg.data.n_shards:
        ap.error(f"--num-shards must be <= {cfg.data.n_shards} to match naming")
    device = torch.device("cuda:0")
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    print(f"motion: {p.duration_s:g}s / {p.n_fine_steps} fine steps -> {p.n_steps} samples")
    out_dir = Path(args.output_dir)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # All final filenames follow the original convention: NNNN-of-0099.npz
    files = [shards_dir / shard_filename(i, cfg.data.n_shards)
             for i in range(args.num_shards)]
    collisions = [str(f) for f in files if f.exists()]
    if collisions and not args.resume:
        raise SystemExit("Existing shard(s) found; refusing to overwrite. "
                         "Use a new --output-dir or --resume. Example: " + collisions[0])

    lookup = cubic_lookup(p, args.cubic_weights, device)
    meta_path = out_dir / "meta.json"
    metadata = {
        "generator": "torch_cuda_vectorized_distributional_reproduction",
        "warning": "CUDA Torch RNG differs from original NumPy RNG: shards are NOT bitwise identical",
        "motion": asdict(p),
        "n_shards": cfg.data.n_shards,
        "shard_indices": list(range(args.num_shards)),
        "trajectories_per_shard": args.trajectories_per_shard,
        "seed": args.seed,
        "shard_seed_rule": "torch.Generator(device='cuda').manual_seed(seed+shard_index)",
        "batch_size": args.batch_size,
        "cubic_weights": args.cubic_weights,
        "fields": list(FIELDS),
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old != metadata:
            raise SystemExit("Existing meta.json is incompatible; choose a new --output-dir.")
    else:
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    started = time.perf_counter()
    for i, path in enumerate(files):
        if path.exists():
            print(f"[{i + 1}/{args.num_shards}] skipping existing {path}")
            continue
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + i)
        parts = {k: [] for k in FIELDS}
        shard_start = time.perf_counter()
        count = 0
        while count < args.trajectories_per_shard:
            take = min(args.batch_size, args.trajectories_per_shard - count)
            batch = simulate_gpu(take, p, lookup, generator, device)
            for k in FIELDS:
                parts[k].append(batch[k])
            count += take
        shard_arrays = {k: np.concatenate(parts[k], axis=0) for k in FIELDS}
        # Atomic-ish write: interrupted generation should not leave a complete
        # final shard filename that --resume could mistakenly skip.
        temp_path = str(path) + ".partial.npz"
        np.savez(temp_path, **shard_arrays)
        os.replace(temp_path, path)
        torch.cuda.synchronize()
        print(f"[{i + 1}/{args.num_shards}] {path} | "
              f"{time.perf_counter()-shard_start:.1f}s/shard | "
              f"{time.perf_counter()-started:.1f}s total", flush=True)
        del shard_arrays, parts
    print(f"DONE: {args.num_shards * args.trajectories_per_shard:,} trajectories, "
          f"{time.perf_counter()-started:.1f}s | data: {shards_dir}")
    print("Next: verify_gpu_data.py --shard-dir " + str(shards_dir))


if __name__ == "__main__":
    main()
