"""Generate the supervised training dataset: simulated rat trajectories written directly as .npz
shards (no intermediate CSV).

Output: <output-dir>/shards/NNNN-of-MMMM.npz plus <output-dir>/meta.json, and data/latest
re-pointed at <output-dir> (train.py reads data/latest/shards by default).

Examples:
    python generate_data.py                          # 100 shards x 10,000 trajectories
    python generate_data.py --num-shards 2 --tag trial
    python generate_data.py --shards 0 1 --output-dir /tmp/check --no-latest
"""
import argparse
import multiprocessing as mp
import os
import time

from config import Config
from grid_cells.data_generation import MotionParams, write_metadata, write_shard


def main():
    cfg = Config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num-shards", type=int, default=None,
                        help="generate shards 0..N-1 of the configured total "
                             f"(default: all {cfg.data.n_shards})")
    parser.add_argument("--shards", type=int, nargs="+", default=None,
                        help="explicit shard indices (overrides --num-shards)")
    parser.add_argument("--trajectories-per-shard", type=int,
                        default=cfg.data.trajectories_per_shard)
    parser.add_argument("--seed", type=int, default=cfg.data.seed)
    parser.add_argument("--output-dir", default=None,
                        help="default: data/datasets/<derived id>")
    parser.add_argument("--tag", default=None, help="suffix for the derived dataset id")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--no-latest", action="store_true",
                        help="do not re-point data/latest at the new dataset")
    args = parser.parse_args()

    total = cfg.data.n_shards
    indices = args.shards if args.shards is not None else list(range(args.num_shards or total))
    p = MotionParams.from_config(cfg)
    dataset_id = (f"square{p.env_size:g}m-{len(indices) * args.trajectories_per_shard}traj-"
                  f"{p.n_steps}steps-seed{args.seed}" + (f"-{args.tag}" if args.tag else ""))
    out_dir = args.output_dir or os.path.join("data", "datasets", dataset_id)
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    write_metadata(os.path.join(out_dir, "meta.json"), p, total,
                   args.trajectories_per_shard, args.seed, indices)

    start = time.time()
    jobs = [(i, total, args.trajectories_per_shard, p, args.seed, shard_dir) for i in indices]
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        for n, path in enumerate(pool.imap_unordered(write_shard, jobs), 1):
            print(f"[{n}/{len(jobs)}] {path} ({time.time() - start:.0f}s)")

    if not args.no_latest:
        latest = os.path.join("data", "latest")
        if os.path.islink(latest):
            os.unlink(latest)
        if not os.path.exists(latest):
            os.symlink(os.path.abspath(out_dir), latest)
            print(f"data/latest -> {out_dir}")
        else:
            print(f"data/latest exists and is not a symlink; left unchanged")
    print(f"done: {len(jobs)} shards in {time.time() - start:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
