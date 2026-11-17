#!/usr/bin/env python3
"""Four-process DMLab collector for the EXISTING B-Online sequential trainer.

SAFETY AND CONTINUITY
* Runs the ORIGINAL train_grid_vision_1350_online_B_sequential.py main().
  Does not modify that file, its checkpoints, learning code, parameters, masks,
  Vision weights, or the existing full-history replay.
* Replaces only its call to online_a.generate_episodes() for training episodes
  with a 4-process prefetcher. Checkpoint fingerprints remain those of the
  unchanged original script. Validation generation is kept serial and identical.
* REQUIRES --resume and a checkpoint with Vision fully frozen (300000 updates).
* Each worker creates one persistent independent DMLab environment. Episode
  seeds, RH movement policy, RGB, motion targets and order stay unchanged.
* Python multiprocessing uses spawn (never fork a live CUDA/DMLab process).

RUN A NO-WRITE 4-ENV PROBE FIRST:
  python -u train_grid_vision_1350_B_parallel4.py --probe-only \
    --out-dir data/checkpoints/grid1350_online_B_seq_run1 --workers 4

RESUME ONLY IF THE ORIGINAL TRAINER IS STOPPED:
  python -u train_grid_vision_1350_B_parallel4.py --workers 4 --resume \
    --rollouts 21429 --val-episodes 100 --eval-every 500 --save-every 25 \
    --out-dir data/checkpoints/grid1350_online_B_seq_run1 --device cuda
"""
from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

# This global exists in each spawned process, NOT in the GPU trainer.
_WORKER_ENV = None
_WORKER_CFG = None


def _worker_init(env_cfg, renderer):
    """One DMLab environment per spawned worker, independent of parent CUDA."""
    global _WORKER_ENV, _WORKER_CFG
    from grid_cells.rl_common import make_dmlab_env
    _WORKER_CFG = env_cfg
    _WORKER_ENV = make_dmlab_env(env_cfg, renderer=renderer)


def _worker_episode(eid, seed_base, yaw_gain, nominal_speed):
    """Generate precisely one original RH episode, return exact input arrays."""
    if _WORKER_ENV is None:
        raise RuntimeError('DMLab worker environment was not initialized')
    import train_grid_vision_1350_online_A as online_a
    store, teleports = online_a.generate_episodes(
        _WORKER_ENV, _WORKER_CFG, np.array([eid], dtype=np.int64),
        seed_base, yaw_gain, nominal_speed)
    # Avoid sending both the same arrays and their reshaped views in store.entries.
    return int(eid), {name: arr[0] for name, arr in store._arrays.items()}, int(teleports), os.getpid()


class ParallelEpisodes:
    """Prefetch exactly 10 future train episodes; consume in original order.

    Prefetch does NOT give future RGB to Vision's replay or Grid; we restrict
    usage to fully frozen Vision. Returned data is fed into the original trainer
    row by row in exactly the original order.
    """
    def __init__(self, workers, renderer, train_seed, original_generate):
        self.workers = workers
        self.renderer = renderer
        self.train_seed = train_seed
        self.original_generate = original_generate
        self.pool = None
        self.pending = {}
        self.last_pid = set()
        self.group_start = None

    def __call__(self, env, cfg, ids, seed_base, yaw_gain, nominal_speed):
        import train_grid_vision_1350_online_A as online_a
        ids = np.asarray(ids, dtype=np.int64)
        # Keep fixed-seed held-out VALIDATION strictly identical/serial.
        if seed_base != self.train_seed or ids.size != 1:
            if self.pending:
                raise RuntimeError('Validation began before 10 train episodes were consumed')
            return self.original_generate(env, cfg, ids, seed_base, yaw_gain, nominal_speed)
        eid = int(ids[0])
        if not self.pending:
            if eid % online_a.BATCH != 0:
                raise RuntimeError(f'Unexpected train episode order at {eid} (expected multiple of 10)')
            if self.pool is None:
                self.pool = concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=mp.get_context('spawn'),
                    initializer=_worker_init, initargs=(cfg, self.renderer))
                print(f'PARALLEL START | workers={self.workers} | spawn | frozen Vision', flush=True)
            self.group_start = eid
            self.last_pid = set()
            # Fill worker queues BEFORE consuming episode 1, so generation
            # overlaps with Vision inference and Grid input preparation.
            self.pending = {
                x: self.pool.submit(_worker_episode, x, seed_base, yaw_gain, nominal_speed)
                for x in range(eid, eid + online_a.BATCH)
            }
        expected = self.group_start + (online_a.BATCH - len(self.pending))
        if eid != expected or eid not in self.pending:
            raise RuntimeError(f'Nonsequential episode request {eid}; expected {expected}')
        returned_id, arrays, teleports, pid = self.pending.pop(eid).result()
        if returned_id != eid:
            raise RuntimeError(f'Worker returned episode {returned_id}, expected {eid}')
        self.last_pid.add(pid)
        # Rebuild original compatible episode store; labels and 3D RGB are exact.
        store = online_a.InMemoryEpisodes(np.array([eid], dtype=np.int64))
        for name, target in store._arrays.items():
            if name not in arrays or arrays[name].shape != target[0].shape:
                raise RuntimeError(f'Worker episode {eid}: invalid array {name}')
            np.copyto(target[0], arrays[name])
        if not self.pending:
            print(f'PARALLEL ROLLOUT COLLECTED | episode_ids={self.group_start}..{eid} '
                  f'| worker_pids_used={len(self.last_pid)}/{self.workers}', flush=True)
        return store, teleports

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        self.pending.clear()


def flag_value(argv, name, default=None):
    for i, item in enumerate(argv):
        if item == name:
            if i+1 == len(argv):
                raise ValueError(f'Missing argument for {name}')
            return argv[i+1]
        if item.startswith(name+'='):
            return item[len(name)+1:]
    return default


def check_frozen_checkpoint(directory: Path):
    import torch
    p = directory.expanduser().resolve() / 'latest.pt'
    if not p.is_file():
        raise FileNotFoundError(f'Original checkpoint not found: {p}')
    ck = torch.load(p, map_location='cpu', weights_only=False)
    updates = int(ck.get('vision_updates', -1))
    rollout = int(ck.get('rollout', -1))
    if updates != 300_000:
        raise RuntimeError(f'Parallel mode requires FROZEN Vision=300000; found {updates}')
    if int(ck.get('grid_updates', -1)) != rollout*14:
        raise RuntimeError('Corrupted or incompatible Grid update counter')
    if ck.get('fingerprint', {}).get('version') != 'online_B_sequential_all_history_raw_v1':
        raise RuntimeError('Not the expected sequential B checkpoint')
    print(f'CHECKPOINT | rollout={rollout}, Grid={rollout*14}, Vision={updates:,}', flush=True)
    return rollout


def probe(workers, original_rollout, train_seed, renderer):
    """Read-only 4-env benchmark. Uses the same next seeds as a resumed run."""
    from config import Config
    from grid_cells.rl_common import make_dmlab_env
    import collect_rh_1350_v3 as rh1350
    import collect_vision_rh_1000_v2 as collector
    from train_grid_vision_1350_online_A import ACTION_REPEAT
    cfg = Config().rl.env
    cfg.level = rh1350.LEVEL_NAME
    cfg.goal_visible = False
    # Original trainer prepares level copy; check it instead of writing.
    rh1350.ensure_level(Path('levels').resolve(), write=False)
    level = Path('levels').resolve() / f'{rh1350.LEVEL_NAME}.lua'
    if not level.exists():
        raise FileNotFoundError(f'Required level is missing: {level}')
    print('PROBE | calibrating original controls in temporary parent DMLab env...', flush=True)
    env = make_dmlab_env(cfg, renderer=renderer)
    try:
        gain, speed = collector.calibrate_controls(
            env, cfg, train_seed+90_000_000, ACTION_REPEAT)
    finally:
        env.close()
    ids = list(range(original_rollout * 10, original_rollout * 10 + workers))
    print(f'PROBE | generating {workers} NEW episodes in parallel, seeds: '
          f'{[train_seed+x for x in ids]}', flush=True)
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context('spawn'),
            initializer=_worker_init, initargs=(cfg, renderer)) as pool:
        fs = [pool.submit(_worker_episode, i, train_seed, gain, speed) for i in ids]
        results = [f.result() for f in fs]
    elapsed = time.monotonic() - started
    assert [r[0] for r in results] == ids
    assert len({r[3] for r in results}) <= workers
    for eid, arr, teleports, pid in results:
        assert arr['images'].shape == (1350,3,64,64)
        assert arr['ego_vel'].shape == (1350,3)
        print(f'  episode_id={eid}, pid={pid}, RGB={arr["images"].shape}, teleports={teleports}', flush=True)
    print(f'PROBE OK | {len(results)} episodes | workers_used={len({r[3] for r in results})} '
          f'| elapsed={elapsed:.1f}s (includes worker startup)', flush=True)
    print('PROBE | No checkpoints, images.bin or training results modified.', flush=True)


def assert_original_not_running():
    """Protect the same checkpoint directory from two concurrent trainers."""
    if not Path('/proc').exists():
        return
    for item in Path('/proc').iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            parts = (item / 'cmdline').read_bytes().split(b'\0')
        except (OSError, PermissionError):
            continue
        # Look at the executable script tokens, not command text in a shell.
        if any(part.endswith(b'/train_grid_vision_1350_online_B_sequential.py') or
               part == b'train_grid_vision_1350_online_B_sequential.py'
               for part in parts):
            raise RuntimeError(f'Original B trainer still running (PID {item.name}). '
                               'Stop it before starting 4-process collector.')


def main():
    q = argparse.ArgumentParser(add_help=False)
    q.add_argument('--workers', type=int, default=4)
    q.add_argument('--probe-only', action='store_true')
    q.add_argument('--help-parallel', action='store_true')
    opts, orig_args = q.parse_known_args()
    if opts.help_parallel:
        print(__doc__)
        return
    if opts.workers != 4:
        raise ValueError('This experiment requires --workers 4 (fixed 4-environment run)')
    if not opts.probe_only and '--resume' not in orig_args:
        raise ValueError('Must pass --resume (never restart Vision/Grid from scratch)')
    out_dir = Path(flag_value(orig_args, '--out-dir',
                              'data/checkpoints/grid1350_online_B_seq_run1'))
    assert_original_not_running()
    rollout = check_frozen_checkpoint(out_dir)
    train_seed = int(flag_value(orig_args, '--train-seed', 75_000_000))
    renderer = flag_value(orig_args, '--renderer', 'hardware')
    if opts.probe_only:
        probe(opts.workers, rollout, train_seed, renderer)
        return

    # CRITICAL: Import original source unchanged, so its SHA256 fingerprint
    # validation sees exactly the same files and hyperparameters as before.
    import train_grid_vision_1350_online_A as online_a
    import train_grid_vision_1350_online_B_sequential as trainer
    original_generate = online_a.generate_episodes
    parallel = ParallelEpisodes(opts.workers, renderer, train_seed, original_generate)
    online_a.generate_episodes = parallel
    # Keep original command line, without wrapper-only flags.
    old_argv = sys.argv
    sys.argv = ['train_grid_vision_1350_online_B_sequential.py', *orig_args]
    try:
        print('PARALLEL RESUME | using original checkpoint fingerprint and trainer unchanged', flush=True)
        trainer.main()
    finally:
        sys.argv = old_argv
        online_a.generate_episodes = original_generate
        parallel.close()


if __name__ == '__main__':
    main()
