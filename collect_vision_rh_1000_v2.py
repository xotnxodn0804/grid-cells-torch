#!/usr/bin/env python3
"""Collect 1000 DeepMind Lab episodes with Raudies-Hasselmo-inspired action control.

This is an IMAGE-ONLY VISION experiment. It stores RGB, real position and heading,
as well as observed speed/turn for diagnostics and a later grid-network study.

Target motor statistics follow grid_cells/data_generation.py, but DeepMind Lab
uses integer mouse/forward commands and Quake physics. The resulting *actual*
trajectories are NOT distributionally identical to the original continuous 2-D
rodent simulator. Inspect speed/turn diagnostics rather than assuming equality.

Designed to produce the *.npy and *_complete.json fields understood by
vision_300k.py train --data-dir <folder>. No modification to prior checkpoints.
The ground-truth position/heading are DMLab observations AFTER each action.
Movement is approximate (on/off forward and integer look), never claimed exact RH.
Each episode is separated by DMLab reset, train/val have distinct seeds.

From the Ubuntu clone root, in .venv-dmlab:
  python collect_vision_rh_1000_v2.py --episodes 1000 --val-episodes 100
  python collect_vision_rh_1000_v2.py --episodes 1000 --val-episodes 100 --resume
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FPS = 60.0
COLLECTOR_VERSION = 2


def wrap_angle(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def wall_zone(ang: float, pos: np.ndarray, min_gap: float, max_gap: float) -> bool:
    """Copied from the existing supervised grid data-generation motion model."""
    if (0 <= ang <= np.pi / 2) and np.any(pos > max_gap):
        return True
    elif (np.pi / 2 <= ang <= np.pi) and (pos[0] < min_gap or pos[1] > max_gap):
        return True
    elif (-np.pi <= ang <= -np.pi / 2) and np.any(pos < min_gap):
        return True
    elif (-np.pi / 2 <= ang <= 0) and (pos[0] > max_gap or pos[1] < min_gap):
        return True
    return False


def wall_redirect(ang: float, pos: np.ndarray, min_gap: float, max_gap: float) -> float:
    half_pi = np.pi / 2
    if 0 <= ang <= np.pi / 2:
        if pos[1] > max_gap: return -half_pi
        elif pos[0] > max_gap: return half_pi
    elif np.pi / 2 <= ang <= np.pi:
        if pos[1] > max_gap: return half_pi
        elif pos[0] < min_gap: return -half_pi
    elif -np.pi <= ang <= -np.pi / 2:
        if pos[1] < min_gap: return -half_pi
        elif pos[0] < min_gap: return half_pi
    else:
        if pos[1] < min_gap: return half_pi
        elif pos[0] > max_gap: return -half_pi
    return 0.0


@dataclass
class RHTargetMotion:
    """Raudies-Hasselmo target increments, matching the repo's parameters.

    Fine samples at 0.02 s; final fraction of a DMLab control interval is
    integrated separately. Real game motion is measured independently.
    """
    arena_m: float = 2.5
    fine_dt: float = 0.02
    speed_scale: float = 0.13
    turn_std_rad_s: float = math.radians(330.0)
    perimeter_m: float = 0.03
    perimeter_reduction: float = 0.25
    last_target_speed: float = 0.0

    def sample(self, rng: np.random.Generator, xy: np.ndarray,
               heading: float, interval_s: float) -> tuple[float, float]:
        # The original 2-D generator uses coordinates [0,L]^2.
        pos = np.asarray(xy, dtype=np.float64).copy() + self.arena_m / 2.0
        theta = heading
        total_distance, total_turn = 0.0, 0.0
        remain = interval_s
        while remain > 1e-9:
            dt = min(self.fine_dt, remain)
            boundary = wall_zone(theta, pos, self.perimeter_m,
                                 self.arena_m - self.perimeter_m)
            if boundary:
                speed = self.last_target_speed * (1.0 - self.perimeter_reduction)
                turn = (wall_redirect(theta, pos, self.perimeter_m,
                                      self.arena_m - self.perimeter_m)
                        + rng.normal(0.0, self.turn_std_rad_s) * dt)
            else:
                speed = rng.rayleigh(self.speed_scale)
                turn = rng.normal(0.0, self.turn_std_rad_s) * dt
            pos += np.array([math.cos(theta), math.sin(theta)]) * speed * dt
            theta = wrap_angle(theta + turn)
            total_distance += speed * dt
            total_turn += turn
            self.last_target_speed = speed
            remain -= dt
        return total_distance / interval_s, total_turn


def read_obs(env, cfg):
    from grid_cells.rl_common import convert_observation
    return convert_observation(env.observations(), cfg)


def action(look: int = 0, forward: bool = False) -> np.ndarray:
    # DMLab: [look_lr_pixels_per_frame, look_ud, strafe, forward, fire, jump, crouch]
    return np.array([int(np.clip(look, -512, 512)), 0, 0, int(forward),
                     0, 0, 0], dtype=np.intc)


def calibrate_controls(env, cfg, seed: int, action_repeat: int) -> tuple[float, float]:
    """Estimate radians per look pixel and m/s from the live DMLab build."""
    interval = action_repeat / FPS
    env.reset(seed=seed)
    prev = read_obs(env, cfg)
    env.step(action(look=20), num_steps=action_repeat)
    if not env.is_running():
        raise RuntimeError("DMLab episode ended during yaw calibration")
    curr = read_obs(env, cfg)
    deg_per_pixel_step = math.degrees(wrap_angle(float(curr["hd"][0] - prev["hd"][0]))) / 20.0
    if abs(deg_per_pixel_step) < 0.05:
        raise RuntimeError(f"Yaw calibration failed: {deg_per_pixel_step:.6g} deg/pixel per step")

    env.reset(seed=seed + 1)
    last = read_obs(env, cfg)
    speeds = []
    for i in range(4):
        env.step(action(forward=True), num_steps=action_repeat)
        if not env.is_running():
            raise RuntimeError("DMLab episode ended during forward calibration")
        now = read_obs(env, cfg)
        displacement = np.linalg.norm(now["pos"] - last["pos"])
        speeds.append(float(displacement / interval))
        last = now
    # Sample before a central-spawn agent is likely to reach a wall.
    # Ignore only the very first step (acceleration), then use the median.
    nominal_speed = float(np.median(speeds[1:]))
    if nominal_speed < 0.03:
        raise RuntimeError(f"Forward calibration failed: {nominal_speed:.4f} m/s")
    print(f"CALIBRATION | yaw={deg_per_pixel_step:+.3f} deg/pixel/action | "
          f"forward={nominal_speed:.3f} m/s", flush=True)
    return deg_per_pixel_step, nominal_speed


SPECS = {
    "images": (np.uint8, (3, 64, 64)),
    "pos": (np.float32, (2,)),
    "hd": (np.float32, ()),
    "speed": (np.float32, ()),       # measured displacement / elapsed time, m/s
    "omega": (np.float32, ()),       # measured heading increment / elapsed time, rad/s
    "ego_vel": (np.float32, (3,)),    # [speed, sin(dheading), cos(dheading)]
    "valid_motion": (np.uint8, ()),  # 0 if teleport/death discontinuity detected
    "target_speed": (np.float32, ()),  # RH desired, NOT actual DMLab speed
    "target_omega": (np.float32, ()),  # RH desired, NOT actual DMLab omega
    "vel_trans": (np.float32, (3,)),  # DMLab VEL.TRANS, original world units/s
    "vel_rot": (np.float32, (3,)),    # DMLab VEL.ROT, deg/s
    "episode_id": (np.int32, ()),    # per-split episode index; useful for grouped evaluation
}


def split_paths(root: Path, split: str) -> dict:
    return {key: root / f"{split}_{key}.npy" for key in SPECS}


def init_split(root: Path, split: str, n_episodes: int, steps: int, resume: bool):
    paths = split_paths(root, split)
    n = n_episodes * steps
    mm = {}
    for key, path in paths.items():
        dtype, shape = SPECS[key]
        expected = (n, *shape)
        if resume:
            if not path.exists():
                raise FileNotFoundError(f"Missing resume array {path}")
            a = np.load(path, mmap_mode="r+")
            if a.shape != expected or a.dtype != np.dtype(dtype):
                raise ValueError(f"Wrong {key} shape/dtype: {a.shape} {a.dtype}; expected {expected} {dtype}")
        else:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}. Use --resume or a new --data-dir")
            a = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=expected)
        mm[key] = a
    return mm


def collect_split(env, cfg, root: Path, split: str, episodes: int,
                  steps: int, seed: int, action_repeat: int, yaw_gain: float,
                  nominal_speed: float, resume: bool) -> None:
    manifest_file = root / f"{split}_progress.json"
    complete_file = root / f"{split}_complete.json"
    has_files = any(p.exists() for p in split_paths(root, split).values())

    if resume and complete_file.is_file():
        meta = json.loads(complete_file.read_text(encoding="utf-8"))
        if (meta.get("episodes"), meta.get("steps_per_episode"), meta.get("seed")) != (episodes, steps, seed):
            raise ValueError(f"Completed {split} dataset does not match requested settings")
        print(f"SKIP completed {split}: {episodes} episodes", flush=True)
        return

    if resume:
        if manifest_file.is_file():
            continuing = True
        elif not has_files:
            # A prior run may have finished TRAIN and stopped before creating VAL.
            # Initialize the unstarted split from scratch, even under --resume.
            continuing = False
        else:
            raise RuntimeError(f"Found {split} arrays but no progress file: {root}. "
                               "Do not guess which rows are valid; choose another folder.")
    else:
        continuing = False
        if manifest_file.exists() or complete_file.exists() or has_files:
            raise FileExistsError(f"Refusing to overwrite existing {split} dataset")

    arrays = init_split(root, split, episodes, steps, resume=continuing)
    if continuing:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        wanted = {"version": COLLECTOR_VERSION, "split": split,
                  "requested_episodes": episodes, "steps_per_episode": steps,
                  "seed": seed, "action_repeat": action_repeat}
        for key, value in wanted.items():
            if manifest.get(key) != value:
                raise ValueError(f"Cannot resume {split}: {key} changed "
                                 f"from {manifest.get(key)!r} to {value!r}")
        finished = int(manifest["completed_episodes"])
    else:
        finished = 0
    if not 0 <= finished <= episodes:
        raise ValueError("Invalid completed episode count")

    def progress(count: int) -> None:
        # The JSON update occurs only after all memmaps have flushed.
        atomic_json(manifest_file, {
            "version": COLLECTOR_VERSION, "split": split,
            "completed_episodes": count, "requested_episodes": episodes,
            "steps_per_episode": steps, "seed": seed,
            "action_repeat": action_repeat,
            "yaw_deg_per_pixel_per_step": yaw_gain,
            "forward_nominal_m_s": nominal_speed,
            "rh_speed_rayleigh_scale_m_s": 0.13,
            "rh_angular_std_deg_s": 330.0,
            "rh_fine_dt_s": 0.02,
            "note": "Target R-H motion converted to DMLab controls; labels from actual DMLab state",
        })

    if not continuing:
        progress(0)  # A crash before episode 1 is now resumable too.

    interval = action_repeat / FPS
    print(f"COLLECT {split}: from episode {finished + 1}/{episodes}, "
          f"{steps} observations/episode", flush=True)
    try:
        for episode in range(finished, episodes):
            rng = np.random.default_rng(seed + episode * 7919)
            env.reset(seed=seed + episode)
            prev = read_obs(env, cfg)
            motor = RHTargetMotion(arena_m=cfg.env_size_m)
            speeds = []
            target_speeds = []
            turns = []
            target_turns = []
            teleports = 0
            for t in range(steps):
                if not env.is_running():
                    raise RuntimeError(f"DMLab episode {episode} ended at step {t}; "
                                       "reduce --steps-per-episode, then use a fresh --data-dir")
                target_speed, target_turn = motor.sample(
                    rng, prev["pos"], float(prev["hd"][0]), interval)
                look = int(np.clip(round(math.degrees(target_turn) / yaw_gain), -512, 512))
                # Stochastic duty cycling approximates velocity at the block level.
                # Target speed is not guaranteed to equal real speed (friction, collision).
                forward = bool(rng.random() < min(1.0, target_speed / nominal_speed))
                env.step(action(look, forward), num_steps=action_repeat)
                if not env.is_running():
                    raise RuntimeError(f"DMLab ended at {t+1}/{steps} in episode {episode}; "
                                       "try --steps-per-episode 1200 in a NEW dataset")
                ob = read_obs(env, cfg)
                displacement = float(np.linalg.norm(ob["pos"] - prev["pos"]))
                dtheta = wrap_angle(float(ob["hd"][0] - prev["hd"][0]))
                valid = displacement <= 0.35  # detect a teleport/reposition discontinuity
                if valid:
                    actual_speed = displacement / interval
                    actual_omega = dtheta / interval
                else:
                    actual_speed, actual_omega, dtheta = 0.0, 0.0, 0.0
                    teleports += 1
                    motor.last_target_speed = 0.0
                j = episode * steps + t
                arrays["images"][j] = ob["image"]
                arrays["pos"][j] = ob["pos"]
                arrays["hd"][j] = float(ob["hd"][0])
                arrays["speed"][j] = actual_speed
                arrays["omega"][j] = actual_omega
                arrays["ego_vel"][j] = (actual_speed, math.sin(dtheta), math.cos(dtheta))
                arrays["valid_motion"][j] = int(valid)
                arrays["target_speed"][j] = target_speed
                arrays["target_omega"][j] = target_turn / interval
                arrays["vel_trans"][j] = ob["vel_trans"]
                arrays["vel_rot"][j] = ob["vel_rot"]
                arrays["episode_id"][j] = episode
                if valid:
                    speeds.append(actual_speed)
                    turns.append(abs(actual_omega))
                    target_speeds.append(target_speed)
                    target_turns.append(abs(target_turn / interval))
                prev = ob
            # Commit only after all arrays have finished writing the whole episode.
            for a in arrays.values():
                a.flush()
            finished = episode + 1
            progress(finished)
            if finished % 10 == 0 or finished == episodes:
                print(f"{split} {finished:,}/{episodes:,} episodes "
                      f"({finished*steps:,} images) | "
                      f"target/actual speed={np.mean(target_speeds):.3f}/{np.mean(speeds):.3f} m/s, "
                      f"target/actual |omega|={math.degrees(np.mean(target_turns)):.1f}/"
                      f"{math.degrees(np.mean(turns)):.1f} deg/s, "
                      f"teleport jumps={teleports}", flush=True)
    except KeyboardInterrupt:
        print(f"Interrupted; last COMPLETED {split} episode={finished}. "
              "Use identical command with --resume.", flush=True)
        raise
    finally:
        for a in arrays.values():
            a.flush()
    atomic_json(complete_file, {
        "split": split, "frames": episodes * steps, "episodes": episodes,
        "steps_per_episode": steps, "seed": seed,
        "env_size_m": cfg.env_size_m, "height": cfg.height, "width": cfg.width,
        "motion": "RH targets / calibrated discrete DeepMind Lab controls",
        "position_and_heading": "measured DMLab ground truth",
        "image_input_only": True,
    })
    print(f"COMPLETE {split}: {episodes} episodes", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=1000,
                   help="TOTAL train+validation episodes (default 1000)")
    p.add_argument("--val-episodes", type=int, default=100,
                   help="Validation episodes, held out by seed (default 100)")
    p.add_argument("--steps-per-episode", type=int, default=1300,
                   help="Action blocks per episode; 1300*4/60=86.7s (<90s timeout)")
    p.add_argument("--data-dir", default="data/vision_rh_1000")
    p.add_argument("--action-repeat", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--renderer", choices=("hardware", "software"), default="hardware")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not (1 < args.episodes and 0 < args.val_episodes < args.episodes):
        p.error("Require --episodes > 1 and 0 < --val-episodes < --episodes")
    if args.steps_per_episode < 1 or args.action_repeat < 1:
        p.error("Positive --steps-per-episode and --action-repeat required")
    if args.steps_per_episode * args.action_repeat >= 5400:
        p.error("Collect fewer than 5400 DMLab frames per episode; e.g. 1300 steps x 4")
    from config import Config
    from grid_cells.rl_common import make_dmlab_env
    cfg = Config()
    expected = (cfg.data.speed_rayleigh_scale, cfg.data.rot_vel_std_deg,
                cfg.data.sim_dt, cfg.data.perimeter_m,
                cfg.data.perimeter_speed_reduction)
    written = (0.13, 330.0, 0.02, 0.03, 0.25)
    if not np.allclose(expected, written):
        raise ValueError("Your Grid Network R-H motion parameters changed; "
                         "update this collector's RHTargetMotion settings before collecting")
    cfg.rl.env.level = "square_arena"
    cfg.rl.env.goal_visible = False
    cfg.rl.env.action_repeat = args.action_repeat
    if cfg.rl.env.height != 64 or cfg.rl.env.width != 64:
        raise ValueError("This collector expects 64x64 images")
    root = Path(args.data_dir).resolve()
    if not args.resume and root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Nonempty dataset folder {root}; choose a new --data-dir")
    if args.resume and not root.exists():
        raise FileNotFoundError(f"Cannot resume non-existing dataset directory {root}")
    root.mkdir(parents=True, exist_ok=True)
    train_episodes = args.episodes - args.val_episodes
    dataset_file = root / "dataset_settings.json"
    settings = {
        "collector_version": COLLECTOR_VERSION, "episodes": args.episodes,
        "val_episodes": args.val_episodes, "steps_per_episode": args.steps_per_episode,
        "action_repeat": args.action_repeat, "seed": args.seed,
        "goal_visible": False, "arena_size_m": 2.5,
    }
    if args.resume:
        if not dataset_file.exists() or json.loads(dataset_file.read_text(encoding="utf-8")) != settings:
            raise ValueError("--resume parameters differ from saved dataset settings")
    else:
        atomic_json(dataset_file, settings)
    print(f"Dataset: {root}\nEpisodes: train={train_episodes}, val={args.val_episodes}; "
          f"images={args.episodes * args.steps_per_episode:,}; "
          f"uncompressed RGB≈{args.episodes*args.steps_per_episode*3*64*64/2**30:.2f} GiB")
    print("WARNING: RH velocity/turn distributions are TARGETS; DMLab discrete actions "
          "and physics yield approximate actual paths. Recording actual motion separately.")
    env = make_dmlab_env(cfg.rl.env, renderer=args.renderer)
    try:
        yaw_gain, nominal_speed = calibrate_controls(env, cfg.rl.env,
                                                     args.seed + 90_000_000,
                                                     args.action_repeat)
        collect_split(env, cfg.rl.env, root, "train", train_episodes,
                      args.steps_per_episode, args.seed, args.action_repeat,
                      yaw_gain, nominal_speed, args.resume)
        collect_split(env, cfg.rl.env, root, "val", args.val_episodes,
                      args.steps_per_episode, args.seed + 10_000_000,
                      args.action_repeat, yaw_gain, nominal_speed, args.resume)
    finally:
        env.close()
    print("COLLECTION COMPLETE")
    print(f"Next train with: python vision_300k.py train --data-dir {root} "
          "--out-dir data/checkpoints/vision_rh_1000 --device cuda --updates 10000")


if __name__ == "__main__":
    main()
