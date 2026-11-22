#!/usr/bin/env python3
"""Disk-backed chronological ring replay for one 64x64 DMLab navigation actor.

Each absolute index k stores the PRE-action observation at step k plus the
resulting action/reward. Storage is a ring; grid sampling returns contiguous
100-frame windows, and explicit reset flags mark teleports/new episodes.
Never feed pos/hd to the ACTOR: those are privileged auxiliary labels only.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


FIELDS = {
    'image': ('u1', (3, 64, 64)),
    'pos': ('f4', (2,)),
    'hd': ('f4', ()),
    'motion': ('f4', (4,)),
    'reset': ('u1', ()),
    'vision_seen': ('u1', ()),
    'episode': ('i4', ()),
    'action': ('i1', ()),
    'reward': ('f4', ()),
    'stamp': ('i8', ()),
}


class DiskRingReplay:
    def __init__(self, root: Path, capacity: int, *, create: bool):
        if capacity < 1_000:
            raise ValueError('Replay capacity must be at least 1,000 frames')
        self.root = Path(root)
        self.capacity = int(capacity)
        self.total = 0
        self.root.mkdir(parents=True, exist_ok=True)
        self.arrays = {}
        for field, (dtype, tail) in FIELDS.items():
            path = self.root / (field + '.bin')
            if create and path.exists():
                raise FileExistsError(f'Refusing to overwrite replay: {path}')
            if not create and not path.is_file():
                raise FileNotFoundError(f'Missing replay field: {path}')
            self.arrays[field] = np.memmap(
                path, dtype=np.dtype(dtype), mode='w+' if create else 'r+',
                shape=(capacity,) + tail)

    @property
    def size(self):
        return min(self.total, self.capacity)

    def load_count(self, total: int):
        if total < 0:
            raise ValueError('Invalid saved replay count')
        self.total = int(total)
        self._verify_recent()

    def _verify_recent(self):
        if self.total == 0:
            return
        # Check contiguous available indexes in the saved snapshot. This
        # detects stale metadata / partially overwritten ring after a crash.
        start = max(0, self.total - self.capacity)
        probe = np.arange(start, self.total, dtype=np.int64)
        if not np.array_equal(self.arrays['stamp'][probe % self.capacity], probe):
            raise RuntimeError('Replay/checkpoint mismatch: ring stamp gap. Preserve files and restart with a fresh out-dir.')

    def add(self, *, image, pos, hd, motion, reset, vision_seen, episode, action, reward):
        k = self.total
        slot = k % self.capacity
        record = {
            'image': np.asarray(image, dtype=np.uint8),
            'pos': np.asarray(pos, dtype=np.float32),
            'hd': np.float32(hd),
            'motion': np.asarray(motion, dtype=np.float32),
            'reset': np.uint8(reset),
            'vision_seen': np.uint8(vision_seen),
            'episode': np.int32(episode),
            'action': np.int8(action),
            'reward': np.float32(reward),
            'stamp': np.int64(k),
        }
        for field, value in record.items():
            self.arrays[field][slot] = value
        self.total += 1

    def read(self, absolute_indices):
        idx = np.asarray(absolute_indices, dtype=np.int64)
        if idx.size == 0:
            raise ValueError('Cannot read empty replay request')
        if idx.min() < max(0, self.total - self.capacity) or idx.max() >= self.total:
            raise IndexError('Index has expired from the replay ring')
        slots = idx % self.capacity
        stamp = self.arrays['stamp'][slots]
        if not np.array_equal(stamp, idx):
            raise RuntimeError('Replay corruption: requested timestamps are not present')
        return {name: np.asarray(arr[slots]).copy() for name, arr in self.arrays.items()
                if name != 'stamp'}

    def sample_vision(self, batch: int, rng: np.random.Generator):
        if self.size < batch:
            raise RuntimeError(f'Need {batch} experiences for Vision, have {self.size}')
        first = self.total - self.size
        ids = rng.choice(np.arange(first, self.total), size=batch, replace=False)
        return self.read(ids)

    def sample_grid(self, paths: int, length: int, rng: np.random.Generator):
        """Return non-overlapping, ordered-internally length-step paths.

        Start points are non-overlapping by choosing an alignment offset and
        selecting distinct windows on that aligned 100-step lattice. Windows
        MAY contain a teleport or episode boundary, marked by reset[t].
        """
        needed = paths * length
        if self.size < needed:
            raise RuntimeError(f'Need {needed} experiences for Grid, have {self.size}')
        first = self.total - self.size
        offset = int(rng.integers(0, self.size - needed + 1)) if self.size == needed else int(rng.integers(0, length))
        if offset + needed > self.size:
            offset = 0
        candidates = np.arange(first + offset, self.total - length + 1, length, dtype=np.int64)
        if len(candidates) < paths:
            # The random offset can leave fewer than `paths` windows near a
            # just-filled buffer. Fall back to first possible alignment.
            candidates = np.arange(first, self.total - length + 1, length, dtype=np.int64)
        starts = rng.choice(candidates, size=paths, replace=False)
        ids = starts[:, None] + np.arange(length, dtype=np.int64)[None, :]
        batch = self.read(ids)
        return batch

    def flush(self):
        # Call BEFORE atomic checkpoint save; checkpoints reference this count.
        for a in self.arrays.values():
            a.flush()


def self_test(tmp: Path):
    tmp = Path(tmp)
    if tmp.exists():
        raise FileExistsError(f'Self-test requires a new temporary path: {tmp}')
    rng = np.random.default_rng(121)
    b = DiskRingReplay(tmp, 1_000, create=True)
    for k in range(1_100):
        image = np.zeros((3, 64, 64), np.uint8)
        image[:, 0, 0] = k % 256
        b.add(image=image, pos=(float(k), -1.0), hd=.1, motion=(0., 0., 0., 1.),
              reset=(k % 250 == 0), vision_seen=(k % 20 == 0), episode=k // 250,
              action=k % 6, reward=float(k == 500))
    assert b.total == 1100 and b.size == 1000
    v = b.sample_vision(32, rng)
    assert v['image'].shape == (32, 3, 64, 64)
    g = b.sample_grid(10, 100, rng)
    assert g['image'].shape == (10, 100, 3, 64, 64)
    assert np.all(np.diff(g['pos'][:, :, 0], axis=1) == 1)
    b.flush()
    del b
    restored = DiskRingReplay(tmp, 1_000, create=False)
    restored.load_count(1100)
    assert np.array_equal(restored.read([100, 1099])['pos'][:, 0], [100, 1099])
    print('REPLAY SELF-TEST PASS: ring overwrite, random Vision, 10x100 Grid, restart', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--test-dir', type=Path, default=Path('/tmp/joint-nav-replay-selftest'))
    a = p.parse_args()
    if a.self_test:
        self_test(a.test_dir)
