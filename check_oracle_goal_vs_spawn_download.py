#!/usr/bin/env python3
"""Read-only audit: is the agent at the debug GOAL when goal_hit==1,
or closer to the most recent post-teleport START?

Run from the grid-cells-torch project root:
  python check_oracle_goal_vs_spawn.py

No retraining or DMLab evaluation. Uses existing CSVs only.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import median


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def xy(row, x='x_m', y='y_m'):
    return float(row[x]), float(row[y])


def summarize_mode(mode, folder):
    paths = sorted(folder.glob('trajectory_ep*.csv'))
    if not paths:
        raise FileNotFoundError(f'No trajectory_ep*.csv in {folder}')
    events, stability = [], []
    for path in paths:
        ep = int(path.stem.split('ep')[-1])
        rows = read_csv(path)
        if not rows:
            continue
        required = {'frame','x_m','y_m','goal_x_m_debug','goal_y_m_debug','goal_hit'}
        missing = required.difference(rows[0])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        goals = {tuple(round(v, 5) for v in xy(r, 'goal_x_m_debug', 'goal_y_m_debug')) for r in rows}
        stability.append((ep, len(goals)))
        hit_indices = [i for i,r in enumerate(rows) if float(r['goal_hit']) > 0.5]
        last_hit = -1
        for hit_num, hit_i in enumerate(hit_indices, start=1):
            # First sampled position after the previous teleport is an approximate re-start.
            start_i = last_hit + 1
            hit_row, start_row = rows[hit_i], rows[start_i]
            hit_xy = xy(hit_row)
            start_xy = xy(start_row)
            goal_xy = xy(hit_row, 'goal_x_m_debug', 'goal_y_m_debug')
            d_goal = distance(hit_xy, goal_xy)
            d_start = distance(hit_xy, start_xy)
            d_closest_on_segment = min(distance(xy(r), goal_xy) for r in rows[start_i:hit_i+1])
            # Distance to previous position; 4 frames are typically one action.
            d_prev = distance(hit_xy, xy(rows[hit_i - 1])) if hit_i else float('nan')
            d_post = distance(hit_xy, xy(rows[hit_i + 1])) if hit_i + 1 < len(rows) else float('nan')
            events.append({
                'mode':mode, 'episode':ep, 'hit_number':hit_num,
                'frame':int(float(hit_row['frame'])),
                'start_x_m':start_xy[0], 'start_y_m':start_xy[1],
                'hit_x_m':hit_xy[0], 'hit_y_m':hit_xy[1],
                'goal_x_m_debug':goal_xy[0], 'goal_y_m_debug':goal_xy[1],
                'distance_hit_to_debug_goal_m':round(d_goal,5),
                'distance_hit_to_teleport_start_m':round(d_start,5),
                'closest_segment_position_to_debug_goal_m':round(d_closest_on_segment,5),
                'before_hit_jump_m':round(d_prev,5),
                'after_hit_jump_m':round(d_post,5),
                'closer_to_start_than_debug_goal':int(d_start < d_goal),
                'debug_goal_positions_within_episode':len(goals),
            })
            last_hit = hit_i
    return events, stability


def describe(label, events, stability):
    print(f'\n====== {label} ======')
    print('Episodes with trajectories:', len(stability))
    print('Episodes with changed debug goal coordinates:', sum(n > 1 for _,n in stability))
    for kind, subset in [('All hits', events), ('Revisits only', [e for e in events if e['hit_number'] >= 2])]:
        if not subset:
            continue
        ds = [e['distance_hit_to_debug_goal_m'] for e in subset]
        ss = [e['distance_hit_to_teleport_start_m'] for e in subset]
        close_goal = sum(d <= .25 for d in ds)
        far_goal = sum(d > .50 for d in ds)
        closer_spawn = sum(s < d for s,d in zip(ss, ds))
        print(f'{kind}: n={len(subset)}, median distance to DEBUG goal={median(ds):.3f}m, '
              f'median distance to last spawn={median(ss):.3f}m')
        print(f'  Within 0.25m of debug goal: {close_goal}/{len(subset)} ({100*close_goal/len(subset):.1f}%)')
        print(f'  Farther than 0.50m from debug goal: {far_goal}/{len(subset)} ({100*far_goal/len(subset):.1f}%)')
        print(f'  Closer to spawn than debug goal: {closer_spawn}/{len(subset)} ({100*closer_spawn/len(subset):.1f}%)')
    print('\nFirst revisits in the four illustrated episodes:')
    for ep in (5,25,78,74):
        match = next((e for e in events if e['episode'] == ep and e['hit_number'] == 2),None)
        if match:
            print(f"  ep {ep:03d}: reward frame={match['frame']} | "
                  f"to debug goal={match['distance_hit_to_debug_goal_m']:.3f}m | "
                  f"to teleport start={match['distance_hit_to_teleport_start_m']:.3f}m | "
                  f"closest on segment to goal={match['closest_segment_position_to_debug_goal_m']:.3f}m")
        else:
            print(f'  ep {ep:03d}: no first revisit recorded')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--normal-dir', type=Path, default=Path('results/oracle_500m_normal'))
    p.add_argument('--zero-dir', type=Path, default=Path('results/oracle_500m_zero'))
    p.add_argument('--out-dir', type=Path, default=Path('results/oracle_500m_compare/goal_audit'))
    args = p.parse_args()
    all_events = []
    for mode, path in [('Normal',args.normal_dir), ('Zero',args.zero_dir)]:
        events, stability = summarize_mode(mode, path)
        describe(mode,events,stability)
        all_events.extend(events)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / 'goal_vs_spawn_events.csv'
    with dest.open('w', encoding='utf-8', newline='') as f:
        if all_events:
            writer=csv.DictWriter(f, fieldnames=list(all_events[0]))
            writer.writeheader()
            writer.writerows(all_events)
    print('\nSaved:', dest)
    print('INTERPRETATION: Distances use pre-step recorded positions and the DEBUG goal marker,')
    print('not the verified collision trigger position. Distances far from the marker indicate')
    print('a mismatch to investigate, NOT automatically that the agent was rewarded at spawn.')
    print('The correct next check is the live Lua goal/reward coordinates and Python position transform.')


if __name__ == '__main__':
    main()
