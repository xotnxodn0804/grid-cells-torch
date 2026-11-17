#!/usr/bin/env python3
"""Draw first-search path and subsequent return paths in DMLab square arena."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_rows(file):
    with file.open(newline='') as f:
        return list(csv.DictReader(f))


def draw(ax, rows, ep, nhits):
    x = np.asarray([float(r['x_m']) for r in rows])
    y = np.asarray([float(r['y_m']) for r in rows])
    gx, gy = float(rows[0]['goal_x_m_debug']), float(rows[0]['goal_y_m_debug'])
    hit_idx = [i for i, r in enumerate(rows) if int(r['goal_hit'])]
    # Each goal hit causes an instant teleport AFTER the logged pre-action position.
    # Break the line between that position and the next row.
    starts = [0] + [i + 1 for i in hit_idx]
    ends = hit_idx + [len(rows) - 1]
    for j, (start, end) in enumerate(zip(starts, ends)):
        if start >= len(rows) or end < start: continue
        ax.plot(x[start:end+1], y[start:end+1],
                color='#232323' if j == 0 else '#4386A7',
                linewidth=1.35 if j == 0 else 1.05,
                alpha=.9 if j == 0 else .8,
                label='First search' if j == 0 else ('After teleport' if j == 1 else None))
        if j > 0:
            ax.scatter(x[start], y[start], s=16, facecolor='white',
                       edgecolor='#4386A7', linewidth=.8, zorder=3)
    ax.scatter([x[0]], [y[0]], c='#232323', s=28, marker='s',
               label='First start', zorder=4)
    ax.scatter([gx], [gy], c='#C44848', s=100, marker='*',
               label='Goal', zorder=5)
    ax.set(xlim=(-1.3,1.3), ylim=(-1.3,1.3),
           xlabel='x (m)', ylabel='y (m)', title=f'Episode {ep} · {nhits} hits')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.12)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--eval-dir', type=Path, required=True)
    p.add_argument('--episode', type=int, default=None,
                   help='Specific episode number (default: highest-hit episodes)')
    p.add_argument('--panels', type=int, default=4)
    p.add_argument('--out', type=Path, default=None)
    args=p.parse_args()
    rows=read_rows(args.eval_dir/'episodes.csv')
    selectable=[]
    for r in rows:
        ep=int(r['episode'])
        file=args.eval_dir/f'trajectory_ep{ep:03d}.csv'
        if file.is_file(): selectable.append((ep, int(r['goals']), file))
    if args.episode is not None:
        selectable=[t for t in selectable if t[0]==args.episode]
    else:
        selectable.sort(key=lambda t: (-t[1], t[0]))
        selectable=selectable[:max(1,args.panels)]
    if not selectable:
        p.error('No matching trajectory CSV. Reevaluate with --save-trajectories 100.')
    n=len(selectable)
    cols=min(n,2)
    rows=(n+cols-1)//cols
    fig, axes=plt.subplots(rows,cols,figsize=(5*cols,5*rows),squeeze=False)
    for ax,(ep,nhits,file) in zip(axes.flat,selectable):
        draw(ax,read_rows(file),ep,nhits)
    for ax in list(axes.flat)[n:]: ax.axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,.99))
    fig.suptitle('Oracle place-cell agent · trajectories (descriptive examples)',y=.995)
    fig.tight_layout(rect=[0,.0,1,.94])
    dest=args.out or (args.eval_dir/'fig2h_trajectories.png')
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest,dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('Saved:', dest)
    print('Goal position shown in plots is debug ground truth; not a policy input.')


if __name__=='__main__':
    main()
