#!/usr/bin/env python3
"""Plot saved SACs from the 1350-step A/B read-only Gridness evaluation.

Run from ~/grid-cells-torch:
  python plot_sac_1350_A.py --npz results/grid1350_A_300K/ratemaps_and_scores.npz

Produces top 32 rate-map/SAC pairs and an 8-page PDF of all 512 SACs.
This script only reads the NPZ: no model loading, optimizer or recomputation.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", type=Path,
                    default=Path("results/grid1350_A_300K/ratemaps_and_scores.npz"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/grid1350_A_300K"))
    args = ap.parse_args()
    if not args.npz.is_file():
        ap.error(f"결과 파일을 찾을 수 없습니다: {args.npz}")
    with np.load(args.npz, allow_pickle=False) as data:
        needed = {"ratemaps", "sacs", "scores_mean"}
        if not needed.issubset(data.files):
            ap.error(f"NPZ에 필요한 항목이 없습니다: {sorted(needed-set(data.files))}; "
                     f"있는 항목: {data.files}")
        maps = data["ratemaps"]
        sacs = data["sacs"]
        scores = data["scores_mean"]
        scores_mm = data["scores_minmax"] if "scores_minmax" in data.files else None
    if maps.shape != (512, 20, 20) or sacs.shape != (512, 39, 39) or scores.shape != (512,):
        ap.error(f"예상과 다른 배열 형태: rate maps={maps.shape}, sacs={sacs.shape}, scores={scores.shape}")
    ranked = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def plot_pair(ax_rate, ax_sac, u):
        ax_rate.imshow(maps[u].T, origin="lower", cmap="viridis", interpolation="nearest")
        ax_sac.imshow(sacs[u].T, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                      interpolation="nearest")
        ax_rate.set_title(f"Unit {u} | G={scores[u]:+.3f}", fontsize=9)
        subtitle = "SAC" if scores_mm is None else f"SAC | minmax={scores_mm[u]:+.3f}"
        ax_sac.set_title(subtitle, fontsize=9)
        for ax in (ax_rate, ax_sac):
            ax.set_xticks([]); ax.set_yticks([])
        for spine in ax_sac.spines.values():
            spine.set_color("#666666"); spine.set_linewidth(.4)

    # Main figure: top 32 units, each with rate map at left and SAC at right.
    fig, axes = plt.subplots(8, 8, figsize=(18, 23), constrained_layout=True)
    for i, u in enumerate(ranked[:32]):
        r, c = divmod(i, 4)
        plot_pair(axes[r, 2*c], axes[r, 2*c+1], int(u))
    fig.suptitle("A / Offline | Grid 300,006 updates | Top 32 units by MEAN Gridness\n"
                 "Rate map (left)  |  Spatial autocorrelogram, SAC (right)  |  100 held-out episodes",
                 fontsize=15)
    png = args.out_dir / "top32_ratemaps_SAC.png"
    fig.savefig(png, dpi=165, facecolor="white")
    plt.close(fig)
    print(f"SAVED | {png}", flush=True)

    pdf_path = args.out_dir / "top32_ratemaps_SAC.pdf"
    with PdfPages(pdf_path) as pdf:
        for start in range(0, 32, 16):
            fig, axes = plt.subplots(4, 8, figsize=(18, 11), constrained_layout=True)
            for i, u in enumerate(ranked[start:start+16]):
                r, c = divmod(i, 4)
                plot_pair(axes[r, 2*c], axes[r, 2*c+1], int(u))
            fig.suptitle(f"A / Offline | Grid 300K | Units ranked {start+1}–{start+16}\n"
                         "Left: rate map | Right: SAC (fixed correlation scale -1 to +1)",
                         fontsize=14)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"SAVED | {pdf_path}", flush=True)

    all_path = args.out_dir / "all512_SAC.pdf"
    with PdfPages(all_path) as pdf:
        for start in range(0, 512, 64):
            fig, axs = plt.subplots(8, 8, figsize=(14, 15), constrained_layout=True)
            for ax, u in zip(axs.flat, ranked[start:start+64]):
                ax.imshow(sacs[u].T, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                          interpolation="nearest")
                ax.set_title(f"#{u}  G={scores[u]:+.2f}", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"A / Offline | All 512 SACs | Mean-gridness ranks {start+1}–{start+64}\n"
                         "Red = positive correlation, blue = negative correlation", fontsize=14)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"SAVED | {all_path}", flush=True)
    print(f"TOP UNIT | #{ranked[0]} G={scores[ranked[0]]:.4f}", flush=True)
    print("READ-ONLY COMPLETE | No training/model files were changed", flush=True)

if __name__ == "__main__":
    main()
