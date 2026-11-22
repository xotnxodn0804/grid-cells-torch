# Joint Navigation: matched-path Gridness analysis (1M vs 21.25M)

**Date:** 2026-11-22  
**Repository:** `xotnxodn0804/grid-cells-torch`  
**Status:** Descriptive analysis complete; visual inspection, field-shuffling significance and causal ablations pending.  
**Source:** `data/evaluations/joint_nav_gridness_1m_21m_100ep/summary.json`, produced by `eval_joint_nav_gridness_1m_21m.py` (not a reconstructed or simulated result).

## Research question

Does continued joint online training of Vision, Grid and A2C Actor-Critic increase hexagonal grid-like periodicity in the 512-dimensional Grid bottleneck? Does gridness covary with the separately evaluated navigation improvement?

## Compared checkpoints

| Model | Actions | Grid updates | Actor updates | Checkpoint (local; Git-ignored) |
|---|---:|---:|---:|---|
| 1M | 1,000,000 | 9,991 | 9,990 | `data/checkpoints/joint_nav_pilot_1m_backup/latest.pt` |
| 21M | 21,247,970 | 212,470 | 212,469 | `data/checkpoints/joint_nav_pilot_21m_backup/latest.pt` |

The code is the same joint navigation architecture at different online training stages. The two Grid **and** Vision networks have different learned weights; this is not a Grid-only controlled intervention.

## Analysis protocol

- One **common 21M Actor-policy trajectory** was generated and both model-specific Grid/Vision pairs received the same RGB observations, noisy motion and visual-cue flags. Thus the rate-map comparison uses matched visited positions, though the two models have different feature encoders.
- 100 DeepMind Lab episodes, 135,000 positional samples, goal hits during collection: 235.
- Arena: 2.5 m × 2.5 m. Binning: 20 × 20 (400 bins), all 400 occupied **over the entire collection**; mean distinct bins per episode: 160.38. Per-bin sample sufficiency is not guaranteed by complete aggregate coverage.
- Observed visual cue fraction: 0.05228148148 (~5.23%).
- Ratemaps: occupancy-normalized, **unsmoothed**, signed activations of 512 *linear bottleneck* units (actual Grid code passed to the Actor, not place-cell predictions). SAC shape: 39 × 39; score maximized across 10 DeepMind-style annuli.
- Mean formula: `(corr60 + corr120)/2 - (corr30 + corr90 + corr150)/3`.
- Min-Max formula: `min(corr60, corr120) - max(corr30, corr90, corr150)`.
- Threshold **> 0.37** is a historical *descriptive* cutoff, **not a shuffled null threshold or a p-value**. Both methods are shown to avoid conflating their scores.
- `split_half_r` is the spatial Pearson correlation between odd/even-episode rate maps where bins overlap; it gauges repeatability of the spatial pattern, **not** hexagonal periodicity by itself.

## Results

| Metric | 1M | 21M |
|---|---:|---:|
| Valid mean gridness scores | 512 | 512 |
| Mean gridness > 0.37 | **4/512 (0.78%)** | **1/512 (0.20%)** |
| Min-Max gridness > 0.37 | **1/512 (0.20%)** | **0/512 (0%)** |
| Highest mean score | **0.75613** (unit 11) | **0.38532** (unit 356) |
| Highest min-max score among top-mean units | **0.72791** (unit 11) | **0.28206** (unit 356) |
| Split-half r of highest-mean unit | **0.84716** | **0.60727** |

### Top units by mean score

- **1M unit 11:** mean = 0.756130, min-max = 0.727912, split-half r = 0.847156. Primary candidate for visual inspection of the rate map and SAC.
- 1M unit 302: mean = 0.430557, min-max = 0.352684, split-half r = 0.737617.
- 1M unit 463: mean = 0.396723, min-max = 0.102388, split-half r = 0.763273.
- 1M unit 360: mean = 0.371720, min-max = 0.360232, split-half r = 0.848693.
- **21M unit 356:** mean = 0.385316, min-max = 0.282064, split-half r = 0.607270. Only unit above 0.37 by the mean formula.
- 21M unit 123: mean = 0.299786, min-max = 0.221267, split-half r = 0.726741.

These are **candidate** grid-like units under a fixed descriptive threshold; do not report the counts as statistically significant grid cells.

## Relationship to independent navigation evaluation

The earlier read-only 100-episode comparison (`eval_joint_nav_1m_vs_21m.py`) reported:

| Metric | 1M | 21M |
|---|---:|---:|
| Mean goals/episode | 0.96 | **2.35** |
| Episodes with at least one goal | 51/100 | 54/100 |
| Total revisits after first hits | 45 | **181** |
| Mean first-hit steps, successful episodes | 613.22 | 305.17 |
| Mean revisit interval, pooled successful revisits | 297.98 | 204.02 |
| Mean actor action entropy | 1.52628 | 0.83944 |

**Observation:** Navigation rewards and revisits increased even as the number of high-gridness bottleneck units fell in this descriptive assay. This does **not** establish that Grid Network is unnecessary, that spatial information has been lost, or that the Actor used remembered goal position. The 21M policy had substantial action concentration (~68.1% action 5), allowing an alternate strategy explanation. These navigation and Gridness outcomes are separate measurements, not a controlled causal test.

## Limitations / needed controls

1. **No field shuffling / multiple-comparison null:** threshold 0.37 is descriptive only; ring maximization can boost scores. Test shuffled positional alignment with an appropriate temporal/field-preserving null and per-unit correction.
2. **Inspect plots:** review unit 11 (1M) and unit 356 (21M), top32 rate maps, SACs, and occupancy map before claiming meaningful hexagonal fields.
3. **Matched path, not isolated Grid perturbation:** Grid **and Vision** weights differ; activity may depend on input representation, activation scaling, and learned policy/state occupancy.
4. **Coverage:** all 400 bins were sampled across runs, but only ~160.38 were visited per episode, and individual bin occupancy and teleport sampling may affect rate maps.
5. **Single seed / stage comparison:** no independent training seeds or temporal replication of gridness scores.
6. **No navigation-memory proof:** compare Goal Memory ON/OFF on paired goal and teleport starting locations. Primary metrics should be return success at fixed horizons, start-to-goal straight-line distance, return actions per meter (careful near zero distance), path-length efficiency, and failure censoring at episode end.

## Reproduce

```bash
cd ~/grid-cells-torch
source .venv-dmlab/bin/activate
python -u eval_joint_nav_gridness_1m_21m.py \
  --mode eval --device cuda --episodes 100 \
  --out-dir data/evaluations/joint_nav_gridness_1m_21m_100ep
```

**Caution:** The evaluator refuses to overwrite a nonempty output folder. Reproduction requires the local checkpoints, DeepMind Lab and training/evaluation code; those `.pt` files are intentionally excluded from Git. The exported figures, CSV and JSON under `results/joint_nav_gridness_20261122/` are a convenient snapshot, not replacements for the local checkpoints. The raw `*_ratemaps_scores.npz` files may be stored locally if size or repository limits make them impractical to commit.

## GitHub artifact index

- Source: `eval_joint_nav_gridness_1m_21m.py` (depends on `train_joint_nav_online.py`, `eval_joint_nav_1m_vs_21m.py` and `grid_cells/`).
- Navigation evaluator: `eval_joint_nav_1m_vs_21m.py` and its 100-episode results under `results/joint_nav_eval_20261122/{episodes.csv,summary.json}`.
- Results: `results/joint_nav_gridness_20261122/summary.json`, per-model `*_gridness_sorted.csv`, plots, and 512-ratemap PDFs.
- Note: `docs/experiments/2026-11-22-joint-gridness.md` (this document).
- Checkpoints: local-only paths above; explicitly **not** committed.

## Next decision

Preserve this 21.25M result before further training. After visual and null-distribution checks, proceed with the planned 21.25M → 50M continuation, saving distinct intermediate model snapshots. Evaluate goal-directed return separately from target discovery.
