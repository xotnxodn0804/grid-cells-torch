# Research handoff — grid-cells-torch

**Updated: 2026-11-10.** This is the entry point for future assistant sessions. The objective is to make the exact methods, outcomes, scripts, and paths recoverable in a new chat.

**GitHub repository:** https://github.com/xotnxodn0804/grid-cells-torch

## Read this first

- [Vision module — setup, training, evaluation, interpretations](docs/experiments/2026-11-10-vision.md)
- [Oracle Place A2C — 50M to 500M, ablations, coordinate audit](docs/experiments/2026-11-10-oracle-a2c.md)
- [Artifact inventory — actual tracked files, exact paths, local-only checkpoints](docs/experiments/2026-11-10-artifact-index.md)
- [Original supervised grid-cell reproduction](README.md)
- [Older plan status, not today's checkpoint](docs/plan/STATUS.md)

## Project branches / conceptual separation

1. **Supervised path integration**: 1M simulated trajectories, LSTM128, bottleneck512, 300 epochs ×1000 steps, gridness analysis. Spatial representation only; not navigation success.
2. **Standalone vision module (supervised)**: RGB 64×64 input, 256 PC +12 HD targets, 100,000 optimizer updates, batch32, validation N=10,000. Mean position error **26.3cm**, median **19.8cm**, mean heading error ~19.2°; actual-vs-predicted and spatial heatmap saved. This module's predicted PC code is NOT the current agent's oracle PC input.
3. **Oracle Place A2C navigation (reinforcement learning)**: 32 DMLab workers, current ground-truth PC+HD and a goal-memory PC+HD code presented only after first reward, LSTM256; 5,400 frames/episode =90s, action repeat4. Goal fixed *within* episode and changes *between* episodes, with teleport after each arrival.

| A2C training frames | Normal goal rewards / episode (100 eval episodes) | First hit found | Revisits |
|---|---:|---:|---:|
| 50M | 2.71 | 51/100 | 220 |
| 100M | 5.62 | 80/100 | 482 |
| 200M | 10.79 | 87/100 | 992 |
| 500M | **10.54** | **81/100** | **973** |
| 500M with goal memory zeroed | **2.90** | **81/100** | **209** |

All above evals use seed 10023, 100 episodes. Success-counts are *episodes with at least one hit*; revisit counts are subsequent hits, NOT percentages.

### Crucial RL coordinate issue

The historical level's exported `ORACLE.GOAL.XY` **Y coordinate is sign-reversed relative to the physical goal location** as inferred from 1,344 reward-event records. Plot historical trajectories with marker `(goal_x_m_debug, -goal_y_m_debug)`; **do not flip agent positions**. On the 500M event audit this correction put 100% of reward events within 25cm of the physical goal marker. The policy's goal-code input during training was NOT corrected and reflects the old convention. Preserve this legacy checkpoint; do **not** silently change goal coding while resuming its weights. It might have learned an implicit coordinate mapping, which is a hypothesis, not proof.

### Major interpretation constraints

- Goal memory removal changes reward performance, but does not prove the internal mechanism or optimal navigation.
- 200M→500M performance roughly plateaued; this alone is not proof of overfitting.
- Vision errors grow near walls; decoder weighted-mean bias contributes even with ideal PC labels. Distal-cue occlusion is *plausible but untested*.
- Figure examples are selectively chosen episodes, not a random representative sample.

## What is actually in GitHub vs local?

Files under `results/vision_20261110/`, `results/oracle_nov2026/`, `scripts/` and these Markdown notes are intended to be committed. **The images and CSVs included with this handoff package are uploaded only after the user unpacks, commits and pushes it.** The local `.gitignore` excludes `data/checkpoints/` and `data/vision_*/`; these **cannot be retrieved from GitHub**. See [artifact index](docs/experiments/2026-11-10-artifact-index.md).

## New chat handoff

Say: **“Read `RESEARCH_HANDOFF.md` in my `xotnxodn0804/grid-cells-torch` GitHub repository and continue from there. For plots, check the artifact index.”** The GitHub repo is an external shared project log, NOT automatic synchronization with model memory.

## Next steps

1. Preserve baseline 500M checkpoint and original Lua level.
2. Validate raw DMLab world goal coordinates against the exported debug goal and reward/teleport timing; fix in a **versioned new run**.
3. Re-train new correctly encoded goal policy from scratch, keeping architecture/seed/eval matched to old model for controlled comparison.
4. Analyze vision decoder bias separately from RGB cue visibility; test wall-facing vs center-facing images at matched location.
