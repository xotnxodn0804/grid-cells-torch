# Research handoff — Grid Cells / Vision / Navigation

**Snapshot date:** 2026-11-17 (KST)  
**Repository:** https://github.com/xotnxodn0804/grid-cells-torch  
**Status of this document:** Reconstructed from recent ChatGPT discussions and the existing GitHub handoff. This is a *research-context snapshot*, NOT a substitute for the actual local source files or binary checkpoints. Verify commands, filenames, and local paths against the working tree before a new run.

## 0. READ FIRST: what is and is not on GitHub

- The remote `main` branch's latest confirmed commit during this snapshot was `9b604814cbd97f6d5d9b6c1e3ed61629299a325d`, with `RESEARCH_HANDOFF.md` last updated **2026-11-10**.
- Existing background and analyses are in `RESEARCH_HANDOFF.md`, `docs/experiments/2026-11-10-vision.md`, `docs/experiments/2026-11-10-oracle-a2c.md`, and `docs/experiments/2026-11-10-artifact-index.md`.
- **Work performed after Nov 10 may be only on the user's laptop.** GitHub cannot recover changes that have never been committed/pushed.
- `data/checkpoints/` and `data/vision_*/` are excluded by the local `.gitignore` according to the earlier artifact index. Model checkpoints, replay buffers, raw data, and large image datasets need separate deliberate backup; a normal `git push` does **not** protect them.
- For the B 78K inspection, the explicit constraint was **read-only analysis**: preserve checkpoints and replay data; never overwrite them during evaluation or debugging.

## 1. Research objective and core architecture

Reproduce and extend the Banino et al. (2018) grid-cell path-integration/navigation work in PyTorch and DeepMind Lab. Relevant threads:

1. **Supervised path integration:** velocity and angular movement -> LSTM -> bottleneck -> place-cell (PC) and head-direction (HD) decoders; investigate grid-like units and spatial representations.
2. **Vision module (`비전모듈`):** RGB 64x64 -> CNN -> spatial/heading code. Combine sparse visual observations with velocity-based integration.
3. **Goal-directed reinforcement learning:** Oracle Place A2C baseline; eventually test navigation based on learned Grid/Vision representations and controlled ablations.

Preferred terminology: call the image-processing CNN **비전모듈**. In mathematical explanations use `l_t` for LSTM cell state and `m_t` for hidden state (instead of `c_t`/`h_t`).

## 2. Older baselines (see original handoff for full provenance)

### 2.1 Supervised path integration baseline

- Training: approximately 1M simulated trajectories; LSTM128, bottleneck512 in the relevant later reproduction; 300 epochs x 1000 steps (confirm against active config/checkpoint before citing numbers).
- Rate maps and gridness analysis used spatial autocorrelograms and rotated-correlation metrics. Gridness threshold `> 0.37` is an analytical threshold, **not** itself a significance test or proof that a cell is a biological grid cell. Earlier experiments also examined shuffling and watershed methods.

### 2.2 Standalone vision baseline

- Dataset: 1,000 episodes, approximately 1.3M images; 64x64 RGB inputs.
- Earlier standalone validation: CE **5.0463**, position **26.3 cm**, HD **19.2 deg** (different experiment/decoder than the later A/B grid runs).
- A later vision setup gave a validation figure around **17.1 cm / 20.6 deg**. Do not confuse this with the original standalone result or with the *combined* Grid/Vision result.

### 2.3 Oracle Place A2C baseline

The **2026-11-10 handoff** documents separate 50M, 100M, 200M, and 500M frame policy evaluations. Examples: 200M -> **10.79** normal goal rewards/episode; 500M -> **10.54**; 500M with goal-memory code zeroed -> **2.90** (100-episode evaluations, same recorded seed). This is an Oracle-input navigation baseline, **not** a demonstration that the latest Grid+Vision network navigates successfully.

**Important legacy bug:** historical exported `ORACLE.GOAL.XY` had a Y-coordinate sign reversal relative to the physical goal. Do not silently reinterpret or resume old policy weights under a changed coding convention; use a new versioned run. See original handoff.

## 3. Most recent Grid + Vision A/B experiments (Nov 11–17)

### Common context

- Arena: approximately **2.5 m x 2.5 m** in DMLab.
- Collection: **1,350 steps / 90 s per episode**, with **4 parallel collection environments** noted in the recent B run.
- Grid network: LSTM with a **512-unit bottleneck**, PC and HD prediction targets; velocity and occasional vision embeddings are combined for training (check active code for exact 272-dimensional input and visual sampling ratio in this run).
- Visual sampling around 5% was explored in earlier versions; the actual A/B config and visibility distribution must be recovered from scripts/logs before claiming an exact matched parameter.

### Experiment A — offline / repeated data

- Method: uses a previously pretrained, **frozen Vision module**, repeatedly training Grid on an earlier pool of **900 training episodes** (verify actual data split on disk).
- Latest reported training length: approximately **300,006 Grid updates**.
- Reported evaluation/analysis summary:

| Metric | A reported |
|---|---:|
| Cross entropy (CE) | **2.1308** |
| Position error | **13.34 cm** |
| HD error | **18.30 deg** |
| Gridness (`mean > 0.37`) | **18/512** |
| Gridness (`min-max > 0.37`) | **9/512** |

- Known older working directory/checkpoint family: `data/checkpoints/grid1350_offline_A/`. Verify the actual exact final checkpoint filename before relying on it.

### Experiment B — newly collected RGB episodes

- Method in current stage: Vision had reached **300,000 updates** and is **frozen**, while **new RGB episodes continue to be generated** and Grid updates proceed. This differs from A's repeated offline 900-episode pool.
- On **2026-11-17**, B was **paused at an intermediate Grid checkpoint**, analysed without changing its checkpoint/replay data, then **resumed toward 300K Grid updates**. The *final* 300K outcome is **not established in this snapshot**.
- Intermediate checkpoint basename: **`grid_78k_analysis.pt`**. Latest verified state:
  - Collection rollouts: **5,575**
  - Grid updates: **78,050**
  - Vision updates: **300,000** (frozen at this stage)
  - Evaluation: **100 held-out episodes**

| Metric | B @ 78,050 |
|---|---:|
| Cross entropy (CE) | **2.1060** |
| Position error | **10.99 cm** |
| HD error | **18.76 deg** |
| Gridness (`mean > 0.37`) | **37/512** |
| Gridness (`min-max > 0.37`) | **13/512** |

- Rate-map inspection of selected channels: e.g. **#14 / #53** appeared as potential grid-like candidates, while **#507 / #195 / #255 / #331** included stripe-like examples. Visual impressions are *screening observations*, not definitive grid classification.

### Interpret A vs B carefully

- B @78K has a numerically lower reported position error and more threshold-crossing gridness units than A @300K; however, these are different training regimes and training steps. Ensure evaluation data, decoding method, smoothing, annulus, and gridness computation match *before* treating the metrics as a controlled head-to-head result.
- Both models can have relatively low position error while many units remain **stripe-like rather than hexagonal grid-like**. Low supervised CE/position error does not uniquely enforce a hexagonal internal representation. This is a working explanation/hypothesis, not a confirmed mechanistic proof.
- Next decisive comparison: evaluate **A vs B at matched Grid update counts and on the same held-out episodes with the same gridness pipeline**, then inspect rate maps and check statistical significance / shuffling as appropriate.

## 4. Immediate next steps in a new chat

1. **DO NOT overwrite B `grid_78k_analysis.pt`, A final checkpoints, or the source replay data.** Preserve originals; new analyses should write to unique output directories.
2. Inspect `git status -sb`, `git diff --stat`, and the current local branch/commit. Compare local work to GitHub `main` to identify uncommitted changes since Nov 10.
3. Save local source changes and experiment reports to GitHub with a reviewed, deliberate staging list. **Do not automatically commit checkpoints or multi-GB datasets.** Keep them in separate backed-up storage and record paths + hashes.
4. Find current B 300K training process/logs; determine if it completed or is still running. The latest supported snapshot only says **resumed toward 300K**.
5. Run a **matched A/B held-out evaluation** at comparable checkpoints; explicitly report CE, position error, HD error, gridness (>0.37 counts, mean/min-max definitions), rate maps, and any shuffle-based thresholds.
6. Check how vision observation frequency, online episode diversity, and supervised loss affect grid/stripe-like maps; consider a no-vision ablation once matched baselines are established.
7. Continue separate navigation/RL experiments only after preserving Grid/Vision baselines and documenting the representation-versus-policy distinction.

## 5. Safe Git commands (local WSL, inside working tree)

**Inspection before adding anything:**

```bash
cd ~/grid-cells-torch
git status -sb
git branch --show-current
git diff --stat
git log -1 --oneline
```

**After copying this handoff into `docs/experiments/2026-11-17-grid-vision-handoff.md`,** review all changes and stage only appropriate *source and documentation* paths. Example:

```bash
# First: visually inspect git status and identify the actual paths to save.
git status --short
# Stage the new handoff explicitly:
git add docs/experiments/2026-11-17-grid-vision-handoff.md
# Stage other needed tracked code edits once reviewed; does not add new untracked files:
git add -u
# Stage any newly created scripts ONE BY ONE with `git add path/to/new_script.py`.
git diff --cached --stat
git diff --cached --name-only
# Only if the staged set is correct:
git commit -m "docs: preserve Nov 17 grid vision A B handoff"
git push origin HEAD
```

Do not run `git reset --hard`, `git clean -fdx`, or force-push as a shortcut. Note: local untracked scripts require explicit staging, and ignored model checkpoints remain outside Git.

## 6. Starter prompt for the next ChatGPT conversation

> My research repo is `xotnxodn0804/grid-cells-torch`. First read `RESEARCH_HANDOFF.md` and `docs/experiments/2026-11-17-grid-vision-handoff.md` (check the branch containing the latter). We are comparing Grid+Vision experiments A and B. A was ~300,006 Grid updates on a repeated 900-episode dataset; B was paused/read-only analysed at 78,050 Grid updates (Vision 300K frozen) and then resumed toward 300K. B 78K gave CE 2.1060, position 10.99cm, HD 18.76 deg, gridness mean>0.37 37/512, min-max>0.37 13/512. Preserve `grid_78k_analysis.pt` and existing replay; do not assume B finished 300K. First inspect local Git status and the current experiment logs, then propose a matched A/B evaluation and next steps. Do not mistake spatial decoding metrics for navigation performance.
