# Image-only Vision Module pilot — 2026-11-05

This is an **exploratory standalone vision experiment**, separate from the supervised 512-unit Grid Network and the RL Actor-Critic navigation system. It does **not** demonstrate goal-directed navigation.

## Scientific question and architecture

Can a single DeepMind Lab RGB observation predict place-cell and head-direction-cell activity at the actual camera pose?

- **Only model input:** a single RGB image of shape `(3,64,64)`, normalized to `[-1,1]`.
- **Targets:** 256 place-cell soft activations generated from **observed DMLab** `(x,y)`; 12 head-direction-cell soft activations from **observed DMLab** heading. Actual coordinates and heading are labels, **not input features**.
- **Network:** the existing `grid_cells/vision.py` (`VisionCNN` 4 conv layers + 256-dimensional embedding + separate place/HD linear heads). No modifications to this shared module: RL code may also depend on it.
- **Objective:** mean of `CE(place logits, place-cell soft target) + CE(HD logits, HD soft target)`. Optimizer RMSprop with learning rate `1e-4`, `alpha=0.99`, `eps=1e-8`; mini-batch size 32; initialization is truncated normal with scale `fan_in**(-1/2)` and truncation at 2 standard deviations, biases zero.
- **Place-cell labels:** 256 Gaussian cells, center locations in the `2.5m x 2.5m` arena, standard deviation `0.4m`. **HD labels:** 12 von Mises cells with `kappa=20`. Both use `build_rl_ensembles(Config(), device)`.
- `mask_prob=0.95`, `mask_mode='whole'` are configured for the future spatial-network feed. **Masking is not applied inside the standalone vision CE loss.**
- **Position metric:** weighted mean of PC centers according to predicted PC probabilities, vs actual DMLab position, in cm. Heading metric: circular mean of predicted HD-cell preferred angles vs actual DMLab heading, in degrees. These metrics are distinct from training CE.

## Data-generation procedure

Source: `collect_vision_rh_1000_v2.py`. DeepMind Lab's custom `levels/square_arena.lua` has a `2.5m x 2.5m` arena, 90-second episode limit and random episode textures/cues. One observation is recorded after every `action_repeat=4` rendered frames (60 simulation frames/s), i.e. approximately 15 observations/s. We use **1,300 observations/episode**, about 86.67 s, avoiding the 90-second timeout.

The collector uses a Raudies–Hasselmo-inspired **target-motion controller**: target speed drawn from a Rayleigh distribution (`scale=0.13 m/s`), target angular velocity from a zero-mean Gaussian (`std=330°/s`), with wall handling adapted from `grid_cells/data_generation.py`. The target speed/turn are converted to discrete DeepMind Lab commands (look pixels and probabilistic forward ON/OFF). Therefore **the realized trajectories do NOT exactly reproduce the continuous 2-D rodent motion model**. In the pilot, observed forward speed exceeded target speed (~0.16 target versus ~0.2–0.25 actual m/s), and this controller was deliberately **not recalibrated** before the final data collection. Actual position/heading recorded *after* each action are the only sources of the training labels. Measured and target velocities are retained for diagnostic use, not input to the standalone vision CNN.

Two separate datasets are concatenated **logically**, with memory mapping (not image copying):

| Local directory (not in Git) | Train episodes | Val episodes | Train images | Val images | Seed |
|---|---:|---:|---:|---:|---:|
| `data/vision_rh_pilot_100` | 90 | 10 | 117,000 | 13,000 | default `1234` |
| `data/vision_rh_extra_900` | 810 | 90 | 1,053,000 | 117,000 | `20261105` |
| **Total** | **900** | **100** | **1,170,000** | **130,000** | disjoint |

Separate training and held-out validation episodes; high correlation among consecutive images means `1.17M` training images are not independent experimental environments. Environment generation, movement, image-to-position ambiguity, and data distribution all influence generalization.

## Results confirmed from console output

| Optimizer updates | Train CE | Validation CE | Train position cm | Val position cm | Train heading ° | Val heading ° |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 5.2332 | 5.2608 | 34.1 | 35.2 | 23.9 | 23.1 |
| 30,000 | 5.1204 | 5.1488 | 29.5 | 30.4 | 22.1 | 21.6 |
| 50,000 | 5.0786 | 5.1099 | 27.6 | 28.6 | 21.1 | 20.5 |

**Status:** 50,000 updates are confirmed. A further run to **100,000 updates** was planned, but its results were not yet provided as of this note. Do **not** report the 100k result until verified. Train and validation evaluations use fixed random subsets of **10,000 images each**; this is not necessarily the full-set CE.

For comparison, the mean validation target entropy (all 130,000 validation observations, computed separately) is **4.851657 nats** for the sum of place + HD target entropies. This is a **label-informed lower bound** on average CE, not necessarily attainable from a single ambiguous RGB image. The entropy and model CE were calculated on different evaluation sample sets, so subtracting them gives an approximate, not an exact, mean KL divergence. Do not import the prior supervised Grid Network's CE target (different labels/environment/inputs) as a criterion of image-only performance.

No universal paper-backed pass/fail threshold has been established for this standalone vision experiment. The historical 30cm / 25° / CE5.3 cutoffs were exploratory only, not validated publication criteria.

## Exact continuation

Run in Ubuntu WSL from `~/grid-cells-torch` with `.venv-dmlab` activated, with **local datasets and checkpoints present**:

```bash
python train_vision_combined.py \
  --data-dirs data/vision_rh_pilot_100 data/vision_rh_extra_900 \
  --out-dir data/checkpoints/vision_rh_combined_1000 \
  --device cuda --updates 100000 --resume
```

`--updates 100000` is the **total** update count; it resumes from checkpoint update 50,000 and does not add 100,000 further updates. To inspect current metrics and latest saved optimizer step:

```bash
tail -n 6 data/checkpoints/vision_rh_combined_1000/metrics.csv
python - <<'PY'
import torch
p='data/checkpoints/vision_rh_combined_1000/latest.pt'
c=torch.load(p, map_location='cpu', weights_only=False)
print('Checkpoint optimizer updates:',c['update'])
PY
```

**Local assets excluded from Git:** `data/vision_*/` (raw RGB and labels) and `data/checkpoints/` (weights and optimizer state). Push only source and this summary. Preserve local checkpoint `latest.pt` and `vision_step*.pt` for replication; GitHub code alone cannot regenerate exact trained parameters.

## Recommended next analyses

1. Check the **100k update** validation curve and compare the best held-out checkpoint vs last. Monitor whether validation CE and position error continue to improve, plateau, or deteriorate.
2. Compute image-free baseline: fit average place/HD target distributions on the training set and score validation; compare constant-position baseline in cm. Also evaluate mean target entropy on exactly the fixed validation indices used for the model to get the paired CE excess KL.
3. Evaluate per-episode and per-location error maps, position/heading occupancy, duplicate/repetitive frames, and performance by unfamiliar episode textures.
4. Only after the standalone experiment is understood, integrate Vision and Grid Network, then Policy LSTM/Actor-Critic. The existing supervised Grid Network was trained in a `2.2m` arena, while the DMLab vision data uses `2.5m`: align place-cell centers, target widths and LSTM input dimensions before integration.

## Versions and historical files

- `grid_cells/vision.py` is the **shared canonical VisionModule implementation**; leave it in place (RL imports it).
- `collect_vision_rh_1000_v2.py` is the **actual collector used for these datasets**. The prior `collect_vision_rh_1000.py` was an earlier draft and is not needed for the reproducible current run.
- `train_vision_combined.py` contains its **own equivalent Sonnet-style weight initializer**, so it does not depend on or require committing the earlier 75-episode pilot script `vision_300k.py`.
- No old collector or pilot script appears on the remote default branch as checked on 2026-11-05, so there is **nothing of that name to delete remotely**.
