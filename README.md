# grid-cells-torch

A PyTorch reproduction and exploratory analysis of the grid-cell network described in:

**Banino et al. (2018), Vector-based navigation using grid-like representations in artificial agents.**

This repository contains a supervised path-integration network, spatial-representation analyses, and an independently implemented reinforcement-learning framework for navigation in DeepMind Lab.

The current experimental focus is a **512-unit bottleneck network trained with Sonnet 1.27-style LSTM initialization**, followed by gridness and grid-spacing analyses.

**Research status (November 2026):** Supervised training and spatial-representation analyses have been completed. Navigation experiments using the newly trained 512-unit network remain a next step.

## Paper and attribution

> Banino, A., Barry, C., Uria, B., Blundell, C., Lillicrap, T., Mirowski, P., Pritzel, A., Chadwick, M. J., Degris, T., Modayil, J., Wayne, G., Soyer, H., Viola, F., Zhang, B., Goroshin, R., Rabinowitz, N., Pascanu, R., Beattie, C., Petersen, S., Sadik, A., Gaffney, S., King, H., Kavukcuoglu, K., Hassabis, D., Hadsell, R. & Kumaran, D. **Vector-based navigation using grid-like representations in artificial agents.** *Nature* 557, 429–433 (2018).

[Original paper](https://doi.org/10.1038/s41586-018-0102-6)

[DeepMind's released supervised-learning code](https://github.com/google-deepmind/grid-cells)

This repository builds on the independent PyTorch implementation by [soluslan/grid-cells-torch](https://github.com/soluslan/grid-cells-torch), with additional GPU data generation, Sonnet initialization, gridness analysis, and grid-spacing experiments.

This is an independent reproduction project, not an official DeepMind implementation.

## 1. Experimental setup

### Supervised grid-cell network

| Component | Configuration |
|---|---|
| Environment | 2.2 × 2.2 m square arena |
| Input | Egocentric velocity and initial spatial information |
| Recurrent network | LSTM, 128 hidden units |
| Bottleneck | 512 units |
| Place-cell output | 256 units |
| Head-direction output | 12 units |
| Bottleneck dropout | 0.5 |
| LSTM initialization | Sonnet 1.27-style truncated normal |
| Training data | 1,000,000 simulated trajectories |
| Trajectory length | 100 steps, 15 seconds |
| Training | 300 epochs × 1,000 steps |
| Total optimizer updates | 300,000 |
| Analysis sample | 4,000 trajectories |

The LSTM initialization was adapted from Sonnet 1.27, including its truncated-normal gate parameters and additional forget-gate bias.

The remaining training architecture follows the existing PyTorch implementation, with the recorded configuration in `config.py`.

### Main checkpoint

`data/checkpoints/sonnet_v127_lstm_512/checkpoint_epoch299.pt`

**Important:** Model checkpoints and generated training datasets are excluded from Git version control and must be preserved locally or backed up separately.

## 2. Supervised learning

The model was trained to predict place-cell and head-direction activity from self-motion inputs.

The resulting network was analyzed to investigate whether spatially structured representations emerged in its bottleneck units.

The principal trained model has:

- 128 LSTM hidden units.
- 512 bottleneck units.
- 300,000 optimizer updates.
- Sonnet 1.27-style LSTM initialization.

Training a path-integration model and demonstrating grid-like activity are distinct from demonstrating successful goal-directed navigation. The latter requires separate reinforcement-learning evaluation.

### Saved figures

[512 bottleneck rate maps](results/sonnet_v127_lstm_512/bottleneck_ratemaps_epoch299.pdf)

[LSTM rate maps](results/sonnet_v127_lstm_512/lstm_ratemaps_epoch299.pdf)

[Initial gridness analysis](results/sonnet_v127_lstm_512/gridness_epoch299.png)

## 3. Gridness analysis

### Analysis method

Spatial firing-rate maps were constructed with a resolution of **20 × 20 bins**, producing **39 × 39 spatial autocorrelograms**.

The analysis uses the 10-annulus configuration from DeepMind's publicly released code:

- Inner radius: 4 bins.
- Outer radii: 10 equally spaced values from 8 to 20 bins.
- Rotations: 30°, 60°, 90°, 120°, and 150°.
- Gridness: the maximum score across the tested annuli.

The mean-formula gridness score is:

\[
G=\frac{r_{60}+r_{120}}{2}-\frac{r_{30}+r_{90}+r_{150}}{3}
\]

Here, each \(r_\theta\) represents the rotational correlation of the spatial autocorrelogram at angle \(\theta\).

**Methodological distinction:** The released DeepMind code's mean formula and 10-annulus configuration differ from the min–max formula and seven expanding annuli described in the paper's Methods. These procedures must not be treated as interchangeable.

### Results

| Measurement | Result |
|---|---:|
| Total bottleneck units | 512 |
| Units with G > 0.37 | 108 (21.1%) |
| Mean observed gridness | 0.1635 |
| Units exceeding their own shuffled 95th percentile | 67 (13.1%) |
| Units with empirical shuffle p ≤ 0.05 | 60 (11.7%) |
| Field shuffles per unit | 100 |

**Interpretation:** The 108/512 result is an exploratory count based on the historical threshold G > 0.37. This threshold was not recalibrated for the current mean-formula scoring protocol.

Field shuffling was also implemented to construct unit-specific null distributions. However, the watershed segmentation, smoothing, and field relocation procedures include implementation choices that have not been established as identical to the original paper.

Consequently, the shuffled classifications are preliminary and require further validation.

Neither the 108/512 count nor the 67/512 count should be presented as an exact reproduction of the paper's reported percentage of grid-like units.

### Saved results

[All 512 rate maps — 20 × 20](results/sonnet_deepmind20_mean/all_512_ratemaps_20x20.pdf)

[Top 32 rate maps and autocorrelograms](results/sonnet_deepmind20_mean/top_32_ratemaps_and_sacs.pdf)

[Unit-level statistics](results/sonnet_deepmind20_mean/unit_statistics.csv)

[Gridness scores](results/sonnet_deepmind20_mean/gridness_scores_sorted.csv)

[Gridness analysis summary](results/sonnet_deepmind20_mean/summary.json)

Additional methodological documentation is available in [README_Gridness_DeepMind20_KO.md](README_Gridness_DeepMind20_KO.md).

## 4. Grid-spacing analysis

Grid spacing was estimated for the **108 bottleneck units selected by the exploratory G > 0.37 criterion**.

### Procedure

1. Calculate a 39 × 39 spatial autocorrelogram for each unit.
2. Identify and exclude the central autocorrelation peak using a threshold of 0.5 and its maximum connected extent along the four cardinal directions.
3. Identify strict local maxima outside the central exclusion region.
4. Select the six local maxima nearest to the autocorrelogram center.
5. Calculate their distances from the center.
6. Define grid spacing as the median of those six distances.

The current square arena and 20-bin resolution correspond to **11 cm per bin**.

Both 8-neighbor and 4-neighbor definitions of local maxima were tested.

### 4-neighbor analysis: 108 units

| Measurement | Result |
|---|---:|
| Analyzed units | 108 |
| Median estimated spacing | 102.59 cm |
| Minimum estimated spacing | 73.79 cm |
| Maximum estimated spacing | 226.77 cm |
| BIC-selected GMM components | 3 |

Estimated GMM component centers:

| Component | Spacing |
|---|---:|
| 1 | 87.90 cm |
| 2 | 123.41 cm |
| 3 | 208.10 cm |

Adjacent component-center ratios were approximately **1.40** and **1.69**.

The three- and four-component models had nearly identical BIC values (1050.043 versus 1050.130), so the exact number of components is not strongly resolved.

### Exploratory outlier sensitivity: 106 units

One minimum-spacing observation and one maximum-spacing observation were removed to examine the sensitivity of GMM selection.

| Measurement | Result |
|---|---:|
| Remaining units | 106 |
| BIC-selected GMM components | 4 |
| Minimum BIC | 1021.31 |

Approximate component centers:

**88.99, 123.38, 169.16, and 210.90 cm.**

This is a sensitivity analysis, not evidence that the removed observations were necessarily invalid.

### Comparison with the paper

Banino et al. reported approximately three characteristic spacing modes near **47, 70, and 106 cm**, with adjacent ratios near **1.49** and **1.51**.

Our measured distributions differ from those values.

An important limitation is that autocorrelogram peaks near the boundary can be estimated from relatively few overlapping spatial bins. In the 4-neighbor analysis, **22 of the 108 units** had at least one selected peak supported by fewer than 100 overlapping pixels.

These low-overlap peaks may inflate grid-spacing estimates and produce apparent large-spacing clusters.

Therefore, the current GMM results are **descriptive and exploratory**, not definitive evidence for discrete biological-like grid modules.

### Saved results

[4-neighbor Grid Spacing summary](results/sonnet_deepmind20_mean/grid_spacing_108_4neighbor/summary.json)

[108-unit Grid Spacing CSV](results/sonnet_deepmind20_mean/grid_spacing_108_4neighbor/grid_spacing_108_units.csv)

[Six peaks per unit](results/sonnet_deepmind20_mean/grid_spacing_108_4neighbor/six_peaks_per_unit.csv)

[4-neighbor peak analysis figures](results/sonnet_deepmind20_mean/grid_spacing_108_4neighbor/all_selected_peak_details.pdf)

[GMM/BIC comparison figure — 108 versus 106 units](results/grid_scale_paper_style_comparison/grid_scale_BIC_108_vs_106.png)

[GMM/BIC values](results/grid_scale_paper_style_comparison/bic_comparison.csv)

The original 8-neighbor results are retained separately under `results/sonnet_deepmind20_mean/grid_spacing_108/`.

## 5. Historical 256-unit baseline

The original repository documented an earlier 256-unit supervised experiment from August 2026.

These results are retained as a historical reference rather than attributed to the current 512-unit experiment.

| Measurement | Historical baseline |
|---|---:|
| Bottleneck units | 256 |
| Training updates | 300,000 |
| Reported top-3 position error at 15 s | 16.0 cm |
| Gridness > 0.37 | 20/256 (7.8%) |
| LSTM units with Gridness > 0.37 | 0/128 |

The historical baseline and current Sonnet experiment do not constitute a controlled comparison of bottleneck size alone: initialization, analysis settings, and potentially other implementation details differ.

Results from different gridness formulas or annulus configurations should not be directly compared as though they use a common validated threshold.

Historical figures remain available in `results/baseline/` and `results/gridness_256/`.

## 6. Navigation and reinforcement learning

The repository also contains a DeepMind Lab-based navigation framework, including:

- `train_rl_agent.py`
- `eval_rl_agent.py`
- `grid_cells/rl_a3c.py`
- `grid_cells/rl_a2c.py`
- `levels/`
- `scripts/setup_dmlab.sh`

The original repository documented earlier A3C training runs that did not demonstrate successful navigation. Its subsequent A2C work and diagnostic experiments are documented in `docs/plan/`.

**These historical RL experiments are separate from the newly trained Sonnet-initialized 512-unit supervised model.**

### Next research objective

The next stage is to investigate goal-directed navigation in a square DeepMind Lab environment, using a policy LSTM trained with reinforcement learning and incorporating the learned spatial representation.

Planned validation stages include:

1. Verify environment observations and agent movement.
2. Establish a random-policy navigation baseline.
3. Train and evaluate a policy LSTM in the square arena.
4. Evaluate whether spatial representations improve goal-directed navigation.
5. Compare navigation behavior with the original paper, while documenting implementation differences.

Successful navigation with the current 512-unit checkpoint has **not yet been demonstrated**.

## 7. Running the supervised experiments

Install the Python dependencies using:

`pip install -r requirements.txt`

Generate the GPU trajectory dataset using `generate_data_gpu.py`, with the data-generation parameters recorded in the script and `config.py`.

To verify the Sonnet initialization:

`python test_sonnet_init.py`

To start a separate Sonnet-initialized training run:

`python train_sonnet_init.py --shard-dir data/datasets/gpu1m/shards`

**Do not reuse an existing checkpoint output directory.** The training wrapper is designed to prevent accidental checkpoint overwrites.

To reproduce the gridness analysis:

`python run_field_shuffle_deepmind20.py --self-test`

To reproduce the 4-neighbor grid-spacing analysis:

`python analyze_108_grid_spacing_4neighbor.py --self-test`

`python analyze_108_grid_spacing_4neighbor.py`

To regenerate the GMM comparison figures:

`python plot_grid_scale_bic_paper.py`

Running the analysis scripts requires the relevant local training checkpoint and trajectory data.

For DeepMind Lab installation and RL training, consult `scripts/setup_dmlab.sh`, `docs/paper/`, and `docs/plan/`. The simulator installation is platform-dependent.

## 8. Repository organization

| Path | Purpose |
|---|---|
| `config.py` | Model, training, and RL configuration |
| `grid_cells/` | Neural-network and evaluation implementation |
| `generate_data_gpu.py` | GPU trajectory generation |
| `train_sonnet_init.py` | Sonnet-style LSTM training |
| `sonnet_init_override.py` | Sonnet initialization adapter |
| `run_field_shuffle_deepmind20.py` | 20 × 20 gridness and field-shuffling analysis |
| `analyze_108_grid_spacing.py` | 8-neighbor grid-spacing analysis |
| `analyze_108_grid_spacing_4neighbor.py` | 4-neighbor grid-spacing analysis |
| `plot_grid_scale_bic_paper.py` | GMM/BIC figures |
| `results/` | Selected figures, CSV data, and experiment summaries |
| `data/` | Local datasets and model checkpoints, excluded from Git |
| `levels/` | Navigation environment definitions |
| `docs/` | Paper notes, experimental plans, and implementation caveats |

## 9. Limitations and reproducibility

- The reproduction is implemented in PyTorch and is not guaranteed to be numerically identical to DeepMind's original TensorFlow/Sonnet implementation.
- Historical and newly implemented gridness evaluation procedures are not directly interchangeable.
- The fixed threshold G > 0.37 was not recalibrated for the current mean-formula analysis.
- Field shuffling uses approximations that require further validation.
- Grid-spacing estimates can be biased by low-overlap autocorrelogram peaks.
- GMM component counts are sensitive to data selection and fitting choices.
- Training checkpoints and trajectory datasets are not stored in GitHub.
- Results from earlier RL baselines must not be confused with navigation performance of the present supervised model.

This repository is maintained as a reproducible research and exploratory-analysis project. Reported measurements should be interpreted in the context of their specific evaluation settings.