# Artifact index — filenames and locations for future ChatGPT conversations

All paths are **relative to** `~/grid-cells-torch` (WSL). Repository: https://github.com/xotnxodn0804/grid-cells-torch . This list refers to tracked results after importing, committing, and pushing the handoff package. The Windows clone `C:\Users\xotnd\GitHubProjects\grid-cells-torch` and the WSL clone **do not automatically sync**.

## Vision figures and data (committed to GitHub after push)

| Requested artifact | GitHub relative path | Generator |
|---|---|---|
| **A histogram** | `results/vision_20261110/vision_position_error_histogram.png` | `plot_vision_position_histogram.py` |
| **B true vs predicted scatter** | `results/vision_20261110/vision_true_vs_pred_scatter.png` | `plot_vision_true_vs_pred_scatter.py` |
| B PDF | `results/vision_20261110/vision_true_vs_pred_scatter.pdf` | same |
| **C heatmap** | `results/vision_20261110/vision_spatial_error_heatmap.png` | `plot_vision_spatial_error_heatmap.py` |
| C PDF | `results/vision_20261110/vision_spatial_error_heatmap.pdf` | same |
| Heatmap cell sample counts | `results/vision_20261110/vision_spatial_sample_counts.png` | heatmap script |
| Per-image validation error | `results/vision_20261110/vision_val_position_errors.csv` | validation evaluation |
| 16×16 mean error and counts | `results/vision_20261110/vision_spatial_error_bins.csv` | heatmap script |

Original WSL files: `results/vision_position_errors/vision_val_position_errors.csv`, `results/vision_position_errors/vision_position_error_histogram.png`, `results/vision_figures/vision_true_vs_pred_scatter.(png|pdf)`, `results/vision_figures/vision_spatial_error_heatmap.(png|pdf)`. The standardized GitHub files are copies, not a change to model weights.

## RL figures, diagnostics and source (after push)

| Requested artifact | GitHub relative path / local working tree |
|---|---|
| 50M normal-vs-zero trajectory with physical goal corrected | `results/oracle_nov2026/fig2h_50m_ycorrected.png` |
| 500M normal-vs-zero trajectory with physical goal corrected | `results/oracle_nov2026/fig2h_500m_ycorrected.png` |
| Goal event coordinate audit CSV | `results/oracle_nov2026/goal_vs_spawn_events.csv` |
| Reward / coordinate source audit report | `results/oracle_nov2026/reward_source_audit.txt` |
| 500M normal 100-episode metrics | `results/oracle_nov2026/episodes_500m_normal.csv` |
| 500M zero 100-episode metrics | `results/oracle_nov2026/episodes_500m_zero.csv` |
| **500M RL training source (not C)** | `train_oracle_place_agent.py` |
| RL evaluator normal / zero | `eval_oracle_place_agent_ablation.py` |
| Lua level implementation | `levels/square_arena_oracle.lua` |
| Source audit generator | `audit_oracle_reward_sources.py` |
| Corrected fig2h generator | `plot_oracle_fig2h_yfixed_generic.py` |

Note: the 500M work is **Python/PyTorch and a Lua level**, not a C-language training program. C/C++ exists in the external DeepMind Lab engine, but the project’s own train script is `.py` and its level is `.lua`.

Original local result folders: `results/oracle_500m_normal`, `results/oracle_500m_zero`, `results/oracle_50m_normal`, `results/oracle_50m_zero`. Each may contain `episodes.csv` and `trajectory_epNNN.csv`; the standard handoff only commits **summary metrics** and representative figures/audit. Full 100-episode raw trajectory arrays can be separately archived if desired.

## Large local-only datasets and checkpoints — NOT GitHub-hosted

| What | Exact *local* WSL path |
|---|---|
| Supervised vision model | `data/checkpoints/vision_rh_combined_1000/latest.pt` |
| A2C 500M trained model | `data/checkpoints/oracle_place_32_run6/latest.pt` |
| A2C 200M-named saved model | `data/checkpoints/oracle_place_32_run5/checkpoint_200m.pt` |
| Vision raw images and labels | `data/vision_*/` |

These paths are excluded by `.gitignore`. **GitHub alone cannot restore the weights or image dataset.** Preserve in separate external backups (or later intentionally upload to a Release / LFS / storage).

## Reproduction / update protocol

- For vision source, see `collect_vision_rh_1000_v2.py`, `train_vision_combined.py`, `grid_cells/vision.py`, and `plot_vision_*.py`.
- For oracle, see `train_oracle_place_agent.py`, `eval_oracle_place_agent_ablation.py`, `levels/square_arena_oracle.lua` and `plot_oracle_fig2h_yfixed_generic.py`.
- Handoff package includes fixed artifacts under the standard `results/` paths. `scripts/collect_nov2026_local_artifacts.sh` copies additional data/code from original WSL results if present, without deleting originals.
- To update GitHub, use `git status`, `git add` targeted files, `git commit`, then `git push` from *the WSL repo*. After pushing, verify the web links actually show the images.
- **New experiment convention:** date, model, git hash, config, seed, raw-vs-plotted coordinate convention, frame/update count, metrics, script command, generated paths, and known caveats in dated experiment note; append any new artifact to this index.
