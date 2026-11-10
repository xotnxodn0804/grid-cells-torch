#!/usr/bin/env bash
# Copy from WSL local experiment output (where present), without changing source datasets/weights.
# Run after unpacking research_handoff_20261110.zip in the root of WSL ~/grid-cells-torch.
set -euo pipefail
cd "$(dirname "$0")/.."
copy_if_present() {
  local src="$1" dst="$2"
  if [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -n "$src" "$dst" || true
    echo "OK      $dst"
  else
    echo "MISSING $src (not copied; may already be bundled)"
  fi
}
copy_if_present results/vision_position_errors/vision_position_error_histogram.png results/vision_20261110/vision_position_error_histogram.png
copy_if_present results/vision_position_errors/vision_val_position_errors.csv results/vision_20261110/vision_val_position_errors.csv
for name in vision_true_vs_pred_scatter.png vision_true_vs_pred_scatter.pdf vision_spatial_error_heatmap.png vision_spatial_error_heatmap.pdf vision_spatial_sample_counts.png vision_spatial_error_bins.csv; do
  copy_if_present "results/vision_figures/$name" "results/vision_20261110/$name"
done
copy_if_present results/oracle_50m_compare/goal_y_corrected/fig2h_normal_vs_zero_corrected.png results/oracle_nov2026/fig2h_50m_ycorrected.png
copy_if_present results/oracle_500m_compare/goal_y_corrected/fig2h_normal_vs_zero_corrected.png results/oracle_nov2026/fig2h_500m_ycorrected.png
copy_if_present results/oracle_500m_compare/goal_audit/goal_vs_spawn_events.csv results/oracle_nov2026/goal_vs_spawn_events.csv
copy_if_present results/oracle_500m_compare/goal_audit/reward_source_audit.txt results/oracle_nov2026/reward_source_audit.txt
copy_if_present results/oracle_500m_normal/episodes.csv results/oracle_nov2026/episodes_500m_normal.csv
copy_if_present results/oracle_500m_zero/episodes.csv results/oracle_nov2026/episodes_500m_zero.csv
printf '\nChecking project source files:\n'
for p in train_oracle_place_agent.py eval_oracle_place_agent_ablation.py make_oracle_square_level.py levels/square_arena_oracle.lua; do
 if [[ -f "$p" ]]; then echo "OK      $p"; else echo "MISSING $p"; fi
done
printf '\nDone. No git commit/push was performed. Review before committing:\n'
git status --short
