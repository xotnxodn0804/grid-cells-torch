# Joint Navigation — 80M vs 169M

## Experiment
- Date: 2026-11-25
- Model: Vision + Grid Network (512D) + A2C
- Evaluation: 100 matched-seed episodes per checkpoint
- Episode: 90 seconds, 1,350 actions
- Goal memory: ON
- Vision cue rate: 0.05

## Results

| Metric | 80M | 169M |
|---|---:|---:|
| Goals/Episode | 7.17 | 12.15 |
| First Hit Success (90s) | 78% | 85% |
| First Hit Actions (successful only) | 337.9 | 347.1 |
| Returns within 10s | 77.7% | 92.4% |
| Total Returns | 639 | 1130 |

10-second returns are measured AFTER teleporting from
a previously discovered goal, not first-hit success.

## Source Code
- [Training](../../train_joint_nav_32actors.py)
- [Recovery/Resume](../../train_joint_nav_32actors_recover.py)
- [Behavior Evaluation](../../eval_joint32_80m_vs_169m.py)
- [Gridness Evaluation](../../eval_joint32_gridness_80m_vs_169m.py)

## Evaluation Command
Run from repository root in the WSL environment:

python eval_joint32_80m_vs_169m.py \
  --mode eval --episodes 100 --device cuda \
  --renderer software --goal-memory on \
  --out-dir results/joint32_80m_vs_169m_100ep

## Files
- report.txt: human-readable summary
- summary.json: machine-readable summary
- episode_metrics.csv: per-episode results
- return_attempts.csv: teleport return attempts
- step_samples.csv.gz: compressed step-level records, if archived
- *.png / *.pdf: analysis figures

## Checkpoints
Local only, not committed:
- data/checkpoints/joint_nav_32actors_200m/latest_backup_80m.pt
- data/checkpoints/joint_nav_32actors_200m/latest_backup_169m.pt

## Interpretation
Navigation improved, especially rapid goal returns.
This does not by itself establish vector-based navigation
or the presence of statistically significant grid cells.

## Research Notes
[Full experiment note](../../docs/experiments/2026-11-25-joint32-80m-169m-and-269m-plan.md)
