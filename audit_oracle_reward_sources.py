#!/usr/bin/env python3
"""Read-only source audit for Oracle Place reward / goal-coordinate mismatches.

Run from ~/grid-cells-torch:
  python audit_oracle_reward_sources.py

Output:
  results/oracle_500m_compare/goal_audit/reward_source_audit.txt

Does not import DeepMind Lab, change weights, simulate, or write to source files.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/oracle_500m_compare/goal_audit/reward_source_audit.txt"

SOURCES = [
    ("Lua active environment", "levels/square_arena_oracle.lua",
     r"(?i)reward|goal|pickup|trigger|teleport|respawn|spawn|player|callback|position|origin|x_m|y_m|observe|state"),
    ("Lua generator", "make_oracle_square_level.py",
     r"(?i)reward|goal|pickup|trigger|teleport|respawn|spawn|player|origin|world|position|grid|meter|level"),
    ("Oracle runner and goal-code encoding", "train_oracle_place_agent.py",
     r"(?i)oracle_goal_xy|def +goal_codes|def +oracle_codes|class +OracleRunner|def +step|def +reset|reward|goal_hd|pos.*100|pos.*400|position|teleport"),
    ("DMLab observation conversion", "grid_cells/rl_common.py",
     r"(?i)def +convert_observation|pos|oracle_goal_xy|map|scal|100|400|grid|xy|observation|state|world"),
    ("Eval reward event recording", "eval_oracle_place_agent_ablation.py",
     r"(?i)goal_hit|oracle_goal_xy|next_obs|env\.step|reward|arrivals|xys|memory|def +goal_codes"),
]

def extract(path, pattern, *, before=4, after=7, max_lines=180):
    if not path.exists():
        return ["[NOT FOUND] " + str(path)]
    contents = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = [i for i, s in enumerate(contents) if re.search(pattern, s)]
    selected = set()
    for i in hits:
        selected.update(range(max(0, i-before), min(len(contents), i+after+1)))
    selected = sorted(selected)
    if len(selected) > max_lines:
        selected = selected[:max_lines]
        truncated = True
    else:
        truncated = False
    ans = [f"Path: {path.relative_to(ROOT)} | {len(contents)} lines | {len(hits)} matches"]
    previous = None
    for i in selected:
        if previous is not None and i > previous+1:
            ans.append("  ...")
        ans.append(f"{i+1:4d} | {contents[i]}")
        previous = i
    if truncated:
        ans.append("[OUTPUT LIMITED] Open original file for the rest.")
    return ans

def main():
    sections = [
        "ORACLE REWARD/GOAL SOURCE AUDIT",
        "This is a SOURCE-CODE INSPECTION report, NOT a proof of the bug.",
        "Question: Is the actual reward/goal world position consistent with",
        "the coordinate exported as oracle_goal_xy and the converted agent pos?",
        "Separate questions: reward timing, teleport timing, goal-memory encoding.",
        "",
    ]
    for title, path, pattern in SOURCES:
        sections += ["", "="*75, title, "="*75]
        sections += extract(ROOT / path, pattern)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print("Saved full report:", OUT.relative_to(ROOT))
    print("Inspected files:")
    for _, name, _ in SOURCES:
        print(f"  {'OK' if (ROOT / name).exists() else 'MISSING':7s} {name}")
    print()
    print("Please share this output file (or paste its contents).")
    print("No training/evaluation rerun or modification of files was performed.")

if __name__ == "__main__":
    main()
