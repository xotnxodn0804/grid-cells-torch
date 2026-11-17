# Oracle Place/HD Actor-Critic sanity test

This package is a **new, independent** experiment for `xotnxodn0804/grid-cells-torch` at commit `27a7ba9`. It does **not** overwrite the existing level, model, checkpoints, or RL trainer. It is not the Banino 2018 Grid Cell Agent; it is an **oracle current-place/HD control** inspired by the paper's Place Cell Agent.

## Install (Ubuntu WSL)

Copy these three Python files to the **repository root**:

- `make_oracle_square_level.py`
- `train_oracle_place_agent.py`
- `eval_oracle_place_agent.py`

Then:

```bash
cd ~/grid-cells-torch
source .venv-dmlab/bin/activate
python make_oracle_square_level.py
```

This generates **only** `levels/square_arena_oracle.lua`; `levels/square_arena.lua` is unchanged. It adds a debug observation (`ORACLE.GOAL.XY`) of the random goal-cell centre; the policy cannot access this value before finding the goal. Check that the output file exists.

## Smoke test (separate checkpoint dir)

```bash
python train_oracle_place_agent.py \
  --vision-checkpoint data/checkpoints/vision_rh_combined_1000/latest.pt \
  --num-envs 2 --device cpu --total-env-frames 12800 \
  --out-dir data/checkpoints/oracle_place_smoke
```

To test GPU and 32 workers, use `--num-envs 32 --device cuda` and a different new `--out-dir`. Expect substantial CPU/RAM load from 32 DeepMind Lab subprocesses; first verify with 2.

## Full training (up to paper's environment-frame budget)

```bash
python train_oracle_place_agent.py \
  --vision-checkpoint data/checkpoints/vision_rh_combined_1000/latest.pt \
  --num-envs 32 --device cuda --n-step 100 --action-repeat 4 \
  --total-env-frames 1000000000 \
  --out-dir data/checkpoints/oracle_place_a2c
```

Default hyperparameters: RMSProp lr=0.0001005, alpha=0.99, eps=1e-8, **momentum=0.0** (independent A2C baseline), discount=0.99, value coefficient=0.5, entropy coefficient=0.00008, 100 **agent actions** per BPTT update. No gradient clipping. To explicitly test the paper's `gradient momentum=0.99` reading, change only `--rmsprop-momentum 0.99` and use a **new output directory**. Never treat these as the same run.

Neural architecture: Pretrained VisionCNN 256-d embedding (weights **copied**, not shared; **frozen** by default), current oracle 256 place+12 HD code, remembered goal oracle 256 place+12 HD code; previous reward and previous action; Policy LSTM hidden=256; 6-action actor, 1-value critic. LSTM, newly created actor and critic weights initialized using truncated Gaussian (`std=1/sqrt(fan_in)` and ±2 std bounds), forget gate bias +1, initial recurrent states zero.

The level itself is 2.5m x 2.5m, 90 seconds / 5400 environment frames, action repeat 4. The goal is hidden and stable within the episode; the agent teleports to a uniformly selected **central 6x6 grid cell** after +10 reward. `ORACLE.GOAL.XY` is the true centre of the goal's map cell; it is only encoded into the policy **after the first reward**. Goal heading code defaults to the heading in the immediately preceding observation (`--goal-hd arrival`), which is an approximation of the orientation at contact. `--goal-hd zero` instead drops this potentially ambiguous heading. These choices are experiment-specific and distinguish this from the original Place Cell Agent.

The actor does **not** receive actual velocity or an RH motion controller. It learns six discrete actions from RL rewards; actual DMLab agent pose is used to construct the oracle spatial codes.

## Evaluation

```bash
python eval_oracle_place_agent.py \
  --checkpoint data/checkpoints/oracle_place_a2c/latest.pt \
  --episodes 100 --device cuda \
  --out-dir results/oracle_place_eval
```

Output: `episodes.csv` includes goal counts and first vs later arrival latency; `trajectory_epXXX.csv` includes actual position and a debug target centre for plotting Fig. 2h-style trajectories. The debug target centre saved to CSV is for **offline evaluation** only. `--argmax` evaluates deterministic actions instead of sampling.

## Key differences from paper

- A2C synchronized 32 subprocess environments, not paper's asynchronous A3C 32 actor threads.
- Ground-truth Place+HD codes, **no Grid Network**. The original paper compares this type of control in its Place Cell Agent.
- Pretrained VisionCNN copied from separate image-only experiment, **frozen** initially; paper uses a separate actor CNN trained via RL.
- Paper RMSProp gradient momentum table shows 0.99; this pilot defaults to 0.0 to match the independently documented more stable A2C setting. Compare separately.
- Ground-truth goal-cell **centre** is used after first hit, rather than the precise agent position at contact. Goal heading uses pre-contact observation; not exact event heading.
- The action vectors are from the independent repo, not specified fully by the paper.
- Checkpoint resume restarts environments at new episodes, so it is **not** bit-for-bit exact continuation; it restores model and optimizer weights.

## Validation status

Python syntax checks and a mocked pure-tensor architecture/return smoke test passed. DeepMind Lab end-to-end execution, exact goal observation availability, 32-worker capacity, and navigation success **have not been tested** in the model's environment. Do not claim successful navigation until training and held-out evaluation succeed.
