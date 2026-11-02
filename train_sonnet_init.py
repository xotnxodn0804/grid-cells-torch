"""Train a separate Sonnet-1.27-LSTM-initialized supervised experiment.

Place beside the expert's train.py and run from the repository root:
    python train_sonnet_init.py --shard-dir data/datasets/gpu1m/shards

Everything else follows the existing train.py/config.py (including neurons_seed=0,
PC sigma=0.01, HD kappa=20, 512 bottleneck units and 300,000 updates).
Never overwrites an existing checkpoint directory.
"""

from __future__ import annotations

import os
import runpy
import sys

from sonnet_init_override import install


def main():
    # Do this before the standard train.py constructs its GridCellsRNN.
    install()

    # Prevent accidents: default to a NEW directory, never reuse the baseline.
    if "--results-dir" not in sys.argv[1:]:
        sys.argv.extend(["--results-dir", "data/checkpoints/sonnet_v127_lstm_512"])

    # train.py writes checkpoints into the specified output directory.
    # Existing code doesn't prevent overwrite, so fail early if any model
    # checkpoint already exists there.
    from argparse import ArgumentParser
    p = ArgumentParser(add_help=False)
    p.add_argument("--results-dir")
    known, _ = p.parse_known_args(sys.argv[1:])
    results_dir = known.results_dir
    if os.path.isdir(results_dir) and any(
        name.startswith("checkpoint_epoch") and name.endswith(".pt")
        for name in os.listdir(results_dir)
    ):
        raise SystemExit(f"REFUSING to overwrite existing checkpoints in: {results_dir}")

    print("[Sonnet 1.27] LSTM: truncated normal, std=1/sqrt(input+hidden),")
    print("              gate bias: truncated normal, forget bias +1")
    print(f"[Sonnet 1.27] new checkpoints: {results_dir}")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    sys.argv[0] = script
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
