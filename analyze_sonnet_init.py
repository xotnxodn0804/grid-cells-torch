"""Analyze a trained Sonnet-LSTM-initialized checkpoint using expert's tools.

Example:
  python analyze_sonnet_init.py --checkpoint data/checkpoints/sonnet_v127_lstm_512/checkpoint_epoch299.pt --analysis gridness

The trained checkpoint always contains actual weights. The patch here makes
sure the evaluation script's *untrained* baseline is initialized with the
same distribution; it does not modify loaded trained weights.
"""

from __future__ import annotations

import os
import runpy
import sys

from sonnet_init_override import install


def main():
    install()
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "analyze_supervised.py")
    sys.argv[0] = script
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
