"""Reproduce Sonnet 1.27 default LSTM initialisation for the supervised model.

Use via train_sonnet_init.py or analyze_sonnet_init.py.  This module deliberately
changes neither grid_cells/model.py nor other parts of the expert's repository.

Source: Sonnet v1.27 gated_rnn.LSTM._create_gate_variables uses ONE concatenated
[input, hidden] matrix and basic.create_linear_initializer(input_size + hidden_size)
for both gate weights and gate biases; `forget_bias=1` is added at runtime.
PyTorch's LSTM uses two weight matrices and two biases.  Independent samples
from the same truncated normal match the Sonnet *distribution*; setting
bias_hh to zero and bias_ih's forget chunk +=1 represents the same forward
computation and training gradients.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def initialize_sonnet_lstm_(lstm: nn.LSTM) -> nn.LSTM:
    """In-place initialization of a single-layer, unidirectional PyTorch LSTM.

    Original Sonnet v1.27: std = 1/sqrt(input_size+hidden_size) for **all**
    LSTM gate weights and the gate bias; truncation is at +/- 2*std.

    Sonnet gate order is (input, candidate, forget, output), PyTorch order is
    (input, forget, candidate, output). As the draws are iid, the distribution
    is invariant to this permutation. The forget +1 is applied to PyTorch's
    2nd gate chunk (indices H:2*H).
    """
    if not isinstance(lstm, nn.LSTM):
        raise TypeError("Expected torch.nn.LSTM")
    if lstm.num_layers != 1 or lstm.bidirectional or lstm.proj_size or not lstm.bias:
        raise ValueError("This initializer supports one-layer, one-direction LSTM with biases, no projection")

    h = lstm.hidden_size
    fan_in = lstm.input_size + h
    std = 1.0 / math.sqrt(fan_in)

    with torch.no_grad():
        for weight in (lstm.weight_ih_l0, lstm.weight_hh_l0):
            nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2 * std, b=2 * std)

        # Sonnet's *single* 4H gate bias is initialized from the same
        # truncated normal as W. PyTorch has two bias vectors; their sum
        # is used in its gate equations.
        nn.init.trunc_normal_(lstm.bias_ih_l0, mean=0.0, std=std,
                              a=-2 * std, b=2 * std)
        nn.init.zeros_(lstm.bias_hh_l0)
        # Sonnet adds forget_bias=1.0 in the forward computation.
        lstm.bias_ih_l0[h:2 * h].add_(1.0)

    return lstm


def install() -> None:
    """Patch only GridCellsRNN construction for this Python process.

    All existing non-LSTM initializers, target generators and data readers
    are left untouched.  This must be called before building the model.
    """
    from grid_cells.model import GridCellsRNN

    if getattr(GridCellsRNN._init_weights, "_sonnet_v127", False):
        return
    original_init = GridCellsRNN._init_weights

    def _patched_init_weights(self):
        original_init(self)
        initialize_sonnet_lstm_(self.lstm)
        # A vision-conditioned RL LSTM is not built in the supervised task.
        # Do not silently change RL experiment specifications here.

    _patched_init_weights._sonnet_v127 = True
    GridCellsRNN._init_weights = _patched_init_weights


if __name__ == "__main__":
    lstm = nn.LSTM(3, 128, batch_first=True)
    initialize_sonnet_lstm_(lstm)
    print("Sonnet 1.27 initialisation OK (LSTM input=3, hidden=128, std=1/sqrt(131))")
