"""Small sanity check that needs only PyTorch (no actual dataset)."""

import math
import torch
from torch import nn

from sonnet_init_override import initialize_sonnet_lstm_


def test_sonnet_v127_lstm_init():
    torch.manual_seed(12)
    lstm = nn.LSTM(3, 128, batch_first=True)
    initialize_sonnet_lstm_(lstm)
    std = 1 / math.sqrt(131)
    assert lstm.weight_ih_l0.abs().max() <= 2 * std + 1e-6
    assert lstm.weight_hh_l0.abs().max() <= 2 * std + 1e-6
    assert torch.count_nonzero(lstm.bias_hh_l0).item() == 0
    assert (lstm.bias_ih_l0[:128].abs() <= 2 * std + 1e-6).all()
    assert ((lstm.bias_ih_l0[128:256] - 1).abs() <= 2 * std + 1e-6).all()
    assert (lstm.bias_ih_l0[256:].abs() <= 2 * std + 1e-6).all()
    assert abs(lstm.weight_hh_l0.std().item() - std * 0.88) < std * .2
    x = torch.randn(3, 100, 3)
    h0 = torch.randn(1, 3, 128)
    c0 = torch.randn(1, 3, 128)
    output, _ = lstm(x, (h0, c0))
    assert output.shape == (3, 100, 128)
    assert torch.isfinite(output).all()


if __name__ == "__main__":
    test_sonnet_v127_lstm_init()
    print("PASS: LSTM initialisation distribution/range, gates and forward pass")
