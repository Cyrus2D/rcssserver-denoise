#!/usr/bin/env python3
"""Export the three frozen Exp4 checkpoints for Helios."""

import argparse
from pathlib import Path

import numpy as np
import torch

CHECKPOINTS = {"player_context": "player", "ball_quantized": "ball", "ball_experts": "forecast"}
WEIGHTS = ("neighbor.0.weight", "neighbor.0.bias", "neighbor.2.weight", "neighbor.2.bias",
           "gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
           "readout.0.weight", "readout.0.bias", "readout.2.weight", "readout.2.bias")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for checkpoint, name in CHECKPOINTS.items():
        state = torch.load(a.runs / f"{checkpoint}.pt", map_location="cpu", weights_only=True)
        fields = WEIGHTS if name == "player" else ("owner.weight", *WEIGHTS)
        with (a.output / f"{name}.weights").open("w", encoding="ascii") as out:
            out.write(f"exp4-{name}-v1\n")
            for field in fields:
                np.savetxt(out, state[field].numpy().reshape(1, -1), fmt="%.9g")
