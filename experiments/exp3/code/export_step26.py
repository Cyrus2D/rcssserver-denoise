#!/usr/bin/env python3
"""Export the frozen Step 26 checkpoint for Helios inference."""

import argparse
from pathlib import Path

import numpy as np
import torch

WEIGHTS = (
    "embedding.weight",
    "neighbor_encoder.0.weight", "neighbor_encoder.0.bias",
    "neighbor_encoder.2.weight", "neighbor_encoder.2.bias",
    "encoder.weight_ih_l0", "encoder.weight_hh_l0",
    "encoder.bias_ih_l0", "encoder.bias_hh_l0",
    "readout.0.weight", "readout.0.bias", "readout.2.weight", "readout.2.bias",
    "bearing.weight", "bearing.bias",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["mode"] != "masked_set_polar":
        parser.error("expected a Step 26 masked_set_polar checkpoint")
    with args.output.open("w", encoding="ascii") as stream:
        stream.write("exp3-player-position-polar-v1\n")
        for name in WEIGHTS:
            np.savetxt(stream, checkpoint["model"][name].numpy().reshape(1, -1), fmt="%.9g")


if __name__ == "__main__":
    main()
