#!/usr/bin/env python3
"""Export the Exp2 circular position-only GRU for the Helios C++ runtime."""

import argparse
from pathlib import Path

import torch

from schema import FEATURES


def export(checkpoint_path, output_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_config = {
        "variant": "summary", "hidden_size": 48, "circular_angles": True,
        "position_residual": False, "landmark_summary": True, "burn_in": 5,
    }
    if any(checkpoint["config"].get(key) != value for key, value in expected_config.items()):
        raise ValueError("expected a Step 05 circular position-only summary checkpoint")

    weights = checkpoint["model"]
    shapes = {
        "input_scales": (len(FEATURES),),
        "gru.weight_ih": (144, len(FEATURES) + 3 + 9 + 16),
        "gru.weight_hh": (144, 48),
        "gru.bias_ih": (144,),
        "gru.bias_hh": (144,),
        "head.weight": (2, 48),
        "head.bias": (2,),
    }
    for key, shape in shapes.items():
        tensor = weights[key]
        if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
            raise ValueError(f"unexpected checkpoint tensor {key}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as output:
        output.write("exp2-self-position-gru-circular-v1\n")
        for key in shapes:
            tensor = weights[key][0] if key.startswith("head.") else weights[key]
            values = (format(value, ".9g") for value in tensor.flatten().tolist())
            output.write(" ".join(values) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.output)
