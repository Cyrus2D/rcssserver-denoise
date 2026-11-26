#!/usr/bin/env python3
"""Export the Exp6 player GRU for the HELIOS C++ launcher."""

import argparse
from pathlib import Path

import torch


LAYOUT = {
    "history.weight_ih_l0": (120, 14),
    "history.weight_hh_l0": (120, 40),
    "history.bias_ih_l0": (120,),
    "history.bias_hh_l0": (120,),
    "readout.0.weight": (40, 40),
    "readout.0.bias": (40,),
    "readout.2.weight": (3, 40),
    "readout.2.bias": (3,),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = tuple(checkpoint.get(key) for key in ("target", "history", "features", "hidden"))
    if metadata != ("player", 8, 14, 40):
        parser.error(f"expected the Exp6 player GRU with 8 frames, 14 features and 40 units, got {metadata}")

    tensors = checkpoint["model"]
    payload = bytearray(b"E6GRU001")
    for name, shape in LAYOUT.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
            parser.error(f"invalid tensor {name}")
        payload.extend(tensor.detach().numpy().astype("<f4", copy=False).tobytes(order="C"))

    args.weights.write_bytes(payload)
    print(args.weights)


if __name__ == "__main__":
    main()
