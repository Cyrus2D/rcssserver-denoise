#!/usr/bin/env python3
"""Measure potential information gain from a second focus at fixed geometry."""

import argparse
import json
from pathlib import Path

import numpy as np

from build_sequences import range_bounds
from replay_normal import quantize_distance
from train_normal import Store

OFFSETS = (0, 2, 5, 10, 20)


def simulate(distance):
    reported = quantize_distance(distance, distance)
    low, high, _ = range_bounds(reported[None], np.zeros((1, len(distance))),
                                np.zeros(1), np.zeros(1))
    candidates = low.ravel()[:, None] + (high - low).ravel()[:, None] * np.linspace(.0001, .9999, 151)
    compatible = np.abs(quantize_distance(candidates, candidates) - reported[:, None]) < .02
    estimates, widths, fallback = [], [], []
    for offset in OFFSETS:
        second = quantize_distance(distance, np.abs(distance - offset))
        matches = compatible & (np.abs(quantize_distance(candidates, np.abs(candidates - offset))
                                       - second[:, None]) < .02)
        have = matches.any(axis=1)
        valid = np.where(have[:, None], matches, compatible)
        valid |= ~valid.any(axis=1)[:, None]
        left = np.where(valid, candidates, np.inf).min(axis=1)
        right = np.where(valid, candidates, -np.inf).max(axis=1)
        center = (candidates * valid).sum(axis=1) / valid.sum(axis=1)
        estimates.append(np.abs(center - distance))
        widths.append(right - left)
        fallback.append(~have)
    return np.stack(estimates, axis=1), np.stack(widths, axis=1), np.stack(fallback, axis=1)


def evaluate(args, split):
    store = Store(args.known, args.events, split)
    distances = []
    for _, rows, _ in store.games:
        mask = (rows[..., 18] == 1) & (rows[..., 2] > 0) & np.isfinite(rows[..., :6]).all(axis=-1)
        distances.append(rows[..., 2][mask])
    distance = np.concatenate(distances)
    batches = [simulate(distance[i:i + 4096]) for i in range(0, len(distance), 4096)]
    error, width, fallback = (np.concatenate(values) for values in zip(*batches))
    masks = {"all": np.ones(len(distance), bool), "under_10m": distance < 10,
             "10_to_25m": (distance >= 10) & (distance < 25), "at_least_25m": distance >= 25}
    return {name: {str(offset): {"n": int(mask.sum()), "mean_radial_error_m": float(error[mask, i].mean()),
                                "mean_width_m": float(width[mask, i].mean()),
                                "grid_fallbacks": int(fallback[mask, i].sum())}
                   for i, offset in enumerate(OFFSETS)}
            for name, mask in masks.items() if mask.any()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("known", "events", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    validation = evaluate(args, "validation")
    chosen = min(validation["all"], key=lambda x: validation["all"][x]["mean_radial_error_m"])
    test = evaluate(args, "test")
    args.output.write_text(json.dumps({"validation_offset_m": chosen, "validation": validation, "test": test},
                                      indent=2) + "\n")
