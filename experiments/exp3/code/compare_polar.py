#!/usr/bin/env python3
"""Compare bounded polar and Cartesian GRU readouts with matched training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_directional import step13
from replay_normal import MAX_AGE
from train_normal import GapGRU, Store, errors, fit, load_models, observer


class PolarGRU(GapGRU):
    def forward(self, feature, anchor, age):
        states, _ = self.gru(feature)
        correction = self.output(states[:, -1]).tanh()
        distance = torch.linalg.vector_norm(anchor, dim=-1)
        bound = .4 + .035 * distance + .22 * age
        # Safe default direction for a zero-range anchor.
        angle = torch.atan2(anchor[:, 1], anchor[:, 0])
        angle = angle + torch.atan2(bound, distance) * correction[:, 1]
        radius = (distance + bound * correction[:, 0]).clamp_min(0)
        return radius[:, None] * torch.stack((angle.cos(), angle.sin()), dim=-1)


def evaluate(args, split):
    store = Store(args.associated, args.events, split)
    models, calibration = load_models(args.runs)
    for mode, model in (("cartesian", GapGRU("cartesian")), ("polar", PolarGRU("polar"))):
        model.load_state_dict(torch.load(args.output / f"{mode}.pt", weights_only=True))
        models[mode] = model.eval()
    for path in args.additional_cartesian:
        name = path.parent.name
        if name in models:
            raise ValueError(f"duplicate checkpoint name: {name}")
        model = GapGRU("cartesian")
        model.load_state_dict(torch.load(path, weights_only=True))
        models[name] = model.eval()
    config = json.loads((args.runs / "audited-validation.json").read_text())["config"]
    methods = {"step13": config["gap"], "cartesian": "cartesian", "polar": "polar",
               **{path.parent.name: path.parent.name for path in args.additional_cartesian}}
    batches = []
    for index, (name, rows, _) in enumerate(store.games):
        paired = np.isfinite(rows[..., :6]).all(axis=-1)
        data = rows[paired]
        scored = {"librcsc": errors(data, observer(data, data[:, 4:6]))[0]}
        for method, mode in methods.items():
            estimate = step13(store, index, models, calibration, {**config, "gap": mode})
            scored[method] = errors(data, observer(data, estimate[paired]))[0]
        if not all(np.isfinite(score).all() for score in scored.values()):
            raise ValueError(f"{name}: non-finite prediction")
        batches.append((data[:, 17], scored))
        print(f"{split}: {index + 1}/{len(store.games)} observers", flush=True)
    age = np.concatenate([batch[0] for batch in batches])
    groups = {"all": np.ones(len(age), bool), "fresh": age == 0,
              "age_1_2": (age >= 1) & (age <= 2),
              "age_3_12": (age >= 3) & (age <= MAX_AGE)}
    results = {key: {} for key in groups}
    for method in batches[0][1]:
        score = np.concatenate([batch[1][method] for batch in batches])
        for key, mask in groups.items():
            values = score[mask]
            results[key][method] = {"n": int(mask.sum()), "mean_m": float(values.mean(dtype=np.float64)),
                                    "p90_m": float(np.quantile(values, .9))}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("events", "known", "associated", "runs", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--train", choices=("cartesian", "polar"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batches", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--additional-cartesian", type=Path, nargs="*", default=[])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if args.train:
        fit(args, args.train, gap_model=PolarGRU if args.train == "polar" else GapGRU)
    else:
        validation = evaluate(args, "validation")
        winner = min((m for m in validation["all"] if m != "librcsc"),
                     key=lambda m: validation["all"][m]["mean_m"])
        test = evaluate(args, "test")
        result = {"validation_winner": winner, "validation": validation, "test": test}
        filename = "expanded-metrics.json" if args.additional_cartesian else "metrics.json"
        (args.output / filename).write_text(json.dumps(result, indent=2) + "\n")
