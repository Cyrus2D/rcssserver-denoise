#!/usr/bin/env python3
"""Compare player and ball GRUs on paired Exp3 observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from build_dataset import CONTEXT, FEATURES, HISTORY, unit

VARIANTS = {"player_history": "player", "player_context": "player",
            "ball_history": "ball", "ball_context": "ball", "ball_experts": "ball", "ball_quantized": "ball",
            "ball_context_no_owner": "ball"}
METRICS = {"player": ("velocity", "body"),
           "ball": ("position", "velocity", "next_position", "next_velocity")}


class Stream:
    def __init__(self, path, ball, stride):
        self.inputs, self.context, self.state, self.truth = [
            np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("input", "context", "state", "truth")
        ]
        with np.load(path / "timing.npz", allow_pickle=False) as t:
            cycles, self.controlled, self.true_control = (t[k] for k in ("cycles", "controlled", "true_control"))
            self.decay = float(t["ball_decay"])
            self_id = int(t["self_id"])
        continuous = np.r_[False, np.diff(cycles) == 1]
        idx = np.arange(len(cycles))
        start = np.maximum.accumulate(np.where(continuous, 0, idx))
        stop = len(cycles) - int(ball)
        possible = np.arange(HISTORY - 1, stop)
        eligible = possible[possible - start[possible] >= HISTORY - 1]
        if ball:
            eligible = eligible[continuous[eligible + 1]]
            truth_ok = np.isfinite(self.truth[eligible, 0, :4]).all(axis=-1)
            truth_ok &= np.isfinite(self.truth[eligible + 1, 0, :4]).all(axis=-1)
            state_ok = np.isfinite(self.state[eligible, 0, :2]).all(axis=-1)
            ends = eligible[truth_ok & state_ok][::stride]
            self.points = np.column_stack((ends, np.zeros(len(ends), dtype=np.int32))).astype(np.int32)
            distance = np.linalg.norm(self.state[:, 1:, :2] - self.state[:, :1, :2], axis=-1)
            nearest = np.nan_to_num(distance, nan=np.inf).argmin(axis=-1) + 1
            if self_id > 11:
                nearest = (nearest + 10) % 22 + 1
            self.owner = np.where(self.controlled, nearest, 0)
            bounds = path / "ball_bounds.npy"
            self.bounds = np.load(bounds, mmap_mode="r", allow_pickle=False) if bounds.is_file() else None
        else:
            valid = np.isfinite(self.truth[eligible, 1:, 2:5]).all(axis=-1)
            valid &= np.isfinite(self.state[eligible, 1:, :2]).all(axis=-1)
            times, tracks = np.nonzero(valid)
            self.points = np.column_stack((eligible[times], tracks + 1))[::stride].astype(np.int32)
            self.owner = np.zeros(len(cycles), dtype=np.int32)

    def batch(self, points, ball):
        end, track = points.T
        history = end[:, None] + np.arange(1 - HISTORY, 1)
        x = self.inputs[history, track[:, None]].copy()
        context = self.context[history, track[:, None]].astype(np.float32)
        stock = self.state[end, track]
        truth = self.truth[end, track]
        vel = np.nan_to_num(stock[:, 2:4])
        if ball:
            pos = stock[:, :2]
            baseline = np.column_stack((pos, vel, pos + vel, self.decay * vel))
            target = np.column_stack((truth[:, :4], self.truth[end + 1, track, :4]))
            impulse = np.linalg.norm(target[:, 6:8] - self.decay * target[:, 2:4], axis=-1) > .2
            tags = np.column_stack((self.controlled[end], self.true_control[end], impulse))
        else:
            direction = stock[:, 4]
            direction = np.where(np.isfinite(direction), direction, np.rad2deg(np.arctan2(vel[:, 1], vel[:, 0])))
            baseline = np.column_stack((vel, unit(direction)))
            target = np.column_stack((truth[:, 2:4], unit(truth[:, 4])))
            tags = np.column_stack((np.isfinite(stock[:, 2:4]).all(axis=-1), np.isfinite(stock[:, 4]),
                                    x[:, -1, 17] > .5, x[:, -1, 18] > .5))
        geometry = self.bounds[end] if ball and self.bounds is not None else np.zeros((len(end), 3), np.float32)
        return x, context, self.owner[history], baseline.astype(np.float32), target.astype(np.float32), geometry, tags


def load_data(root, split, ball, stride):
    manifest = json.loads((root / "manifest.json").read_text())["games"]
    return [Stream(root / f'{g["name"]}-{side}', ball, stride)
            for g in manifest if g["split"] == split for side in "lr"]


class Model(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.ball = VARIANTS[variant] == "ball"
        self.quantized = variant == "ball_quantized"
        self.use_owner = variant != "ball_context_no_owner"
        self.context = variant.endswith("context") or variant in {"ball_experts", "ball_quantized", "ball_context_no_owner"}
        self.experts = variant == "ball_experts"
        if self.context:
            self.neighbor = nn.Sequential(nn.Linear(CONTEXT // 4, 16), nn.SiLU(), nn.Linear(16, 12))
        if self.ball and self.context:
            self.owner = nn.Embedding(23, 4, padding_idx=0)
        self.gru = nn.GRU(FEATURES + 26 * self.context + 4 * (self.ball and self.context), 64, batch_first=True)
        out = 8 if self.ball else 3
        self.readout = nn.Sequential(nn.Linear(64 + 3 * self.quantized, 48), nn.SiLU(), nn.Linear(48, out * (2 if self.experts else 1)))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, x, context, owner, base, control, geometry):
        inputs = [x]
        if self.context:
            slots = context.unflatten(-1, (2, 2, CONTEXT // 4))
            available = slots[..., 7:8]
            count = available.sum(dim=-2)
            pool = (self.neighbor(slots) * available).sum(dim=-2) / count.clamp_min(1)
            inputs.append(torch.cat((pool, count / 2), dim=-1).flatten(-2))
            if self.ball:
                inputs.append(self.owner(owner if self.use_owner else torch.zeros_like(owner)))
        state, _ = self.gru(torch.cat(inputs, dim=-1))
        hidden = state[:, -1]
        if self.quantized:
            origin = x[:, -1, 19:21] * 55
            face = torch.atan2(x[:, -1, 24], x[:, -1, 23])
            angle = face + torch.atan2(x[:, -1, 14], x[:, -1, 13])
            low, high, available = geometry.unbind(dim=-1)
            width = (high - low).clamp_min(.01)
            anchor = torch.linalg.vector_norm(base[:, :2] - origin, dim=-1).clamp(low, high)
            hidden = torch.cat((hidden, torch.stack((width / 3, (anchor - low) / width, available), dim=-1)), dim=1)
        delta = self.readout(hidden)
        if self.experts:
            delta = delta.reshape(-1, 2, 8)[torch.arange(len(delta)), control.long()]
        if self.ball:
            bounds = base.new_tensor([4., 4., 2., 2., 5., 5., 3., 3.])
            corrected = base + bounds * delta.tanh()
            if self.quantized:
                radius = (anchor + width * delta[:, 0].tanh()).clamp(low, high)
                angle = angle + (torch.pi / 360) * delta[:, 1].tanh()
                polar = origin + radius[:, None] * torch.stack((angle.cos(), angle.sin()), dim=1)
                corrected = torch.cat((torch.where(available[:, None] > 0, polar, corrected[:, :2]),
                                       corrected[:, 2:]), dim=1)
            return corrected
        angle = torch.atan2(base[:, 3], base[:, 2]) + torch.pi * delta[:, 2].tanh()
        return torch.cat((base[:, :2] + 1.5 * delta[:, :2].tanh(),
                          torch.stack((angle.cos(), angle.sin()), dim=1)), dim=1)


def predict(model, batch):
    x, context, owner, baseline, target, geometry, tags = batch
    return model(*map(torch.from_numpy, (x, context, owner, baseline, tags[:, 0].copy(), geometry))), torch.from_numpy(target)


def error(prediction, target, ball):
    p = prediction.reshape(len(target), -1, 2)
    t = target.reshape(len(target), -1, 2)
    values = torch.linalg.vector_norm(p - t, dim=-1)
    if not ball:
        values[:, 1] = torch.rad2deg(2 * torch.asin((values[:, 1] / 2).clamp(max=1)))
    return values


def loss(pred, target, ball):
    if ball:
        return F.smooth_l1_loss(pred, target, beta=.2)
    velocity = F.smooth_l1_loss(pred[:, :2], target[:, :2], beta=.1)
    body = (1 - (pred[:, 2:] * target[:, 2:]).sum(dim=1)).mean()
    return velocity + .3 * body


def normalized(errors, ball):
    scales = errors.new_tensor([.12, .15, .22, .25] if ball else [.15, 15.])
    return (errors / scales).mean()


def train(args):
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    ball = VARIANTS[args.train] == "ball"
    training = load_data(args.dataset, "train", ball, 1)
    validation = load_data(args.dataset, "validation", ball, 32 if not ball else 6)
    model = Model(args.train)
    optim = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    weights = np.array([len(s.points) for s in training])
    best = float("inf")
    epochs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for i in rng.choice(len(training), args.batches, p=weights / weights.sum()):
            stream = training[i]
            indices = rng.integers(len(stream.points), size=args.batch)
            prediction, truth = predict(model, stream.batch(stream.points[indices], ball))
            value = loss(prediction, truth, ball)
            optim.zero_grad()
            value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optim.step()
        model.eval()
        errors = []
        with torch.inference_mode():
            for stream in validation:
                for start in range(0, len(stream.points), 1024):
                    pred, truth = predict(model, stream.batch(stream.points[start:start+1024], ball))
                    errors.append(error(pred, truth, ball))
        metrics = torch.cat(errors)
        score = float(normalized(metrics, ball))
        epochs.append({"epoch": epoch, "selection_score": score, "mean": metrics.mean(dim=0).tolist()})
        print(args.train, epoch, epochs[-1], flush=True)
        if score < best:
            best = score
            torch.save(model.state_dict(), args.output / f"{args.train}.pt")
        (args.output / f"{args.train}-training.json").write_text(json.dumps(epochs, indent=2) + "\n")


@torch.inference_mode()
def evaluate(args, split, ball, names, body_policy=None):
    data = load_data(args.dataset, split, ball, 1 if split == "test" else (16 if not ball else 3))
    models = {}
    for name in names:
        model = Model(name).eval()
        model.load_state_dict(torch.load(args.output / f"{name}.pt", weights_only=True))
        models[name] = model
    collected = {name: [] for name in ("stock", *names)}
    flags = []
    events = {name: [] for name in collected} if ball else None
    for i, stream in enumerate(data):
        for start in range(0, len(stream.points), 1024):
            batch = stream.batch(stream.points[start:start+1024], ball)
            truth = torch.from_numpy(batch[4])
            stock = torch.from_numpy(batch[3])
            collected["stock"].append(error(stock, truth, ball).numpy())
            if ball:
                events["stock"].append(torch.linalg.vector_norm(stock[:, 6:8] - stream.decay * stock[:, 2:4], dim=1).numpy() > .2)
            for name, model in models.items():
                pred, _ = predict(model, batch)
                collected[name].append(error(pred, truth, ball).numpy())
                if ball:
                    events[name].append(torch.linalg.vector_norm(pred[:, 6:8] - stream.decay * pred[:, 2:4], dim=1).numpy() > .2)
            flags.append(batch[-1])
        if (i + 1) % 10 == 0:
            print(f"{split} {'ball' if ball else 'player'}: {i+1}/{len(data)} observers", flush=True)
    t = np.concatenate(flags)
    masks = {"all": np.ones(len(t), bool)}
    if ball:
        masks.update(estimated_control=t[:, 0], estimated_free=~t[:, 0],
                     truth_in_kicking_range=t[:, 1], truth_free=~t[:, 1], next_impulse=t[:, 2],
                     next_coasting=~t[:, 2], controlled_next_impulse=t[:, 0] & t[:, 2],
                     free_next_impulse=~t[:, 0] & t[:, 2])
    else:
        masks.update(stock_velocity=t[:, 0], stock_body=t[:, 1],
                     raw_motion=t[:, 2], no_raw_motion=~t[:, 2], raw_body=t[:, 3], no_raw_body=~t[:, 3])
    scores = {name: np.concatenate(values) for name, values in collected.items()}
    if not ball:
        if body_policy is None:
            body_policy = {group: min(("stock", "player_context"),
                                      key=lambda key: scores[key][masks[group], 1].mean())
                           for group in ("raw_body", "no_raw_body")}
        gated = scores["player_context"].copy()
        for group, source in body_policy.items():
            gated[masks[group], 1] = scores[source][masks[group], 1]
        scores["player_context_gated"] = gated
    result = {}
    for name, a in scores.items():
        if not np.isfinite(a).all():
            raise ValueError(f"{split}/{name}: invalid error")
        result[name] = {key: {"n": int(mask.sum()), "mean": a[mask].mean(axis=0, dtype=np.float64).tolist(),
                              "p90": np.quantile(a[mask], .9, axis=0).tolist()}
                        for key, mask in masks.items() if mask.any()}
    event_metrics = {}
    if ball:
        event_metrics["control_proxy"] = {
            "estimated_control": int(t[:, 0].sum()), "truth_in_kicking_range": int(t[:, 1].sum()),
            "both": int((t[:, 0] & t[:, 1]).sum()), "n": len(t),
        }
        for name, batches in events.items():
            predicted = np.concatenate(batches)
            actual = t[:, 2]
            event_metrics[name] = {"predicted_impulse": int(predicted.sum()),
                                   "true_impulse": int(actual.sum()),
                                   "correct_impulse": int((predicted & actual).sum())}
    return result, event_metrics, body_policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", choices=VARIANTS)
    mode.add_argument("--evaluate", choices=("player", "ball"))
    parser.add_argument("--include-quantized", action="store_true")
    parser.add_argument("--include-owner-ablation", action="store_true")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batches", type=int, default=192)
    parser.add_argument("--batch", type=int, default=192)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.train:
        train(args)
    else:
        names = [name for name, target in VARIANTS.items() if target == args.evaluate
                 and (name != "ball_quantized" or args.include_quantized)
                 and (name != "ball_context_no_owner" or args.include_owner_ablation)]
        ball = args.evaluate == "ball"
        validation, validation_events, body_policy = evaluate(args, "validation", ball, names)
        winner = min(validation, key=lambda m: float(normalized(torch.tensor(validation[m]["all"]["mean"]), ball)))
        print("Validation selection", winner, flush=True)
        test, test_events, _ = evaluate(args, "test", ball, names, body_policy)
        result = {"metric_order": METRICS[args.evaluate], "winner": winner, "validation": validation, "test": test}
        if body_policy is not None:
            result["body_policy"] = body_policy
        if ball:
            result["event_metrics"] = {"validation": validation_events, "test": test_events}
        (args.output / f"{args.evaluate}-metrics.json").write_text(json.dumps(result, indent=2) + "\n")
