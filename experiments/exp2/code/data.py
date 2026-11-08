"""Read memory-mapped game shards and serve causal training and evaluation sequences."""

import hashlib
import json
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from landmark_slots import LANDMARK_COUNT
from schema import FEATURES, ROW, SCHEMA_VERSION, STATE


@dataclass(frozen=True)
class Segment:
    file: str
    start: int
    end: int

    @property
    def length(self):
        return self.end - self.start


class Store:
    def __init__(self, root):
        self.root = root
        manifest = (root / "manifest.json").read_bytes()
        self.fingerprint = hashlib.sha256(manifest).hexdigest()
        self.manifest = json.loads(manifest)
        if (self.manifest["schema_version"] != SCHEMA_VERSION
                or tuple(self.manifest["features"]) != FEATURES or tuple(self.manifest["state"]) != STATE):
            raise ValueError("The dataset schema does not match this version of exp2")
        self._arrays = {}

    def segments(self, split):
        for game in self.manifest["games"]:
            if game["split"] == split:
                for span in game["segments"]:
                    yield Segment(game["file"], span["start"], span["end"])

    def rows(self, file, start, end):
        if file not in self._arrays:
            array = np.load(self.root / file, mmap_mode="r", allow_pickle=False)
            if array.dtype != ROW:
                raise ValueError(f"Unexpected record format in {file}")
            self._arrays[file] = array
        return self._arrays[file][start:end]


class Windows(Dataset):
    def __init__(self, store, steps, stride):
        self.store, self.steps, self.stride = store, steps, stride
        self.segments, self.ends = [], []
        for segment in store.segments("train"):
            if segment.length > steps:
                count = 1 + (segment.length - steps - 1) // stride
                self.segments.append(segment)
                self.ends.append(count + (self.ends[-1] if self.ends else 0))
        if not self.ends:
            raise ValueError("No training segments are long enough for the selected window")

    def __len__(self):
        return self.ends[-1]

    def window(self, index):
        i = bisect_right(self.ends, index)
        segment = self.segments[i]
        first = self.ends[i - 1] if i else 0
        start = segment.start + (index - first) * self.stride
        return segment.file, start, self.store.rows(segment.file, start, start + self.steps + 1)

    def __getitem__(self, index):
        _, _, rows = self.window(index)
        return tuple(torch.from_numpy(np.array(array, copy=True))
                     for array in (rows["features"][1:], rows["librcsc"][0], rows["truth"][1:]))


def evaluation_batches(store, split, batch_size, burn_in, player_physics=None, landmark_slots=None):
    segments = sorted((s for s in store.segments(split) if s.length > burn_in + 1),
                      key=lambda s: s.length)
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        steps = batch[-1].length - 1
        features = np.zeros((len(batch), steps, len(FEATURES)), dtype=np.float32)
        initial = np.empty((len(batch), len(STATE)), dtype=np.float32)
        truth = np.full((len(batch), steps, len(STATE)), np.nan, dtype=np.float32)
        librcsc = np.full_like(truth, np.nan)
        if player_physics is not None:
            motion = np.broadcast_to(player_physics.base, (len(batch), steps, len(player_physics.base))).copy()
        if landmark_slots is not None:
            landmarks = np.zeros((len(batch), steps, LANDMARK_COUNT, 2), dtype=np.float32)
            landmarks[..., 0] = -1
        for j, segment in enumerate(batch):
            rows = store.rows(segment.file, segment.start, segment.end)
            n = len(rows) - 1
            initial[j] = rows["librcsc"][0]
            features[j, :n] = rows["features"][1:]
            truth[j, :n] = rows["truth"][1:]
            librcsc[j, :n] = rows["librcsc"][1:]
            if player_physics is not None:
                motion[j, :n] = player_physics.rows(segment.file, segment.start + 1, segment.end)
            if landmark_slots is not None:
                landmarks[j, :n] = landmark_slots.rows(segment.file, segment.start + 1, segment.end)
        arrays = (features, initial, truth, librcsc)
        if player_physics is not None:
            arrays += (motion,)
        if landmark_slots is not None:
            arrays += (landmarks,)
        yield tuple(torch.from_numpy(array) for array in arrays)
