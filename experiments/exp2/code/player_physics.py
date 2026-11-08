"""Versioned sidecar containing the physics known before each transition."""

import json
from pathlib import Path

import numpy as np

MOTION_FIELDS = ("inertia", "dash_rate", "decay", "speed_max")
TYPE_FIELDS = ("inertia_moment", "dash_power_rate", "player_decay", "player_speed_max")
MOTION_ROW = np.dtype([("type_id", "<i2"), ("values", "<f4", (len(MOTION_FIELDS),))])
MOTION_VERSION = 1


class PlayerPhysics:
    def __init__(self, root, store, split):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if (manifest["schema_version"] != MOTION_VERSION
                or tuple(manifest["fields"]) != MOTION_FIELDS
                or manifest["dataset_fingerprint"] != store.fingerprint
                or manifest["split"] != split):
            raise ValueError("Player physics index does not match the dataset, schema, and split")
        self.base = np.array([store.manifest["physics"][name] for name in MOTION_FIELDS], dtype=np.float32)
        self.files = {entry["file"]: entry["rows"] for entry in manifest["games"]}
        self.arrays = {}
        expected = {entry["file"]: entry["rows"] for entry in store.manifest["games"] if entry["split"] == split}
        if self.files != expected:
            raise ValueError("Player physics index does not cover the selected split")

    def rows(self, file, start, end):
        if file not in self.arrays:
            array = np.load(self.root / file, mmap_mode="r", allow_pickle=False)
            if array.dtype != MOTION_ROW or len(array) != self.files[file]:
                raise ValueError(f"Unexpected motion index format in {file}")
            self.arrays[file] = array
        return self.arrays[file]["values"][start:end]
