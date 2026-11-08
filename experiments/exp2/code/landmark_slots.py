"""Memory-mapped fixed slots for librcsc MarkerID values 0 through 54."""

import json
from pathlib import Path

import numpy as np

LANDMARK_COUNT = 55  # 53 flags and two goals in librcsc/rcsc/types.h.
LANDMARK_VERSION = 1


class LandmarkSlots:
    def __init__(self, root, store):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if (manifest["schema_version"] != LANDMARK_VERSION
                or manifest["landmarks"] != LANDMARK_COUNT
                or manifest["dataset_fingerprint"] != store.fingerprint):
            raise ValueError("Landmark slots do not match the dataset")
        self.files = {entry["file"]: entry["rows"] for entry in manifest["games"]}
        self.arrays = {}
        expected = {entry["file"]: entry["rows"] for entry in store.manifest["games"]
                    if entry["split"] in manifest["splits"]}
        if self.files != expected:
            raise ValueError("Landmark slots do not cover the indexed splits")

    def rows(self, file, start, end):
        if file not in self.arrays:
            if file not in self.files:
                raise ValueError(f"No landmark slots for {file}")
            array = np.load(self.root / file, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.float32 or array.shape != (self.files[file], LANDMARK_COUNT, 2):
                raise ValueError(f"Unexpected landmark slot format in {file}")
            self.arrays[file] = array
        return self.arrays[file][start:end]
