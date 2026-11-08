"""Versioned, row-oriented storage shared by the builder and trainer."""

import numpy as np

SCHEMA_VERSION = 1
FEATURES = (
    "dash_power", "dash_dir", "turn", "neck_turn", "effort", "dash", "turn_cmd", "tackle",
    "speed", "speed_dir", "neck", "has_speed", "has_neck", "collision",
    "see", "fix_pos", "fix_face", "fix_x", "fix_y", "fix_angle",
    "fix_err_x", "fix_err_y", "fix_err_angle", "pos_age", "face_age",
    "narrow", "normal", "wide",
    "land_count", "land_mean_dist", "land_min_dist", "land_bearing_spread", "land_span", "line_count",
)
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURES)}
STATE = ("x", "y", "vx", "vy", "body_rad", "neck_rad")
ROW = np.dtype([
    ("cycle", "<i4"),
    ("features", "<f4", (len(FEATURES),)),
    ("truth", "<f4", (len(STATE),)),
    ("librcsc", "<f4", (len(STATE),)),
])
PHYSICS_KEYS = {
    "decay": "player_decay", "speed_max": "player_speed_max",
    "accel_max": "player_accel_max", "dash_rate": "dash_power_rate", "inertia": "inertia_moment",
    "side_dash_rate": "side_dash_rate", "back_dash_rate": "back_dash_rate",
    "min_neck": "minneckang", "max_neck": "maxneckang",
}
