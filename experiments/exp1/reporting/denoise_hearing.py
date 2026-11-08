"""Compare decoded teammate position reports with same-target local references."""

from functools import lru_cache

import numpy as np
import pandas as pd

# Fixed packet lengths from librcsc/rcsc/common/say_message_parser.h.
PACKET_LENGTHS = {
    "b": 6, "p": 10, "i": 3, "g": 5, "e": 8, "o": 2, "d": 2, "w": 1, "F": 2, "h": 4,
    "s": 2, "r": 2, "c": 2, "D": 4, "G": 10, "P": 4, "Q": 7, "R": 10, "S": 5, "T": 5,
    "O": 5, "B": 10,
}
MESSAGE_NAMES = {
    "b": "Ball", "p": "Pass", "B": "Ball + player", "G": "Ball + goalie",
    "P": "One player", "Q": "Two players", "R": "Three players",
    "T": "Teammate", "O": "Opponent", "S": "Sender position",
    "g": "Goalie", "e": "Goalie + player",
}
MESSAGE_TYPES = tuple(MESSAGE_NAMES.values())
TYPE_CODES = {tag: i for i, tag in enumerate(MESSAGE_NAMES)}
FAMILIES = ("bpBG", "BPQRTOSe", "gGe")
TARGETS = ("Ball", "Teammate", "Opponent", "Opponent goalie")
CONTEXTS = ("Fresh visual", "Unseen / 1–2", "Unseen / 3–5", "Unseen / 6+", "Unseen / unknown age")
REFERENCES = ("Fresh visual", "Prior-cycle baseline")
DISTANCES = ("Under 10 m", "10 to under 40 m", "40 m and farther", "Unknown distance")
MATERIAL_DIFFERENCE_M = 0.05
HEARING_COLUMNS = ("sender_unum", "raw_msg", "decoded")
PAYLOAD_METRICS = {
    "message_pos_error": ("Decoded position error", "m"),
    "message_ball_vel_error": ("Decoded ball velocity error", "m/cycle"),
}
CANDIDATE_METRICS = {
    "candidate_pos_error": ("Heard position error", "m"),
    "reference_pos_error": ("Local reference error", "m"),
    "heard_better": ("Heard position closer by more than 5 cm", "%"),
    "heard_worse": ("Heard position farther off by more than 5 cm", "%"),
}
POSITION_ITEMS = r"(?:^|;)player\((?P<unum>\d+) (?P<x>[-+.\deE]+) (?P<y>[-+.\deE]+) [^)]*\)"
CYCLE_STRIDE = 2**20
OBJECT_STRIDE = 32


@lru_cache(maxsize=32768)
def packet_types(packet):
    """Return one unambiguous spatial payload type per family."""
    tags = []
    offset = 0
    while offset < len(packet):
        tag = packet[offset]
        length = PACKET_LENGTHS.get(tag)
        if length is None or offset + length > len(packet):
            return (-1, -1, -1)
        tags.append(tag)
        offset += length
    result = []
    for family in FAMILIES:
        matches = [tag for tag in tags if tag in family]
        result.append(TYPE_CODES[matches[0]] if len(matches) == 1 else -1)
    return tuple(result)


def candidates(heard, objects, object_keys, side):
    tags = np.array([packet_types(packet) for packet in heard["raw_msg"]], dtype=int).reshape(-1, 3)
    ball = heard.assign(target_id=0, message=tags[:, 0], reported_error=heard["n_ball_pos_d"],
                        ball_velocity_error=heard["n_ball_vel_d"])
    ball = ball[(ball["message"] >= 0) & ball["reported_error"].notna()]

    goalie_mask = (objects["object_id"] != 0) & (objects["is_goalie"] == 1) & (objects["is_teammate"] == 0)
    goalies = objects.loc[goalie_mask, ["object_id"]].assign(key=object_keys[goalie_mask])
    goalies = goalies.drop_duplicates("key", keep=False).set_index("key")["object_id"]
    goalie = heard.assign(target_id=heard["key"].map(goalies), message=tags[:, 2],
                          reported_error=heard["n_goalie_pos_d"], is_goalie_payload=True)
    goalie = goalie[(goalie["message"] >= 0) & goalie["target_id"].notna() & goalie["reported_error"].notna()]

    player_types = heard.assign(message=tags[:, 1])
    player_types = player_types[player_types["message"] >= 0]
    players = player_types["decoded"].str.extractall(POSITION_ITEMS).droplevel("match").astype(float)
    unum = players["unum"]
    players["target_id"] = unum if side == "l" else np.where(unum <= 11, unum + 11, unum - 11)
    players = players[(unum >= 1) & (unum <= 22)].join(player_types[["message", "key"]])
    return pd.concat([
        ball[["key", "target_id", "message", "reported_error", "ball_velocity_error"]],
        goalie[["key", "target_id", "message", "reported_error", "is_goalie_payload"]],
        players[["key", "target_id", "message", "x", "y"]],
    ], ignore_index=True)


def summarize_hearing(summary, agent, objects, object_keys, heard, side):
    # The logger's decoded string aggregates by sender and time. If that
    # sender has multiple packets in a cycle we cannot attribute items reliably.
    heard = heard.drop_duplicates(["key", "sender_unum"], keep=False).reset_index(drop=True)
    reports = candidates(heard, objects, object_keys, side)
    index = pd.Index(object_keys * OBJECT_STRIDE + objects["object_id"].to_numpy(dtype=np.int64))
    keys = reports["key"].to_numpy(dtype=np.int64) * OBJECT_STRIDE + reports["target_id"].to_numpy(dtype=np.int64)
    positions = index.get_indexer(keys)
    valid = positions >= 0
    reports, positions, keys = reports.loc[valid], positions[valid], keys[valid]
    current = objects.iloc[positions]
    target = np.select([current["object_id"] == 0, current["is_teammate"] == 1,
                        reports["is_goalie_payload"].eq(True)], [0, 1, 3], default=2)
    message = reports["message"].to_numpy(dtype=int)
    truth_x, truth_y = (current[f"tru_pos_{axis}"].to_numpy() for axis in ("x", "y"))
    decoded_error = np.hypot(reports["x"].to_numpy() - truth_x, reports["y"].to_numpy() - truth_y)
    heard_error = np.where(reports["reported_error"].notna(), reports["reported_error"], decoded_error)
    summary.add("heard_message_payload", {"message_pos_error": heard_error,
                                          "message_ball_vel_error": reports["ball_velocity_error"].to_numpy()},
                target, message)

    seen = current["seen"].to_numpy() == 1
    reliable_visual = (target == 0) | current["match_method"].eq("unum").to_numpy()
    reference_error = np.where(seen & reliable_visual, current["n_ls_pos_d"], np.nan)
    # The prior-cycle baselines were available before this message arrived.
    # Earlier teammate reports may already be part of those prior states.
    previous_rows = index.get_indexer(keys - CYCLE_STRIDE * OBJECT_STRIDE)
    has_previous = (previous_rows >= 0) & (current["stopped"].to_numpy() == 0) & ~seen
    stale = np.flatnonzero(has_previous)
    previous = objects.iloc[previous_rows[stale]]
    ball = target[stale] == 0
    prior_agent = agent.reindex(reports["key"].to_numpy(dtype=np.int64)[stale] - CYCLE_STRIDE)
    prior_x, prior_y = (
        np.where(ball, prior_agent[f"next_ball_pos_{axis}"], previous[f"trk_pos_{axis}"])
        for axis in ("x", "y")
    )
    known = previous["n_trk_pos_d"].notna().to_numpy() & (ball | previous["track_match"].eq("unum").to_numpy())
    stale_reference = np.hypot(prior_x - truth_x[stale], prior_y - truth_y[stale])
    reference_error[stale] = np.where(known, stale_reference, np.nan)
    compare = np.isfinite(heard_error) & np.isfinite(reference_error)
    delta = heard_error[compare] - reference_error[compare]
    compared = {
        "candidate_pos_error": heard_error[compare],
        "reference_pos_error": reference_error[compare],
        "heard_better": 100 * (delta < -MATERIAL_DIFFERENCE_M),
        "heard_worse": 100 * (delta > MATERIAL_DIFFERENCE_M),
    }
    age = current["trk_seen_pos_count"].to_numpy()
    context = np.select([seen, (age >= 1) & (age < 3), (age >= 3) & (age < 6),
                         (age >= 6) & (age < 1000)], [0, 1, 2, 3], default=4)
    distance = current["true_dist"].to_numpy()
    distance_band = np.select([distance < 10, distance < 40, np.isfinite(distance)], [0, 1, 2], default=3)
    summary.add("heard_candidate_context", compared, target[compare], context[compare])
    summary.add("heard_candidate_distance", compared, target[compare], context[compare], distance_band[compare])
    summary.add("heard_candidate_message", compared, target[compare], (~seen[compare]).astype(int), message[compare])

