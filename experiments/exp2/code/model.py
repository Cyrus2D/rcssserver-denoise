"""Differentiable motion model and optional recurrent visual-fusion gates."""

import math

import torch
from torch import nn

from schema import FEATURES, FEATURE_INDEX as F

POSITION_GAIN = 0.80
FACE_GAIN = 0.95
INPUT_SCALES = {
    "dash_power": 100, "dash_dir": math.pi, "turn": math.pi, "neck_turn": math.pi,
    "speed": 1.2, "speed_dir": math.pi, "neck": math.pi / 2,
    "fix_x": 55, "fix_y": 35, "fix_angle": math.pi,
    "fix_err_x": 0.10, "fix_err_y": 0.10, "fix_err_angle": 0.02,
    "pos_age": 5, "face_age": 5, "land_count": 20, "land_mean_dist": 60,
    "land_min_dist": 40, "land_span": 60, "line_count": 2,
}
LANDMARK_EMBEDDING = 16
CIRCULAR_INPUTS = ("dash_dir", "speed_dir", "fix_angle")
CIRCULAR_MASKS = ("dash", "has_speed", "fix_face")
LANDMARK_SUMMARY = ("land_count", "land_mean_dist", "land_min_dist", "land_bearing_spread", "land_span", "line_count")
MAX_POSITION_RESIDUAL = 0.10


def wrap(angle):
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


def unit_vector(angle):
    return torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)


def limit_norm(vector, limit):
    length = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return vector * (limit / length.clamp(min=1e-6)).clamp(max=1)


class Tracker(nn.Module):
    """State is x, y, vx, vy, body, neck. Angles are radians."""

    def __init__(self, physics, hidden_size=48, landmark_count=0, *,
                 circular_angles=False, position_residual=False, landmark_summary=True):
        super().__init__()
        self.physics = dict(physics)
        self.hidden_size = hidden_size
        self.landmark_count = landmark_count
        self.circular_angles = circular_angles
        self.landmark_summary = landmark_summary
        self.position_residual = position_residual
        self.register_buffer("input_scales", torch.tensor([INPUT_SCALES.get(name, 1.0) for name in FEATURES]))
        self.register_buffer("fixed_gains", torch.tensor([POSITION_GAIN, FACE_GAIN]))
        if hidden_size:
            # Six prior-state features plus three visual innovations.
            landmark_features = LANDMARK_EMBEDDING if landmark_count else 0
            circular_features = len(CIRCULAR_INPUTS) if circular_angles else 0
            self.gru = nn.GRUCell(len(FEATURES) + 9 + landmark_features + circular_features, hidden_size)
            self.head = nn.Linear(hidden_size, 2)
            nn.init.zeros_(self.head.weight)
            with torch.no_grad():
                self.head.bias.copy_(torch.logit(self.fixed_gains))
        if landmark_count:
            self.landmark_encoder = nn.Linear(landmark_count * 4, LANDMARK_EMBEDDING, bias=False)
        if position_residual:
            self.offset_head = nn.Linear(hidden_size, 2)
            nn.init.zeros_(self.offset_head.weight)
            nn.init.zeros_(self.offset_head.bias)

    def initial_hidden(self, state):
        return state.new_zeros((len(state), self.hidden_size))

    def network_inputs(self, inputs):
        features = inputs / self.input_scales
        if not self.landmark_summary:
            features[:, [F[name] for name in LANDMARK_SUMMARY]] = 0
        if self.circular_angles:
            indices = [F[name] for name in CIRCULAR_INPUTS]
            angles = inputs[:, indices]
            mask = inputs[:, [F[name] for name in CIRCULAR_MASKS]]
            features[:, indices] = angles.sin() * mask
            features = torch.cat((features, angles.cos() * mask), dim=-1)
        return features

    def step(self, inputs, state, hidden, motion=None, landmark_embedding=None, return_prior=False):
        p = self.physics
        inertia, dash_rate, decay, speed_max = (p[name] for name in ("inertia", "dash_rate", "decay", "speed_max"))
        if motion is not None:
            inertia, dash_rate, decay, speed_max = motion.unbind(dim=-1)
            decay, speed_max = decay[:, None], speed_max[:, None]
        old_position, old_velocity = state[:, :2], state[:, 2:4]
        old_body, old_neck = state[:, 4], state[:, 5]
        old_speed = torch.linalg.vector_norm(old_velocity, dim=-1)
        turn = inputs[:, F["turn"]] / (1 + inertia * old_speed)
        body = wrap(old_body + turn)
        neck = (old_neck + inputs[:, F["neck_turn"]]).clamp(
            math.radians(p["min_neck"]), math.radians(p["max_neck"]),
        )
        neck = torch.where(inputs[:, F["has_neck"]] > 0.5, inputs[:, F["neck"]], neck)
        face = wrap(body + neck)

        dash_dir = inputs[:, F["dash_dir"]]
        side_fraction = dash_dir.abs().clamp(max=math.pi) / (math.pi / 2)
        directional_rate = torch.where(
            side_fraction <= 1,
            1 + (p["side_dash_rate"] - 1) * side_fraction,
            p["side_dash_rate"] + (p["back_dash_rate"] - p["side_dash_rate"]) * (side_fraction - 1),
        )
        dash = inputs[:, F["dash_power"]] * inputs[:, F["effort"]] * dash_rate * directional_rate
        accel = limit_norm(dash[:, None] * unit_vector(old_body + dash_dir), p["accel_max"])
        move = limit_norm(old_velocity + accel, speed_max)
        position = old_position + move
        predicted_position = position
        velocity = move * decay

        has_position = inputs[:, F["fix_pos"]] > 0.5
        has_face = inputs[:, F["fix_face"]] > 0.5
        fix_position = inputs[:, [F["fix_x"], F["fix_y"]]]
        position_delta = torch.where(has_position[:, None], fix_position - position, 0)
        face_delta = torch.where(has_face, wrap(inputs[:, F["fix_angle"]] - face), 0)
        if self.hidden_size:
            prior = torch.cat((position / 55, velocity / 1.2, unit_vector(face)), dim=-1)
            innovations = torch.cat((position_delta / 0.10, face_delta[:, None] / 0.02), dim=-1)
            parts = (self.network_inputs(inputs), prior, innovations)
            if landmark_embedding is not None:
                parts += (landmark_embedding,)
            features = torch.cat(parts, dim=-1).clamp(-12, 12)
            hidden = self.gru(features, hidden)
            gains = self.head(hidden).sigmoid()
        else:
            gains = self.fixed_gains.expand(len(state), 2)

        position = position + gains[:, :1] * position_delta
        if self.position_residual:
            local_offset = limit_norm(self.offset_head(hidden).tanh(), 1) * MAX_POSITION_RESIDUAL
            forward = unit_vector(body)
            left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
            offset = local_offset[:, :1] * forward + local_offset[:, 1:] * left
            position = position + torch.where(has_position[:, None], offset, 0)
        face = wrap(face + gains[:, 1] * face_delta)
        body = wrap(face - neck)
        sensed_velocity = inputs[:, F["speed"], None] * unit_vector(face + inputs[:, F["speed_dir"]])
        velocity = torch.where((inputs[:, F["has_speed"]] > 0.5)[:, None], sensed_velocity, velocity)
        state = torch.cat((position, velocity, body[:, None], neck[:, None]), dim=-1)
        if return_prior:
            return state, hidden, gains, predicted_position
        return state, hidden, gains

    def landmark_features(self, inputs, landmarks):
        if landmarks is None:
            return inputs.new_zeros((*inputs.shape[:2], LANDMARK_EMBEDDING))
        distance, bearing = landmarks.unbind(dim=-1)
        visible = distance >= 0
        values = torch.stack((visible.to(distance.dtype), distance / 60,
                              bearing.sin(), bearing.cos()), dim=-1)
        values = torch.where(visible[..., None], values, 0)
        return self.landmark_encoder(values.flatten(start_dim=-2)).tanh()

    def forward(self, inputs, state, reference=None, motion=None, landmarks=None, return_prior=False):
        """Roll out, optionally retaining librcsc velocity and orientation each cycle."""
        hidden = self.initial_hidden(state)
        embeddings = self.landmark_features(inputs, landmarks) if self.landmark_count else None
        states, gains = [], []
        priors = []
        for step, current in enumerate(inputs.unbind(dim=1)):
            parameters = motion[:, step] if motion is not None else None
            embedding = embeddings[:, step] if embeddings is not None else None
            result = self.step(current, state, hidden, parameters, embedding, return_prior)
            state, hidden, weights = result[:3]
            if return_prior:
                priors.append(result[3])
            if reference is not None:
                measured = reference[:, step, 2:].reshape(len(state), 2, 2)
                fallback = state[:, 2:].reshape(len(state), 2, 2)
                valid = torch.isfinite(measured).all(dim=-1, keepdim=True)
                retained = torch.where(valid, measured, fallback).flatten(start_dim=1)
                state = torch.cat((state[:, :2], retained), dim=-1)
            states.append(state)
            gains.append(weights)
        rollout = torch.stack(states, dim=1), torch.stack(gains, dim=1)
        if return_prior:
            return (*rollout, torch.stack(priors, dim=1))
        return rollout
