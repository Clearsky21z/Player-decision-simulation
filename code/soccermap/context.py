from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_CONTEXT_DIM = 8
DEFAULT_CONTEXT_FEATURE_NAMES = (
    "team_score",
    "opponent_score",
    "score_diff",
    "match_time",
    "phase_early",
    "phase_mid",
    "phase_late",
    "phase_extra",
)


def context_feature_names(context_dim: int = DEFAULT_CONTEXT_DIM) -> list[str]:
    names = list(DEFAULT_CONTEXT_FEATURE_NAMES[:context_dim])
    if context_dim > len(DEFAULT_CONTEXT_FEATURE_NAMES):
        names.extend(
            f"context_extra_{idx}"
            for idx in range(len(DEFAULT_CONTEXT_FEATURE_NAMES), context_dim)
        )
    return names


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _encode_match_phase(minute: float) -> np.ndarray:
    phase = np.zeros(4, dtype=np.float32)
    if minute < 30.0:
        phase[0] = 1.0
    elif minute < 60.0:
        phase[1] = 1.0
    elif minute < 90.0:
        phase[2] = 1.0
    else:
        phase[3] = 1.0
    return phase


def build_context_features(
    event_slice: Optional[pd.DataFrame],
    *,
    actor_row: Optional[pd.Series] = None,
    context_dim: int = DEFAULT_CONTEXT_DIM,
) -> np.ndarray:
    """
    Build a compact, non-spatial game-context vector for late fusion.

    The first 8 slots are pure game-state features:
      - scoreline from the acting team's perspective (2)
      - score differential (1)
      - normalized match time (1)
      - match phase one-hot (4)

    If ``context_dim`` is smaller we truncate; if larger we zero-pad.
    """
    if context_dim <= 0:
        return np.zeros((0,), dtype=np.float32)

    base = np.zeros(DEFAULT_CONTEXT_DIM, dtype=np.float32)
    if event_slice is None or event_slice.empty:
        return _resize_context(base, context_dim)

    if actor_row is None:
        actor = event_slice.loc[event_slice["actor"] == True]
        actor_row = None if actor.empty else actor.iloc[0]

    if actor_row is None:
        return _resize_context(base, context_dim)

    team_score = _coerce_float(actor_row.get("team_score"))
    opponent_score = _coerce_float(actor_row.get("opponent_score"))
    score_diff = _coerce_float(actor_row.get("score_diff"), default=team_score - opponent_score)

    base[0] = min(team_score / 5.0, 1.0)
    base[1] = min(opponent_score / 5.0, 1.0)
    base[2] = float(np.clip(score_diff / 3.0, -1.0, 1.0))

    minute = _coerce_float(actor_row.get("minute"))
    second = _coerce_float(actor_row.get("second"))
    total_seconds = max(0.0, minute * 60.0 + second)
    base[3] = min(total_seconds / (120.0 * 60.0), 1.0)
    base[4:8] = _encode_match_phase(minute)

    return _resize_context(base, context_dim)


def _resize_context(features: np.ndarray, context_dim: int) -> np.ndarray:
    if context_dim == len(features):
        return features
    if context_dim < len(features):
        return features[:context_dim].copy()

    out = np.zeros(context_dim, dtype=np.float32)
    out[: len(features)] = features
    return out