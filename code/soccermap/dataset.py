from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .channels import create_11_channels
from .config import GridSpec
from .context import DEFAULT_CONTEXT_DIM, build_context_features


@dataclass(frozen=True)
class PassSample:
    event_id: str
    channels: torch.Tensor         # (11,L,W)
    context_features: torch.Tensor # (context_dim,)
    dest_index: int                # for selection model (flattened index)
    completed: Optional[int]       # 0/1 for success model
    dest_lw: Tuple[int, int]       # (l_idx,w_idx)
    actor_player_name: Optional[str] = None  # for player embedding lookup


class PassDataset(Dataset):
    """
    Builds pass samples from expanded_df.

    You can use it for:
      - pass success (BCE at destination pixel)
      - pass selection (CE over all pixels)
    """
    def __init__(
        self,
        expanded_df: pd.DataFrame,
        *,
        grid: GridSpec = GridSpec(),
        only_passes: bool = True,
        team_filter: Optional[str] = None,
        compute_velocities: bool = False,
        max_time_gap: float = 5.0,
        max_match_distance: float = 15.0,
        context_dim: int = DEFAULT_CONTEXT_DIM,
    ):
        self.expanded_df = expanded_df
        self.grid = grid
        self.compute_velocities = compute_velocities
        self.max_time_gap = max_time_gap
        self.max_match_distance = max_match_distance
        self.context_dim = context_dim
        self._event_slices = {
            str(event_id): event_slice
            for event_id, event_slice in expanded_df.groupby("event_id", sort=False)
        }

        # one row per event (actor rows)
        actor = expanded_df[expanded_df["actor"] == True].copy()
        if only_passes:
            actor = actor[actor["event_type"] == "Pass"]
        if team_filter:
            actor = actor[actor["team"] == team_filter]

        # must have end_location
        actor = actor[actor["end_location"].notna()]
        sort_cols = ["total_seconds"]
        if "event_index" in actor.columns:
            sort_cols.append("event_index")
        self.actor_events = actor.sort_values(sort_cols).reset_index(drop=True)

        context_vectors = []
        for _, row in self.actor_events.iterrows():
            event_slice = self._event_slices.get(str(row["event_id"]))
            context_vectors.append(
                build_context_features(
                    event_slice,
                    actor_row=row,
                    context_dim=self.context_dim,
                )
            )
        self.actor_events["context_features"] = context_vectors

    def __len__(self) -> int:
        return len(self.actor_events)

    def __getitem__(self, idx: int) -> PassSample:
        row = self.actor_events.iloc[idx]
        eid = str(row["event_id"])

        # compute velocities using previous pass event (or previous event with 360 if you prefer)
        # Velocity channels have been removed from the active model variant.
        # Keep ``compute_velocities`` in the Dataset API for compatibility, but
        # the emitted channel tensor is now the 11-channel no-velocity version.
        chans = create_11_channels(self.expanded_df, eid, self.grid)
        if chans is None:
            # fallback: return a zero sample (rare), but keep deterministic
            chans = np.zeros((11, self.grid.L, self.grid.W), dtype=np.float32)

        # destination index
        end_loc = row["end_location"]
        l_idx, w_idx, _, _ = self.grid.sb_to_grid(float(end_loc[0]), float(end_loc[1]))
        dest_index = int(l_idx * self.grid.W + w_idx)

        completed = row.get("pass_completed")
        completed = None if pd.isna(completed) else int(completed)

        actor_name = row.get("player_name") if "player_name" in row.index else None
        context_features = row.get("context_features")
        if not isinstance(context_features, np.ndarray):
            context_features = np.zeros((self.context_dim,), dtype=np.float32)

        return PassSample(
            event_id=eid,
            channels=torch.tensor(chans, dtype=torch.float32),
            context_features=torch.tensor(context_features, dtype=torch.float32),
            dest_index=dest_index,
            completed=completed,
            dest_lw=(int(l_idx), int(w_idx)),
            actor_player_name=actor_name,
        )
