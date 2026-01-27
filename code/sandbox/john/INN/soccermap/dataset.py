from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .channels import create_13_channels, compute_player_velocities
from .config import GridSpec


@dataclass(frozen=True)
class PassSample:
    event_id: str
    channels: torch.Tensor         # (13,L,W)
    dest_index: int                # for selection model (flattened index)
    completed: Optional[int]       # 0/1 for success model
    dest_lw: Tuple[int, int]       # (l_idx,w_idx)


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
        compute_velocities: bool = False,
        max_time_gap: float = 5.0,
        max_match_distance: float = 15.0,
    ):
        self.expanded_df = expanded_df
        self.grid = grid
        self.compute_velocities = compute_velocities
        self.max_time_gap = max_time_gap
        self.max_match_distance = max_match_distance

        # one row per event (actor rows)
        actor = expanded_df[expanded_df["actor"] == True].copy()
        if only_passes:
            actor = actor[actor["event_type"] == "Pass"]
        # must have end_location
        actor = actor[actor["end_location"].notna()]
        self.actor_events = actor.sort_values("total_seconds").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.actor_events)

    def __getitem__(self, idx: int) -> PassSample:
        row = self.actor_events.iloc[idx]
        eid = str(row["event_id"])

        # compute velocities using previous pass event (or previous event with 360 if you prefer)
        vel = None
        if self.compute_velocities and idx > 0:
            prev_eid = str(self.actor_events.iloc[idx - 1]["event_id"])
            vel = compute_player_velocities(
                self.expanded_df, eid, previous_event_id=prev_eid,
                max_time_gap=self.max_time_gap,
                max_match_distance=self.max_match_distance,
            )

        chans = create_13_channels(self.expanded_df, eid, self.grid, velocity_dict=vel)
        if chans is None:
            # fallback: return a zero sample (rare), but keep deterministic
            chans = np.zeros((13, self.grid.L, self.grid.W), dtype=np.float32)

        # destination index
        end_loc = row["end_location"]
        l_idx, w_idx, _, _ = self.grid.sb_to_grid(float(end_loc[0]), float(end_loc[1]))
        dest_index = int(l_idx * self.grid.W + w_idx)

        completed = row.get("pass_completed")
        completed = None if pd.isna(completed) else int(completed)

        return PassSample(
            event_id=eid,
            channels=torch.tensor(chans, dtype=torch.float32),
            dest_index=dest_index,
            completed=completed,
            dest_lw=(int(l_idx), int(w_idx)),
        )
