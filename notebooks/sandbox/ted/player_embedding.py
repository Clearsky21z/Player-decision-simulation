
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.path import Path as MplPath
from torch.utils.data import Dataset

# ---- Existing project modules (code/ must be on sys.path) ----
from load_data import (
    load_match_data,
    build_events_dataframe,
    load_lineup_dataframe,
    find_seasons_for_competition,
    load_team_matches,
    find_matches_with_360,
)
from feature_engineering import (
    create_player_id_mapping as _create_base_player_mapping,
)
from data_cleaning import validate_location


# =====================================================================
# Section 1 — Grid specification  (from soccermap/config.py)
# =====================================================================

@dataclass(frozen=True)
class GridSpec:
    """Pitch discretisation matching StatsBomb coordinates (120x80)."""
    L: int = 120
    W: int = 80
    sb_L: float = 120.0
    sb_W: float = 80.0

    @property
    def scale_L(self) -> float:
        return self.L / self.sb_L

    @property
    def scale_W(self) -> float:
        return self.W / self.sb_W

    def sb_to_grid(self, x: float, y: float) -> Tuple[int, int, float, float]:
        l = x * self.scale_L
        w = y * self.scale_W
        l_idx = int(np.clip(np.floor(l), 0, self.L - 1))
        w_idx = int(np.clip(np.floor(w), 0, self.W - 1))
        return l_idx, w_idx, float(l), float(w)

    def grid_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        gx = np.arange(self.L, dtype=np.float32)[:, None]
        gy = np.arange(self.W, dtype=np.float32)[None, :]
        gx = np.repeat(gx, self.W, axis=1)
        gy = np.repeat(gy, self.L, axis=0)
        return gx, gy

    def goal_location(self) -> Tuple[float, float]:
        return float(self.L), float(self.W) / 2.0


# =====================================================================
# Section 2 — Player ID mapping  (wraps feature_engineering)
# =====================================================================

def build_player_id_mapping(
    events_df: pd.DataFrame,
    lineup_df: Optional[pd.DataFrame] = None,
) -> Dict[str, int]:
    """
    Build a global  player-name → integer-index  mapping.

    Index **0 is reserved** for unknown / padding.

    Wraps ``feature_engineering.create_player_id_mapping`` (which starts
    at 0) and shifts all indices by +1.  Optionally merges players from
    *lineup_df* so that substitutes who never touched the ball still get
    an embedding slot.

    Parameters
    ----------
    events_df : DataFrame with a ``player`` column  (from ``build_events_dataframe``).
    lineup_df : DataFrame from ``load_lineup_dataframe`` (optional).
    """
    # base mapping from events  {name: 0-based idx}
    base = _create_base_player_mapping(events_df)
    all_names = set(base.keys())

    # add lineup players that may not appear in events
    if lineup_df is not None and "player_name" in lineup_df.columns:
        all_names.update(lineup_df["player_name"].dropna().unique())

    # deterministic ordering, 1-based  (0 = padding)
    sorted_names = sorted(all_names)
    return {name: i + 1 for i, name in enumerate(sorted_names)}


# =====================================================================
# Section 3 — Expand with identity  (adapted from soccermap/expand.py)
#
#   Input:  combined events from  load_match_data()
#           lineup_df   from  load_lineup_dataframe()
# =====================================================================

@dataclass(frozen=True)
class ExpandedMatchWithIdentity:
    event_df: pd.DataFrame
    expanded_df: pd.DataFrame


def _pass_end_location(event: Dict[str, Any]) -> Optional[List[float]]:
    p = event.get("pass")
    if not isinstance(p, dict):
        return None
    end_loc = p.get("end_location")
    if isinstance(end_loc, list) and len(end_loc) >= 2:
        return end_loc[:2]
    return None


def _pass_completed(event: Dict[str, Any]) -> Optional[int]:
    p = event.get("pass")
    if not isinstance(p, dict):
        return None
    return 1 if p.get("outcome") is None else 0


def build_expanded_dfs_with_identity(
    combined: List[Dict[str, Any]],
    lineup_df: pd.DataFrame,
    player_id_mapping: Optional[Dict[str, int]] = None,
    keep_all_events: bool = False,
) -> ExpandedMatchWithIdentity:
    """
    Build expanded DataFrames from *combined* events (output of
    ``load_match_data``), preserving actor identity.

    Each combined event already carries ``freeze_frame`` and
    ``visible_area`` (merged by ``load_match_data``).

    Parameters
    ----------
    combined        : list of event dicts from ``load_match_data()``.
    lineup_df       : DataFrame from ``load_lineup_dataframe()``.
    player_id_mapping : {player_name: int} with 0 = padding.
    keep_all_events : keep events even when freeze-frame is missing.
    """
    # Team names from lineup
    teams = lineup_df["team_name"].unique().tolist() if lineup_df is not None else []
    team1 = teams[0] if len(teams) > 0 else None
    team2 = teams[1] if len(teams) > 1 else None

    if player_id_mapping is None:
        player_id_mapping = {}

    event_rows: List[Dict[str, Any]] = []
    expanded_rows: List[Dict[str, Any]] = []

    for ev in combined:
        ev_id = ev.get("id")
        if ev_id is None:
            continue

        freeze = ev.get("freeze_frame", [])
        visible_area = ev.get("visible_area")
        has_ff = isinstance(freeze, list) and len(freeze) > 0

        if not has_ff and not keep_all_events:
            continue

        team = (ev.get("team") or {}).get("name")
        opp_team = None
        if team1 and team2 and team:
            opp_team = team2 if team == team1 else (team1 if team == team2 else None)

        # Actor identity
        player_info = ev.get("player") or {}
        player_name = player_info.get("name")
        player_sb_id = player_info.get("id")
        player_idx = player_id_mapping.get(player_name, 0) if player_name else 0

        minute = ev.get("minute")
        second = ev.get("second")
        period = ev.get("period")
        event_type = (ev.get("type") or {}).get("name")
        loc = ev.get("location")
        end_loc = _pass_end_location(ev)
        completed = _pass_completed(ev)

        event_rows.append({
            "event_id": ev_id,
            "minute": minute,
            "second": second,
            "period": period,
            "event_type": event_type,
            "team": team,
            "opponent_team": opp_team,
            "event_location": loc,
            "end_location": end_loc,
            "pass_completed": completed,
            "visible_area": visible_area,
            "player_name": player_name,
            "player_sb_id": player_sb_id,
            "player_idx": player_idx,
        })

        # Actor row
        expanded_rows.append({
            "event_id": ev_id,
            "actor": True,
            "teammate": True,
            "keeper": False,
            "team": team,
            "opponent_team": opp_team,
            "minute": minute,
            "second": second,
            "period": period,
            "event_type": event_type,
            "event_location": loc,
            "end_location": end_loc,
            "pass_completed": completed,
            "ff_location": None,
            "ff_idx": None,
            "visible_area": visible_area,
            "player_name": player_name,
            "player_sb_id": player_sb_id,
            "player_idx": player_idx,
        })

        if not has_ff:
            continue

        ff_idx = 0
        for p in freeze:
            if not isinstance(p, dict):
                continue
            if p.get("actor") is True:
                continue
            ploc = p.get("location")
            if not (isinstance(ploc, list) and len(ploc) >= 2):
                continue

            teammate = bool(p.get("teammate"))
            pteam = team if teammate else opp_team

            expanded_rows.append({
                "event_id": ev_id,
                "actor": False,
                "teammate": teammate,
                "keeper": bool(p.get("keeper")),
                "team": pteam,
                "opponent_team": (opp_team if teammate else team),
                "minute": minute,
                "second": second,
                "period": period,
                "event_type": event_type,
                "event_location": loc,
                "end_location": end_loc,
                "pass_completed": completed,
                "ff_location": ploc[:2],
                "ff_idx": ff_idx,
                "visible_area": visible_area,
                "player_name": None,
                "player_sb_id": None,
                "player_idx": 0,
            })
            ff_idx += 1

    event_df = pd.DataFrame(event_rows)
    expanded_df = pd.DataFrame(expanded_rows)

    if not expanded_df.empty:
        expanded_df["total_seconds"] = (
            expanded_df["minute"].fillna(0).astype(float) * 60.0
            + expanded_df["second"].fillna(0).astype(float)
        )

    return ExpandedMatchWithIdentity(event_df=event_df, expanded_df=expanded_df)


# =====================================================================
# Section 4 — 14-channel creation  (from soccermap/channels.py)
# =====================================================================

VelKey = Tuple[str, int]  # (team_name, ff_idx)


def compute_player_velocities(
    expanded_df: pd.DataFrame,
    event_id: str,
    previous_event_id: Optional[str] = None,
    *,
    max_time_gap: float = 5.0,
    max_match_distance: float = 15.0,
) -> Dict[VelKey, Tuple[float, float]]:
    """Velocity estimation by nearest-neighbor matching across consecutive events."""
    cur = expanded_df[expanded_df["event_id"] == event_id]
    if cur.empty:
        return {}

    cur_actor = cur[cur["actor"] == True]
    if cur_actor.empty:
        return {}
    cur_t = float(cur_actor.iloc[0]["total_seconds"])

    if previous_event_id is None:
        all_actor = expanded_df[expanded_df["actor"] == True].copy()
        prev_candidates = all_actor[
            (all_actor["total_seconds"] < cur_t)
            & (all_actor["total_seconds"] >= cur_t - max_time_gap)
        ]
        if prev_candidates.empty:
            return {}
        previous_event_id = str(
            prev_candidates.sort_values("total_seconds", ascending=False).iloc[0]["event_id"]
        )

    prev = expanded_df[expanded_df["event_id"] == previous_event_id]
    if prev.empty:
        return {}

    prev_actor = prev[prev["actor"] == True]
    if prev_actor.empty:
        return {}
    prev_t = float(prev_actor.iloc[0]["total_seconds"])
    dt = cur_t - prev_t
    if dt <= 0 or dt > max_time_gap:
        return {}

    cur_players = cur[cur["actor"] == False].copy()
    prev_players = prev[prev["actor"] == False].copy()

    cur_players = cur_players[
        cur_players["ff_location"].notna()
        & cur_players["team"].notna()
        & cur_players["ff_idx"].notna()
    ]
    prev_players = prev_players[
        prev_players["ff_location"].notna()
        & prev_players["team"].notna()
        & prev_players["ff_idx"].notna()
    ]

    vel: Dict[VelKey, Tuple[float, float]] = {}

    for team_name in cur_players["team"].dropna().unique():
        cur_team = cur_players[cur_players["team"] == team_name]
        prev_team = prev_players[prev_players["team"] == team_name]
        if cur_team.empty or prev_team.empty:
            continue

        cur_xy = np.vstack(cur_team["ff_location"].to_list()).astype(np.float32)
        prev_xy = np.vstack(prev_team["ff_location"].to_list()).astype(np.float32)

        d2 = ((cur_xy[:, None, :] - prev_xy[None, :, :]) ** 2).sum(axis=2)
        d = np.sqrt(d2)

        nearest = d.argmin(axis=1)
        nearest_dist = d[np.arange(d.shape[0]), nearest]

        for row_i, (j, dist) in enumerate(zip(nearest, nearest_dist)):
            if dist > max_match_distance:
                continue
            curr_row = cur_team.iloc[row_i]
            prev_row = prev_team.iloc[int(j)]

            cx, cy = curr_row["ff_location"]
            px, py = prev_row["ff_location"]
            vx = float(cx - px) / dt
            vy = float(cy - py) / dt

            key: VelKey = (team_name, int(curr_row["ff_idx"]))
            vel[key] = (vx, vy)

    return vel


def create_channel_visibility_mask(
    visible_area: Optional[List[float]],
    grid: GridSpec = GridSpec(),
) -> np.ndarray:
    """Channel 14: visibility mask (1=visible, 0=not visible)."""
    if visible_area is None or len(visible_area) < 6 or len(visible_area) % 2 != 0:
        return np.ones((grid.L, grid.W), dtype=np.float32)

    polygon_points = [
        (visible_area[i], visible_area[i + 1])
        for i in range(0, len(visible_area), 2)
    ]
    polygon_path = MplPath(polygon_points)

    x_coords, y_coords = np.meshgrid(np.arange(grid.L), np.arange(grid.W))
    points = np.vstack([x_coords.ravel(), y_coords.ravel()]).T
    inside = polygon_path.contains_points(points)
    return inside.reshape(grid.W, grid.L).T.astype(np.float32)


def create_14_channels(
    expanded_df: pd.DataFrame,
    event_id: str,
    grid: GridSpec = GridSpec(),
    *,
    velocity_dict: Optional[Dict[VelKey, Tuple[float, float]]] = None,
    visible_area: Optional[List[float]] = None,
) -> Optional[np.ndarray]:
    """
    Create a (14, L, W) channel tensor for a single event.

    Channels:
      1-2  : teammate / opponent locations (sparse counts)
      3-4  : teammate vx, vy
      5-6  : opponent vx, vy
      7    : distance to ball
      8    : distance to goal
      9-10 : sin/cos angle between (cell->goal) and (cell->ball)
      11   : angle to goal (radians)
      12-13: sin/cos ball-velocity vs ball->teammate direction
      14   : visibility mask
    """
    ev = expanded_df[expanded_df["event_id"] == event_id]
    if ev.empty:
        return None

    actor = ev[ev["actor"] == True]
    if actor.empty:
        return None
    actor = actor.iloc[0]

    ball_loc = actor["event_location"]
    if not (isinstance(ball_loc, list) and len(ball_loc) >= 2):
        return None

    ball_l_idx, ball_w_idx, ball_l, ball_w = grid.sb_to_grid(
        float(ball_loc[0]), float(ball_loc[1])
    )

    end_loc = actor.get("end_location")
    ball_vel = np.array([0.0, 0.0], dtype=np.float32)
    if isinstance(end_loc, list) and len(end_loc) >= 2:
        _, _, end_l, end_w = grid.sb_to_grid(float(end_loc[0]), float(end_loc[1]))
        v = np.array([end_l - ball_l, end_w - ball_w], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            ball_vel = v / n

    goal_l, goal_w = grid.goal_location()
    chans = np.zeros((14, grid.L, grid.W), dtype=np.float32)

    # --- sparse channels ---
    players = ev[ev["actor"] == False].copy()
    if not players.empty:
        mates = players[players["teammate"] == True]
        opps = players[players["teammate"] == False]

        def _place_sparse(df: pd.DataFrame, ch_loc: int, ch_vx: int, ch_vy: int):
            for _, row in df.iterrows():
                loc = row["ff_location"]
                if not (isinstance(loc, list) and len(loc) >= 2):
                    continue
                l_idx, w_idx, _, _ = grid.sb_to_grid(float(loc[0]), float(loc[1]))
                chans[ch_loc, l_idx, w_idx] += 1.0

                if (
                    velocity_dict is not None
                    and row.get("team") is not None
                    and row.get("ff_idx") is not None
                ):
                    key = (str(row["team"]), int(row["ff_idx"]))
                    if key in velocity_dict:
                        vx_sb, vy_sb = velocity_dict[key]
                        chans[ch_vx, l_idx, w_idx] = float(vx_sb) * grid.scale_L
                        chans[ch_vy, l_idx, w_idx] = float(vy_sb) * grid.scale_W

        _place_sparse(mates, 0, 2, 3)
        _place_sparse(opps, 1, 4, 5)

        # ball-velocity vs teammate direction (channels 12-13)
        if np.linalg.norm(ball_vel) > 1e-6:
            bv_l, bv_w = float(ball_vel[0]), float(ball_vel[1])
            for _, row in mates.iterrows():
                loc = row["ff_location"]
                if not (isinstance(loc, list) and len(loc) >= 2):
                    continue
                l_idx, w_idx, l, w = grid.sb_to_grid(float(loc[0]), float(loc[1]))
                vec = np.array([l - ball_l, w - ball_w], dtype=np.float32)
                n = float(np.linalg.norm(vec))
                if n <= 1e-6:
                    continue
                u = vec / n
                cos = bv_l * float(u[0]) + bv_w * float(u[1])
                sin = bv_l * float(u[1]) - bv_w * float(u[0])
                chans[11, l_idx, w_idx] = sin
                chans[12, l_idx, w_idx] = cos

    # --- dense channels ---
    gx, gy = grid.grid_mesh()

    chans[6] = np.sqrt((gx - ball_l) ** 2 + (gy - ball_w) ** 2)
    chans[7] = np.sqrt((gx - goal_l) ** 2 + (gy - goal_w) ** 2)

    vgx = goal_l - gx
    vgy = goal_w - gy
    vbx = ball_l - gx
    vby = ball_w - gy

    norm_g = np.sqrt(vgx**2 + vgy**2)
    norm_b = np.sqrt(vbx**2 + vby**2)
    denom = norm_g * norm_b
    mask = denom > 1e-6

    dot = vgx * vbx + vgy * vby
    cross = vgx * vby - vgy * vbx

    cos_arr = np.zeros_like(dot, dtype=np.float32)
    sin_arr = np.zeros_like(dot, dtype=np.float32)
    cos_arr[mask] = (dot[mask] / denom[mask]).astype(np.float32)
    sin_arr[mask] = (cross[mask] / denom[mask]).astype(np.float32)
    cos_arr = np.clip(cos_arr, -1.0, 1.0)

    chans[8] = sin_arr
    chans[9] = cos_arr
    chans[10] = np.arctan2(goal_w - gy, goal_l - gx).astype(np.float32)

    # visibility
    vis = visible_area if visible_area is not None else actor.get("visible_area")
    chans[13] = create_channel_visibility_mask(vis, grid)

    return chans


# =====================================================================
# Section 5 — Dataset with player identity
# =====================================================================

@dataclass(frozen=True)
class PassSampleWithIdentity:
    event_id: str
    channels: torch.Tensor       # (14, L, W)
    actor_id: int                # player embedding index
    dest_index: int              # flattened destination index
    completed: Optional[int]     # 0/1
    dest_lw: Tuple[int, int]


class PassDatasetWithIdentity(Dataset):
    """PyTorch Dataset that yields PassSampleWithIdentity (channels + actor ID)."""

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

        actor = expanded_df[expanded_df["actor"] == True].copy()
        if only_passes:
            actor = actor[actor["event_type"] == "Pass"]
        actor = actor[actor["end_location"].notna()]

        # keep only events where actor identity is known
        if "player_idx" in actor.columns:
            actor = actor[actor["player_idx"] > 0]

        self.actor_events = actor.sort_values("total_seconds").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.actor_events)

    def __getitem__(self, idx: int) -> PassSampleWithIdentity:
        row = self.actor_events.iloc[idx]
        eid = str(row["event_id"])
        actor_id = int(row.get("player_idx", 0))

        vel = None
        if self.compute_velocities and idx > 0:
            prev_eid = str(self.actor_events.iloc[idx - 1]["event_id"])
            vel = compute_player_velocities(
                self.expanded_df,
                eid,
                previous_event_id=prev_eid,
                max_time_gap=self.max_time_gap,
                max_match_distance=self.max_match_distance,
            )

        chans = create_14_channels(self.expanded_df, eid, self.grid, velocity_dict=vel)
        if chans is None:
            chans = np.zeros((14, self.grid.L, self.grid.W), dtype=np.float32)

        end_loc = row["end_location"]
        l_idx, w_idx, _, _ = self.grid.sb_to_grid(float(end_loc[0]), float(end_loc[1]))
        dest_index = int(l_idx * self.grid.W + w_idx)

        completed = row.get("pass_completed")
        completed = None if pd.isna(completed) else int(completed)

        return PassSampleWithIdentity(
            event_id=eid,
            channels=torch.tensor(chans, dtype=torch.float32),
            actor_id=actor_id,
            dest_index=dest_index,
            completed=completed,
            dest_lw=(int(l_idx), int(w_idx)),
        )


# =====================================================================
# Section 6 — SoccerMap with player embedding
# =====================================================================

# ---- CNN building blocks (from soccermap/model.py) ----

class SymmetricPadConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int = 1, mode: str = "reflect"):
        super().__init__()
        pad = k // 2
        if pad > 0:
            if mode == "reflect":
                self.pad = nn.ReflectionPad2d(pad)
            elif mode == "replicate":
                self.pad = nn.ReplicationPad2d(pad)
            else:
                raise ValueError(f"Unknown pad_mode={mode}")
        else:
            self.pad = nn.Identity()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class Conv5x5FeatBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pad_mode: str = "reflect"):
        super().__init__()
        self.c1 = SymmetricPadConv2d(in_ch, out_ch, k=5, stride=1, mode=pad_mode)
        self.c2 = SymmetricPadConv2d(out_ch, out_ch, k=5, stride=1, mode=pad_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.c2(F.relu(self.c1(x))))


class PredictionHead(nn.Module):
    def __init__(self, in_ch: int, pred_ch: int = 32):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, pred_ch, kernel_size=1)
        self.c2 = nn.Conv2d(pred_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c2(F.relu(self.c1(x)))


class Upsample2xBlock(nn.Module):
    def __init__(self, up_ch: int = 32, pad_mode: str = "reflect"):
        super().__init__()
        self.c1 = SymmetricPadConv2d(1, up_ch, k=3, stride=1, mode=pad_mode)
        self.c2 = SymmetricPadConv2d(up_ch, 1, k=3, stride=1, mode=pad_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.c2(F.relu(self.c1(x)))


class FusePair(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([a, b], dim=1))


# ---- Model ----

class SoccerMapWithEmbedding(nn.Module):
    """
    SoccerMap with a learnable player-embedding layer.

    The actor's embedding vector is broadcast spatially and concatenated
    with the 14 base channels before being fed into the multi-scale CNN.

    After training, ``get_player_embeddings()`` returns the learned vectors
    which can be compared via cosine similarity for player-similarity analysis.
    """

    def __init__(
        self,
        num_players: int,
        embed_dim: int = 8,
        base_channels: int = 14,
        feat_channels: int = 32,
        pred_channels: int = 32,
        up_channels: int = 32,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        self.num_players = num_players
        self.embed_dim = embed_dim

        # Player embedding table.  Index 0 = padding (unknown player).
        self.player_embedding = nn.Embedding(
            num_embeddings=num_players + 1,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        total_in = base_channels + embed_dim  # 14 + D

        # Multi-scale feature extraction
        self.feat1 = Conv5x5FeatBlock(total_in, feat_channels, pad_mode)
        self.pool1 = nn.MaxPool2d(2)

        self.feat2 = Conv5x5FeatBlock(feat_channels, feat_channels, pad_mode)
        self.pool2 = nn.MaxPool2d(2)

        self.feat3 = Conv5x5FeatBlock(feat_channels, feat_channels, pad_mode)

        # Prediction heads
        self.pred1 = PredictionHead(feat_channels, pred_channels)
        self.pred2 = PredictionHead(feat_channels, pred_channels)
        self.pred3 = PredictionHead(feat_channels, pred_channels)

        # Upsampling + fusion
        self.up3_to_2 = Upsample2xBlock(up_channels, pad_mode)
        self.fuse23 = FusePair()

        self.up23_to_1 = Upsample2xBlock(up_channels, pad_mode)
        self.fuse123 = FusePair()

    def forward(self, x: torch.Tensor, actor_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 14, L, W)  base channel tensor
        actor_ids : (N,)    long tensor of player embedding indices

        Returns
        -------
        logits : (N, 1, L, W)
        """
        N, C, L, W = x.shape

        # Look up actor embedding and broadcast spatially
        embed = self.player_embedding(actor_ids)                  # (N, D)
        embed_spatial = embed[:, :, None, None].expand(N, self.embed_dim, L, W)

        # Concatenate: (N, 14+D, L, W)
        x = torch.cat([x, embed_spatial], dim=1)

        # Encoder
        h1 = self.feat1(x)
        p1 = self.pred1(h1)

        h2 = self.feat2(self.pool1(h1))
        p2 = self.pred2(h2)

        h3 = self.feat3(self.pool2(h2))
        p3 = self.pred3(h3)

        # Decoder
        p3_up = self.up3_to_2(p3)
        p23 = self.fuse23(p2, p3_up)

        p23_up = self.up23_to_1(p23)
        p123 = self.fuse123(p1, p23_up)

        return p123  # (N, 1, L, W) logits

    # ---- Embedding utilities ----

    def get_player_embeddings(self) -> np.ndarray:
        """Return all player embeddings as numpy array, shape (num_players+1, D)."""
        return self.player_embedding.weight.detach().cpu().numpy()

    def get_player_similarity(self, idx_a: int, idx_b: int) -> float:
        """Cosine similarity between two players (by embedding index)."""
        w = self.player_embedding.weight.detach()
        a = w[idx_a]
        b = w[idx_b]
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

    def get_similarity_matrix(self, exclude_padding: bool = True) -> np.ndarray:
        """
        Full pairwise cosine-similarity matrix.

        Returns shape (P, P) where P = num_players  (or num_players+1 if
        ``exclude_padding=False``).
        """
        w = self.player_embedding.weight.detach()
        if exclude_padding:
            w = w[1:]  # skip index-0 padding
        w_norm = F.normalize(w, dim=1)
        return (w_norm @ w_norm.T).cpu().numpy()


# =====================================================================
# Section 7 — Loss functions & surfaces
# =====================================================================

def _gather_dest_logits(logits: torch.Tensor, dest_index: torch.Tensor) -> torch.Tensor:
    N = logits.shape[0]
    flat = logits.view(N, -1)
    return flat.gather(1, dest_index.view(-1, 1)).squeeze(1)


def pass_selection_loss(logits: torch.Tensor, dest_index: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over all pixels for pass destination prediction."""
    N = logits.shape[0]
    flat = logits.view(N, -1)
    return F.cross_entropy(flat, dest_index)


def pass_success_loss(
    logits: torch.Tensor,
    dest_index: torch.Tensor,
    completed: torch.Tensor,
) -> torch.Tensor:
    """BCE at destination pixel for pass success prediction."""
    chosen = _gather_dest_logits(logits, dest_index)
    return F.binary_cross_entropy_with_logits(chosen, completed.float())


@torch.no_grad()
def pass_selection_surface(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over all pixels -> probability distribution over pitch."""
    N, _, H, W = logits.shape
    flat = logits.view(N, -1)
    return torch.softmax(flat, dim=1).view(N, H, W)


@torch.no_grad()
def pass_success_surface(logits: torch.Tensor) -> torch.Tensor:
    """Sigmoid -> per-pixel success probability."""
    return torch.sigmoid(logits).squeeze(1)
