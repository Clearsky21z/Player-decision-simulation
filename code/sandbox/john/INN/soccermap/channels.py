from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from .config import GridSpec

VelKey = Tuple[str, int]  # (team_name, ff_idx) -> (vx, vy) in StatsBomb units/sec


# =============================================================================
# Small extraction/helpers
# =============================================================================
def _get_event_slice(expanded_df: pd.DataFrame, event_id: str) -> pd.DataFrame:
    return expanded_df.loc[expanded_df["event_id"] == event_id]


def _get_actor_row(ev: pd.DataFrame) -> Optional[pd.Series]:
    actor = ev.loc[ev["actor"] == True]
    if actor.empty:
        return None
    return actor.iloc[0]


def _get_players(ev: pd.DataFrame) -> pd.DataFrame:
    return ev.loc[ev["actor"] == False].copy()


def _safe_loc_xy(loc) -> Optional[Tuple[float, float]]:
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        try:
            return float(loc[0]), float(loc[1])
        except Exception:
            return None
    return None


def _sb_to_grid_point(grid: GridSpec, x: float, y: float):
    """
    Returns:
      l_idx, w_idx: integer indices
      l, w        : continuous coordinates in grid-space (consistent with grid.grid_mesh())
    """
    return grid.sb_to_grid(x, y)


def _normalize(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _grid_to_sb_xy(grid: GridSpec, l: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert grid-space coordinates -> StatsBomb coordinates (x,y).

    Assumes GridSpec uses:
      scale_L = L / 120, scale_W = W / 80  (grid units per SB unit)
    so:
      x = l / scale_L, y = w / scale_W
    """
    x = l / float(grid.scale_L)
    y = w / float(grid.scale_W)
    return x, y


def _place_sparse_counts(
        chans: np.ndarray,
        df: pd.DataFrame,
        grid: GridSpec,
        ch_loc: int,
) -> None:
    """Sparse count map at player cells."""
    for _, row in df.iterrows():
        loc = row.get("ff_location")
        xy = _safe_loc_xy(loc)
        if xy is None:
            continue
        l_idx, w_idx, _, _ = _sb_to_grid_point(grid, xy[0], xy[1])
        chans[ch_loc, l_idx, w_idx] += 1.0


def _place_sparse_velocities(
        chans: np.ndarray,
        df: pd.DataFrame,
        grid: GridSpec,
        ch_vx: int,
        ch_vy: int,
        velocity_dict: Optional[Dict[VelKey, Tuple[float, float]]],
) -> None:
    """Sparse vx/vy maps at player cells (if velocity_dict exists)."""
    if velocity_dict is None:
        return

    for _, row in df.iterrows():
        loc = row.get("ff_location")
        xy = _safe_loc_xy(loc)
        if xy is None:
            continue

        team = row.get("team")
        ff_idx = row.get("ff_idx")
        if team is None or ff_idx is None:
            continue

        key = (str(team), int(ff_idx))
        if key not in velocity_dict:
            continue

        l_idx, w_idx, _, _ = _sb_to_grid_point(grid, xy[0], xy[1])

        vx_sb, vy_sb = velocity_dict[key]  # SB units/sec
        # SB units/sec -> grid units/sec
        vx = float(vx_sb) * float(grid.scale_L)
        vy = float(vy_sb) * float(grid.scale_W)

        chans[ch_vx, l_idx, w_idx] = vx
        chans[ch_vy, l_idx, w_idx] = vy


def _ball_direction_from_pass_end(grid: GridSpec, ball_lw: Tuple[float, float], end_loc) -> np.ndarray:
    """
    (Heuristic) direction based on pass start->end.
    Note: Paper says "ball-carrier velocity"; this is "pass direction".
    """
    ball_l, ball_w = ball_lw
    xy = _safe_loc_xy(end_loc)
    if xy is None:
        return np.zeros((2,), dtype=np.float32)

    _, _, end_l, end_w = _sb_to_grid_point(grid, xy[0], xy[1])
    v = np.array([end_l - ball_l, end_w - ball_w], dtype=np.float32)
    return _normalize(v).astype(np.float32)


def _dense_distance(gx: np.ndarray, gy: np.ndarray, cx: float, cy: float) -> np.ndarray:
    return np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2).astype(np.float32)


def _dense_angle_to_point(gx: np.ndarray, gy: np.ndarray, tx: float, ty: float) -> np.ndarray:
    """Angle in radians from each cell to target point (tx,ty)."""
    return np.arctan2((ty - gy), (tx - gx)).astype(np.float32)


def _sin_cos_between(
        ax: np.ndarray, ay: np.ndarray,
        bx: np.ndarray, by: np.ndarray,
        eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each cell, compute sin/cos between vectors a and b.
    """
    norm_a = np.sqrt(ax**2 + ay**2)
    norm_b = np.sqrt(bx**2 + by**2)
    denom = norm_a * norm_b

    mask = denom > eps

    dot = ax * bx + ay * by
    cross = ax * by - ay * bx

    cos = np.zeros_like(dot, dtype=np.float32)
    sin = np.zeros_like(dot, dtype=np.float32)
    cos[mask] = (dot[mask] / denom[mask]).astype(np.float32)
    sin[mask] = (cross[mask] / denom[mask]).astype(np.float32)

    cos = np.clip(cos, -1.0, 1.0)
    sin = np.clip(sin, -1.0, 1.0)
    return sin, cos


# =============================================================================
# Velocity estimation (unchanged API, cleaner internals)
# =============================================================================
def compute_player_velocities(
        expanded_df: pd.DataFrame,
        event_id: str,
        previous_event_id: Optional[str] = None,
        *,
        max_time_gap: float = 5.0,
        max_match_distance: float = 15.0,
) -> Dict[VelKey, Tuple[float, float]]:
    """
    Velocity estimation by nearest-neighbor matching across consecutive events.

    - Match by team name + proximity.
    - Key by (team, ff_idx) from CURRENT frame.
    """
    cur = _get_event_slice(expanded_df, event_id)
    if cur.empty:
        return {}

    cur_actor = _get_actor_row(cur)
    if cur_actor is None:
        return {}

    cur_t = float(cur_actor["total_seconds"])

    if previous_event_id is None:
        all_actor = expanded_df.loc[expanded_df["actor"] == True].copy()
        prev_candidates = all_actor[
            (all_actor["total_seconds"] < cur_t) &
            (all_actor["total_seconds"] >= cur_t - max_time_gap)
            ]
        if prev_candidates.empty:
            return {}
        previous_event_id = str(prev_candidates.sort_values("total_seconds", ascending=False).iloc[0]["event_id"])

    prev = _get_event_slice(expanded_df, previous_event_id)
    if prev.empty:
        return {}

    prev_actor = _get_actor_row(prev)
    if prev_actor is None:
        return {}

    prev_t = float(prev_actor["total_seconds"])
    dt = cur_t - prev_t
    if dt <= 0 or dt > max_time_gap:
        return {}

    cur_players = _get_players(cur)
    prev_players = _get_players(prev)

    # keep only rows with needed info
    cur_players = cur_players[
        cur_players["ff_location"].notna() &
        cur_players["team"].notna() &
        cur_players["ff_idx"].notna()
        ]
    prev_players = prev_players[
        prev_players["ff_location"].notna() &
        prev_players["team"].notna() &
        prev_players["ff_idx"].notna()
        ]

    vel: Dict[VelKey, Tuple[float, float]] = {}

    for team_name in cur_players["team"].dropna().unique():
        cur_team = cur_players.loc[cur_players["team"] == team_name]
        prev_team = prev_players.loc[prev_players["team"] == team_name]
        if cur_team.empty or prev_team.empty:
            continue

        cur_xy = np.vstack(cur_team["ff_location"].to_list()).astype(np.float32)   # (Nc,2) SB coords
        prev_xy = np.vstack(prev_team["ff_location"].to_list()).astype(np.float32) # (Np,2)

        d2 = ((cur_xy[:, None, :] - prev_xy[None, :, :]) ** 2).sum(axis=2)
        d = np.sqrt(d2)

        nearest = d.argmin(axis=1)
        nearest_dist = d[np.arange(d.shape[0]), nearest]

        # IMPORTANT: keep row alignment using iloc on the team-sliced frames
        cur_team_iloc = cur_team.reset_index(drop=True)
        prev_team_iloc = prev_team.reset_index(drop=True)

        for i, (j, dist) in enumerate(zip(nearest, nearest_dist)):
            if float(dist) > max_match_distance:
                continue

            curr_row = cur_team_iloc.iloc[int(i)]
            prev_row = prev_team_iloc.iloc[int(j)]

            cx, cy = curr_row["ff_location"]
            px, py = prev_row["ff_location"]
            vx = float(cx - px) / dt
            vy = float(cy - py) / dt

            key: VelKey = (str(team_name), int(curr_row["ff_idx"]))
            vel[key] = (vx, vy)

    return vel

# =============================================================================
# Main channel creator (same API, refactored)
# =============================================================================
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

    Channel layout (0-based indices):
      0  : teammate locations (sparse counts)
      1  : opponent locations (sparse counts)
      2  : teammate vx (sparse)
      3  : teammate vy (sparse)
      4  : opponent vx (sparse)
      5  : opponent vy (sparse)
      6  : distance to ball (dense)
      7  : distance to goal (dense)
      8  : sin between (cell->goal) and (cell->ball) (dense)
      9  : cos between (cell->goal) and (cell->ball) (dense)
      10 : angle to goal in radians (dense)
      11 : sin between pass-dir and (ball->teammate) (sparse at teammate cells)
      12 : cos between pass-dir and (ball->teammate) (sparse at teammate cells)
      13 : visibility mask (dense)
    """
    ev = _get_event_slice(expanded_df, event_id)
    if ev.empty:
        return None

    actor = _get_actor_row(ev)
    if actor is None:
        return None

    ball_xy = _safe_loc_xy(actor.get("event_location"))
    if ball_xy is None:
        return None

    # If not passed in, try to pull visible_area from actor row (common in expanded tables)
    if visible_area is None:
        va = actor.get("visible_area")
        if isinstance(va, list):
            visible_area = va

    # ball origin in grid coords
    ball_l_idx, ball_w_idx, ball_l, ball_w = _sb_to_grid_point(grid, ball_xy[0], ball_xy[1])
    ball_lw = (float(ball_l), float(ball_w))

    # direction feature (heuristic)
    pass_dir = _ball_direction_from_pass_end(grid, ball_lw, actor.get("end_location"))  # (2,)

    goal_l, goal_w = grid.goal_location()

    chans = np.zeros((14, grid.L, grid.W), dtype=np.float32)

    # ---- sparse players ----
    players = _get_players(ev)
    if not players.empty:
        mates = players.loc[players["teammate"] == True]
        opps = players.loc[players["teammate"] == False]

        # locations
        _place_sparse_counts(chans, mates, grid, ch_loc=0)
        _place_sparse_counts(chans, opps, grid, ch_loc=1)

        # velocities (optional)
        _place_sparse_velocities(chans, mates, grid, ch_vx=2, ch_vy=3, velocity_dict=velocity_dict)
        _place_sparse_velocities(chans, opps, grid, ch_vx=4, ch_vy=5, velocity_dict=velocity_dict)

        # pass-dir vs (ball->teammate) angle at teammate cells (channels 11-12)
        if float(np.linalg.norm(pass_dir)) > 1e-6:
            dv_l, dv_w = float(pass_dir[0]), float(pass_dir[1])
            for _, row in mates.iterrows():
                loc = row.get("ff_location")
                xy = _safe_loc_xy(loc)
                if xy is None:
                    continue

                l_idx, w_idx, l, w = _sb_to_grid_point(grid, xy[0], xy[1])
                vec = np.array([float(l) - ball_lw[0], float(w) - ball_lw[1]], dtype=np.float32)
                u = _normalize(vec).astype(np.float32)
                if float(np.linalg.norm(u)) <= 1e-6:
                    continue

                # cos = dot, sin = 2D cross
                cos = dv_l * float(u[0]) + dv_w * float(u[1])
                sin = dv_l * float(u[1]) - dv_w * float(u[0])

                chans[11, l_idx, w_idx] = float(sin)
                chans[12, l_idx, w_idx] = float(cos)

    # ---- dense fields ----
    gx, gy = grid.grid_mesh()  # shape (L,W) in grid-space

    # 6: distance to ball
    chans[6] = _dense_distance(gx, gy, ball_lw[0], ball_lw[1])

    # 7: distance to goal
    chans[7] = _dense_distance(gx, gy, float(goal_l), float(goal_w))

    # 8-9: sin/cos between (cell->goal) and (cell->ball)
    vgx = (float(goal_l) - gx)
    vgy = (float(goal_w) - gy)
    vbx = (ball_lw[0] - gx)
    vby = (ball_lw[1] - gy)
    sin, cos = _sin_cos_between(vgx, vgy, vbx, vby)
    chans[8] = sin
    chans[9] = cos

    # 10: angle to goal (radians)
    chans[10] = _dense_angle_to_point(gx, gy, float(goal_l), float(goal_w))

    # 13: visibility
    chans[13] = create_channel_visibility_mask(visible_area, grid)

    return chans


def create_channels_for_events(
        expanded_df: pd.DataFrame,
        event_ids: List[str],
        grid: GridSpec = GridSpec(),
        *,
        compute_velocities: bool = False,
        max_time_gap: float = 5.0,
        max_match_distance: float = 15.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Batch channel creation.

    If compute_velocities=True, we sort by time and compute velocities using the previous event.
    """
    # sort by actor time
    times: List[Tuple[str, float]] = []
    for eid in event_ids:
        ev_actor = expanded_df.loc[(expanded_df["event_id"] == eid) & (expanded_df["actor"] == True)]
        if ev_actor.empty:
            continue
        t = float(ev_actor.iloc[0]["total_seconds"])
        times.append((eid, t))

    times.sort(key=lambda x: x[1])
    sorted_ids = [eid for eid, _ in times]

    out: List[np.ndarray] = []
    valid: List[str] = []

    prev_id: Optional[str] = None
    for eid in sorted_ids:
        vel = None
        if compute_velocities and prev_id is not None:
            vel = compute_player_velocities(
                expanded_df,
                eid,
                previous_event_id=prev_id,
                max_time_gap=max_time_gap,
                max_match_distance=max_match_distance,
            )

        chans = create_14_channels(expanded_df, eid, grid, velocity_dict=vel)
        if chans is not None:
            out.append(chans)
            valid.append(eid)
            prev_id = eid

    if len(out) == 0:
        return np.zeros((0, 14, grid.L, grid.W), dtype=np.float32), []

    return np.stack(out, axis=0), valid

# TODO: potential channel 1/2 update:
# instead of sparse counts, we can do Gaussian kernels centered at player locations?


def create_channel_visibility_mask(
    visible_area: Optional[List[float]],
    grid: GridSpec = GridSpec(),
) -> np.ndarray:
    """
    Channel 14: Visibility mask (dense).
    1 = visible, 0 = not visible.

    Args:
        visible_area: Flat list [x1, y1, x2, y2, ...] defining visible polygon
        grid: GridSpec defining the grid dimensions

    Returns:
        np.array of shape (grid.L, grid.W)
    """
    # If no valid visible area, assume entire pitch is visible
    if visible_area is None or len(visible_area) < 6 or len(visible_area) % 2 != 0:
        return np.ones((grid.L, grid.W), dtype=np.float32)
    # Convert flat array to polygon points
    polygon_points = [
        (visible_area[i], visible_area[i+1])
        for i in range(0, len(visible_area), 2)
    ]

    # Create matplotlib Path
    polygon_path = MplPath(polygon_points)

    # Create grid of all points (vectorized)
    x_coords, y_coords = np.meshgrid(np.arange(grid.L), np.arange(grid.W))
    points = np.vstack([x_coords.ravel(), y_coords.ravel()]).T

    # Test all points
    inside = polygon_path.contains_points(points)
    channel = inside.reshape(grid.W, grid.L).T.astype(np.float32)

    return channel
