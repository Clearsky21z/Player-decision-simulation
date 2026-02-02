from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from .config import GridSpec


VelKey = Tuple[str, int]  # (team_name, ff_idx) -> (vx,vy) in StatsBomb units per second


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

    Fix vs notebook:
    - We match by absolute team name + proximity (not just teammate flag).
    - We key by (team, ff_idx) so it's stable and does not depend on DataFrame indices.

    Notes:
    - This is a heuristic. It works best when the previous event is close in time and play is continuous.
    """
    cur = expanded_df[expanded_df["event_id"] == event_id]
    if cur.empty:
        return {}

    cur_actor = cur[cur["actor"] == True]
    if cur_actor.empty:
        return {}

    cur_t = float(cur_actor.iloc[0]["total_seconds"])

    if previous_event_id is None:
        # pick closest previous actor event within time window
        all_actor = expanded_df[expanded_df["actor"] == True].copy()
        prev_candidates = all_actor[(all_actor["total_seconds"] < cur_t) & (all_actor["total_seconds"] >= cur_t - max_time_gap)]
        if prev_candidates.empty:
            return {}
        previous_event_id = str(prev_candidates.sort_values("total_seconds", ascending=False).iloc[0]["event_id"])

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

    # drop missing
    cur_players = cur_players[cur_players["ff_location"].notna() & cur_players["team"].notna() & cur_players["ff_idx"].notna()]
    prev_players = prev_players[prev_players["ff_location"].notna() & prev_players["team"].notna() & prev_players["ff_idx"].notna()]

    vel: Dict[VelKey, Tuple[float, float]] = {}

    # process each team separately
    for team_name in cur_players["team"].dropna().unique():
        cur_team = cur_players[cur_players["team"] == team_name]
        prev_team = prev_players[prev_players["team"] == team_name]
        if cur_team.empty or prev_team.empty:
            continue

        cur_xy = np.vstack(cur_team["ff_location"].to_list()).astype(np.float32)  # (Nc,2) in SB coords
        prev_xy = np.vstack(prev_team["ff_location"].to_list()).astype(np.float32)  # (Np,2)

        # pairwise distances (Nc,Np)
        # Nc is at most 21-ish; Np also small, so this is fine.
        d2 = ((cur_xy[:, None, :] - prev_xy[None, :, :]) ** 2).sum(axis=2)
        d = np.sqrt(d2)

        # greedy matching: for each current player, take nearest previous under threshold
        # (could be improved with Hungarian matching, but we keep it simple)
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

    Channels (matches notebook intent):
      1-2  : teammate / opponent locations (sparse counts)
      3-4  : teammate vx, vy (sparse)
      5-6  : opponent vx, vy (sparse)
      7    : distance to ball (dense)
      8    : distance to goal (dense)
      9-10 : sin/cos angle between (cell->goal) and (cell->ball) (dense)
      11   : angle to goal in radians (dense)
      12-13: sin/cos angle between ball velocity dir and (ball->teammate) (sparse at teammate cells)
      14   : visibility mask (dense)

    Returns None if event not found or has no actor row.
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

    # ball origin in grid coords
    ball_l_idx, ball_w_idx, ball_l, ball_w = grid.sb_to_grid(float(ball_loc[0]), float(ball_loc[1]))

    # pass end location for ball velocity direction
    end_loc = actor.get("end_location")
    ball_vel = np.array([0.0, 0.0], dtype=np.float32)  # in grid coords (dl, dw)
    if isinstance(end_loc, list) and len(end_loc) >= 2:
        _, _, end_l, end_w = grid.sb_to_grid(float(end_loc[0]), float(end_loc[1]))
        v = np.array([end_l - ball_l, end_w - ball_w], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            ball_vel = v / n

    goal_l, goal_w = grid.goal_location()

    chans = np.zeros((14, grid.L, grid.W), dtype=np.float32)

    # ----- sparse channels: player locations + velocities -----
    players = ev[ev["actor"] == False].copy()
    if not players.empty:
        # teammates (relative to actor)
        mates = players[players["teammate"] == True]
        opps = players[players["teammate"] == False]

        def _place_sparse(df: pd.DataFrame, ch_loc: int, ch_vx: int, ch_vy: int):
            for _, row in df.iterrows():
                loc = row["ff_location"]
                if not (isinstance(loc, list) and len(loc) >= 2):
                    continue
                l_idx, w_idx, _, _ = grid.sb_to_grid(float(loc[0]), float(loc[1]))
                chans[ch_loc, l_idx, w_idx] += 1.0

                if velocity_dict is not None and row.get("team") is not None and row.get("ff_idx") is not None:
                    key = (str(row["team"]), int(row["ff_idx"]))
                    if key in velocity_dict:
                        vx_sb, vy_sb = velocity_dict[key]
                        # convert SB-units/sec -> grid-units/sec
                        vx = float(vx_sb) * grid.scale_L
                        vy = float(vy_sb) * grid.scale_W
                        chans[ch_vx, l_idx, w_idx] = vx
                        chans[ch_vy, l_idx, w_idx] = vy

        _place_sparse(mates, 0, 2, 3)
        _place_sparse(opps, 1, 4, 5)

        # ball-velocity vs teammate direction (channels 12-13) at teammate cells
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
                # cos = dot, sin = cross (2D)
                cos = bv_l * float(u[0]) + bv_w * float(u[1])
                sin = bv_l * float(u[1]) - bv_w * float(u[0])
                chans[11, l_idx, w_idx] = sin
                chans[12, l_idx, w_idx] = cos

    # ----- dense channels (vectorized) -----
    gx, gy = grid.grid_mesh()  # gx,gy shape (L,W)

    # 7: distance to ball
    chans[6] = np.sqrt((gx - ball_l) ** 2 + (gy - ball_w) ** 2)

    # 8: distance to goal
    chans[7] = np.sqrt((gx - goal_l) ** 2 + (gy - goal_w) ** 2)

    # 9-10: sin/cos between (cell->goal) and (cell->ball)
    vgx = (goal_l - gx)
    vgy = (goal_w - gy)
    vbx = (ball_l - gx)
    vby = (ball_w - gy)

    norm_g = np.sqrt(vgx**2 + vgy**2)
    norm_b = np.sqrt(vbx**2 + vby**2)
    denom = norm_g * norm_b
    # avoid /0
    mask = denom > 1e-6

    dot = vgx * vbx + vgy * vby
    cross = vgx * vby - vgy * vbx

    cos = np.zeros_like(dot, dtype=np.float32)
    sin = np.zeros_like(dot, dtype=np.float32)
    cos[mask] = (dot[mask] / denom[mask]).astype(np.float32)
    sin[mask] = (cross[mask] / denom[mask]).astype(np.float32)

    cos = np.clip(cos, -1.0, 1.0)
    chans[8] = sin
    chans[9] = cos

    # 11: angle to goal in radians
    chans[10] = np.arctan2((goal_w - gy), (goal_l - gx)).astype(np.float32)

    # 14: visibility mask
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
    # sort by time (actor rows)
    times = []
    for eid in event_ids:
        ev = expanded_df[(expanded_df["event_id"] == eid) & (expanded_df["actor"] == True)]
        if ev.empty:
            continue
        t = float(ev.iloc[0]["total_seconds"])
        times.append((eid, t))
    times.sort(key=lambda x: x[1])
    sorted_ids = [eid for eid, _ in times]

    out = []
    valid = []

    prev_id = None
    for eid in sorted_ids:
        vel = None
        if compute_velocities and prev_id is not None:
            vel = compute_player_velocities(
                expanded_df, eid, previous_event_id=prev_id,
                max_time_gap=max_time_gap, max_match_distance=max_match_distance
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
