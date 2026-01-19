from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

GRID_W = 104  # length axis (x)
GRID_H = 68   # width axis (y)

OUT_DIR = "channel_images"


# =========================
# Utilities
# =========================
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def to_grid_xy(x: float, y: float) -> Tuple[int, int]:
    """
    Map StatsBomb metric coordinates (0..120, 0..80) to grid indices.
    Returns (gx, gy) where gx in [0, GRID_W-1], gy in [0, GRID_H-1].
    """
    gx = int(np.floor((x / PITCH_LENGTH) * (GRID_W - 1)))
    gy = int(np.floor((y / PITCH_WIDTH) * (GRID_H - 1)))
    gx = int(clamp(gx, 0, GRID_W - 1))
    gy = int(clamp(gy, 0, GRID_H - 1))
    return gx, gy


def flip_lr_location(loc: List[float]) -> List[float]:
    """Flip a [x,y] location so that attack direction becomes left->right."""
    return [PITCH_LENGTH - float(loc[0]), PITCH_WIDTH - float(loc[1])]


def normalize_frame_left_to_right(
        actor_loc: List[float],
        end_loc: Optional[List[float]],
        freeze_frame: List[Dict],
) -> Tuple[List[float], Optional[List[float]], List[Dict], bool]:
    """
    Heuristic normalization: if end_x < start_x, treat this as right->left play
    and flip all coordinates (actor, end_loc, freeze-frame players).
    """
    flipped = False
    if end_loc is not None and len(end_loc) >= 2:
        if float(end_loc[0]) < float(actor_loc[0]):
            flipped = True

    if not flipped:
        return actor_loc, end_loc, freeze_frame, False

    actor_loc2 = flip_lr_location(actor_loc)
    end_loc2 = flip_lr_location(end_loc) if end_loc is not None else None

    ff2 = []
    for p in freeze_frame:
        p2 = dict(p)
        if p2.get("location") is not None and len(p2["location"]) >= 2:
            p2["location"] = flip_lr_location(p2["location"])
        ff2.append(p2)

    return actor_loc2, end_loc2, ff2, True


def build_meshgrid_xy() -> Tuple[np.ndarray, np.ndarray]:
    """
    Return meshgrid arrays for cell centers in metric coordinates:
    X: shape (GRID_H, GRID_W), Y: shape (GRID_H, GRID_W)
    """
    xs = np.linspace(0, PITCH_LENGTH, GRID_W, dtype=np.float32)
    ys = np.linspace(0, PITCH_WIDTH, GRID_H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)  # default indexing: (y,x)
    return X, Y


def angle_between(u: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cos(theta) and sin(theta) between vectors u and v.
    u, v shapes: (...,2)
    Returns: cos_theta, sin_theta (same leading shape)
    """
    dot = np.sum(u * v, axis=-1)
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    denom = (nu * nv) + eps
    cos_t = dot / denom
    cos_t = np.clip(cos_t, -1.0, 1.0)
    # sin via sqrt(1-cos^2) but preserve sign using 2D cross product (z-component)
    cross_z = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t**2))
    sin_t = np.where(cross_z >= 0, sin_t, -sin_t)
    return cos_t, sin_t


# =========================
# Loading (minimal, standalone)
# =========================
def load_match_json(match_id: str, base_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load events, three-sixty, lineups JSON for one match.
    base_dir should point to StatsBomb 'data' folder which contains:
      events/, three-sixty/, lineups/
    """
    base = Path(base_dir)
    events_path = base / "events" / f"{match_id}.json"
    threesixty_path = base / "three-sixty" / f"{match_id}.json"
    lineups_path = base / "lineups" / f"{match_id}.json"

    with open(events_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    with open(threesixty_path, "r", encoding="utf-8") as f:
        threesixty = json.load(f)

    with open(lineups_path, "r", encoding="utf-8") as f:
        lineups = json.load(f)

    return events, threesixty, lineups


def merge_freeze_frames(events: List[Dict], threesixty: List[Dict]) -> List[Dict]:
    """
    Attach freeze_frame + visible_area to each event if available.
    StatsBomb 360 uses 'event_uuid' to reference events 'id'.
    """
    lookup = {ff["event_uuid"]: ff for ff in threesixty}
    combined = []
    for ev in events:
        ev_id = ev.get("id")
        if ev_id in lookup:
            ev = dict(ev)
            ev["freeze_frame"] = lookup[ev_id].get("freeze_frame", [])
            ev["visible_area"] = lookup[ev_id].get("visible_area", [])
        else:
            ev = dict(ev)
            ev["freeze_frame"] = []
            ev["visible_area"] = []
        combined.append(ev)
    return combined


def events_to_df(combined_events: List[Dict]) -> pd.DataFrame:
    """
    Build a light events DataFrame with fields used for channel creation and velocity estimation.
    """
    rows = []
    for ev in combined_events:
        ev_type = (ev.get("type") or {}).get("name")
        row = {
            "event_id": ev.get("id"),
            "match_id": ev.get("match_id"),
            "team": (ev.get("team") or {}).get("name"),
            "player": (ev.get("player") or {}).get("name"),
            "player_id": (ev.get("player") or {}).get("id"),
            "type": ev_type,
            "minute": ev.get("minute"),
            "second": ev.get("second"),
            "location": ev.get("location"),
            "freeze_frame": ev.get("freeze_frame", []),
        }
        # end_location for pass/carry/shot
        if ev_type == "Pass":
            row["end_location"] = (ev.get("pass") or {}).get("end_location")
            row["pass_outcome"] = 0 if (ev.get("pass") or {}).get("outcome") else 1  # 1=completed, 0=failed
        elif ev_type == "Carry":
            row["end_location"] = (ev.get("carry") or {}).get("end_location")
            row["pass_outcome"] = np.nan
        elif ev_type == "Shot":
            row["end_location"] = (ev.get("shot") or {}).get("end_location")
            row["pass_outcome"] = np.nan
        else:
            row["end_location"] = None
            row["pass_outcome"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# Velocity estimation (event-to-event, actor only)
# =========================
def total_seconds(minute: int, second: int) -> float:
    return float(minute) * 60.0 + float(second)


def estimate_actor_velocity_vector(
        events_df: pd.DataFrame,
        smooth_window: int = 3,
        min_dt: float = 0.5,
        max_speed: float = 12.0,
) -> pd.DataFrame:
    """
    Estimate (vx, vy) for each player's events using displacement to next event.
    Output columns:
      - vx, vy  (metric units per second)
      - speed
    """
    df = events_df.copy()

    required = ["match_id", "player_id", "minute", "second", "location", "event_id"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Sort in time order per player
    df = df.sort_values(["match_id", "player_id", "minute", "second", "event_id"]).reset_index(drop=True)

    df["t"] = df.apply(lambda r: total_seconds(r["minute"], r["second"]), axis=1)
    g = df.groupby(["match_id", "player_id"], sort=False)

    df["t_next"] = g["t"].shift(-1)
    df["loc_next"] = g["location"].shift(-1)

    def safe_vec(row):
        loc = row["location"]
        loc2 = row["loc_next"]
        if not isinstance(loc, (list, tuple)) or not isinstance(loc2, (list, tuple)):
            return (0.0, 0.0, 0.0)

        if len(loc) < 2 or len(loc2) < 2:
            return (0.0, 0.0, 0.0)

        dt = row["t_next"] - row["t"]
        if pd.isna(dt):
            return (0.0, 0.0, 0.0)

        dt = max(float(dt), min_dt)
        dx = float(loc2[0]) - float(loc[0])
        dy = float(loc2[1]) - float(loc[1])

        vx = dx / dt
        vy = dy / dt
        spd = float(np.sqrt(vx * vx + vy * vy))

        # Filter unrealistic speeds
        if spd > max_speed:
            return (0.0, 0.0, 0.0)

        return (vx, vy, spd)

    vecs = df.apply(safe_vec, axis=1, result_type="expand")
    df["vx"] = vecs[0].astype(float)
    df["vy"] = vecs[1].astype(float)
    df["speed"] = vecs[2].astype(float)

    # Smooth per player to reduce event noise
    if smooth_window > 1:
        df["vx"] = g["vx"].transform(lambda x: x.rolling(smooth_window, min_periods=1).mean())
        df["vy"] = g["vy"].transform(lambda x: x.rolling(smooth_window, min_periods=1).mean())
        df["speed"] = g["speed"].transform(lambda x: x.rolling(smooth_window, min_periods=1).mean())

    return df


# =========================
# Channel construction (13 channels)
# =========================
def build_13_channels_for_pass(pass_row: pd.Series, actor_v: Tuple[float, float]) -> np.ndarray:
    """
    Build 13 channels (GRID_H, GRID_W, 13) for one pass event.

    Channel order (matches paper description):
      1  attacker presence (sparse)
      2  attacker vx (sparse)
      3  attacker vy (sparse)
      4  defender presence (sparse)
      5  defender vx (sparse)
      6  defender vy (sparse)
      7  dist to ball (dense)
      8  dist to goal (dense)
      9  sin(angle between vectors to goal and to ball) (dense)
      10 cos(angle between vectors to goal and to ball) (dense)
      11 angle to goal (radians) (dense)
      12 sin(angle between actor velocity vector and vector to each teammate) (sparse)
      13 cos(angle between actor velocity vector and vector to each teammate) (sparse)
    """
    # Initialize channels
    C = np.zeros((GRID_H, GRID_W, 13), dtype=np.float32)

    actor_loc = pass_row["location"]
    end_loc = pass_row.get("end_location")
    freeze_frame = pass_row.get("freeze_frame", [])

    if not (isinstance(actor_loc, (list, tuple)) and len(actor_loc) >= 2):
        raise ValueError("Pass row has invalid actor location.")

    # Normalize to left->right using a simple heuristic (see top of file)
    actor_loc, end_loc, freeze_frame, flipped = normalize_frame_left_to_right(actor_loc, end_loc, freeze_frame)

    # Define ball and goal (opponent goal is at x=120, y=40 after left->right normalization)
    ball_x, ball_y = float(actor_loc[0]), float(actor_loc[1])
    goal_x, goal_y = PITCH_LENGTH, PITCH_WIDTH / 2.0

    # ===== 1-6 Sparse: player positions (+ velocities where available)
    # In StatsBomb 360 freeze_frame, teammate/opponent flags exist; 'actor' identifies ball carrier.
    # We add actor as an attacker as well.
    for p in freeze_frame:
        loc = p.get("location")
        if not (isinstance(loc, (list, tuple)) and len(loc) >= 2):
            continue

        px, py = float(loc[0]), float(loc[1])
        gx, gy = to_grid_xy(px, py)

        is_teammate = bool(p.get("teammate"))
        is_actor = bool(p.get("actor"))

        if is_teammate or is_actor:
            # attacker presence
            C[gy, gx, 0] = 1.0

            # attacker velocity components:
            # we only know actor velocity vector; teammates (non-actor) set to 0
            if is_actor:
                C[gy, gx, 1] = float(actor_v[0])
                C[gy, gx, 2] = float(actor_v[1])
            else:
                C[gy, gx, 1] = 0.0
                C[gy, gx, 2] = 0.0

        else:
            # defender presence
            C[gy, gx, 3] = 1.0
            # defender velocity set to 0 due to data limitation
            C[gy, gx, 4] = 0.0
            C[gy, gx, 5] = 0.0

    # ===== 7-11 Dense: distance & angle fields
    X, Y = build_meshgrid_xy()

    # Distance to ball
    C[:, :, 6] = np.sqrt((X - ball_x) ** 2 + (Y - ball_y) ** 2)

    # Distance to goal
    C[:, :, 7] = np.sqrt((X - goal_x) ** 2 + (Y - goal_y) ** 2)

    # Angle between (goal - cell) and (ball - cell): store sin & cos
    v_goal = np.stack([goal_x - X, goal_y - Y], axis=-1)
    v_ball = np.stack([ball_x - X, ball_y - Y], axis=-1)
    cos_t, sin_t = angle_between(v_goal, v_ball)
    C[:, :, 8] = sin_t.astype(np.float32)
    C[:, :, 9] = cos_t.astype(np.float32)

    # Absolute angle (radians) from cell to goal
    C[:, :, 10] = np.arctan2(goal_y - Y, goal_x - X).astype(np.float32)

    # ===== 12-13 Sparse: sin/cos of angle between actor velocity vector and vector to each teammate
    vx, vy = float(actor_v[0]), float(actor_v[1])
    v_actor = np.array([vx, vy], dtype=np.float32)
    v_actor_norm = np.linalg.norm(v_actor)

    if v_actor_norm < 1e-6:
        # If actor velocity is near zero, sin=0, cos=1 is a reasonable default for all teammates
        default_cos = 1.0
        default_sin = 0.0
    else:
        default_cos = None
        default_sin = None

    ax, ay = float(actor_loc[0]), float(actor_loc[1])

    for p in freeze_frame:
        if not bool(p.get("teammate")):
            continue
        loc = p.get("location")
        if not (isinstance(loc, (list, tuple)) and len(loc) >= 2):
            continue

        tx, ty = float(loc[0]), float(loc[1])
        gx, gy = to_grid_xy(tx, ty)

        v_to_tm = np.array([tx - ax, ty - ay], dtype=np.float32)
        if np.linalg.norm(v_to_tm) < 1e-6:
            continue

        if default_cos is not None:
            C[gy, gx, 11] = default_sin
            C[gy, gx, 12] = default_cos
        else:
            cos_a, sin_a = angle_between(v_actor.reshape(1, 2), v_to_tm.reshape(1, 2))
            C[gy, gx, 11] = float(sin_a[0])
            C[gy, gx, 12] = float(cos_a[0])

    return C


# =========================
# Plotting
# =========================
def save_channel_image(
        channel_2d: np.ndarray,
        out_path: str,
        title: str,
        description: str,
        actor_pt=None,
        teammate_pts=None,
        opponent_pts=None,
        value_unit: str = "",
        is_sparse: bool = False,
) -> None:
    """
    Save one channel as an image with:
    - axis labels (meters)
    - overlay markers (actor/teammates/opponents)
    - a short description box

    Note:
    - channel_2d is (GRID_H, GRID_W) with origin='lower'.
    - grid is 104x68 matching SoccerMap paper.
    """
    plt.figure(figsize=(8, 5))
    plt.imshow(channel_2d, origin="lower", aspect="auto")
    plt.title(title)
    cb = plt.colorbar()
    if value_unit:
        cb.set_label(value_unit)

    # Axis ticks mapped to meters for readability
    xticks = np.linspace(0, GRID_W - 1, 7).astype(int)
    yticks = np.linspace(0, GRID_H - 1, 6).astype(int)
    plt.xticks(xticks, [f"{(x/(GRID_W-1))*PITCH_LENGTH:.0f}" for x in xticks])
    plt.yticks(yticks, [f"{(y/(GRID_H-1))*PITCH_WIDTH:.0f}" for y in yticks])
    plt.xlabel("Pitch length x (meters, left → right)")
    plt.ylabel("Pitch width y (meters, bottom → top)")

    # Overlay markers (no need to change channel computation)
    # Use simple markers so the viewer knows what dots represent.
    if actor_pt is not None:
        plt.scatter([actor_pt[0]], [actor_pt[1]], marker="x", s=80, linewidths=2, label="Ball carrier (actor)")

    if teammate_pts:
        xs = [p[0] for p in teammate_pts]
        ys = [p[1] for p in teammate_pts]
        plt.scatter(xs, ys, marker="o", s=30, label="Teammates (non-actor)")

    if opponent_pts:
        xs = [p[0] for p in opponent_pts]
        ys = [p[1] for p in opponent_pts]
        plt.scatter(xs, ys, marker="s", s=30, label="Opponents")

    # Description box
    # Keep it short and consistent across channels.
    info_lines = [
        description,
        "Sparse channel: non-zero only at player locations." if is_sparse else "Dense channel: defined at every grid cell.",
    ]
    plt.gca().text(
        0.01, -0.18,
        "\n".join(info_lines),
        transform=plt.gca().transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    # Legend (only if markers exist)
    if (actor_pt is not None) or teammate_pts or opponent_pts:
        plt.legend(loc="upper right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def extract_overlay_points(pass_row: pd.Series):
    """
    Extract overlay points for plotting:
    - ball carrier (actor)
    - teammates
    - opponents

    Returns grid coordinates (gx, gy) for each group.
    """
    actor_loc = pass_row["location"]
    end_loc = pass_row.get("end_location")
    freeze_frame = pass_row.get("freeze_frame", [])

    # Use the same left-to-right normalization as channel creation (keep consistent)
    actor_loc, end_loc, freeze_frame, _ = normalize_frame_left_to_right(actor_loc, end_loc, freeze_frame)

    # Ball carrier point in grid
    ax, ay = float(actor_loc[0]), float(actor_loc[1])
    a_gx, a_gy = to_grid_xy(ax, ay)

    tm_pts = []
    opp_pts = []

    for p in freeze_frame:
        loc = p.get("location")
        if not (isinstance(loc, (list, tuple)) and len(loc) >= 2):
            continue
        px, py = float(loc[0]), float(loc[1])
        gx, gy = to_grid_xy(px, py)

        # Skip actor here; we plot actor separately
        if bool(p.get("actor")):
            continue

        if bool(p.get("teammate")):
            tm_pts.append((gx, gy))
        else:
            opp_pts.append((gx, gy))

    return (a_gx, a_gy), tm_pts, opp_pts

def main():
    base_dir = r"E:\R\open-data-master\data"
    match_id = "3788741"

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load & merge
    events, threesixty, _ = load_match_json(match_id, base_dir=base_dir)
    combined = merge_freeze_frames(events, threesixty)
    df = events_to_df(combined)

    # Keep only Pass events with valid locations and non-empty freeze_frame
    df_pass = df[
        (df["type"] == "Pass")
        & df["location"].notna()
        & df["end_location"].notna()
        & df["freeze_frame"].apply(lambda x: isinstance(x, list) and len(x) > 0)
        & df["player_id"].notna()
        ].copy()

    if len(df_pass) == 0:
        raise RuntimeError("No valid Pass events with freeze-frame found for this match_id.")

    # Estimate actor velocity vectors for all events
    df_vel = estimate_actor_velocity_vector(df, smooth_window=3)

    # Pick one pass to visualize
    pass_row = df_pass.iloc[0]
    ev_id = pass_row["event_id"]

    # Get actor velocity at this exact event
    row_v = df_vel[df_vel["event_id"] == ev_id]
    if len(row_v) == 0:
        actor_v = (0.0, 0.0)
    else:
        actor_v = (float(row_v.iloc[0]["vx"]), float(row_v.iloc[0]["vy"]))

    # Build 13 channels
    channels = build_13_channels_for_pass(pass_row, actor_v=actor_v)

    # Extract overlay points once
    actor_pt, teammate_pts, opponent_pts = extract_overlay_points(pass_row)

    channel_meta = [
        ("C1 attacker_presence",
         "Attacking team occupancy: 1 at each attacker grid cell (including actor).", "", True),
        ("C2 attacker_vx",
         "Attacker x-velocity (m/s) at attacker cells. With open-data, only actor vx is estimated; others may be 0.", "m/s", True),
        ("C3 attacker_vy",
         "Attacker y-velocity (m/s) at attacker cells. With open-data, only actor vy is estimated; others may be 0.", "m/s", True),
        ("C4 defender_presence",
         "Defending team occupancy: 1 at each defender grid cell.", "", True),
        ("C5 defender_vx",
         "Defender x-velocity (m/s) at defender cells. With open-data, usually 0 unless you implement defender velocity estimation.", "m/s", True),
        ("C6 defender_vy",
         "Defender y-velocity (m/s) at defender cells. With open-data, usually 0 unless you implement defender velocity estimation.", "m/s", True),
        ("C7 dist_to_ball",
         "Euclidean distance from each grid cell to the ball (actor location).", "meters", False),
        ("C8 dist_to_goal",
         "Euclidean distance from each grid cell to the opponent goal center (120, 40).", "meters", False),
        ("C9 sin(angle(goal,ball))",
         "sin of the angle between vectors (cell→goal) and (cell→ball).", "", False),
        ("C10 cos(angle(goal,ball))",
         "cos of the angle between vectors (cell→goal) and (cell→ball).", "", False),
        ("C11 angle_to_goal_rad",
         "Absolute angle from each grid cell pointing to the opponent goal (radians).", "radians", False),
        ("C12 sin(actor_vel_to_teammate)",
         "sin of the angle between actor velocity and vector (actor→teammate), placed at teammate cells.", "", True),
        ("C13 cos(actor_vel_to_teammate)",
         "cos of the angle between actor velocity and vector (actor→teammate), placed at teammate cells.", "", True),
    ]

    for i, (name, desc, unit, is_sparse) in enumerate(channel_meta):
        out_path = os.path.join(OUT_DIR, f"{i+1:02d}_{name}.png")
        save_channel_image(
            channels[:, :, i],
            out_path=out_path,
            title=name,
            description=desc,
            actor_pt=actor_pt,
            teammate_pts=teammate_pts,
            opponent_pts=opponent_pts,
            value_unit=unit,
            is_sparse=is_sparse,
        )

    print(f"Saved {len(channel_meta)} channel images to: {OUT_DIR}")

if __name__ == "__main__":
    main()


