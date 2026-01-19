
from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from load_data import load_match_data


PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

GRID_W = 104  # x axis (length)
GRID_H = 68   # y axis (width)

CHANNEL_NAMES = [
    "C1 attacker_presence",
    "C2 attacker_vx",
    "C3 attacker_vy",
    "C4 defender_presence",
    "C5 defender_vx",
    "C6 defender_vy",
    "C7 dist_to_ball",
    "C8 dist_to_goal",
    "C9 sin(angle(goal,ball))",
    "C10 cos(angle(goal,ball))",
    "C11 angle_to_goal_rad",
    "C12 sin(actor_vel_to_teammate)",
    "C13 cos(actor_vel_to_teammate)",
]



# Utilities

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def to_grid_xy(x: float, y: float) -> Tuple[int, int]:
    """Map StatsBomb coordinates (0..120, 0..80) to grid indices (gx,gy)."""
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
) -> Tuple[List[float], Optional[List[float]], List[Dict]]:
    """
    Heuristic normalization:
    If end_x < start_x, flip all coordinates (actor, end_loc, freeze-frame players).
    """
    flipped = False
    if end_loc is not None and len(end_loc) >= 2:
        if float(end_loc[0]) < float(actor_loc[0]):
            flipped = True

    if not flipped:
        return actor_loc, end_loc, freeze_frame

    actor_loc2 = flip_lr_location(actor_loc)
    end_loc2 = flip_lr_location(end_loc) if end_loc is not None else None

    ff2 = []
    for p in freeze_frame:
        p2 = dict(p)
        if p2.get("location") is not None and len(p2["location"]) >= 2:
            p2["location"] = flip_lr_location(p2["location"])
        ff2.append(p2)

    return actor_loc2, end_loc2, ff2


def build_meshgrid_xy() -> Tuple[np.ndarray, np.ndarray]:
    """Return meshgrid arrays for cell centers in metric coordinates."""
    xs = np.linspace(0, PITCH_LENGTH, GRID_W, dtype=np.float32)
    ys = np.linspace(0, PITCH_WIDTH, GRID_H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)  # (y,x)
    return X, Y


def angle_between(u: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cos(theta) and sin(theta) between vectors u and v.
    u, v shapes: (...,2)
    """
    dot = np.sum(u * v, axis=-1)
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    denom = (nu * nv) + eps
    cos_t = dot / denom
    cos_t = np.clip(cos_t, -1.0, 1.0)

    cross_z = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t**2))
    sin_t = np.where(cross_z >= 0, sin_t, -sin_t)
    return cos_t, sin_t



# Parsing: combined events -> DataFrame (self-contained)

def events_to_df(combined_events: List[Dict]) -> pd.DataFrame:
    """
    Build a light events DataFrame with fields needed for:
    - pass filtering
    - pass_outcome label
    - velocity estimation
    - channel building
    """
    rows = []
    for ev in combined_events:
        ev_type = (ev.get("type") or {}).get("name")

        player_dict = ev.get("player") or {}
        player_id = player_dict.get("id")
        player_name = player_dict.get("name")

        row = {
            "event_id": ev.get("id"),
            "match_id": ev.get("match_id"),
            "team": (ev.get("team") or {}).get("name"),
            "player": player_name,
            "player_id": player_id,
            "type": ev_type,
            "minute": ev.get("minute"),
            "second": ev.get("second"),
            "location": ev.get("location"),
            "freeze_frame": ev.get("freeze_frame", []),
        }

        if ev_type == "Pass":
            pinfo = ev.get("pass") or {}
            row["end_location"] = pinfo.get("end_location")
            # StatsBomb: completed pass has no "outcome"; unsuccessful pass has "outcome"
            row["pass_outcome"] = 0 if pinfo.get("outcome") is not None else 1
        else:
            row["end_location"] = None
            row["pass_outcome"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)



# Velocity estimation (actor-only)

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
    """
    df = events_df.copy()

    df = df[df["player_id"].notna() & df["location"].notna()].copy()
    df["t"] = df.apply(lambda r: total_seconds(r["minute"], r["second"]), axis=1)
    df = df.sort_values(["match_id", "player_id", "t", "event_id"]).reset_index(drop=True)

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

        if spd > max_speed:
            return (0.0, 0.0, 0.0)

        return (vx, vy, spd)

    vecs = df.apply(safe_vec, axis=1, result_type="expand")
    df["vx"] = vecs[0].astype(float)
    df["vy"] = vecs[1].astype(float)
    df["speed"] = vecs[2].astype(float)

    if smooth_window > 1:
        df["vx"] = g["vx"].transform(lambda x: x.rolling(smooth_window, min_periods=1).mean())
        df["vy"] = g["vy"].transform(lambda x: x.rolling(smooth_window, min_periods=1).mean())
        df["speed"] = g["speed"].transform(lambda x: x.rolling(smooth_window, min_periods=1).mean())

    return df



# 13-channel construction

def build_13_channels_for_pass(pass_row: pd.Series, actor_v: Tuple[float, float]) -> np.ndarray:
    """
    Build channels on 104x68 grid.
    Returns: (GRID_H, GRID_W, 13)
    """
    C = np.zeros((GRID_H, GRID_W, 13), dtype=np.float32)

    actor_loc = pass_row["location"]
    end_loc = pass_row.get("end_location")
    freeze_frame = pass_row.get("freeze_frame", [])

    if not (isinstance(actor_loc, (list, tuple)) and len(actor_loc) >= 2):
        raise ValueError("Invalid actor location in pass row.")

    actor_loc, end_loc, freeze_frame = normalize_frame_left_to_right(actor_loc, end_loc, freeze_frame)

    ball_x, ball_y = float(actor_loc[0]), float(actor_loc[1])
    goal_x, goal_y = PITCH_LENGTH, PITCH_WIDTH / 2.0

    # Channels 1-6 (sparse): positions (+ velocities where available)
    # Only estimate actor velocity; others remain 0 in velocity channels.
    for p in freeze_frame:
        loc = p.get("location")
        if not (isinstance(loc, (list, tuple)) and len(loc) >= 2):
            continue

        px, py = float(loc[0]), float(loc[1])
        gx, gy = to_grid_xy(px, py)

        is_teammate = bool(p.get("teammate"))
        is_actor = bool(p.get("actor"))

        if is_teammate or is_actor:
            C[gy, gx, 0] = 1.0  # attacker presence
            if is_actor:
                C[gy, gx, 1] = float(actor_v[0])
                C[gy, gx, 2] = float(actor_v[1])
        else:
            C[gy, gx, 3] = 1.0  # defender presence

    # Channels 7-11 (dense): distance/angle fields
    X, Y = build_meshgrid_xy()

    C[:, :, 6] = np.sqrt((X - ball_x) ** 2 + (Y - ball_y) ** 2)  # dist to ball
    C[:, :, 7] = np.sqrt((X - goal_x) ** 2 + (Y - goal_y) ** 2)  # dist to goal

    v_goal = np.stack([goal_x - X, goal_y - Y], axis=-1)
    v_ball = np.stack([ball_x - X, ball_y - Y], axis=-1)
    cos_t, sin_t = angle_between(v_goal, v_ball)
    C[:, :, 8] = sin_t.astype(np.float32)
    C[:, :, 9] = cos_t.astype(np.float32)

    C[:, :, 10] = np.arctan2(goal_y - Y, goal_x - X).astype(np.float32)

    # Channels 12-13 (sparse): angle between actor velocity and vector to each teammate
    vx, vy = float(actor_v[0]), float(actor_v[1])
    v_actor = np.array([vx, vy], dtype=np.float32)
    v_actor_norm = np.linalg.norm(v_actor)

    ax, ay = float(actor_loc[0]), float(actor_loc[1])

    for p in freeze_frame:
        if not bool(p.get("teammate")):
            continue
        if bool(p.get("actor")):
            continue

        loc = p.get("location")
        if not (isinstance(loc, (list, tuple)) and len(loc) >= 2):
            continue

        tx, ty = float(loc[0]), float(loc[1])
        gx, gy = to_grid_xy(tx, ty)

        v_to_tm = np.array([tx - ax, ty - ay], dtype=np.float32)
        if np.linalg.norm(v_to_tm) < 1e-6:
            continue

        if v_actor_norm < 1e-6:
            C[gy, gx, 11] = 0.0
            C[gy, gx, 12] = 1.0
        else:
            cos_a, sin_a = angle_between(v_actor.reshape(1, 2), v_to_tm.reshape(1, 2))
            C[gy, gx, 11] = float(sin_a[0])
            C[gy, gx, 12] = float(cos_a[0])

    return C



# Feature extraction (per-event summary)

def channel_summary_vector(channels: np.ndarray) -> np.ndarray:
    """
    Convert (H,W,13) -> 13 scalars for correlation/importance analysis.
    """
    v = np.zeros((13,), dtype=np.float32)

    # Presence: sum
    v[0] = float(np.sum(channels[:, :, 0]))
    v[3] = float(np.sum(channels[:, :, 3]))

    # Velocities: mean abs
    v[1] = float(np.mean(np.abs(channels[:, :, 1])))
    v[2] = float(np.mean(np.abs(channels[:, :, 2])))
    v[4] = float(np.mean(np.abs(channels[:, :, 4])))
    v[5] = float(np.mean(np.abs(channels[:, :, 5])))

    # Dense: mean
    v[6] = float(np.mean(channels[:, :, 6]))
    v[7] = float(np.mean(channels[:, :, 7]))
    v[8] = float(np.mean(channels[:, :, 8]))
    v[9] = float(np.mean(channels[:, :, 9]))
    v[10] = float(np.mean(channels[:, :, 10]))

    # Relational: mean abs
    v[11] = float(np.mean(np.abs(channels[:, :, 11])))
    v[12] = float(np.mean(np.abs(channels[:, :, 12])))

    return v


def point_biserial_corr(x: np.ndarray, y01: np.ndarray) -> float:
    """Pearson correlation between continuous x and binary y in {0,1}."""
    if np.std(x) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y01)[0, 1])



# Plot

def plot_corr_heatmap(corr: np.ndarray, labels: List[str], title: str, save_path: str) -> None:
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, origin="lower", aspect="equal")
    plt.title(title)
    plt.colorbar(label="Pearson correlation")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_importance_bar(scores: np.ndarray, labels: List[str], title: str, save_path: str) -> None:
    """
    Plot channel importance in fixed C1 -> C13 order (no sorting).
    """
    abs_scores = np.abs(scores)
    x = np.arange(len(labels))

    plt.figure(figsize=(10, 5))
    plt.bar(x, abs_scores)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("|Correlation with pass_outcome|")
    plt.title(title)

    # annotate sign (+ / -)
    for i in range(len(scores)):
        sgn = "+" if scores[i] >= 0 else "-"
        plt.text(i, abs_scores[i], sgn, ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



# Main pipeline

def build_event_table(
        base_dir: str,
        match_ids: List[str],
        max_passes_total: int = 300,
        progress_every: int = 25,
) -> pd.DataFrame:
    """
    Build (n_events, 13) summary feature table for pass events.
    """
    rows = []
    passes_collected = 0

    for match_id in match_ids:
        print(f"\n[INFO] Loading match_id={match_id}")
        combined = load_match_data(match_id, base_dir=base_dir)
        df = events_to_df(combined)

        df_pass = df[
            (df["type"] == "Pass")
            & df["location"].notna()
            & df["end_location"].notna()
            & df["freeze_frame"].apply(lambda x: isinstance(x, list) and len(x) > 0)
            & df["player_id"].notna()
            & df["pass_outcome"].notna()
            ].copy()

        print(f"[INFO] Pass events with freeze-frame: {len(df_pass)}")
        if len(df_pass) == 0:
            continue

        df_vel = estimate_actor_velocity_vector(df, smooth_window=3)

        for _, pass_row in df_pass.iterrows():
            ev_id = pass_row["event_id"]

            row_v = df_vel[df_vel["event_id"] == ev_id]
            if len(row_v) == 0:
                actor_v = (0.0, 0.0)
            else:
                actor_v = (float(row_v.iloc[0]["vx"]), float(row_v.iloc[0]["vy"]))

            channels = build_13_channels_for_pass(pass_row, actor_v=actor_v)
            feat = channel_summary_vector(channels)

            out = {CHANNEL_NAMES[i]: float(feat[i]) for i in range(13)}
            out["pass_outcome"] = int(pass_row["pass_outcome"])
            out["match_id"] = str(match_id)
            out["event_id"] = ev_id
            out["player_id"] = int(pass_row["player_id"])
            rows.append(out)

            passes_collected += 1

            if passes_collected % progress_every == 0:
                print(f"[INFO] Processed {passes_collected} pass events...")

            if passes_collected >= max_passes_total:
                break

        if passes_collected >= max_passes_total:
            break

    if len(rows) == 0:
        raise RuntimeError("No pass rows collected. Check base_dir and match_ids (must have 360).")

    print(f"\n[INFO] Finished. Total pass events used: {len(rows)}")
    return pd.DataFrame(rows)


def main():

    # Must contain: events/, three-sixty/, (lineups/ optional)
    base_dir = r"E:\R\open-data-master\data"

    # ---- Match IDs that exist locally and have 360 ----
    match_ids = [
        "3788741",
    ]

    # If it feels slow, reduce this to 50/100 first
    max_passes_total = 250

    df_feat = build_event_table(
        base_dir=base_dir,
        match_ids=match_ids,
        max_passes_total=max_passes_total,
        progress_every=25,
    )

    X = df_feat[CHANNEL_NAMES].values.astype(np.float32)
    corr = np.corrcoef(X, rowvar=False)

    heatmap_path = "channel_correlation_heatmap.png"
    plot_corr_heatmap(
        corr,
        CHANNEL_NAMES,
        title="Channel-Channel Correlation (per-event summary features)",
        save_path=heatmap_path,
    )
    print(f"[INFO] Saved heatmap to: {os.path.abspath(heatmap_path)}")

    y = df_feat["pass_outcome"].values.astype(np.float32)
    imp = np.array([point_biserial_corr(X[:, i], y) for i in range(13)], dtype=np.float32)

    bar_path = "channel_importance_bar.png"
    plot_importance_bar(
        imp,
        CHANNEL_NAMES,
        title="Channel Importance Proxy: |corr with pass_outcome|",
        save_path=bar_path,
    )
    print(f"[INFO] Saved importance bar plot to: {os.path.abspath(bar_path)}")

    # Print top correlated pairs
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)
    pairs = []
    for i in range(13):
        for j in range(i + 1, 13):
            pairs.append((abs_corr[i, j], i, j))
    pairs.sort(reverse=True)

    print("\nTop 10 most correlated channel pairs (by |corr|):")
    for k in range(min(10, len(pairs))):
        val, i, j = pairs[k]
        print(f"{k+1:02d}. |corr|={val:.3f}  {CHANNEL_NAMES[i]}  vs  {CHANNEL_NAMES[j]}")

    print("\nTop channels by |corr with pass_outcome|:")
    order = np.argsort(np.abs(imp))[::-1]
    for k in range(13):
        i = order[k]
        print(f"{k+1:02d}. corr={imp[i]:+.3f}  {CHANNEL_NAMES[i]}")

    out_csv = "channel_summary_features.csv"
    df_feat.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved feature table to: {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    main()




