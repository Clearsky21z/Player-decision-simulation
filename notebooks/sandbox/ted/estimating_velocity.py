"""
Velocity Features Module
Functions to estimate player movement speed and distance based on event data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union

# Constants
MAX_VALID_VELOCITY = 12.0  # Meters/second (approx 36 km/h, near world record speed)
MIN_TIME_DELTA = 0.5  # Minimum seconds to avoid division by zero artifacts


def calculate_euclidean_distance(loc1: List[float], loc2: List[float]) -> float:
    """
    Calculate Euclidean distance between two coordinate points.

    Args:
        loc1: [x, y] start location
        loc2: [x, y] end location

    Returns:
        Distance in pitch units (yards/meters depending on data source)
    """
    if loc1 is None or loc2 is None:
        return 0.0

    # Ensure inputs are valid lists/arrays with at least 2 elements
    if len(loc1) < 2 or len(loc2) < 2:
        return 0.0

    try:
        x1, y1 = float(loc1[0]), float(loc1[1])
        x2, y2 = float(loc2[0]), float(loc2[1])
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    except (ValueError, TypeError):
        return 0.0


def calculate_total_seconds(minute: int, second: int) -> float:
    """
    Convert match minute and second to total seconds.

    Args:
        minute: Match minute
        second: Match second

    Returns:
        Total seconds from match start
    """
    return minute * 60 + second


def estimate_velocity(df: pd.DataFrame,
                      smooth_window: int = 3) -> pd.DataFrame:
    """
    Estimate player velocity based on displacement between consecutive events.

    Since 'duration' is not in the loaded data, this function approximates
    velocity by calculating:
    (Distance to Next Event) / (Time to Next Event)

    Args:
        df: Events DataFrame containing 'player_id', 'location', 'minute', 'second'
        smooth_window: Window size for rolling average smoothing (default 3)

    Returns:
        DataFrame with added columns: 'dist_to_next', 'time_delta', 'estimated_velocity'
    """
    df = df.copy()

    # 1. Validation: Ensure required columns exist
    required_cols = ["player_id", "match_id", "minute", "second", "location"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' missing. Cannot calculate velocity.")
            return df

    # 2. Sort Data: Crucial for sequential calculation
    # We sort by match, player, and time to ensure events are ordered correctly
    df = df.sort_values(by=["match_id", "player_id", "minute", "second", "event_id"])

    # 3. Calculate Time Variables
    df["total_seconds"] = df.apply(
        lambda x: calculate_total_seconds(x["minute"], x["second"]), axis=1
    )

    # 4. Group by Player and Match to calculate deltas
    # We use transform to keep the original index and shape
    grouped = df.groupby(["match_id", "player_id"])

    # Calculate time to next event
    df["next_total_seconds"] = grouped["total_seconds"].shift(-1)
    df["time_delta"] = df["next_total_seconds"] - df["total_seconds"]

    # Calculate location of next event
    df["next_location"] = grouped["location"].shift(-1)

    # 5. Handle Carry Events (Special Case)
    # If the current event has an 'end_location' (like a Carry), the distance
    # traveled is primarily within the event itself.
    # Logic:
    # If end_location exists: dist = dist(location, end_location)
    # If no end_location: dist = dist(location, next_location)

    def get_displacement(row):
        # If we have an end_location (e.g., Carry), use it for distance
        if isinstance(row.get("end_location"), (list, tuple, np.ndarray)):
            return calculate_euclidean_distance(row["location"], row["end_location"])

        # Otherwise, calculate distance to the next event
        return calculate_euclidean_distance(row["location"], row["next_location"])

    df["dist_travelled"] = df.apply(get_displacement, axis=1)

    # 6. Calculate Velocity
    # We enforce a minimum time delta to avoid infinity/noise with simultaneous events
    df["valid_time_delta"] = df["time_delta"].apply(lambda x: max(x, MIN_TIME_DELTA) if pd.notnull(x) else None)

    df["estimated_velocity"] = df["dist_travelled"] / df["valid_time_delta"]

    # 7. Clean and Smooth Results
    # Filter unrealistic velocities (data errors or teleportation artifacts)
    df.loc[df["estimated_velocity"] > MAX_VALID_VELOCITY, "estimated_velocity"] = np.nan

    # Apply rolling smoothing if requested (fills NaNs and reduces noise)
    if smooth_window > 1:
        df["estimated_velocity"] = grouped["estimated_velocity"].transform(
            lambda x: x.rolling(window=smooth_window, min_periods=1).mean()
        )

    # Fill remaining NaNs (last events of halves) with 0
    df["estimated_velocity"] = df["estimated_velocity"].fillna(0.0)

    # Drop temporary calculation columns if desired, but keeping them for debugging
    # df = df.drop(columns=["next_total_seconds", "next_location", "valid_time_delta"])

    return df


def get_player_speed_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate aggregate speed statistics per player.

    Args:
        df: DataFrame with 'estimated_velocity' column

    Returns:
        DataFrame with player speed metrics
    """
    if "estimated_velocity" not in df.columns:
        return pd.DataFrame()

    stats = df.groupby(["player_id", "player"]).agg(
        avg_velocity=("estimated_velocity", "mean"),
        max_velocity=("estimated_velocity", "max"),
        total_distance=("dist_travelled", "sum"),
        events_count=("event_id", "count")
    ).reset_index()

    return stats.sort_values(by="max_velocity", ascending=False)