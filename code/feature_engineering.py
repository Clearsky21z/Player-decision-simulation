"""
Feature Engineering Module (6.3)
Functions to extract context features for decision-making analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union

PITCH_LENGTH = 120
PITCH_WIDTH = 80


def get_pitch_zone(x: float, y: float, 
                  pitch_length: int = PITCH_LENGTH,
                  pitch_width: int = PITCH_WIDTH) -> str:
    """
    Divide pitch into zones: defensive, middle, attacking.
    """
    if x < pitch_length / 3:
        return "defensive"
    elif x < 2 * pitch_length / 3:
        return "middle"
    else:
        return "attacking"


def encode_pitch_zone(x: float, pitch_length: int = PITCH_LENGTH) -> np.ndarray:
    """
    One-hot encode pitch zone.
    
    Args:
        x: X coordinate
        pitch_length: Maximum X coordinate
    
    Returns:
        One-hot encoded array [defensive, middle, attacking]
    """
    zone = get_pitch_zone(x, 0, pitch_length)
    encoding = np.zeros(3)
    
    if zone == "defensive":
        encoding[0] = 1
    elif zone == "middle":
        encoding[1] = 1
    else:  # attacking
        encoding[2] = 1
    
    return encoding


def calculate_pressure_metrics(freeze_frame: List[Dict], 
                               actor_location: List[float],
                               distance_threshold: float = 10.0) -> Dict[str, float]:
    """
    Calculate pressure metrics from freeze-frame data.
    
    Args:
        freeze_frame: List of player positions in freeze-frame
        actor_location: [x, y] location of actor
        distance_threshold: Distance threshold for "nearby" (default 10 units)
    
    Returns:
        Dictionary with:
        - opponent_count: Number of nearby opponents
        - teammate_count: Number of nearby teammates
        - nearest_opponent_dist: Distance to nearest opponent
        - nearest_teammate_dist: Distance to nearest teammate
        - pressure_ratio: opponent_count / (opponent_count + teammate_count + 1)
    """
    if actor_location is None or len(actor_location) < 2:
        return {
            "opponent_count": 0.0,
            "teammate_count": 0.0,
            "nearest_opponent_dist": np.inf,
            "nearest_teammate_dist": np.inf,
            "pressure_ratio": 0.0,
        }
    
    actor_x, actor_y = float(actor_location[0]), float(actor_location[1])
    
    opponent_distances = []
    teammate_distances = []
    opponent_count = 0
    teammate_count = 0
    
    for p in freeze_frame:
        loc = p.get("location")
        if not loc or len(loc) < 2:
            continue
        
        try:
            px, py = float(loc[0]), float(loc[1])
            dist = np.sqrt((px - actor_x)**2 + (py - actor_y)**2)
            
            if dist < distance_threshold:
                if p.get("teammate"):
                    teammate_count += 1
                    teammate_distances.append(dist)
                elif not p.get("actor"):  # Not the actor
                    opponent_count += 1
                    opponent_distances.append(dist)
        except (ValueError, TypeError):
            continue
    
    nearest_opponent_dist = min(opponent_distances) if opponent_distances else np.inf
    nearest_teammate_dist = min(teammate_distances) if teammate_distances else np.inf
    
    total_nearby = opponent_count + teammate_count
    pressure_ratio = opponent_count / (total_nearby + 1) if total_nearby > 0 else 0.0
    
    return {
        "opponent_count": float(opponent_count),
        "teammate_count": float(teammate_count),
        "nearest_opponent_dist": float(nearest_opponent_dist) if nearest_opponent_dist != np.inf else 999.0,
        "nearest_teammate_dist": float(nearest_teammate_dist) if nearest_teammate_dist != np.inf else 999.0,
        "pressure_ratio": pressure_ratio,
    }


def count_players_in_zones(freeze_frame: List[Dict],
                          actor_location: List[float],
                          pitch_length: int = PITCH_LENGTH) -> Dict[str, int]:
    """
    Count teammates and opponents in different pitch zones.
    
    Args:
        freeze_frame: List of player positions
        actor_location: [x, y] location of actor
        pitch_length: Maximum X coordinate
    
    Returns:
        Dictionary with counts for each zone and player type
    """
    zone_counts = {
        "teammates_defensive": 0,
        "teammates_middle": 0,
        "teammates_attacking": 0,
        "opponents_defensive": 0,
        "opponents_middle": 0,
        "opponents_attacking": 0,
    }
    
    for p in freeze_frame:
        if p.get("actor"):
            continue
        
        loc = p.get("location")
        if not loc or len(loc) < 2:
            continue
        
        try:
            x = float(loc[0])
            zone = get_pitch_zone(x, 0, pitch_length)
            
            key_prefix = "teammates" if p.get("teammate") else "opponents"
            key = f"{key_prefix}_{zone}"
            
            if key in zone_counts:
                zone_counts[key] += 1
        except (ValueError, TypeError):
            continue
    
    return zone_counts


def normalize_game_time(minute: int, second: int = 0, 
                        match_duration: int = 90) -> float:
    """
    Normalize game time to [0, 1].
    
    Args:
        minute: Match minute
        second: Match second
        match_duration: Expected match duration in minutes
    
    Returns:
        Normalized time in [0, 1]
    """
    total_seconds = minute * 60 + second
    max_seconds = match_duration * 60
    return min(1.0, total_seconds / max_seconds)


def get_match_phase(minute: int) -> str:
    """
    Categorize match phase based on minute.
    
    Args:
        minute: Match minute
    
    Returns:
        Phase: "early" (0-30), "mid" (30-60), "late" (60-90), "extra" (90+)
    """
    if minute < 30:
        return "early"
    elif minute < 60:
        return "mid"
    elif minute < 90:
        return "late"
    else:
        return "extra"


def encode_match_phase(minute: int) -> np.ndarray:
    """
    One-hot encode match phase.
    
    Args:
        minute: Match minute
    
    Returns:
        One-hot encoded array [early, mid, late, extra]
    """
    phase = get_match_phase(minute)
    encoding = np.zeros(4)
    
    phase_map = {"early": 0, "mid": 1, "late": 2, "extra": 3}
    encoding[phase_map[phase]] = 1
    
    return encoding


def extract_context_features(event_row: pd.Series, 
                            context_dim: int = 16) -> np.ndarray:
    """
    Extract comprehensive context features for decision-making.
    
    This is the main function for feature engineering. It combines:
    - Pitch zone encoding
    - Pressure metrics
    - Passing options
    - Game state
    - Match phase
    
    Args:
        event_row: Series or dict with event data
        context_dim: Dimension of context feature vector
    
    Returns:
        numpy array of shape (context_dim,)
    """
    features = np.zeros(context_dim)
    idx = 0
    
    # Pitch zone (one-hot: defensive, middle, attacking) - 3 features
    location = event_row.get("location")
    if location and isinstance(location, (list, tuple)) and len(location) >= 2:
        x = float(location[0])
        zone_encoding = encode_pitch_zone(x)
        features[idx:idx+3] = zone_encoding
    idx += 3
    
    # Pressure metrics - 5 features
    freeze_frame = event_row.get("freeze_frame", [])
    if freeze_frame and location:
        pressure = calculate_pressure_metrics(freeze_frame, location)
        features[idx] = min(pressure["opponent_count"] / 5.0, 1.0)  # Normalized opponent count
        features[idx+1] = min(pressure["teammate_count"] / 5.0, 1.0)  # Normalized teammate count
        features[idx+2] = min(pressure["nearest_opponent_dist"] / 20.0, 1.0)  # Normalized nearest opponent
        features[idx+3] = pressure["pressure_ratio"]  # Pressure ratio
        features[idx+4] = min(pressure["nearest_teammate_dist"] / 20.0, 1.0) if pressure["nearest_teammate_dist"] < 999 else 1.0
    idx += 5
    
    # Game time (normalized) - 1 feature
    minute = event_row.get("minute", 45)
    second = event_row.get("second", 0)
    features[idx] = normalize_game_time(minute, second)
    idx += 1
    
    # Match phase (one-hot: early, mid, late, extra) - 4 features
    if idx + 4 <= context_dim:
        phase_encoding = encode_match_phase(minute)
        features[idx:idx+4] = phase_encoding
        idx += 4
    
    # Passing options in different zones - 6 features (if space available)
    if idx + 6 <= context_dim and freeze_frame:
        zone_counts = count_players_in_zones(freeze_frame, location)
        # Normalize counts
        features[idx] = min(zone_counts.get("teammates_defensive", 0) / 5.0, 1.0)
        features[idx+1] = min(zone_counts.get("teammates_middle", 0) / 5.0, 1.0)
        features[idx+2] = min(zone_counts.get("teammates_attacking", 0) / 5.0, 1.0)
        features[idx+3] = min(zone_counts.get("opponents_defensive", 0) / 5.0, 1.0)
        features[idx+4] = min(zone_counts.get("opponents_middle", 0) / 5.0, 1.0)
        features[idx+5] = min(zone_counts.get("opponents_attacking", 0) / 5.0, 1.0)
    
    return features


def add_pitch_zones_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pitch zone columns to DataFrame.
    
    Args:
        df: Events DataFrame
    
    Returns:
        DataFrame with added 'start_zone' and 'end_zone' columns
    """
    df = df.copy()
    
    def get_zone_from_location(loc):
        if loc is None or not isinstance(loc, (list, tuple)) or len(loc) < 2:
            return None
        return get_pitch_zone(loc[0], loc[1])
    
    if "location" in df.columns:
        df["start_zone"] = df["location"].apply(get_zone_from_location)
    
    if "end_location" in df.columns:
        df["end_zone"] = df["end_location"].apply(get_zone_from_location)
    
    return df


def add_context_features_to_df(df: pd.DataFrame, 
                               context_dim: int = 16) -> pd.DataFrame:
    """
    Add context features to DataFrame.
    
    Args:
        df: Events DataFrame
        context_dim: Dimension of context feature vector
    
    Returns:
        DataFrame with added 'context_features' column
    """
    df = df.copy()
    
    context_features = []
    for _, row in df.iterrows():
        features = extract_context_features(row, context_dim=context_dim)
        context_features.append(features)
    
    df["context_features"] = context_features
    
    return df


def create_player_id_mapping(df: pd.DataFrame) -> Dict[str, int]:
    """
    Create mapping from player names to unique integer IDs.
    
    Args:
        df: Events DataFrame with 'player' column
    
    Returns:
        Dictionary mapping player names to integer IDs
    """
    if "player" not in df.columns:
        return {}
    
    unique_players = df["player"].dropna().unique()
    player_to_id = {player: idx for idx, player in enumerate(sorted(unique_players))}
    
    return player_to_id


def add_player_ids_to_df(df: pd.DataFrame, 
                         player_id_mapping: Dict[str, int]) -> pd.DataFrame:
    """
    Add player_id column to DataFrame using mapping.
    
    Args:
        df: Events DataFrame
        player_id_mapping: Dictionary mapping player names to IDs
    
    Returns:
        DataFrame with added 'player_id' column
    """
    df = df.copy()
    
    def map_player_id(player_name):
        if pd.isna(player_name):
            return None
        return player_id_mapping.get(player_name)
    
    df["player_id"] = df["player"].apply(map_player_id)
    
    return df

