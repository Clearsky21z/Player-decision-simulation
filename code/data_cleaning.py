import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


def validate_location(location, pitch_length: int = 120, pitch_width: int = 80) -> bool:
    """
    Validate that a location is within pitch bounds and properly formatted.
    
    Args:
        location: Location as [x, y] or None
        pitch_length: Maximum x coordinate
        pitch_width: Maximum y coordinate
    
    Returns:
        True if location is valid, False otherwise
    """
    if location is None:
        return False
    
    if not isinstance(location, (list, tuple, np.ndarray)):
        return False
    
    if len(location) < 2:
        return False
    
    try:
        x, y = float(location[0]), float(location[1])
        return 0 <= x <= pitch_length and 0 <= y <= pitch_width
    except (ValueError, TypeError):
        return False


def validate_freeze_frame(freeze_frame) -> bool:
    """
    Validate that freeze_frame data is properly formatted.
    
    Args:
        freeze_frame: Freeze frame list or None
    
    Returns:
        True if freeze_frame is valid, False otherwise
    """
    if freeze_frame is None:
        return False
    
    if not isinstance(freeze_frame, (list, tuple)):
        return False
    
    if len(freeze_frame) == 0:
        return False
    
    # Check that at least one entry has a location
    has_location = any(
        isinstance(p, dict) and p.get("location") is not None 
        for p in freeze_frame
    )
    
    return has_location


def filter_valid_events(df: pd.DataFrame, 
                        event_types: Optional[List[str]] = None,
                        require_location: bool = True,
                        require_end_location: bool = True,
                        require_freeze_frame: bool = True,
                        min_freeze_frame_size: int = 1) -> pd.DataFrame:
    """
    Filter DataFrame to keep only valid events with required data.
    
    Args:
        df: Events DataFrame
        event_types: List of event types to keep (None = all types)
        require_location: Require start location
        require_end_location: Require end location
        require_freeze_frame: Require freeze frame data
        min_freeze_frame_size: Minimum number of players in freeze frame
    
    Returns:
        Filtered DataFrame
    """
    valid_df = df.copy()
    
    # Filter by event type
    if event_types:
        valid_df = valid_df[valid_df["type"].isin(event_types)].copy()
    
    # Filter by location
    if require_location:
        valid_df = valid_df[
            valid_df["location"].apply(lambda loc: validate_location(loc))
        ].copy()
    
    # Filter by end_location
    if require_end_location:
        valid_df = valid_df[
            valid_df["end_location"].notna() & 
            valid_df["end_location"].apply(lambda loc: validate_location(loc))
        ].copy()
    
    # Filter by freeze_frame
    if require_freeze_frame:
        valid_df = valid_df[
            valid_df["freeze_frame"].apply(
                lambda ff: validate_freeze_frame(ff) and len(ff) >= min_freeze_frame_size
            )
        ].copy()
    
    return valid_df


def identify_edge_cases(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Identify edge cases that might need special handling.
    
    Args:
        df: Events DataFrame
    
    Returns:
        Dictionary with keys:
        - 'penalty_box': Events in penalty box
        - 'set_pieces': Free kicks, corners, throw-ins
        - 'goalkeeper': Events by goalkeepers
        - 'out_of_bounds': Events with locations out of bounds (shouldn't happen after filtering)
    """
    edge_cases = {}
    
    # Penalty box events (x > 88 or x < 12, y between 18 and 62 for StatsBomb)
    if len(df) > 0 and "location" in df.columns:
        penalty_box_mask = df["location"].apply(
            lambda loc: loc is not None and len(loc) >= 2 and (
                (loc[0] > 88 or loc[0] < 12) and 18 < loc[1] < 62
            )
        )
        edge_cases["penalty_box"] = df[penalty_box_mask].copy()
    
    # Set pieces
    set_piece_types = ["Free Kick", "Corner", "Throw-in", "Kick Off"]
    if "type" in df.columns:
        edge_cases["set_pieces"] = df[df["type"].isin(set_piece_types)].copy()
    
    # Goalkeeper events (filter by position if available, or by player name pattern)
    if "player" in df.columns:
        # This is a heuristic - might need adjustment
        gk_pattern = df["player"].str.contains("keeper|GK|Goalkeeper", case=False, na=False)
        edge_cases["goalkeeper"] = df[gk_pattern].copy()
    
    # Out of bounds (shouldn't exist after validation, but check anyway)
    if len(df) > 0 and "location" in df.columns:
        out_of_bounds = df[
            ~df["location"].apply(lambda loc: validate_location(loc))
        ].copy()
        edge_cases["out_of_bounds"] = out_of_bounds
    
    return edge_cases


def handle_missing_data(df: pd.DataFrame, 
                        fill_strategy: str = "drop") -> pd.DataFrame:
    """
    Handle missing data in the DataFrame.
    
    Args:
        df: Events DataFrame
        fill_strategy: Strategy for handling missing data
            - "drop": Drop rows with missing critical data
            - "forward_fill": Forward fill where appropriate (not recommended for locations)
    
    Returns:
        DataFrame with missing data handled
    """
    cleaned_df = df.copy()
    
    if fill_strategy == "drop":
        # Drop rows with missing critical columns
        critical_cols = ["match_id", "team", "type", "location"]
        for col in critical_cols:
            if col in cleaned_df.columns:
                cleaned_df = cleaned_df[cleaned_df[col].notna()].copy()
    
    elif fill_strategy == "forward_fill":
        # Only forward fill non-spatial data (not recommended for locations)
        fillable_cols = ["minute", "second"]
        for col in fillable_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(method="ffill")
    
    return cleaned_df


def validate_data_consistency(df: pd.DataFrame, 
                             match_ids: List[str]) -> Dict[str, any]:
    """
    Validate data consistency across matches.
    
    Args:
        df: Events DataFrame
        match_ids: List of expected match IDs
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        "matches_found": df["match_id"].nunique() if len(df) > 0 else 0,
        "matches_expected": len(match_ids),
        "missing_matches": [],
        "events_per_match": {},
        "coverage_issues": [],
    }
    
    if len(df) == 0:
        validation["coverage_issues"].append("No events found in DataFrame")
        return validation
    
    # Check for missing matches
    found_match_ids = set(df["match_id"].unique())
    expected_match_ids = set(match_ids)
    validation["missing_matches"] = list(expected_match_ids - found_match_ids)
    
    # Events per match
    match_counts = df["match_id"].value_counts()
    validation["events_per_match"] = match_counts.to_dict()
    
    # Check for matches with suspiciously low event counts
    if len(match_counts) > 0:
        mean_events = match_counts.mean()
        std_events = match_counts.std()
        low_threshold = mean_events - 2 * std_events
        
        low_coverage = match_counts[match_counts < low_threshold]
        if len(low_coverage) > 0:
            validation["coverage_issues"].append(
                f"Matches with low event counts: {low_coverage.to_dict()}"
            )
    
    return validation
