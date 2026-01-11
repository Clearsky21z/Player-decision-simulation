"""
Data Loading Module
Functions to load matches for any team, competition, and season.
Has single-match and multi-match collection functionality.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict


# ============================================================================
# SINGLE MATCH LOADING FUNCTIONS
# ============================================================================

def load_match_data(match_id, base_dir="../data/open-data/data"):
    """
    Load and join Events + 360 freeze-frame data for a single match.
    
    Args:
        match_id: Match ID (string or int)
        base_dir: Base directory for StatsBomb data
    
    Returns:
        List of combined event dictionaries with freeze-frame data attached
    """
    base = Path(base_dir)
    events_file = base / "events" / f"{match_id}.json"
    three_sixty_file = base / "three-sixty" / f"{match_id}.json"

    # Load events
    with open(events_file, "r", encoding="utf-8") as f:
        events = json.load(f)

    # Load freeze frames
    with open(three_sixty_file, "r", encoding="utf-8") as f:
        freeze_frames = json.load(f)

    # Build lookup dict for freeze frames
    freeze_lookup = {ff["event_uuid"]: ff for ff in freeze_frames}

    # Merge where possible
    combined = []
    for ev in events:
        ev_id = ev.get("id")
        if ev_id in freeze_lookup:
            ff = freeze_lookup[ev_id]
            ev["freeze_frame"] = ff.get("freeze_frame", [])
            ev["visible_area"] = ff.get("visible_area", [])
        combined.append(ev)

    return combined


def build_events_dataframe(combined):
    """
    Build events DataFrame from combined event list.
    One row per event.
    
    Args:
        combined: List of combined event dictionaries
    
    Returns:
        DataFrame with one row per event
    """
    rows = []
    for ev in combined:
        row = {
            "event_id": ev.get("id"),
            "match_id": ev.get("match_id"),
            "team": ev.get("team", {}).get("name"),
            "player": ev.get("player", {}).get("name"),
            "type": ev.get("type", {}).get("name"),
            "minute": ev.get("minute"),
            "second": ev.get("second"),
            "location": ev.get("location"),  # [x,y]
            "freeze_frame": ev.get("freeze_frame", []),
            "visible_area": ev.get("visible_area", []),
        }

        # For Pass, Carry, Shot: add end_location
        if row["type"] == "Pass":
            row["end_location"] = ev.get("pass", {}).get("end_location")
        elif row["type"] == "Carry":
            row["end_location"] = ev.get("carry", {}).get("end_location")
        elif row["type"] == "Shot":
            row["end_location"] = ev.get("shot", {}).get("end_location")

        rows.append(row)

    return pd.DataFrame(rows)


def build_freeze_frames_dataframe(events_df):
    """
    Build freeze-frames DataFrame from events DataFrame.
    One row per player in each freeze frame.
    
    Args:
        events_df: Events DataFrame with freeze_frame column
    
    Returns:
        DataFrame with one row per player in each freeze frame
    """
    rows = []
    for _, ev in events_df.iterrows():
        event_id = ev["event_id"]
        team = ev["team"]
        player = ev.get("player")  # actor name from events JSON
        ev_type = ev["type"]
        minute, second = ev["minute"], ev["second"]
        location = ev["location"]
        end_location = ev.get("end_location")

        for ff in ev["freeze_frame"]:
            rows.append({
                "event_id": event_id,
                "team": team,
                "player_name": player, 
                "event_type": ev_type,
                "minute": minute,
                "second": second,
                "event_location": location,
                "end_location": end_location,
                "ff_location": ff.get("location"),   # [x,y]
                "teammate": ff.get("teammate"),
                "actor": ff.get("actor"),
                "keeper": ff.get("keeper"),
            })

    return pd.DataFrame(rows)


def load_lineup_dataframe(match_id, base_dir="../data/open-data/data"):
    """
    Load lineup JSON into a DataFrame.
    
    Args:
        match_id: Match ID (string or int)
        base_dir: Base directory for StatsBomb data
    
    Returns:
        DataFrame with columns: team_name, player_name, player_id, position
    """
    lineup_file = Path(base_dir) / "lineups" / f"{match_id}.json"
    with open(lineup_file, "r", encoding="utf-8") as f:
        lineup_data = json.load(f)
    
    rows = []
    for team in lineup_data:
        team_name = team["team_name"]
        for p in team["lineup"]:
            pname = p["player_name"]
            pid = p["player_id"]
            # take first listed position (simplification)
            pos = p["positions"][0]["position"] if p.get("positions") else None
            rows.append({
                "team_name": team_name,
                "player_name": pname,
                "player_id": pid,
                "position": pos
            })
    
    return pd.DataFrame(rows)


def attach_actor_metadata(events_df, lineup_df):
    """
    Merge player_id and position from lineup into events DataFrame.
    Uses case-insensitive matching for team and player names.
    
    Args:
        events_df: Events DataFrame
        lineup_df: Lineup DataFrame from load_lineup_dataframe()
    
    Returns:
        DataFrame with player_id and position columns added
    """
    events_df = events_df.copy()
    
    # Create lowercase columns for matching
    events_df["team_lower"] = events_df["team"].str.lower()
    events_df["player_lower"] = events_df["player"].str.lower()
    
    lineup_df = lineup_df.copy()
    lineup_df["team_name_lower"] = lineup_df["team_name"].str.lower()
    lineup_df["player_name_lower"] = lineup_df["player_name"].str.lower()
    
    # Merge
    merged = events_df.merge(
        lineup_df[["team_name_lower", "player_name_lower", "player_id", "position"]],
        left_on=["team_lower", "player_lower"],
        right_on=["team_name_lower", "player_name_lower"],
        how="left"
    )
    
    # Keep only the columns we want (drop the temporary lowercase columns)
    events_df["player_id"] = merged["player_id"]
    events_df["position"] = merged["position"]
    events_df = events_df.drop(columns=["team_lower", "player_lower"])
    
    return events_df


def load_full_match(match_id, base_dir="../data/open-data/data"):
    """
    Load all data for a match into DataFrames:
    - events_df: one row per event with player_id and position
    - freeze_frames_df: one row per player in freeze-frames
    - lineup_df: DataFrame with player lineup info
    
    Args:
        match_id: Match ID (string or int)
        base_dir: Base directory for StatsBomb data
    
    Returns:
        Tuple of (events_df, freeze_frames_df, lineup_df)
    """
    # Step 1: Load events + freeze frames
    combined = load_match_data(match_id, base_dir=base_dir)
    
    # Step 2: Build events DataFrame
    events_df = build_events_dataframe(combined)
    
    # Step 3: Build freeze frames DataFrame
    freeze_frames_df = build_freeze_frames_dataframe(events_df)
    
    # Step 4: Load lineup data (using DataFrame version)
    lineup_df = load_lineup_dataframe(match_id, base_dir=base_dir)
    
    # Step 5: Attach player metadata to events
    events_df = attach_actor_metadata(events_df, lineup_df)
    
    return events_df, freeze_frames_df, lineup_df


# ============================================================================
# MULTI-MATCH / TEAM COLLECTION FUNCTIONS
# ============================================================================

def load_competitions(base_dir="../data/open-data/data"):
    """
    Load all available competitions from competitions.json.
    
    Args:
        base_dir: Base directory for StatsBomb data
    
    Returns:
        List of competition dictionaries
    """
    base_path = Path(base_dir)
    competitions_file = base_path / "competitions.json"
        
    with open(competitions_file, "r", encoding="utf-8") as f:
        competitions = json.load(f)
    
    return competitions


def find_competitions(competition_name=None, country_name=None, base_dir="../data/open-data/data"):
    """
    Find competitions matching given criteria.
    
    Args:
        competition_name: Competition name (partial match, case-insensitive)
        country_name: Country name (partial match, case-insensitive)
        base_dir: Base directory for StatsBomb data
    
    Returns:
        List of matching competition dictionaries
    """
    competitions = load_competitions(base_dir)
    
    if competition_name is None and country_name is None:
        return competitions
    
    matches = []
    for comp in competitions:
        match = True
        
        if competition_name:
            comp_name = comp.get("competition_name", "")
            if competition_name.lower() not in comp_name.lower():
                match = False
        
        if country_name:
            country = comp.get("country_name", "")
            if country_name.lower() not in country.lower():
                match = False
        
        if match:
            matches.append(comp)
    
    return matches


def find_seasons_for_competition(competition_id, base_dir="../data/open-data/data"):
    """
    Find all seasons available for a given competition.
    
    Args:
        competition_id: Competition ID
        base_dir: Base directory for StatsBomb data
    
    Returns:
        List of season dictionaries with competition_id, season_id, season_name, etc.
    """
    competitions = load_competitions(base_dir)
    
    seasons = [
        {
            "competition_id": comp["competition_id"],
            "season_id": comp["season_id"],
            "season_name": comp.get("season_name", ""),
            "competition_name": comp.get("competition_name", ""),
            "country_name": comp.get("country_name", ""),
            "match_available_360": comp.get("match_available_360"),
        }
        for comp in competitions
        if comp.get("competition_id") == competition_id
    ]
    
    return seasons


def load_team_matches(team_name, competition_id=None, season_id=None, 
                     base_dir="../data/open-data/data"):
    """
    Load all matches for a specific team from a competition and season.
    
    Args:
        team_name: Team name
        competition_id: Competition ID (optional)
        season_id: Season ID (optional)
        base_dir: Base directory for StatsBomb data
    
    Returns:
        List of match dictionaries with match_id, match_date, teams, scores, etc.
    """
    base_path = Path(base_dir)
    
    # If specific competition and season provided, load directly
    if competition_id is not None and season_id is not None:
        matches_file = base_path / "matches" / str(competition_id) / f"{season_id}.json"
        
        with open(matches_file, "r", encoding="utf-8") as f:
            matches = json.load(f)
    else:
        # Search across competitions/seasons
        competitions = load_competitions(base_dir)
        
        if competition_id is not None:
            competitions = [c for c in competitions if c.get("competition_id") == competition_id]
        if season_id is not None:
            competitions = [c for c in competitions if c.get("season_id") == season_id]
        
        matches = []
        for comp in competitions:
            comp_id = comp.get("competition_id")
            seas_id = comp.get("season_id")
            
            matches_file = base_path / "matches" / str(comp_id) / f"{seas_id}.json"
            
            if not matches_file.exists():
                continue  # Skip this season if file not found
            
            try:
                with open(matches_file, "r", encoding="utf-8") as f:
                    season_matches = json.load(f)
                    matches.extend(season_matches)
            except Exception:
                continue
    
    # Filter for team matches (case-insensitive)
    team_matches = []
    for match in matches:
        home_team = match.get("home_team", {}).get("home_team_name", "")
        away_team = match.get("away_team", {}).get("away_team_name", "")
        
        match_home = team_name.lower() in home_team.lower()
        match_away = team_name.lower() in away_team.lower()
        
        if match_home or match_away:
            team_matches.append({
                "match_id": match["match_id"],
                "match_date": match.get("match_date", ""),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": match.get("home_score", 0),
                "away_score": match.get("away_score", 0),
                "is_home": match_home,
                "competition_id": match.get("competition", {}).get("competition_id"),
                "season_id": match.get("season", {}).get("season_id"),
                "competition_name": match.get("competition", {}).get("competition_name", ""),
                "season_name": match.get("season", {}).get("season_name", ""),
            })
    
    return team_matches


def find_matches_with_360(matches, base_dir="../data/open-data/data"):
    """
    Find which matches have 360 freeze-frame data available.
    
    Args:
        matches: List of match dictionaries (from load_team_matches)
        base_dir: Base directory for StatsBomb data
    
    Returns:
        List of match IDs (integers) with 360 data
    """
    base_path = Path(base_dir)
    three_sixty_dir = base_path / "three-sixty"
    
    if not three_sixty_dir.exists():
        return []
    
    matches_with_360 = []
    for match in matches:
        match_id = match["match_id"]
        three_sixty_file = three_sixty_dir / f"{match_id}.json"
        
        if three_sixty_file.exists():
            matches_with_360.append(match_id)
    
    return matches_with_360


def load_all_team_data(team_name, matches, matches_with_360, base_dir="../data/open-data/data",
                       max_matches=None):
    """
    Load all team matches with 360 data and combine into unified DataFrames.
    
    Args:
        team_name: Team name (used for filtering events)
        matches: List of match dictionaries (from load_team_matches)
        matches_with_360: List of match IDs with 360 data (from find_matches_with_360)
        base_dir: Base directory for StatsBomb data
        max_matches: Maximum number of matches to load (None = all)
    
    Returns:
        Tuple of:
        - combined_events_df: DataFrame with all team events
        - combined_freeze_frames_df: DataFrame with expanded freeze-frame data
        - player_stats: Dictionary of player statistics
    """
    all_events = []
    all_freeze_frames = []
    player_stats = defaultdict(lambda: {"events": 0, "passes": 0, "matches": set()})
    
    matches_to_load = matches[:max_matches] if max_matches else matches
    
    # Determine correct base_dir
    test_match_id = matches_with_360[0] if matches_with_360 else None
    if test_match_id:
        test_paths = [
            Path(base_dir) / "events" / f"{test_match_id}.json",
            Path("data/open-data/data/events") / f"{test_match_id}.json",
        ]
        for test_path in test_paths:
            if test_path.exists():
                base_dir = str(test_path.parent.parent)
                break
    
    for match in matches_to_load:
        match_id = match["match_id"]
        match_id_str = str(match_id)
        
        # Check if match has 360 data
        if match_id not in matches_with_360:
            continue
        
        try:
            # Load full match data
            events_df, freeze_frames_df, lineup_df = load_full_match(match_id_str, base_dir=base_dir)
            
            # Filter for team events (case-insensitive)
            team_events = events_df[
                events_df["team"].str.contains(team_name, case=False, na=False)
            ].copy()
            
            if len(team_events) > 0:
                # Filter freeze frames for team events
                team_event_ids = set(team_events["event_id"])
                team_freeze_frames = freeze_frames_df[
                    freeze_frames_df["event_id"].isin(team_event_ids)
                ].copy()
                
                all_events.append(team_events)
                all_freeze_frames.append(team_freeze_frames)
                
                # Track player stats
                for _, row in team_events.iterrows():
                    player = row.get("player")
                    if player:
                        player_stats[player]["events"] += 1
                        player_stats[player]["matches"].add(match_id_str)
                        if row["type"] == "Pass":
                            player_stats[player]["passes"] += 1
        
        except Exception:
            continue
    
    # Combine all events
    if all_events:
        combined_events_df = pd.concat(all_events, ignore_index=True)
        combined_freeze_frames_df = pd.concat(all_freeze_frames, ignore_index=True)
    else:
        # Create empty DataFrames with expected columns
        combined_events_df = pd.DataFrame(columns=[
            "event_id", "match_id", "team", "player", "type", 
            "minute", "second", "location", "freeze_frame", 
            "visible_area", "end_location", "player_id", "position"
        ])
        combined_freeze_frames_df = pd.DataFrame(columns=[
            "event_id", "team", "player_name", "event_type",
            "minute", "second", "event_location", "end_location",
            "ff_location", "teammate", "actor", "keeper"
        ])
    
    # Convert sets to counts for player_stats
    player_stats_dict = {
        player: {
            "events": stats["events"],
            "passes": stats["passes"],
            "matches": len(stats["matches"])
        }
        for player, stats in player_stats.items()
    }
    
    return combined_events_df, combined_freeze_frames_df, player_stats_dict


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_data_summary(combined_events, combined_freeze_frames, player_stats, team_name="Team"):
    """
    Generate summary statistics for the loaded data.
    
    Args:
        combined_events: Combined events DataFrame
        combined_freeze_frames: Combined freeze frames DataFrame
        player_stats: Player statistics dictionary
        team_name: Team name for summary
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "team_name": team_name,
        "total_events": len(combined_events),
        "total_freeze_frames": len(combined_freeze_frames),
        "unique_matches": combined_events["match_id"].nunique() if len(combined_events) > 0 else 0,
        "unique_players": len(player_stats),
        "event_types": combined_events["type"].value_counts().to_dict() if len(combined_events) > 0 else {},
        "top_players": dict(sorted(
            player_stats.items(), 
            key=lambda x: x[1]["events"], 
            reverse=True
        )[:10]),
        "events_with_player_id": combined_events["player_id"].notna().sum() if "player_id" in combined_events.columns else 0,
        "events_with_location": combined_events["location"].notna().sum() if len(combined_events) > 0 else 0,
    }
    
    return summary


def print_data_summary(summary):
    """
    Print formatted data summary.
    
    Args:
        summary: Dictionary from get_data_summary()
    """
    team_name = summary.get("team_name", "Team")
    print("=" * 70)
    print(f"DATA SUMMARY - {team_name.upper()}")
    print("=" * 70)
    print(f"Total events:              {summary['total_events']:,}")
    print(f"Total freeze frame rows:   {summary['total_freeze_frames']:,}")
    print(f"Unique matches:            {summary['unique_matches']}")
    print(f"Unique players:            {summary['unique_players']}")
    print(f"Events with player_id:     {summary['events_with_player_id']:,}")
    print(f"Events with location:      {summary['events_with_location']:,}")
    
    if summary['event_types']:
        print(f"\nEvent types:")
        for ev_type, count in sorted(summary['event_types'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {ev_type}: {count:,}")
    
    if summary['top_players']:
        print(f"\nTop 10 players by event count:")
        for player, stats in summary['top_players'].items():
            print(f"  {player}: {stats['events']} events ({stats['passes']} passes) "
                  f"in {stats['matches']} matches")
    
    print("=" * 70)
