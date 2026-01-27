from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .statsbomb_io import get_team_names_from_lineups


@dataclass(frozen=True)
class ExpandedMatch:
    """
    event_df: one row per event (only events that have 360 data, unless keep_all_events=True)
    expanded_df: multiple rows per event: one "actor row" + all freeze-frame players excluding actor
    """
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
    """
    StatsBomb convention:
      - pass["outcome"] exists when it's not completed (e.g., Out, Incomplete, Offside, etc.)
      - completed passes often have no "outcome" field.
    Returns:
      1 for completed, 0 for not completed, None if not a pass.
    """
    p = event.get("pass")
    if not isinstance(p, dict):
        return None
    outcome = p.get("outcome")
    return 1 if outcome is None else 0


def build_expanded_dfs(
    events: List[Dict[str, Any]],
    threesixty: List[Dict[str, Any]],
    lineups: Optional[List[Dict[str, Any]]] = None,
    keep_all_events: bool = False,
) -> ExpandedMatch:
    """
    Merge StatsBomb events + 360 freeze frames into:
      - event_df: one row per event
      - expanded_df: one actor row + all freeze-frame players (excluding actor) per event

    IMPORTANT:
    - We create a dedicated actor row so that actor==True is unique per event.
      (360 freeze frames include an actor player as well; we drop that actor entry to avoid duplicates.)
    - We attach a stable ff_idx for each freeze-frame player within an event
      so velocity matching doesn't depend on DataFrame row indices.
    """
    # Map 360 by event uuid
    ff_by_uuid: Dict[str, Dict[str, Any]] = {
        d.get("event_uuid"): d for d in threesixty if isinstance(d, dict) and d.get("event_uuid")
    }

    team1, team2 = (None, None)
    if lineups is not None:
        team1, team2 = get_team_names_from_lineups(lineups)

    event_rows: List[Dict[str, Any]] = []
    expanded_rows: List[Dict[str, Any]] = []

    for ev in events:
        ev_id = ev.get("id")
        if ev_id is None:
            continue

        ff = ff_by_uuid.get(ev_id)
        if ff is None and not keep_all_events:
            continue

        team = (ev.get("team") or {}).get("name")
        # infer opponent team if possible
        opp_team = None
        if team1 and team2 and team:
            opp_team = team2 if team == team1 else (team1 if team == team2 else None)

        minute = ev.get("minute")
        second = ev.get("second")
        period = ev.get("period")
        event_type = (ev.get("type") or {}).get("name")
        loc = ev.get("location")  # ball location for the event

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
            "visible_area": (ff or {}).get("visible_area"),
        })

        # Actor row (unique)
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
            "visible_area": (ff or {}).get("visible_area"),
        })

        if ff is None:
            continue

        freeze = ff.get("freeze_frame") or []
        # Add all players EXCLUDING actor entry from freeze_frame
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
                "visible_area": ff.get("visible_area"),
            })
            ff_idx += 1

    event_df = pd.DataFrame(event_rows)
    expanded_df = pd.DataFrame(expanded_rows)

    # Useful derived columns
    if not expanded_df.empty:
        expanded_df["total_seconds"] = expanded_df["minute"].fillna(0).astype(float) * 60.0 + expanded_df["second"].fillna(0).astype(float)

    return ExpandedMatch(event_df=event_df, expanded_df=expanded_df)
