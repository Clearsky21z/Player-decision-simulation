from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def events_path(data_root: str | Path, match_id: str | int) -> Path:
    return Path(data_root) / "events" / f"{match_id}.json"


def threesixty_path(data_root: str | Path, match_id: str | int) -> Path:
    return Path(data_root) / "three-sixty" / f"{match_id}.json"


def lineups_path(data_root: str | Path, match_id: str | int) -> Path:
    return Path(data_root) / "lineups" / f"{match_id}.json"


def load_events(data_root: str | Path, match_id: str | int) -> List[Dict[str, Any]]:
    return _read_json(events_path(data_root, match_id))


def load_threesixty(data_root: str | Path, match_id: str | int) -> List[Dict[str, Any]]:
    return _read_json(threesixty_path(data_root, match_id))


def load_lineups(data_root: str | Path, match_id: str | int) -> List[Dict[str, Any]]:
    return _read_json(lineups_path(data_root, match_id))


def get_team_names_from_lineups(lineups: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """
    lineups file is a list with one entry per team.
    Each entry usually has {"team_name": "...", "lineup":[...]}.
    """
    if not isinstance(lineups, list) or len(lineups) < 2:
        return None, None
    t1 = lineups[0].get("team_name")
    t2 = lineups[1].get("team_name")
    return t1, t2
