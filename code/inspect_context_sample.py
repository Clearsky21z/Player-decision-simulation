from __future__ import annotations

import argparse
from typing import Optional

from soccermap.context import DEFAULT_CONTEXT_DIM, context_feature_names
from soccermap.dataset import PassDataset
from soccermap.expand import build_expanded_dfs
from soccermap.statsbomb_io import load_events, load_lineups, load_threesixty


def _format_location(loc: object) -> str:
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        return f"[{float(loc[0]):.1f}, {float(loc[1]):.1f}]"
    return "None"


def _phase_label(minute: float) -> str:
    if minute < 30.0:
        return "early"
    if minute < 60.0:
        return "mid"
    if minute < 90.0:
        return "late"
    return "extra"


def _build_dataset(
    *,
    data_root: str,
    match_id: str,
    team_filter: Optional[str],
    compute_velocities: bool,
    context_dim: int,
) -> PassDataset:
    events = load_events(data_root, match_id)
    threesixty = load_threesixty(data_root, match_id)
    lineups = load_lineups(data_root, match_id)
    expanded = build_expanded_dfs(events, threesixty, lineups)
    return PassDataset(
        expanded.expanded_df,
        only_passes=True,
        team_filter=team_filter,
        compute_velocities=compute_velocities,
        context_dim=context_dim,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect the game-state context vector for one pass sample."
    )
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--match_id", type=str, required=True)
    ap.add_argument("--sample_idx", type=int, required=True)
    ap.add_argument("--team_filter", type=str, default="")
    ap.add_argument("--context_dim", type=int, default=DEFAULT_CONTEXT_DIM)
    ap.add_argument("--compute_velocities", action="store_true")
    args = ap.parse_args()

    team_filter = args.team_filter.strip() or None
    ds = _build_dataset(
        data_root=args.data_root,
        match_id=args.match_id,
        team_filter=team_filter,
        compute_velocities=args.compute_velocities,
        context_dim=args.context_dim,
    )

    if len(ds) == 0:
        raise RuntimeError(
            "No pass samples found for the requested match/team filter."
        )

    if args.sample_idx < 0 or args.sample_idx >= len(ds):
        raise IndexError(
            f"sample_idx {args.sample_idx} is out of range for dataset of size {len(ds)}"
        )

    sample = ds[args.sample_idx]
    row = ds.actor_events.iloc[args.sample_idx]

    team = row.get("team")
    opponent_team = row.get("opponent_team")
    minute = float(row.get("minute", 0))
    second = float(row.get("second", 0))
    team_score = int(row.get("team_score", 0))
    opponent_score = int(row.get("opponent_score", 0))
    score_diff = int(row.get("score_diff", team_score - opponent_score))

    print("SAMPLE")
    print(f"match_id: {args.match_id}")
    print(f"sample_idx: {args.sample_idx}")
    print(f"dataset_size: {len(ds)}")
    print(f"event_id: {sample.event_id}")
    print(f"player: {sample.actor_player_name}")
    print(f"team: {team}")
    print(f"opponent_team: {opponent_team}")
    print(f"minute_second: {int(minute)}:{int(second):02d}")
    print(f"phase_label: {_phase_label(minute)}")
    print(f"score_raw: {team_score}-{opponent_score}")
    print(f"score_diff_raw: {score_diff}")
    print(f"event_location: {_format_location(row.get('event_location'))}")
    print(f"end_location: {_format_location(row.get('end_location'))}")
    print(f"dest_lw: {sample.dest_lw}")
    print(f"dest_index: {sample.dest_index}")
    print(f"channels_shape: {tuple(sample.channels.shape)}")
    print(f"context_shape: {tuple(sample.context_features.shape)}")

    print("\nCONTEXT_FEATURES")
    for name, value in zip(
        context_feature_names(args.context_dim),
        sample.context_features.tolist(),
    ):
        print(f"{name}: {float(value):.6f}")

    if args.context_dim > DEFAULT_CONTEXT_DIM:
        print("\nNOTE")
        print(
            f"context_dim={args.context_dim} is larger than the default "
            f"{DEFAULT_CONTEXT_DIM}, so trailing entries are zero-padded."
        )


if __name__ == "__main__":
    main()
