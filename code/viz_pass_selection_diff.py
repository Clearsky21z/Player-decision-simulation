from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from soccermap.context import DEFAULT_CONTEXT_DIM
from soccermap.dataset import PassDataset
from soccermap.expand import build_expanded_dfs
from soccermap.statsbomb_io import load_events, load_lineups, load_threesixty
from soccermap.viz import plot_pass_selection_difference_surface
from viz_pass_selection import (
    _event_metadata,
    _build_model_from_ckpt,
    _infer_match_id,
    _slugify,
    predict_pass_selection_embed,
    resolve_player_name,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--match_id", type=str, default="")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--sample_idx", type=int, default=0)
    ap.add_argument("--player_a", type=str, required=True)
    ap.add_argument("--player_b", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--compute_velocities", action="store_true")
    ap.add_argument("--team_filter", type=str, default="")
    ap.add_argument("--q", type=float, default=0.995)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    match_id = _infer_match_id(ckpt, args.match_id)

    events = load_events(args.data_root, match_id)
    threesixty = load_threesixty(args.data_root, match_id)
    lineups = load_lineups(args.data_root, match_id)
    m = build_expanded_dfs(events, threesixty, lineups)

    dataset_context_dim = ckpt.get("context_dim", DEFAULT_CONTEXT_DIM)
    team_filter = args.team_filter.strip() or None
    ds = PassDataset(
        m.expanded_df,
        compute_velocities=args.compute_velocities,
        only_passes=True,
        team_filter=team_filter,
        context_dim=dataset_context_dim,
    )

    if len(ds) == 0:
        raise ValueError(
            f"No pass samples found for match_id={match_id}"
            + (f" with team_filter={team_filter!r}" if team_filter else "")
        )
    if args.sample_idx < 0 or args.sample_idx >= len(ds):
        raise IndexError(
            f"sample_idx {args.sample_idx} out of range for match_id={match_id}; "
            f"dataset contains {len(ds)} sample(s)."
        )

    model, uses_player_embed = _build_model_from_ckpt(ckpt, args.device)
    if not uses_player_embed:
        raise ValueError("viz_pass_selection_diff.py requires a player-conditioned checkpoint.")

    player_id_mapping = ckpt.get("player_id_mapping", {})
    resolved_player_a, player_a_id = resolve_player_name(args.player_a, player_id_mapping)
    resolved_player_b, player_b_id = resolve_player_name(args.player_b, player_id_mapping)

    sample = ds[args.sample_idx]
    event_index, _ = _event_metadata(m.expanded_df, sample.event_id)
    _, _, prob_a = predict_pass_selection_embed(
        model,
        sample,
        player_a_id,
    )
    _, _, prob_b = predict_pass_selection_embed(
        model,
        sample,
        player_b_id,
    )
    diff = prob_a - prob_b

    if args.out.strip():
        out_path = Path(args.out)
    else:
        out_path = Path("viz") / (
            f"{Path(args.ckpt).stem}_match_{match_id}_sample_{args.sample_idx}_"
            f"{_slugify(resolved_player_a)}_minus_{_slugify(resolved_player_b)}.png"
        )

    title = (
        f"{resolved_player_a} - {resolved_player_b} pass probability difference\n"
        f"match_id={match_id} | sample_idx={args.sample_idx} | "
        f"event_idx={event_index} | event_id={sample.event_id}"
    )
    plot_pass_selection_difference_surface(
        diff,
        m.expanded_df,
        sample.event_id,
        title=title,
        out_path=str(out_path),
        show=False,
        q=args.q,
    )

    print(
        f"visualized difference match_id={match_id} sample_idx={args.sample_idx} "
        f"event_idx={event_index} event_id={sample.event_id} "
        f"players={resolved_player_a!r}-{resolved_player_b!r}"
    )
    print(
        f"diff summary: min={float(np.min(diff)):.6e} "
        f"max={float(np.max(diff)):.6e} "
        f"mean_abs={float(np.mean(np.abs(diff))):.6e}"
    )
    print(f"saved visualization -> {out_path}")


if __name__ == "__main__":
    main()
