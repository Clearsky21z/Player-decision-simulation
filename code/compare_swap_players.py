from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from soccermap.context import DEFAULT_CONTEXT_DIM
from soccermap.dataset import PassDataset
from soccermap.expand import build_expanded_dfs
from soccermap.model import SoccerMapConfig, SoccerMapWithPlayerEmbed
from soccermap.statsbomb_io import load_events, load_lineups, load_threesixty
from soccermap.viz import _extract_scene, _extract_visible_area, _to_img_yx


def _predict_surface(model, sample, actor_id: int, device: str):
    with torch.no_grad():
        x = sample.channels.unsqueeze(0).to(device)
        context = sample.context_features.unsqueeze(0).to(device)
        actor_ids = torch.tensor([actor_id], dtype=torch.long, device=device)
        logits = model(x, actor_ids, context)
        flat = logits.view(1, -1)
        prob_flat = torch.softmax(flat, dim=1)[0]
        l_bins, w_bins = logits.shape[2], logits.shape[3]
        prob_lw = prob_flat.view(l_bins, w_bins).cpu().numpy()
        peak_idx = int(prob_flat.argmax().item())
        peak_l, peak_w = peak_idx // w_bins, peak_idx % w_bins
        peak_x = peak_l * (120.0 / l_bins)
        peak_y = peak_w * (80.0 / w_bins)
    return prob_lw, (peak_x, peak_y)


def _draw_scene(ax, expanded_df, event_id: str, label: str):
    passer_xy, dest_xy, _, atk_xy, dfn_xy = _extract_scene(expanded_df, event_id)
    visible_area_pts = _extract_visible_area(expanded_df, event_id)

    if atk_xy.size:
        ax.scatter(atk_xy[:, 0], atk_xy[:, 1], s=60, c="dodgerblue", edgecolors="black", zorder=5)
    if dfn_xy.size:
        ax.scatter(dfn_xy[:, 0], dfn_xy[:, 1], s=60, c="red", edgecolors="black", zorder=5)
    if visible_area_pts is not None:
        closed = np.vstack([visible_area_pts, visible_area_pts[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="lime", linewidth=2, zorder=4)
    if passer_xy is not None:
        ax.scatter([passer_xy[0]], [passer_xy[1]], s=140, c="gold", edgecolors="black", zorder=6)
    if dest_xy is not None:
        ax.scatter([dest_xy[0]], [dest_xy[1]], s=150, marker="x", c="black", zorder=7)
    if passer_xy is not None and dest_xy is not None:
        ax.annotate("", xy=dest_xy, xytext=passer_xy, arrowprops=dict(color="black", width=1.5, headwidth=8))
    ax.set_xlim(0, 120)
    ax.set_ylim(80, 0)
    ax.set_aspect("equal")
    ax.set_title(label, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--match_id", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--sample_idx", type=int, required=True)
    ap.add_argument("--players", type=str, required=True, help="Comma-separated player names for swap_player comparison")
    ap.add_argument("--out_prefix", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--compute_velocities", action="store_true")
    ap.add_argument("--team_filter", type=str, default="Bayer Leverkusen")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    events = load_events(args.data_root, args.match_id)
    threesixty = load_threesixty(args.data_root, args.match_id)
    lineups = load_lineups(args.data_root, args.match_id)
    match = build_expanded_dfs(events, threesixty, lineups)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    state = ckpt["model_state"]
    cfg_kwargs = ckpt.get("config", {})
    cfg = SoccerMapConfig(**cfg_kwargs) if cfg_kwargs else SoccerMapConfig()
    player_id_mapping = ckpt["player_id_mapping"]

    ds = PassDataset(
        match.expanded_df,
        compute_velocities=args.compute_velocities,
        only_passes=True,
        team_filter=args.team_filter,
        context_dim=ckpt.get("context_dim", DEFAULT_CONTEXT_DIM),
    )
    sample = ds[args.sample_idx]

    model = SoccerMapWithPlayerEmbed(
        num_players=ckpt["num_players"],
        embed_dim=ckpt.get("embed_dim", 8),
        context_dim=ckpt.get("context_dim", 0),
        context_hidden_dim=ckpt.get("context_hidden_dim", 16),
        context_embed_dim=ckpt.get("context_embed_dim", 8),
        cfg=cfg,
    ).to(args.device)
    model.load_state_dict(state)
    model.eval()

    original_actor = sample.actor_player_name
    players = [p.strip() for p in args.players.split(",") if p.strip()]
    compare_names = [original_actor] + players

    surfaces = {}
    peaks = {}
    original_id = player_id_mapping.get(original_actor, 0)
    original_surface, original_peak = _predict_surface(model, sample, original_id, args.device)
    surfaces[original_actor] = original_surface
    peaks[original_actor] = original_peak

    for player_name in players:
        actor_id = player_id_mapping.get(player_name, 0)
        surfaces[player_name], peaks[player_name] = _predict_surface(model, sample, actor_id, args.device)

    # Difference maps relative to original actor.
    n_cols = len(compare_names)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 6), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    for ax, player_name in zip(axes, compare_names):
        diff = surfaces[player_name] - original_surface
        img = _to_img_yx(diff)
        vmax = np.max(np.abs(img))
        vmax = max(vmax, 1e-12)
        im = ax.imshow(
            img,
            extent=[0, 120, 80, 0],
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            alpha=0.85,
            interpolation="nearest",
        )
        label = "Original" if player_name == original_actor else f"{player_name}\nminus original"
        _draw_scene(ax, match.expanded_df, sample.event_id, label)
        peak_x, peak_y = peaks[player_name]
        ax.scatter([peak_x], [peak_y], s=150, marker="x", c="blue", zorder=8)

    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    diff_path = out_prefix.with_name(out_prefix.name + "_difference_maps.png")
    fig.savefig(diff_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Predicted destination displacement summary.
    base_x, base_y = original_peak
    rows = []
    for player_name in compare_names:
        peak_x, peak_y = peaks[player_name]
        displacement = float(np.hypot(peak_x - base_x, peak_y - base_y))
        rows.append(
            {
                "player": player_name,
                "peak_x": peak_x,
                "peak_y": peak_y,
                "displacement_from_original": displacement,
            }
        )
    df = pd.DataFrame(rows)
    csv_path = out_prefix.with_name(out_prefix.name + "_displacement.csv")
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(df["player"], df["displacement_from_original"], color=["black"] + ["steelblue"] * (len(df) - 1))
    ax.set_ylabel("Predicted destination displacement")
    ax.set_title(f"match_id={args.match_id} sample_idx={args.sample_idx}")
    ax.tick_params(axis="x", rotation=20)
    bar_path = out_prefix.with_name(out_prefix.name + "_displacement.png")
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"saved difference maps -> {diff_path}")
    print(f"saved displacement csv -> {csv_path}")
    print(f"saved displacement plot -> {bar_path}")


if __name__ == "__main__":
    main()
