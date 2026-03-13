from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mplsoccer import Pitch

from soccermap.statsbomb_io import load_events, load_threesixty, load_lineups
from soccermap.expand import build_expanded_dfs
from soccermap.dataset import PassDataset
from soccermap.model import SoccerMap, SoccerMapConfig
from soccermap.viz import _extract_scene, _to_img_yx, plot_pass_selection_surface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--match_id", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--sample_idx", type=int, default=0)
    ap.add_argument("--out", type=str, default="viz/pass_selection_surface.png")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--compute_velocities", action="store_true")

    # NEW
    ap.add_argument("--scale", type=str, default="log", choices=["log", "percentile", "linear"])
    ap.add_argument("--q", type=float, default=0.995)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--cmap", type=str, default="RdBu_r")

    args = ap.parse_args()

    events = load_events(args.data_root, args.match_id)
    threesixty = load_threesixty(args.data_root, args.match_id)
    lineups = load_lineups(args.data_root, args.match_id)
    m = build_expanded_dfs(events, threesixty, lineups)

    ds = PassDataset(m.expanded_df, compute_velocities=args.compute_velocities, only_passes=True)
    sample = ds[args.sample_idx]

    model = SoccerMap(SoccerMapConfig()).to(args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        x = sample.channels.unsqueeze(0).to(args.device)   # (1,13,L,W)
        logits = model(x)                                  # (1,1,L,W)

        # Softmax over all cells (selection distribution)
        flat = logits.view(1, -1)
        prob_flat = torch.softmax(flat, dim=1)[0]           # (L*W,)
        p_dest = float(prob_flat[sample.dest_index].cpu().item())

        L = logits.shape[2]
        W = logits.shape[3]
        prob_LW = prob_flat.view(L, W).cpu().numpy()

    title = f"Sample idx={args.sample_idx} | p(dest cell)={p_dest:.6f} | true complete={sample.completed}"

    out_path = Path(args.out)
    plot_pass_selection_surface(
        prob_LW,
        m.expanded_df,
        sample.event_id,
        title=title,
        out_path=str(out_path),
        show=False,
        scale=args.scale,
        q=args.q,
        eps=args.eps,
        cmap=args.cmap,
    )
    print(f"saved visualization -> {out_path}")


def plot_pass_selection_embed(
    model,
    ds,
    expanded_df,
    player_id_mapping: Dict[str, int],
    *,
    sample_idx: Optional[int] = None,
    out_path: Optional[str] = None,
    show: bool = True,
    white_threshold: float = 1e-9,
    cmap_name: str = "YlOrRd",
    figsize=(12, 8),
) -> plt.Figure:
    """
    Run inference on a single PassDataset sample using a model with player
    embeddings, then plot the pass selection probability surface with players,
    passer arrow, visibility mask boundary, and destination marker.

    Parameters
    ----------
    model : SoccerMapWithPlayerEmbed (already in .eval() mode)
    ds : PassDataset
    expanded_df : pd.DataFrame from build_expanded_dfs
    player_id_mapping : dict mapping player name -> embedding id
    sample_idx : index into ds; random if None
    out_path : if provided, save figure to this path
    show : call plt.show() if True
    white_threshold : probability below this renders as white
    cmap_name : colormap name
    figsize : figure size
    """
    if sample_idx is None:
        sample_idx = random.randint(0, len(ds) - 1)

    sample = ds[sample_idx]

    actor_name = sample.actor_player_name
    actor_id = player_id_mapping.get(actor_name, 0)

    # ---- Forward pass ----
    with torch.no_grad():
        x = sample.channels.unsqueeze(0)
        actor_ids = torch.tensor([actor_id], dtype=torch.long)
        logits = model(x, actor_ids)

        flat = logits.view(1, -1)
        prob_flat = torch.softmax(flat, dim=1)[0]
        L, W = logits.shape[2], logits.shape[3]
        prob_LW = prob_flat.view(L, W).cpu().numpy()
        p_dest = float(prob_flat[sample.dest_index].item())

    # ---- Extract scene info ----
    passer_xy, dest_xy, completed, atk_xy, dfn_xy = _extract_scene(
        expanded_df, sample.event_id
    )

    # ---- Visibility mask (channel 13) ----
    vis_mask = sample.channels[13].cpu().numpy()

    # ---- Plot ----
    pitch = Pitch(pitch_type="statsbomb", line_color="black", linewidth=1.2)
    fig, ax = pitch.draw(figsize=figsize)

    # probability field
    img = _to_img_yx(prob_LW)
    eps = 1e-12
    img_plot = np.clip(img, eps, None)
    vmax = max(float(np.quantile(img_plot, 0.995)), white_threshold * 10)

    cmap = cm.get_cmap(cmap_name).copy()
    cmap.set_under("white")

    norm = mcolors.LogNorm(vmin=white_threshold, vmax=vmax)
    im = ax.imshow(
        img_plot,
        extent=pitch.extent,
        origin="upper",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        alpha=0.85,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # visibility mask boundary
    vis_img = _to_img_yx(vis_mask)
    ax.contour(
        vis_img,
        levels=[0.5],
        colors="lime",
        linewidths=2,
        extent=pitch.extent,
        origin="upper",
    )

    # players
    if atk_xy.size:
        pitch.scatter(
            atk_xy[:, 0], atk_xy[:, 1],
            s=80, ax=ax, color="dodgerblue",
            edgecolors="black", zorder=5, label="Teammates",
        )
    if dfn_xy.size:
        pitch.scatter(
            dfn_xy[:, 0], dfn_xy[:, 1],
            s=80, ax=ax, color="red",
            edgecolors="black", zorder=5, label="Opponents",
        )

    # passer + pass arrow + destination
    if passer_xy:
        pitch.scatter(
            [passer_xy[0]], [passer_xy[1]],
            s=160, ax=ax, color="gold",
            edgecolors="black", zorder=6,
            label=f"Passer: {actor_name}",
        )
    if dest_xy:
        pitch.scatter(
            [dest_xy[0]], [dest_xy[1]],
            s=160, marker="x", ax=ax,
            color="black", zorder=6, label="True destination",
        )
    if passer_xy and dest_xy:
        pitch.arrows(
            passer_xy[0], passer_xy[1],
            dest_xy[0], dest_xy[1],
            ax=ax, width=2, headwidth=6, headlength=6, color="black",
        )

    ax.set_title(
        f"{actor_name} (embed_id={actor_id})\n"
        f"p(dest)={p_dest:.6f} | completed={completed} | sample_idx={sample_idx}",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


if __name__ == "__main__":
    main()
