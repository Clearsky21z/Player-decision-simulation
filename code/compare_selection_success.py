from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from mplsoccer import Pitch

from soccermap.context import DEFAULT_CONTEXT_DIM
from soccermap.dataset import PassDataset
from soccermap.expand import build_expanded_dfs
from soccermap.model import (
    SoccerMap,
    SoccerMapConfig,
    SoccerMapWithPlayerEmbed,
    pass_selection_surface,
    pass_success_surface,
)
from soccermap.statsbomb_io import load_events, load_lineups, load_threesixty
from soccermap.viz import _extract_scene, _to_img_yx


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_data_root = project_root / "data" / "leverkusen_data"
    default_sel = project_root / "checkpoints" / "pass_selection_15ch_context8_ce.pt"
    default_succ = project_root / "checkpoints" / "pass_success_15ch_context8.pt"

    ap = argparse.ArgumentParser(
        description="Visualize selection, success, and combined pass surfaces on the same sample."
    )
    ap.add_argument("--data_root", type=str, default=str(default_data_root))
    ap.add_argument("--match_id", type=str, required=True)
    ap.add_argument("--selection_ckpt", type=str, default=str(default_sel))
    ap.add_argument("--success_ckpt", type=str, default=str(default_succ))
    ap.add_argument("--old_selection_ckpt", type=str, default="")
    ap.add_argument("--sample_idx", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--compute_velocities", action="store_true")
    ap.add_argument("--out", type=str, default="viz/selection_success_compare.png")
    ap.add_argument("--swap_player", type=str, default="")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(project_root / "code" / "sandbox" / "cris" / "viz"),
    )
    ap.add_argument("--selection_scale", type=str, default="log", choices=["log", "linear"])
    ap.add_argument("--selection_eps", type=float, default=1e-12)
    ap.add_argument("--selection_q", type=float, default=0.995)
    ap.add_argument(
        "--save_mode",
        type=str,
        default="core",
        choices=["core", "all"],
        help="`core` saves selection-only and combined views; `all` also saves old-selection and success-only views.",
    )
    return ap.parse_args()


def build_model_from_ckpt(ckpt: Dict, device: str) -> Tuple[torch.nn.Module, bool, Dict[str, int]]:
    state = ckpt.get("model_state", ckpt)
    cfg_kwargs = ckpt.get("config", {})
    cfg = SoccerMapConfig(**cfg_kwargs) if cfg_kwargs else SoccerMapConfig()

    uses_player_embed = any(k.startswith("player_embedding.") for k in state.keys())
    player_id_mapping = ckpt.get("player_id_mapping", {})

    if uses_player_embed:
        model = SoccerMapWithPlayerEmbed(
            num_players=ckpt.get("num_players", 0),
            embed_dim=ckpt.get("embed_dim", 8),
            context_dim=ckpt.get("context_dim", 0),
            context_hidden_dim=ckpt.get("context_hidden_dim", 16),
            context_embed_dim=ckpt.get("context_embed_dim", 8),
            cfg=cfg,
        ).to(device)
    else:
        model = SoccerMap(cfg).to(device)

    model.load_state_dict(state)
    model.eval()
    return model, uses_player_embed, player_id_mapping


def align_sample_to_ckpt(sample, ckpt: Dict):
    state = ckpt.get("model_state", ckpt)
    cfg_kwargs = ckpt.get("config", {})
    cfg = SoccerMapConfig(**cfg_kwargs) if cfg_kwargs else SoccerMapConfig()
    expected_channels = cfg.in_channels
    expected_context_dim = int(ckpt.get("context_dim", sample.context_features.shape[0]))

    channels = sample.channels
    if channels.shape[0] > expected_channels:
        channels = channels[:expected_channels]
    elif channels.shape[0] < expected_channels:
        pad = expected_channels - channels.shape[0]
        channels = torch.cat(
            [
                channels,
                torch.zeros(
                    pad,
                    channels.shape[1],
                    channels.shape[2],
                    dtype=channels.dtype,
                ),
            ],
            dim=0,
        )

    context = sample.context_features
    if context.shape[0] > expected_context_dim:
        context = context[:expected_context_dim]
    elif context.shape[0] < expected_context_dim:
        pad = expected_context_dim - context.shape[0]
        context = torch.cat([context, torch.zeros(pad, dtype=context.dtype)], dim=0)

    return replace(sample, channels=channels, context_features=context)


def predict_surface(
    ckpt_path: str,
    sample,
    *,
    device: str,
    task_hint: Optional[str] = None,
    swap_player: str = "",
) -> Tuple[np.ndarray, float, str]:
    ckpt = torch.load(ckpt_path, map_location=device)
    sample = align_sample_to_ckpt(sample, ckpt)
    model, uses_player_embed, player_id_mapping = build_model_from_ckpt(ckpt, device)

    with torch.no_grad():
        x = sample.channels.unsqueeze(0).to(device)
        if uses_player_embed:
            actor_name = swap_player.strip() or sample.actor_player_name
            actor_id = player_id_mapping.get(actor_name, 0)
            actor_ids = torch.tensor([actor_id], dtype=torch.long, device=device)
            context = sample.context_features.unsqueeze(0).to(device)
            logits = model(x, actor_ids, context)
        else:
            logits = model(x)

    task = task_hint or ckpt.get("task", "pass_selection")
    if task == "pass_success":
        surf = pass_success_surface(logits)[0].cpu().numpy()
    else:
        surf = pass_selection_surface(logits)[0].cpu().numpy()

    L, W = surf.shape
    dest_l, dest_w = sample.dest_lw
    dest_prob = float(surf[dest_l, dest_w])
    return surf, dest_prob, task


def draw_surface(
    ax,
    pitch: Pitch,
    surface: np.ndarray,
    expanded_df,
    event_id: str,
    *,
    title: str,
    is_selection: bool,
    selection_scale: str,
    selection_eps: float,
    selection_q: float,
):
    passer_xy, dest_xy, _, atk_xy, dfn_xy = _extract_scene(expanded_df, event_id)
    img = _to_img_yx(surface)

    if is_selection and selection_scale == "log":
        img_plot = np.clip(img, selection_eps, None)
        vmax = float(np.quantile(img_plot, selection_q))
        vmax = max(vmax, selection_eps * 10)
        norm = mcolors.LogNorm(vmin=selection_eps, vmax=vmax)
        im = ax.imshow(
            img_plot,
            extent=pitch.extent,
            origin="upper",
            aspect="auto",
            cmap="YlOrRd",
            norm=norm,
            alpha=0.92,
            interpolation="nearest",
        )
    else:
        vmax = 1.0 if not is_selection else float(max(np.quantile(img, selection_q), selection_eps))
        im = ax.imshow(
            img,
            extent=pitch.extent,
            origin="upper",
            aspect="auto",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=vmax,
            alpha=0.92,
            interpolation="nearest",
        )

    if atk_xy.size:
        pitch.scatter(atk_xy[:, 0], atk_xy[:, 1], s=55, ax=ax, color="dodgerblue", edgecolors="black", zorder=5)
    if dfn_xy.size:
        pitch.scatter(dfn_xy[:, 0], dfn_xy[:, 1], s=55, ax=ax, color="red", edgecolors="black", zorder=5)
    if passer_xy is not None:
        pitch.scatter([passer_xy[0]], [passer_xy[1]], s=120, ax=ax, color="gold", edgecolors="black", zorder=6)
    if dest_xy is not None:
        pitch.scatter([dest_xy[0]], [dest_xy[1]], s=120, marker="x", ax=ax, color="black", zorder=6)
    if passer_xy is not None and dest_xy is not None:
        pitch.arrows(
            passer_xy[0], passer_xy[1],
            dest_xy[0], dest_xy[1],
            ax=ax,
            width=2,
            headwidth=6,
            headlength=6,
            headaxislength=5,
            color="black",
        )
    ax.set_title(title, fontsize=10)
    return im


def plot_single_surface(
    surface: np.ndarray,
    sample,
    expanded_df,
    *,
    title: str,
    out_path: Path,
    is_selection: bool,
    selection_scale: str,
    selection_eps: float,
    selection_q: float,
    visibility_mask: Optional[np.ndarray] = None,
):
    pitch = Pitch(pitch_type="statsbomb", line_color="black", linewidth=1.2)
    fig, ax = pitch.draw(figsize=(12, 8))

    passer_xy, dest_xy, completed, atk_xy, dfn_xy = _extract_scene(expanded_df, sample.event_id)
    img = _to_img_yx(surface)

    if is_selection and selection_scale == "log":
        img_plot = np.clip(img, selection_eps, None)
        vmax = float(np.quantile(img_plot, selection_q))
        vmax = max(vmax, selection_eps * 10)
        norm = mcolors.LogNorm(vmin=selection_eps, vmax=vmax)
        im = ax.imshow(
            img_plot,
            extent=pitch.extent,
            origin="upper",
            aspect="auto",
            cmap="YlOrRd",
            norm=norm,
            alpha=0.92,
            interpolation="nearest",
        )
    else:
        vmax = 1.0 if not is_selection else float(max(np.quantile(img, selection_q), selection_eps))
        im = ax.imshow(
            img,
            extent=pitch.extent,
            origin="upper",
            aspect="auto",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=vmax,
            alpha=0.92,
            interpolation="nearest",
        )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if visibility_mask is not None:
        vis_img = _to_img_yx(visibility_mask)
        ax.contour(
            vis_img,
            levels=[0.5],
            colors="lime",
            linewidths=2,
            extent=pitch.extent,
            origin="upper",
        )

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
    if passer_xy is not None:
        pitch.scatter(
            [passer_xy[0]], [passer_xy[1]],
            s=160, ax=ax, color="gold",
            edgecolors="black", zorder=6, label="Ball carrier position",
        )
    if dest_xy is not None:
        pitch.scatter(
            [dest_xy[0]], [dest_xy[1]],
            s=160, marker="x", ax=ax,
            color="black", zorder=6, label="True destination",
        )
    if passer_xy is not None and dest_xy is not None:
        pitch.arrows(
            passer_xy[0], passer_xy[1],
            dest_xy[0], dest_xy[1],
            ax=ax, width=2, headwidth=6, headlength=6, color="black",
        )

    peak_idx = int(surface.argmax())
    L, W = surface.shape
    peak_l, peak_w = peak_idx // W, peak_idx % W
    peak_x = peak_l * (120.0 / L)
    peak_y = peak_w * (80.0 / W)
    if passer_xy is not None:
        pitch.arrows(
            passer_xy[0], passer_xy[1],
            peak_x, peak_y,
            ax=ax, width=2, headwidth=6, headlength=6, color="blue",
        )
    pitch.scatter(
        [peak_x], [peak_y],
        s=160, marker="x", ax=ax,
        color="blue", zorder=6, label="Predicted destination",
    )

    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_difference_surface(
    surface: np.ndarray,
    sample,
    expanded_df,
    *,
    title: str,
    out_path: Path,
    visibility_mask: Optional[np.ndarray] = None,
):
    pitch = Pitch(pitch_type="statsbomb", line_color="black", linewidth=1.2)
    fig, ax = pitch.draw(figsize=(12, 8))

    passer_xy, dest_xy, completed, atk_xy, dfn_xy = _extract_scene(expanded_df, sample.event_id)
    img = _to_img_yx(surface)
    vmax = float(np.max(np.abs(img)))
    vmax = max(vmax, 1e-9)

    im = ax.imshow(
        img,
        extent=pitch.extent,
        origin="upper",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        alpha=0.92,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if visibility_mask is not None:
        vis_img = _to_img_yx(visibility_mask)
        ax.contour(
            vis_img,
            levels=[0.5],
            colors="lime",
            linewidths=2,
            extent=pitch.extent,
            origin="upper",
        )

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
    if passer_xy is not None:
        pitch.scatter(
            [passer_xy[0]], [passer_xy[1]],
            s=160, ax=ax, color="gold",
            edgecolors="black", zorder=6, label="Ball carrier position",
        )
    if dest_xy is not None:
        pitch.scatter(
            [dest_xy[0]], [dest_xy[1]],
            s=160, marker="x", ax=ax,
            color="black", zorder=6, label="True destination",
        )
    if passer_xy is not None and dest_xy is not None:
        pitch.arrows(
            passer_xy[0], passer_xy[1],
            dest_xy[0], dest_xy[1],
            ax=ax, width=2, headwidth=6, headlength=6, color="black",
        )

    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def normalize_surface(surface: np.ndarray) -> np.ndarray:
    total = float(surface.sum())
    if total <= 0.0:
        return surface
    return surface / total


def main():
    args = parse_args()

    events = load_events(args.data_root, args.match_id)
    threesixty = load_threesixty(args.data_root, args.match_id)
    lineups = load_lineups(args.data_root, args.match_id)
    expanded = build_expanded_dfs(events, threesixty, lineups)

    sel_ckpt = torch.load(args.selection_ckpt, map_location=args.device)
    dataset_context_dim = sel_ckpt.get("context_dim", DEFAULT_CONTEXT_DIM)
    ds = PassDataset(
        expanded.expanded_df,
        compute_velocities=args.compute_velocities,
        only_passes=True,
        context_dim=dataset_context_dim,
    )
    sample = ds[args.sample_idx]

    selection_surface, selection_dest_prob, _ = predict_surface(
        args.selection_ckpt,
        sample,
        device=args.device,
        task_hint="pass_selection",
        swap_player=args.swap_player,
    )
    success_surface, success_dest_prob, _ = predict_surface(
        args.success_ckpt,
        sample,
        device=args.device,
        task_hint="pass_success",
        swap_player=args.swap_player,
    )

    combined_surface = selection_surface * success_surface
    combined_surface = normalize_surface(combined_surface)
    combined_dest_prob = float(combined_surface[sample.dest_lw[0], sample.dest_lw[1]])

    sharp_alpha = 2.0
    combined_sharp_surface = normalize_surface(selection_surface * (success_surface ** sharp_alpha))
    combined_sharp_dest_prob = float(
        combined_sharp_surface[sample.dest_lw[0], sample.dest_lw[1]]
    )

    eps = 1e-12
    log_beta = 2.0
    log_combined_surface = normalize_surface(
        np.exp(np.log(np.clip(selection_surface, eps, None)) + log_beta * np.log(np.clip(success_surface, eps, None)))
    )
    log_combined_dest_prob = float(
        log_combined_surface[sample.dest_lw[0], sample.dest_lw[1]]
    )

    old_selection_surface = None
    old_selection_dest_prob = None
    if args.old_selection_ckpt.strip():
        old_ckpt = torch.load(args.old_selection_ckpt, map_location=args.device)
        old_context_dim = old_ckpt.get("context_dim", DEFAULT_CONTEXT_DIM)
        old_ds = PassDataset(
            expanded.expanded_df,
            compute_velocities=args.compute_velocities,
            only_passes=True,
            context_dim=old_context_dim,
        )
        old_sample = old_ds[args.sample_idx]
        old_selection_surface, old_selection_dest_prob, _ = predict_surface(
            args.old_selection_ckpt,
            old_sample,
            device=args.device,
            task_hint="pass_selection",
            swap_player=args.swap_player,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    visibility_mask = sample.channels[-1].cpu().numpy()
    actor_name = sample.actor_player_name or "unknown"
    display_name = args.swap_player.strip() or actor_name
    title_prefix = (
        f"Swapped to {display_name}" if args.swap_player.strip() else f"Original passer: {display_name}"
    )
    file_prefix = "swapped" if args.swap_player.strip() else "original"

    if old_selection_surface is not None:
        plot_single_surface(
            old_selection_surface,
            sample,
            expanded.expanded_df,
            title=(
                f"{title_prefix} | old selection checkpoint\n"
                f"Original passer: {actor_name} | completed={sample.completed} | sample_idx={args.sample_idx}"
            ),
            out_path=out_dir / f"sample_{args.sample_idx}_{file_prefix}_old_selection.png",
            is_selection=True,
            selection_scale=args.selection_scale,
            selection_eps=args.selection_eps,
            selection_q=args.selection_q,
            visibility_mask=visibility_mask,
        )

    plot_single_surface(
        selection_surface,
        sample,
        expanded.expanded_df,
        title=(
            f"{title_prefix} | pass selection only\n"
            f"Original passer: {actor_name} | completed={sample.completed} | sample_idx={args.sample_idx}"
        ),
        out_path=out_dir / f"sample_{args.sample_idx}_{file_prefix}_selection_only.png",
        is_selection=True,
        selection_scale=args.selection_scale,
        selection_eps=args.selection_eps,
        selection_q=args.selection_q,
        visibility_mask=visibility_mask,
    )
    plot_single_surface(
        success_surface,
        sample,
        expanded.expanded_df,
        title=(
            f"{title_prefix} | pass success only\n"
            f"Original passer: {actor_name} | completed={sample.completed} | sample_idx={args.sample_idx}"
        ),
        out_path=out_dir / f"sample_{args.sample_idx}_{file_prefix}_pass_success_only.png",
        is_selection=False,
        selection_scale=args.selection_scale,
        selection_eps=args.selection_eps,
        selection_q=args.selection_q,
        visibility_mask=visibility_mask,
    )
    plot_single_surface(
        combined_surface,
        sample,
        expanded.expanded_df,
        title=(
            f"{title_prefix} | combined: pass selection x pass success\n"
            f"Original passer: {actor_name} | completed={sample.completed} | sample_idx={args.sample_idx}"
        ),
        out_path=out_dir / f"sample_{args.sample_idx}_{file_prefix}_combined_selection_success.png",
        is_selection=True,
        selection_scale=args.selection_scale,
        selection_eps=args.selection_eps,
        selection_q=args.selection_q,
        visibility_mask=visibility_mask,
    )
    plot_single_surface(
        combined_sharp_surface,
        sample,
        expanded.expanded_df,
        title=(
            f"{title_prefix} | sharp combined: selection x success^{sharp_alpha:.0f}\n"
            f"Original passer: {actor_name} | completed={sample.completed} | sample_idx={args.sample_idx}"
        ),
        out_path=out_dir / f"sample_{args.sample_idx}_{file_prefix}_combined_sharp_success2.png",
        is_selection=True,
        selection_scale=args.selection_scale,
        selection_eps=args.selection_eps,
        selection_q=args.selection_q,
        visibility_mask=visibility_mask,
    )
    plot_single_surface(
        log_combined_surface,
        sample,
        expanded.expanded_df,
        title=(
            f"{title_prefix} | log combined: log(sel) + {log_beta:.0f}*log(succ)\n"
            f"Original passer: {actor_name} | completed={sample.completed} | sample_idx={args.sample_idx}"
        ),
        out_path=out_dir / f"sample_{args.sample_idx}_{file_prefix}_combined_log_success2.png",
        is_selection=True,
        selection_scale=args.selection_scale,
        selection_eps=args.selection_eps,
        selection_q=args.selection_q,
        visibility_mask=visibility_mask,
    )
    plot_difference_surface(
        combined_surface - selection_surface,
        sample,
        expanded.expanded_df,
        title=(
            f"{title_prefix} | difference: combined - selection\n"
            f"Original passer: {actor_name} | completed={sample.completed} | sample_idx={args.sample_idx}"
        ),
        out_path=out_dir / f"sample_{args.sample_idx}_{file_prefix}_difference_combined_minus_selection.png",
        visibility_mask=visibility_mask,
    )
    plot_difference_surface(
        combined_sharp_surface - selection_surface,
        sample,
        expanded.expanded_df,
        title=(
            f"{title_prefix} | difference: sharp combined - selection\n"
            f"Original passer: {actor_name} | completed={sample.completed} | sample_idx={args.sample_idx}"
        ),
        out_path=out_dir / f"sample_{args.sample_idx}_{file_prefix}_difference_sharp_minus_selection.png",
        visibility_mask=visibility_mask,
    )
    plot_difference_surface(
        log_combined_surface - selection_surface,
        sample,
        expanded.expanded_df,
        title=(
            f"{title_prefix} | difference: log combined - selection\n"
            f"Original passer: {actor_name} | completed={sample.completed} | sample_idx={args.sample_idx}"
        ),
        out_path=out_dir / f"sample_{args.sample_idx}_{file_prefix}_difference_log_minus_selection.png",
        visibility_mask=visibility_mask,
    )

    ncols = 4 if old_selection_surface is not None else 3
    pitch = Pitch(pitch_type="statsbomb", line_color="black", linewidth=1.2)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 6.5))
    if ncols == 1:
        axes = [axes]

    ims = []
    panel_idx = 0
    if old_selection_surface is not None:
        pitch.draw(ax=axes[panel_idx])
        ims.append(
            draw_surface(
                axes[panel_idx],
                pitch,
                old_selection_surface,
                expanded.expanded_df,
                sample.event_id,
                title=f"Old Selection\np(dest)={old_selection_dest_prob:.6f}",
                is_selection=True,
                selection_scale=args.selection_scale,
                selection_eps=args.selection_eps,
                selection_q=args.selection_q,
            )
        )
        panel_idx += 1

    pitch.draw(ax=axes[panel_idx])
    ims.append(
        draw_surface(
            axes[panel_idx],
            pitch,
            selection_surface,
            expanded.expanded_df,
            sample.event_id,
            title=f"New Selection\np(dest)={selection_dest_prob:.6f}",
            is_selection=True,
            selection_scale=args.selection_scale,
            selection_eps=args.selection_eps,
            selection_q=args.selection_q,
        )
    )
    panel_idx += 1

    pitch.draw(ax=axes[panel_idx])
    ims.append(
        draw_surface(
            axes[panel_idx],
            pitch,
            success_surface,
            expanded.expanded_df,
            sample.event_id,
            title=f"Pass Success\np(success @ dest)={success_dest_prob:.6f}",
            is_selection=False,
            selection_scale=args.selection_scale,
            selection_eps=args.selection_eps,
            selection_q=args.selection_q,
        )
    )
    panel_idx += 1

    pitch.draw(ax=axes[panel_idx])
    ims.append(
        draw_surface(
            axes[panel_idx],
            pitch,
            combined_surface,
            expanded.expanded_df,
            sample.event_id,
            title=f"Combined\np(dest weighted)={combined_dest_prob:.6f}",
            is_selection=True,
            selection_scale=args.selection_scale,
            selection_eps=args.selection_eps,
            selection_q=args.selection_q,
        )
    )

    fig.suptitle(
        f"match={args.match_id} sample_idx={args.sample_idx} actor={actor_name} completed={sample.completed}",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved comparison -> {out_path}")
    print(f"saved single-panel visualizations -> {out_dir}")


if __name__ == "__main__":
    main()
