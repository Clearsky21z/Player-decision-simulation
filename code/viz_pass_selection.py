from __future__ import annotations

import argparse
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from mplsoccer import Pitch

from soccermap.context import DEFAULT_CONTEXT_DIM
from soccermap.dataset import PassDataset, PassSample
from soccermap.expand import build_expanded_dfs
from soccermap.model import SoccerMap, SoccerMapConfig, SoccerMapWithPlayerEmbed
from soccermap.statsbomb_io import load_events, load_lineups, load_threesixty
from soccermap.viz import _extract_scene, _to_img_yx, plot_pass_selection_surface


def _infer_match_id(ckpt: dict, cli_match_id: str) -> str:
    match_id = (cli_match_id or "").strip()
    if match_id:
        return match_id

    holdout = str(ckpt.get("holdout_match_id", "") or "").strip()
    if holdout:
        return holdout

    train_ids = ckpt.get("train_match_ids") or []
    if len(train_ids) == 1:
        return str(train_ids[0])

    raise ValueError(
        "Could not infer match_id from checkpoint. Pass --match_id explicitly."
    )


def _slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.strip().lower().replace(" ", "_")
    value = re.sub(r"[^a-z0-9_-]+", "", value)
    return value


def _normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def resolve_player_name(query: str, player_id_mapping: Dict[str, int]) -> Tuple[str, int]:
    if query in player_id_mapping:
        return query, int(player_id_mapping[query])

    normalized_query = _normalize_name(query)
    if not normalized_query:
        raise KeyError("Empty player name query.")

    exact_normalized: Dict[str, str] = {}
    token_matches: list[Tuple[int, int, str]] = []
    query_tokens = normalized_query.split()

    for candidate in player_id_mapping:
        normalized_candidate = _normalize_name(candidate)
        exact_normalized.setdefault(normalized_candidate, candidate)
        candidate_tokens = normalized_candidate.split()

        if normalized_query == normalized_candidate:
            return candidate, int(player_id_mapping[candidate])

        if query_tokens and all(token in candidate_tokens for token in query_tokens):
            token_matches.append((len(candidate_tokens), len(candidate), candidate))

    if normalized_query in exact_normalized:
        resolved = exact_normalized[normalized_query]
        return resolved, int(player_id_mapping[resolved])

    if token_matches:
        token_matches.sort()
        resolved = token_matches[0][2]
        return resolved, int(player_id_mapping[resolved])

    preview = ", ".join(sorted(player_id_mapping.keys())[:8])
    raise KeyError(
        f"player {query!r} not found in checkpoint player_id_mapping. "
        f"Examples: {preview}"
    )


def _event_metadata(expanded_df, event_id: str) -> tuple[Optional[int], Optional[int]]:
    actor = expanded_df.loc[
        (expanded_df["event_id"] == event_id) & (expanded_df["actor"] == True)
    ]
    if actor.empty:
        return None, None

    row = actor.iloc[0]
    event_index = row.get("event_index")
    completed = row.get("pass_completed")

    try:
        event_index = None if event_index is None else int(event_index)
    except (TypeError, ValueError):
        event_index = None

    try:
        completed = None if completed is None else int(completed)
    except (TypeError, ValueError):
        completed = None

    return event_index, completed


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _load_dataset(
    *,
    data_root: str,
    match_id: str,
    compute_velocities: bool,
    team_filter: Optional[str],
    context_dim: int,
):
    events = load_events(data_root, match_id)
    threesixty = load_threesixty(data_root, match_id)
    lineups = load_lineups(data_root, match_id)
    expanded = build_expanded_dfs(events, threesixty, lineups)
    ds = PassDataset(
        expanded.expanded_df,
        compute_velocities=compute_velocities,
        only_passes=True,
        team_filter=team_filter,
        context_dim=context_dim,
    )
    return expanded, ds


def _build_model_from_ckpt(ckpt: dict, device: str):
    state = ckpt.get("model_state", ckpt)
    cfg_kwargs = ckpt.get("config", {})
    cfg = SoccerMapConfig(**cfg_kwargs) if cfg_kwargs else SoccerMapConfig()
    uses_player_embed = any(k.startswith("player_embedding.") for k in state.keys())
    if "input_channels" in ckpt:
        expected_input_channels = int(ckpt["input_channels"])
    elif uses_player_embed:
        expected_input_channels = int(state["feat1.c1.conv.weight"].shape[1]) - int(
            ckpt.get("embed_dim", 8)
        )
    else:
        expected_input_channels = int(state["feat1.c1.conv.weight"].shape[1])

    if uses_player_embed:
        model = SoccerMapWithPlayerEmbed(
            num_players=ckpt.get("num_players", 0),
            embed_dim=ckpt.get("embed_dim", 8),
            context_dim=ckpt.get("context_dim", 0),
            context_hidden_dim=ckpt.get("context_hidden_dim", 16),
            context_embed_dim=ckpt.get("context_embed_dim", 8),
            late_fusion=ckpt.get("late_fusion", True),
            cfg=cfg,
        ).to(device)
    else:
        model = SoccerMap(cfg).to(device)

    model.load_state_dict(state)
    model.eval()
    model.expected_input_channels = expected_input_channels
    return model, uses_player_embed


def _adapt_channels_for_model(channels: torch.Tensor, expected_input_channels: int) -> torch.Tensor:
    actual_input_channels = int(channels.shape[0])
    if actual_input_channels == expected_input_channels:
        return channels
    if actual_input_channels == 15 and expected_input_channels == 14:
        return torch.cat([channels[:2], channels[3:]], dim=0)
    if actual_input_channels == 15 and expected_input_channels == 11:
        return torch.cat([channels[:3], channels[7:]], dim=0)
    if actual_input_channels == 11 and expected_input_channels == 15:
        out = torch.zeros(
            (15, channels.shape[1], channels.shape[2]),
            dtype=channels.dtype,
            device=channels.device,
        )
        out[0] = channels[0]
        out[1] = channels[1]
        out[2] = channels[2]
        out[7] = channels[3]
        out[8] = channels[4]
        out[9] = channels[5]
        out[10] = channels[6]
        out[11] = channels[7]
        out[12] = channels[8]
        out[13] = channels[9]
        out[14] = channels[10]
        return out
    if actual_input_channels == 11 and expected_input_channels == 14:
        out = torch.zeros(
            (14, channels.shape[1], channels.shape[2]),
            dtype=channels.dtype,
            device=channels.device,
        )
        out[0] = channels[0]
        out[1] = channels[1]
        out[6] = channels[3]
        out[7] = channels[4]
        out[8] = channels[5]
        out[9] = channels[6]
        out[10] = channels[7]
        out[11] = channels[8]
        out[12] = channels[9]
        out[13] = channels[10]
        return out
    raise RuntimeError(
        f"Checkpoint expects {expected_input_channels} input channels, but dataset produced "
        f"{actual_input_channels}. No automatic adapter is defined for this combination."
    )


@torch.no_grad()
def predict_pass_selection_embed(
    model: SoccerMapWithPlayerEmbed,
    sample: PassSample,
    actor_id: int,
):
    device = _model_device(model)
    expected_input_channels = int(
        getattr(model, "expected_input_channels", sample.channels.shape[0])
    )
    x = _adapt_channels_for_model(sample.channels, expected_input_channels).unsqueeze(0).to(device)
    context = sample.context_features.unsqueeze(0).to(device)
    actor_ids = torch.tensor([actor_id], dtype=torch.long, device=device)
    logits = model(x, actor_ids, context)

    flat = logits.view(1, -1)
    prob_flat = torch.softmax(flat, dim=1)[0]
    p_dest = float(prob_flat[sample.dest_index].detach().cpu().item())
    prob_LW = prob_flat.view(logits.shape[2], logits.shape[3]).detach().cpu().numpy()
    return logits, p_dest, prob_LW


def _visibility_channel_index(num_channels: int) -> Optional[int]:
    if num_channels == 11:
        return 10
    if num_channels == 14:
        return 13
    if num_channels == 15:
        return 14
    return None


def _grid_index_to_statsbomb_xy(prob_LW: np.ndarray, l_idx: int, w_idx: int) -> tuple[float, float]:
    L, W = prob_LW.shape
    x = (float(l_idx) + 0.5) * (120.0 / float(L))
    y = (float(w_idx) + 0.5) * (80.0 / float(W))
    return x, y


def plot_pass_selection_embed(
    model: SoccerMapWithPlayerEmbed,
    ds: PassDataset,
    expanded_df,
    player_id_mapping: Dict[str, int],
    *,
    sample_idx: Optional[int] = None,
    swap_player: Optional[str] = None,
    out_path: Optional[str] = None,
    show: bool = True,
    scale: str = "log",
    q: float = 0.995,
    eps: float = 1e-12,
    cmap: str = "RdBu_r",
):
    if sample_idx is None:
        sample_idx = random.randint(0, len(ds) - 1)
    if sample_idx < 0 or sample_idx >= len(ds):
        raise IndexError(f"sample_idx {sample_idx} out of range for dataset size {len(ds)}")

    sample = ds[sample_idx]
    if swap_player:
        resolved_name, actor_id = resolve_player_name(swap_player, player_id_mapping)
    else:
        actor_name = sample.actor_player_name or ""
        resolved_name, actor_id = resolve_player_name(actor_name, player_id_mapping)

    _, p_dest, prob_LW = predict_pass_selection_embed(model, sample, actor_id)
    flat = prob_LW.reshape(-1)
    peak_idx = int(np.argmax(flat))
    peak_l = peak_idx // prob_LW.shape[1]
    peak_w = peak_idx % prob_LW.shape[1]
    peak_x, peak_y = _grid_index_to_statsbomb_xy(prob_LW, peak_l, peak_w)

    passer_xy, dest_xy, completed, atk_xy, dfn_xy = _extract_scene(expanded_df, sample.event_id)
    visibility_channel_idx = _visibility_channel_index(int(sample.channels.shape[0]))
    visibility_mask = None
    if visibility_channel_idx is not None and visibility_channel_idx < sample.channels.shape[0]:
        visibility_mask = sample.channels[visibility_channel_idx].detach().cpu().numpy()

    event_index, completed = _event_metadata(expanded_df, sample.event_id)

    pitch = Pitch(pitch_type="statsbomb", line_color="black", linewidth=1.2)
    fig, ax = pitch.draw(figsize=(12, 8))

    img = _to_img_yx(prob_LW)
    img_plot = np.clip(img, eps, None)
    vmax = float(np.quantile(img_plot, q))
    vmax = max(vmax, eps * 10)
    render_cmap = plt.get_cmap(cmap).copy()
    render_cmap.set_under("white")

    if scale == "log":
        norm = mcolors.LogNorm(vmin=eps, vmax=vmax)
    elif scale == "percentile":
        vmax = max(float(np.quantile(img, q)), eps)
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
        img_plot = img
    else:
        norm = None
        img_plot = img

    im = ax.imshow(
        img_plot,
        extent=pitch.extent,
        origin="upper",
        aspect="auto",
        cmap=render_cmap,
        norm=norm,
        alpha=0.88,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    if visibility_mask is not None:
        vis_img = _to_img_yx(visibility_mask)
        ax.contour(
            vis_img,
            levels=[0.5],
            colors="lime",
            linewidths=2.2,
            extent=pitch.extent,
            origin="upper",
        )

    if atk_xy.size:
        pitch.scatter(
            atk_xy[:, 0],
            atk_xy[:, 1],
            s=80,
            ax=ax,
            color="dodgerblue",
            edgecolors="black",
            zorder=5,
            label="Teammates",
        )
    if dfn_xy.size:
        pitch.scatter(
            dfn_xy[:, 0],
            dfn_xy[:, 1],
            s=80,
            ax=ax,
            color="red",
            edgecolors="black",
            zorder=5,
            label="Opponents",
        )

    if passer_xy is not None:
        pitch.scatter(
            [passer_xy[0]],
            [passer_xy[1]],
            s=160,
            ax=ax,
            color="gold",
            edgecolors="black",
            zorder=6,
            label="Ball carrier position",
        )

    if dest_xy is not None:
        pitch.scatter(
            [dest_xy[0]],
            [dest_xy[1]],
            s=170,
            marker="x",
            ax=ax,
            color="black",
            zorder=7,
            label="True destination",
        )
        if passer_xy is not None:
            pitch.arrows(
                passer_xy[0],
                passer_xy[1],
                dest_xy[0],
                dest_xy[1],
                ax=ax,
                width=2,
                headwidth=6,
                headlength=6,
                headaxislength=5,
                color="black",
            )

    if passer_xy is not None:
        pitch.arrows(
            passer_xy[0],
            passer_xy[1],
            peak_x,
            peak_y,
            ax=ax,
            width=2,
            headwidth=6,
            headlength=6,
            headaxislength=5,
            color="blue",
        )

    pitch.scatter(
        [peak_x],
        [peak_y],
        s=170,
        marker="x",
        ax=ax,
        color="blue",
        zorder=7,
        label="Predicted destination",
    )

    if swap_player:
        title = (
            f"Swapped to {resolved_name} (embed_id={actor_id})\n"
            f"Original passer: {sample.actor_player_name} | completed={completed} | "
            f"sample_idx={sample_idx} | event_idx={event_index}"
        )
    else:
        title = (
            f"{resolved_name} (embed_id={actor_id})\n"
            f"Original passer: {sample.actor_player_name} | completed={completed} | "
            f"sample_idx={sample_idx} | event_idx={event_index}"
        )
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8)

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, prob_LW, (peak_x, peak_y), p_dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--match_id", type=str, default="")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--sample_idx", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--compute_velocities", action="store_true")
    ap.add_argument("--swap_player", type=str, default="")
    ap.add_argument("--team_filter", type=str, default="")
    ap.add_argument("--scale", type=str, default="log", choices=["log", "percentile", "linear"])
    ap.add_argument("--q", type=float, default=0.995)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--cmap", type=str, default="RdBu_r")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    match_id = _infer_match_id(ckpt, args.match_id)
    dataset_context_dim = int(ckpt.get("context_dim", DEFAULT_CONTEXT_DIM))
    team_filter = args.team_filter.strip() or None

    expanded, ds = _load_dataset(
        data_root=args.data_root,
        match_id=match_id,
        compute_velocities=args.compute_velocities,
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
    sample = ds[args.sample_idx]
    player_id_mapping = ckpt.get("player_id_mapping", {})

    if uses_player_embed:
        fig, prob_LW, peak_xy, p_dest = plot_pass_selection_embed(
            model,
            ds,
            expanded.expanded_df,
            player_id_mapping,
            sample_idx=args.sample_idx,
            swap_player=args.swap_player.strip() or None,
            out_path=str(
                Path(args.out)
                if args.out.strip()
                else Path("viz")
                / (
                    f"{Path(args.ckpt).stem}_match_{match_id}_sample_{args.sample_idx}_"
                    f"{_slugify((args.swap_player.strip() or sample.actor_player_name or 'actual_passer'))}.png"
                )
            ),
            show=False,
            scale=args.scale,
            q=args.q,
            eps=args.eps,
            cmap=args.cmap,
        )
        requested_name = args.swap_player.strip() or (sample.actor_player_name or "")
        resolved_name, actor_id = resolve_player_name(requested_name, player_id_mapping)
        out_path = (
            Path(args.out)
            if args.out.strip()
            else Path("viz")
            / (
                f"{Path(args.ckpt).stem}_match_{match_id}_sample_{args.sample_idx}_"
                f"{_slugify(resolved_name or 'actual_passer')}.png"
            )
        )
    else:
        x = _adapt_channels_for_model(
            sample.channels,
            int(getattr(model, "expected_input_channels", sample.channels.shape[0])),
        ).unsqueeze(0).to(args.device)
        with torch.no_grad():
            logits = model(x)
            flat = logits.view(1, -1)
            prob_flat = torch.softmax(flat, dim=1)[0]
            p_dest = float(prob_flat[sample.dest_index].detach().cpu().item())
            prob_LW = prob_flat.view(logits.shape[2], logits.shape[3]).detach().cpu().numpy()
        resolved_name = sample.actor_player_name or ""
        out_path = (
            Path(args.out)
            if args.out.strip()
            else Path("viz")
            / (
                f"{Path(args.ckpt).stem}_match_{match_id}_sample_{args.sample_idx}_"
                f"{_slugify(resolved_name or 'actual_passer')}.png"
            )
        )
        event_index, completed = _event_metadata(expanded.expanded_df, sample.event_id)
        title = (
            f"sample_idx={args.sample_idx} | event_idx={event_index} | event_id={sample.event_id}\n"
            f"embed_actor={resolved_name} | true_actor={sample.actor_player_name} | "
            f"p(dest)={p_dest:.6f} | completed={completed}"
        )
        plot_pass_selection_surface(
            prob_LW,
            expanded.expanded_df,
            sample.event_id,
            title=title,
            out_path=str(out_path),
            show=False,
            scale=args.scale,
            q=args.q,
            eps=args.eps,
            cmap=args.cmap,
        )

    event_index, completed = _event_metadata(expanded.expanded_df, sample.event_id)
    print(
        f"visualized match_id={match_id} sample_idx={args.sample_idx} "
        f"event_idx={event_index} event_id={sample.event_id}"
    )
    print(
        f"embed_actor={resolved_name!r} true_actor={sample.actor_player_name!r} "
        f"completed={completed} p_dest={p_dest:.6f}"
    )
    print(f"saved visualization -> {out_path}")


if __name__ == "__main__":
    main()
