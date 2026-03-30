from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from soccermap.context import DEFAULT_CONTEXT_DIM
from soccermap.dataset import PassDataset
from soccermap.expand import build_expanded_dfs
from soccermap.model import SoccerMap, SoccerMapConfig, SoccerMapWithPlayerEmbed
from soccermap.statsbomb_io import load_events, load_lineups, load_threesixty


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_data_root = project_root / "data" / "leverkusen_data"
    default_ckpt = project_root / "checkpoints" / "testing_with_new_stuff3.pt"

    ap = argparse.ArgumentParser(
        description="Inspect relative influence of CNN channels, player embedding, and context branch."
    )
    ap.add_argument("--ckpt", type=str, default=str(default_ckpt))
    ap.add_argument("--data_root", type=str, default=str(default_data_root))
    ap.add_argument("--match_id", type=str, default="3895302")
    ap.add_argument("--team_filter", type=str, default="Bayer Leverkusen")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=128)
    ap.add_argument("--compute_velocities", action="store_true")
    return ap.parse_args()


def build_model_from_ckpt(
    ckpt: Dict,
    device: str,
) -> Tuple[torch.nn.Module, bool, Dict[str, int]]:
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
            late_fusion=ckpt.get("late_fusion", True),
            cfg=cfg,
        ).to(device)
    else:
        model = SoccerMap(cfg).to(device)

    model.load_state_dict(state)
    model.eval()
    return model, uses_player_embed, player_id_mapping


def build_dataset(
    data_root: str,
    match_id: str,
    *,
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


def batch_indices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, end


def gather_batch(
    ds: PassDataset,
    idxs: List[int],
    *,
    device: str,
    uses_player_embed: bool,
    player_id_mapping: Dict[str, int],
    expected_in_channels: Optional[int] = None,
):
    batch = [ds[i] for i in idxs]
    channels = torch.stack([b.channels for b in batch]).to(device)
    dest = torch.tensor([b.dest_index for b in batch], dtype=torch.long, device=device)
    context = torch.stack([b.context_features for b in batch]).to(device)

    if expected_in_channels is not None and channels.shape[1] != expected_in_channels:
        if channels.shape[1] > expected_in_channels:
            channels = channels[:, :expected_in_channels]
        else:
            pad = expected_in_channels - channels.shape[1]
            channels = torch.cat(
                [
                    channels,
                    torch.zeros(
                        channels.shape[0],
                        pad,
                        channels.shape[2],
                        channels.shape[3],
                        dtype=channels.dtype,
                        device=channels.device,
                    ),
                ],
                dim=1,
            )

    actor_ids = None
    if uses_player_embed:
        actor_ids = torch.tensor(
            [player_id_mapping.get(b.actor_player_name, 0) for b in batch],
            dtype=torch.long,
            device=device,
        )

    return channels, context, dest, actor_ids


def pass_selection_loss(logits: torch.Tensor, dest_index: torch.Tensor) -> torch.Tensor:
    flat = logits.view(logits.shape[0], -1)
    return F.cross_entropy(flat, dest_index)


def forward_variant(
    model: torch.nn.Module,
    channels: torch.Tensor,
    context: torch.Tensor,
    actor_ids: Optional[torch.Tensor],
    *,
    zero_channels: bool = False,
    zero_context: bool = False,
    zero_embed: bool = False,
) -> torch.Tensor:
    if isinstance(model, SoccerMapWithPlayerEmbed):
        run_channels = torch.zeros_like(channels) if zero_channels else channels
        run_context = torch.zeros_like(context) if zero_context else context
        run_actor_ids = torch.zeros_like(actor_ids) if zero_embed else actor_ids
        return model(run_channels, run_actor_ids, run_context)

    run_channels = torch.zeros_like(channels) if zero_channels else channels
    return model(run_channels)


def inspect_direct_weights(model: torch.nn.Module) -> List[str]:
    lines: List[str] = []

    if not isinstance(model, SoccerMapWithPlayerEmbed):
        lines.append("Checkpoint has no player/context branch; only spatial CNN is present.")
        return lines

    if not model.late_fusion:
        lines.append("Direct learned weights")
        lines.append("- late fusion head is disabled for this checkpoint; selection uses only early concat + FiLM")
    else:
        lines.append("Direct learned weights")
        lines.append("- late fusion head (`embed_head`) mixes final CNN logit map + embedding dims + context dims")

    with torch.no_grad():
        late_w = model.embed_head.weight.detach().view(-1)
        cnn_w = late_w[:1]
        embed_w = late_w[1 : 1 + model.embed_dim]
        context_w = late_w[1 + model.embed_dim :]

        def block_stats(name: str, w: torch.Tensor) -> str:
            if w.numel() == 0:
                return f"  {name:<12} dims=0"
            abs_mean = float(w.abs().mean().item())
            l2 = float(w.norm().item())
            return f"  {name:<12} dims={w.numel():>2} abs_mean={abs_mean:.6f} l2={l2:.6f}"

        lines.append(block_stats("cnn_logit", cnn_w))
        lines.append(block_stats("embedding", embed_w))
        lines.append(block_stats("context", context_w))

        denom = float(cnn_w.abs().sum() + embed_w.abs().sum() + context_w.abs().sum() + 1e-12)
        lines.append("  normalized abs-weight share in late fusion head:")
        lines.append(f"    cnn_logit = {float(cnn_w.abs().sum() / denom):.4f}")
        lines.append(f"    embedding = {float(embed_w.abs().sum() / denom):.4f}")
        lines.append(f"    context   = {float(context_w.abs().sum() / denom):.4f}")

        first_conv = model.feat1.c1.conv.weight.detach()
        spatial_w = first_conv[:, : model.cfg.in_channels]
        embed_input_w = first_conv[:, model.cfg.in_channels :]
        lines.append("- first conv (`feat1.c1`) sees raw spatial channels plus broadcast embedding")
        lines.append(
            f"  spatial_input dims={spatial_w.numel():>5} abs_mean={float(spatial_w.abs().mean().item()):.6f} "
            f"l2={float(spatial_w.norm().item()):.6f}"
        )
        lines.append(
            f"  embed_input   dims={embed_input_w.numel():>5} abs_mean={float(embed_input_w.abs().mean().item()):.6f} "
            f"l2={float(embed_input_w.norm().item()):.6f}"
        )

        for name in ("film1", "film2", "film3"):
            layer = getattr(model, name)
            lines.append(
                f"- {name:<5} weight_l2={float(layer.weight.detach().norm().item()):.6f} "
                f"bias_l2={float(layer.bias.detach().norm().item()):.6f}"
            )

    return lines


@torch.no_grad()
def inspect_context_encoder_activity(
    model: torch.nn.Module,
    ds: PassDataset,
    *,
    device: str,
    batch_size: int,
    max_samples: int,
    uses_player_embed: bool,
    player_id_mapping: Dict[str, int],
) -> List[str]:
    if not isinstance(model, SoccerMapWithPlayerEmbed) or model.context_ffn is None or model.context_dim <= 0:
        return ["Context encoder activity", "- model has no active context encoder"]

    n = min(len(ds), max_samples)
    if n == 0:
        return ["Context encoder activity", "- no samples available"]

    raw_context_sum = None
    raw_context_sq_sum = None
    context_embed_sum = None
    context_embed_sq_sum = None
    raw_l2_total = 0.0
    embed_l2_total = 0.0
    count = 0

    for start, end in batch_indices(n, batch_size):
        idxs = list(range(start, end))
        _, context, _, _ = gather_batch(
            ds,
            idxs,
            device=device,
            uses_player_embed=uses_player_embed,
            player_id_mapping=player_id_mapping,
            expected_in_channels=getattr(model.cfg, "in_channels", None),
        )
        context_embed = model.context_ffn(context)

        if raw_context_sum is None:
            raw_context_sum = context.sum(dim=0)
            raw_context_sq_sum = (context ** 2).sum(dim=0)
            context_embed_sum = context_embed.sum(dim=0)
            context_embed_sq_sum = (context_embed ** 2).sum(dim=0)
        else:
            raw_context_sum += context.sum(dim=0)
            raw_context_sq_sum += (context ** 2).sum(dim=0)
            context_embed_sum += context_embed.sum(dim=0)
            context_embed_sq_sum += (context_embed ** 2).sum(dim=0)

        raw_l2_total += float(context.norm(dim=1).sum().item())
        embed_l2_total += float(context_embed.norm(dim=1).sum().item())
        count += len(idxs)

    raw_mean = raw_context_sum / count
    raw_var = raw_context_sq_sum / count - raw_mean ** 2
    raw_std = torch.sqrt(torch.clamp(raw_var, min=0.0))

    embed_mean = context_embed_sum / count
    embed_var = context_embed_sq_sum / count - embed_mean ** 2
    embed_std = torch.sqrt(torch.clamp(embed_var, min=0.0))

    lines = ["Context encoder activity"]
    lines.append(f"- samples used: {count}")
    lines.append(f"- raw context mean sample l2: {raw_l2_total / count:.6f}")
    lines.append(f"- encoded context mean sample l2: {embed_l2_total / count:.6f}")
    lines.append("- raw context dims:")
    for i in range(model.context_dim):
        lines.append(
            f"  raw[{i:02d}] mean={float(raw_mean[i].item()):.6f} std={float(raw_std[i].item()):.6f}"
        )
    lines.append("- encoded context dims:")
    for i in range(model.context_embed_dim):
        lines.append(
            f"  enc[{i:02d}] mean={float(embed_mean[i].item()):.6f} std={float(embed_std[i].item()):.6f}"
        )
    lines.append("- if encoded context l2 or std values are near zero, the context branch has collapsed")
    return lines


@torch.no_grad()
def inspect_ablation_effects(
    model: torch.nn.Module,
    ds: PassDataset,
    *,
    device: str,
    batch_size: int,
    max_samples: int,
    uses_player_embed: bool,
    player_id_mapping: Dict[str, int],
) -> List[str]:
    totals = {
        "baseline_loss": 0.0,
        "no_context_loss": 0.0,
        "no_embed_loss": 0.0,
        "no_channels_loss": 0.0,
        "no_context_surface_l1": 0.0,
        "no_embed_surface_l1": 0.0,
        "no_channels_surface_l1": 0.0,
    }
    count = 0
    raw_dataset_channels = None

    n = min(len(ds), max_samples)
    for start, end in batch_indices(n, batch_size):
        idxs = list(range(start, end))
        channels, context, dest, actor_ids = gather_batch(
            ds,
            idxs,
            device=device,
            uses_player_embed=uses_player_embed,
            player_id_mapping=player_id_mapping,
            expected_in_channels=getattr(model.cfg, "in_channels", None),
        )
        if raw_dataset_channels is None and len(idxs) > 0:
            raw_dataset_channels = int(ds[idxs[0]].channels.shape[0])

        baseline_logits = forward_variant(model, channels, context, actor_ids)
        baseline_probs = torch.softmax(baseline_logits.view(baseline_logits.shape[0], -1), dim=1)

        totals["baseline_loss"] += float(pass_selection_loss(baseline_logits, dest).item()) * len(idxs)

        no_context_logits = forward_variant(
            model, channels, context, actor_ids, zero_context=True
        )
        no_context_probs = torch.softmax(no_context_logits.view(no_context_logits.shape[0], -1), dim=1)
        totals["no_context_loss"] += float(pass_selection_loss(no_context_logits, dest).item()) * len(idxs)
        totals["no_context_surface_l1"] += float((baseline_probs - no_context_probs).abs().mean().item()) * len(idxs)

        if uses_player_embed:
            no_embed_logits = forward_variant(
                model, channels, context, actor_ids, zero_embed=True
            )
            no_embed_probs = torch.softmax(no_embed_logits.view(no_embed_logits.shape[0], -1), dim=1)
            totals["no_embed_loss"] += float(pass_selection_loss(no_embed_logits, dest).item()) * len(idxs)
            totals["no_embed_surface_l1"] += float((baseline_probs - no_embed_probs).abs().mean().item()) * len(idxs)

        no_channels_logits = forward_variant(
            model, channels, context, actor_ids, zero_channels=True
        )
        no_channels_probs = torch.softmax(no_channels_logits.view(no_channels_logits.shape[0], -1), dim=1)
        totals["no_channels_loss"] += float(pass_selection_loss(no_channels_logits, dest).item()) * len(idxs)
        totals["no_channels_surface_l1"] += float((baseline_probs - no_channels_probs).abs().mean().item()) * len(idxs)

        count += len(idxs)

    if count == 0:
        return ["No samples available."]

    def avg(name: str) -> float:
        return totals[name] / count

    lines = ["Ablation effects on real samples"]
    lines.append(f"- samples used: {count}")
    if raw_dataset_channels is not None:
        lines.append(
            f"- dataset channels: {raw_dataset_channels} | checkpoint expects: {model.cfg.in_channels}"
        )
        if raw_dataset_channels != model.cfg.in_channels:
            lines.append(
                "- note: channels were auto-aligned to the checkpoint before evaluation"
            )
    lines.append(f"- baseline CE loss: {avg('baseline_loss'):.6f}")
    lines.append(
        f"- zero context : loss={avg('no_context_loss'):.6f}  "
        f"delta={avg('no_context_loss') - avg('baseline_loss'):+.6f}  "
        f"surface_l1={avg('no_context_surface_l1'):.6f}"
    )
    if uses_player_embed:
        lines.append(
            f"- zero embed   : loss={avg('no_embed_loss'):.6f}  "
            f"delta={avg('no_embed_loss') - avg('baseline_loss'):+.6f}  "
            f"surface_l1={avg('no_embed_surface_l1'):.6f}"
        )
    lines.append(
        f"- zero channels: loss={avg('no_channels_loss'):.6f}  "
        f"delta={avg('no_channels_loss') - avg('baseline_loss'):+.6f}  "
        f"surface_l1={avg('no_channels_surface_l1'):.6f}"
    )
    lines.append("")
    lines.append("Interpretation")
    lines.append("- bigger positive loss delta means that branch mattered more for correct predictions")
    lines.append("- bigger surface_l1 means removing that branch changed the predicted map more")
    if model.late_fusion:
        lines.append("- `embed_head` weights alone are not enough because embedding also enters early via concat + FiLM")
        lines.append("- for pass-selection, the late broadcast embedding can only add a constant logit offset across the pitch")
    else:
        lines.append("- this checkpoint already bypasses late fusion, so player effects come only from early concat + FiLM")
    return lines


def main():
    args = parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    context_dim = ckpt.get("context_dim", DEFAULT_CONTEXT_DIM)
    model, uses_player_embed, player_id_mapping = build_model_from_ckpt(ckpt, args.device)

    print("=" * 80)
    print("Checkpoint summary")
    print("=" * 80)
    print("checkpoint:", args.ckpt)
    print("uses_player_embed:", uses_player_embed)
    print("cfg.in_channels:", model.cfg.in_channels)
    if uses_player_embed:
        print("embed_dim:", model.embed_dim)
        print("context_dim:", model.context_dim)
        print("context_embed_dim:", model.context_embed_dim)
        print("num_players:", ckpt.get("num_players", 0))

    print()
    print("=" * 80)
    print("Weight inspection")
    print("=" * 80)
    for line in inspect_direct_weights(model):
        print(line)

    ds = build_dataset(
        args.data_root,
        args.match_id,
        team_filter=args.team_filter or None,
        compute_velocities=args.compute_velocities,
        context_dim=context_dim,
    )

    print()
    print("=" * 80)
    print("Context encoder")
    print("=" * 80)
    for line in inspect_context_encoder_activity(
        model,
        ds,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        uses_player_embed=uses_player_embed,
        player_id_mapping=player_id_mapping,
    ):
        print(line)

    print()
    print("=" * 80)
    print("Prediction ablation")
    print("=" * 80)
    for line in inspect_ablation_effects(
        model,
        ds,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        uses_player_embed=uses_player_embed,
        player_id_mapping=player_id_mapping,
    ):
        print(line)


if __name__ == "__main__":
    main()
