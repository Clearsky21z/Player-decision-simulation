from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split

from soccermap.statsbomb_io import load_events, load_threesixty, load_lineups
from soccermap.context import context_feature_names
from soccermap.expand import build_expanded_dfs, build_player_id_mapping
from soccermap.dataset import PassDataset
from soccermap.model import (
    SoccerMapConfig,
    SoccerMapWithPlayerEmbed,
    pass_selection_kl_loss,
    pass_selection_loss,
    pass_selection_teammate_kl_loss,
    pass_selection_surface,
)


def list_available_match_ids(data_root: str) -> List[str]:
    """
    Returns match_ids (as strings) that have events + three-sixty + lineups json files.
    """
    root = Path(data_root)
    ev_dir = root / "events"
    ts_dir = root / "three-sixty"
    lu_dir = root / "lineups"

    ids: List[str] = []
    for p in sorted(ev_dir.glob("*.json")):
        mid = p.stem
        if (ts_dir / f"{mid}.json").exists() and (lu_dir / f"{mid}.json").exists():
            ids.append(mid)
    return ids


def parse_id_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _batch_to_tensors(
    batch: Sequence,
    player_id_mapping: dict[str, int],
    device: str,
):
    channels = torch.stack([b.channels for b in batch]).to(device)
    dest = torch.tensor([b.dest_index for b in batch], dtype=torch.long, device=device)
    actor_ids = torch.tensor(
        [player_id_mapping.get(b.actor_player_name, 0) for b in batch],
        dtype=torch.long,
        device=device,
    )
    return channels, dest, actor_ids


def _infer_input_channels(ds) -> int:
    sample = ds[0]
    return int(sample.channels.shape[0])


def _embedding_norm_summary(model: SoccerMapWithPlayerEmbed) -> tuple[float, float]:
    weights = model.player_embedding.weight.detach()[1:]
    if weights.numel() == 0:
        return 0.0, 0.0
    norms = weights.norm(dim=1)
    return float(norms.mean().item()), float(norms.max().item())


def _roll_to_distinct_nonzero(actor_ids: torch.Tensor) -> torch.Tensor:
    if actor_ids.numel() <= 1:
        return actor_ids

    swapped = actor_ids.roll(shifts=1)
    same_or_unknown = (swapped == actor_ids) | (swapped == 0) | (actor_ids == 0)

    if torch.all(~same_or_unknown):
        return swapped

    nonzero = actor_ids[actor_ids != 0].unique()
    if nonzero.numel() <= 1:
        return actor_ids

    replacement = nonzero.roll(shifts=1)
    out = actor_ids.clone()
    for src, dst in zip(nonzero.tolist(), replacement.tolist()):
        mask = actor_ids == src
        out[mask] = dst
    return out


@torch.no_grad()
def _swap_surface_delta(
    model: SoccerMapWithPlayerEmbed,
    channels: torch.Tensor,
    actor_ids: torch.Tensor,
) -> float:
    swapped_ids = _roll_to_distinct_nonzero(actor_ids)
    if torch.equal(swapped_ids, actor_ids):
        return 0.0

    orig_surface = pass_selection_surface(model(channels, actor_ids))
    swapped_surface = pass_selection_surface(model(channels, swapped_ids))
    return float((orig_surface - swapped_surface).abs().mean().item())


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, required=True)

    ap.add_argument("--match_id", type=str, default="")

    ap.add_argument("--train_match_ids", type=str, default="")

    ap.add_argument("--holdout_match_id", type=str, default="")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--compute_velocities", action="store_true")
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--embed_team", type=str, default="Bayer Leverkusen")
    ap.add_argument("--loss", type=str, default="teammate_kl",
                    choices=["kl", "ce", "teammate_kl"],
                    help="Loss function: kl (Gaussian KL), ce (cross-entropy), teammate_kl (teammate-weighted KL)")
    ap.add_argument("--embed_dim", type=int, default=8)
    ap.add_argument("--context_dim", type=int, default=0)
    ap.add_argument("--context_hidden_dim", type=int, default=16)
    ap.add_argument("--context_embed_dim", type=int, default=8)
    ap.add_argument("--out_ckpt", type=str, default="checkpoints/pass_selection.pt")
    args = ap.parse_args()

    train_ids = parse_id_list(args.train_match_ids)

    if not train_ids:
        if args.match_id.strip():
            train_ids = [args.match_id.strip()]
        else:
            train_ids = list_available_match_ids(args.data_root)

    holdout = args.holdout_match_id.strip()
    if holdout:
        train_ids = [mid for mid in train_ids if mid != holdout]

    if not train_ids:
        raise RuntimeError("No training matches found. Check your StatsBombData folder/files.")

    print(f"Training on {len(train_ids)} match(es). First few: {train_ids[:8]}{'...' if len(train_ids) > 8 else ''}")
    if holdout:
        print(f"Holding out match_id={holdout} (NOT used in training)")

    # --- collect lineups and build player ID mapping ---
    all_lineups = []
    for mid in train_ids:
        all_lineups.append(load_lineups(args.data_root, mid))

    player_id_mapping = build_player_id_mapping(all_lineups, team_name=args.embed_team)
    num_players = len(player_id_mapping)
    print(f"Player ID mapping: {num_players} unique players (team={args.embed_team})")

    # --- build datasets per match and concat ---
    ds_list = []
    total_passes = 0

    for mid in train_ids:
        events = load_events(args.data_root, mid)
        threesixty = load_threesixty(args.data_root, mid)
        lineups = load_lineups(args.data_root, mid)

        m = build_expanded_dfs(events, threesixty, lineups)
        ds_mid = PassDataset(
            m.expanded_df,
            compute_velocities=args.compute_velocities,
            only_passes=True,
            team_filter=args.embed_team,
            context_dim=args.context_dim,
        )

        ds_list.append(ds_mid)
        total_passes += len(ds_mid)
        print(f"  match {mid}: {len(ds_mid)} pass samples")

    full_ds = ConcatDataset(ds_list)
    print(f"Total samples: {len(full_ds)} (sum={total_passes})")

    input_channels = _infer_input_channels(full_ds)
    if input_channels != SoccerMapConfig().in_channels:
        raise RuntimeError(
            f"Dataset produces {input_channels} channels but SoccerMapConfig expects "
            f"{SoccerMapConfig().in_channels}. Update the config or channel builder before training."
        )

    # --- train / val split ---
    n_total = len(full_ds)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    print(f"Train: {n_train}  Val: {n_val}")

    collate = lambda batch: batch  # PassDataset returns dataclasses
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    fixed_val_batch = next(iter(val_dl), None)

    model = SoccerMapWithPlayerEmbed(
        num_players=num_players,
        embed_dim=args.embed_dim,
        context_dim=args.context_dim,
        context_hidden_dim=args.context_hidden_dim,
        context_embed_dim=args.context_embed_dim,
        late_fusion=False,
        cfg=SoccerMapConfig(),
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(
        f"Model conditioning: input_concat+film | late_fusion=off | loss: {args.loss} | "
        f"input_channels={input_channels} | context_dim={args.context_dim}"
    )

    def _compute_loss(logits, dest, channels):
        if args.loss == "kl":
            return pass_selection_kl_loss(logits, dest)
        elif args.loss == "ce":
            return pass_selection_loss(logits, dest)
        else:  # teammate_kl
            return pass_selection_teammate_kl_loss(logits, dest, channels[:, 0])

    for ep in range(args.epochs):
        # --- training ---
        model.train()
        train_total = 0.0
        train_n = 0
        for batch in train_dl:
            channels, dest, actor_ids = _batch_to_tensors(batch, player_id_mapping, args.device)
            context = torch.stack([b.context_features for b in batch]).to(args.device)

            logits = model(channels, actor_ids, context)
            loss = _compute_loss(logits, dest, channels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_total += float(loss.item()) * len(channels)
            train_n += len(channels)

        # --- validation ---
        model.eval()
        val_total = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_dl:
                channels, dest, actor_ids = _batch_to_tensors(batch, player_id_mapping, args.device)
                context = torch.stack([b.context_features for b in batch]).to(args.device)

                logits = model(channels, actor_ids, context)
                loss = _compute_loss(logits, dest, channels)

                val_total += float(loss.item()) * len(channels)
                val_n += len(channels)

        train_loss = train_total / max(train_n, 1)
        val_loss = val_total / max(val_n, 1)
        mean_norm, max_norm = _embedding_norm_summary(model)
        swap_delta = 0.0
        if fixed_val_batch:
            fixed_channels, _, fixed_actor_ids = _batch_to_tensors(fixed_val_batch, player_id_mapping, args.device)
            swap_delta = _swap_surface_delta(model, fixed_channels, fixed_actor_ids)
        print(
            f"epoch {ep+1}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  embed_norm_mean={mean_norm:.4f}  "
            f"embed_norm_max={max_norm:.4f}  swap_l1={swap_delta:.6f}"
        )

    # Save checkpoint for visualization / later evaluation
    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_type": "soccermap_player_conditioned",
            "conditioning": "input_concat+film",
            "late_fusion": False,
            "loss": args.loss,
            "train_match_ids": train_ids,
            "holdout_match_id": holdout,
            "config": SoccerMapConfig().__dict__,
            "input_channels": input_channels,
            "player_id_mapping": player_id_mapping,
            "num_players": num_players,
            "embed_dim": args.embed_dim,
            "context_dim": args.context_dim,
            "context_hidden_dim": args.context_hidden_dim,
            "context_embed_dim": args.context_embed_dim,
            "context_feature_names": context_feature_names(args.context_dim),
        },
        out_path,
    )
    print(f"saved checkpoint -> {out_path}")


if __name__ == "__main__":
    main()
