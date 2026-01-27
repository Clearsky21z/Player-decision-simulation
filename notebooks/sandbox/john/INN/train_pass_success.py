from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, ConcatDataset

from soccermap.statsbomb_io import load_events, load_threesixty, load_lineups
from soccermap.expand import build_expanded_dfs
from soccermap.dataset import PassDataset
from soccermap.model import SoccerMap, SoccerMapConfig, pass_success_loss


def list_available_match_ids(data_root: str) -> List[str]:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)

    # Backwards compatible
    ap.add_argument("--match_id", type=str, default="")

    # Option 2
    ap.add_argument("--train_match_ids", type=str, default="")
    ap.add_argument("--holdout_match_id", type=str, default="")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--compute_velocities", action="store_true")
    ap.add_argument("--out_ckpt", type=str, default="checkpoints/pass_success.pt")
    args = ap.parse_args()

    # --- decide training match ids ---
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
        raise RuntimeError("No training matches found. Check your StatsBombData files.")

    print(f"Training on {len(train_ids)} match(es). First few: {train_ids[:8]}{'...' if len(train_ids) > 8 else ''}")
    if holdout:
        print(f"Holding out match_id={holdout} (NOT used in training)")

    # --- build datasets per match and concat ---
    ds_list = []
    total_samples = 0
    for mid in train_ids:
        events = load_events(args.data_root, mid)
        threesixty = load_threesixty(args.data_root, mid)
        lineups = load_lineups(args.data_root, mid)

        m = build_expanded_dfs(events, threesixty, lineups)
        ds_mid = PassDataset(
            m.expanded_df,
            compute_velocities=args.compute_velocities,
            only_passes=True,
        )
        ds_list.append(ds_mid)
        total_samples += len(ds_mid)
        print(f"  match {mid}: {len(ds_mid)} pass samples")

    ds = ConcatDataset(ds_list)
    print(f"Total training samples: {len(ds)} (sum={total_samples})")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )

    model = SoccerMap(SoccerMapConfig()).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for ep in range(args.epochs):
        total = 0.0
        n = 0
        for batch in dl:
            # batch is list[PassSample]
            # Keep only samples with completion label (should be all for Pass)
            batch2 = [b for b in batch if b.completed is not None]
            if not batch2:
                continue

            channels = torch.stack([b.channels for b in batch2]).to(args.device)
            dest = torch.tensor([b.dest_index for b in batch2], dtype=torch.long, device=args.device)
            comp = torch.tensor([b.completed for b in batch2], dtype=torch.float32, device=args.device)

            logits = model(channels)  # (N,1,L,W)
            loss = pass_success_loss(logits, dest, comp)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item()) * len(channels)
            n += len(channels)

        print(f"epoch {ep+1}/{args.epochs}  loss={total/max(n,1):.4f}")

    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "train_match_ids": train_ids,
            "holdout_match_id": holdout,
            "config": SoccerMapConfig().__dict__,
        },
        out_path,
    )
    print(f"saved checkpoint -> {out_path}")


if __name__ == "__main__":
    main()
