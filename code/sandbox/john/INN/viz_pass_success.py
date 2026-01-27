from __future__ import annotations

import argparse
from pathlib import Path

import torch

from soccermap.statsbomb_io import load_events, load_threesixty, load_lineups
from soccermap.expand import build_expanded_dfs
from soccermap.dataset import PassDataset
from soccermap.model import SoccerMap, SoccerMapConfig
from soccermap.viz import plot_pass_success_surface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--match_id", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--sample_idx", type=int, default=0)
    ap.add_argument("--out", type=str, default="viz/pass_success_surface.png")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--compute_velocities", action="store_true")

    ap.add_argument("--cmap", type=str, default="RdBu_r")

    ap.add_argument("--vmin", type=float, default=0.0)
    ap.add_argument("--vmax", type=float, default=1.0)

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
        x = sample.channels.unsqueeze(0).to(args.device)
        logits = model(x)  # (1,1,L,W)
        prob = torch.sigmoid(logits)[0, 0]  # (L,W) in [0,1]

        p_dest = float(prob.view(-1)[sample.dest_index].cpu().item())
        prob_LW = prob.cpu().numpy()

    title = f"Sample idx={args.sample_idx} | p(dest cell)={p_dest:.6f} | true complete={sample.completed}"

    out_path = Path(args.out)
    plot_pass_success_surface(
        prob_LW,
        m.expanded_df,
        sample.event_id,
        title=title,
        out_path=str(out_path),
        show=False,
        cmap=args.cmap,
    )
    print(f"saved visualization -> {out_path}")


if __name__ == "__main__":
    main()
