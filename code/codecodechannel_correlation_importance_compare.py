from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from soccermap.statsbomb_io import load_events, load_threesixty, load_lineups
from soccermap.context import DEFAULT_CONTEXT_DIM
from soccermap.expand import build_expanded_dfs
from soccermap.dataset import PassDataset
from soccermap.model import (
    SoccerMap,
    SoccerMapConfig,
    SoccerMapWithPlayerEmbed,
    pass_selection_loss,
    pass_selection_kl_loss,
    pass_selection_teammate_kl_loss,
    pass_success_loss,
)

CHANNEL_NAMES = [
    "Teammate location",
    "Opponent location",
    "Teammate vx",
    "Teammate vy",
    "Opponent vx",
    "Opponent vy",
    "Distance to ball",
    "Distance to goal",
    "Goal-ball sin",
    "Goal-ball cos",
    "Angle to goal",
    "Pass-dir sin",
    "Pass-dir cos",
    "Visibility mask",
]

def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]

    default_data_root = project_root / "data" / "leverkusen_data"
    default_ckpt = project_root / "checkpoints" / "testing_with_new_stuff3.pt"
    default_out_dir = project_root / "viz" / "channel_analysis_compare"

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data_root",
        type=str,
        default=str(default_data_root),
        help="Path to StatsBomb data root",
    )
    ap.add_argument(
        "--match_id",
        type=str,
        default="3895302",
        help="StatsBomb match id",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default=str(default_ckpt),
        help="Path to model checkpoint",
    )
    ap.add_argument(
        "--task",
        type=str,
        default="pass_selection",
        choices=["pass_selection", "pass_success"],
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=256)
    ap.add_argument("--compute_velocities", action="store_true")
    ap.add_argument("--team_filter", type=str, default="")
    ap.add_argument("--out_dir", type=str, default=str(default_out_dir))
    ap.add_argument(
        "--loss_modes",
        nargs="+",
        default=["ce", "gaussian_kl", "teammate_kl"],
        choices=["ce", "gaussian_kl", "teammate_kl"],
        help="Loss functions to compare for pass_selection",
    )

    return ap.parse_args()


def build_dataset(
        data_root: str,
        match_id: str,
        *,
        compute_velocities: bool,
        team_filter: Optional[str],
        context_dim: int,
) -> Tuple[PassDataset, object]:
    events = load_events(data_root, match_id)
    threesixty = load_threesixty(data_root, match_id)
    lineups = load_lineups(data_root, match_id)
    expanded = build_expanded_dfs(events, threesixty, lineups)

    ds = PassDataset(
        expanded.expanded_df,
        only_passes=True,
        team_filter=team_filter,
        compute_velocities=compute_velocities,
        context_dim=context_dim,
    )
    return ds, expanded


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
            cfg=cfg,
        ).to(device)
    else:
        model = SoccerMap(cfg).to(device)

    model.load_state_dict(state)
    model.eval()
    return model, uses_player_embed, player_id_mapping


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
):
    batch = [ds[i] for i in idxs]

    channels = torch.stack([b.channels for b in batch]).to(device)
    context = torch.stack([b.context_features for b in batch]).to(device)
    dest = torch.tensor([b.dest_index for b in batch], dtype=torch.long, device=device)

    completed = torch.tensor(
        [float(b.completed) if b.completed is not None else 0.0 for b in batch],
        dtype=torch.float32,
        device=device,
    )

    actor_ids = None
    if uses_player_embed:
        actor_ids = torch.tensor(
            [player_id_mapping.get(b.actor_player_name, 0) for b in batch],
            dtype=torch.long,
            device=device,
        )

    return channels, context, dest, completed, actor_ids


def forward_logits(
        model: torch.nn.Module,
        channels: torch.Tensor,
        context: torch.Tensor,
        actor_ids: Optional[torch.Tensor],
        uses_player_embed: bool,
) -> torch.Tensor:
    if uses_player_embed:
        return model(channels, actor_ids, context)
    return model(channels)


def compute_loss_for_batch(
        task: str,
        loss_mode: str,
        logits: torch.Tensor,
        dest: torch.Tensor,
        completed: torch.Tensor,
        channels: torch.Tensor,
) -> torch.Tensor:
    if task == "pass_success":
        return pass_success_loss(logits, dest, completed)

    if loss_mode == "ce":
        return pass_selection_loss(logits, dest)
    if loss_mode == "gaussian_kl":
        return pass_selection_kl_loss(logits, dest)
    if loss_mode == "teammate_kl":
        teammate_channel = channels[:, 0, :, :]
        return pass_selection_teammate_kl_loss(logits, dest, teammate_channel)

    raise ValueError(f"Unsupported loss_mode: {loss_mode}")


@torch.no_grad()
def evaluate_baseline(
        model: torch.nn.Module,
        ds: PassDataset,
        *,
        task: str,
        loss_mode: str,
        device: str,
        batch_size: int,
        max_samples: int,
        uses_player_embed: bool,
        player_id_mapping: Dict[str, int],
) -> float:
    n = min(len(ds), max_samples)
    indices = list(range(n))

    total = 0.0
    count = 0

    for start, end in tqdm(list(batch_indices(n, batch_size)), desc=f"baseline-{loss_mode}"):
        idxs = indices[start:end]

        channels, context, dest, completed, actor_ids = gather_batch(
            ds,
            idxs,
            device=device,
            uses_player_embed=uses_player_embed,
            player_id_mapping=player_id_mapping,
        )

        logits = forward_logits(model, channels, context, actor_ids, uses_player_embed)
        loss = compute_loss_for_batch(task, loss_mode, logits, dest, completed, channels)

        bs = channels.shape[0]
        total += float(loss.item()) * bs
        count += bs

    return total / max(count, 1)


@torch.no_grad()
def evaluate_channel_importance(
        model: torch.nn.Module,
        ds: PassDataset,
        *,
        task: str,
        loss_mode: str,
        device: str,
        batch_size: int,
        max_samples: int,
        uses_player_embed: bool,
        player_id_mapping: Dict[str, int],
        baseline: float,
) -> np.ndarray:
    n = min(len(ds), max_samples)
    indices = list(range(n))

    deltas = np.zeros(len(CHANNEL_NAMES), dtype=np.float64)

    for ch in tqdm(range(len(CHANNEL_NAMES)), desc=f"channel-ablation-{loss_mode}"):
        total = 0.0
        count = 0

        for start, end in batch_indices(n, batch_size):
            idxs = indices[start:end]

            channels, context, dest, completed, actor_ids = gather_batch(
                ds,
                idxs,
                device=device,
                uses_player_embed=uses_player_embed,
                player_id_mapping=player_id_mapping,
            )

            channels[:, ch, :, :] = 0.0

            logits = forward_logits(model, channels, context, actor_ids, uses_player_embed)
            loss = compute_loss_for_batch(task, loss_mode, logits, dest, completed, channels)

            bs = channels.shape[0]
            total += float(loss.item()) * bs
            count += bs

        ablated = total / max(count, 1)
        deltas[ch] = ablated - baseline

    return deltas


@torch.no_grad()
def compute_channel_correlation(
        ds: PassDataset,
        *,
        max_samples: int,
) -> np.ndarray:
    n = min(len(ds), max_samples)
    flat_by_channel = [[] for _ in range(len(CHANNEL_NAMES))]
    rng = np.random.default_rng(0)

    for i in tqdm(range(n), desc="channel-correlation"):
        sample = ds[i]
        arr = sample.channels.detach().cpu().numpy()
        for ch in range(len(CHANNEL_NAMES)):
            flat_by_channel[ch].append(arr[ch].reshape(-1))

    stacked = []
    for ch in range(len(CHANNEL_NAMES)):
        x = np.concatenate(flat_by_channel[ch]).astype(np.float64)
        if np.std(x) < 1e-12:
            x = x + 1e-12 * rng.standard_normal(size=x.shape)
        stacked.append(x)

    X = np.vstack(stacked)
    corr = np.corrcoef(X)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def plot_channel_importance(
        deltas: np.ndarray,
        out_path: Path,
        *,
        baseline_loss: float,
        task: str,
        loss_mode: str,
) -> None:
    order = np.argsort(deltas)
    vals = deltas[order]
    names = [CHANNEL_NAMES[i] for i in order]

    fig, ax = plt.subplots(figsize=(11.0, 7.2))

    colors = ["#2C7FB8" if v >= 0 else "#D95F5F" for v in vals]
    y = np.arange(len(vals))

    ax.barh(
        y,
        vals,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        height=0.72,
    )

    left_limit = min(-0.02, float(vals.min()) * 1.15)
    right_limit = max(0.05, float(vals.max()) * 1.10)
    ax.set_xlim(left_limit, right_limit)

    ax.axvline(0.0, color="#4A4A4A", linewidth=1.0)

    for i, v in enumerate(vals):
        if abs(v) < 0.005:
            x_text = 0.005
            ha = "left"
        else:
            offset = 0.01 if v >= 0 else -0.01
            x_text = v + offset
            ha = "left" if v >= 0 else "right"

        ax.text(
            x_text,
            i,
            f"{v:.4f}",
            va="center",
            ha=ha,
            fontsize=9,
            color="#222222",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Loss increase after zeroing one channel", fontsize=11)
    ax.set_title(
        f"Channel importance ({task}, {loss_mode})\nBaseline loss = {baseline_loss:.4f}",
        fontsize=15,
        pad=14,
    )

    ax.grid(axis="x", linestyle="--", alpha=0.30)
    ax.grid(axis="y", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_channel_correlation(
        corr: np.ndarray,
        out_path: Path,
        *,
        task: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(CHANNEL_NAMES)))
    ax.set_yticks(np.arange(len(CHANNEL_NAMES)))
    ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CHANNEL_NAMES, fontsize=9)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(
                j,
                i,
                f"{corr[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    ax.set_title(f"Inter-channel correlation ({task})", fontsize=13, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_channel_importance_csv(deltas: np.ndarray, out_path: Path) -> None:
    order = np.argsort(deltas)[::-1]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("channel_idx,channel_name,delta_loss\n")
        for i in order:
            f.write(f"{i},{CHANNEL_NAMES[i]},{deltas[i]:.12f}\n")


def save_channel_correlation_csv(corr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        header = "channel_name," + ",".join(CHANNEL_NAMES)
        f.write(header + "\n")
        for i, name in enumerate(CHANNEL_NAMES):
            row = [name] + [f"{corr[i, j]:.12f}" for j in range(len(CHANNEL_NAMES))]
            f.write(",".join(row) + "\n")


def print_channel_examples(ds: PassDataset, num_examples: int = 3) -> None:
    print("\nexample channel tensor shapes:")
    for i in range(min(num_examples, len(ds))):
        sample = ds[i]
        print(f"{i:02d}: {tuple(sample.channels.shape)}")


def run_one_loss_mode(
        *,
        loss_mode: str,
        args: argparse.Namespace,
        model: torch.nn.Module,
        ds: PassDataset,
        uses_player_embed: bool,
        player_id_mapping: Dict[str, int],
        base_out_dir: Path,
) -> Tuple[float, np.ndarray]:
    print("\n" + "=" * 80)
    print(f"running loss mode: {loss_mode}")
    print("=" * 80)

    baseline = evaluate_baseline(
        model,
        ds,
        task=args.task,
        loss_mode=loss_mode,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        uses_player_embed=uses_player_embed,
        player_id_mapping=player_id_mapping,
    )

    deltas = evaluate_channel_importance(
        model,
        ds,
        task=args.task,
        loss_mode=loss_mode,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        uses_player_embed=uses_player_embed,
        player_id_mapping=player_id_mapping,
        baseline=baseline,
    )

    loss_out_dir = base_out_dir / loss_mode
    loss_out_dir.mkdir(parents=True, exist_ok=True)

    plot_channel_importance(
        deltas,
        loss_out_dir / f"{args.task}_{loss_mode}_channel_importance.png",
        baseline_loss=baseline,
        task=args.task,
        loss_mode=loss_mode,
        )

    save_channel_importance_csv(
        deltas,
        loss_out_dir / f"{args.task}_{loss_mode}_channel_importance.csv",
        )

    print("\nbaseline loss:")
    print(f"{baseline:.12f}")

    print("\nper-channel importance:")
    for i, v in enumerate(deltas):
        print(f"{i:02d}  {CHANNEL_NAMES[i]:<18}  {v:.12f}")

    return baseline, deltas


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root not found: {data_root}\n"
            f"Please run this script from the project root, e.g.\n"
            f"python code/{Path(__file__).name}"
        )

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Please run this script from the project root, e.g.\n"
            f"python code/{Path(__file__).name}"
        )

    events_file = data_root / "events" / f"{args.match_id}.json"
    threesixty_file = data_root / "three-sixty" / f"{args.match_id}.json"
    lineups_file = data_root / "lineups" / f"{args.match_id}.json"

    if not events_file.exists():
        raise FileNotFoundError(f"Missing events file: {events_file}")
    if not threesixty_file.exists():
        raise FileNotFoundError(f"Missing three-sixty file: {threesixty_file}")
    if not lineups_file.exists():
        raise FileNotFoundError(f"Missing lineups file: {lineups_file}")

    ckpt = torch.load(str(ckpt_path), map_location=args.device)
    context_dim = ckpt.get("context_dim", DEFAULT_CONTEXT_DIM)

    ds, _ = build_dataset(
        str(data_root),
        args.match_id,
        compute_velocities=args.compute_velocities,
        team_filter=args.team_filter or None,
        context_dim=context_dim,
    )

    if len(ds) == 0:
        raise RuntimeError("No pass samples found for the requested match / team filter.")

    model, uses_player_embed, player_id_mapping = build_model_from_ckpt(ckpt, args.device)

    print("uses_player_embed:", uses_player_embed)
    print("context_dim:", context_dim)
    print("context_feature_names from ckpt:", ckpt.get("context_feature_names", None))
    print("dataset size:", len(ds))
    print("num samples used:", min(len(ds), args.max_samples))

    print_channel_examples(ds, num_examples=3)

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "pass_success":
        active_loss_modes = ["ce"]
    else:
        active_loss_modes = args.loss_modes

    summary_rows = []

    for loss_mode in active_loss_modes:
        baseline, _ = run_one_loss_mode(
            loss_mode=loss_mode,
            args=args,
            model=model,
            ds=ds,
            uses_player_embed=uses_player_embed,
            player_id_mapping=player_id_mapping,
            base_out_dir=out_dir,
        )
        summary_rows.append((loss_mode, baseline))

    corr = compute_channel_correlation(
        ds,
        max_samples=args.max_samples,
    )

    shared_dir = out_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    plot_channel_correlation(
        corr,
        shared_dir / f"{args.task}_channel_correlation.png",
        task=args.task,
        )
    save_channel_correlation_csv(
        corr,
        shared_dir / f"{args.task}_channel_correlation.csv",
        )

    summary_path = out_dir / f"{args.task}_baseline_summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("loss_mode,baseline_loss\n")
        for loss_mode, baseline in summary_rows:
            f.write(f"{loss_mode},{baseline:.12f}\n")

    print("\nshared correlation files saved to:")
    print(shared_dir / f"{args.task}_channel_correlation.png")
    print(shared_dir / f"{args.task}_channel_correlation.csv")

    print("\nbaseline summary file saved to:")
    print(summary_path)


if __name__ == "__main__":
    main()
