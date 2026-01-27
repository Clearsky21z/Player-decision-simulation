from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mplsoccer import Pitch


# -----------------------------
# Helpers
# -----------------------------
def _safe_xy(xy) -> Optional[Tuple[float, float]]:
    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
        try:
            return float(xy[0]), float(xy[1])
        except Exception:
            return None
    return None


def _make_pitch() -> Pitch:
    # StatsBomb coordinate system: x in [0,120], y in [0,80]
    return Pitch(pitch_type="statsbomb", line_color="black", linewidth=1.2)


def _to_img_yx(prob_LW: np.ndarray) -> np.ndarray:
    """
    We store surfaces as (L, W) like the paper: length_bins x width_bins.
    Matplotlib expects (rows, cols) = (y, x).
    StatsBomb pitch uses x horizontal, y vertical -> we need (W, L).
    """
    arr = np.asarray(prob_LW, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"prob surface must be 2D, got shape={arr.shape}")
    return arr.T  # (W, L) == (y, x)


def _extract_scene(expanded_df: pd.DataFrame, event_id: str):
    ev = expanded_df.loc[expanded_df["event_id"] == event_id]
    if ev.empty:
        raise ValueError(f"event_id {event_id} not found in expanded_df")

    actor = ev.loc[ev["actor"] == True]
    if actor.empty:
        raise ValueError(f"event_id {event_id} has no actor row")
    actor_row = actor.iloc[0]

    passer_xy = _safe_xy(actor_row.get("event_location"))
    dest_xy = _safe_xy(actor_row.get("end_location"))
    completed = actor_row.get("pass_completed")

    players = ev.loc[ev["actor"] == False].copy()
    atk = players.loc[players["teammate"] == True]
    dfn = players.loc[players["teammate"] == False]

    atk_xy = np.array([_safe_xy(xy) for xy in atk["ff_location"].tolist() if _safe_xy(xy) is not None], dtype=float)
    dfn_xy = np.array([_safe_xy(xy) for xy in dfn["ff_location"].tolist() if _safe_xy(xy) is not None], dtype=float)

    return passer_xy, dest_xy, completed, atk_xy, dfn_xy


def _save_or_show(fig: plt.Figure, out_path: Optional[str], show: bool) -> None:
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------
# Main plotting funcs
# -----------------------------
def plot_pass_selection_surface(
        prob_LW: np.ndarray,
        expanded_df: pd.DataFrame,
        event_id: str,
        *,
        title: str = "",
        out_path: Optional[str] = None,
        show: bool = False,
        # scaling / color
        scale: str = "log",     # "log" | "percentile" | "linear"
        q: float = 0.995,       # percentile for vmax (log/percentile)
        eps: float = 1e-12,     # avoids log(0)
        cmap: str = "RdBu_r",   # red-blue tone
        alpha: float = 0.92,
        interpolation: str = "nearest",
) -> plt.Figure:
    """
    PASS SELECTION surface (softmax over all pixels).
    These probabilities are tiny across most cells, so "log" is usually best.
    """
    passer_xy, dest_xy, completed, atk_xy, dfn_xy = _extract_scene(expanded_df, event_id)

    pitch = _make_pitch()
    fig, ax = pitch.draw(figsize=(11, 7))

    img = _to_img_yx(prob_LW)

    if scale == "log":
        img_plot = np.clip(img, eps, None)
        vmax = float(np.quantile(img_plot, q))
        vmax = max(vmax, eps * 10)
        norm = mcolors.LogNorm(vmin=eps, vmax=vmax)
        im = ax.imshow(
            img_plot,
            extent=pitch.extent,
            origin="upper",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            alpha=alpha,
            interpolation=interpolation,
        )
    elif scale == "percentile":
        vmax = float(np.quantile(img, q))
        vmax = max(vmax, 1e-12)
        im = ax.imshow(
            img,
            extent=pitch.extent,
            origin="upper",
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            alpha=alpha,
            interpolation=interpolation,
        )
    elif scale == "linear":
        im = ax.imshow(
            img,
            extent=pitch.extent,
            origin="upper",
            aspect="auto",
            cmap=cmap,
            alpha=alpha,
            interpolation=interpolation,
        )
    else:
        raise ValueError("scale must be one of: 'log', 'percentile', 'linear'")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=9)

    # players
    if atk_xy.size:
        pitch.scatter(atk_xy[:, 0], atk_xy[:, 1], s=70, ax=ax, label="Attacking")
    if dfn_xy.size:
        pitch.scatter(dfn_xy[:, 0], dfn_xy[:, 1], s=70, ax=ax, label="Defending")

    # passer/dest + arrow
    if passer_xy is not None:
        pitch.scatter([passer_xy[0]], [passer_xy[1]], s=120, ax=ax, label="Passer")
    if dest_xy is not None:
        pitch.scatter([dest_xy[0]], [dest_xy[1]], s=140, marker="x", ax=ax, label="True destination")
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

    if not title:
        title = f"event_id={event_id} | completed={completed}"
    ax.set_title(title)
    ax.legend(loc="upper left")

    _save_or_show(fig, out_path, show)
    return fig


def plot_pass_success_surface(
        prob_LW: np.ndarray,          # (L,W) success probs in [0,1]
        expanded_df: pd.DataFrame,
        event_id: str,
        *,
        title: str = "",
        out_path: Optional[str] = None,
        show: bool = False,
        cmap: str = "RdBu_r",
        alpha: float = 0.92,
        vmin: float = 0.0,
        vmax: float = 1.0,
        interpolation: str = "nearest",
) -> plt.Figure:
    """
    PASS SUCCESS surface (sigmoid per pixel).
    """
    passer_xy, dest_xy, completed, atk_xy, dfn_xy = _extract_scene(expanded_df, event_id)

    pitch = _make_pitch()
    fig, ax = pitch.draw(figsize=(11, 7))

    img = _to_img_yx(prob_LW)

    im = ax.imshow(
        img,
        extent=pitch.extent,
        origin="upper",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        interpolation=interpolation,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=9)

    if atk_xy.size:
        pitch.scatter(atk_xy[:, 0], atk_xy[:, 1], s=70, ax=ax, label="Attacking")
    if dfn_xy.size:
        pitch.scatter(dfn_xy[:, 0], dfn_xy[:, 1], s=70, ax=ax, label="Defending")

    if passer_xy is not None:
        pitch.scatter([passer_xy[0]], [passer_xy[1]], s=120, ax=ax, label="Passer")
    if dest_xy is not None:
        pitch.scatter([dest_xy[0]], [dest_xy[1]], s=140, marker="x", ax=ax, label="True destination")
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

    if not title:
        title = f"event_id={event_id} | completed={completed}"
    ax.set_title(title)
    ax.legend(loc="upper left")

    _save_or_show(fig, out_path, show)
    return fig
