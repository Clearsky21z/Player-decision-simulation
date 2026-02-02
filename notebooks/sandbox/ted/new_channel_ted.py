import numpy as np
import pandas as pd

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


### Decomposition of pass vectors

def _safe_xy(loc):
    """Return (x,y) floats from a StatsBomb location-like field, else (nan,nan)."""
    if loc is None or not isinstance(loc, (list, tuple)) or len(loc) < 2:
        return (np.nan, np.nan)
    try:
        return (float(loc[0]), float(loc[1]))
    except (TypeError, ValueError):
        return (np.nan, np.nan)

def decompose_pass_vector(
    start_loc,
    end_loc,
    *,
    attacking_left_to_right=True,
    return_angle=True,
    eps=1e-9
):
    """
    Decompose a pass vector into (dx, dy), length, direction (cos,sin), and forward/lateral components.

    Parameters
    ----------
    start_loc, end_loc : list/tuple like [x, y] (StatsBomb event 'location' and pass 'end_location')
    attacking_left_to_right : bool
        If False, we flip dx so "forward" is still positive toward the team's attacking direction.
    return_angle : bool
        If True, also return theta = atan2(dy, dx) in radians.
    eps : float
        Numerical stability for normalization.

    Returns
    -------
    dict with keys:
        dx, dy, length, cos, sin, forward, lateral, (optional) theta
    """
    sx, sy = _safe_xy(start_loc)
    ex, ey = _safe_xy(end_loc)

    dx = ex - sx
    dy = ey - sy

    forward = dx if attacking_left_to_right else -dx
    lateral = dy  # keep sign as pitch y-axis; if you want "toward right touchline", ensure consistent convention.

    length = float(np.sqrt(dx * dx + dy * dy)) if np.isfinite(dx) and np.isfinite(dy) else np.nan

    if not np.isfinite(length) or length < eps:
        cos_t, sin_t = (0.0, 0.0)
        theta = 0.0
    else:
        cos_t = float(dx / length)
        sin_t = float(dy / length)
        theta = float(np.arctan2(dy, dx))

    out = {
        "dx": float(dx) if np.isfinite(dx) else np.nan,
        "dy": float(dy) if np.isfinite(dy) else np.nan,
        "length": length,
        "cos": cos_t,
        "sin": sin_t,
        "forward": float(forward) if np.isfinite(forward) else np.nan,
        "lateral": float(lateral) if np.isfinite(lateral) else np.nan,
    }
    if return_angle:
        out["theta"] = theta
    return out

def add_pass_vector_features(
    df: pd.DataFrame,
    *,
    start_col="location",
    end_col="end_location",
    attacking_left_to_right_col=None,
    default_attacking_left_to_right=True,
    prefix="passvec_",
    drop_if_missing_end=True
) -> pd.DataFrame:
    """
    Add pass vector decomposition columns to a StatsBomb-like events DataFrame.

    If you have per-event attacking direction, pass its column name via attacking_left_to_right_col.
    Otherwise uses default_attacking_left_to_right for all rows.

    Adds columns:
      {prefix}dx, {prefix}dy, {prefix}length, {prefix}cos, {prefix}sin,
      {prefix}forward, {prefix}lateral, {prefix}theta
    """
    out = df.copy()

    if drop_if_missing_end:
        mask = out[end_col].apply(lambda x: isinstance(x, (list, tuple)) and len(x) >= 2)
        out = out[mask].copy()

    def _row_feats(row):
        attack_dir = (
            bool(row[attacking_left_to_right_col])
            if attacking_left_to_right_col is not None and attacking_left_to_right_col in row
            else default_attacking_left_to_right
        )
        return decompose_pass_vector(
            row.get(start_col),
            row.get(end_col),
            attacking_left_to_right=attack_dir,
            return_angle=True,
        )

    feats = out.apply(_row_feats, axis=1, result_type="expand")
    feats = feats.add_prefix(prefix)
    out = pd.concat([out, feats], axis=1)

    return out