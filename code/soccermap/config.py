from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """
    Grid spec for the model input/output.

    We store spatial tensors as (L, W) where:
      - L corresponds to StatsBomb x (pitch length, 0..120)
      - W corresponds to StatsBomb y (pitch width,  0..80)

    So a grid cell index is (l_idx, w_idx) and lives in:
      l_idx in [0, L-1], w_idx in [0, W-1]
    """
    L: int = 120  # grid length
    # The SoccerMap paper uses 104x68. We keep StatsBomb coords (120x80)
    # and discretize to (L,W) = (104,68) by default.
    W: int = 80   # grid width
    sb_L: float = 120.0
    sb_W: float = 80.0

    @property
    def scale_L(self) -> float:
        return self.L / self.sb_L

    @property
    def scale_W(self) -> float:
        return self.W / self.sb_W

    def sb_to_grid(self, x: float, y: float) -> Tuple[int, int, float, float]:
        """
        Convert StatsBomb (x,y) to grid indices (l_idx,w_idx) and floats (l,w).
        With L=120, W=80 this becomes 1:1 in units.
        """
        # scaled continuous coords (still keep floats for dense channels)
        l = x * self.scale_L
        w = y * self.scale_W

        # indices for sparse channels / destination bins
        l_idx = int(np.clip(np.floor(l), 0, self.L - 1))
        w_idx = int(np.clip(np.floor(w), 0, self.W - 1))
        return l_idx, w_idx, float(l), float(w)


    def grid_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (gx, gy) each shape (L, W) where:
          gx[l,w] = l, gy[l,w] = w
        """
        gx = np.arange(self.L, dtype=np.float32)[:, None]
        gy = np.arange(self.W, dtype=np.float32)[None, :]
        gx = np.repeat(gx, self.W, axis=1)
        gy = np.repeat(gy, self.L, axis=0)
        return gx, gy

    def goal_location(self) -> Tuple[float, float]:
        """
        Right-side goal center in grid coordinates.
        We place it at (L, W/2) to mimic "just beyond the endline".
        """
        return float(self.L), float(self.W) / 2.0
