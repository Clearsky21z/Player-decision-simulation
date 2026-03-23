from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------- Config --------

@dataclass(frozen=True)
class SoccerMapConfig:
    """
    Matches the paper's described components.
    """
    in_channels: int = 14

    feat_channels: int = 32

    pred_channels: int = 32
    up_channels: int = 32

    pad_mode: str = "reflect"


# -------- Low-level helpers --------

class SymmetricPadConv2d(nn.Module):
    """
    "Symmetric padding" in practice:
      - ReflectionPad2d or ReplicationPad2d
      - Conv2d with padding=0
    """
    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int = 1, mode: str = "reflect"):
        super().__init__()
        pad = k // 2
        if pad > 0:
            if mode == "reflect":
                self.pad = nn.ReflectionPad2d(pad)
            elif mode == "replicate":
                self.pad = nn.ReplicationPad2d(pad)
            else:
                raise ValueError(f"Unknown pad_mode={mode}, use 'reflect' or 'replicate'.")
        else:
            self.pad = nn.Identity()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


# -------- Building blocks from the paper --------

class Conv5x5FeatBlock(nn.Module):
    """
    Two 5x5 conv layers (stride 1) each followed by ReLU, with symmetric padding.
    """
    def __init__(self, in_ch: int, out_ch: int, pad_mode: str):
        super().__init__()
        self.c1 = SymmetricPadConv2d(in_ch, out_ch, k=5, stride=1, mode=pad_mode)
        self.c2 = SymmetricPadConv2d(out_ch, out_ch, k=5, stride=1, mode=pad_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        return x


class PredictionHead(nn.Module):
    """
    Prediction layers:
      1x1 conv(32) + ReLU + 1x1 conv(1), linear output (logits)
    """
    def __init__(self, in_ch: int, pred_ch: int = 32):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, pred_ch, kernel_size=1, stride=1, padding=0)
        self.c2 = nn.Conv2d(pred_ch, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.c1(x))
        return self.c2(x)  # linear logits


class Upsample2xBlock(nn.Module):
    """
    Upsampling:
      2x nearest neighbor upsample
      3x3 conv(32) + ReLU
      3x3 conv(1) linear

    Uses symmetric padding for the 3x3 convs as well.
    """
    def __init__(self, up_ch: int = 32, pad_mode: str = "reflect"):
        super().__init__()
        self.c1 = SymmetricPadConv2d(1, up_ch, k=3, stride=1, mode=pad_mode)
        self.c2 = SymmetricPadConv2d(up_ch, 1, k=3, stride=1, mode=pad_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.c1(x))
        return self.c2(x)  # linear logits


class FusePair(nn.Module):
    """
    Fusion:
      concat two maps -> 1x1 conv(1), linear
    """
    def __init__(self):
        super().__init__()
        self.fuse = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([a, b], dim=1))


# -------- SoccerMap --------

class SoccerMap(nn.Module):
    """
    Paper structure:

      1x:   feat -> pred p1
      1/2x: pool -> feat -> pred p2
      1/4x: pool -> feat -> pred p3

      p3 -> upsample(2x) -> fuse with p2 -> p23 (1/2x)
      p23 -> upsample(2x) -> fuse with p1 -> p123 (1x)

    Returns logits at 1x: (N,1,H,W)
    """
    def __init__(self, cfg: SoccerMapConfig = SoccerMapConfig()):
        super().__init__()
        self.cfg = cfg

        self.feat1 = Conv5x5FeatBlock(cfg.in_channels, cfg.feat_channels, cfg.pad_mode)
        self.pool1 = nn.MaxPool2d(2)

        self.feat2 = Conv5x5FeatBlock(cfg.feat_channels, cfg.feat_channels, cfg.pad_mode)
        self.pool2 = nn.MaxPool2d(2)

        self.feat3 = Conv5x5FeatBlock(cfg.feat_channels, cfg.feat_channels, cfg.pad_mode)

        self.pred1 = PredictionHead(cfg.feat_channels, cfg.pred_channels)
        self.pred2 = PredictionHead(cfg.feat_channels, cfg.pred_channels)
        self.pred3 = PredictionHead(cfg.feat_channels, cfg.pred_channels)

        self.up3_to_2 = Upsample2xBlock(cfg.up_channels, cfg.pad_mode)
        self.fuse23 = FusePair()

        self.up23_to_1 = Upsample2xBlock(cfg.up_channels, cfg.pad_mode)
        self.fuse123 = FusePair()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.feat1(x)
        p1 = self.pred1(h1)

        h2 = self.feat2(self.pool1(h1))
        p2 = self.pred2(h2)

        h3 = self.feat3(self.pool2(h2))
        p3 = self.pred3(h3)

        p3_up = self.up3_to_2(p3)
        p23 = self.fuse23(p2, p3_up)

        p23_up = self.up23_to_1(p23)
        p123 = self.fuse123(p1, p23_up)

        return p123  # logits



class SoccerMapWithPlayerEmbed(nn.Module):
    """
    SoccerMap with FiLM-conditioned spatial features and late-fused game context.
    """

    def __init__(
        self,
        num_players: int,
        embed_dim: int = 8,
        context_dim: int = 0,
        context_hidden_dim: int = 16,
        context_embed_dim: int = 8,
        cfg: SoccerMapConfig = SoccerMapConfig(),
    ):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.context_embed_dim = context_embed_dim if context_dim > 0 else 0

        # Player embedding table (index 0 = unknown / padding)
        self.player_embedding = nn.Embedding(
            num_embeddings=num_players + 1,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        self.context_ffn: Optional[nn.Sequential]
        if self.context_dim > 0:
            self.context_ffn = nn.Sequential(
                nn.Linear(context_dim, context_hidden_dim),
                nn.ReLU(),
                nn.Linear(context_hidden_dim, self.context_embed_dim),
                nn.ReLU(),
            )
        else:
            self.context_ffn = None

        # --- Backbone ---
        # Condition the model early by concatenating the player embedding to
        # the input channels before feature extraction begins.
        self.feat1 = Conv5x5FeatBlock(cfg.in_channels + embed_dim, cfg.feat_channels, cfg.pad_mode)
        self.pool1 = nn.MaxPool2d(2)

        self.feat2 = Conv5x5FeatBlock(cfg.feat_channels, cfg.feat_channels, cfg.pad_mode)
        self.pool2 = nn.MaxPool2d(2)

        self.feat3 = Conv5x5FeatBlock(cfg.feat_channels, cfg.feat_channels, cfg.pad_mode)

        self.pred1 = PredictionHead(cfg.feat_channels, cfg.pred_channels)
        self.pred2 = PredictionHead(cfg.feat_channels, cfg.pred_channels)
        self.pred3 = PredictionHead(cfg.feat_channels, cfg.pred_channels)

        self.up3_to_2 = Upsample2xBlock(cfg.up_channels, cfg.pad_mode)
        self.fuse23 = FusePair()

        self.up23_to_1 = Upsample2xBlock(cfg.up_channels, cfg.pad_mode)
        self.fuse123 = FusePair()

        # --- Late fusion head for identity + compact game context ---
        self.embed_head = nn.Conv2d(1 + embed_dim + self.context_embed_dim, 1, kernel_size=1)
        # FiLM layers modulate intermediate feature maps using the player
        # embedding so different actors can alter the spatial representation.
        self.film1 = nn.Linear(embed_dim, 2 * cfg.feat_channels)
        self.film2 = nn.Linear(embed_dim, 2 * cfg.feat_channels)
        self.film3 = nn.Linear(embed_dim, 2 * cfg.feat_channels)
        nn.init.zeros_(self.film1.weight)
        nn.init.zeros_(self.film1.bias)
        nn.init.zeros_(self.film2.weight)
        nn.init.zeros_(self.film2.bias)
        nn.init.zeros_(self.film3.weight)
        nn.init.zeros_(self.film3.bias)


    def _apply_film(self, h: torch.Tensor, embed: torch.Tensor, film: nn.Linear) -> torch.Tensor:
        gamma_beta = film(embed)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(
        self,
        x: torch.Tensor,
        actor_ids: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 14, H, W) channel tensor
        actor_ids : (N,) long tensor of player embedding indices
        context : (N, context_dim) dense match-context features

        Returns
        -------
        logits : (N, 1, H, W)
        """
        N, _, H, W = x.shape

        embed = self.player_embedding(actor_ids)  # (N, D)
        embed_spatial = embed[:, :, None, None].expand(N, self.embed_dim, H, W)
        x = torch.cat([x, embed_spatial], dim=1)

        h1 = self.feat1(x)
        h1 = self._apply_film(h1, embed, self.film1)
        p1 = self.pred1(h1)

        h2 = self.feat2(self.pool1(h1))
        h2 = self._apply_film(h2, embed, self.film2)
        p2 = self.pred2(h2)

        h3 = self.feat3(self.pool2(h2))
        h3 = self._apply_film(h3, embed, self.film3)
        p3 = self.pred3(h3)

        p3_up = self.up3_to_2(p3)
        p23 = self.fuse23(p2, p3_up)

        p23_up = self.up23_to_1(p23)
        p123 = self.fuse123(p1, p23_up)  # (N, 1, H, W)

        # --- LATE FUSION ---
        # Reuse embed from early fusion; broadcast spatially for concat.
        embed_spatial = embed[:, :, None, None].expand(N, self.embed_dim, H, W)  # (N, D, H, W)
        fusion_parts = [p123, embed_spatial]

        if self.context_ffn is not None:
            if context is None:
                context = torch.zeros(
                    (N, self.context_dim),
                    dtype=x.dtype,
                    device=x.device,
                )
            elif context.shape != (N, self.context_dim):
                raise ValueError(
                    f"context must have shape {(N, self.context_dim)}, got {tuple(context.shape)}"
                )
            context_embed = self.context_ffn(context)
            context_spatial = context_embed[:, :, None, None].expand(N, self.context_embed_dim, H, W)
            fusion_parts.append(context_spatial)

        fused = torch.cat(fusion_parts, dim=1)
        logits = self.embed_head(fused)                        # (N, 1, H, W)
        # --- END LATE FUSION ---

        return logits


# -------- Losses / surfaces --------

def _gather_dest_logits(logits: torch.Tensor, dest_index: torch.Tensor) -> torch.Tensor:
    """
    logits: (N,1,H,W)
    dest_index: (N,) flattened index = y*W + x
    """
    N, _, H, W = logits.shape
    flat = logits.view(N, -1)
    return flat.gather(1, dest_index.view(-1, 1)).squeeze(1)


def pass_success_loss(logits: torch.Tensor, dest_index: torch.Tensor, completed: torch.Tensor) -> torch.Tensor:
    """
    Paper target-location loss:
      log-loss at destination pixel vs outcome y in {0,1}.
    """
    chosen = _gather_dest_logits(logits, dest_index)
    return F.binary_cross_entropy_with_logits(chosen, completed.float())


def pass_selection_loss(logits: torch.Tensor, dest_index: torch.Tensor) -> torch.Tensor:
    """
    """
    N, _, H, W = logits.shape
    flat = logits.view(N, -1)
    return F.cross_entropy(flat, dest_index)


def pass_selection_kl_loss(logits: torch.Tensor, dest_index: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """
    KL-divergence pass-selection loss.

    Target: 2-D Gaussian centred on the true destination cell with std=sigma,
            normalised to a valid probability distribution over the H*W grid.
    Predicted: log-softmax of the model's flattened spatial logits.
    """
    N, _, H, W = logits.shape
    flat_logits = logits.view(N, H * W)
    log_p = F.log_softmax(flat_logits, dim=1)

    # build Gaussian target for each sample
    coords_l = torch.arange(H, device=logits.device).float()
    coords_w = torch.arange(W, device=logits.device).float()
    grid_l, grid_w = torch.meshgrid(coords_l, coords_w, indexing="ij")  # (H,W)

    dest_l = (dest_index // W).float() # (N,)
    dest_w = (dest_index %  W).float() # (N,)

    sigma = 2.0
    diff_l = grid_l.unsqueeze(0) - dest_l.view(N, 1, 1) # (N,H,W)
    diff_w = grid_w.unsqueeze(0) - dest_w.view(N, 1, 1)
    gauss  = torch.exp(-(diff_l**2 + diff_w**2) / (2 * sigma**2))
    q = gauss.view(N, H * W)
    q = q / q.sum(dim=1, keepdim=True) # normalize to valid dist

    return F.kl_div(log_p, q, reduction="batchmean")


def pass_selection_teammate_kl_loss(
    logits: torch.Tensor,
    dest_index: torch.Tensor,
    teammate_channel: torch.Tensor,
    sigma: float = 2.0,
    dest_weight: float = 0.5,
) -> torch.Tensor:
    """
    The target is a
    mixture of Gaussians placed at every teammate location (from channel 0) plus
    the true destination.  The destination component receives ``dest_weight`` of
    the total mass; the remaining ``1 - dest_weight`` is split evenly across
    teammate locations.  This produces multi-modal, non-circular targets shaped
    by the actual game geometry.
    """
    N, _, H, W = logits.shape
    flat_logits = logits.view(N, H * W)
    log_p = F.log_softmax(flat_logits, dim=1)

    coords_l = torch.arange(H, device=logits.device).float()
    coords_w = torch.arange(W, device=logits.device).float()
    grid_l, grid_w = torch.meshgrid(coords_l, coords_w, indexing="ij")  # (H, W)

    # destination Gaussian — same as original KL loss
    dest_l = (dest_index // W).float()
    dest_w = (dest_index %  W).float()
    diff_l = grid_l.unsqueeze(0) - dest_l.view(N, 1, 1)
    diff_w = grid_w.unsqueeze(0) - dest_w.view(N, 1, 1)
    dest_gauss = torch.exp(-(diff_l ** 2 + diff_w ** 2) / (2 * sigma ** 2))  # (N, H, W)

    # teammate Gaussians — one per nonzero cell in teammate_channel
    tm_gauss = torch.zeros(N, H, W, device=logits.device)
    for i in range(N):
        locs = (teammate_channel[i] > 0).nonzero(as_tuple=False)  # (K, 2)
        if locs.shape[0] == 0:
            continue
        tl = locs[:, 0].float()  # (K,)
        tw = locs[:, 1].float()  # (K,)
        dl = grid_l.unsqueeze(0) - tl.view(-1, 1, 1)  # (K, H, W)
        dw = grid_w.unsqueeze(0) - tw.view(-1, 1, 1)
        per_tm = torch.exp(-(dl ** 2 + dw ** 2) / (2 * sigma ** 2))  # (K, H, W)
        tm_gauss[i] = per_tm.mean(dim=0)  # average over teammates

    # normalise each component independently before mixing
    dest_flat = dest_gauss.view(N, H * W)
    dest_flat = dest_flat / dest_flat.sum(dim=1, keepdim=True).clamp(min=1e-8)

    tm_flat = tm_gauss.view(N, H * W)
    tm_sum = tm_flat.sum(dim=1, keepdim=True)
    has_teammates = (tm_sum > 1e-8).float()
    tm_flat = tm_flat / tm_sum.clamp(min=1e-8)

    # mix: if no teammates detected, fall back to dest-only
    q = dest_weight * dest_flat + (1.0 - dest_weight) * has_teammates * tm_flat
    # re-normalise (handles the no-teammate fallback case)
    q = q / q.sum(dim=1, keepdim=True).clamp(min=1e-8)

    return F.kl_div(log_p, q, reduction="batchmean")


@torch.no_grad()
def pass_success_surface(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits).squeeze(1)  # (N,H,W) in [0,1]


@torch.no_grad()
def pass_selection_surface(logits: torch.Tensor) -> torch.Tensor:
    N, _, H, W = logits.shape
    flat = logits.view(N, -1)
    return torch.softmax(flat, dim=1).view(N, H, W)  # sums to 1

