from __future__ import annotations

from dataclasses import dataclass
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

# TODO
# we need to create a new loss function for pass selection model
# 1. We have to make the true pass a guassian field N(0,2)
# 2. then our loss function will be minimizing KL divergence between the predicted distribution and the true distribution

@torch.no_grad()
def pass_success_surface(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits).squeeze(1)  # (N,H,W) in [0,1]


@torch.no_grad()
def pass_selection_surface(logits: torch.Tensor) -> torch.Tensor:
    N, _, H, W = logits.shape
    flat = logits.view(N, -1)
    return torch.softmax(flat, dim=1).view(N, H, W)  # sums to 1
