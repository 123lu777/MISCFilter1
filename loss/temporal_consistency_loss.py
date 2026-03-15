"""
loss/temporal_consistency_loss.py
----------------------------------
Temporal consistency loss for video deblurring.

Penalises large differences between consecutive deblurred frames beyond what
the optical-flow field predicts.  Two complementary terms are provided:

  TemporalConsistencyLoss
      Computes an L1/Charbonnier distance between warped consecutive frames
      using a provided optical-flow field.  If no flow is given, falls back
      to a plain L1 difference (requires zero-motion assumption).

  FlowWarpLoss
      Encourages the deblurred output to be consistent with the optical flow
      by minimising the photometric error after flow-based warping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flow_warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp tensor *x* using 2-D optical flow *flow*.

    Args:
        x:    (B, C, H, W) feature map to warp.
        flow: (B, 2, H, W) optical flow in pixels.  flow[:, 0] = dx (x-axis),
              flow[:, 1] = dy (y-axis).

    Returns:
        Warped tensor of the same shape as *x*.
    """
    B, C, H, W = x.shape
    device = x.device

    # Normalised grid  [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    # (H, W) → (1, H, W)
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)

    # flow: (B, 2, H, W)  → (B, H, W)
    flow_x = flow[:, 0, :, :]   # displacement along x (cols)
    flow_y = flow[:, 1, :, :]   # displacement along y (rows)

    # Add displacement and normalise to [-1, 1]
    new_x = ((grid_x + flow_x) / (W - 1.0)) * 2.0 - 1.0
    new_y = ((grid_y + flow_y) / (H - 1.0)) * 2.0 - 1.0

    # grid_sample expects (B, H, W, 2) with (x, y)
    grid = torch.stack([new_x, new_y], dim=-1)  # (B, H, W, 2)

    warped = F.grid_sample(x, grid,
                           mode='bilinear',
                           padding_mode='border',
                           align_corners=True)
    return warped


# ---------------------------------------------------------------------------
# Temporal consistency loss
# ---------------------------------------------------------------------------

class TemporalConsistencyLoss(nn.Module):
    """Penalise temporal inconsistency between consecutive deblurred frames.

    Given two consecutive model outputs ``out_t`` and ``out_{t+1}`` and the
    optical flow ``flow_{t→t+1}``, the loss minimises the Charbonnier distance
    between the warped ``out_t`` and ``out_{t+1}``.

    When *flow* is not provided, the loss reduces to a plain L1 difference
    (assumes stationary scene between consecutive frames).

    Args:
        eps:    Charbonnier smoothing term.
        weight: Scalar weight for the loss value.
    """

    def __init__(self, eps: float = 1e-3, weight: float = 1.0):
        super().__init__()
        self.eps    = eps
        self.weight = weight

    def forward(self,
                out_t: torch.Tensor,
                out_t1: torch.Tensor,
                flow: torch.Tensor = None
                ) -> torch.Tensor:
        """Compute temporal consistency loss.

        Args:
            out_t:  (B, C, H, W) deblurred frame at time t.
            out_t1: (B, C, H, W) deblurred frame at time t+1.
            flow:   (B, 2, H, W) optical flow from t to t+1 in pixels.
                    If None, a plain L1 difference is used.

        Returns:
            Scalar loss tensor.
        """
        if flow is not None:
            warped = _flow_warp(out_t, flow)
        else:
            warped = out_t

        diff = warped - out_t1
        loss = torch.mean(torch.sqrt(diff * diff + self.eps ** 2))
        return self.weight * loss


# ---------------------------------------------------------------------------
# Flow warp photometric loss
# ---------------------------------------------------------------------------

class FlowWarpLoss(nn.Module):
    """Encourage output photometric consistency across frames via flow warping.

    Computes the Charbonnier distance between *frame_t* warped by *flow* and
    *frame_{t+1}*.  Useful as an auxiliary self-supervised loss.

    Args:
        eps:    Charbonnier smoothing term.
        weight: Scalar weight.
    """

    def __init__(self, eps: float = 1e-3, weight: float = 0.1):
        super().__init__()
        self.eps    = eps
        self.weight = weight

    def forward(self,
                frame_t: torch.Tensor,
                frame_t1: torch.Tensor,
                flow: torch.Tensor
                ) -> torch.Tensor:
        """Compute flow warp photometric loss.

        Args:
            frame_t:  (B, C, H, W) frame at time t (deblurred or input).
            frame_t1: (B, C, H, W) frame at time t+1.
            flow:     (B, 2, H, W) optical flow from t to t+1.

        Returns:
            Scalar loss tensor.
        """
        warped = _flow_warp(frame_t, flow)
        diff   = warped - frame_t1
        loss   = torch.mean(torch.sqrt(diff * diff + self.eps ** 2))
        return self.weight * loss
