"""
loss/motion_prior_loss.py
--------------------------
Motion-prior and physics-constraint losses for wind turbine blade deblurring.

Two loss modules are provided:

  MotionPriorLoss
      Constrains the predicted blur kernels (or motion maps) to be consistent
      with the optical-flow speed labels extracted from real video data.

  PhysicsConstraintLoss
      Enforces physically plausible rotation parameters (RPM range, exposure
      time range, radially increasing blur magnitude tip→root) using soft
      penalty terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Motion prior loss
# ---------------------------------------------------------------------------

class MotionPriorLoss(nn.Module):
    """Constrain predicted motion/blur maps to match optical-flow-derived labels.

    The loss computes a Charbonnier distance between a predicted per-pixel
    motion magnitude map and a target map derived from optical flow
    (``flow_to_psf_label`` in optical_flow_utils.py).

    Args:
        eps:    Charbonnier smoothing term.
        weight: Scalar weight for the loss value.
    """

    def __init__(self, eps: float = 1e-3, weight: float = 0.1):
        super().__init__()
        self.eps    = eps
        self.weight = weight

    def forward(self,
                pred_motion: torch.Tensor,
                flow_label: torch.Tensor
                ) -> torch.Tensor:
        """Compute motion-prior loss.

        Args:
            pred_motion: (B, 1, H, W) or (B, H, W) predicted motion magnitude
                         map output by the model (values in pixels/frame).
            flow_label:  (B, 1, H, W) or (B, H, W) target motion magnitude
                         derived from optical flow (same units).

        Returns:
            Scalar loss tensor.
        """
        # Ensure 4-D (B, 1, H, W)
        if pred_motion.dim() == 3:
            pred_motion = pred_motion.unsqueeze(1)
        if flow_label.dim() == 3:
            flow_label = flow_label.unsqueeze(1)

        # Resize target to match predicted if spatial dims differ
        if pred_motion.shape[-2:] != flow_label.shape[-2:]:
            flow_label = F.interpolate(flow_label, size=pred_motion.shape[-2:],
                                       mode='bilinear', align_corners=False)

        diff = pred_motion - flow_label.to(pred_motion.device)
        loss = torch.mean(torch.sqrt(diff * diff + self.eps ** 2))
        return self.weight * loss


# ---------------------------------------------------------------------------
# Physics constraint loss
# ---------------------------------------------------------------------------

class PhysicsConstraintLoss(nn.Module):
    """Soft penalty loss enforcing physical plausibility of rotation parameters.

    Three penalty terms are available (each controlled by a weight):

    1. RPM range penalty  – predicted RPM outside [rpm_min, rpm_max].
    2. Exposure time penalty – predicted exposure outside [exp_min, exp_max].
    3. Radial velocity gradient – blade tip region should have higher blur
       than root region (tip_speed > root_speed).

    All penalties use a smooth ReLU (softplus) to be differentiable.

    Args:
        rpm_min:    Minimum plausible RPM.
        rpm_max:    Maximum plausible RPM.
        exp_min:    Minimum plausible exposure time (seconds).
        exp_max:    Maximum plausible exposure time (seconds).
        w_rpm:      Weight for RPM penalty.
        w_exp:      Weight for exposure penalty.
        w_radial:   Weight for radial velocity gradient penalty.
    """

    def __init__(self,
                 rpm_min: float = 5.0,
                 rpm_max: float = 30.0,
                 exp_min: float = 1e-4,
                 exp_max: float = 0.1,
                 w_rpm: float = 0.01,
                 w_exp: float = 0.01,
                 w_radial: float = 0.05):
        super().__init__()
        self.rpm_min  = rpm_min
        self.rpm_max  = rpm_max
        self.exp_min  = exp_min
        self.exp_max  = exp_max
        self.w_rpm    = w_rpm
        self.w_exp    = w_exp
        self.w_radial = w_radial

    @staticmethod
    def _penalty_range(val: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        """Return softplus penalty when *val* is outside [lo, hi]."""
        lo_t = torch.tensor(lo, dtype=val.dtype, device=val.device)
        hi_t = torch.tensor(hi, dtype=val.dtype, device=val.device)
        below = F.softplus(lo_t - val)
        above = F.softplus(val - hi_t)
        return (below + above).mean()

    def forward(self,
                pred_rpm: torch.Tensor = None,
                pred_exposure: torch.Tensor = None,
                motion_map: torch.Tensor = None
                ) -> torch.Tensor:
        """Compute physics constraint loss.

        Args:
            pred_rpm:      (B,) or scalar predicted RPM values.
            pred_exposure: (B,) or scalar predicted exposure time (seconds).
            motion_map:    (B, 1, H, W) predicted motion magnitude map.
                           Used to enforce tip > root blur gradient.

        Returns:
            Scalar loss tensor.
        """
        device = (pred_rpm.device if pred_rpm is not None else
                  pred_exposure.device if pred_exposure is not None else
                  motion_map.device if motion_map is not None else
                  torch.device('cpu'))
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        if pred_rpm is not None and self.w_rpm > 0:
            loss = loss + self.w_rpm * self._penalty_range(
                pred_rpm.float(), self.rpm_min, self.rpm_max)

        if pred_exposure is not None and self.w_exp > 0:
            loss = loss + self.w_exp * self._penalty_range(
                pred_exposure.float(), self.exp_min, self.exp_max)

        if motion_map is not None and self.w_radial > 0:
            # motion_map: (B, 1, H, W)
            if motion_map.dim() == 3:
                motion_map = motion_map.unsqueeze(1)
            H = motion_map.shape[2]
            tip_speed  = motion_map[:, :, :H // 3, :].mean()
            root_speed = motion_map[:, :, 2 * H // 3:, :].mean()
            # Penalise if root_speed >= tip_speed
            radial_penalty = F.softplus(root_speed - tip_speed)
            loss = loss + self.w_radial * radial_penalty

        return loss
