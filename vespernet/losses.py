import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_retinex_pseudo_gt(low: torch.Tensor, high: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    ratio = low / (high + eps)
    pseudo = ratio.mean(dim=1, keepdim=True)
    pseudo = F.avg_pool2d(pseudo, kernel_size=5, stride=1, padding=2)
    return pseudo.clamp(0.0, 1.0)


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    loss_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    loss_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return loss_h + loss_w


def reconstruction_consistency_loss(illumination: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(illumination * high, low)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def color_consistency_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_sum = pred.sum(dim=1, keepdim=True) + eps
    target_sum = target.sum(dim=1, keepdim=True) + eps
    return F.l1_loss(pred / pred_sum, target / target_sum)


def sample_mask(reference: torch.Tensor, mask_prob: float) -> torch.Tensor:
    batch_size, _, height, width = reference.shape
    mask = torch.full((batch_size, 1, height, width), mask_prob, device=reference.device)
    return torch.bernoulli(mask)


def construct_masked_reflectance(
    proxy: torch.Tensor,
    mask_prob: float,
    sigma_min: float,
    sigma_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = sample_mask(proxy, mask_prob)
    local_mean = F.avg_pool2d(proxy, kernel_size=5, stride=1, padding=2)
    sigma = random.uniform(sigma_min, sigma_max)
    noise = local_mean + torch.randn_like(proxy) * sigma
    mixed = (1.0 - mask) * proxy + mask * noise
    return mixed, mask


@dataclass
class StageLIlluminationLoss(nn.Module):
    lambda_illum: float = 1.0
    lambda_tv: float = 0.1
    lambda_recon: float = 1.0

    def __post_init__(self) -> None:
        super().__init__()

    def forward(self, prediction: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> dict:
        pseudo = compute_retinex_pseudo_gt(low, high)
        loss_illum = F.l1_loss(prediction, pseudo)
        loss_tv = tv_loss(prediction)
        loss_recon = reconstruction_consistency_loss(prediction, low, high)
        total = self.lambda_illum * loss_illum + self.lambda_tv * loss_tv + self.lambda_recon * loss_recon
        return {
            "total": total,
            "illumination": loss_illum,
            "tv": loss_tv,
            "reconstruction": loss_recon,
        }


@dataclass
class StageRReflectanceLoss(nn.Module):
    lambda_grad: float = 0.1
    lambda_color: float = 0.5
    lambda_delta: float = 0.15

    def __post_init__(self) -> None:
        super().__init__()

    def forward(
        self,
        restored: torch.Tensor,
        target: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        masked_only: bool = False,
    ) -> dict:
        diff = torch.abs(restored - target)
        if masked_only and mask is not None:
            reconstruction = (mask * diff).sum() / (mask.sum() + 1e-6)
        else:
            reconstruction = diff.mean()
        grad = gradient_loss(restored, target)
        color = color_consistency_loss(restored, target)
        delta = residual.abs().mean() if residual is not None else torch.zeros(1, device=restored.device)
        total = reconstruction + self.lambda_grad * grad + self.lambda_color * color + self.lambda_delta * delta
        return {
            "total": total,
            "reconstruction": reconstruction,
            "gradient": grad,
            "color": color,
            "delta_regularization": delta,
        }
