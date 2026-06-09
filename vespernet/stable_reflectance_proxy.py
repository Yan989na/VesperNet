import torch
import torch.nn as nn


class StableReflectanceProxy(nn.Module):
    """Lower-bounded and clamped Retinex proxy construction."""

    def __init__(self, tau: float = 0.10, eps: float = 1e-6, pmax: float = 5.0) -> None:
        super().__init__()
        self.tau = float(tau)
        self.eps = float(eps)
        self.pmax = float(pmax)

    def lower_bound(self, illumination: torch.Tensor) -> torch.Tensor:
        return torch.clamp(illumination, min=self.tau)

    def forward(self, image: torch.Tensor, illumination: torch.Tensor) -> torch.Tensor:
        bounded = self.lower_bound(illumination)
        proxy = image / (bounded + self.eps)
        return torch.clamp(proxy, 0.0, self.pmax)
