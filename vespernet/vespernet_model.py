from typing import Dict

import torch
import torch.nn as nn

from .stable_reflectance_proxy import StableReflectanceProxy


class VesperNet(nn.Module):
    def __init__(
        self,
        illumination: nn.Module,
        lumi_rest: nn.Module,
        omega: float = 15.0,
        tau: float = 0.10,
        eps: float = 1e-6,
        illum_adjust_mode: str = "gamma",
        pmax: float = 5.0,
        disable_residual: bool = False,
    ) -> None:
        super().__init__()
        self.illumination = illumination
        self.lumi_rest = lumi_rest
        self.omega = float(omega)
        self.illum_adjust_mode = illum_adjust_mode
        self.proxy = StableReflectanceProxy(tau=tau, eps=eps, pmax=pmax)
        self.disable_residual = disable_residual

    @property
    def tau(self) -> float:
        return self.proxy.tau

    @tau.setter
    def tau(self, value: float) -> None:
        self.proxy.tau = float(value)

    @property
    def eps(self) -> float:
        return self.proxy.eps

    @property
    def pmax(self) -> float:
        return self.proxy.pmax

    def compute_illumination(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        illumination_raw = self.illumination(image)
        if self.illum_adjust_mode == "gamma":
            illumination_enhanced = torch.clamp(illumination_raw.pow(1.0 / self.omega), 0.0, 1.0)
        elif self.illum_adjust_mode == "linear":
            illumination_enhanced = torch.clamp(illumination_raw * self.omega, 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported illum_adjust_mode: {self.illum_adjust_mode}")
        return illumination_raw, illumination_enhanced

    def compute_stable_reflectance_proxy(self, image: torch.Tensor, illumination: torch.Tensor) -> torch.Tensor:
        return self.proxy(image, illumination)

    def forward_reflectance(
        self,
        image: torch.Tensor,
        illumination_raw: torch.Tensor,
        illumination_enhanced: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        proxy = self.compute_stable_reflectance_proxy(image, illumination_raw)
        restoration_input = torch.cat([proxy, illumination_enhanced], dim=1)
        residual = self.lumi_rest(restoration_input)
        if self.disable_residual:
            reflectance = torch.clamp(residual, 0.0)
        else:
            reflectance = torch.clamp(proxy - residual, 0.0)
        return {
            "proxy": proxy,
            "residual": residual,
            "reflectance": reflectance,
        }

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        illumination_raw, illumination_enhanced = self.compute_illumination(image)
        reflectance_outputs = self.forward_reflectance(image, illumination_raw, illumination_enhanced)
        enhanced = torch.clamp(reflectance_outputs["reflectance"] * illumination_enhanced, 0.0, 1.0)
        return {
            "L_T": illumination_raw,
            "L_e": illumination_enhanced,
            "stable_reflectance_proxy": reflectance_outputs["proxy"],
            "delta": reflectance_outputs["residual"],
            "R_e": reflectance_outputs["reflectance"],
            "I_hat": enhanced,
        }
