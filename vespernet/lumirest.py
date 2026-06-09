import torch
import torch.nn as nn


def _rotate(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(x, k, dims=(2, 3))


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class RotEqBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rotation-equivariant response via shared weights over 0/90/180/270.
        # Batched: group (0°, 180°) and (90°, 270°) since they share spatial dims.
        # 2 conv calls instead of 4 → ~2x faster.
        B = x.shape[0]
        x2 = _rotate(x, 2)   # 180° — same HxW as original
        x1 = _rotate(x, 1)   # 90°  — HxW swapped
        x3 = _rotate(x, 3)   # 270° — HxW swapped (same as 90°)

        # Batch 0° + 180° (same shape)
        y02 = self.conv(torch.cat([x, x2], dim=0))
        y0, y2 = y02.split(B, dim=0)

        # Batch 90° + 270° (same shape)
        y13 = self.conv(torch.cat([x1, x3], dim=0))
        y1, y3 = y13.split(B, dim=0)

        # Rotate back and average
        y = (y0 + _rotate(y1, -1) + _rotate(y2, -2) + _rotate(y3, -3)) * 0.25
        return self.act(y)


class FusionMaskNetwork(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, f_v: torch.Tensor, f_e: torch.Tensor) -> torch.Tensor:
        # This mask is a structural gating mechanism inside LumiRest.
        # It is NOT a blind-spot / masked-pixel training mask.
        m = self.mask(torch.cat([f_v, f_e], dim=1))
        return m


class LumiRest(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        disable_rotation: bool = False,
        disable_le_cond: bool = False,
        disable_gate: bool = False,
    ) -> None:
        super().__init__()
        self.disable_rotation = disable_rotation
        self.disable_le_cond = disable_le_cond
        self.disable_gate = disable_gate
        # Input: StableReflectanceProxy (3ch) + L_e (1ch) = 4 channels.
        # When disable_le_cond, only the reflectance proxy is used as input.
        in_ch = 3 if disable_le_cond else 4
        BlockCls = ConvBlock if disable_rotation else RotEqBlock
        self.v1 = ConvBlock(in_ch, base_channels)
        self.v2 = ConvBlock(base_channels, base_channels)
        self.v3 = ConvBlock(base_channels, base_channels)

        self.e1 = BlockCls(in_ch, base_channels)
        self.e2 = BlockCls(base_channels, base_channels)
        self.e3 = BlockCls(base_channels, base_channels)

        self.fusion_mask = FusionMaskNetwork(base_channels)
        self.out_conv = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.disable_le_cond:
            x = x[:, :3]  # drop L_e channel, use only the reflectance proxy
        f_v = self.v3(self.v2(self.v1(x)))
        f_e = self.e3(self.e2(self.e1(x)))
        if self.disable_gate:
            # Ablation: equal-weight average instead of learned spatial gating
            f = 0.5 * f_v + 0.5 * f_e
        else:
            m = self.fusion_mask(f_v, f_e)
            # M_f gates between Vanilla and EQ branches per-pixel.
            f = m * f_v + (1.0 - m) * f_e
        delta = self.out_conv(f)
        return delta
