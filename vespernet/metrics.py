from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return (kernel_1d[:, None] * kernel_1d[None, :]).unsqueeze(0).unsqueeze(0)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    mse = torch.mean((pred - target) ** 2)
    return (10.0 * torch.log10(1.0 / (mse + eps))).item()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    channels = pred.shape[1]
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    window = _gaussian_window(window_size, sigma, pred.device, pred.dtype).repeat(channels, 1, 1, 1)

    mu_pred = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu_target = F.conv2d(target, window, padding=window_size // 2, groups=channels)
    mu_pred_sq = mu_pred * mu_pred
    mu_target_sq = mu_target * mu_target
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu_target_sq
    sigma_cross = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu_cross

    ssim_map = ((2 * mu_cross + c1) * (2 * sigma_cross + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )
    return ssim_map.mean().item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


class LPIPSWrapper:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            import lpips

            self._model = lpips.LPIPS(net="alex").to(self.device)
            self._model.eval()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        self._ensure_model()
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        pred = pred.to(self.device) * 2.0 - 1.0
        target = target.to(self.device) * 2.0 - 1.0
        with torch.inference_mode():
            return self._model(pred, target).item()


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_model: Optional[LPIPSWrapper] = None,
    compute_lpips: bool = True,
) -> Dict[str, float]:
    metrics = {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
        "mae": compute_mae(pred, target),
    }
    if compute_lpips:
        lpips_model = lpips_model or LPIPSWrapper(pred.device if pred.is_cuda else None)
        metrics["lpips"] = lpips_model(pred, target)
    return metrics


def compute_cdr(score_table: Mapping[str, Mapping[str, float]], higher_is_better: bool = True) -> Dict[str, float]:
    if not score_table:
        return {}
    methods = sorted({method for dataset_scores in score_table.values() for method in dataset_scores})
    per_method_scores: Dict[str, list[float]] = {method: [] for method in methods}

    for dataset_scores in score_table.values():
        values = [dataset_scores[method] for method in methods if method in dataset_scores]
        if not values:
            continue
        value_min = min(values)
        value_max = max(values)
        denom = value_max - value_min + 1e-8
        for method in methods:
            if method not in dataset_scores:
                continue
            value = dataset_scores[method]
            if higher_is_better:
                normalized = (value - value_min) / denom
            else:
                normalized = (value_max - value) / denom
            per_method_scores[method].append(normalized)

    cdr = {}
    for method, scores in per_method_scores.items():
        if not scores:
            continue
        values = torch.tensor(scores, dtype=torch.float32)
        cdr[method] = (values.mean() - values.std(unbiased=False)).item()
    return cdr
