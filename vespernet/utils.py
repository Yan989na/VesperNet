import csv
import json
import random
from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import yaml


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    base_ref = config.pop("_base_", None)
    if base_ref is None:
        return config
    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base_config = load_config(str(base_path))
    return merge_dicts(base_config, config)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def iter_image_paths(path: str) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted(candidate for candidate in root.iterdir() if candidate.is_file() and is_image_file(candidate))


def load_image(path: str, resize: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if resize is not None:
        image = image.resize(resize, Image.BILINEAR)
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_image(tensor: torch.Tensor, path: str) -> None:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_json(payload: Mapping, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_csv(rows: Iterable[Mapping], path: str) -> None:
    rows = list(rows)
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_state_dict(module: torch.nn.Module, path: str, device: torch.device) -> None:
    state_dict = torch.load(path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    module.load_state_dict(state_dict, strict=True)


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def numpy_to_tensor_image(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array.transpose(2, 0, 1)).float().to(device)


def apply_gray_world(image: torch.Tensor, strength: float = 0.3) -> torch.Tensor:
    channel_mean = image.mean(dim=(1, 2), keepdim=True)
    global_mean = channel_mean.mean()
    gain = global_mean / (channel_mean + 1e-8)
    return torch.clamp(image * (1.0 - strength + strength * gain), 0.0, 1.0)


def apply_bilateral_filter(
    image: torch.Tensor,
    diameter: int = 9,
    sigma_color: float = 30.0,
    sigma_space: float = 9.0,
) -> torch.Tensor:
    image_np = (tensor_to_numpy_image(image) * 255.0).astype(np.uint8)
    filtered = cv2.bilateralFilter(image_np, diameter, sigma_color, sigma_space)
    filtered = filtered.astype(np.float32) / 255.0
    return numpy_to_tensor_image(filtered, image.device)


def apply_auto_contrast(image: torch.Tensor, clip_pct: float = 1.0) -> torch.Tensor:
    image_np = tensor_to_numpy_image(image)
    low = np.percentile(image_np, clip_pct)
    high = np.percentile(image_np, 100.0 - clip_pct)
    if high - low < 1e-8:
        return image
    stretched = np.clip((image_np - low) / (high - low), 0.0, 1.0).astype(np.float32)
    return numpy_to_tensor_image(stretched, image.device)


def apply_unsharp_mask(image: torch.Tensor, sigma: float = 1.0, strength: float = 0.3) -> torch.Tensor:
    image_np = tensor_to_numpy_image(image)
    blurred = cv2.GaussianBlur(image_np, (0, 0), sigma)
    sharpened = np.clip(image_np + strength * (image_np - blurred), 0.0, 1.0).astype(np.float32)
    return numpy_to_tensor_image(sharpened, image.device)


def apply_postprocess(image: torch.Tensor, config: Mapping) -> torch.Tensor:
    output = image
    if config.get("bilateral", False):
        output = apply_bilateral_filter(
            output,
            diameter=int(config.get("bilateral_d", 9)),
            sigma_color=float(config.get("bilateral_sigma_color", 30.0)),
            sigma_space=float(config.get("bilateral_sigma_space", 9.0)),
        )
    if config.get("contrast", False):
        output = apply_auto_contrast(output, clip_pct=float(config.get("contrast_clip_pct", 1.0)))
    if config.get("sharpen", False):
        output = apply_unsharp_mask(
            output,
            sigma=float(config.get("sharpen_sigma", 1.0)),
            strength=float(config.get("sharpen_strength", 0.3)),
        )
    if config.get("gray_world", False):
        output = apply_gray_world(output, strength=float(config.get("gray_world_strength", 0.3)))
    return output


def build_model_from_config(
    config: Mapping,
    device: torch.device,
    load_weights: bool = True,
    tau_override: Optional[float] = None,
):
    from .illumination_net import IlluminationNet
    from .lumirest import LumiRest
    from .vespernet_model import VesperNet

    model_cfg = config["model"]
    constants = config["constants"]
    model = VesperNet(
        illumination=IlluminationNet(
            base_channels=int(model_cfg.get("illumination_channels", 32)),
            arch=str(model_cfg.get("illum_arch", "default")),
        ),
        lumi_rest=LumiRest(
            base_channels=int(model_cfg.get("lumi_rest_channels", 32)),
            disable_rotation=bool(model_cfg.get("disable_rotation", False)),
            disable_le_cond=bool(model_cfg.get("disable_le_cond", False)),
            disable_gate=bool(model_cfg.get("disable_gate", False)),
        ),
        omega=float(constants.get("omega", 15.0)),
        tau=float(tau_override if tau_override is not None else constants.get("tau", 0.10)),
        eps=float(constants.get("eps", 1e-6)),
        illum_adjust_mode=str(constants.get("illum_adjust_mode", "gamma")),
        pmax=float(constants.get("pmax", 5.0)),
        disable_residual=bool(model_cfg.get("disable_residual", False)),
    ).to(device)

    if load_weights:
        pretrained = config["pretrained"]
        load_state_dict(model.illumination, str(pretrained["illumination"]), device)
        load_state_dict(model.lumi_rest, str(pretrained["lumi_rest"]), device)

    return model


def tta_forward(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    transforms = [
        (lambda x: x, lambda x: x),
        (lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
        (lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[2])),
        (lambda x: torch.flip(x, dims=[2, 3]), lambda x: torch.flip(x, dims=[2, 3])),
    ]
    accum = torch.zeros_like(image[0])
    for forward_fn, inverse_fn in transforms:
        pred = model(forward_fn(image))["I_hat"][0]
        accum = accum + inverse_fn(pred.unsqueeze(0))[0]
    return accum / len(transforms)
