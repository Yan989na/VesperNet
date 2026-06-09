import argparse
import random
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vespernet.datasets import LowLightDataset
from vespernet.losses import StageRReflectanceLoss, construct_masked_reflectance
from vespernet.utils import build_model_from_config, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LumiRest (Stage-R) with frozen IlluminationNet.")
    parser.add_argument("--config", type=str, default="configs/train_lumirest_stage_r.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LowLightDataset(
        mode=cfg["data"]["mode"],
        source_low_dir=cfg["data"].get("source_low_dir"),
        source_high_dir=cfg["data"].get("source_high_dir"),
        target_low_dir=cfg["data"].get("target_low_dir"),
        resize=tuple(cfg["data"]["resize"]) if cfg["data"].get("resize") else None,
        file_list=cfg["data"].get("file_list"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model_from_config(cfg, device, load_weights=False)
    pretrained_illum = cfg["pretrained"]["illumination"]
    model.illumination.load_state_dict(torch.load(pretrained_illum, map_location=device))
    for parameter in model.illumination.parameters():
        parameter.requires_grad = False
    model.illumination.eval()

    optimizer = torch.optim.Adam(model.lumi_rest.parameters(), lr=float(cfg["train"]["lr"]))
    criterion = StageRReflectanceLoss(
        lambda_grad=float(cfg["train"].get("lambda_grad", 0.1)),
        lambda_color=float(cfg["train"].get("lambda_color", 0.5)),
        lambda_delta=float(cfg["train"].get("lambda_delta", 0.15)),
    )

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    periodic_dir = save_dir / "stage_r"
    periodic_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / cfg["train"]["save_name"]
    save_interval = int(cfg["train"].get("save_interval", 20))
    mask_prob = float(cfg["train"].get("mask_prob", 0.2))
    full_input_prob = float(cfg["train"].get("full_input_prob", 0.5))
    sigma_min = float(cfg["noise"]["sigma_min"])
    sigma_max = float(cfg["noise"]["sigma_max"])
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and bool(cfg["train"].get("amp", True)))

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        model.illumination.eval()
        running = {"total": 0.0, "reconstruction": 0.0, "gradient": 0.0, "color": 0.0, "delta_regularization": 0.0}
        for batch in tqdm(loader, desc=f"Stage-R {epoch}", ncols=80):
            low = batch["low"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                illumination_raw, illumination_enhanced = model.compute_illumination(low)
                proxy = model.compute_stable_reflectance_proxy(low, illumination_raw)

            masked_proxy, mask = construct_masked_reflectance(proxy, mask_prob, sigma_min, sigma_max)
            if random.random() < full_input_prob:
                restoration_input = proxy
                use_mask = False
            else:
                restoration_input = masked_proxy
                use_mask = True

            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                residual = model.lumi_rest(torch.cat([restoration_input, illumination_enhanced], dim=1))
                restored = torch.clamp(restoration_input - residual, 0.0)
                losses = criterion(restored, proxy, residual=residual, mask=mask, masked_only=use_mask)

            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            for key in running:
                running[key] += float(losses[key].item())

        steps = max(1, len(loader))
        print(
            f"[Stage-R][Epoch {epoch}] "
            f"loss={running['total'] / steps:.6f} "
            f"ss={running['reconstruction'] / steps:.6f} "
            f"grad={running['gradient'] / steps:.6f} "
            f"color={running['color'] / steps:.6f} "
            f"delta={running['delta_regularization'] / steps:.6f}"
        )
        if epoch % save_interval == 0:
            torch.save(model.lumi_rest.state_dict(), periodic_dir / f"epoch_{epoch:04d}.pth")

    torch.save(model.lumi_rest.state_dict(), save_path)
    print(f"Saved LumiRest checkpoint to {save_path}")


if __name__ == "__main__":
    main()
