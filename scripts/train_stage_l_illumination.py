import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vespernet.datasets import LowLightDataset
from vespernet.losses import StageLIlluminationLoss
from vespernet.utils import build_model_from_config, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IlluminationNet (Stage-L).")
    parser.add_argument("--config", type=str, default="configs/train_illumination_stage_l.yaml")
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
    optimizer = torch.optim.Adam(model.illumination.parameters(), lr=float(cfg["train"]["lr"]))
    criterion = StageLIlluminationLoss(
        lambda_illum=float(cfg["train"].get("lambda_illum", 1.0)),
        lambda_tv=float(cfg["train"].get("lambda_tv", 0.1)),
        lambda_recon=float(cfg["train"].get("lambda_recon", 1.0)),
    )

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    periodic_dir = save_dir / "stage_l"
    periodic_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / cfg["train"]["save_name"]
    save_interval = int(cfg["train"].get("save_interval", 10))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and bool(cfg["train"].get("amp", True)))

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        running = {"total": 0.0, "illumination": 0.0, "tv": 0.0, "reconstruction": 0.0}
        for batch in tqdm(loader, desc=f"Stage-L {epoch}", ncols=80):
            low = batch["low"].to(device, non_blocking=True)
            high = batch["high"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                prediction, _ = model.compute_illumination(low)
                losses = criterion(prediction, low, high)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            for key in running:
                running[key] += float(losses[key].item())

        steps = max(1, len(loader))
        print(
            f"[Stage-L][Epoch {epoch}] "
            f"loss={running['total'] / steps:.6f} "
            f"illum={running['illumination'] / steps:.6f} "
            f"tv={running['tv'] / steps:.6f} "
            f"recon={running['reconstruction'] / steps:.6f}"
        )
        if epoch % save_interval == 0:
            torch.save(model.illumination.state_dict(), periodic_dir / f"epoch_{epoch:04d}.pth")

    torch.save(model.illumination.state_dict(), save_path)
    print(f"Saved IlluminationNet checkpoint to {save_path}")


if __name__ == "__main__":
    main()
