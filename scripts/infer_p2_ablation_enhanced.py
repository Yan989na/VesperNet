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
from vespernet.metrics import LPIPSWrapper, compute_all_metrics
from vespernet.utils import apply_postprocess, build_model_from_config, load_config, save_csv, save_image, save_json, tta_forward


def summarize(metrics_rows: list[dict]) -> dict:
    if not metrics_rows:
        return {}
    summary = {"count": len(metrics_rows)}
    for key in ("psnr", "ssim", "mae", "lpips"):
        values = [row[key] for row in metrics_rows if key in row]
        if values:
            values_tensor = torch.tensor(values, dtype=torch.float32)
            summary[key] = {
                "mean": float(values_tensor.mean()),
                "std": float(values_tensor.std(unbiased=False)),
                "min": float(values_tensor.min()),
                "max": float(values_tensor.max()),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 ablation-enhanced inference.")
    parser.add_argument("--config", type=str, default="configs/inference_p2_ablation.yaml")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--gt-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-lpips", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.input_dir:
        cfg["data"]["test_low_dir"] = args.input_dir
        if not args.gt_dir:
            cfg["data"]["test_high_dir"] = None
    if args.gt_dir:
        cfg["data"]["test_high_dir"] = args.gt_dir
    if args.output_dir:
        cfg["inference"]["output_dir"] = args.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg, device, load_weights=True)
    model.eval()

    dataset = LowLightDataset(
        mode="paired_by_index" if cfg["data"].get("test_high_dir") else "target_low_only",
        source_low_dir=cfg["data"]["test_low_dir"] if cfg["data"].get("test_high_dir") else None,
        source_high_dir=cfg["data"].get("test_high_dir"),
        target_low_dir=None if cfg["data"].get("test_high_dir") else cfg["data"]["test_low_dir"],
        resize=tuple(cfg["data"]["resize"]) if cfg["data"].get("resize") else None,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(cfg["data"].get("num_workers", 0)))

    output_dir = Path(cfg["inference"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    lpips_model = None if args.no_lpips else LPIPSWrapper(device)
    rows: list[dict] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="P2 inference", ncols=80):
            low = batch["low"].to(device, non_blocking=True)
            name = batch["name"][0]
            prediction = tta_forward(model, low) if cfg["inference"].get("tta", False) else model(low)["I_hat"][0]
            prediction = apply_postprocess(prediction, cfg["inference"])
            save_image(prediction, str(output_dir / name))
            if "high" in batch:
                high = batch["high"][0].to(device, non_blocking=True)
                metrics = compute_all_metrics(prediction, high, lpips_model=lpips_model, compute_lpips=not args.no_lpips)
                metrics["image_name"] = name
                rows.append(metrics)

    if rows:
        save_csv(rows, str(output_dir / "metrics_per_image.csv"))
        save_json(
            {
                "protocol": "P2",
                "dataset": cfg["data"].get("dataset_name", "unknown"),
                "statistics": summarize(rows),
            },
            str(output_dir / "metrics_summary.json"),
        )


if __name__ == "__main__":
    main()
