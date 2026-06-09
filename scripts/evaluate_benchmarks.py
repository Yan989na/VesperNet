import argparse
from copy import deepcopy
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
from vespernet.utils import apply_postprocess, build_model_from_config, load_config, save_csv, save_json, tta_forward


BENCHMARKS = {
    "LOLv1-eval15": ("data/LOL/eval15/low", "data/LOL/eval15/high"),
    "LOLv2-Rea": ("data/LOLv2-Rea/low", "data/LOLv2-Rea/high"),
    "LSRW": ("data/LSRW/low", "data/LSRW/high"),
    "BrighteningTrain": ("data/BrighteningTrain/low", "data/BrighteningTrain/high"),
    "SICE": ("data/SICE/low", "data/SICE/high"),
}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    summary = {"count": len(rows)}
    for key in ("psnr", "ssim", "mae", "lpips"):
        values = [row[key] for row in rows if key in row]
        if values:
            tensor = torch.tensor(values, dtype=torch.float32)
            summary[key] = {
                "mean": float(tensor.mean()),
                "std": float(tensor.std(unbiased=False)),
                "min": float(tensor.min()),
                "max": float(tensor.max()),
            }
    return summary


def evaluate_dataset(model, cfg: dict, device: torch.device, lpips_model, compute_lpips: bool) -> dict:
    dataset = LowLightDataset(
        mode="paired_by_index",
        source_low_dir=cfg["data"]["test_low_dir"],
        source_high_dir=cfg["data"]["test_high_dir"],
        target_low_dir=None,
        resize=tuple(cfg["data"]["resize"]) if cfg["data"].get("resize") else None,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(cfg["data"].get("num_workers", 0)))
    rows = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=cfg["data"]["dataset_name"], ncols=80):
            low = batch["low"].to(device, non_blocking=True)
            high = batch["high"][0].to(device, non_blocking=True)
            prediction = tta_forward(model, low) if cfg["inference"].get("tta", False) else model(low)["I_hat"][0]
            prediction = apply_postprocess(prediction, cfg["inference"])
            metrics = compute_all_metrics(prediction, high, lpips_model=lpips_model, compute_lpips=compute_lpips)
            metrics["image_name"] = batch["name"][0]
            rows.append(metrics)
    return {"rows": rows, "summary": summarize(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VesperNet across benchmark datasets.")
    parser.add_argument("--config", type=str, default="configs/inference_p1.yaml")
    parser.add_argument("--datasets", nargs="+", default=list(BENCHMARKS.keys()))
    parser.add_argument("--output-dir", type=str, default="outputs/benchmarks")
    parser.add_argument("--no-lpips", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg, device, load_weights=True)
    model.eval()
    lpips_model = None if args.no_lpips else LPIPSWrapper(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregated_rows = []
    summary = {}

    for name in args.datasets:
        if name not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {name}")
        dataset_cfg = deepcopy(cfg)
        dataset_cfg["data"]["dataset_name"] = name
        dataset_cfg["data"]["test_low_dir"], dataset_cfg["data"]["test_high_dir"] = BENCHMARKS[name]
        result = evaluate_dataset(model, dataset_cfg, device, lpips_model, not args.no_lpips)
        summary[name] = result["summary"]
        for row in result["rows"]:
            row["dataset"] = name
            aggregated_rows.append(row)

    save_csv(aggregated_rows, str(output_dir / "metrics_per_image.csv"))
    save_json(summary, str(output_dir / "benchmark_summary.json"))


if __name__ == "__main__":
    main()
