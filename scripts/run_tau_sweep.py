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
from vespernet.utils import apply_postprocess, build_model_from_config, load_config, save_csv, save_json, tta_forward


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference-time tau sensitivity analysis.")
    parser.add_argument("--config", type=str, default="configs/inference_p1.yaml")
    parser.add_argument("--taus", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--output-dir", type=str, default="outputs/tau_sweep")
    parser.add_argument("--no-lpips", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LowLightDataset(
        mode="paired_by_index" if cfg["data"].get("test_high_dir") else "target_low_only",
        source_low_dir=cfg["data"]["test_low_dir"] if cfg["data"].get("test_high_dir") else None,
        source_high_dir=cfg["data"].get("test_high_dir"),
        target_low_dir=None if cfg["data"].get("test_high_dir") else cfg["data"]["test_low_dir"],
        resize=tuple(cfg["data"]["resize"]) if cfg["data"].get("resize") else None,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(cfg["data"].get("num_workers", 0)))
    lpips_model = None if args.no_lpips else LPIPSWrapper(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for tau in args.taus:
        model = build_model_from_config(cfg, device, load_weights=True, tau_override=tau)
        model.eval()
        tau_rows = []
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"tau={tau:.2f}", ncols=80):
                low = batch["low"].to(device, non_blocking=True)
                prediction = tta_forward(model, low) if cfg["inference"].get("tta", False) else model(low)["I_hat"][0]
                prediction = apply_postprocess(prediction, cfg["inference"])
                row = {"tau": tau, "image_name": batch["name"][0]}
                if "high" in batch:
                    high = batch["high"][0].to(device, non_blocking=True)
                    row.update(compute_all_metrics(prediction, high, lpips_model=lpips_model, compute_lpips=not args.no_lpips))
                tau_rows.append(row)
                rows.append(row)
        save_json({"tau": tau, "statistics": summarize(tau_rows)}, str(output_dir / f"tau_{tau:.2f}.json"))

    save_csv(rows, str(output_dir / "tau_sweep.csv"))


if __name__ == "__main__":
    main()
