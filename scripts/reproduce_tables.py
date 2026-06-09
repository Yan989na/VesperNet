import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vespernet.metrics import compute_cdr
from vespernet.utils import save_csv


def run_subprocess(command: list[str]) -> None:
    subprocess.run(command, check=True)


def reproduce_main(args) -> None:
    command = [
        sys.executable,
        "scripts/evaluate_benchmarks.py",
        "--config",
        args.config,
        "--output-dir",
        args.output_dir,
    ]
    if args.no_lpips:
        command.append("--no-lpips")
    run_subprocess(command)


def reproduce_tau(args) -> None:
    command = [
        sys.executable,
        "scripts/run_tau_sweep.py",
        "--config",
        args.config,
        "--output-dir",
        args.output_dir,
        "--taus",
        *[str(tau) for tau in args.taus],
    ]
    if args.no_lpips:
        command.append("--no-lpips")
    run_subprocess(command)


def reproduce_cdr(args) -> None:
    input_path = Path(args.input_csv)
    score_table: dict[str, dict[str, float]] = defaultdict(dict)
    with input_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            score_table[row["dataset"]][row["method"]] = float(row[args.metric])
    cdr = compute_cdr(score_table, higher_is_better=not args.lower_is_better)
    rows = [{"method": method, f"{args.metric}_cdr": value} for method, value in sorted(cdr.items())]
    save_csv(rows, args.output_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce benchmark tables and derived metrics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    main_parser = subparsers.add_parser("main", help="Reproduce the main benchmark table for VesperNet.")
    main_parser.add_argument("--config", type=str, default="configs/inference_p1.yaml")
    main_parser.add_argument("--output-dir", type=str, default="outputs/reproduce_main")
    main_parser.add_argument("--no-lpips", action="store_true")
    main_parser.set_defaults(func=reproduce_main)

    tau_parser = subparsers.add_parser("tau", help="Reproduce the tau sensitivity table.")
    tau_parser.add_argument("--config", type=str, default="configs/inference_p1.yaml")
    tau_parser.add_argument("--output-dir", type=str, default="outputs/reproduce_tau")
    tau_parser.add_argument("--taus", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.15, 0.20])
    tau_parser.add_argument("--no-lpips", action="store_true")
    tau_parser.set_defaults(func=reproduce_tau)

    cdr_parser = subparsers.add_parser("cdr", help="Compute CDR from a multi-method summary CSV.")
    cdr_parser.add_argument("--input-csv", type=str, required=True)
    cdr_parser.add_argument("--metric", type=str, required=True)
    cdr_parser.add_argument("--output-csv", type=str, required=True)
    cdr_parser.add_argument("--lower-is-better", action="store_true")
    cdr_parser.set_defaults(func=reproduce_cdr)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
