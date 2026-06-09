# Evaluation Protocol

## Metrics

The release computes:

- `PSNR`
- `SSIM`
- `MAE`
- `LPIPS`

Example:

```bash
python scripts/evaluate_benchmarks.py --config configs/inference_p1.yaml --output-dir outputs/benchmarks_p1
```

## CDR

Cross-Domain Robustness (CDR) is defined across compared methods, not from a single-method run alone.

For each dataset, metrics are min-max normalized across methods:

- for higher-is-better metrics such as PSNR and SSIM, direct normalization is used;
- for lower-is-better metrics such as LPIPS, reversed normalization is used.

The final score is:

```text
CDR = mean(normalized_score_across_datasets) - std(normalized_score_across_datasets)
```

Example:

```bash
python scripts/reproduce_tables.py cdr \
  --input-csv results/method_comparison.csv \
  --metric psnr \
  --output-csv results/psnr_cdr.csv
```

The input CSV must contain at least:

- `dataset`
- `method`
- the selected metric column

## Tau Sensitivity

```bash
python scripts/run_tau_sweep.py --config configs/inference_p1.yaml --output-dir outputs/tau_sweep
```

This evaluates the fixed trained checkpoints while changing only the inference-time lower bound `tau`.
