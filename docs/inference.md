# Inference

Three protocol scripts are provided.

## P1 Enhanced

```bash
python scripts/infer_p1_enhanced.py --config configs/inference_p1.yaml
```

Protocol:

- TTA enabled
- bilateral filtering enabled
- no contrast stretch
- no unsharp mask
- no gray-world correction

## P2 Ablation-Enhanced

```bash
python scripts/infer_p2_ablation_enhanced.py --config configs/inference_p2_ablation.yaml
```

Protocol:

- TTA enabled
- bilateral filtering enabled
- contrast stretch enabled
- unsharp mask enabled
- gray-world correction enabled

## P3 Clean

```bash
python scripts/infer_p3_clean.py --config configs/inference_p3_clean.yaml
```

Protocol:

- no TTA
- no bilateral filtering
- no contrast stretch
- no unsharp mask
- no gray-world correction

## Custom Input Folder

```bash
python scripts/infer_p3_clean.py \
  --config configs/inference_p3_clean.yaml \
  --input-dir examples/input \
  --output-dir outputs/example_clean
```

If `--gt-dir` is provided, the scripts also write `metrics_per_image.csv` and `metrics_summary.json`.
