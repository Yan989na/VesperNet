# VesperNet

Official implementation of the paper **Stable Reflectance Decomposition for Cross-Domain Low-Light Image Enhancement**.

VesperNet stabilizes Retinex enhancement before restoration starts. The pipeline first estimates illumination with `IlluminationNet`, constructs a stable reflectance proxy with a lower-bounded illumination denominator and a clamped reflectance proxy, then restores reflectance residuals with `LumiRest`, and finally reconstructs the enhanced image. Training uses mixed supervision: paired illumination learning in Stage-L and masked self-supervised reflectance restoration in Stage-R.

## Method Summary

- Stable Retinex reflectance proxy construction from the input image and estimated illumination.
- Lower-bounded illumination denominator with the default fixed threshold `tau = 0.10`.
- Clamped reflectance proxy with the default fixed bound `Pmax = 5.0`.
- `LumiRest` reflectance-domain residual restoration with luminance guidance.
- Mixed supervision with paired illumination learning and masked self-supervised reflectance restoration.

## Repository Layout

```text
VesperNet/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
├── configs/
├── vespernet/
├── scripts/
├── pretrained/
├── examples/
├── assets/
└── docs/
```

## Installation

Install from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate vespernet
```

## Dataset Preparation

Expected layout:

```text
data/
├── LOL/
│   ├── our485/
│   │   ├── low/
│   │   └── high/
│   └── eval15/
│       ├── low/
│       └── high/
├── LOLv2-Rea/
│   ├── low/
│   └── high/
├── LSRW/
│   ├── low/
│   └── high/
├── BrighteningTrain/
│   ├── low/
│   └── high/
└── SICE/
    ├── low/
    └── high/
```

Datasets used by the release:

- `LOL/our485`: source-domain training set for Stage-L and Stage-R.
- `LOL/eval15`: source-domain held-out evaluation set.
- `LOLv2-Rea`: real-scene cross-domain benchmark.
- `LSRW`: real-world cross-domain benchmark.
- `BrighteningTrain`: paired cross-domain benchmark.
- `SICE`: cross-domain evaluation split arranged as paired low/high images.

Detailed notes are in [docs/dataset_preparation.md](docs/dataset_preparation.md).

## Training

Stage-L illumination training:

```bash
python scripts/train_stage_l_illumination.py --config configs/train_illumination_stage_l.yaml
```

Stage-R LumiRest training with frozen IlluminationNet:

```bash
python scripts/train_stage_r_lumirest.py --config configs/train_lumirest_stage_r.yaml
```

## Inference

P1 enhanced inference:

```bash
python scripts/infer_p1_enhanced.py --config configs/inference_p1.yaml
```

P2 ablation-enhanced inference:

```bash
python scripts/infer_p2_ablation_enhanced.py --config configs/inference_p2_ablation.yaml
```

P3 clean inference:

```bash
python scripts/infer_p3_clean.py --config configs/inference_p3_clean.yaml
```

To run on a custom folder:

```bash
python scripts/infer_p1_enhanced.py \
  --config configs/inference_p1.yaml \
  --input-dir examples/input \
  --output-dir outputs/example_p1
```

More usage notes are in [docs/inference.md](docs/inference.md).

## Evaluation

Evaluate the five benchmarks with PSNR, SSIM, MAE, and LPIPS:

```bash
python scripts/evaluate_benchmarks.py --config configs/inference_p1.yaml --output-dir outputs/benchmarks_p1
```

Run the inference-time `tau` sensitivity analysis:

```bash
python scripts/run_tau_sweep.py --config configs/inference_p1.yaml --output-dir outputs/tau_sweep
```

Reproduce the main VesperNet benchmark table entry points:

```bash
python scripts/reproduce_tables.py main --config configs/inference_p1.yaml --output-dir outputs/reproduce_main
python scripts/reproduce_tables.py tau --config configs/inference_p1.yaml --output-dir outputs/reproduce_tau
```

Compute CDR from a multi-method summary CSV:

```bash
python scripts/reproduce_tables.py cdr \
  --input-csv results/method_comparison.csv \
  --metric psnr \
  --output-csv results/psnr_cdr.csv
```

`CDR` is defined across compared methods, so it requires a CSV containing at least the columns `dataset`, `method`, and the selected metric. Evaluation protocol details are in [docs/evaluation_protocol.md](docs/evaluation_protocol.md).

## Pretrained Weights

The latest paper-consistent checkpoints included in this release are:

- `pretrained/illum_ckpt.pth`
- `pretrained/lumi_rest_ckpt.pth`

The inference configs load these filenames by default. To use the shipped weights without editing paths:

```bash
python scripts/infer_p3_clean.py --config configs/inference_p3_clean.yaml --input-dir examples/input --output-dir outputs/example_p3
```

## Reproducibility Notes

- Default fixed parameters: `tau = 0.10`, `Pmax = 5.0`, and `omega = 15.0`.
- Source-fixed LOL-only training setting.
- No target-domain tuning.
- P1 uses fixed TTA plus fixed bilateral filtering.
- P2 is for ablation-only enhanced comparison.
- P3 is the clean forward pass without TTA or hand-crafted post-processing.

## Citation

Please cite the metadata in [CITATION.cff](CITATION.cff).

## Additional Documentation

- [docs/training.md](docs/training.md)
- [docs/inference.md](docs/inference.md)
- [docs/evaluation_protocol.md](docs/evaluation_protocol.md)

## Originality and Third-Party Resources

This repository contains the authors' official implementation of VesperNet for the manuscript "Stable Reflectance Decomposition for Cross-Domain Low-Light Image Enhancement." The model implementation, training scripts, inference scripts, and evaluation protocol released here correspond to the authors' submitted revision.

This repository does not redistribute third-party datasets. Users should download LOL, LOLv2-Rea, LSRW, BrighteningTrain, and SICE from their official sources and arrange them according to the documented directory structure. Third-party Python packages are used under their respective licenses.

The VesperNet model code is authored by the paper authors; external libraries are used only as dependencies.
