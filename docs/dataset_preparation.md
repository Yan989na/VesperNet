# Dataset Preparation

This release expects paired low/high folders for all training and benchmark datasets.

## Directory Layout

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

## Notes Per Dataset

- `LOL/our485`: used for Stage-L and Stage-R training.
- `LOL/eval15`: used as the source-domain held-out benchmark.
- `LOLv2-Rea`, `LSRW`, `BrighteningTrain`, and `SICE`: used for zero-shot cross-domain evaluation.
- If your filenames do not match exactly across paired folders, create a text file with `low_path|high_path` pairs and pass it through the `file_list` field in the config.

## Validation

Before training or evaluation, verify:

- each `low/` folder contains only image files;
- each paired benchmark has matching filenames or a valid `file_list`;
- images are stored in RGB-compatible formats such as `png`, `jpg`, or `jpeg`.
