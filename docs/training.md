# Training

VesperNet uses a two-stage training pipeline.

## Stage-L: IlluminationNet

```bash
python scripts/train_stage_l_illumination.py --config configs/train_illumination_stage_l.yaml
```

Stage-L learns illumination from paired low/high LOL source images using:

- Retinex pseudo-illumination supervision;
- total-variation smoothness;
- reconstruction consistency.

Default output:

```text
runs/stage_l/illum_ckpt.pth
```

## Stage-R: LumiRest

```bash
python scripts/train_stage_r_lumirest.py --config configs/train_lumirest_stage_r.yaml
```

Stage-R freezes `IlluminationNet`, constructs the stable reflectance proxy, and trains `LumiRest` with:

- masked self-supervised reflectance restoration;
- gradient preservation;
- color consistency;
- residual regularization.

Default output:

```text
runs/stage_r/lumi_rest_ckpt.pth
```

## Fixed Defaults

- `tau = 0.10`
- `Pmax = 5.0`
- `omega = 15.0`
- `mask_prob = 0.2`
- `full_input_prob = 0.5`

These defaults match the paper release configuration.
