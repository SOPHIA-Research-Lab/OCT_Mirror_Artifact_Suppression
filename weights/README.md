# Trained weights

Weights are not stored in this repository. They are available upon request from the corresponding author (see the main `README.md`).

Expected file names:

| Model | Configuration | File | Format |
|---|---|---|---|
| PC-CGAN reference | M3 | `pccgan_M3_G_best.pth` | `state_dict` of `octmirror.models.pccgan.UNetGenerator(in_ch=2, out_ch=2)` (the `weight/G_best.pth` file written by `scripts/train_pccgan.py`). |
| Conditional diffusion (reported model) | `--family bscan --kind dlogamp` | `diffusion/ema_best.pth` | The `ema_best.pth` file written by `legacy/diffComplexD.py`: a dictionary with the EMA weights (`ema_weights`), the architecture (`arch`), the normalization settings (`norm_cfg`) and the selected epoch. Load it with `diffComplexD.load_trained_path`, or point `--out` of the `preview` stage to the folder that contains it. |

`*.pth` and `*.pt` files are ignored by git.
