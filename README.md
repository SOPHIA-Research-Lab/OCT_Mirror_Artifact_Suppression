# Amplitude-Phase Trade-offs in Deep Learning Reconstruction for Mirror Artifact Suppression in Optical Coherence Tomography

Official code for **"Amplitude-Phase Trade-offs in Deep Learning Reconstruction for Mirror Artifact Suppression in Optical Coherence Tomography"**.

> Status: manuscript under review. The paper link and citation will be added upon acceptance.

**Authors:** Diego Pulgarín, Sebastian Ruiz-Lopera, Juan José Cadavid-Muñoz, Carlos Trujillo, René Restrepo.
Applied Optics and Electronic Instrumentation Research Group, EAFIT University (Medellín, Colombia). S. Ruiz-Lopera is also affiliated with MIT EECS and the Wellman Center for Photomedicine (Harvard Medical School / MGH).

## Abstract

Fourier-domain optical coherence tomography (FD-OCT) is affected by complex-conjugate mirror artifacts that can hinder the visualization of true tissue structures and compromise quantitative analysis. Most deep learning approaches address this problem in the intensity domain, often disregarding phase information required for coherent post-processing. In this work, we investigate mirror artifact suppression directly in the complex domain using two generative strategies: a phase-constrained conditional generative adversarial network (PC-CGAN) and a conditional diffusion model. The proposed methods are evaluated against representative learned and analytical approaches, including an intensity-based generative complex conjugate artifact removal network (CCARNet), the Hilbert-transform analytic signal, and the Dispersion-Encoded Full-Range method (DEFR). All methods are assessed using a joint criterion that considers both amplitude reconstruction and phase preservation. In addition to standard image quality metrics, we analyze axial and lateral phase continuity and validate physical consistency through a bulk-motion correction experiment. The results show a clear trade-off between structural fidelity and phase preservation: the diffusion model achieves the highest structural similarity, while the phase-constrained generative adversarial network is the only method that preserves phase relationships compatible with optical coherence tomography (OCT) processing. These findings highlight the importance of evaluating artifact removal methods in OCT using both intensity and phase information.

**Keywords:** complex conjugate artifact removal, complex-field diffusion models, generative adversarial networks, optical coherence tomography.

## What this repository contains

| Component | Description |
|---|---|
| `src/octmirror/preprocessing.py` | Numerical mirror-artifact generation, log-amplitude normalization and its inverse, tomogram loading. |
| `src/octmirror/datasets.py` | PyTorch dataset for paired (corrupted, clean) complex B-scans. |
| `src/octmirror/models/pccgan.py` | PC-CGAN: U-Net generator and PatchGAN discriminator. |
| `src/octmirror/losses.py` | Hinge adversarial losses, circular phase loss, phase-gradient loss. |
| `src/octmirror/training.py` | PC-CGAN training loop. |
| `src/octmirror/metrics.py` | Image metrics (SSIM, PSNR, MSE, histogram similarity) and phase metrics (WPC, CCC, PG-SSIM). |
| `src/octmirror/stats.py` | Bootstrap confidence intervals, Wilcoxon tests, Holm-Bonferroni correction. |
| `src/octmirror/inference.py` | Loading PC-CGAN weights and running inference. |
| `scripts/` | PC-CGAN command-line entry points: `train_pccgan.py`, `evaluate_pccgan.py`, `infer.py`. |
| `configs/` | One YAML per PC-CGAN ablation: `pccgan/m2..m8.yaml`. |
| `legacy/diffComplexD.py` | Conditional diffusion model: training, self-tests and sampling (self-contained script). |
| `legacy/evaluate_diffusion_D.py` | Evaluation of the diffusion runs against the Hilbert, DEFR and oracle baselines. |
| `legacy/` (other files) | Original PC-CGAN training and evaluation scripts, used as reference by the golden tests. |
| `data/sample_volumes/` | Two small test volumes for inference. |
| `weights/` | Instructions to request the trained weights. |

**Scope of this release.**
- The PC-CGAN is provided with training, evaluation and inference code in the `octmirror` package.
- The conditional diffusion model is provided with training, sampling and evaluation code in two self-contained scripts in `legacy/`. It is not part of the `octmirror` package.
- The Hilbert-transform analytic signal and DEFR baselines are computed inside `legacy/evaluate_diffusion_D.py`. The package stub `src/octmirror/baselines.py` is not implemented.
- CCARNet and the bulk-motion correction experiment are not included.

## Installation

```bash
git clone [https://github.com/Diegoapulgarin/OCT_Mirror_Artifact_Suppression.git](https://github.com/Diegoapulgarin/OCT_Mirror_Artifact_Suppression.git)
cd OCT_Mirror_Artifact_Suppression
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

A CUDA GPU is recommended for training and for diffusion sampling. PC-CGAN inference runs on CPU. The diffusion scripts also need `diffusers` and `segmentation-models-pytorch` (included in `requirements.txt`).

## Trained weights

Weights are **not** stored in this repository because of their size and GitHub's file-size limits. They are available upon request from the corresponding author (see Contact). See `weights/README.md` for the expected file names:

| Model | Configuration | Expected file |
|---|---|---|
| PC-CGAN reference | M3 | `weights/pccgan_M3_G_best.pth` |
| Conditional diffusion (reported model) | `--family bscan --kind dlogamp` | `weights/diffusion/ema_best.pth` |

## Inference

The two sample volumes in `data/sample_volumes/` (optic nerve and salmon skin, each of size Z×X×Y = 512×512×10) are artifact-free. By default, `infer.py` first simulates the mirror artifact numerically and then reconstructs the field.

```bash
# PC-CGAN (M3), all 10 B-scans, with metrics against the clean volume
python scripts/infer.py --model pccgan \
    --weights weights/pccgan_M3_G_best.pth \
    --input data/sample_volumes/optic_nerve.npy \
    --out-dir outputs/pccgan_optic_nerve \
    --compute-metrics

```

Outputs: a complex `.npy` reconstruction, amplitude (dB) and phase PNGs, and `metrics.csv` when `--compute-metrics` is used.

The diffusion model is sampled with the `preview` stage of its training script. It simulates the artifact on the central B-scan of every volume in `--data-val`, reconstructs it with the model found in `--out` and writes a figure and `preview_metrics.csv` to `<out>/previews/`:

```bash
python legacy/diffComplexD.py --stage preview \
    --out weights/diffusion \
    --data-val data/sample_volumes
```

Two options need care:
- `--simulate-mirror / --no-simulate-mirror`: use `--no-simulate-mirror` if your volume already contains the artifact.
- `--scale-from {target,input}`: the log-amplitude normalization is inverted using min/max values from either the clean target or the input. The paper's evaluation protocol uses the clean target, which is only available when the artifact is simulated. With real corrupted data, use `--scale-from input`.

### Data format

Tomograms are `.npy` arrays with axes **(Z, X, Y)**: depth, fast-scan, slow-scan. Both layouts are accepted: a complex array `(Z, X, Y)` or a real array `(Z, X, Y, 2)` whose last axis holds real and imaginary parts. The mirror artifact is generated numerically from an artifact-free complex tomogram, following the Fourier-equivalent procedure described in the paper: fringes are recovered, only their real part is kept, and the result is transformed back to the depth domain. The pipeline works B-scan by B-scan, with log-amplitude normalization per slice.

## Training

Data paths are passed on the command line. Folder names and file lists live in the configs.

```bash
# PC-CGAN, reference configuration (M3)
python scripts/train_pccgan.py --config configs/pccgan/m3.yaml \
    --data-root /path/to/Data --out-dir runs/pccgan

# Conditional diffusion, reported configuration
python legacy/diffComplexD.py --stage train --family bscan --kind dlogamp \
    --data /path/to/your/training/data \
    --data-val /path/to/your/validation/data \
    --out runs/diffusion
```

The diffusion training files are listed in `TRAIN_FILES` at the top of `legacy/diffComplexD.py`; `--data` is the folder that contains them.

### PC-CGAN ablations (M2–M8)

Common settings: L1 weight 100, amplitude weight 5, generator/discriminator learning rates 3e-5 / 5e-6, Adam betas (0.5, 0.999), 300 epochs, batch size 32, hinge adversarial loss, gradient clipping 1.0, seed 0, 90/10 train/validation split by B-scan.

| ID | Phase weight | Phase weighting | Phase-gradient loss |
|---|---|---|---|
| M2 | 0 | none | no |
| **M3 (reference)** | 20 | geometric mean of amplitudes | no |
| M4 | 20 | none | no |
| M5 | 2 | geometric | no |
| M6 | 60 | geometric | no |
| M7 | 150 | geometric | no |
| M8 | 20 | geometric | yes (weight 20) |

### Diffusion model and ablations

The reported model is trained with `--family bscan --kind dlogamp` on full 512×512 B-scans. Default settings: 40 epochs, batch size 8, AdamW (learning rate 1e-4, weight decay 0.01, linear warmup and cosine decay), EMA decay 0.9999, seed 0. The checkpoint is selected on an internal validation block held out from each training volume; the test set is never used for selection. The formulation is documented in the docstring of the script.

The same script trains the alternatives used in the ablations: the other `--kind` values (`mask`, `null`, `dmask`) and the `roi` family (`--family roi`, diffusion or deterministic regressors). Run `python legacy/diffComplexD.py --help` for all options and `--stage selftest` to check the installation.

**ROI size.** In the `roi` family, `--roi` sets the side of the square ROIs. The default, 512, uses the full B-scan. Smaller values (for example 256) reduce memory use and training time, which is useful for quick experiments or GPUs with limited VRAM; the memory of the U-Net, in particular its self-attention layers, grows quickly with image size. The `bscan` family always uses the full B-scan; if it does not fit on your GPU, reduce `--batch-size`. Evaluate a model with the ROI size it was trained with (`evaluate_diffusion_D.py --roi`).

## Evaluation of the PC-CGAN

```bash
python scripts/evaluate_pccgan.py \
    --ablation-root /path/to/AblationPCCGAN \
    --validation-dir /path/to/validationCxPaper \
    --out-dir outputs/evaluation
```

`--ablation-root` must contain one folder per configuration (`M2_ComplexCGAN_NoPhase`, `M3_PCCGAN_Reference`, …), each with `weight/G_best.pth` and `metrics_epoch.csv`.

**Protocol.** Each validation volume contributes up to 40 evenly spaced B-scans. Intensity metrics (SSIM, PSNR, MSE, histogram cosine similarity) are computed on despeckled, normalized amplitude in dB using non-local means with `h = 0.1`. Phase metrics (WPC, CCC, PG-SSIM) use the raw complex field without filtering. Reported values are means with 95% bootstrap confidence intervals (2000 resamples, seed 42). Paired Wilcoxon tests compare every configuration against M3, with Holm-Bonferroni correction across all 42 tests as one family.

**Comparability.** Statistics are paired across B-scans and describe B-scan-level differences. They do not establish subject-level independence, because several B-scans come from the same volume. The training/validation split inside the training script is also done per B-scan. The independent validation set used for the reported results is separate from the training data.

## Evaluation of the diffusion models

Set `DIFF_ROOT` (folder with one sub-folder per training run) and `VALIDATION_DATA_DIR` at the top of `legacy/evaluate_diffusion_D.py`, then run:

```bash
python legacy/evaluate_diffusion_D.py            # full 512x512 B-scans (default)
python legacy/evaluate_diffusion_D.py --roi 256  # runs trained on 256x256 ROIs
```

The script evaluates the `best` and `final` checkpoints of every run with the same intensity and phase metrics as the PC-CGAN evaluation, plus mirror suppression and consistency with the measurement, and compares them with the Hilbert, DEFR and oracle baselines. Results go to `DIFF_ROOT/evaluation_D_<roi>/`.

## Limitations

- Each B-scan is processed independently. No constraint enforces phase continuity along the slow-scan axis.
- The diffusion model is a standard Euclidean Gaussian formulation. In the paper it reaches the highest structural similarity but loses phase coherence.
- The conclusions apply to the evaluated methods, datasets and objectives. They do not constitute validation for Doppler OCT, OCT angiography or elastography.

## Reproducibility notes

See `docs/REPRODUCIBILITY.md` for the seeds, data splits and evaluation protocol used by the code.

## Data availability

The sample volumes are for demonstration and testing of the inference code. The remaining training and validation data, and the trained weights, are available upon reasonable request, subject to applicable ethical approval, institutional data-management requirements and restrictions on the underlying acquisitions.

## Citation

The paper is under review. Until it is published, please cite this repository:

```bibtex
@software{pulgarin_oct_mirror_artifact_suppression,
  author = {Pulgar{\'i}n, Diego and Ruiz-Lopera, Sebastian and Cadavid-Mu{\~n}oz, Juan Jos{\'e} and Trujillo, Carlos and Restrepo, Ren{\'e}},
  title  = {OCT\_Mirror\_Artifact\_Suppression},
  url    = {[https://github.com/Diegoapulgarin/OCT_Mirror_Artifact_Suppression](https://github.com/Diegoapulgarin/OCT_Mirror_Artifact_Suppression)},
  note   = {Code for the paper "Amplitude-Phase Trade-offs in Deep Learning Reconstruction for Mirror Artifact Suppression in Optical Coherence Tomography"}
}
```

## License

<TBD: MIT or Apache-2.0>. See `LICENSE`.

## Contact

Diego Pulgarín, EAFIT University: dapulgaris@eafit.edu.co. Weight requests and questions: open an issue or write to this address.
