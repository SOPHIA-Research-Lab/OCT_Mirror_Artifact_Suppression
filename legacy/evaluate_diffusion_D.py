# -*- coding: utf-8 -*-
"""
===============================================================================
UNIFIED EVALUATION -- DIFFUSION MODELS AND REGRESSORS (D) + CLASSICAL BASELINES
===============================================================================
Walks the run folders in DIFF_ROOT, samples every model trained with diffComplexD.py on the
validation (test) set and computes the task metrics with the same code as the PC-CGAN evaluation
(the metric functions are copied verbatim).

WHAT IS EVALUATED
  * Every run folder trained at the ROI of this script (--roi), in its TWO checkpoints: 'best'
    (selected with the internal validation of the training) and 'final' (last). roi-family
    diffusion: every variant of SAMPLING_VARIANTS (G1 without DC, G2 exact DC, G3 late DC).
    Regressors: one pass ('R_direct'). bscan family: the variants of
    diffComplexD.bscan_eval_variants (D_raw / D_dc for dlogamp, D_mask for dmask).
    Method = '<run>/<checkpoint>/<variant>'.
  * Baselines, computed ONCE and cached (reused while the data, the ROI and their parameters do
    not change):
      BASELINE_NoOp_Input     the corrupted input as is.
      HILBERT_KeepLower/Upper analytic signal along k = keep one half-space.
      HILBERT_OracleHalf      the best half-space per ROI, chosen WITH the ground truth. ORACLE,
                              declared as such: the choice cannot be decided from the data.
      DEFR_DispEnc            advanced DEFR (Koettig et al. 2012) on an input WITH DISPERSION
                              ENCODING. DIFFERENT INPUT from all other methods: flagged in
                              input_type.
      ORACLE_WienerLocal      BOUND of the Wiener head: real gain with the local powers of the two
                              sources taken from the ground truth (sigma = 3 px). ORACLE.
      ORACLE_RatioMask        same with instantaneous powers: the best possible real mask.

PROBLEM CONVENTION: each ROI of ROI_SIZE is an independent acquisition with its own artifact (as
in training). The artifact is NOT cropped from the artifact of the full B-scan: cropping in depth
would break the z <-> -z pairs. With ROI_SIZE = 512 (default) the ROI is the full B-scan, the
protocol of the paper. All methods, baselines included, solve exactly the same problem per ROI.

SCALE: every prediction is delivered as a PHYSICAL complex field. No method is de-normalized with
ground-truth statistics, so the energy metrics (suppression) share the scale by construction.

NEW METRICS with respect to the PC-CGAN evaluator
  ccc_vs_mirror_solution  CCC against S~ = conj(S(-z)), the other exact solution. If it is high,
                          the model chose the mirror mode: a "phase failure" that is in fact an
                          orientation failure.
  dc_residual             ||M(pred) - y|| / ||y||: consistency with the measurement.
  ccc_tissue_dom / ccc_ambiguous / nrmse_tissue_dom
                          metrics per pixel class defined by the forward model: tissue-dominant
                          (recoverable) and ambiguous zone (bounded).

OUTPUTS in EVAL_DIR: per_roi_metrics.csv, summary_metrics.csv, per_volume_summary.csv,
metric_task_correlation.csv, wilcoxon_vs_noop.csv, plots/ (panels with the SAME B-scan for all
methods, decile and despeckling curves).

USAGE
    python evaluate_diffusion_D.py                  # evaluates everything not cached yet (ROI 512)
    python evaluate_diffusion_D.py --roi 256        # runs trained on 256 x 256 ROIs
    python evaluate_diffusion_D.py --force          # re-samples every run
    python evaluate_diffusion_D.py --runs RUN1 RUN2 # only these folders
    python evaluate_diffusion_D.py --baselines-only # only (re)build the baseline cache
===============================================================================
"""
import os
import sys
import json
import glob
import hashlib
import argparse
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift, ifftshift
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_nl_means
import cv2
from scipy.stats import wilcoxon, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ===================== CONFIGURATION ===================== #

DIFF_ROOT = 'put/your/output/path/here'     # folder with one sub-folder per training run
RUN_NAMES = []            # empty = every folder of DIFF_ROOT that contains a trained model
VALIDATION_DATA_DIR = 'put/your/validation/data/path/here'
ROI_SIZE = 512            # 512 = full B-scan (paper protocol); 256 = runs trained on 256 x 256 ROIs
EVAL_DIR = os.path.join(DIFF_ROOT, f'evaluation_D_{ROI_SIZE}')
CACHE_DIR = os.path.join(EVAL_DIR, 'cache')
PLOTS_DIR = os.path.join(EVAL_DIR, 'plots')
PRED_DIR = os.path.join(EVAL_DIR, 'predictions')
ERROR_LOG_PATH = os.path.join(EVAL_DIR, 'evaluation_errors.log')

N_BSCANS_PER_VOLUME = 40
SAVE_PREDICTIONS = True           # ~0.5 GB per run and variant; allows new metrics without re-sampling

# Sampling variants of roi-family diffusion. n_samples > 1 = mean of K samples (approx. posterior mean).
SAMPLING_VARIANTS = {
    'G1_plain':   dict(variant='G1_plain',   n_samples=1),
    'G2_dc_full': dict(variant='G2_dc_full', n_samples=1),
    'G3_dc_late': dict(variant='G3_dc_late', n_samples=1),
    # 'G2_dc_full_mean4': dict(variant='G2_dc_full', n_samples=4),
}
PANEL_PRIMARY_VARIANTS = ('G2_dc_full', 'R_direct', 'D_mask', 'D_dc')   # variant that represents each run in the global panel
PANEL_BASELINES = ['HILBERT_OracleHalf', 'DEFR_DispEnc', 'ORACLE_WienerLocal']
SAMPLING_STEPS = 100
SAMPLE_BATCH = 64
SAMPLE_SEED = 0                    # same seed per ROI in every variant -> paired comparison

# DEFR: advanced DEFR algorithm (defr_advanced, Koettig et al. 2012), vectorized over A-lines
# (identical to the per-A-line version up to 1e-16), with the parameters of the reference
# implementation:  phi(k) = BETA2 * k^2 - phi[N/2],  k = linspace(0, 1, N)
# Convention fixes with respect to the reference implementation (see defr_batch):
#   1) the encoding is exp(+i phi), the one the algorithm assumes; the reference script induced
#      exp(-i phi) and DEFR extracted the MIRROR copy: output ~ conj(GT shifted 1 px)/2, CCC ~0.
#   2) no flip in z (the flip was not undone when saving).
#   3) zero delay moved from N/2 to index 0 (no-fftshift convention of the algorithm) and back.
#   4) output x2: the real measurement halves the amplitude (as does the analytic signal).
DEFR_BETA2 = 1000.0
DEFR_MAX_ITER = 50
DEFR_THRESH_DB = 20.0
DEFR_BATCH = 32

# Metrics: same constants as the PC-CGAN evaluation
SEPARABILITY_CONTRAST_THRESHOLDS = (3.0, 10.0, 30.0)
MIRROR_DOMINANT_CONTRAST = 10.0
NOISE_FLOOR_PERCENTILE = 50.0
CLASS_CONTRAST = 3.0               # tissue-dominant: |S(z)| > 3 |S(-z)|; ambiguous: between 1/3 and 3
DESPECKLE_H_SWEEP = [0.10, 0.08, 0.06, 0.04, 0.02, 0.00]
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 42
IDENTITY_COLLAPSE_GATE_FILTER = 'nlm_h010'
IDENTITY_COLLAPSE_THRESHOLD = 0.85
MIRROR_SUPPRESSION_MIN = 0.50
ROBUST_MEDIAN_METRICS = ('mirror_suppression', 'mirror_residual_db', 'dc_residual')
NOOP_BASELINE_NAME = 'BASELINE_NoOp_Input'
METRIC_VERSION = 'D1'              # change it if a metric is modified -> invalidates the caches

METRIC_COLUMNS = ['ssim', 'psnr', 'mse', 'hist_cosine_similarity', 'wpc', 'ccc', 'pgssim',
                  'mirror_suppression', 'mirror_residual_db',
                  'ccc_vs_mirror_solution', 'dc_residual', 'ccc_tissue_dom', 'ccc_ambiguous',
                  'nrmse_tissue_dom']
KEY_COLS = ['tomogram_file', 'bscan_index', 'z0', 'x0']

N_JOBS = int(os.environ.get('SLURM_NTASKS', os.environ.get('SLURM_CPUS_PER_TASK', max(1, (os.cpu_count() or 2) - 1))))


def log_error(message):
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(ERROR_LOG_PATH, 'a') as f:
        f.write(message + "\n")
    print(message)


def mirrorArtifact(array):
    """Forward model, identical to the one of PC-CGAN and of diffComplexD.py (axis 0 = depth)."""
    fringes = ifftshift(ifft(ifftshift(array, axes=0), axis=0), axes=0)
    return fftshift(fft(fftshift(fringes.real, axes=0), axis=0), axes=0)


# ===================== COPIED VERBATIM FROM THE PC-CGAN EVALUATION ===================== #
def simcos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def calculate_ssim(image1, image2):
    assert image1.shape == image2.shape, "Images must have the same size."
    ssim_value, _ = ssim(image1, image2, full=True, data_range=1)
    return ssim_value

def calculate_mse(image_original, image_reconstructed):
    assert image_original.shape == image_reconstructed.shape, "Images must have the same size."
    mse = np.mean((image_original - image_reconstructed) ** 2)
    return mse

def calculate_psnr(image1, image2, max_val=1.0):
    assert image1.shape == image2.shape, "Images must have the same size."
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10(max_val**2 / mse)
    return psnr

# NOTE: method 'cosine-similarity' returns a SIMILARITY (1 = identical), whereas a histogram
# difference follows the "lower is better" convention. The value is stored as-is under the column
# name `hist_cosine_similarity`, without inverting it, so the direction of interpretation is
# decided explicitly when building the final table.
def histogram_difference(image1, image2, method="cosine-similarity"):
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 1])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 1])
    if method == "kullback-leibler":
        hist1 /= hist1.sum()
        hist2 /= hist2.sum()
    if method == "chi-squared":
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    elif method == "kullback-leibler":
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_KL_DIV)
    elif method == 'cosine-similarity':
        return simcos(np.ravel(hist1), np.ravel(hist2))
    else:
        raise ValueError("Unknown method")

def weighted_phase_coherence(pred_cx, gt_cx):
    w    = np.abs(pred_cx) * np.abs(gt_cx)
    diff = np.angle(pred_cx) - np.angle(gt_cx)
    return float(np.sum(w * np.cos(diff)) / (np.sum(w) + 1e-12))

def masked_complex_coherence(pred_cx, gt_cx, percentile=50):
    mask   = np.abs(gt_cx) > np.percentile(np.abs(gt_cx), percentile)
    d1, d2 = pred_cx[mask], gt_cx[mask]
    piston = np.angle(np.sum(d1 * np.conj(d2)))
    d1c    = d1 * np.exp(-1j * piston)
    num    = np.abs(np.sum(d1c * np.conj(d2)))
    den    = np.sqrt(np.sum(np.abs(d1c)**2) * np.sum(np.abs(d2)**2))
    return float(num / (den + 1e-12))

def phase_gradient_ssim(pred_cx, gt_cx, percentile=50):
    mask     = np.abs(gt_cx) > np.percentile(np.abs(gt_cx), percentile)
    phi_pred = np.angle(pred_cx)
    phi_gt   = np.angle(gt_cx)
    dpz_pred = np.angle(np.exp(1j * np.diff(phi_pred, axis=0, append=phi_pred[-1:])))
    dpx_pred = np.angle(np.exp(1j * np.diff(phi_pred, axis=1, append=phi_pred[:, -1:])))
    dpz_gt   = np.angle(np.exp(1j * np.diff(phi_gt,   axis=0, append=phi_gt[-1:])))
    dpx_gt   = np.angle(np.exp(1j * np.diff(phi_gt,   axis=1, append=phi_gt[:, -1:])))
    for arr in [dpz_pred, dpx_pred, dpz_gt, dpx_gt]:
        arr[~mask] = 0.0
    sz = ssim(dpz_pred, dpz_gt, data_range=2*np.pi)
    sx = ssim(dpx_pred, dpx_gt, data_range=2*np.pi)
    return float((sz + sx) / 2.0), float(sz), float(sx)

# Phase coherence stratified by amplitude decile of the ground truth.
# Reason: 68.5 % of the WPC weight was measured to fall on the 1 % brightest pixels (dynamic range
# of 99 dB). WPC does not describe the coherence of the tomogram but that of a handful of pixels.
# The per-decile curve does not depend on any weighting and shows WHERE each method works.
N_AMPLITUDE_DECILES = 10
DECILE_COLUMNS = ['coh_d%d' % i for i in range(1, N_AMPLITUDE_DECILES + 1)]


def coherence_by_amplitude_decile(pred_cx, gt_cx, n_bins=N_AMPLITUDE_DECILES):
    """
    Mean phase coherence cos(phi_pred - phi_gt) inside each decile of |GT|.

    It complements WPC without replacing it. WPC collapses everything into a scalar with a weighting
    that, with 99 dB of dynamic range, puts 68.5 % of the weight on 1 % of the pixels: a method can
    win WPC by being right only on the brightest layers. This curve is UNWEIGHTED inside each decile,
    so it exposes the full profile: decile 1 = background/noise (where the phase is physically
    undefined and every method should give ~0), decile 10 = strongest signal.

    The global piston is not removed, as in weighted_phase_coherence(), so that the curve and the
    scalar WPC measure the same quantity and can be compared.
    """
    amp = np.abs(gt_cx).ravel()
    cosd = np.cos(np.angle(pred_cx).ravel() - np.angle(gt_cx).ravel())
    edges = np.percentile(amp, np.linspace(0, 100, n_bins + 1))
    out = {}
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (amp >= lo) & (amp <= hi) if i == n_bins - 1 else (amp >= lo) & (amp < hi)
        out['coh_d%d' % (i + 1)] = float(np.mean(cosd[m])) if m.any() else np.nan
    return out


def mirror_of(arr):
    """
    Mirror row along the depth axis: out[z] = arr[(-z) mod Z].
    With this, mirrorArtifact(S) == (S + conj(mirror_of(S))) / 2 exactly (verified to 3e-16).
    """
    return np.roll(np.flip(arr, axis=0), 1, axis=0)


def decompose_mirror(target_cx):
    """
    Splits the corrupted B-scan into its two physical components:
      true_term   = S(z)/2            -- the true tissue
      mirror_term = conj(S(-z))/2     -- the conjugate copy folded at the zero delay
    Their sum is exactly the input. Both are derived from the ground truth, so this only exists
    in evaluation (it is not information available at inference).
    """
    true_term = target_cx / 2.0
    mirror_term = np.conj(mirror_of(target_cx)) / 2.0
    return true_term, mirror_term


def separability_stats(target_cx):
    """
    Separability map. Each pixel observes the SUM of the tissue and its mirror, so the complex
    value of each one is only determined where one clearly dominates the other. The fraction of
    separable pixels bounds how much of the speckle of the ground truth is recoverable IN PRINCIPLE,
    and therefore which unfiltered SSIM is achievable. It is a property of the DATA, not the model.
    """
    true_term, mirror_term = decompose_mirror(target_cx)
    a, b = np.abs(true_term), np.abs(mirror_term)
    floor = np.percentile(np.abs(target_cx), NOISE_FLOOR_PERCENTILE)
    valid = np.abs(target_cx) > floor
    ratio = a / (b + 1e-12)
    out = {}
    for t in SEPARABILITY_CONTRAST_THRESHOLDS:
        sep = (ratio > t) | (ratio < 1.0 / t)
        out['separable_frac_x%g' % t] = float(np.mean(sep[valid])) if valid.any() else np.nan
    out['mirror_dominant_frac'] = float(np.mean((ratio < 1.0 / MIRROR_DOMINANT_CONTRAST)[valid])) if valid.any() else np.nan
    out['median_contrast_db'] = float(20 * np.log10(np.median(ratio[valid]) + 1e-12)) if valid.any() else np.nan
    out['mirror_overlap_corr'] = mirror_overlap_severity(target_cx)
    return out


def mirror_region_mask(target_cx):
    """Pixels above the noise floor where the mirror term dominates the true tissue."""
    true_term, mirror_term = decompose_mirror(target_cx)
    floor = np.percentile(np.abs(target_cx), NOISE_FLOOR_PERCENTILE)
    dominant = np.abs(mirror_term) > MIRROR_DOMINANT_CONTRAST * np.abs(true_term)
    return dominant & (np.abs(mirror_of(target_cx)) > floor)



def mirror_residual_metrics(pred_cx, gt_cx, input_cx, mask):
    """
    Artifact suppression, measured DIRECTLY on the region dominated by the mirror. SSIM/PSNR/WPC
    are global measures dominated by the bulk tissue, which input and ground truth share, so they
    barely respond to whether the artifact disappeared.

      mirror_suppression = (E_input - E_pred) / (E_input - E_gt)
        ~1.0  the residual dropped to the level of the ground truth -> artifact removed
        ~0.0  the residual stays at the level of the input          -> artifact intact (identity copy)
        <0    the prediction put MORE energy there than the input
      mirror_residual_db = 10*log10(E_pred / E_input): raw residual, not normalized by the ground truth.
    """
    if mask is None or not mask.any():
        return dict(mirror_suppression=np.nan, mirror_residual_db=np.nan, mirror_mask_frac=0.0)
    e_pred = float(np.sum(np.abs(pred_cx[mask]) ** 2))
    e_gt = float(np.sum(np.abs(gt_cx[mask]) ** 2))
    e_in = float(np.sum(np.abs(input_cx[mask]) ** 2))
    denom = e_in - e_gt
    supp = (e_in - e_pred) / denom if abs(denom) > 1e-20 else np.nan
    return dict(
        mirror_suppression=float(supp),
        mirror_residual_db=float(10 * np.log10((e_pred + 1e-30) / (e_in + 1e-30))),
        mirror_mask_frac=float(np.mean(mask)),
    )


def despeckle_nlm(image, h=0.10, patch_size=5, patch_distance=6):
    return denoise_nl_means(image, h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=True)

def bootstrap_ci_by_volume(df_subset, metric_col, n_boot=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED):
    """
    95% percentile bootstrap resampling whole volumes (blocks of ~N_BSCANS_PER_VOLUME consecutive
    B-scans sharing a tomogram_file) instead of individual B-scans. With only a few independent
    volumes, resampling per B-scan would treat within-volume B-scans as independent and understate
    the true interval, so each bootstrap draw resamples (with replacement) the per-volume means.
    """
    rng = np.random.default_rng(seed)
    volume_means = df_subset.groupby('tomogram_file')[metric_col].mean().values
    volume_means = np.asarray(volume_means, dtype=np.float64)
    volume_means = volume_means[np.isfinite(volume_means)]
    if len(volume_means) == 0:
        return np.nan, np.nan
    n = len(volume_means)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(volume_means, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    lower = np.percentile(boot_means, 2.5)
    upper = np.percentile(boot_means, 97.5)
    return float(lower), float(upper)


def holm_bonferroni(pvalues):
    """
    Manual Holm-Bonferroni step-down correction.
    Sort p-values ascending, adjust each by (n - rank), enforce monotonicity via running max.
    """
    pvalues = np.asarray(pvalues, dtype=np.float64)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = pvalues[idx] * (n - rank)
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def mirror_overlap_severity(input_cx):
    """
    Proxy for how severely the mirror artifact overlaps the true signal in a given B-scan: Pearson
    correlation between the (dB) intensity image and its own depth-axis (axis=0) flip. The mirror
    term is, by construction, a duplicate of the real signal folded across depth zero, so a higher
    correlation here means the artifact sits closer on top of real structure (a harder case) rather
    than in an emptier region of the image (an easier one).
    """
    intensity = 20 * np.log10(np.abs(input_cx) + 1e-12)
    flipped = np.flip(intensity, axis=0)
    a = intensity.ravel() - intensity.mean()
    b = flipped.ravel() - flipped.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

# ===================== END OF COPIED CODE ===================== #


def _despeckle_variant_name(h):
    return 'nlm_none' if h <= 0.0 else f'nlm_h{int(round(h * 100)):03d}'


DESPECKLE_VARIANTS = {
    _despeckle_variant_name(h): (lambda img, h=h: img if h <= 0.0 else despeckle_nlm(img, h=h))
    for h in DESPECKLE_H_SWEEP
}


# ===================== NEW METRICS ===================== #

def class_masks(gt_cx):
    """Tissue-dominant pixels (recoverable in principle) and ambiguous zone, above the noise floor."""
    t = np.abs(gt_cx)
    m = np.abs(mirror_of(gt_cx))
    valid = t > np.percentile(t, NOISE_FLOOR_PERCENTILE)
    ratio = t / (m + 1e-30)
    tissue = valid & (ratio > CLASS_CONTRAST)
    ambiguous = valid & (ratio <= CLASS_CONTRAST) & (ratio >= 1.0 / CLASS_CONTRAST)
    return tissue, ambiguous


def coherence_on_mask(pred_cx, gt_cx, mask, min_pixels=50):
    """Same definition as masked_complex_coherence (piston removed), on a given mask."""
    if mask.sum() < min_pixels:
        return np.nan
    d1, d2 = pred_cx[mask], gt_cx[mask]
    piston = np.angle(np.sum(d1 * np.conj(d2)))
    d1c = d1 * np.exp(-1j * piston)
    num = np.abs(np.sum(d1c * np.conj(d2)))
    den = np.sqrt(np.sum(np.abs(d1c) ** 2) * np.sum(np.abs(d2) ** 2))
    return float(num / (den + 1e-12))


def nrmse_on_mask(pred_cx, gt_cx, mask, min_pixels=50):
    if mask.sum() < min_pixels:
        return np.nan
    return float(np.linalg.norm(pred_cx[mask] - gt_cx[mask]) / (np.linalg.norm(gt_cx[mask]) + 1e-30))


def roi_metrics(task):
    """
    All metrics of one ROI. Same intensity protocol as the PC-CGAN evaluator (dB normalized with
    the range of the ground truth, NLM sweep), but with the three images already in a common
    physical scale.
      task = (pred_cx, gt_cx, input_cx, identity_input_cx, meta)
    identity_input_cx is the input the method ACTUALLY received (for DEFR, the dispersion-encoded
    one); input_cx is the standard measurement and sets the scale of the suppression.
    """
    pred_cx, gt_cx, input_cx, ident_cx, meta = task
    pred_cx = pred_cx.astype(np.complex128)
    gt_cx = gt_cx.astype(np.complex128)
    input_cx = input_cx.astype(np.complex128)
    ident_cx = ident_cx.astype(np.complex128)
    try:
        intensity_pred = 20 * np.log10(np.abs(pred_cx) + 1e-12)
        intensity_gt = 20 * np.log10(np.abs(gt_cx) + 1e-12)
        intensity_input = 20 * np.log10(np.abs(ident_cx) + 1e-12)
        gt_min, gt_max = intensity_gt.min(), intensity_gt.max()
        norm_pred = np.clip((intensity_pred - gt_min) / (gt_max - gt_min), 0, 1)
        norm_gt = np.clip((intensity_gt - gt_min) / (gt_max - gt_min), 0, 1)
        norm_input = np.clip((intensity_input - gt_min) / (gt_max - gt_min), 0, 1)

        wpc_val = weighted_phase_coherence(pred_cx, gt_cx)
        ccc_val = masked_complex_coherence(pred_cx, gt_cx)
        pgssim_val, pgssim_axial, pgssim_lateral = phase_gradient_ssim(pred_cx, gt_cx)
        mirror_stats = mirror_residual_metrics(pred_cx, gt_cx, input_cx, mirror_region_mask(gt_cx))
        decile_stats = coherence_by_amplitude_decile(pred_cx, gt_cx)
        tissue, amb = class_masks(gt_cx)
        extra = dict(
            ccc_vs_mirror_solution=masked_complex_coherence(pred_cx, np.conj(mirror_of(gt_cx))),
            dc_residual=float(np.linalg.norm(mirrorArtifact(pred_cx) - input_cx) /
                              (np.linalg.norm(input_cx) + 1e-30)),
            ccc_tissue_dom=coherence_on_mask(pred_cx, gt_cx, tissue),
            ccc_ambiguous=coherence_on_mask(pred_cx, gt_cx, amb),
            nrmse_tissue_dom=nrmse_on_mask(pred_cx, gt_cx, tissue),
            frac_tissue_dom=float(tissue.mean()), frac_ambiguous=float(amb.mean()),
        )
        rows = []
        for filter_name, filter_fn in DESPECKLE_VARIANTS.items():
            fp, fg, fi = filter_fn(norm_pred), filter_fn(norm_gt), filter_fn(norm_input)
            rows.append(dict(
                **meta, despeckle_filter=filter_name,
                ssim=calculate_ssim(fp, fg), psnr=calculate_psnr(fp, fg), mse=calculate_mse(fp, fg),
                hist_cosine_similarity=histogram_difference(fp, fg, method='cosine-similarity'),
                identity_collapse_score=calculate_ssim(fp, fi),
                wpc=wpc_val, ccc=ccc_val, pgssim=pgssim_val,
                pgssim_axial=pgssim_axial, pgssim_lateral=pgssim_lateral,
                **mirror_stats, **decile_stats, **extra))
        return rows
    except Exception as ex:
        return [dict(**meta, despeckle_filter='ERROR', error=f'{ex}\n{traceback.format_exc(limit=3)}')]


def compute_metrics_parallel(preds, samples, meta_common, ident_inputs=None, jobs=None):
    """preds (N,r,r) aligned with samples['gt'/'y']. Returns a DataFrame of rows."""
    jobs = jobs or N_JOBS
    tasks = []
    for i, m in enumerate(samples['meta']):
        tasks.append((preds[i], samples['gt'][i], samples['y'][i],
                      samples['y'][i] if ident_inputs is None else ident_inputs[i],
                      dict(**meta_common, **m)))
    rows = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context('fork')) as ex:
            for r in ex.map(roi_metrics, tasks, chunksize=8):
                rows.extend(r)
    else:
        for t in tasks:
            rows.extend(roi_metrics(t))
    df = pd.DataFrame(rows)
    if 'error' in df.columns:
        bad = df[df['despeckle_filter'] == 'ERROR']
        for _, r in bad.iterrows():
            log_error(f"[{r.get('method')}] ROI {r.get('tomogram_file')}/{r.get('bscan_index')}: {r['error']}")
        df = df[df['despeckle_filter'] != 'ERROR'].drop(columns=['error'])
    return df


# ===================== DATA ===================== #

def select_bscan_indices(num_bscans):
    if num_bscans <= N_BSCANS_PER_VOLUME:
        return np.arange(num_bscans)
    return np.unique(np.linspace(0, num_bscans - 1, N_BSCANS_PER_VOLUME, dtype=int))


def roi_origins(shape, roi=None):
    roi = ROI_SIZE if roi is None else roi
    Z, X = shape
    return [(z0, x0) for z0 in range(0, Z - roi + 1, roi) for x0 in range(0, X - roi + 1, roi)]


def build_roi_samples():
    """
    All evaluation ROIs. The artifact is generated ON each ROI (double precision).
    Visual panel: the central B-scan among the selected ones of each volume (the same for every
    method).
    """
    gts, ys, meta, fingerprint = [], [], [], []
    files = sorted(glob.glob(os.path.join(VALIDATION_DATA_DIR, '*.npy')))
    for fp in files:
        fn = os.path.basename(fp)
        raw = np.load(fp, mmap_mode='r')
        fingerprint.append((fn, tuple(raw.shape), os.path.getsize(fp)))
        idx = select_bscan_indices(raw.shape[2])
        panel_b = int(idx[len(idx) // 2])
        for bi in idx:
            b = np.asarray(raw[:, :, bi])
            bs = (b[..., 0] + 1j * b[..., 1]) if raw.ndim == 4 else b
            for (z0, x0) in roi_origins(bs.shape, ROI_SIZE):
                g = np.asarray(bs[z0:z0 + ROI_SIZE, x0:x0 + ROI_SIZE], dtype=np.complex128)
                gts.append(g.astype(np.complex64))
                ys.append(mirrorArtifact(g).astype(np.complex64))
                meta.append(dict(tomogram_file=fn, bscan_index=int(bi), z0=int(z0), x0=int(x0),
                                 panel=bool(int(bi) == panel_b)))
        print(f'  {fn}: {raw.shape} -> {len(idx)} B-scans, {len(roi_origins(raw.shape[:2], ROI_SIZE))} ROIs/B-scan')
    samples = dict(gt=np.stack(gts), y=np.stack(ys), meta=meta)
    print(f'Total: {len(meta)} ROIs of {ROI_SIZE}x{ROI_SIZE} '
          f'({sum(m["panel"] for m in meta)} in the visual panel)')
    return samples, fingerprint


def data_key(fingerprint, extra):
    s = json.dumps(dict(fp=fingerprint, roi=ROI_SIZE, nb=N_BSCANS_PER_VOLUME, sweep=DESPECKLE_H_SWEEP,
                        metric_version=METRIC_VERSION, **extra), sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:10]


# ===================== BASELINES ===================== #

def _to_fr(S):
    return ifftshift(ifft(ifftshift(S, axes=-2), axis=-2), axes=-2)


def _from_fr(f):
    return fftshift(fft(fftshift(f, axes=-2), axis=-2), axes=-2)


def hilbert_half(y, keep):
    """
    Analytic signal of the real fringe along k: 2x one depth half-space and 1x the self-paired rows
    (zero delay c and row 0). Identical to scipy.signal.hilbert on the fringes, but expressed in
    depth. keep = 'lower' (rows > c) or 'upper' (< c).
    """
    N = y.shape[-2]
    c = N // 2
    w = np.zeros(N)
    if keep == 'lower':
        w[c + 1:] = 2.0
    else:
        w[1:c] = 2.0
    w[c] = 1.0
    w[0] = 1.0
    return y * w[:, None]


def defr_advanced_batch(P, phi, max_iter=DEFR_MAX_ITER, threshold_db=DEFR_THRESH_DB):
    """
    Advanced DEFR vectorized over columns. P: (N, C) real.
    Same step-by-step logic as the per-A-line version (maximum difference 1.8e-16): at each
    iteration every active A-line takes its peak from c1 (compensated signal) or from c2[:N/2]
    (uncompensated), accumulates it in d1/d2 and subtracts its contribution in both channels. An
    A-line is deactivated when its peak falls below the threshold, like the break of the original.
    At the end d1 += c1.
    """
    N, C = P.shape
    half = N // 2
    p1p = ifft(np.exp(1j * phi)); p1n = ifft(np.exp(-1j * phi)); p2n = ifft(np.exp(-2j * phi))
    c1 = ifft(P * np.exp(-1j * phi)[:, None], axis=0)
    c2 = ifft(P, axis=0)
    d1 = np.zeros((N, C), dtype=complex)
    d2 = np.zeros((N, C), dtype=complex)
    T = np.maximum(np.abs(c1).max(0), np.abs(c2[:half]).max(0)) * 10 ** (-threshold_db / 20.0)
    n = np.arange(N)[:, None]
    cols = np.arange(C)
    active = np.ones(C, dtype=bool)
    for _ in range(max_iter):
        a1 = np.abs(c1); n1 = a1.argmax(0); val1 = a1[n1, cols]
        a2 = np.abs(c2[:half]); n2 = a2.argmax(0); val2 = a2[n2, cols]
        active &= ~(np.maximum(val1, val2) < T)
        if not active.any():
            break
        u1 = active & (val1 > val2)
        u2 = active & ~(val1 > val2)
        A = np.where(u1, c1[n1, cols], 0)
        d1[n1[u1], cols[u1]] += A[u1]
        c1[n1[u1], cols[u1]] = 0.0
        c1 -= np.conj(A) * p2n[(n + n1) % N]                          # roll(p2n, -n1)
        c2 -= np.where(n1 <= half, A * p1p[(n - n1) % N],               # roll(p1p,  n1)
                       np.conj(A) * p1n[(n + n1) % N])                  # roll(p1n, -n1)
        B = np.where(u2, c2[n2, cols], 0)
        d2[n2[u2], cols[u2]] += B[u2]
        c2[n2[u2], cols[u2]] = 0.0
        c1 -= B * p1n[(n - n2) % N] + np.conj(B) * p1n[(n + n2) % N]    # roll(p1n, n2), roll(p1n, -n2)
    d1 += c1
    return d1, d2


def defr_batch(gt_batch, beta2=DEFR_BETA2, max_iter=DEFR_MAX_ITER, threshold_db=DEFR_THRESH_DB):
    """
    Batch of ROIs (n, Z, X) in the convention of the pipeline (zero delay at Z/2) ->
    (physical DEFR estimate, depth-encoded measurement), both (n, Z, X).
    """
    n, Z, X = gt_batch.shape
    b = np.roll(gt_batch.astype(np.complex128), -(Z // 2), axis=1)         # zero delay -> index 0
    cols = b.transpose(1, 0, 2).reshape(Z, n * X)                          # (Z, n*X): one column per A-line
    k = np.linspace(0, 1, Z)
    phi = beta2 * k ** 2
    phi -= phi[Z // 2]
    P = np.real(fft(cols, axis=0) * np.exp(1j * phi)[:, None])             # encoded real fringe
    D1, _ = defr_advanced_batch(P, phi, max_iter, threshold_db)
    est = 2.0 * np.roll(D1.reshape(Z, n, X).transpose(1, 0, 2), Z // 2, axis=1)
    y_d = np.roll(ifft(P, axis=0).reshape(Z, n, X).transpose(1, 0, 2), Z // 2, axis=1)
    return est.astype(np.complex64), y_d.astype(np.complex64)


def oracle_wiener(gt, y, sigma):
    """
    BOUND of the Wiener head (ORACLE, uses the ground truth). In the folded domain, with L and U' the
    two sources of each pixel and R_L, R_U their local powers (|.|^2 smoothed with sigma px;
    sigma = 0 = instantaneous powers, the best possible real mask pixel by pixel):
        g = (R_L - R_U) / (R_L + R_U),   L = Y (1 + g),   U' = Y (1 - g).
    It is the best any method that only splits the measurement with a real gain can achieve.
    """
    from scipy.ndimage import gaussian_filter
    g128, y128 = gt.astype(np.complex128), y.astype(np.complex128)
    N = g128.shape[-2]
    c = N // 2
    L = g128[..., c:, :]
    Up = np.conj(np.flip(g128[..., 1:c + 1, :], axis=-2))
    Y = y128[..., c:, :]
    PL, PU = np.abs(L) ** 2, np.abs(Up) ** 2
    if sigma > 0:
        sg = (0, sigma, sigma) if PL.ndim == 3 else (sigma, sigma)
        PL, PU = gaussian_filter(PL, sg, mode='reflect'), gaussian_filter(PU, sg, mode='reflect')
    gain = (PL - PU) / (PL + PU + 1e-300)
    Lh, Uh = Y * (1 + gain), Y * (1 - gain)
    out = np.concatenate([y128[..., :1, :], np.flip(np.conj(Uh[..., 1:, :]), axis=-2),
                          (Lh[..., :1, :] + np.conj(Uh[..., :1, :])) / 2, Lh[..., 1:, :]], axis=-2)
    return out.astype(np.complex64)


WIENER_ORACLE_SIGMA = 3.0


def _baseline_specs():
    """(name, parameters for the cache key). Each baseline has its own cache."""
    return [
        (NOOP_BASELINE_NAME, {}),
        ('HILBERT_KeepLower', {}),
        ('HILBERT_KeepUpper', {}),
        ('HILBERT_OracleHalf', {}),
        ('DEFR_DispEnc', dict(beta2=DEFR_BETA2, it=DEFR_MAX_ITER, thr=DEFR_THRESH_DB, impl='defr_advanced+conv_fix')),
        ('ORACLE_WienerLocal', dict(sigma=WIENER_ORACLE_SIGMA)),
        ('ORACLE_RatioMask', dict(sigma=0.0)),
    ]


def _compute_baseline(name, samples):
    """-> (prediction, input_type, oracle, input for the identity-collapse score or None, extra panel or None)"""
    y, gt = samples['y'], samples['gt']
    if name == NOOP_BASELINE_NAME:
        return y, 'standard', 'none', None, None
    if name == 'HILBERT_KeepLower':
        return hilbert_half(y, 'lower'), 'standard', 'none', None, None
    if name == 'HILBERT_KeepUpper':
        return hilbert_half(y, 'upper'), 'standard', 'none', None, None
    if name == 'HILBERT_OracleHalf':
        hl, hu = hilbert_half(y, 'lower'), hilbert_half(y, 'upper')
        pick = np.array([masked_complex_coherence(hl[i].astype(np.complex128), gt[i].astype(np.complex128)) >=
                         masked_complex_coherence(hu[i].astype(np.complex128), gt[i].astype(np.complex128))
                         for i in range(len(gt))])
        print(f'  Hilbert oracle: lower half-space chosen in {100 * pick.mean():.0f}% of the ROIs')
        return np.where(pick[:, None, None], hl, hu), 'standard', 'half_selection_from_gt', None, None
    if name == 'DEFR_DispEnc':
        est = np.zeros_like(gt); yd = np.zeros_like(gt)
        for i0 in range(0, len(gt), DEFR_BATCH):
            est[i0:i0 + DEFR_BATCH], yd[i0:i0 + DEFR_BATCH] = defr_batch(gt[i0:i0 + DEFR_BATCH])
            print(f'    DEFR {min(i0 + DEFR_BATCH, len(gt))}/{len(gt)}', end='\r')
        print()
        return est, 'dispersion_encoded', 'none', yd, yd
    if name == 'ORACLE_WienerLocal':
        return oracle_wiener(gt, y, WIENER_ORACLE_SIGMA), 'standard', 'source_powers_from_gt', None, None
    if name == 'ORACLE_RatioMask':
        return oracle_wiener(gt, y, 0.0), 'standard', 'source_powers_from_gt', None, None
    raise ValueError(name)


def build_baselines(samples, fingerprint, force=False):
    """Each baseline is computed ONCE and kept in its own cache: adding one does not recompute the others."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    panel_idx = np.array([i for i, m in enumerate(samples['meta']) if m['panel']])
    dfs, panel = [], {}
    for name, params in _baseline_specs():
        key = data_key(fingerprint, dict(baseline=name, **params))
        csv_path = os.path.join(CACHE_DIR, f'baseline_{name}_{key}.csv.gz')
        pnl_path = os.path.join(CACHE_DIR, f'baseline_{name}_{key}_panel.npz')
        if os.path.exists(csv_path) and os.path.exists(pnl_path) and not force:
            dfs.append(pd.read_csv(csv_path))
            pz = np.load(pnl_path)
            panel.update({k: pz[k] for k in pz.files})
            print(f'  {name}: cache')
            continue
        print(f'  {name}: computing...')
        pred, itype, oracle, ident, extra = _compute_baseline(name, samples)
        common = dict(method=name, family='baseline', run='', checkpoint='', variant='', epoch=-1,
                      input_type=itype, oracle=oracle)
        df = compute_metrics_parallel(pred, samples, common, ident_inputs=ident)
        df.to_csv(csv_path, index=False)
        pn = {name: pred[panel_idx]}
        if extra is not None:
            pn[name + '__input'] = extra[panel_idx]
        np.savez_compressed(pnl_path, **pn)
        dfs.append(df); panel.update(pn)
    return pd.concat(dfs, ignore_index=True), panel


# ===================== TRAINED RUNS (diffComplexD.py, both families) ===================== #

CHECKPOINTS = (('best', 'ema_best.pth'), ('final', 'ema_final.pth'))


def discover_runs(requested=None):
    names = requested or RUN_NAMES or sorted(d for d in os.listdir(DIFF_ROOT)
                                             if os.path.isdir(os.path.join(DIFF_ROOT, d)))
    runs = []
    for n in names:
        d = os.path.join(DIFF_ROOT, n)
        if any(os.path.exists(os.path.join(d, fn)) for _, fn in CHECKPOINTS):
            runs.append((n, d))
        elif requested:
            log_error(f'[{n}] no ema_best.pth nor ema_final.pth (training not finished); skipped.')
    return runs


def evaluate_run(run_name, run_dir, samples, fingerprint, force=False):
    """
    Evaluates the two checkpoints of each run: 'best' (selected with the internal validation) and
    'final' (last). roi-family diffusion: every SAMPLING_VARIANTS. roi-family regressor: one pass
    'R_direct'. bscan family: diffComplexD.bscan_eval_variants. Method = '<run>/<checkpoint>/<variant>'.
    """
    import torch
    import diffComplexD as dd
    panel_idx = np.array([i for i, m in enumerate(samples['meta']) if m['panel']])
    dfs, panel = [], {}
    for ck_name, fn in CHECKPOINTS:
        path = os.path.join(run_dir, fn)
        if not os.path.exists(path):
            continue
        try:
            model, arch, cfg, epoch = dd.load_trained_path(path)
        except Exception as ex:
            print(f'  [{run_name}/{ck_name}] could not be loaded ({ex}); skipped.')
            continue
        if int(arch.get('roi_size', -1)) != ROI_SIZE:
            print(f'  [{run_name}] trained at ROI {arch.get("roi_size")} != {ROI_SIZE}; skipped.')
            return [], {}
        if dd.arch_family(arch) == 'bscan':
            variants = {k: dict(variant=v, n_samples=n) for k, (v, n) in dd.bscan_eval_variants(arch).items()}
            fam_label = f'bscan-{arch["kind"]}'
            is_reg = arch['kind'] in ('mask', 'null')
        else:
            is_reg = dd.arch_get(arch, 'model') == 'regress'
            variants = {'R_direct': dict(variant='R_direct', n_samples=1)} if is_reg else SAMPLING_VARIANTS
            fam_label = 'regress' if is_reg else 'diffusion'
        mtime = os.path.getmtime(path)
        for vname, vcfg in variants.items():
            key = data_key(fingerprint, dict(run=run_name, ck=ck_name, ep=epoch, mt=mtime, var=vcfg,
                                             steps=SAMPLING_STEPS, seed=SAMPLE_SEED))
            tag = f'{run_name}_{ck_name}_{vname}_{key}'
            csv_path = os.path.join(CACHE_DIR, f'run_{tag}.csv.gz')
            pnl_path = os.path.join(CACHE_DIR, f'run_{tag}_panel.npy')
            method = f'{run_name}/{ck_name}/{vname}'
            if os.path.exists(csv_path) and os.path.exists(pnl_path) and not force:
                print(f'  {method}: cache')
                dfs.append(pd.read_csv(csv_path)); panel[method] = np.load(pnl_path)
                continue
            print(f'  {method} (epoch {epoch}): {"one pass" if is_reg else f"{SAMPLING_STEPS} DDIM steps"} '
                  f'on {len(samples["y"])} ROIs...')
            pred = dd.reconstruct_rois(model, samples['y'], cfg, vcfg['variant'], steps=SAMPLING_STEPS,
                                       batch=SAMPLE_BATCH, seed=SAMPLE_SEED, n_samples=vcfg['n_samples'])
            if SAVE_PREDICTIONS:
                os.makedirs(os.path.join(PRED_DIR, run_name), exist_ok=True)
                np.save(os.path.join(PRED_DIR, run_name, f'{ck_name}_{vname}.npy'), pred)
            common = dict(method=method, family=fam_label, run=run_name,
                          checkpoint=ck_name, variant=vname, epoch=epoch, input_type='standard', oracle='none')
            df = compute_metrics_parallel(pred, samples, common)
            df.to_csv(csv_path, index=False)
            np.save(pnl_path, pred[panel_idx])
            dfs.append(df); panel[method] = pred[panel_idx]
        del model
        torch.cuda.empty_cache()
    return dfs, panel


# ===================== SUMMARIES ===================== #

def summarize(df):
    rows = []
    for (method, filt), g in df.groupby(['method', 'despeckle_filter']):
        first = g.iloc[0]
        row = dict(method=method, despeckle_filter=filt, family=first['family'],
                   input_type=first['input_type'], oracle=first['oracle'], n_rois=len(g))
        for m in METRIC_COLUMNS:
            v = g[m].values.astype(float)
            row[f'{m}_mean'] = float(np.nanmean(v))
            row[f'{m}_median'] = float(np.nanmedian(v))
            row[f'{m}_std'] = float(np.nanstd(v))
            lo, hi = bootstrap_ci_by_volume(g, m)
            row[f'{m}_ci95_lower'], row[f'{m}_ci95_upper'] = lo, hi
        row['mirror_suppression_frac_ok'] = float(np.nanmean(g['mirror_suppression'].values > MIRROR_SUPPRESSION_MIN))
        row['identity_collapse_score_mean'] = float(np.nanmean(g['identity_collapse_score']))
        rows.append(row)
    s = pd.DataFrame(rows)
    noop = s[s['method'] == NOOP_BASELINE_NAME].set_index('despeckle_filter')
    for m in ('ssim', 'ccc', 'ccc_tissue_dom'):
        s[f'{m}_gain_over_noop'] = s[f'{m}_mean'] - s['despeckle_filter'].map(noop[f'{m}_mean'])
    gate = s['despeckle_filter'] == IDENTITY_COLLAPSE_GATE_FILTER
    s['SUSPECTED_IDENTITY_COLLAPSE'] = False
    s.loc[gate, 'SUSPECTED_IDENTITY_COLLAPSE'] = (
        (s.loc[gate, 'identity_collapse_score_mean'] > IDENTITY_COLLAPSE_THRESHOLD)
        | (s.loc[gate, 'mirror_suppression_median'] < MIRROR_SUPPRESSION_MIN))
    s.loc[s['method'] == NOOP_BASELINE_NAME, 'SUSPECTED_IDENTITY_COLLAPSE'] = False
    return s


def per_volume_table(df, filt='nlm_none'):
    keys = ['mirror_suppression', 'ccc', 'ccc_tissue_dom', 'ccc_ambiguous', 'ccc_vs_mirror_solution',
            'dc_residual', 'ssim', 'wpc']
    return (df[df['despeckle_filter'] == filt].groupby(['method', 'tomogram_file'])[keys]
            .median().reset_index())


def metric_task_correlation(summary, filt='nlm_none'):
    """Rank correlation of each metric with artifact suppression (which metrics track the task)."""
    s = summary[summary['despeckle_filter'] == filt]
    rows = []
    for m in METRIC_COLUMNS:
        if m == 'mirror_suppression':
            continue
        col = f'{m}_median' if m in ROBUST_MEDIAN_METRICS else f'{m}_mean'
        a, b = s[col].values, s['mirror_suppression_median'].values
        ok = np.isfinite(a) & np.isfinite(b)
        rho, p = spearmanr(a[ok], b[ok]) if ok.sum() >= 4 else (np.nan, np.nan)
        rows.append(dict(metric=m, spearman_rho_vs_suppression=rho, p_value=p, n_methods=int(ok.sum())))
    return pd.DataFrame(rows)


def wilcoxon_vs_noop(df, filt='nlm_none', metrics=('mirror_suppression', 'ccc', 'ccc_tissue_dom',
                                                     'ccc_ambiguous', 'ssim')):
    d = df[df['despeckle_filter'] == filt]
    ref = d[d['method'] == NOOP_BASELINE_NAME].set_index(KEY_COLS)
    rows = []
    for method, g in d.groupby('method'):
        if method == NOOP_BASELINE_NAME:
            continue
        g = g.set_index(KEY_COLS)
        common = g.index.intersection(ref.index)
        for m in metrics:
            a, b = g.loc[common, m].values.astype(float), ref.loc[common, m].values.astype(float)
            ok = np.isfinite(a) & np.isfinite(b)
            try:
                stat, p = wilcoxon(a[ok], b[ok])
            except ValueError:
                stat, p = np.nan, 1.0
            rows.append(dict(method=method, metric=m, n=int(ok.sum()), median_diff=float(np.median(a[ok] - b[ok]))
                             if ok.any() else np.nan, statistic=stat, p_raw=p))
    out = pd.DataFrame(rows)
    if not out.empty:
        out['p_holm'] = holm_bonferroni(out['p_raw'].fillna(1.0).values)
    return out


# ===================== PLOTS ===================== #

def _db(a):
    return 20 * np.log10(np.abs(a) + 1e-12)


def _assemble(rois, metas):
    Z = max(m['z0'] for m in metas) + ROI_SIZE
    X = max(m['x0'] for m in metas) + ROI_SIZE
    out = np.zeros((Z, X), dtype=np.complex64)
    for r, m in zip(rois, metas):
        out[m['z0']:m['z0'] + ROI_SIZE, m['x0']:m['x0'] + ROI_SIZE] = r
    return out


def render_panel(samples, panel, methods, out_path, title):
    """Same B-scan for every method. One row per volume."""
    pidx = [i for i, m in enumerate(samples['meta']) if m['panel']]
    metas = [samples['meta'][i] for i in pidx]
    files = sorted(set(m['tomogram_file'] for m in metas))
    cols = ['Input', 'GT'] + methods
    fig, axs = plt.subplots(len(files), len(cols), figsize=(2.6 * len(cols), 2.6 * len(files)), squeeze=False)
    for r, fn in enumerate(files):
        sel = [k for k, m in enumerate(metas) if m['tomogram_file'] == fn]
        mm = [metas[k] for k in sel]
        gt = _assemble(samples['gt'][[pidx[k] for k in sel]], mm)
        imgs = [_assemble(samples['y'][[pidx[k] for k in sel]], mm), gt] + \
               [_assemble(panel[m][sel], mm) if m in panel else None for m in methods]
        gdb = _db(gt)
        vmax = np.percentile(gdb, 99.9)
        vmin = max(np.percentile(gdb, 1), vmax - 70)
        for c, im in enumerate(imgs):
            ax = axs[r, c]
            if im is not None:
                ax.imshow(_db(im), cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
            for m in mm:
                ax.axhline(m['z0'] - 0.5, color='y', lw=0.3); ax.axvline(m['x0'] - 0.5, color='y', lw=0.3)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c].replace('/', '\n'), fontsize=7)
        axs[r, 0].set_ylabel(f"{fn[:20]}\nb={mm[0]['bscan_index']}", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  Panel: {out_path}')


def plot_decile_curves(df, out_path, filt='nlm_none'):
    d = df[df['despeckle_filter'] == filt]
    cols = [c for c in d.columns if c.startswith('coh_d')]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, g in d.groupby('method'):
        ax.plot(range(1, len(cols) + 1), g[cols].mean().values, marker='o', ms=3, label=method)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('decile of |GT|  (1 = background, 10 = strongest signal)')
    ax.set_ylabel('mean cos(dphi), unweighted')
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_despeckle_curves(summary, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    order = {_despeckle_variant_name(h): h for h in DESPECKLE_H_SWEEP}
    for method, g in summary.groupby('method'):
        g = g.assign(h=g['despeckle_filter'].map(order)).sort_values('h')
        ax.plot(g['h'], g['ssim_mean'], marker='o', ms=3, label=method)
    ax.set_xlabel('NLM h (0 = unfiltered)'); ax.set_ylabel('mean SSIM')
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


# ===================== MAIN ===================== #

def main():
    global N_JOBS, ROI_SIZE, EVAL_DIR, CACHE_DIR, PLOTS_DIR, PRED_DIR, ERROR_LOG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None)
    ap.add_argument('--force', action='store_true', help='re-sample runs even if they are cached')
    ap.add_argument('--force-baselines', action='store_true')
    ap.add_argument('--baselines-only', action='store_true')
    ap.add_argument('--jobs', type=int, default=None)
    ap.add_argument('--roi', type=int, default=ROI_SIZE,
                    help='512 = full B-scan (bscan family, paper protocol); '
                         '256 = runs trained on 256 x 256 ROIs (roi family)')
    args = ap.parse_args()
    if args.jobs:
        N_JOBS = args.jobs
    ROI_SIZE = args.roi
    EVAL_DIR = os.path.join(DIFF_ROOT, f'evaluation_D_{ROI_SIZE}')
    CACHE_DIR = os.path.join(EVAL_DIR, 'cache')
    PLOTS_DIR = os.path.join(EVAL_DIR, 'plots')
    PRED_DIR = os.path.join(EVAL_DIR, 'predictions')
    ERROR_LOG_PATH = os.path.join(EVAL_DIR, 'evaluation_errors.log')
    print(f'ROI {ROI_SIZE} -> results in {EVAL_DIR}')
    for d in (EVAL_DIR, CACHE_DIR, PLOTS_DIR):
        os.makedirs(d, exist_ok=True)
    print(f'Processes for metrics: {N_JOBS}')

    samples, fingerprint = build_roi_samples()
    g0 = samples['gt'][0].astype(np.complex128)
    err = np.linalg.norm(mirrorArtifact(g0) - (g0 / 2 + np.conj(mirror_of(g0)) / 2)) / np.linalg.norm(mirrorArtifact(g0))
    print(f'  [self-test] forward-model identity on a real ROI: {err:.2e} '
          f'{"OK" if err < 1e-6 else "FAIL: check the z-axis convention"}')

    sep = pd.DataFrame([dict(**m, **separability_stats(samples['gt'][i].astype(np.complex128)))
                        for i, m in enumerate(samples['meta'])])
    sep.to_csv(os.path.join(EVAL_DIR, 'separability_per_roi.csv'), index=False)
    print('  Separability (fraction of pixels with contrast > t):',
          {t: round(float(sep[f'separable_frac_x{t:g}'].mean()), 3) for t in SEPARABILITY_CONTRAST_THRESHOLDS})

    base_df, panel = build_baselines(samples, fingerprint, force=args.force_baselines)
    all_dfs = [base_df]
    if not args.baselines_only:
        runs = discover_runs(args.runs)
        print(f'Runs to evaluate: {[r[0] for r in runs]}')
        for name, d in runs:
            dfs, pnl = evaluate_run(name, d, samples, fingerprint, force=args.force)
            all_dfs.extend(dfs); panel.update(pnl)
            if dfs:
                run_methods = [m for m in pnl if m.startswith(name + '/')]
                render_panel(samples, panel, PANEL_BASELINES + run_methods,
                             os.path.join(PLOTS_DIR, f'panel_{name}.png'),
                             f'{name}: same B-scan for every method (DEFR receives the encoded input)')

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(os.path.join(EVAL_DIR, 'per_roi_metrics.csv'), index=False)
    summary = summarize(df)
    summary.to_csv(os.path.join(EVAL_DIR, 'summary_metrics.csv'), index=False)
    per_volume_table(df).to_csv(os.path.join(EVAL_DIR, 'per_volume_summary.csv'), index=False)
    corr = metric_task_correlation(summary)
    corr.to_csv(os.path.join(EVAL_DIR, 'metric_task_correlation.csv'), index=False)
    wilcoxon_vs_noop(df).to_csv(os.path.join(EVAL_DIR, 'wilcoxon_vs_noop.csv'), index=False)
    plot_decile_curves(df, os.path.join(PLOTS_DIR, 'coherence_by_decile.png'))
    plot_despeckle_curves(summary, os.path.join(PLOTS_DIR, 'despeckle_ssim_curves.png'))
    primary = [m for m in panel if m.count('/') == 2 and m.split('/')[1] == 'best'
               and m.split('/')[2] in PANEL_PRIMARY_VARIANTS]
    render_panel(samples, panel, PANEL_BASELINES + primary,
                 os.path.join(PLOTS_DIR, 'panel_all_runs.png'),
                 'All runs (selected checkpoint) vs. baselines')

    s = summary[summary['despeckle_filter'] == 'nlm_none'].sort_values('mirror_suppression_median', ascending=False)
    print('\n' + '=' * 118)
    print(f'{"method":<34s}{"input":<11s}{"sup(med)":>9s}{"%ok":>6s}{"CCC":>7s}{"CCCtis":>8s}{"CCCamb":>8s}'
          f'{"CCC_S~":>8s}{"dc_res":>9s}{"SSIM":>7s}{"collapse":>9s}')
    for _, r in s.iterrows():
        col = summary[(summary['method'] == r['method']) & (summary['despeckle_filter'] == IDENTITY_COLLAPSE_GATE_FILTER)]
        flag = 'YES' if (not col.empty and bool(col['SUSPECTED_IDENTITY_COLLAPSE'].iloc[0])) else ''
        tag = ' (oracle)' if r['oracle'] != 'none' else ''
        print(f'{(r["method"] + tag)[:33]:<34s}{r["input_type"][:10]:<11s}{r["mirror_suppression_median"]:>9.3f}'
              f'{100 * r["mirror_suppression_frac_ok"]:>5.0f}%{r["ccc_mean"]:>7.3f}{r["ccc_tissue_dom_mean"]:>8.3f}'
              f'{r["ccc_ambiguous_mean"]:>8.3f}{r["ccc_vs_mirror_solution_mean"]:>8.3f}'
              f'{r["dc_residual_median"]:>9.2e}{r["ssim_mean"]:>7.3f}{flag:>9s}')
    print('=' * 118)
    print('Rows with a different input (dispersion_encoded) or an oracle do NOT share conditions with the rest.')
    print(f'Results in {EVAL_DIR}')


if __name__ == '__main__':
    main()
