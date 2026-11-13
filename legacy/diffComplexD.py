# -*- coding: utf-8 -*-
"""
===============================================================================
COMPLEX-FIELD MIRROR-ARTIFACT SUPPRESSION -- DIFFUSION MODELS AND REGRESSORS (D)
===============================================================================
Training script for the diffusion model reported in the paper and for the deterministic
regressors and alternative formulations used in the ablations. Evaluation is done with
evaluate_diffusion_D.py (same folder), which imports this module.

FORWARD MODEL. The corrupted measurement is y = M(S) = FFT(Re(IFFT(S))) along depth (axis 0,
with the fftshift convention of the pipeline), which equals (S + conj(S(-z)))/2. The field S and
its mirror solution S~ = conj(S(-z)) produce exactly the same measurement.

TWO MODEL FAMILIES (--family)

  bscan (default) -- full 512 x 512 B-scan with its zero delay at row 256 (the protocol of the
      paper). The network sees the structure through a 4-channel input computed ONLY from the
      measurement y:
          [ log|y| normalized (percentiles 1 and 99.9 of y itself),  cos(angle y),  sin(angle y),
            signed depth (z - c)/c ]
      The log amplitude gives the anatomy at full resolution; the phase enters as a unit vector
      (no jump at +-pi). The measured phasor is exactly recoverable from these channels.

      Formulations (--kind):
        dlogamp  DIFFUSION MODEL REPORTED IN THE PAPER. Diffusion over [log amplitude, cos, sin] of
                 the target: amplitude and phase are generated as separate channels, with the
                 measured phase given as input so that it can be copied, and the complex field is
                 then recomposed as a phasor, S = 10^(la/20) exp(i atan2(sin, cos))
                 (logamp_to_complex). Output variants: D_raw, and D_dc (exact projection onto the
                 measurement).
        dmask    Diffusion over the gain g = 2m - 1 (one antisymmetric channel), conditioned on the
                 4 input channels. Target: power-ratio mask of the ground truth, smoothed with
                 --mask-sigma. S = (1 + g) y: exact data consistency. Several samples give the
                 orientation uncertainty.
        mask     Regressor. "On which side of the zero delay does this energy belong?" The network
                 gives a logit o(z); the mask m(z) = sigmoid(o(z) - o(-z)) satisfies
                 m(z) + m(-z) = 1 by construction and S = 2 m y. The phase comes from the
                 measurement: where tissue dominates it is the true phase; where the mirror
                 dominates, the phase of the weak tissue is unrecoverable. Exact consistency with
                 the measurement (self-test). It cannot hallucinate phase or produce speckle-free
                 patches.
        null     Same backbone and input, free complex output in the null-space component:
                 S_n = y_n + (u - M u). mask vs. null separates "reformulation" from
                 "input/backbone".

      Backbones. Regressors: segmentation-models-pytorch U-Net with an ImageNet-PRETRAINED encoder
      (ConvNeXt-S from timm by default); with few training volumes, an encoder that already
      recognizes edges, continuity and shapes is the main lever against the generalization gap.
      Diffusion: diffusers UNet2DModel (attention only at 32x32 and 16x16, standard normalization
      and residuals), v-prediction, cosine schedule, DDIM sampling.

      Geometry augmentation (--zpd-shift). Before generating the artifact, the ground truth is
      shifted in depth by a random number of rows (equivalent to moving the reference arm), so
      each B-scan produces many different overlap configurations. The shift is circular: with
      moderate shifts, what wraps around is almost always noise floor. Global phase rotation and
      lateral flip are kept. NEVER a flip in z.

  roi -- square ROIs of --roi pixels (default 512 = full B-scan; 256 = tiled problem, which needs
      less GPU memory and is useful for quick tests or GPUs with limited VRAM). Each ROI is treated
      as an independent acquisition with its zero delay at ROI//2; folding is done inside the ROI.
      The complex field is normalized by an envelope computed only from y (no oracle), and the
      artifact and normalization are generated on the GPU per batch.

      Models (--model):
        diffusion  DDPM training and DDIM sampling in the normalized domain, with optional EXACT
                   data consistency (DC). Sampling variants: G1_plain (no DC), G2_dc_full (DC at
                   every step), G3_dc_late (DC in the last 30 % of the steps).
        regress    DETERMINISTIC REGRESSOR with the same U-Net (fixed t = 0, no x_t). Loss: complex
                   MSE in the normalized domain, the same for all heads. It answers whether this
                   architecture can choose the orientation without any diffusion component.

      Domain (--domain) and head (--head) of the regressor:
        unfolded + null   The network predicts u; S_n = y_n + (u - M u). DC by construction.
        folded   + null   FOLDED DOMAIN. With the zero delay at c = N/2, pixel c+j is paired with
                          c-j. With L[j] = S[c+j] and U'[j] = conj(S[c-j]), j = 0..c-1,
                          Y[j] = y[c+j] = (L + U')/2 exactly. Each folded pixel contains its two
                          partners, so the pairing becomes LOCAL. The network predicts d and
                          L = Y + d, U' = Y - d: DC by construction (self-test (g)).
        folded   + wiener WIENER HEAD. The network predicts a REAL gain g = tanh(.) in [-1, 1]:
                          L = Y (1 + g), U' = Y (1 - g). Under circular Gaussian speckle with
                          independent sources, the MMSE estimator has exactly this form with
                          g = (R_L - R_U)/(R_L + R_U). The output phase is that of the measurement:
                          the head cannot hallucinate phase.
      In the unfolded domain a signed depth channel (z - c)/c is added, and in the folded domain
      the distance to the zero delay j/(c-1): the same geometric information in both domains, so
      that unfolded vs. folded isolates ONLY the representation.

SHARED TRAINING INFRASTRUCTURE
  * LR with linear warmup + cosine decay down to --lr-min. AdamW with beta2 = 0.99.
  * EMA 0.9999 with warmup (timm ModelEmaV3, use_warmup): it does not drag the initial weights.
  * INTERNAL VALIDATION split: from each training volume, the final contiguous block of B-scans
    (--val-frac) is held out, with a guard band (--val-guard) that is not used at all: neighbouring
    B-scans are almost duplicates. The test set (VALIDATION_DATA_DIR) NEVER takes part in the
    selection.
  * Checkpoint selection with the TASK metric: among the internal evaluations with median
    suppression >= --supp-floor, the one with the highest CCC; if none reaches the floor, the one
    with the highest suppression. ema_best.pth (selected) and ema_final.pth (last) are saved.
  * Per-epoch log: gradient norm (mean and max), skipped non-finite steps and, for roi diffusion,
    the loss per range of t. In the bscan family a step with a non-finite gradient is discarded
    BEFORE touching the weights, and the run stops after --max-bad-steps consecutive bad steps.

USAGE
    python diffComplexD.py --stage selftest [--family bscan|roi]
    python diffComplexD.py --stage train --family bscan --kind dlogamp --out /path/to/run
    python diffComplexD.py --stage train --family roi --model regress --domain folded --head null --out /path/to/run
    python diffComplexD.py --stage preview --out /path/to/run [--which best|final]
===============================================================================
"""

import os
import csv
import glob
import json
import math
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
from numpy.fft import fft, ifft, fftshift, ifftshift

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
except ImportError:      # PyTorch < 2.3
    from contextlib import nullcontext
    def sdpa_kernel(_):
        return nullcontext()
    class SDPBackend:
        FLASH_ATTENTION = EFFICIENT_ATTENTION = None


# ============================== CONFIGURATION ==============================

TRAIN_DATA_DIR = 'put/your/data/path/here'                    # folder that contains TRAIN_FILES
VALIDATION_DATA_DIR = 'put/your/validation/data/path/here'    # TEST set (evaluator and previews)

TRAIN_FILES = [
    'Human_fingertip_skin.npy', 'Human_nail_plate.npy', 'Chicken_thigh_tissue.npy',
    'fovea.npy', 'perifovea.npy', 'macula.npy', 'optic_nerve.npy',
]

NUM_WORKERS = 8
EMA_DECAY = 0.9999
PREVIEW_SEED = 1234

# Internal validation (both families)
VAL_FRAC = 0.05
VAL_GUARD = 16
VAL_BSCANS = 8            # B-scans per volume evaluated (spread over the reserved block)
VAL_EVERY = 5
VAL_STEPS = 50            # sampling steps of the internal validation (cheaper than for previews)
SUPP_FLOOR = 0.50

# --- roi family ---
ROI_SIZE = 512            # 512 = full B-scan; 256 (tiles) lowers memory for quick tests / small GPUs
ROI_MODE = 'random'
CROPS_PER_BSCAN = 4       # random crops per B-scan and epoch (with ROI_SIZE = 512 the crop is the
                          # whole B-scan, so this only repeats it with different augmentations)

ENV_SIGMA_Z = 6.0
ENV_SIGMA_X = 6.0
ENV_SCALE = 2.0
ENV_FLOOR_DB = 100.0
X0_CLAMP = 20.0

NUM_HEADS = 4
HIRES_ATTENTION = True
NUM_TIME_STEPS = 1000
SAMPLING_STEPS = 100

DC_LATE_FRACTION = 0.30
VARIANTS = {
    'G1_plain':   dict(dc=False, dc_late=False),
    'G2_dc_full': dict(dc=True,  dc_late=False),
    'G3_dc_late': dict(dc=True,  dc_late=True),
}
REGRESS_VARIANTS = ('R_direct',)

MIRROR_DOMINANT_CONTRAST = 10.0
NOISE_FLOOR_PERCENTILE = 50.0
CLASS_CONTRAST = 3.0

# --- bscan family ---
BSCAN_SIZE = 512
ENC_DEFAULT = 'tu-convnext_small'
DIFF_BLOCKS = (64, 128, 128, 256, 256, 512)
DIFF_STEPS_TRAIN = 1000
DIFF_STEPS_SAMPLE = 50
KINDS = ('mask', 'null', 'dmask', 'dlogamp')

# Options whose default depends on the family (applied when the option is not given).
FAMILY_DEFAULTS = {
    'roi':   dict(epochs=70, batch_size=16, lr=4e-5, wd=0.0, snapshot_every=5, out='runs/D_roi'),
    'bscan': dict(epochs=40, batch_size=8, lr=1e-4, wd=0.01, snapshot_every=10, out='runs/D_bscan'),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def arch_family(arch):
    """
    'bscan' or 'roi'. The family is recognized by the fields of the architecture (only bscan
    models have 'kind'), not by the 'family' tag, so checkpoints written during development,
    which carry other tags, still load.
    """
    return 'bscan' if 'kind' in arch else 'roi'


def arch_get(arch, key, default=None):
    """roi family: defaults for checkpoints written before 'model'/'domain'/'head' existed."""
    defaults = dict(model='diffusion', domain='unfolded', head='null', depth_chan=False,
                    cond_env=True, out_channels=2)
    return arch.get(key, defaults.get(key, default))


def variants_for(arch):
    return list(REGRESS_VARIANTS) if arch_get(arch, 'model') == 'regress' else list(VARIANTS)


def preview_variants_for(arch):
    return list(REGRESS_VARIANTS) if arch_get(arch, 'model') == 'regress' else ['G1_plain', 'G2_dc_full']


# ============================== OPERATORS (torch) ==============================

def to_fringe_t(S):
    return torch.fft.ifftshift(torch.fft.ifft(torch.fft.ifftshift(S, dim=-2), dim=-2), dim=-2)


def from_fringe_t(f):
    return torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(f, dim=-2), dim=-2), dim=-2)


def torch_mirror(S):
    """Forward model M: FFT( Re( IFFT(S) ) ) with the shift convention of the pipeline."""
    return from_fringe_t(to_fringe_t(S).real.to(S.dtype))


def mirror_of_t(a):
    """out[z] = a[(-z) mod Z]. With this, M(S) = (S + conj(mirror_of(S)))/2."""
    return torch.roll(torch.flip(a, dims=[-2]), 1, dims=-2)


def c2ch(z):
    """complex (B,Z,X) -> real (B,2,Z,X)"""
    return torch.stack([z.real, z.imag], dim=1)


def ch2c(x):
    """real (B,2,Z,X) -> complex (B,Z,X)"""
    return torch.complex(x[:, 0].float(), x[:, 1].float())


_KERNEL_CACHE = {}


def _gauss1d(sigma, dev, dtype=torch.float32):
    key = (float(sigma), str(dev), dtype)
    if key not in _KERNEL_CACHE:
        r = max(1, int(math.ceil(3.0 * sigma)))
        x = torch.arange(-r, r + 1, dtype=dtype, device=dev)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        _KERNEL_CACHE[key] = (k / k.sum(), r)
    return _KERNEL_CACHE[key]


def envelope_t(y, sz=ENV_SIGMA_Z, sx=ENV_SIGMA_X, scale=ENV_SCALE, floor_db=ENV_FLOOR_DB):
    """
    Real envelope (B,Z,X) computed ONLY from the measurement y. It is symmetric under mirror_of,
    so M commutes with the division by it.
      - CIRCULAR border in z: a circular convolution with a symmetric kernel commutes with z -> -z.
      - reflect border in x: M acts per A-line, any dependence on x is valid.
      - explicit final symmetrization to absorb rounding errors.
      - floor relative to the GLOBAL maximum of each sample (a global maximum keeps the symmetry).
    """
    p = (y.real ** 2 + y.imag ** 2).unsqueeze(1)          # (B,1,Z,X), same precision as y
    kz, rz = _gauss1d(sz, p.device, p.dtype)
    kx, rx = _gauss1d(sx, p.device, p.dtype)
    p = F.conv2d(F.pad(p, (0, 0, rz, rz), mode='circular'), kz.view(1, 1, -1, 1))
    p = F.conv2d(F.pad(p, (rx, rx, 0, 0), mode='reflect'), kx.view(1, 1, 1, -1))
    p = p.squeeze(1)
    p = 0.5 * (p + mirror_of_t(p))
    pmax = p.amax(dim=(-2, -1), keepdim=True)
    p = torch.maximum(p, pmax * (10.0 ** (-floor_db / 10.0)) + torch.finfo(p.dtype).tiny)
    return scale * torch.sqrt(p)


def envelope_channel(e):
    """log10(e) normalized per sample with percentiles of the input itself: ~[-1, 1]."""
    le = torch.log10(e)
    q = torch.quantile(le.flatten(1), torch.tensor([0.01, 0.5, 0.99], device=e.device), dim=1)
    lo, med, hi = q[0].view(-1, 1, 1), q[1].view(-1, 1, 1), q[2].view(-1, 1, 1)
    return (le - med) / (0.5 * (hi - lo) + 1e-6)


def normalize_measurement(y, cfg):
    """
    Physical complex y (B,Z,X) -> (y_n, envelope e, log-envelope channel), all from y.
    This is the ONLY thing used at inference; the ground truth is not involved.
    """
    e = envelope_t(y, cfg['env_sigma_z'], cfg['env_sigma_x'], cfg['env_scale'], cfg['env_floor_db'])
    return y / e, e, envelope_channel(e)


def build_condition(yn, envch, arch):
    """Conditioning channels for the domain. Same geometric content in both domains."""
    B, N, X = yn.shape
    c = N // 2
    use_env = arch_get(arch, 'cond_env')
    use_depth = arch_get(arch, 'depth_chan')
    if arch_get(arch, 'domain') == 'folded':
        Y = yn[:, c:, :]
        ch = [Y.real, Y.imag]
        if use_env:
            ch.append(envch[:, c:, :])
        if use_depth:
            j = torch.arange(c, device=yn.device, dtype=torch.float32) / max(1, c - 1)
            ch.append(j.view(1, c, 1).expand(B, c, X))
    else:
        ch = [yn.real, yn.imag]
        if use_env:
            ch.append(envch)
        if use_depth:
            z = (torch.arange(N, device=yn.device, dtype=torch.float32) - c) / c
            ch.append(z.view(1, N, 1).expand(B, N, X))
    return torch.stack(ch, dim=1).float()


def cond_channels(domain, cond_env, depth_chan):
    return 2 + int(bool(cond_env)) + int(bool(depth_chan))


# ---------- folded domain ----------
# With c = N/2:  L[j] = S[c+j],  U'[j] = conj(S[c-j]),  j = 0..c-1,   Y[j] = y[c+j] = (L + U')/2.
# Self-paired rows: c (zero delay, j = 0, where U'[0] = conj(L[0]) and Y is real) and 0 (border).

def fold_t(S):
    """S (B,N,X) -> (L, U'), each (B,c,X)."""
    c = S.shape[-2] // 2
    return S[:, c:, :], torch.conj(torch.flip(S[:, 1:c + 1, :], dims=[-2]))


def unfold_t(L, Up, y_full):
    """
    Inverse of fold_t. The zero-delay row is averaged, (L[0] + conj U'[0])/2, which makes it
    consistent with a real Y there; row 0 (self-paired, outside the folding) takes the measurement.
    For any d: M(unfold(Y+d, Y-d)) = y  (exact data consistency, self-test (g)).
    """
    return torch.cat([y_full[:, :1, :],
                      torch.flip(torch.conj(Up[:, 1:, :]), dims=[-2]),
                      ((L[:, :1, :] + torch.conj(Up[:, :1, :])) / 2),
                      L[:, 1:, :]], dim=1)


def fold_np(S):
    c = S.shape[-2] // 2
    return S[..., c:, :], np.conj(np.flip(S[..., 1:c + 1, :], axis=-2))


def unfold_np(L, Up, y_full):
    return np.concatenate([y_full[..., :1, :], np.flip(np.conj(Up[..., 1:, :]), axis=-2),
                           (L[..., :1, :] + np.conj(Up[..., :1, :])) / 2, L[..., 1:, :]], axis=-2)


def dc_project(x0, yn):
    """EXACT data consistency in the normalized domain: S = y_n + (I - M) S_hat."""
    z = ch2c(x0)
    return c2ch(yn + (z - torch_mirror(z)))


# ============================== OPERATORS (numpy) ==============================

def mirrorArtifact(array):
    """Forward model. Identical to the one used by PC-CGAN and by the evaluator."""
    fringes = ifftshift(ifft(ifftshift(array, axes=0), axis=0), axes=0)
    return fftshift(fft(fftshift(fringes.real, axes=0), axis=0), axes=0)


def mirror_of(arr):
    return np.roll(np.flip(arr, axis=0), 1, axis=0)


# ============================== DATA ==============================

def load_volume_bscans(filepath, max_per_volume=None):
    """
    Loads a volume (Z,X,Y) complex or (Z,X,Y,2) and returns (nB, Z, X) contiguous complex64.
    It is read with mmap and in blocks so that a full complex128 volume is not duplicated in RAM.
    """
    raw = np.load(filepath, mmap_mode='r')
    Y = raw.shape[2]
    idx = np.arange(Y) if max_per_volume is None else \
        np.unique(np.linspace(0, Y - 1, min(max_per_volume, Y), dtype=int))
    out = np.empty((len(idx), raw.shape[0], raw.shape[1]), dtype=np.complex64)
    for k0 in range(0, len(idx), 64):
        sel = idx[k0:k0 + 64]
        blk = np.asarray(raw[:, :, sel])
        if raw.ndim == 4:
            out[k0:k0 + len(sel)] = (blk[..., 0] + 1j * blk[..., 1]).transpose(2, 0, 1)
        else:
            out[k0:k0 + len(sel)] = blk.transpose(2, 0, 1)
    print(f'  {os.path.basename(filepath)}: {raw.shape} -> {out.shape} complex64 '
          f'({out.nbytes / 2**30:.2f} GiB)')
    return out


def roi_origins(shape, roi=ROI_SIZE):
    """Regular non-overlapping tiling ('tile' mode and evaluation)."""
    Z, X = shape
    return [(z0, x0) for z0 in range(0, Z - roi + 1, roi) for x0 in range(0, X - roi + 1, roi)]


class CleanBscanDataset(Dataset):
    """
    Stores ONLY clean B-scans. Returns (2, roi, roi) float32 [Re, Im] crops of the ground truth.
    The artifact and the normalization are computed afterwards, on the GPU, per batch.
    The random crop uses torch (seeded differently per worker); the global np.random would be
    duplicated identically in all forked workers.
    """

    def __init__(self, volumes, roi=ROI_SIZE, mode=ROI_MODE, crops_per_bscan=CROPS_PER_BSCAN):
        self.vols, self.roi, self.mode = volumes, roi, mode
        self.index = []
        for v, arr in enumerate(volumes):
            nb, Z, X = arr.shape
            if Z < roi or X < roi:
                print(f'  WARNING: volume {v} {arr.shape} is smaller than the ROI, skipped'); continue
            tiles = roi_origins((Z, X), roi)
            reps = len(tiles) if mode == 'tile' else (1 if mode == 'center' else crops_per_bscan)
            for b in range(nb):
                for k in range(reps):
                    self.index.append((v, b, k))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        v, b, k = self.index[i]
        bs = self.vols[v][b]
        Z, X = bs.shape
        r = self.roi
        if self.mode == 'tile':
            z0, x0 = roi_origins((Z, X), r)[k]
        elif self.mode == 'center':
            z0, x0 = (Z - r) // 2, (X - r) // 2
        else:
            z0 = int(torch.randint(0, Z - r + 1, (1,)))
            x0 = int(torch.randint(0, X - r + 1, (1,)))
        c = bs[z0:z0 + r, x0:x0 + r]
        return torch.from_numpy(np.stack([c.real, c.imag]).astype(np.float32))


def split_volume(nb, val_frac=VAL_FRAC, guard=VAL_GUARD):
    """Indices [0, n_train) training, [n_train, n_train+guard) discarded, the rest validation."""
    n_val = max(4, int(round(val_frac * nb)))
    n_train = nb - n_val - guard
    if n_train < 1:
        raise ValueError(f'Volume with {nb} B-scans: not enough for val={n_val} + guard={guard}')
    return n_train, nb - n_val


def gpu_make_batch(gt2, cfg, augment=True):
    """
    roi family. Clean crops (B,2,r,r) -> dict with S_n (normalized target), y_n, e and the envelope
    channel. Everything in fp32; autocast only wraps the network.
    """
    S = ch2c(gt2.to(device, non_blocking=True))
    B = S.shape[0]
    if augment and cfg['aug_phase']:
        th = torch.rand(B, 1, 1, device=device) * (2 * math.pi)
        S = S * torch.polar(torch.ones_like(th), th)
    if augment and cfg['aug_flip']:
        fl = torch.rand(B, device=device) < 0.5
        if fl.any():
            S[fl] = torch.flip(S[fl], dims=[-1])
    y = torch_mirror(S)
    yn, e, envch = normalize_measurement(y, cfg)
    return dict(Sn=S / e, yn=yn, e=e, envch=envch)


def gpu_batch(gt2, cfg, augment=True):
    """bscan family. Clean B-scans (B,2,N,X) -> (S, y = M(S)), with phase/flip/zero-delay augmentation."""
    S = ch2c(gt2.to(device, non_blocking=True))
    B, N, X = S.shape
    if augment:
        if cfg['aug_phase']:
            th = torch.rand(B, 1, 1, device=device) * (2 * math.pi)
            S = S * torch.polar(torch.ones_like(th), th)
        if cfg['aug_flip']:
            fl = torch.rand(B, device=device) < 0.5
            if fl.any():
                S[fl] = torch.flip(S[fl], dims=[-1])
        if cfg['zpd_shift'] > 0:
            do = torch.rand(B) < cfg['zpd_shift_p']
            sh = torch.randint(-cfg['zpd_shift'], cfg['zpd_shift'] + 1, (B,))
            for b in range(B):
                if do[b] and int(sh[b]) != 0:
                    S[b] = torch.roll(S[b], int(sh[b]), dims=0)
    return S, torch_mirror(S)


# ============================== roi FAMILY: MODEL ==============================
# DDPM-style U-Net with sinusoidal time embeddings, residual blocks and self-attention.

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps, embed_dim):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        embeds = self.embeddings[t].to(x.device)
        return embeds[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, C, num_groups, dropout_prob):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x, embeddings):
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x


class Attention(nn.Module):
    def __init__(self, C, num_heads, dropout_prob):
        super().__init__()
        self.proj1 = nn.Linear(C, C * 3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        # 'b L (K H C)': keeps C as the fastest component (stride 1) -> flash/efficient kernels.
        x = rearrange(x, 'b L (K H C) -> K b H L C', K=3, H=self.num_heads)
        q, k, v = x[0].contiguous(), x[1].contiguous(), x[2].contiguous()
        if q.is_cuda:
            # Restricted list: failing is preferred over falling back to the math backend (O(N^2) memory).
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                x = F.scaled_dot_product_attention(q, k, v, is_causal=False,
                                                   dropout_p=self.dropout_prob if self.training else 0.0)
        else:   # CPU (self-test): no backend restriction
            x = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')


class UnetLayer(nn.Module):
    def __init__(self, upscale, attention, num_groups, dropout_prob, num_heads, C):
        super().__init__()
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C // 2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C * 2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(C, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        return self.conv(x), x


class UNET(nn.Module):
    def __init__(self, Channels=[64, 128, 256, 512, 512, 384],
                 Attentions=[False, False, False, True, True, True],
                 Upscales=[False, False, False, True, True, True],
                 num_groups=32, dropout_prob=0.1, num_heads=NUM_HEADS,
                 input_channels=5, output_channels=2, time_steps=NUM_TIME_STEPS):
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        out_channels = (Channels[-1] // 2) + Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels // 2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels))
        for i in range(self.num_layers):
            setattr(self, f'Layer{i+1}', UnetLayer(
                upscale=Upscales[i], attention=Attentions[i], num_groups=num_groups,
                dropout_prob=dropout_prob, C=Channels[i], num_heads=num_heads))

    def forward(self, x, t, condition=None):
        if condition is not None:
            x = torch.cat((x, condition), dim=1)
        x = self.shallow_conv(x)
        residuals = []
        for i in range(self.num_layers // 2):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)
            x, r = layer(x, embeddings)
            residuals.append(r)
        for i in range(self.num_layers // 2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, embeddings)[0], residuals[self.num_layers - i - 1]), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))


def build_model_roi(arch):
    attn = [False, False, False, True, True, bool(arch['hires_attn'])]
    m = UNET(Channels=arch.get('channels', [64, 128, 256, 512, 512, 384]),
             Attentions=attn, num_heads=int(arch['num_heads']),
             input_channels=int(arch['in_channels']),
             output_channels=int(arch_get(arch, 'out_channels'))).to(device)
    m.arch = dict(arch)      # travels with the model (and with the EMA copy): reconstruct_rois reads it
    return m


class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps=NUM_TIME_STEPS):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False).to(device)
        self.alpha = torch.cumprod(1 - self.beta, dim=0).to(device)


# ============================== roi FAMILY: AUXILIARY LOSSES (diffusion) ==============================

def phase_loss_circular(x0_hat, x0, alpha_t):
    """
    1 - cos(dphi) weighted by sqrt(|x0_hat||x0|) * alpha_t.
    Signed representation: atan2 gives the physical phase directly (no shift).
    """
    fr, fi = x0_hat[:, 0], x0_hat[:, 1]
    rr, ri = x0[:, 0], x0[:, 1]
    delta = torch.atan2(fi, fr) - torch.atan2(ri, rr)
    amp_f = torch.sqrt(fr ** 2 + fi ** 2 + 1e-8)
    amp_r = torch.sqrt(rr ** 2 + ri ** 2 + 1e-8)
    w = torch.sqrt(amp_f * amp_r + 1e-8) * alpha_t.view(-1, 1, 1)
    return (w * (1 - torch.cos(delta))).sum() / (w.sum() + 1e-8)


def phase_loss_coherence(x0_hat, x0, alpha_t, win=8):
    """
    1 - Re<x0_hat, x0>_w / (||x0_hat||_w ||x0||_w) in local win x win windows.
    No atan2: the gradient with respect to Re/Im does not scale as 1/|z| in weak pixels.
    Uses Re (not |.|): a constant local phase offset IS penalized, because the absolute phase
    with respect to the measurement is physical. Weighted by alpha_t like the circular loss.
    """
    num = F.avg_pool2d(x0_hat[:, 0:1] * x0[:, 0:1] + x0_hat[:, 1:2] * x0[:, 1:2], win)
    p1 = F.avg_pool2d(x0_hat[:, 0:1] ** 2 + x0_hat[:, 1:2] ** 2, win)
    p2 = F.avg_pool2d(x0[:, 0:1] ** 2 + x0[:, 1:2] ** 2, win)
    coh = num / (torch.sqrt(p1 * p2) + 1e-6)
    w = alpha_t.view(-1, 1, 1, 1)
    return ((1 - coh) * w).sum() / (w.sum() * coh[0].numel() + 1e-8)


# ============================== roi FAMILY: SAMPLING / INFERENCE ==============================

def _autocast(amp_dtype):
    if amp_dtype is not None and device.type == 'cuda':
        return torch.autocast('cuda', dtype=amp_dtype)
    return torch.autocast('cuda', enabled=False)


@torch.no_grad()
def sample_ddim(model, cond, yn, dc=False, dc_late=False, steps=SAMPLING_STEPS,
                clamp=X0_CLAMP, generator=None, amp_dtype=None, diag=None):
    """
    DDIM eta=0 in the normalized domain, with optional and EXACT data consistency.
    Note: eta=0 is deterministic GIVEN the seed, but it is still a SAMPLE of the posterior
    (the randomness moves to x_T); it is not the posterior mean.
    """
    sched = DDPM_Scheduler()
    ts = torch.linspace(NUM_TIME_STEPS - 1, 0, steps).long().tolist()
    b, _, Z, X = cond.shape
    x = torch.randn((b, 2, Z, X), device=device, generator=generator)
    dc_start = int((1 - DC_LATE_FRACTION) * steps) if dc_late else 0
    for i, t in enumerate(ts):
        tt = torch.full((b,), t, dtype=torch.long)
        a_t = sched.alpha[t].view(1, 1, 1, 1)
        ctx = torch.autocast('cuda', dtype=amp_dtype) if (amp_dtype is not None and device.type == 'cuda') \
            else torch.autocast('cuda', enabled=False)
        with ctx:
            eps = model(x, tt, cond)
        eps = eps.float()
        x0 = ((x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)).clamp(-clamp, clamp)
        if dc and i >= dc_start:
            x0 = dc_project(x0, yn)
        if diag is not None and (i % max(1, steps // 10) == 0 or i == len(ts) - 1):
            z = ch2c(x0)
            res = (torch.linalg.vector_norm(torch_mirror(z) - yn) /
                   (torch.linalg.vector_norm(yn) + 1e-30)).item()
            sat = (x0.abs() >= clamp * (1 - 1e-6)).float().mean().item()
            diag.append(dict(step=i, t=int(t), dc_residual=res, saturated_frac=sat))
        if i < len(ts) - 1:
            a_prev = sched.alpha[ts[i + 1]].view(1, 1, 1, 1)
            eps_r = (x - torch.sqrt(a_t) * x0) / torch.sqrt(1 - a_t)
            x = torch.sqrt(a_prev) * x0 + torch.sqrt(1 - a_prev) * eps_r
        else:
            x = x0
    return x


def regress_forward(model, yn, envch, amp_dtype=None):
    """
    Deterministic regressor. Returns the estimated S_n (B,N,X), complex, in the normalized domain,
    with data consistency guaranteed by construction for the three heads.
    """
    arch = model.arch
    cond = build_condition(yn, envch, arch)
    t0 = torch.zeros(cond.shape[0], dtype=torch.long)
    with _autocast(amp_dtype):
        out = model(cond, t0)
    out = out.float()
    if arch_get(arch, 'domain') == 'unfolded':
        if arch_get(arch, 'head') != 'null':
            raise ValueError('the wiener head only exists in the folded domain')
        u = ch2c(out)
        return yn + (u - torch_mirror(u))
    c = yn.shape[-2] // 2
    Y = yn[:, c:, :]
    if arch_get(arch, 'head') == 'wiener':
        d = torch.tanh(out[:, 0]) * Y
    else:
        d = ch2c(out)
    return unfold_t(Y + d, Y - d, yn)


@torch.no_grad()
def roi_reconstruct(model, y_rois, cfg, variant=None, steps=SAMPLING_STEPS, batch=32, seed=0,
                    n_samples=1, amp_dtype=torch.bfloat16, return_std=False):
    """
    roi family. y_rois: complex numpy (N, r, r) = measurement per ROI. Returns the estimated
    PHYSICAL complex field (N, r, r) complex64, de-normalized with the envelope of y (no oracle).
    Diffusion: variant G1/G2/G3 and n_samples (mean of K samples). Regressor: a single pass
    (variant 'R_direct'; n_samples is ignored).
    """
    arch = model.arch
    is_reg = arch_get(arch, 'model') == 'regress'
    variant = variant or ('R_direct' if is_reg else 'G2_dc_full')
    out = np.zeros(y_rois.shape, dtype=np.complex64)
    std = np.zeros(y_rois.shape, dtype=np.float32) if return_std else None
    for i0 in range(0, len(y_rois), batch):
        y = torch.from_numpy(np.ascontiguousarray(y_rois[i0:i0 + batch])).to(device).to(torch.complex64)
        yn, e, envch = normalize_measurement(y, cfg)
        if is_reg:
            out[i0:i0 + len(y)] = (regress_forward(model, yn, envch, amp_dtype) * e).cpu().numpy()
            continue
        v = VARIANTS[variant]
        cond = build_condition(yn, envch, arch)
        acc, acc2 = None, None
        for k in range(n_samples):
            g = torch.Generator(device=device).manual_seed(int(seed) + 1000003 * k + i0)
            x = sample_ddim(model, cond, yn, dc=v['dc'], dc_late=v['dc_late'], steps=steps,
                            clamp=cfg.get('x0_clamp', X0_CLAMP), generator=g, amp_dtype=amp_dtype)
            S = ch2c(x) * e
            acc = S if acc is None else acc + S
            if return_std:
                acc2 = S.abs() ** 2 if acc2 is None else acc2 + S.abs() ** 2
        mean = acc / n_samples
        out[i0:i0 + len(y)] = mean.cpu().numpy()
        if return_std:
            std[i0:i0 + len(y)] = torch.sqrt((acc2 / n_samples - mean.abs() ** 2).clamp_min(0)).cpu().numpy()
    return (out, std) if return_std else out


# ============================== bscan FAMILY: REPRESENTATION ==============================

def bscan_is_diffusion(arch):
    return arch['kind'] in ('dmask', 'dlogamp')


def bscan_eval_variants(arch):
    """Variants the evaluator computes for each kind of model: name -> (variant, n_samples)."""
    k = arch['kind']
    if k in ('mask', 'null'):
        return {'R_direct': ('R_direct', 1)}
    if k == 'dmask':
        return {'D_mask': ('D_mask', 1), 'D_mask_mean4': ('D_mask', 4)}
    return {'D_raw': ('D_raw', 1), 'D_dc': ('D_dc', 1)}


def bscan_preview_variants(arch):
    vs = []
    for v, _ in bscan_eval_variants(arch).values():
        if v not in vs:
            vs.append(v)
    return vs[:2]


def input_features(y):
    """
    y complex (B,N,X) -> (input (B,4,N,X), (q01, q999) in dB per sample).
    Everything is computed from the measurement; nothing from the ground truth.
    """
    B, N, X = y.shape
    amp = y.abs().float()
    la = 20 * torch.log10(amp + 1e-30)
    q = torch.quantile(la.flatten(1), torch.tensor([0.01, 0.999], device=y.device), dim=1)
    q01, q999 = q[0].view(B, 1, 1), q[1].view(B, 1, 1)
    a = ((la - q01) / (q999 - q01 + 1e-6)).clamp(-0.5, 1.5)
    ph = y / (amp + 1e-30)
    z = ((torch.arange(N, device=y.device, dtype=torch.float32) - N // 2) / (N // 2)).view(1, N, 1).expand(B, N, X)
    return torch.stack([a, ph.real.float(), ph.imag.float(), z], dim=1), (q01, q999)


def smooth_power(p, sigma):
    """Gaussian smoothing (B,N,X), circular border in z: commutes with z -> -z (exact antisymmetry)."""
    if sigma <= 0:
        return p
    k, r = _gauss1d(sigma, p.device, p.dtype)
    q = p.unsqueeze(1)
    q = F.conv2d(F.pad(q, (0, 0, r, r), mode='circular'), k.view(1, 1, -1, 1))
    q = F.conv2d(F.pad(q, (r, r, 0, 0), mode='reflect'), k.view(1, 1, 1, -1))
    return q.squeeze(1)


def ratio_gain_target(S, sigma):
    """g* = (P(z) - P(-z)) / (P(z) + P(-z)) with P = smoothed |S|^2. Antisymmetric, in [-1, 1]."""
    P = S.abs() ** 2
    Pm = mirror_of_t(P)
    Ps, Pms = smooth_power(P, sigma), smooth_power(Pm, sigma)
    return (Ps - Pms) / (Ps + Pms + 1e-30)


def antisym(g):
    return 0.5 * (g - mirror_of_t(g))


def mask_from_logits(o):
    """m(z) = sigmoid(o(z) - o(-z)):  m(z) + m(-z) = 1 exactly."""
    return torch.sigmoid(o - mirror_of_t(o))


def logamp_target(S, q01, q999):
    """
    dlogamp target: [2a-1, cos, sin] of the ground truth, with the log amplitude a normalized with
    the percentiles of the MEASUREMENT. Amplitude and phase are separate channels.
    """
    la = 20 * torch.log10(S.abs() + 1e-30)
    a = ((la - q01) / (q999 - q01 + 1e-6)).clamp(-0.5, 1.5)
    ph = S / (S.abs() + 1e-30)
    return torch.stack([2 * a - 1, ph.real, ph.imag], dim=1).float()


def logamp_to_complex(x, q01, q999):
    """
    Recomposes the complex field (phasor) from the generated amplitude and phase channels:
    amplitude 10^(la/20) from the de-normalized log amplitude, phase atan2(sin, cos).
    """
    a = (x[:, 0] + 1) / 2
    la = a * (q999 - q01) + q01
    c, s = x[:, 1], x[:, 2]
    return torch.polar(10 ** (la / 20), torch.atan2(s, c))


# ============================== bscan FAMILY: MODELS ==============================

class DiffNet(nn.Module):
    """diffusers UNet2DModel: input [x_t, condition] -> v."""

    def __init__(self, x_ch, cond_ch, blocks=DIFF_BLOCKS, layers=2):
        super().__init__()
        from diffusers import UNet2DModel
        nb = len(blocks)
        self.unet = UNet2DModel(
            sample_size=BSCAN_SIZE, in_channels=x_ch + cond_ch, out_channels=x_ch, layers_per_block=layers,
            block_out_channels=tuple(blocks),
            down_block_types=('DownBlock2D',) * (nb - 2) + ('AttnDownBlock2D',) * 2,
            up_block_types=('AttnUpBlock2D',) * 2 + ('UpBlock2D',) * (nb - 2),
            norm_num_groups=32)

    def forward(self, x, t, cond):
        return self.unet(torch.cat([x, cond], dim=1), t).sample


def build_model_bscan(arch, pretrained=False):
    kind = arch['kind']
    if kind in ('mask', 'null'):
        import segmentation_models_pytorch as smp
        m = smp.Unet(encoder_name=arch['encoder'], encoder_weights='imagenet' if pretrained else None,
                     in_channels=4, classes=1 if kind == 'mask' else 2)
    else:
        m = DiffNet(1 if kind == 'dmask' else 3, 4, arch.get('blocks', DIFF_BLOCKS), arch.get('layers', 2))
    m = m.to(device)
    m.arch = dict(arch)
    return m


def zero_init_output(model):
    """Regressors: output at zero -> they start at the identity (mask 0.5 / u = 0), the no-op."""
    conv = model.segmentation_head[0]
    nn.init.zeros_(conv.weight); nn.init.zeros_(conv.bias)


def make_schedulers():
    from diffusers import DDPMScheduler, DDIMScheduler
    tr = DDPMScheduler(num_train_timesteps=DIFF_STEPS_TRAIN, beta_schedule='squaredcos_cap_v2',
                       prediction_type='v_prediction', clip_sample=False)
    return tr, DDIMScheduler.from_config(tr.config)


# ============================== bscan FAMILY: INFERENCE ==============================

def forward_det(model, y, cfg, amp_dtype=None):
    """Regressors. Physical complex y (B,N,X) -> (estimated physical S, gain g or None)."""
    feats, _ = input_features(y)
    with _autocast(amp_dtype):
        out = model(feats)
    out = out.float()
    if model.arch['kind'] == 'mask':
        m = mask_from_logits(out[:, 0])
        return 2 * m * y, 2 * m - 1
    e = envelope_t(y, cfg['env_sigma_z'], cfg['env_sigma_x'], cfg['env_scale'], cfg['env_floor_db'])
    yn = y / e
    u = ch2c(out)
    return (yn + (u - torch_mirror(u))) * e, None


@torch.no_grad()
def sample_diff(model, y, variant, steps=DIFF_STEPS_SAMPLE, seed=0, amp_dtype=None):
    """Diffusion (DDIM, eta = 0). Returns (estimated physical S, gain g or None)."""
    _, sch = make_schedulers()
    sch.set_timesteps(steps)
    feats, (q01, q999) = input_features(y)
    kind = model.arch['kind']
    B, N, X = y.shape
    g_ = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn((B, 1 if kind == 'dmask' else 3, N, X), device=device, generator=g_)
    for t in sch.timesteps:
        with _autocast(amp_dtype):
            v = model(x, t.to(device).expand(B), feats)
        x = sch.step(v.float(), t, x, eta=0.0).prev_sample
    if kind == 'dmask':
        g = antisym(x[:, 0]).clamp(-1, 1)
        return (1 + g) * y, g
    # dlogamp: the generated amplitude and phase channels are recomposed into a complex phasor
    S = logamp_to_complex(x, q01, q999)
    if variant == 'D_dc':
        S = y + (S - torch_mirror(S))
    return S, None


@torch.no_grad()
def bscan_reconstruct(model, y_rois, cfg, variant=None, steps=DIFF_STEPS_SAMPLE, batch=8, seed=0,
                      n_samples=1, amp_dtype=torch.bfloat16, return_gain=False):
    """bscan family. Same interface as roi_reconstruct (y_rois are full B-scans)."""
    arch = model.arch
    variant = variant or list(bscan_eval_variants(arch).values())[0][0]
    steps = min(int(steps), 200) if bscan_is_diffusion(arch) else steps
    batch = min(batch, 8)
    out = np.zeros(y_rois.shape, dtype=np.complex64)
    gains = np.zeros(y_rois.shape, dtype=np.float32) if return_gain else None
    for i0 in range(0, len(y_rois), batch):
        y = torch.from_numpy(np.ascontiguousarray(y_rois[i0:i0 + batch])).to(device).to(torch.complex64)
        if not bscan_is_diffusion(arch):
            S, g = forward_det(model, y, cfg, amp_dtype)
        else:
            acc, gacc = 0, 0
            for k in range(n_samples):
                S, g = sample_diff(model, y, variant, steps, seed + 1000003 * k + i0, amp_dtype)
                acc = acc + S
                gacc = gacc + (g if g is not None else 0)
            S, g = acc / n_samples, (gacc / n_samples if g is not None else None)
        out[i0:i0 + len(y)] = S.cpu().numpy()
        if return_gain and g is not None:
            gains[i0:i0 + len(y)] = g.cpu().numpy()
    return (out, gains) if return_gain else out


def det_loss(S_hat, S, y, cfg, w):
    """Complex MSE normalized by the envelope of y + L1 of the log amplitude (what is judged visually)."""
    e = envelope_t(y, cfg['env_sigma_z'], cfg['env_sigma_x'], cfg['env_scale'], cfg['env_floor_db'])
    cm = ((S_hat - S) / e).abs().pow(2).mean()
    eps = 0.05 * e
    la = (torch.log(S_hat.abs() + eps) - torch.log(S.abs() + eps)).abs().mean()
    return w['cmse'] * cm + w['logamp'] * la, dict(cmse=cm.item(), logamp=la.item())


# ============================== COMMON INTERFACE ==============================

@torch.no_grad()
def reconstruct_rois(model, y_rois, cfg, variant=None, steps=None, batch=None, seed=0, n_samples=1,
                     amp_dtype=torch.bfloat16):
    """
    Common interface with the evaluator, for both families. y_rois: complex numpy (N, r, r) =
    measurement per ROI (full B-scans for the bscan family). Returns the estimated PHYSICAL complex
    field (N, r, r) complex64.
    """
    if arch_family(model.arch) == 'bscan':
        return bscan_reconstruct(model, y_rois, cfg, variant, steps=steps or DIFF_STEPS_SAMPLE,
                                 batch=batch or 8, seed=seed, n_samples=n_samples, amp_dtype=amp_dtype)
    return roi_reconstruct(model, y_rois, cfg, variant, steps=steps or SAMPLING_STEPS,
                           batch=batch or 32, seed=seed, n_samples=n_samples, amp_dtype=amp_dtype)


# ============================== QUICK METRICS (monitoring and internal validation) ==============================
# Exact copies of those in the evaluator, so that both measure the same thing.

def masked_complex_coherence(pred_cx, gt_cx, percentile=50):
    mask = np.abs(gt_cx) > np.percentile(np.abs(gt_cx), percentile)
    d1, d2 = pred_cx[mask], gt_cx[mask]
    piston = np.angle(np.sum(d1 * np.conj(d2)))
    d1c = d1 * np.exp(-1j * piston)
    num = np.abs(np.sum(d1c * np.conj(d2)))
    den = np.sqrt(np.sum(np.abs(d1c) ** 2) * np.sum(np.abs(d2) ** 2))
    return float(num / (den + 1e-12))


def mirror_region_mask(target_cx):
    true_term = target_cx / 2.0
    mirror_term = np.conj(mirror_of(target_cx)) / 2.0
    floor = np.percentile(np.abs(target_cx), NOISE_FLOOR_PERCENTILE)
    dominant = np.abs(mirror_term) > MIRROR_DOMINANT_CONTRAST * np.abs(true_term)
    return dominant & (np.abs(mirror_of(target_cx)) > floor)


def mirror_suppression(pred_cx, gt_cx, input_cx, mask):
    if mask is None or not mask.any():
        return np.nan
    e_pred = float(np.sum(np.abs(pred_cx[mask]) ** 2))
    e_gt = float(np.sum(np.abs(gt_cx[mask]) ** 2))
    e_in = float(np.sum(np.abs(input_cx[mask]) ** 2))
    d = e_in - e_gt
    return float((e_in - e_pred) / d) if abs(d) > 1e-20 else np.nan


def dc_residual_np(pred_cx, input_cx):
    return float(np.linalg.norm(mirrorArtifact(pred_cx) - input_cx) / (np.linalg.norm(input_cx) + 1e-30))


def class_masks(gt_cx):
    t = np.abs(gt_cx)
    m = np.abs(mirror_of(gt_cx))
    valid = t > np.percentile(t, NOISE_FLOOR_PERCENTILE)
    ratio = t / (m + 1e-30)
    return valid & (ratio > CLASS_CONTRAST), valid & (ratio <= CLASS_CONTRAST) & (ratio >= 1.0 / CLASS_CONTRAST)


def coherence_on_mask(pred_cx, gt_cx, mask, min_pixels=50):
    if mask.sum() < min_pixels:
        return np.nan
    d1, d2 = pred_cx[mask], gt_cx[mask]
    piston = np.angle(np.sum(d1 * np.conj(d2)))
    d1c = d1 * np.exp(-1j * piston)
    return float(np.abs(np.sum(d1c * np.conj(d2))) /
                 (np.sqrt(np.sum(np.abs(d1c) ** 2) * np.sum(np.abs(d2) ** 2)) + 1e-12))


def quick_metrics(preds, gts, ys):
    """Averages over a set of ROIs; the suppression (a ratio) uses the median, as in the evaluator."""
    ccc, cccm, ccct, ccca, sup, dcr = [], [], [], [], [], []
    for p, g, y in zip(preds, gts, ys):
        p = p.astype(np.complex128); g = g.astype(np.complex128); y = y.astype(np.complex128)
        ccc.append(masked_complex_coherence(p, g))
        cccm.append(masked_complex_coherence(p, np.conj(mirror_of(g))))
        tis, amb = class_masks(g)
        ccct.append(coherence_on_mask(p, g, tis))
        ccca.append(coherence_on_mask(p, g, amb))
        sup.append(mirror_suppression(p, g, y, mirror_region_mask(g)))
        dcr.append(dc_residual_np(p, y))
    return dict(ccc=float(np.nanmean(ccc)), ccc_vs_mirror_solution=float(np.nanmean(cccm)),
                ccc_tissue_dom=float(np.nanmean(ccct)), ccc_ambiguous=float(np.nanmean(ccca)),
                mirror_suppression_median=float(np.nanmedian(sup)), dc_residual=float(np.nanmean(dcr)))


# ============================== INTERNAL VALIDATION ==============================

def build_internal_val(vol_list, names, roi, n_bscans=VAL_BSCANS, val_frac=VAL_FRAC, guard=VAL_GUARD):
    """Tiled ROIs of n_bscans B-scans per volume, taken from the reserved block."""
    gts = []
    for arr, nm in zip(vol_list, names):
        nb = arr.shape[0]
        _, v0 = split_volume(nb, val_frac, guard)
        idx = np.unique(np.linspace(v0, nb - 1, min(n_bscans, nb - v0), dtype=int))
        for b in idx:
            for (z0, x0) in roi_origins(arr.shape[1:], roi):
                gts.append(arr[b, z0:z0 + roi, x0:x0 + roi])
    gts = np.stack(gts).astype(np.complex64)
    ys = np.stack([mirrorArtifact(g.astype(np.complex128)).astype(np.complex64) for g in gts])
    print(f'  Internal validation: {len(gts)} ROIs from the reserved blocks of {len(names)} volumes')
    return dict(gt=gts, y=ys)


def roi_run_internal_val(model, cfg, val, steps=VAL_STEPS, amp_dtype=None):
    model.eval()
    var = 'R_direct' if arch_get(model.arch, 'model') == 'regress' else 'G2_dc_full'
    pred = roi_reconstruct(model, val['y'], cfg, var, steps=steps, batch=64, seed=PREVIEW_SEED,
                           amp_dtype=amp_dtype)
    m = quick_metrics(pred, val['gt'], val['y'])
    noop = quick_metrics(val['y'], val['gt'], val['y'])
    model.train()
    return m, noop


def bscan_run_internal_val(model, cfg, val, amp_dtype):
    model.eval()
    pred = bscan_reconstruct(model, val['y'], cfg, None, steps=cfg['val_steps'], seed=PREVIEW_SEED,
                             amp_dtype=amp_dtype)
    m = quick_metrics(pred, val['gt'], val['y'])
    model.train()
    return m


def select_best(history, floor=SUPP_FLOOR):
    """Selection rule: fidelity (CCC) restricted to the evaluations that pass the suppression floor."""
    ok = [h for h in history if h['mirror_suppression_median'] >= floor]
    if ok:
        return max(ok, key=lambda h: h['ccc']), f'max CCC with suppression >= {floor}'
    return max(history, key=lambda h: h['mirror_suppression_median']), 'max suppression (none reaches the floor)'


# ============================== PREVIEWS (test set; monitoring only) ==============================

def load_preview_set(val_dir=VALIDATION_DATA_DIR, roi=ROI_SIZE):
    """One B-scan (the central one) per validation volume, split into ROI tiles."""
    items = []
    files = sorted(glob.glob(os.path.join(val_dir, '*.npy')))
    for fp in files:
        fn = os.path.basename(fp)
        raw = np.load(fp, mmap_mode='r')
        bi = raw.shape[2] // 2
        b = np.asarray(raw[:, :, bi])
        bs = (b[..., 0] + 1j * b[..., 1]) if raw.ndim == 4 else b
        bs = bs.astype(np.complex64)
        tiles = roi_origins(bs.shape, roi)
        if not tiles:
            print(f'  WARNING preview: {fn} {bs.shape} is smaller than the ROI, skipped'); continue
        gts = np.stack([bs[z0:z0 + roi, x0:x0 + roi] for (z0, x0) in tiles])
        ys = np.stack([mirrorArtifact(g.astype(np.complex128)).astype(np.complex64) for g in gts])
        items.append(dict(file=fn, bscan_index=int(bi), shape=bs.shape, tiles=tiles, gt=gts, y=ys))
    return items


def _assemble(rois, tiles, shape, roi):
    Z = max(z0 for z0, _ in tiles) + roi
    X = max(x0 for _, x0 in tiles) + roi
    out = np.zeros((Z, X), dtype=rois.dtype)
    for r, (z0, x0) in zip(rois, tiles):
        out[z0:z0 + roi, x0:x0 + roi] = r
    return out


def _db(a):
    return 20 * np.log10(np.abs(a) + 1e-12)


def roi_run_preview(model, cfg, items, out_dir, tag, roi=ROI_SIZE, steps=SAMPLING_STEPS, amp_dtype=None):
    """Figure + quick metrics on one B-scan per volume of the TEST set. It selects nothing."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    pvars = preview_variants_for(model.arch)
    rows, recon = [], {}
    all_y = np.concatenate([it['y'] for it in items])
    offs = np.cumsum([0] + [len(it['y']) for it in items])
    for var in pvars:
        pred = roi_reconstruct(model, all_y, cfg, var, steps=steps, batch=64, seed=PREVIEW_SEED,
                               amp_dtype=amp_dtype)
        for j, it in enumerate(items):
            p = pred[offs[j]:offs[j + 1]]
            recon[(it['file'], var)] = p
            rows.append(dict(tag=tag, file=it['file'], bscan_index=it['bscan_index'], variant=var,
                             **quick_metrics(p, it['gt'], it['y'])))
    for it in items:
        rows.append(dict(tag=tag, file=it['file'], bscan_index=it['bscan_index'], variant='NoOp_input',
                         **quick_metrics(it['y'], it['gt'], it['y'])))

    cols = ['Input (tiles)', 'GT'] + pvars + ['dphi_z GT', f'dphi_z {pvars[-1]}']
    fig, axs = plt.subplots(len(items), len(cols), figsize=(3.0 * len(cols), 3.0 * len(items)), squeeze=False)
    for j, it in enumerate(items):
        gt = _assemble(it['gt'], it['tiles'], it['shape'], roi)
        yy = _assemble(it['y'], it['tiles'], it['shape'], roi)
        imgs = [yy, gt] + [_assemble(recon[(it['file'], v)], it['tiles'], it['shape'], roi) for v in pvars]
        gdb = _db(gt)
        vmax = np.percentile(gdb, 99.9)
        vmin = max(np.percentile(gdb, 1), vmax - 70)
        for c, im in enumerate(imgs):
            axs[j, c].imshow(_db(im), cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
        for c, im in zip((len(imgs), len(imgs) + 1), (gt, imgs[-1])):
            axs[j, c].imshow(np.angle(im[1:] * np.conj(im[:-1])), cmap='twilight', vmin=-np.pi, vmax=np.pi,
                             aspect='auto')
        for c in range(len(cols)):
            axs[j, c].set_xticks([]); axs[j, c].set_yticks([])
            for z0, x0 in it['tiles']:
                axs[j, c].axhline(z0 - 0.5, color='y', lw=0.3); axs[j, c].axvline(x0 - 0.5, color='y', lw=0.3)
            if j == 0:
                axs[j, c].set_title(cols[c], fontsize=9)
        r2 = [r for r in rows if r['file'] == it['file'] and r['variant'] == pvars[-1]][0]
        axs[j, 0].set_ylabel(f"{it['file'][:22]}\nb={it['bscan_index']}\nCCC={r2['ccc']:.2f} "
                             f"sup={r2['mirror_suppression_median']:.2f}", fontsize=7)
    fig.suptitle(f'Control inference ({tag}) -- EMA, fixed seed {PREVIEW_SEED} -- TEST set, '
                 f'monitoring only', fontsize=11)
    fig.tight_layout()
    png = os.path.join(out_dir, f'preview_{tag}.png')
    fig.savefig(png, dpi=110)
    plt.close(fig)

    csv_path = os.path.join(out_dir, 'preview_metrics.csv')
    new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f'  [preview {tag}] {png}')
    for var in pvars + ['NoOp_input']:
        rr = [r for r in rows if r['variant'] == var]
        print(f'    {var:<11s} CCC={np.mean([r["ccc"] for r in rr]):.3f}  '
              f'CCCtis={np.nanmean([r["ccc_tissue_dom"] for r in rr]):.3f}  '
              f'CCC_vs_S~={np.mean([r["ccc_vs_mirror_solution"] for r in rr]):.3f}  '
              f'sup(med)={np.nanmedian([r["mirror_suppression_median"] for r in rr]):.3f}  '
              f'dc_res={np.mean([r["dc_residual"] for r in rr]):.2e}')
    model.train()
    return png


def bscan_run_preview(model, cfg, items, out_dir, tag, amp_dtype=None):
    """Full B-scan, one per volume of the TEST set. Monitoring only."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    arch = model.arch
    pv = bscan_preview_variants(arch)
    has_gain = arch['kind'] in ('mask', 'dmask')
    ys = np.concatenate([it['y'] for it in items]); gts = np.concatenate([it['gt'] for it in items])
    rows, preds, gains = [], {}, None
    for v in pv:
        res = bscan_reconstruct(model, ys, cfg, v, steps=cfg['val_steps'], seed=PREVIEW_SEED,
                                amp_dtype=amp_dtype, return_gain=has_gain)
        p, gg = res if has_gain else (res, None)
        preds[v] = p
        gains = gg if gains is None else gains
        for i, it in enumerate(items):
            rows.append(dict(tag=tag, file=it['file'], variant=v, **quick_metrics(p[i:i + 1], gts[i:i + 1], ys[i:i + 1])))
    for i, it in enumerate(items):
        rows.append(dict(tag=tag, file=it['file'], variant='NoOp_input',
                         **quick_metrics(ys[i:i + 1], gts[i:i + 1], ys[i:i + 1])))
    cols = ['Input', 'GT'] + pv + ['dphi_z GT', f'dphi_z {pv[-1]}'] + (['gain g'] if has_gain else [])
    fig, axs = plt.subplots(len(items), len(cols), figsize=(3.1 * len(cols), 3.1 * len(items)), squeeze=False)
    for i, it in enumerate(items):
        gdb = _db(gts[i]); vmax = np.percentile(gdb, 99.9); vmin = max(np.percentile(gdb, 1), vmax - 70)
        imgs = [ys[i], gts[i]] + [preds[v][i] for v in pv]
        for c, im in enumerate(imgs):
            axs[i, c].imshow(_db(im), cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
        for c, im in zip((len(imgs), len(imgs) + 1), (gts[i], preds[pv[-1]][i])):
            axs[i, c].imshow(np.angle(im[1:] * np.conj(im[:-1])), cmap='twilight', vmin=-np.pi, vmax=np.pi, aspect='auto')
        if has_gain:
            axs[i, -1].imshow(gains[i], cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        for c in range(len(cols)):
            axs[i, c].set_xticks([]); axs[i, c].set_yticks([])
            axs[i, c].axhline(BSCAN_SIZE // 2 - 0.5, color='y', lw=0.4)
            if i == 0:
                axs[i, c].set_title(cols[c], fontsize=9)
        r = [q for q in rows if q['file'] == it['file'] and q['variant'] == pv[-1]][0]
        axs[i, 0].set_ylabel(f"{it['file'][:22]}\nCCC={r['ccc']:.2f} sup={r['mirror_suppression_median']:.2f}", fontsize=7)
    fig.suptitle(f'{arch["kind"]} ({tag}) -- EMA -- TEST set, monitoring only. '
                 f'Yellow line = zero delay', fontsize=10)
    fig.tight_layout()
    png = os.path.join(out_dir, f'preview_{tag}.png')
    fig.savefig(png, dpi=100); plt.close(fig)
    path = os.path.join(out_dir, 'preview_metrics.csv')
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f'  [preview {tag}] {png}')
    for v in pv + ['NoOp_input']:
        rr = [q for q in rows if q['variant'] == v]
        print(f'    {v:<12s} CCC={np.mean([q["ccc"] for q in rr]):.3f}  CCCtis={np.nanmean([q["ccc_tissue_dom"] for q in rr]):.3f}  '
              f'CCC_vs_S~={np.mean([q["ccc_vs_mirror_solution"] for q in rr]):.3f}  '
              f'sup(med)={np.nanmedian([q["mirror_suppression_median"] for q in rr]):.3f}')
    model.train()


# ============================== TRAINING (shared helpers) ==============================

def set_seed(seed=0, deterministic=False):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def lr_lambda_factory(total_iters, warmup_iters, lr, lr_min):
    def f(step):
        if step < warmup_iters:
            return (step + 1) / max(1, warmup_iters)
        prog = min(1.0, (step - warmup_iters) / max(1, total_iters - warmup_iters))
        return (lr_min / lr) + (1 - lr_min / lr) * 0.5 * (1 + math.cos(math.pi * prog))
    return f


def load_training_volumes(args, required_shape=None):
    """Loads TRAIN_FILES from --data and splits each volume into training / guard / internal validation."""
    vols, names = [], []
    for fn in TRAIN_FILES:
        p = os.path.join(args.data, fn)
        if not os.path.exists(p):
            print(f'  MISSING {fn}, skipped'); continue
        v = load_volume_bscans(p, args.max_per_volume)
        if required_shape is not None and tuple(v.shape[1:]) != tuple(required_shape):
            print(f'  {fn}: B-scan {v.shape[1:]} != {required_shape[0]}x{required_shape[1]}, skipped'); continue
        vols.append(v); names.append(fn)
    views = []
    for arr, nm in zip(vols, names):
        n_tr, v0 = split_volume(arr.shape[0], args.val_frac, args.val_guard)
        views.append(arr[:n_tr])        # view, no copy
        print(f'  {nm}: training [0,{n_tr})  guard [{n_tr},{v0})  internal validation [{v0},{arr.shape[0]})')
    return vols, names, views


def make_ema(model, decay):
    from timm.utils import ModelEmaV3
    try:
        return ModelEmaV3(model, decay=decay, use_warmup=True, warmup_power=0.75), True
    except TypeError:            # timm versions without EMA warmup
        print('  WARNING: this timm version does not support EMA warmup; a fixed decay is used.')
        return ModelEmaV3(model, decay=decay), False


# ============================== roi FAMILY: TRAINING ==============================

def norm_cfg_from_args(args):
    return dict(env_sigma_z=args.env_sigma_z, env_sigma_x=args.env_sigma_x, env_scale=args.env_scale,
                env_floor_db=args.env_floor_db, cond_env=not args.no_cond_env, x0_clamp=args.x0_clamp,
                aug_phase=args.aug_phase, aug_flip=args.aug_flip, roi=args.roi, norm='envelope')


def arch_from_args(args):
    model_type, domain, head = args.model, args.domain, args.head
    if model_type == 'diffusion' and domain != 'unfolded':
        raise ValueError('Diffusion in the folded domain is not implemented.')
    if head == 'wiener' and domain != 'folded':
        raise ValueError('--head wiener requires --domain folded')
    depth = args.depth_chan if args.depth_chan is not None else (model_type == 'regress')
    cond_env = not args.no_cond_env
    n_cond = cond_channels(domain, cond_env, depth)
    return dict(family='roi', model=model_type, domain=domain, head=head, depth_chan=bool(depth),
                cond_env=cond_env, num_heads=args.num_heads, hires_attn=bool(args.hires_attn),
                in_channels=n_cond + (2 if model_type == 'diffusion' else 0),
                out_channels=1 if head == 'wiener' else 2,
                roi_size=args.roi, channels=[64, 128, 256, 512, 512, 384],
                phase_loss=args.phase_loss if model_type == 'diffusion' else 'none',
                lambda_phase=float(args.lambda_phase) if model_type == 'diffusion' else 0.0)


@torch.no_grad()
def report_data_stats(dl, cfg, n_batches=8):
    xs = []
    for k, gt2 in enumerate(dl):
        b = gpu_make_batch(gt2, cfg, augment=True)
        xs.append(c2ch(b['Sn']).flatten().cpu())
        if k + 1 >= n_batches:
            break
    x = torch.cat(xs)
    q = torch.quantile(x.abs()[::max(1, x.numel() // 2_000_000)], torch.tensor([0.99, 0.999]))
    print(f'  [data] x0: std={x.std():.3f}  |x0| p99={q[0]:.2f} p99.9={q[1]:.2f} max={x.abs().max():.2f}  '
          f'frac>|clamp={cfg["x0_clamp"]}|={(x.abs() > cfg["x0_clamp"]).float().mean():.2e}')


T_BINS = 4   # diffusion loss per quartile of t: [0,250) [250,500) [500,750) [750,1000)


def roi_train(args):
    set_seed(args.seed, deterministic=args.deterministic)
    cfg = norm_cfg_from_args(args)
    arch = arch_from_args(args)
    is_reg = arch['model'] == 'regress'
    print(f'roi family | model: {arch["model"]} | domain {arch["domain"]} | head {arch["head"]} | '
          f'input channels {arch["in_channels"]} | output {arch["out_channels"]}')

    print('Loading training volumes (clean B-scans only)...')
    vols, names, train_views = load_training_volumes(args)
    ds = CleanBscanDataset(train_views, roi=args.roi, mode=args.roi_mode, crops_per_bscan=args.crops_per_bscan)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
                    prefetch_factor=4 if args.workers else None)
    val = build_internal_val(vols, names, args.roi, args.val_bscans, args.val_frac, args.val_guard)
    print(f'{len(ds)} crops/epoch -> {len(dl)} iterations')

    model = build_model_roi(arch)
    if is_reg:
        # Output layer at zero: the regressor STARTS at the identity (d = 0 -> S = y, the no-op, which
        # is also the MMSE estimator under the symmetry S <-> S~) and only learns deviations from it.
        nn.init.zeros_(model.output_conv.weight); nn.init.zeros_(model.output_conv.bias)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.wd)
    total_iters = args.epochs * len(dl)
    warmup_iters = int(args.warmup_epochs * len(dl))
    sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(total_iters, warmup_iters, args.lr, args.lr_min))
    ema, ema_takes_step = make_ema(model, args.ema_decay)
    ema.module.arch = dict(arch)
    ddpm = DDPM_Scheduler()
    amp_dtype = torch.bfloat16 if args.amp else None
    print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M | '
          f'{total_iters} iterations, warmup {warmup_iters}, LR {args.lr:g} -> {args.lr_min:g}')

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, 'checkpoint.pth')
    log_path = os.path.join(args.out, 'train_log.csv')
    val_path = os.path.join(args.out, 'val_metrics.csv')
    start_ep, step, history = 0, 0, []
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck.get('arch', {}).get('model') != arch['model'] or ck['arch'].get('domain') != arch['domain'] \
                or ck['arch'].get('head') != arch['head']:
            raise RuntimeError('The checkpoint in --out belongs to another configuration: use a different --out.')
        model.load_state_dict(ck['weights']); ema.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['optimizer']); sch.load_state_dict(ck['lr_sched'])
        start_ep, step, history = ck['epoch'], ck['global_step'], ck.get('val_history', [])
        print(f'RESUMING from epoch {start_ep} (step {step})')
    elif args.resume:
        print('--resume requested but there is no checkpoint; starting from scratch.')

    print(f'Config: {vars(args)}')
    report_data_stats(dl, cfg)
    preview_items = load_preview_set(args.data_val, args.roi) if args.data_val else []

    def save_ckpt(ep):
        torch.save({'weights': model.state_dict(), 'ema': ema.state_dict(),
                    'ema_weights': ema.module.state_dict(), 'optimizer': opt.state_dict(),
                    'lr_sched': sch.state_dict(), 'epoch': ep, 'global_step': step,
                    'val_history': history, 'arch': arch, 'norm_cfg': cfg, 'args': vars(args)}, ckpt_path)

    def save_ema(path, ep, extra=None):
        torch.save({'ema_weights': ema.module.state_dict(), 'arch': arch, 'norm_cfg': cfg, 'epoch': ep,
                    **(extra or {})}, path)

    for ep in range(start_ep, args.epochs):
        model.train()
        tot, n, t0, skipped = 0.0, 0, time.time(), 0
        gn_sum, gn_max = 0.0, 0.0
        tb_sum, tb_cnt = np.zeros(T_BINS), np.zeros(T_BINS)
        for gt2 in dl:
            b = gpu_make_batch(gt2, cfg, augment=True)
            Sn = b['Sn']
            if is_reg:
                Sn_hat = regress_forward(model, b['yn'], b['envch'], amp_dtype)
                loss = (Sn_hat - Sn).abs().pow(2).mean()
            else:
                x0 = c2ch(Sn)
                cond = build_condition(b['yn'], b['envch'], arch)
                B = x0.shape[0]
                t = torch.randint(0, NUM_TIME_STEPS, (B,))
                eps = torch.randn_like(x0)
                a = ddpm.alpha[t].view(B, 1, 1, 1).to(device)
                xt = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * eps
                with _autocast(amp_dtype):
                    eps_hat = model(xt, t, cond)
                eps_hat = eps_hat.float()
                per = (eps_hat - eps).pow(2).mean(dim=(1, 2, 3))
                loss = per.mean()
                with torch.no_grad():
                    bins = (t * T_BINS // NUM_TIME_STEPS).clamp(max=T_BINS - 1).numpy()
                    pv = per.detach().cpu().numpy()
                    np.add.at(tb_sum, bins, pv); np.add.at(tb_cnt, bins, 1)
                if arch['lambda_phase'] > 0 and arch['phase_loss'] != 'none':
                    x0_hat = ((xt - torch.sqrt(1 - a) * eps_hat) / torch.sqrt(a)).clamp(-args.x0_clamp, args.x0_clamp)
                    fn = phase_loss_circular if arch['phase_loss'] == 'circular' else phase_loss_coherence
                    loss = loss + arch['lambda_phase'] * fn(x0_hat, x0, a.view(-1))
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                      args.grad_clip if args.grad_clip > 0 else float('inf')))
            opt.step(); sch.step(); step += 1
            if ema_takes_step:
                ema.update(model, step=step)
            else:
                ema.update(model)
            tot += loss.item(); n += 1
            gn_sum += gn; gn_max = max(gn_max, gn)
        dt = time.time() - t0
        tb = tb_sum / np.maximum(tb_cnt, 1)
        print(f'Epoch {ep+1}/{args.epochs} | loss {tot/max(n,1):.5f} | |g| mean {gn_sum/max(n,1):.3f} '
              f'max {gn_max:.2f} | lr {sch.get_last_lr()[0]:.2e} | skipped {skipped}'
              + ('' if is_reg else ' | per t: ' + ' '.join(f'{v:.4f}' for v in tb))
              + f' | {dt/60:.1f} min | ~{(args.epochs-ep-1)*dt/3600:.1f} h left', flush=True)
        new = not os.path.exists(log_path)
        with open(log_path, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['epoch', 'loss', 'grad_norm_mean', 'grad_norm_max', 'lr', 'skipped', 'minutes']
                           + [f'loss_t{k}' for k in range(T_BINS)])
            w.writerow([ep + 1, tot / max(n, 1), gn_sum / max(n, 1), gn_max, sch.get_last_lr()[0], skipped,
                        dt / 60] + list(tb))

        is_last = (ep + 1) == args.epochs
        if (ep + 1) % args.val_every == 0 or is_last:
            m, noop = roi_run_internal_val(ema.module, cfg, val, steps=args.val_steps, amp_dtype=amp_dtype)
            history.append(dict(epoch=ep + 1, **m))
            best, rule = select_best(history, args.supp_floor)
            is_best = best['epoch'] == ep + 1
            if is_best:
                save_ema(os.path.join(args.out, 'ema_best.pth'), ep + 1, dict(selection=best, rule=rule))
            new = not os.path.exists(val_path)
            with open(val_path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['epoch'] + list(m.keys()) + ['selected_so_far'])
                if new:
                    w.writeheader()
                w.writerow(dict(epoch=ep + 1, **m, selected_so_far=int(is_best)))
            print(f'  [internal val ep{ep+1}] CCC={m["ccc"]:.3f} (no-op {noop["ccc"]:.3f})  '
                  f'CCCtis={m["ccc_tissue_dom"]:.3f} (no-op {noop["ccc_tissue_dom"]:.3f})  '
                  f'CCCamb={m["ccc_ambiguous"]:.3f}  CCC_vs_S~={m["ccc_vs_mirror_solution"]:.3f}  '
                  f'sup(med)={m["mirror_suppression_median"]:.3f}  dc_res={m["dc_residual"]:.2e}'
                  f'{"  <- BEST" if is_best else ""}  [{rule}; best ep{best["epoch"]}]', flush=True)
        save_ckpt(ep + 1)
        if args.snapshot_every and (ep + 1) % args.snapshot_every == 0:
            save_ema(os.path.join(args.out, f'ema_ep{ep+1:03d}.pth'), ep + 1)
        if preview_items and args.preview_every and (ep + 1) % args.preview_every == 0 and not is_last:
            roi_run_preview(ema.module, cfg, preview_items, os.path.join(args.out, 'previews'),
                            f'ep{ep+1:03d}', roi=args.roi, steps=args.steps, amp_dtype=amp_dtype)

    # --- END: last and selected checkpoints, always (also when resuming a completed training) ---
    save_ema(os.path.join(args.out, 'ema_final.pth'), args.epochs)
    if history:
        best, rule = select_best(history, args.supp_floor)
        with open(os.path.join(args.out, 'selection.json'), 'w') as f:
            json.dump(dict(rule=rule, selected=best, history=history), f, indent=2)
        print(f'Selected checkpoint: epoch {best["epoch"]} ({rule})')
    if preview_items:
        roi_run_preview(ema.module, cfg, preview_items, os.path.join(args.out, 'previews'),
                        f'final_ep{args.epochs:03d}', roi=args.roi, steps=args.steps, amp_dtype=amp_dtype)
        bp = os.path.join(args.out, 'ema_best.pth')
        if os.path.exists(bp):
            mb, _, _, epb = load_trained_path(bp)
            roi_run_preview(mb, cfg, preview_items, os.path.join(args.out, 'previews'),
                            f'best_ep{epb:03d}', roi=args.roi, steps=args.steps, amp_dtype=amp_dtype)
    print('Training finished. Full evaluation: evaluate_diffusion_D.py')


# ============================== bscan FAMILY: TRAINING ==============================

def bscan_train(args):
    set_seed(args.seed, deterministic=args.deterministic)
    arch = dict(family='bscan', kind=args.kind, encoder=args.encoder, roi_size=BSCAN_SIZE,
                blocks=list(DIFF_BLOCKS), layers=2)
    cfg = dict(env_sigma_z=ENV_SIGMA_Z, env_sigma_x=ENV_SIGMA_X, env_scale=ENV_SCALE, env_floor_db=ENV_FLOOR_DB,
               aug_phase=args.aug_phase, aug_flip=args.aug_flip, zpd_shift=args.zpd_shift,
               zpd_shift_p=args.zpd_shift_p, mask_sigma=args.mask_sigma, val_steps=args.val_steps)
    w = dict(cmse=args.w_cmse, logamp=args.w_logamp)
    diff = bscan_is_diffusion(arch)
    print(f'bscan family | kind={args.kind} | ' + (
        f'encoder={args.encoder} ({"not pretrained" if args.no_pretrained else "ImageNet-pretrained"})'
        if not diff else 'diffusers UNet2DModel'))

    print('Loading training volumes (clean B-scans only)...')
    vols, names, views = load_training_volumes(args, required_shape=(BSCAN_SIZE, BSCAN_SIZE))
    ds = CleanBscanDataset(views, roi=BSCAN_SIZE, mode='tile')
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers,
                    pin_memory=True, persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    val = build_internal_val(vols, names, BSCAN_SIZE, args.val_bscans, args.val_frac, args.val_guard)
    items = load_preview_set(args.data_val, BSCAN_SIZE) if args.data_val else []
    print(f'{len(ds)} B-scans/epoch -> {len(dl)} iterations')

    model = build_model_bscan(arch, pretrained=(not diff) and not args.no_pretrained)
    if not diff:
        zero_init_output(model)
    opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.wd)
    total, warm = args.epochs * len(dl), int(args.warmup_epochs * len(dl))
    sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(total, warm, args.lr, args.lr_min))
    ema, ema_step = make_ema(model, args.ema_decay)
    ema.module.arch = dict(arch)
    tr_sched = make_schedulers()[0] if diff else None
    amp = torch.bfloat16 if args.amp else None
    print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M | {total} iterations')

    os.makedirs(args.out, exist_ok=True)
    ck_path = os.path.join(args.out, 'checkpoint.pth')
    start, step, hist = 0, 0, []
    if args.resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        if ck['arch'].get('kind') != arch['kind']:
            raise RuntimeError('The checkpoint in --out belongs to another configuration: use a different --out.')
        model.load_state_dict(ck['weights']); ema.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['optimizer']); sch.load_state_dict(ck['lr_sched'])
        start, step, hist = ck['epoch'], ck['global_step'], ck.get('val_history', [])
        print(f'RESUMING from epoch {start}')
    print(f'Config: {vars(args)}')

    def save_ema(path, ep, extra=None):
        torch.save({'ema_weights': ema.module.state_dict(), 'arch': arch, 'norm_cfg': cfg, 'epoch': ep, **(extra or {})}, path)

    log_path = os.path.join(args.out, 'train_log.csv')
    bad_streak = 0
    for ep in range(start, args.epochs):
        model.train()
        t0, n, tot, skipped, gsum, gmax = time.time(), 0, 0.0, 0, 0.0, 0.0
        parts = {}
        for gt2 in dl:
            S, y = gpu_batch(gt2, cfg, augment=True)
            if not diff:
                S_hat, _ = forward_det(model, y, cfg, amp)
                loss, lp = det_loss(S_hat, S, y, cfg, w)
            else:
                feats, (q01, q999) = input_features(y)
                # dmask: antisymmetric power-ratio gain; dlogamp: [log amplitude, cos, sin] of the target
                x0 = ratio_gain_target(S, cfg['mask_sigma']).unsqueeze(1).float() if args.kind == 'dmask' \
                    else logamp_target(S, q01, q999)
                noise = torch.randn_like(x0)
                t = torch.randint(0, DIFF_STEPS_TRAIN, (x0.shape[0],), device=device)
                xt = tr_sched.add_noise(x0, noise, t)
                with _autocast(amp):
                    v = model(xt, t, feats)
                loss = F.mse_loss(v.float(), tr_sched.get_velocity(x0, noise, t))
                lp = dict(v_mse=loss.item())
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1; bad_streak += 1
            else:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(gn):           # non-finite gradient: discarded BEFORE touching the weights
                    opt.zero_grad(set_to_none=True); skipped += 1; bad_streak += 1
                else:
                    opt.step(); sch.step(); step += 1; bad_streak = 0
                    ema.update(model, step=step) if ema_step else ema.update(model)
                    tot += loss.item(); n += 1; gsum += float(gn); gmax = max(gmax, float(gn))
                    for k_, v_ in lp.items():
                        parts[k_] = parts.get(k_, 0.0) + v_
            if bad_streak >= args.max_bad_steps:
                raise SystemExit(f'ABORTED: {bad_streak} consecutive non-finite steps in epoch {ep+1}. '
                                 f'Check the LR / data. Last healthy checkpoint in {ck_path}')
        dt = time.time() - t0
        pstr = ' '.join(f'{k_} {v_ / max(n, 1):.4f}' for k_, v_ in parts.items())
        print(f'Epoch {ep+1}/{args.epochs} | loss {tot/max(n,1):.5f} ({pstr}) | |g| mean {gsum/max(n,1):.3f} '
              f'max {gmax:.2f} | lr {sch.get_last_lr()[0]:.2e} | discarded {skipped} | {dt/60:.1f} min | '
              f'~{(args.epochs-ep-1)*dt/3600:.1f} h left', flush=True)
        new = not os.path.exists(log_path)
        with open(log_path, 'a', newline='') as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(['epoch', 'loss', 'grad_mean', 'grad_max', 'lr', 'skipped', 'minutes'] + list(parts))
            wr.writerow([ep + 1, tot / max(n, 1), gsum / max(n, 1), gmax, sch.get_last_lr()[0], skipped, dt / 60]
                        + [v_ / max(n, 1) for v_ in parts.values()])
        last = (ep + 1) == args.epochs
        if (ep + 1) % args.val_every == 0 or last:
            m = bscan_run_internal_val(ema.module, cfg, val, amp)
            hist.append(dict(epoch=ep + 1, **m))
            best, rule = select_best(hist, args.supp_floor)
            isb = best['epoch'] == ep + 1
            if isb:
                save_ema(os.path.join(args.out, 'ema_best.pth'), ep + 1, dict(selection=best, rule=rule))
            print(f'  [internal val ep{ep+1}] CCC={m["ccc"]:.3f}  CCCtis={m["ccc_tissue_dom"]:.3f}  '
                  f'CCCamb={m["ccc_ambiguous"]:.3f}  CCC_vs_S~={m["ccc_vs_mirror_solution"]:.3f}  '
                  f'sup(med)={m["mirror_suppression_median"]:.3f}{"  <- BEST" if isb else ""}  [best ep{best["epoch"]}]',
                  flush=True)
        torch.save({'weights': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': opt.state_dict(),
                    'lr_sched': sch.state_dict(), 'epoch': ep + 1, 'global_step': step, 'val_history': hist,
                    'arch': arch, 'norm_cfg': cfg, 'args': vars(args)}, ck_path)
        if args.snapshot_every and (ep + 1) % args.snapshot_every == 0:
            save_ema(os.path.join(args.out, f'ema_ep{ep+1:03d}.pth'), ep + 1)
        if items and args.preview_every and (ep + 1) % args.preview_every == 0 and not last:
            bscan_run_preview(ema.module, cfg, items, os.path.join(args.out, 'previews'), f'ep{ep+1:03d}', amp)

    save_ema(os.path.join(args.out, 'ema_final.pth'), args.epochs)
    if hist:
        best, rule = select_best(hist, args.supp_floor)
        with open(os.path.join(args.out, 'selection.json'), 'w') as f:
            json.dump(dict(rule=rule, selected=best, history=hist), f, indent=2)
        print(f'Selected checkpoint: epoch {best["epoch"]} ({rule})')
    if items:
        bscan_run_preview(ema.module, cfg, items, os.path.join(args.out, 'previews'), f'final_ep{args.epochs:03d}', amp)
        bp = os.path.join(args.out, 'ema_best.pth')
        if os.path.exists(bp):
            mb, _, cb, eb = load_trained_path(bp)
            bscan_run_preview(mb, cb, items, os.path.join(args.out, 'previews'), f'best_ep{eb:03d}', amp)
    print('Training finished. Full evaluation: evaluate_diffusion_D.py')


# ============================== LOADING TRAINED MODELS ==============================

def load_trained_path(path):
    """(EMA model in eval mode, arch, norm_cfg, epoch) from a .pth written by this script (either family)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if 'arch' not in ck or 'ema_weights' not in ck:
        raise RuntimeError(f'{path} does not contain EMA weights written by this script')
    arch, cfg = ck['arch'], ck['norm_cfg']
    if arch_family(arch) == 'bscan':
        model = build_model_bscan(arch, pretrained=False)
    else:
        if 'cond_env' not in arch:                 # checkpoints written before this field existed
            arch = dict(arch, cond_env=cfg.get('cond_env', True))
        model = build_model_roi(arch)
    model.load_state_dict(ck['ema_weights'])
    model.eval()
    return model, arch, cfg, ck.get('epoch')


def load_trained(out_dir, which='best'):
    order = ['ema_best.pth', 'ema_final.pth'] if which == 'best' else ['ema_final.pth', 'ema_best.pth']
    for fn in order + ['checkpoint.pth']:
        p = os.path.join(out_dir, fn)
        if os.path.exists(p):
            return load_trained_path(p)
    raise FileNotFoundError(f'No checkpoint in {out_dir}')


# ============================== SELF-TESTS ==============================

def _reporter():
    state = dict(ok=True)

    def rep(tag, val, tol):
        good = bool(val < tol)
        state['ok'] &= good
        print(f'  {tag:<52s}: {val:.2e}  {"OK" if good else "FAIL"}')
    return rep, state


def roi_selftest():
    print('=' * 74); print('SELF-TEST (roi family)'); print('=' * 74)
    rep, state = _reporter()
    rng = np.random.default_rng(0)
    N, X = 256, 64
    R = np.full((N, 1), 1e-4); R[150:200] = 1.0; R[60:70] = 100.0
    S = (np.sqrt(R / 2) * (rng.standard_normal((N, X)) + 1j * rng.standard_normal((N, X))))

    a = mirrorArtifact(S)
    St = torch.from_numpy(S)[None].to(torch.complex128)
    rep('(a) torch_mirror vs numpy', np.linalg.norm(a - torch_mirror(St).numpy()[0]) / np.linalg.norm(a), 1e-10)
    y64 = torch_mirror(St)
    e64 = envelope_t(y64)
    rep('(b) symmetric envelope', (torch.abs(e64 - mirror_of_t(e64)).max() / e64.max()).item(), 1e-12)
    rep('    M(S/e) == M(S)/e  (float64)', (torch.linalg.vector_norm(torch_mirror(St / e64) - y64 / e64) /
                                            torch.linalg.vector_norm(y64 / e64)).item(), 1e-12)
    cfg = dict(env_sigma_z=ENV_SIGMA_Z, env_sigma_x=ENV_SIGMA_X, env_scale=ENV_SCALE, env_floor_db=ENV_FLOOR_DB)
    S32 = St.to(torch.complex64)
    y = torch_mirror(S32)
    yn, e, envch = normalize_measurement(y, cfg)
    xc = dc_project(torch.randn(1, 2, N, X), yn)
    rep('(c) DC from pure noise (unfolded domain)', (torch.linalg.vector_norm(torch_mirror(ch2c(xc)) - yn) /
                                                     torch.linalg.vector_norm(yn)).item(), 1e-4)
    rep('(d) M(S~) == M(S)  (exact ambiguity)',
        np.linalg.norm(mirrorArtifact(np.conj(mirror_of(S))) - a) / np.linalg.norm(a), 1e-10)
    rep('(e) round trip S -> S/e -> S', (torch.linalg.vector_norm(ch2c(c2ch(S32 / e)) * e - S32) /
                                         torch.linalg.vector_norm(S32)).item(), 1e-5)
    # (f) folding
    L, Up = fold_t(St)
    Y = y64[:, N // 2:, :]
    rep('(f) folding: Y == (L + U\')/2', (torch.linalg.vector_norm(Y - (L + Up) / 2) /
                                          torch.linalg.vector_norm(Y)).item(), 1e-12)
    rec = unfold_t(L, Up, y64)
    rep('    unfold(fold(S)) == S  (rows 1..N-1)', (torch.linalg.vector_norm((rec - St)[:, 1:]) /
                                                   torch.linalg.vector_norm(St)).item(), 1e-12)
    d = torch.randn_like(Y)
    Sd = unfold_t(Y + d, Y - d, y64)
    rep('(g) DC by construction (folded): M(S_d) == y', (torch.linalg.vector_norm(torch_mirror(Sd) - y64) /
                                                         torch.linalg.vector_norm(y64)).item(), 1e-12)
    g = torch.tanh(torch.randn(Y.shape, dtype=torch.float64))
    Sw = unfold_t(Y * (1 + g), Y * (1 - g), y64)
    rep('    same with the Wiener head', (torch.linalg.vector_norm(torch_mirror(Sw) - y64) /
                                          torch.linalg.vector_norm(y64)).item(), 1e-12)
    Ln, Upn = fold_np(S)
    rep('    fold/unfold numpy == torch', np.linalg.norm(unfold_np(Ln, Upn, a) - rec.numpy()[0]) /
        np.linalg.norm(S), 1e-12)
    # (h) regressors (reduced U-Net) with DC by construction
    small = [32, 64, 128, 256, 256, 192]
    for dom, head in (('unfolded', 'null'), ('folded', 'null'), ('folded', 'wiener')):
        ar = dict(family='roi', model='regress', domain=dom, head=head, depth_chan=True, cond_env=True,
                  num_heads=NUM_HEADS, hires_attn=False, in_channels=cond_channels(dom, True, True),
                  out_channels=1 if head == 'wiener' else 2, channels=small)
        m = build_model_roi(ar).eval()       # the model lives on `device` (GPU on the cluster)
        yn_d, env_d = yn.to(device), envch.to(device)
        with torch.no_grad():
            Sh = regress_forward(m, yn_d, env_d)
        rep(f'(h) regressor {dom}/{head}: ||M(S_n)-y_n||', (torch.linalg.vector_norm(torch_mirror(Sh) - yn_d) /
                                                           torch.linalg.vector_norm(yn_d)).item(), 1e-4)
    ar = dict(family='roi', model='diffusion', num_heads=NUM_HEADS, hires_attn=False, in_channels=5, channels=small)
    m = build_model_roi(ar).eval()
    with torch.no_grad():
        o = m(torch.randn(2, 2, 64, 64, device=device), torch.tensor([0, 999]), torch.randn(2, 3, 64, 64, device=device))
    print(f'  (i) diffusion U-Net (reduced) forward                : {tuple(o.shape)}  OK')
    print('=' * 74 + ('\n' if state['ok'] else '\n  THERE ARE FAILURES: do not launch training.\n'))
    return state['ok']


def bscan_selftest():
    print('=' * 74); print('SELF-TEST (bscan family)'); print('=' * 74)
    rep, state = _reporter()
    rng = np.random.default_rng(0)
    N, X = 128, 64
    R = np.full((N, 1), 1e-4); R[70:110] = 1.0; R[20:26] = 50.0
    S = torch.from_numpy(np.sqrt(R / 2) * (rng.standard_normal((N, X)) + 1j * rng.standard_normal((N, X))))[None].to(torch.complex128).to(device)
    y = torch_mirror(S)
    o = torch.randn(1, N, X, dtype=torch.float64, device=device)
    m = mask_from_logits(o)
    rep('(a) mask: m(z) + m(-z) == 1', (m + mirror_of_t(m) - 1).abs().max().item(), 1e-12)
    rep('    exact DC with S = 2 m y', ((torch_mirror(2 * m * y) - y).abs().max() / y.abs().max()).item(), 1e-12)
    g = ratio_gain_target(S, 1.0)
    rep('(b) antisymmetric target g*', (g + mirror_of_t(g)).abs().max().item(), 1e-10)
    rep('    exact DC with S = (1 + g*) y', ((torch_mirror((1 + g) * y) - y).abs().max() / y.abs().max()).item(), 1e-12)
    feats, (q01, q999) = input_features(y.to(torch.complex64))
    ph = torch.complex(feats[:, 1], feats[:, 2])
    rep('(c) phase recoverable from the input: y == |y| (cos + i sin)',
        ((y.abs().float() * ph - y.to(torch.complex64)).abs().max() / y.abs().max()).item(), 1e-5)
    x = logamp_target(S.to(torch.complex64), q01, q999)
    Sr = logamp_to_complex(x, q01, q999)
    msk = (x[:, 0] > -1.99) & (x[:, 0] < 1.99)
    rep('(d) dlogamp: [log-amp, cos, sin] -> complex (round trip)',
        ((Sr - S.to(torch.complex64)).abs()[msk].max() / S.abs().max()).item(), 1e-4)
    for kind in KINDS:
        ar = dict(family='bscan', kind=kind, encoder='resnet18', roi_size=BSCAN_SIZE,
                  blocks=[32, 32, 64, 64, 64, 64], layers=1)
        mm = build_model_bscan(ar).eval()
        if kind in ('mask', 'null'):
            zero_init_output(mm)
            yy = y.to(torch.complex64).expand(2, N, X).contiguous()
            Sh, _ = forward_det(mm, yy, dict(env_sigma_z=ENV_SIGMA_Z, env_sigma_x=ENV_SIGMA_X,
                                             env_scale=ENV_SCALE, env_floor_db=ENV_FLOOR_DB))
            rep(f'(e) {kind}: zero-initialized output == no-op', ((Sh - yy).abs().max() / yy.abs().max()).item(), 1e-5)
        else:
            with torch.no_grad():
                out = mm(torch.randn(1, 1 if kind == 'dmask' else 3, 64, 64, device=device),
                         torch.tensor([5], device=device), torch.randn(1, 4, 64, 64, device=device))
            print(f'  (f) {kind}: UNet2DModel forward {tuple(out.shape)}  OK')
    print('=' * 74 + ('\n' if state['ok'] else '\n  THERE ARE FAILURES: do not launch training.\n'))
    return state['ok']


def run_selftest(family):
    return bscan_selftest() if family == 'bscan' else roi_selftest()


# ============================== MAIN ==============================

def build_argparser():
    ap = argparse.ArgumentParser(
        description='Diffusion models and regressors for complex-field mirror-artifact suppression.')
    ap.add_argument('--stage', choices=['selftest', 'train', 'preview'], default='selftest')
    ap.add_argument('--family', choices=['bscan', 'roi'], default='bscan',
                    help='bscan: full 512x512 B-scan with amplitude/phase input (paper). '
                         'roi: square ROIs with envelope-normalized complex field.')
    ap.add_argument('--data', default=TRAIN_DATA_DIR, help='folder that contains TRAIN_FILES')
    ap.add_argument('--data-val', default=VALIDATION_DATA_DIR, help='TEST set, only used for previews')
    ap.add_argument('--out', default=None, help='run folder (default: runs/D_<family>)')
    ap.add_argument('--which', choices=['best', 'final'], default='best', help='only for --stage preview')
    # bscan family
    ap.add_argument('--kind', choices=KINDS, default='dlogamp', help='bscan family formulation')
    ap.add_argument('--encoder', default=ENC_DEFAULT, help='bscan regressors: timm/smp encoder')
    ap.add_argument('--no-pretrained', action='store_true', help='bscan regressors: random encoder init')
    ap.add_argument('--zpd-shift', type=int, default=64, help='maximum zero-delay shift in rows (0 = no augmentation)')
    ap.add_argument('--zpd-shift-p', type=float, default=0.7)
    ap.add_argument('--mask-sigma', type=float, default=1.0, help='power smoothing of the dmask target')
    ap.add_argument('--w-cmse', type=float, default=1.0)
    ap.add_argument('--w-logamp', type=float, default=0.1)
    ap.add_argument('--max-bad-steps', type=int, default=200)
    # roi family: model
    ap.add_argument('--model', choices=['diffusion', 'regress'], default='diffusion')
    ap.add_argument('--domain', choices=['unfolded', 'folded'], default='unfolded')
    ap.add_argument('--head', choices=['null', 'wiener'], default='null')
    ap.add_argument('--depth-chan', dest='depth_chan', action='store_true', default=None)
    ap.add_argument('--no-depth-chan', dest='depth_chan', action='store_false')
    ap.add_argument('--num-heads', type=int, default=NUM_HEADS)
    ap.add_argument('--hires-attn', action='store_true', default=HIRES_ATTENTION)
    ap.add_argument('--no-hires-attn', dest='hires_attn', action='store_false')
    # roi family: data and normalization
    ap.add_argument('--roi', type=int, default=ROI_SIZE,
                    help='ROI side (512 = full B-scan; 256 for quick tests or GPUs with limited VRAM)')
    ap.add_argument('--roi-mode', choices=['random', 'tile', 'center'], default=ROI_MODE)
    ap.add_argument('--crops-per-bscan', type=int, default=CROPS_PER_BSCAN)
    ap.add_argument('--env-sigma-z', type=float, default=ENV_SIGMA_Z)
    ap.add_argument('--env-sigma-x', type=float, default=ENV_SIGMA_X)
    ap.add_argument('--env-scale', type=float, default=ENV_SCALE)
    ap.add_argument('--env-floor-db', type=float, default=ENV_FLOOR_DB)
    ap.add_argument('--x0-clamp', type=float, default=X0_CLAMP)
    ap.add_argument('--no-cond-env', action='store_true')
    # roi family: diffusion
    ap.add_argument('--phase-loss', choices=['none', 'circular', 'coherence'], default='circular')
    ap.add_argument('--lambda-phase', type=float, default=0.0)
    ap.add_argument('--steps', type=int, default=SAMPLING_STEPS, help='DDIM steps of the previews')
    ap.add_argument('--channels-last', action='store_true')
    # optimization (both families; None = family default, see FAMILY_DEFAULTS)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--lr-min', type=float, default=1e-6)
    ap.add_argument('--warmup-epochs', type=float, default=1.0)
    ap.add_argument('--beta2', type=float, default=0.99)
    ap.add_argument('--wd', type=float, default=None)
    ap.add_argument('--ema-decay', type=float, default=EMA_DECAY)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--deterministic', action='store_true')
    # data (both families)
    ap.add_argument('--workers', type=int, default=NUM_WORKERS)
    ap.add_argument('--max-per-volume', type=int, default=None)
    ap.add_argument('--aug-phase', action='store_true', default=True)
    ap.add_argument('--no-aug-phase', dest='aug_phase', action='store_false')
    ap.add_argument('--aug-flip', action='store_true', default=True)
    ap.add_argument('--no-aug-flip', dest='aug_flip', action='store_false')
    # internal validation / selection
    ap.add_argument('--val-frac', type=float, default=VAL_FRAC)
    ap.add_argument('--val-guard', type=int, default=VAL_GUARD)
    ap.add_argument('--val-bscans', type=int, default=VAL_BSCANS)
    ap.add_argument('--val-every', type=int, default=VAL_EVERY)
    ap.add_argument('--val-steps', type=int, default=VAL_STEPS)
    ap.add_argument('--supp-floor', type=float, default=SUPP_FLOOR)
    # logging
    ap.add_argument('--preview-every', type=int, default=5, help='0 = only at the end')
    ap.add_argument('--snapshot-every', type=int, default=None, help='EMA snapshot every N epochs (0 = no)')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--amp', action='store_true', default=True)
    ap.add_argument('--no-amp', dest='amp', action='store_false')
    return ap


def resolve_family_defaults(args):
    for key, value in FAMILY_DEFAULTS[args.family].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


if __name__ == '__main__':
    args = resolve_family_defaults(build_argparser().parse_args())
    ok = run_selftest(args.family)
    if args.stage == 'train':
        if not ok:
            raise SystemExit('Self-test failed: aborting.')
        (bscan_train if args.family == 'bscan' else roi_train)(args)
    elif args.stage == 'preview':
        model, arch, cfg, ep = load_trained(args.out, args.which)
        amp_dtype = torch.bfloat16 if args.amp else None
        out_dir = os.path.join(args.out, 'previews')
        tag = f'manual_{args.which}_ep{ep}'
        if arch_family(arch) == 'bscan':
            bscan_run_preview(model, cfg, load_preview_set(args.data_val, BSCAN_SIZE), out_dir, tag, amp_dtype)
        else:
            items = load_preview_set(args.data_val, arch['roi_size'])
            roi_run_preview(model, cfg, items, out_dir, tag, roi=arch['roi_size'], steps=args.steps,
                            amp_dtype=amp_dtype)
