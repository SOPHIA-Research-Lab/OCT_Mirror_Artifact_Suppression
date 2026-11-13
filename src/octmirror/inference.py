"""Loading the trained PC-CGAN generator and reconstructing complex B-scans."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from octmirror.models.pccgan import UNetGenerator
from octmirror.preprocessing import inverse_log_scale, log_scale


def load_pccgan(weights: str, device) -> UNetGenerator:
    """Load a PC-CGAN generator ``state_dict`` (e.g. ``weight/G_best.pth``) in eval mode."""
    G = UNetGenerator(in_ch=2, out_ch=2)
    state_dict = torch.load(weights, map_location=device)
    G.load_state_dict(state_dict)
    G.to(device)
    G.eval()
    return G


def run_pccgan(G: nn.Module, input_cx: np.ndarray, smax: np.ndarray, smin: np.ndarray,
               device) -> np.ndarray:
    """Reconstruct one complex B-scan with the PC-CGAN generator.

    The input is log-scaled with its own limits, passed through ``G``, and the output is
    de-normalized with the given ``smax`` / ``smin`` (no clipping). In the paper's
    evaluation protocol these are the clean target's limits.

    Args:
        G: Generator in eval mode.
        input_cx: Mirror-corrupted complex B-scan (Z, X); Z and X must be multiples of 128.
        smax: Maximum log10 amplitude, shape (1, 1, 1).
        smin: Minimum log10 amplitude, shape (1, 1, 1).
        device: Torch device of ``G``.

    Returns:
        Reconstructed complex B-scan (Z, X).
    """
    logslices_in, _, _, _, _ = log_scale(input_cx[np.newaxis, ...])
    input_tensor = torch.tensor(
        logslices_in[0].transpose(2, 0, 1), dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        fake_img = G(input_tensor)  # (1, 2, Z, X)

    fake_np = fake_img[0].detach().cpu().numpy().transpose(1, 2, 0)[np.newaxis, ...]  # (1, Z, X, 2)
    pred_rec = inverse_log_scale(fake_np, smax, smin, max_log_amp=None)
    return pred_rec[0, :, :, 0] + 1j * pred_rec[0, :, :, 1]


def scale_limits(cx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-slice log10-amplitude limits ``(smax, smin)`` of one complex B-scan, shape (1, 1, 1)."""
    _, smax, smin, _, _ = log_scale(cx[np.newaxis, ...])
    return smax, smin
