import os

import numpy as np
import pytest

import aliases
from legacy_loader import load_legacy
from octmirror.preprocessing import (
    inverse_log_scale,
    load_tomogram,
    log_scale,
    select_bscan_indices,
)

LEGACY_SOURCES = [
    "torchPCCGAN_AblationM2_M8.py",
    "evaluate_ablation_unified.py",
]


def random_complex(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


# (a) round trip
@pytest.mark.parametrize("max_log_amp", [None, 30.0])
def test_log_scale_round_trip(max_log_amp):
    x = random_complex((3, 32, 24), seed=1)
    logslices, smax, smin, _, _ = log_scale(x)
    rec = inverse_log_scale(logslices, smax, smin, max_log_amp=max_log_amp)
    rec_cx = rec[..., 0] + 1j * rec[..., 1]
    # The minimum-amplitude pixel of each slice is normalized to amplitude 0, so its phase
    # is lost and it is reconstructed with phase 0.
    amp = np.abs(x)
    is_min = amp == amp.min(axis=(1, 2), keepdims=True)
    np.testing.assert_allclose(rec_cx[~is_min], x[~is_min], rtol=1e-5)
    np.testing.assert_allclose(np.abs(rec_cx[is_min]), amp[is_min], rtol=1e-5)


def test_inverse_log_scale_clip_only_affects_large_amplitudes():
    slices = np.full((1, 4, 4, 2), 1.0)  # normalized amplitude sqrt(2)
    smax = np.full((1, 1, 1), 40.0)
    smin = np.full((1, 1, 1), 0.0)
    clipped = inverse_log_scale(slices, smax, smin, max_log_amp=30.0)
    amp = np.abs(clipped[..., 0] + 1j * clipped[..., 1])
    np.testing.assert_allclose(amp, 1e30)


# (d) golden tests against legacy/
@pytest.mark.parametrize("source", LEGACY_SOURCES + ["diffComplexD.py", "evaluate_diffusion_D.py"])
def test_mirror_artifact_matches_legacy(source):
    legacy = load_legacy(source, ["mirrorArtifact"])
    for shape in [(64, 16, 3), (32, 32)]:
        x = random_complex(shape, seed=2)
        np.testing.assert_array_equal(aliases.mirrorArtifact(x), legacy.mirrorArtifact(x))


@pytest.mark.parametrize("source", LEGACY_SOURCES)
@pytest.mark.parametrize("as_channels", [False, True])
def test_log_scale_matches_legacy(source, as_channels):
    legacy = load_legacy(source, ["logScale"])
    x = random_complex((3, 32, 24), seed=3)
    if as_channels:
        x = np.stack([x.real, x.imag], axis=-1)
    new = aliases.logScale(x)
    old = legacy.logScale(x)
    for a, b in zip(new, old):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("source", ["torchPCCGAN_AblationM2_M8.py", "evaluate_ablation_unified.py"])
def test_inverse_log_scale_without_clip_matches_pccgan_legacy(source):
    legacy = load_legacy(source, ["inverseLogScale"])
    rng = np.random.default_rng(4)
    slices = rng.uniform(0, 1, (2, 16, 16, 2))
    smax = rng.uniform(30, 40, (2, 1, 1))  # large enough that a clip at 30 would matter
    smin = rng.uniform(0, 1, (2, 1, 1))
    np.testing.assert_array_equal(
        aliases.inverseLogScale(slices, smax, smin), legacy.inverseLogScale(slices, smax, smin)
    )


def test_select_bscan_indices_matches_legacy_evaluation():
    legacy = load_legacy("evaluate_ablation_unified.py", ["select_bscan_indices", "N_BSCANS_PER_VOLUME"])
    for n in [1, 10, 40, 41, 100, 512, 1000]:
        np.testing.assert_array_equal(select_bscan_indices(n, 40), legacy.select_bscan_indices(n))


@pytest.mark.parametrize("n_y", [10, 40, 41, 100, 512])
def test_select_bscan_indices_matches_linspace_rule(n_y):
    n = 40
    if n_y > n:
        expected = np.linspace(0, n_y - 1, n).astype(int)
    else:
        expected = np.arange(n_y)
    np.testing.assert_array_equal(select_bscan_indices(n_y, n), expected)


@pytest.mark.parametrize("as_channels", [False, True])
def test_load_tomogram_matches_legacy(tmp_path, as_channels):
    legacy = load_legacy("evaluate_ablation_unified.py", ["load_tomogram_complex"])
    tom = random_complex((16, 12, 5), seed=7)
    stored = np.stack([tom.real, tom.imag], axis=-1) if as_channels else tom
    path = os.path.join(tmp_path, "vol.npy")
    np.save(path, stored)
    new = load_tomogram(path, verbose=False)
    np.testing.assert_array_equal(new, legacy.load_tomogram_complex(path))
    np.testing.assert_array_equal(new, tom)
