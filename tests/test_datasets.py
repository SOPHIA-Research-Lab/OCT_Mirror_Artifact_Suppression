import os
import random

import numpy as np
import torch

from legacy_loader import load_legacy
from octmirror.datasets import (
    PairedComplexDatasetWithScale,
    build_pccgan_datasets,
)


def _write_volumes(root, folder, names, shape, seed=0):
    os.makedirs(os.path.join(root, folder), exist_ok=True)
    rng = np.random.default_rng(seed)
    vols = {}
    for k, name in enumerate(names):
        tom = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        stored = np.stack([tom.real, tom.imag], axis=-1) if k % 2 else tom
        np.save(os.path.join(root, folder, name), stored)
        vols[name] = tom
    return vols


def test_seeded_split_equals_global_seed_shuffle():
    a = list(range(100))
    random.Random(0).shuffle(a)
    b = list(range(100))
    random.seed(0)
    random.shuffle(b)
    assert a == b


def test_pccgan_dataset_matches_legacy_dataset():
    legacy = load_legacy("torchPCCGAN_AblationM2_M8.py", ["ArrayDataset"])
    rng = np.random.default_rng(0)
    data = [(rng.uniform(size=(8, 6, 2)), rng.uniform(size=(8, 6, 2))) for _ in range(3)]
    max_min = [(rng.uniform(size=(1, 1)), rng.uniform(size=(1, 1))) for _ in range(3)]
    new = PairedComplexDatasetWithScale(data, max_min)
    old = legacy.ArrayDataset(data, max_min)
    assert len(new) == len(old)
    for i in range(len(new)):
        for a, b in zip(new[i], old[i]):
            if isinstance(a, torch.Tensor):
                assert a.dtype == torch.float32 and a.shape == (2, 8, 6)
                assert torch.equal(a, b)
            else:
                np.testing.assert_array_equal(a, b)


def test_build_pccgan_datasets_reproduces_legacy_pipeline(tmp_path):
    legacy = load_legacy("torchPCCGAN_AblationM2_M8.py", ["mirrorArtifact", "logScale"])
    root = str(tmp_path)
    vols = _write_volumes(root, "phase", ["A.npy", "B.npy", "C.npy"], (32, 32, 6))
    files = ["A.npy", "B.npy"]
    train_ds, val_ds = build_pccgan_datasets(root, ["phase"], files, split_seed=0)

    # legacy loop (verbatim logic), followed by the seeded shuffle
    pairs, max_min = [], []
    for file in os.listdir(os.path.join(root, "phase")):
        if file not in files:
            continue
        tom = vols[file]
        tomcc = legacy.mirrorArtifact(tom)
        tom = np.transpose(tom, (2, 0, 1))
        tomcc = np.transpose(tomcc, (2, 0, 1))
        t, smax, smin, _, _ = legacy.logScale(tom)
        c, _, _, _, _ = legacy.logScale(tomcc)
        pairs += [(c[i], t[i]) for i in range(t.shape[0])]
        max_min += [(smax[i], smin[i]) for i in range(t.shape[0])]
    indices = list(range(len(pairs)))
    random.seed(0)
    random.shuffle(indices)
    n_train = len(pairs) - int(0.1 * len(pairs))
    assert len(train_ds) == n_train and len(val_ds) == len(pairs) - n_train
    for ds, idx in [(train_ds, indices[:n_train]), (val_ds, indices[n_train:])]:
        for k, i in enumerate(idx):
            np.testing.assert_array_equal(ds.data[k][0], pairs[i][0])
            np.testing.assert_array_equal(ds.data[k][1], pairs[i][1])
            np.testing.assert_array_equal(ds.max_min[k][0], max_min[i][0])
            np.testing.assert_array_equal(ds.max_min[k][1], max_min[i][1])
