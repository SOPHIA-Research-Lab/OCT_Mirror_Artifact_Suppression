import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from octmirror.inference import load_pccgan, run_pccgan, scale_limits
from octmirror.models.pccgan import UNetGenerator
from octmirror.preprocessing import inverse_log_scale, log_scale

ROOT = Path(__file__).resolve().parents[1]


def test_load_and_run_pccgan_matches_evaluation_path(tmp_path):
    torch.manual_seed(0)
    G = UNetGenerator()
    torch.save(G.state_dict(), tmp_path / "G_best.pth")
    G2 = load_pccgan(str(tmp_path / "G_best.pth"), "cpu")
    assert not G2.training

    rng = np.random.default_rng(0)
    inp = rng.standard_normal((128, 128)) + 1j * rng.standard_normal((128, 128))
    tgt = rng.standard_normal((128, 128)) + 1j * rng.standard_normal((128, 128))
    smax, smin = scale_limits(tgt)
    pred = run_pccgan(G2, inp, smax, smin, "cpu")

    log_in, _, _, _, _ = log_scale(inp[np.newaxis])
    with torch.no_grad():
        fake = G.eval()(torch.tensor(log_in[0].transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0))
    rec = inverse_log_scale(fake[0].numpy().transpose(1, 2, 0)[np.newaxis], smax, smin)
    np.testing.assert_array_equal(pred, rec[0, ..., 0] + 1j * rec[0, ..., 1])


def _run_infer(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "infer.py"), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_infer_help():
    res = _run_infer("--help")
    assert res.returncode == 0 and "--scale-from" in res.stdout


@pytest.mark.parametrize("extra, message", [
    (["--no-simulate-mirror", "--scale-from", "target"], "--scale-from target requires"),
    (["--no-simulate-mirror", "--compute-metrics"], "--compute-metrics requires"),
])
def test_infer_rejects_invalid_combinations(extra, message):
    res = _run_infer("--model", "pccgan", "--weights", "w.pth", "--input", "x.npy",
                     "--out-dir", "o", *extra)
    assert res.returncode == 2 and message in res.stderr


def test_infer_pccgan_end_to_end(tmp_path):
    torch.manual_seed(0)
    torch.save(UNetGenerator().state_dict(), tmp_path / "G.pth")
    rng = np.random.default_rng(0)
    tom = rng.standard_normal((128, 128, 3)) + 1j * rng.standard_normal((128, 128, 3))
    np.save(tmp_path / "vol.npy", np.stack([tom.real, tom.imag], axis=-1))
    out = tmp_path / "out"
    res = _run_infer("--model", "pccgan", "--weights", str(tmp_path / "G.pth"),
                     "--input", str(tmp_path / "vol.npy"), "--out-dir", str(out),
                     "--bscan-indices", "0", "2", "--device", "cpu", "--compute-metrics")
    assert res.returncode == 0, res.stderr
    rec = np.load(out / "vol_pccgan_reconstruction.npy")
    assert rec.shape == (128, 128, 2) and np.iscomplexobj(rec)
    df = pd.read_csv(out / "metrics.csv")
    assert list(df["bscan_index"]) == [0, 2]
    assert set(df["despeckle_filter"]) == {"nlm_h010"}
    assert (out / "png" / "vol_y0002_phase.png").exists()
    assert (out / "png" / "vol_y0000_amplitude_db.png").exists()
