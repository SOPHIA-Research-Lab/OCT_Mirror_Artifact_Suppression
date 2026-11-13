import pytest
import torch

from legacy_loader import load_legacy
from octmirror.models.pccgan import PatchGANDiscriminator, UNetGenerator

GAN = "torchPCCGAN_AblationM2_M8.py"


# (b) shapes
def test_unet_generator_shape():
    G = UNetGenerator(in_ch=2, out_ch=2).eval()
    with torch.no_grad():
        y = G(torch.rand(1, 2, 512, 512))
    assert y.shape == (1, 2, 512, 512)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_patchgan_shape():
    D = PatchGANDiscriminator(in_ch=2).eval()
    with torch.no_grad():
        y = D(torch.rand(1, 2, 512, 512), torch.rand(1, 2, 512, 512))
    assert y.shape == (1, 1, 30, 30)


# golden tests: identical architectures and outputs
@pytest.mark.parametrize("source", [GAN, "evaluate_ablation_unified.py"])
def test_pccgan_models_match_legacy(source):
    legacy = load_legacy(source, ["GeneratorTf", "DiscriminatorTf"])
    torch.manual_seed(0)
    G_old = legacy.GeneratorTf(in_ch=2, out_ch=2)
    torch.manual_seed(0)
    G_new = UNetGenerator(in_ch=2, out_ch=2)
    assert G_old.state_dict().keys() == G_new.state_dict().keys()
    for k, v in G_old.state_dict().items():
        assert torch.equal(v, G_new.state_dict()[k])
    G_new.load_state_dict(G_old.state_dict())
    x = torch.rand(2, 2, 256, 256)
    with torch.no_grad():
        assert torch.equal(G_old.eval()(x), G_new.eval()(x))

    D_old = legacy.DiscriminatorTf(in_ch=2)
    D_new = PatchGANDiscriminator(in_ch=2)
    D_new.load_state_dict(D_old.state_dict())
    a, b = torch.rand(2, 2, 128, 128), torch.rand(2, 2, 128, 128)
    with torch.no_grad():
        assert torch.equal(D_old.eval()(a, b), D_new.eval()(a, b))
