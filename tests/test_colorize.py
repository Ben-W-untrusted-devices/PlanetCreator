import numpy as np
import torch

from planetcreator.colorize import ColorizeConfig, Colorizer, tile_infer
from planetcreator.features import COND_CHANNELS, build_cond, slope_magnitude
from planetcreator.models import UNet


def _batch(b=2, n=64):
    g = torch.Generator().manual_seed(0)
    return {
        "height": torch.randn(b, 1, n, n, generator=g) * 2000,
        "lat": torch.rand(b, 1, n, n, generator=g) * 180 - 90,
        "tavg": torch.rand(b, 1, n, n, generator=g) * 60 - 30,
        "trange": torch.rand(b, 1, n, n, generator=g) * 40,
        "prec": torch.rand(b, 1, n, n, generator=g) * 5000,
    }


def test_slope_of_plane():
    h = torch.arange(16.0).repeat(16, 1)[None, None] * 3  # 3 m/px in x
    s = slope_magnitude(h)
    assert torch.allclose(s[..., 1:-1, 1:-1], torch.tensor(3.0))


def test_build_cond_shape_and_range():
    c = build_cond(_batch())
    assert c.shape == (2, len(COND_CHANNELS), 64, 64)
    assert c.abs().max() < 5  # everything roughly unit-scaled
    assert set(c[:, 2].unique().tolist()) <= {0.0, 1.0}  # ocean mask is binary


def test_unet_shape():
    m = UNet(7, 3, base=8, depth=3)
    assert m(torch.zeros(1, 7, 64, 64)).shape == (1, 3, 64, 64)


def test_colorizer_train_step_reduces_loss():
    torch.manual_seed(0)
    m = Colorizer(ColorizeConfig(base=8, depth=2))
    b = _batch()
    target = torch.rand(2, 3, 64, 64)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    losses = []
    for _ in range(15):
        loss, _ = m.loss(m(build_cond(b)), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_tile_infer_matches_direct(tmp_path):
    torch.manual_seed(0)
    m = Colorizer(ColorizeConfig(base=8, depth=2)).eval()
    b = _batch(b=1, n=96)
    layers = {k: v[0, 0].numpy() for k, v in b.items()}
    direct = m(build_cond(b))[0].permute(1, 2, 0).detach().numpy()
    tiled = tile_infer(m, layers, tile=48, overlap=16, batch=4).astype(np.float32) / 255
    # Blending over the receptive field differs slightly; require close agreement.
    assert np.abs(tiled - direct).mean() < 0.02
    m.save(tmp_path / "m.pt", step=3)
    m2, ck = Colorizer.load(tmp_path / "m.pt")
    assert ck["step"] == 3 and m2.cfg == m.cfg
