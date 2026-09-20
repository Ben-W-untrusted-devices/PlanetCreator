import numpy as np
import torch

from planetcreator.bake import CTX_LAYERS, DIST_NAMES
from planetcreator.features import (
    COND_CHANNELS,
    HEIGHT_SCALE,
    build_cond,
    build_ctx,
    noise_field,
    px_km_for_face,
)
from planetcreator.terrain import TerrainConfig, TerrainNet, infer_face

P, PAD, F = 64, 224, 8  # small config: pad = 3.5 P


def _batch(b=2):
    g = torch.Generator().manual_seed(0)
    cs = (P + 2 * PAD) // F
    out = {
        "height": torch.randn(b, 1, P, P, generator=g) * 2000,
        "rgb": torch.rand(b, 3, P, P, generator=g),
        "lat": torch.rand(b, 1, P, P, generator=g) * 180 - 90,
        "tavg": torch.rand(b, 1, P, P, generator=g) * 60 - 30,
        "trange": torch.rand(b, 1, P, P, generator=g) * 40,
        "prec": torch.rand(b, 1, P, P, generator=g) * 5000,
        "water": (torch.rand(b, 1, P, P, generator=g) < 0.3).float(),
        "flowacc": torch.rand(b, 1, P, P, generator=g) * 6,
    }
    for d in DIST_NAMES:
        out[d] = torch.rand(b, 1, P, P, generator=g) * 8
    out["noise"] = torch.rand(b, 1, P, P, generator=g) * 2 - 1
    for name in CTX_LAYERS:
        out[f"ctx_{name}"] = torch.rand(b, 1, cs, cs, generator=g) * (4000 if name == "height" else 1)
    return out


def _cfg(mode, adv=0.0):
    return TerrainConfig(mode=mode, base=8, depth=2, ctx_base=8, ctx_pad=PAD, ctx_factor=F, adv_weight=adv)


def test_cond_shapes_both_modes():
    b = _batch()
    for mode in ("colour", "joint"):
        c = build_cond(b, mode, ctx_pad=PAD, ctx_factor=F)
        assert c.shape == (2, len(COND_CHANNELS), P, P) and c.abs().max() < 10
    assert build_ctx(b).shape == (2, len(CTX_LAYERS), (P + 2 * PAD) // F, (P + 2 * PAD) // F)


def test_slope_is_resolution_invariant():
    b = _batch(b=1)
    b["height"] = torch.arange(float(P)).repeat(P, 1)[None, None] * 1.0
    lo = {k: v.clone() for k, v in b.items()}
    lo["height"] = b["height"] * 4.0
    hi = build_cond(b, px_km=px_km_for_face(4096), ctx_pad=PAD, ctx_factor=F)[0, 1, 5:-5, 5:-5]
    lo = build_cond(lo, px_km=px_km_for_face(1024), ctx_pad=PAD, ctx_factor=F)[0, 1, 5:-5, 5:-5]
    assert torch.allclose(hi, lo, atol=1e-5)


def test_forward_and_train_step_each_mode():
    for mode in ("colour", "joint"):
        torch.manual_seed(0)
        m = TerrainNet(_cfg(mode, adv=0.1))
        d = m.make_discriminator()
        b = _batch()
        out, cond, _ctx = m.predict(b)
        assert out["rgb"].shape == (2, 3, P, P)
        if mode == "joint":
            assert out["height"].shape == (2, 1, P, P)
            # mean-preserving: every 8x8 block of the output averages to the coarse texel
            coarse = b["ctx_height"][..., PAD // F : PAD // F + P // F, PAD // F : PAD // F + P // F] / HEIGHT_SCALE
            block = torch.nn.functional.avg_pool2d(out["height"], F)
            assert torch.allclose(block, coarse, atol=1e-4)
        target = m.target(b)
        opt = torch.optim.Adam(m.parameters(), 1e-2)
        losses = []
        for _ in range(8):
            out, cond, _ctx = m.predict(b)
            logits = d(cond, m.stack_out(out))
            loss, parts = m.loss(out, target, logits)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert torch.isfinite(torch.tensor(losses)).all() and "adv" in parts
        dl = m.d_loss(d(cond, m.stack_out(target if mode == "colour" else {"rgb": target["rgb"], "height": target["height"]})), logits.detach())
        assert torch.isfinite(dl)


def test_infer_face_matches_direct(tmp_path):
    torch.manual_seed(0)
    m = TerrainNet(_cfg("joint")).eval()
    n = 2 * P
    cs_face = (n + 2 * PAD) // F
    rng = np.random.default_rng(0)
    fine = {k: rng.random((n, n)).astype(np.float32) * 10 for k in ("lat", "tavg", "trange", "prec")}
    ctx = {k: rng.random((cs_face, cs_face)).astype(np.float32) * (4000 if k == "height" else 1) for k in CTX_LAYERS}
    res = infer_face(m, fine, ctx, tile=P, overlap=16, batch=4)
    assert res["rgb"].shape == (n, n, 3) and res["height"].shape == (n, n)
    # Direct evaluation of the top-left tile agrees with the blended result away from tile edges.
    b = {k: torch.from_numpy(v[:P, :P])[None, None] for k, v in fine.items()}
    cs = (P + 2 * PAD) // F
    b.update({f"ctx_{k}": torch.from_numpy(v[:cs, :cs])[None, None] for k, v in ctx.items()})
    b["noise"] = torch.from_numpy(noise_field(0, 0, 0, P))[None, None]
    with torch.no_grad():
        direct = m.predict(b)[0]["height"][0, 0].numpy() * HEIGHT_SCALE
    assert np.abs(res["height"][:P // 2, :P // 2] - direct[:P // 2, :P // 2]).mean() < 5.0
    m.save(tmp_path / "m.pt", step=3)
    m2, ck = TerrainNet.load(tmp_path / "m.pt")
    assert ck["step"] == 3 and m2.cfg == m.cfg


def test_noise_field_is_deterministic_and_position_keyed():
    a = noise_field(2, 100, 200, 16)
    b = noise_field(2, 100, 200, 16)
    assert np.array_equal(a, b) and a.min() >= -1 and a.max() <= 1 and abs(a.mean()) < 0.2
    # a window shifted by (0, 4) shares its overlapping pixels exactly
    c = noise_field(2, 100, 204, 16)
    assert np.array_equal(a[:, 4:], c[:, :12])
    assert not np.array_equal(noise_field(3, 100, 200, 16), a)


def test_mean_preserving_upsample_is_exact_and_smooth():
    from planetcreator.features import mean_preserving_upsample

    torch.manual_seed(0)
    c = torch.randn(1, 1, 12, 12) * 1000
    up = mean_preserving_upsample(c, 8)
    assert torch.allclose(torch.nn.functional.avg_pool2d(up, 8), c, atol=1e-3)
    # smooth: the largest pixel-to-pixel step is far below the coarse texel differences
    step = (up[..., 1:] - up[..., :-1]).abs().max()
    assert step < 0.35 * (c[..., 1:] - c[..., :-1]).abs().max()
