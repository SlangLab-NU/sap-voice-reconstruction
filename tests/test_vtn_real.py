"""Real-data, believable VTN tests (skip if fixture audio absent).

- forward on real source/target mels (shape/finiteness on real data, not random tensors)
- overfit-single-batch with the ESPnet-style loss: assert reconstruction *collapses* and
  the encoder-decoder attention *diagonalizes* — i.e. the model genuinely learns, rather
  than merely running a step without error.
"""
import torch
import torch.nn.functional as F

from sap.data.features import MelExtractor
from sap.models.vc.vtn.losses import VTNLoss
from sap.models.vc.vtn.model import VTN, VTNConfig


def _batch(pairs, ext, k):
    src = [ext.from_wav(p["source"]) for p in pairs[:k]]
    tgt = [ext.from_wav(p["target"]) for p in pairs[:k]]
    sl = torch.tensor([m.shape[0] for m in src])
    tl = torch.tensor([m.shape[0] for m in tgt])

    def pad(ms, T):
        return torch.stack([F.pad(m, (0, 0, 0, T - m.shape[0])) for m in ms])

    return pad(src, int(sl.max())), pad(tgt, int(tl.max())), sl, tl


def test_real_pair_forward(vtn_pairs):
    ext = MelExtractor()
    model = VTN(VTNConfig(d_model=32, nhead=4, num_encoder_layers=2,
                          num_decoder_layers=2, dim_feedforward=64,
                          prenet_dim=32, postnet_channels=32)).eval()
    for p in vtn_pairs:
        s = ext.from_wav(p["source"]).unsqueeze(0)
        t = ext.from_wav(p["target"]).unsqueeze(0)
        out = model(s, t)
        assert out["mel_after"].shape == t.shape
        assert out["attn"].shape == (1, t.shape[1], s.shape[1])
        assert torch.isfinite(out["mel_after"]).all()


def test_overfit_single_batch(vtn_pairs):
    """The believable test: overfit a real batch and require it to actually fit."""
    torch.manual_seed(0)
    ext = MelExtractor()
    src, tgt, sl, tl = _batch(vtn_pairs, ext, k=2)

    model = VTN(VTNConfig(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                          dim_feedforward=128, dropout=0.0, prenet_dim=64,
                          prenet_dropout=0.1, postnet_channels=128, postnet_layers=5))
    crit = VTNLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    @torch.no_grad()
    def measure():
        model.eval()
        out = model(src, tgt, sl, tl)
        _, stats = crit(out, tgt, tl, sl)
        model.train()
        return {k: float(v) for k, v in stats.items()}

    init = measure()
    for _ in range(180):
        model.train()
        opt.zero_grad()
        out = model(src, tgt, sl, tl)
        loss, _ = crit(out, tgt, tl, sl)
        loss.backward()
        opt.step()
    final = measure()

    # (1) reconstruction collapses
    assert final["mel_mse"] < 0.3 * init["mel_mse"], (init, final)
    assert final["mel_l1"] < 0.5 * init["mel_l1"], (init, final)
    # (3) attention diagonalizes (guided-attn loss drops)
    assert final["guided"] < init["guided"], (init, final)
