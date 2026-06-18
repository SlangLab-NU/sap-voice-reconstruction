#!/usr/bin/env python3
"""Overfit the VTN on the fixture pairs and render audio for listening.

Proof-of-life demo (not a test): overfits the real fixture batch, then vocodes —
per pair — the original source, the target mel, and the model's predicted mel through
the same Griffin-Lim baseline, into ``OUT_DIR`` so they can be compared by ear.

Compare **pred_GL vs target_GL** (same vocoder) — that isolates "did the model fit the
target mel" from "how good is Griffin-Lim". Run:  .venv/bin/python scripts/vtn_overfit_demo.py
"""
import json
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

from sap.data.features import MelExtractor
from sap.data.vocoder import GriffinLimVocoder
from sap.models.vc.vtn.losses import VTNLoss
from sap.models.vc.vtn.model import VTN, VTNConfig

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
OUT_DIR = Path("/projects/aanchan/vtn_listen/overfit")
STEPS = 400
SR = 24000


def slug(s):
    return "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_")


def main():
    torch.manual_seed(0)
    pairs = [json.loads(l) for l in (FIXTURES / "vtn_pairs.jsonl").read_text().splitlines() if l.strip()]
    pairs = [p for p in pairs if (FIXTURES / p["source"]).exists()]
    assert pairs, "fixture audio missing"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ext = MelExtractor()
    voc = GriffinLimVocoder(n_iter=60)
    srcs = [ext.from_wav(FIXTURES / p["source"]) for p in pairs]
    tgts = [ext.from_wav(FIXTURES / p["target"]) for p in pairs]
    sl = torch.tensor([m.shape[0] for m in srcs])
    tl = torch.tensor([m.shape[0] for m in tgts])
    pad = lambda ms, T: torch.stack([F.pad(m, (0, 0, 0, T - m.shape[0])) for m in ms])
    src, tgt = pad(srcs, int(sl.max())), pad(tgts, int(tl.max()))

    model = VTN(VTNConfig(d_model=64, nhead=4, num_encoder_layers=3, num_decoder_layers=3,
                          dim_feedforward=256, dropout=0.0, prenet_dim=64, prenet_dropout=0.1,
                          postnet_channels=256, postnet_layers=5))
    crit = VTNLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(STEPS):
        opt.zero_grad()
        loss, stats = crit(model(src, tgt, sl, tl), tgt, tl, sl)
        loss.backward(); opt.step()
        if step % 50 == 0 or step == STEPS - 1:
            print(f"step {step:4d}  total={float(loss):.3f}  mel_mse={float(stats['mel_mse']):.3f}  "
                  f"guided={float(stats['guided']):.4f}")

    model.eval()
    with torch.no_grad():
        out = model(src, tgt, sl, tl)
    for i, p in enumerate(pairs):
        base = f"pair{i+1}_{slug(p['transcript'])}"
        # original source (already on disk) -> copy at its own rate via resample to SR for fairness
        src_wav, src_sr = sf.read(str(FIXTURES / p["source"]), dtype="float32")
        sf.write(OUT_DIR / f"{base}_source{src_sr//1000}k.wav", src_wav, src_sr)
        sf.write(OUT_DIR / f"{base}_target_GL.wav", voc(tgts[i]).cpu().numpy(), SR)
        sf.write(OUT_DIR / f"{base}_pred_GL.wav", voc(out["mel_after"][i][: tl[i]]).cpu().numpy(), SR)
        print(f"  wrote {base}: source / target_GL / pred_GL")
    print(f"\nDone -> {OUT_DIR}  (compare *_pred_GL vs *_target_GL)")


if __name__ == "__main__":
    main()
