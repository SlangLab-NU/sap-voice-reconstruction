#!/usr/bin/env python3
"""Synthesize from a trained VTN checkpoint and render audio (Issue 9, Griffin-Lim path).

For a few short SAP source utterances, writes per utterance (all through the SAME Griffin-Lim
vocoder, so differences are the model's, not the vocoder's):
  - <id>_source.wav        original atypical input (the model's input)
  - <id>_target_GL.wav     the synthetic target mel the model was trained toward
  - <id>_recon_tf_GL.wav   teacher-forced reconstruction (best case — feeds ground-truth target)
  - <id>_synth_free_GL.wav free-running synthesis (real inference: autoregressive, no target)

Runs fine on CPU. Example:
  .venv/bin/python scripts/infer_vtn.py \
      --checkpoint /projects/aanchan/exp/vtn_mg_run1/checkpoints/latest.pt \
      --split test --num 4 --max-duration 4.0 \
      --out-dir /projects/aanchan/vtn_listen/infer_run1
"""
import argparse
from pathlib import Path

import soundfile as sf
import torch

from sap.data.manifest import VTNManifestDataset
from sap.data.vocoder import build_vocoder
from sap.models.vc.vtn.model import VTN, VTNConfig


def load_vtn(checkpoint_path, device="cpu"):
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    model = VTN(VTNConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, int(ckpt.get("step", -1))


def slug(s):
    return "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_")[:40]


def main():
    p = argparse.ArgumentParser(description="VTN synthesis + Griffin-Lim render")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest-dir", default="/projects/aanchan/data/manifests")
    p.add_argument("--split", default="test")
    p.add_argument("--num", type=int, default=4)
    p.add_argument("--max-duration", type=float, default=4.0,
                   help="pick source utts <= this many seconds (keeps free-running fast)")
    p.add_argument("--max-len", type=int, default=600, help="max synthesis frames")
    p.add_argument("--out-dir", default="/projects/aanchan/vtn_listen/infer_run1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--vocoder", default="griffinlim", choices=["griffinlim", "hifigan"])
    p.add_argument("--vocoder-checkpoint", default=None, help="required for --vocoder hifigan")
    p.add_argument("--etiology", default=None,
                   help="comma-separated source etiologies to sample, e.g. 'Stroke,Cerebral Palsy'")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, step = load_vtn(args.checkpoint, args.device)
    print(f"loaded VTN @ step {step} from {args.checkpoint}")
    voc = build_vocoder(args.vocoder, checkpoint=args.vocoder_checkpoint, device=args.device)
    etis = [e.strip() for e in args.etiology.split(",")] if args.etiology else None
    ds = VTNManifestDataset(args.manifest_dir, args.split, max_duration=args.max_duration,
                            etiologies=etis)
    print(f"{len(ds)} utts available (split={args.split}, etiology={etis}, <= {args.max_duration}s)")
    sr = voc.sample_rate

    for i in range(min(args.num, len(ds))):
        ex = ds[i]
        uid, text = ex["id"], ex["text"]
        src = ex["source_mel"].unsqueeze(0).to(args.device)
        tgt = ex["target_mel"].unsqueeze(0).to(args.device)
        base = f"{i + 1:02d}_{slug(ex['etiology'] or 'na')}_{slug(text)}"
        print(f"[{i + 1}] {ex['etiology']} | {text!r} | spk {ex['speaker'][:8]} "
              f"(src {src.shape[1]} fr, tgt {tgt.shape[1]} fr)")

        with torch.no_grad():
            tf = model(src, tgt)["mel_after"][0]              # teacher-forced
            free = model.inference(src, max_len=args.max_len)  # free-running
        print(f"     free-running produced {free['n_frames']} frames")

        sf.write(out / f"{base}_source.wav", ds._src[uid].load_audio()[0],
                 int(ds._src[uid].sampling_rate))
        sf.write(out / f"{base}_target_GL.wav", voc(ex["target_mel"]).cpu().numpy(), sr)
        sf.write(out / f"{base}_recon_tf_GL.wav", voc(tf).cpu().numpy(), sr)
        sf.write(out / f"{base}_synth_free_GL.wav", voc(free["mel_after"][0]).cpu().numpy(), sr)

    print(f"\nDone -> {out}  (compare *_synth_free_GL vs *_target_GL; *_recon_tf_GL = best case)")


if __name__ == "__main__":
    main()
