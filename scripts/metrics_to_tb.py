#!/usr/bin/env python3
"""Render a VTN run's metrics.jsonl into TensorBoard event files.

The GPU container writes metrics.jsonl (canonical; no tensorboard dep there). Run this in
the CPU .venv (which has tensorboard) to get the TB UI without touching the container:

    .venv/bin/python scripts/metrics_to_tb.py /projects/aanchan/exp/vtn_run1
    .venv/bin/tensorboard --logdir /projects/aanchan/exp/vtn_run1/tb

Re-run anytime to refresh (rewrites the tb/ dir).
"""
import json
import shutil
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def main():
    exp = Path(sys.argv[1])
    tb = exp / "tb"
    if tb.exists():
        shutil.rmtree(tb)
    writer = SummaryWriter(str(tb))
    n = 0
    for line in (exp / "metrics.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tag, step = r.get("split", "train"), r.get("step")
        for k, v in r.items():
            if isinstance(v, (int, float)) and k != "step":
                writer.add_scalar(f"{tag}/{k}", v, step)
        n += 1
    writer.close()
    print(f"wrote {n} records -> {tb}\n  view: .venv/bin/tensorboard --logdir {tb}")


if __name__ == "__main__":
    main()
