"""VTN training loop (Issue 8) — config-driven, cluster-friendly.

Consumes the Lhotse manifest dataloader (:mod:`sap.data.manifest`), the VTN model, and the
ESPnet-style :class:`~sap.models.vc.vtn.losses.VTNLoss` (mel L1+MSE before/after + stop BCE +
guided attention). Writes a structured experiment dir: ``config.json``, ``metrics.jsonl``
(per-log-step train + per-val-step metrics), and ``checkpoints/`` (``step_*.pt`` + ``latest.pt``,
each with model/optimizer state and the model config). No hardcoded home/user paths — all roots
come from :class:`TrainConfig`; on the NU cluster point ``exp_dir`` at ``/scratch/aa.mohan/...``.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Dict, Optional

import torch

from sap.data.manifest import DEFAULT_MANIFEST_DIR, make_dataloader
from sap.models.vc.vtn.losses import VTNLoss
from sap.models.vc.vtn.model import VTN, VTNConfig


@dataclass
class TrainConfig:
    manifest_dir: str = str(DEFAULT_MANIFEST_DIR)
    train_split: str = "train"
    val_split: str = "val"
    exp_dir: str = "experiments/vtn"
    model: VTNConfig = field(default_factory=VTNConfig)
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_steps: int = 100_000
    log_every: int = 50
    val_every: int = 1000
    ckpt_every: int = 1000
    val_batches: int = 20
    num_workers: int = 4
    seed: int = 0
    device: str = "auto"
    bce_pos_weight: float = 5.0
    guided_weight: float = 1.0
    # caps for quick/smoke runs (None = full split)
    max_train_ids: Optional[int] = None
    max_val_ids: Optional[int] = None


def _device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _to_device(batch: Dict, device) -> Dict:
    for k in ("src_mel", "tgt_mel", "src_lens", "tgt_lens"):
        batch[k] = batch[k].to(device)
    return batch


class VTNTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = _device(cfg.device)
        self.exp = Path(cfg.exp_dir)
        (self.exp / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.exp / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        self._metrics = open(self.exp / "metrics.jsonl", "a")

        self._ids = self._subset_ids(cfg.train_split, cfg.max_train_ids)
        self.train_loader = make_dataloader(
            cfg.manifest_dir, cfg.train_split, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, ids=self._ids)
        self.val_loader = make_dataloader(
            cfg.manifest_dir, cfg.val_split, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, ids=self._subset_ids(cfg.val_split, cfg.max_val_ids))

        self.model = VTN(cfg.model).to(self.device)
        self.crit = VTNLoss(bce_pos_weight=cfg.bce_pos_weight, guided_weight=cfg.guided_weight)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr,
                                    weight_decay=cfg.weight_decay)
        self.step = 0

    def _subset_ids(self, split, cap):
        if cap is None:
            return None
        from sap.data.manifest import load_pair_cutsets
        src, tgt = load_pair_cutsets(self.cfg.manifest_dir, split)
        return sorted(set(src.ids) & set(tgt.ids))[:cap]

    def _log(self, record: Dict):
        self._metrics.write(json.dumps(record) + "\n")
        self._metrics.flush()
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in record.items()), flush=True)

    def _run_batch(self, batch, train: bool):
        batch = _to_device(batch, self.device)
        out = self.model(batch["src_mel"], batch["tgt_mel"], batch["src_lens"], batch["tgt_lens"])
        loss, stats = self.crit(out, batch["tgt_mel"], batch["tgt_lens"], batch["src_lens"])
        if train:
            self.opt.zero_grad()
            loss.backward()
            if self.cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()
        return float(loss.detach()), {k: float(v.detach()) for k, v in stats.items()}

    @torch.no_grad()
    def validate(self) -> Dict:
        self.model.eval()
        tot = 0.0
        n = 0
        for batch in islice(self.val_loader, self.cfg.val_batches):
            loss, _ = self._run_batch(batch, train=False)
            tot += loss
            n += 1
        self.model.train()
        return {"step": self.step, "split": "val", "val_loss": tot / max(n, 1)}

    def save_checkpoint(self, tag: str):
        path = self.exp / "checkpoints" / f"{tag}.pt"
        torch.save({"step": self.step, "model": self.model.state_dict(),
                    "optim": self.opt.state_dict(), "model_config": asdict(self.cfg.model)}, path)
        torch.save({"step": self.step, "model": self.model.state_dict(),
                    "optim": self.opt.state_dict(), "model_config": asdict(self.cfg.model)},
                   self.exp / "checkpoints" / "latest.pt")
        return path

    def train(self):
        self.model.train()
        t0 = time.time()
        done = False
        while not done:
            for batch in self.train_loader:
                self.step += 1
                loss, stats = self._run_batch(batch, train=True)
                if self.step % self.cfg.log_every == 0:
                    self._log({"step": self.step, "split": "train", "loss": loss,
                               "mel_mse": stats["mel_mse"], "bce": stats["bce"],
                               "guided": stats["guided"], "sps": self.step / (time.time() - t0)})
                if self.cfg.val_every and self.step % self.cfg.val_every == 0:
                    self._log(self.validate())
                if self.cfg.ckpt_every and self.step % self.cfg.ckpt_every == 0:
                    self.save_checkpoint(f"step_{self.step}")
                if self.step >= self.cfg.max_steps:
                    done = True
                    break
        self.save_checkpoint(f"step_{self.step}")
        self._metrics.close()
        return self.exp


def train(cfg: TrainConfig):
    return VTNTrainer(cfg).train()
