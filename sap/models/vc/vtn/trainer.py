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
import os
import time
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from sap.data.manifest import (
    DEFAULT_MANIFEST_DIR,
    VTNManifestDataset,
    collate_vtn,
    make_dataloader,
)
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
    # resume: "auto" (load exp_dir/checkpoints/latest.pt if present), "none", or a path
    resume: str = "auto"
    # drop pairs whose source or target exceeds this many seconds (None = keep all)
    max_duration: Optional[float] = None
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

        # Distributed (torchrun) detection — additive: single-process path is unchanged.
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.dist = self.world_size > 1
        self.is_main = self.rank == 0
        if self.dist:
            import torch.distributed as dist
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = _device(cfg.device)

        # Only the main rank writes config/metrics/tb/checkpoints.
        self.exp = Path(cfg.exp_dir)
        self._metrics = None
        self._tb = None
        if self.is_main:
            (self.exp / "checkpoints").mkdir(parents=True, exist_ok=True)
            (self.exp / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
            self._metrics = open(self.exp / "metrics.jsonl", "a")
            try:  # optional TB (absent in the GPU container; rendered offline from jsonl)
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(str(self.exp / "tb"))
            except Exception:
                pass

        # --- data (DistributedSampler shards the train set across ranks) ---
        self._ids = self._subset_ids(cfg.train_split, cfg.max_train_ids)
        self.train_ds = VTNManifestDataset(cfg.manifest_dir, cfg.train_split, ids=self._ids,
                                           max_duration=cfg.max_duration)
        if self.dist:
            from torch.utils.data.distributed import DistributedSampler
            self.train_sampler = DistributedSampler(self.train_ds, shuffle=True, drop_last=True)
            self.train_loader = DataLoader(self.train_ds, batch_size=cfg.batch_size,
                                           sampler=self.train_sampler, num_workers=cfg.num_workers,
                                           collate_fn=collate_vtn)
        else:
            self.train_sampler = None
            self.train_loader = DataLoader(self.train_ds, batch_size=cfg.batch_size, shuffle=True,
                                           num_workers=cfg.num_workers, collate_fn=collate_vtn)
        self.val_loader = None
        if self.is_main:  # validation runs on the main rank only
            self.val_loader = make_dataloader(
                cfg.manifest_dir, cfg.val_split, batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, ids=self._subset_ids(cfg.val_split, cfg.max_val_ids),
                max_duration=cfg.max_duration)

        # --- model / loss / optim ---
        self._raw = VTN(cfg.model).to(self.device)
        if self.dist:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self._raw, device_ids=[self.local_rank])
        else:
            self.model = self._raw
        self.crit = VTNLoss(bce_pos_weight=cfg.bce_pos_weight, guided_weight=cfg.guided_weight)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr,
                                    weight_decay=cfg.weight_decay)
        self.step = 0
        self._maybe_resume()

    def _barrier(self):
        if self.dist:
            import torch.distributed as dist
            dist.barrier()

    def _maybe_resume(self):
        if self.cfg.resume == "none":
            return
        if self.cfg.resume == "auto":
            path = self.exp / "checkpoints" / "latest.pt"
        else:
            path = Path(self.cfg.resume)
        if not path.exists():
            return
        ckpt = torch.load(path, map_location=self.device)
        self._raw.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optim"])
        self.step = ckpt["step"]
        if self.is_main:
            print(f"[resume] loaded {path} at step {self.step}", flush=True)

    def _subset_ids(self, split, cap):
        if cap is None:
            return None
        from sap.data.manifest import load_pair_cutsets
        src, tgt = load_pair_cutsets(self.cfg.manifest_dir, split)
        return sorted(set(src.ids) & set(tgt.ids))[:cap]

    def _log(self, record: Dict):
        self._metrics.write(json.dumps(record) + "\n")
        self._metrics.flush()
        if self._tb is not None:
            tag = record.get("split", "train")
            for k, v in record.items():
                if isinstance(v, float) and k not in ("sps",):
                    self._tb.add_scalar(f"{tag}/{k}", v, record["step"])
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
        # Run on the UNWRAPPED module (self._raw), never the DDP wrapper — validating
        # through DDP on rank 0 alone desyncs collectives and deadlocks the other ranks.
        self._raw.eval()
        tot = 0.0
        n = 0
        for batch in islice(self.val_loader, self.cfg.val_batches):
            batch = _to_device(batch, self.device)
            out = self._raw(batch["src_mel"], batch["tgt_mel"], batch["src_lens"], batch["tgt_lens"])
            loss, _ = self.crit(out, batch["tgt_mel"], batch["tgt_lens"], batch["src_lens"])
            tot += float(loss.detach())
            n += 1
        self._raw.train()
        return {"step": self.step, "split": "val", "val_loss": tot / max(n, 1)}

    def save_checkpoint(self, tag: str):
        payload = {"step": self.step, "model": self._raw.state_dict(),
                   "optim": self.opt.state_dict(), "model_config": asdict(self.cfg.model)}
        path = self.exp / "checkpoints" / f"{tag}.pt"
        torch.save(payload, path)
        torch.save(payload, self.exp / "checkpoints" / "latest.pt")
        return path

    def train(self):
        self.model.train()
        t0 = time.time()
        done = False
        epoch = 0
        while not done:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            for batch in self.train_loader:
                self.step += 1
                loss, stats = self._run_batch(batch, train=True)
                if self.is_main and self.step % self.cfg.log_every == 0:
                    self._log({"step": self.step, "split": "train", "loss": loss,
                               "mel_mse": stats["mel_mse"], "bce": stats["bce"],
                               "guided": stats["guided"], "sps": self.step / (time.time() - t0)})
                if self.cfg.val_every and self.step % self.cfg.val_every == 0:
                    if self.is_main:
                        self._log(self.validate())
                    self._barrier()
                if self.cfg.ckpt_every and self.step % self.cfg.ckpt_every == 0:
                    if self.is_main:
                        self.save_checkpoint(f"step_{self.step}")
                    self._barrier()
                if self.step >= self.cfg.max_steps:
                    done = True
                    break
            epoch += 1
        if self.is_main:
            self.save_checkpoint(f"step_{self.step}")
            (self.exp / "DONE").write_text(f"step {self.step}\n")  # tells the chain to stop
            self._metrics.close()
            if self._tb is not None:
                self._tb.close()
        self._barrier()
        if self.dist:
            import torch.distributed as dist
            dist.destroy_process_group()
        return self.exp


def train(cfg: TrainConfig):
    return VTNTrainer(cfg).train()
