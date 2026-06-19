"""Smoke test for the VTN trainer (real manifest data, tiny/capped; skip if absent)."""
import json

from sap.models.vc.vtn.model import VTNConfig
from sap.models.vc.vtn.trainer import TrainConfig, VTNTrainer, train


def test_trainer_runs_logs_checkpoints_and_learns(manifest_dir, tmp_path):
    cfg = TrainConfig(
        manifest_dir=str(manifest_dir),
        train_split="val", val_split="val",   # use val (smaller) for a fast smoke
        exp_dir=str(tmp_path / "vtn_smoke"),
        model=VTNConfig(d_model=32, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                        dim_feedforward=64, prenet_dim=32, postnet_channels=32),
        batch_size=4, lr=1e-3, max_steps=24, log_every=4, val_every=12, ckpt_every=12,
        val_batches=2, num_workers=0, max_train_ids=8, max_val_ids=4, device="cpu",
    )
    exp = train(cfg)

    # structured experiment dir
    assert (exp / "config.json").exists()
    assert (exp / "checkpoints" / "latest.pt").exists()
    metrics = [json.loads(l) for l in (exp / "metrics.jsonl").read_text().splitlines() if l.strip()]
    assert metrics, "no metrics logged"

    # a validation record was written
    assert any(m.get("split") == "val" for m in metrics)

    # training loss went down over the (short) run
    train_losses = [m["loss"] for m in metrics if m.get("split") == "train"]
    assert len(train_losses) >= 2
    assert train_losses[-1] < train_losses[0], train_losses


def test_trainer_resumes_from_checkpoint(manifest_dir, tmp_path):
    base = dict(
        manifest_dir=str(manifest_dir), train_split="val", val_split="val",
        exp_dir=str(tmp_path / "vtn_resume"),
        model=VTNConfig(d_model=32, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                        dim_feedforward=64, prenet_dim=32, postnet_channels=32),
        batch_size=4, max_steps=8, log_every=4, val_every=0, ckpt_every=0,
        num_workers=0, max_train_ids=8, max_val_ids=4, device="cpu",
    )
    train(TrainConfig(**base))                      # runs to step 8, writes latest.pt
    resumed = VTNTrainer(TrainConfig(**base))        # resume="auto" picks up latest.pt
    assert resumed.step == 8
