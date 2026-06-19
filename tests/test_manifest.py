"""Tests for the Lhotse-manifest VTN data path (skip if manifests/lhotse absent)."""
import torch

from sap.data.manifest import VTNManifestDataset, collate_vtn, make_dataloader
from sap.models.vc.vtn.model import VTN, VTNConfig


def test_source_target_pairing_is_1to1(val_cutsets):
    src, tgt = val_cutsets
    sids, tids = set(src.ids), set(tgt.ids)
    assert sids == tids and len(sids) > 0  # same id set -> exact 1:1 join


def _small_ids(val_cutsets, n=4):
    src, tgt = val_cutsets
    return sorted(set(src.ids) & set(tgt.ids))[:n]


def test_dataset_item(manifest_dir, val_cutsets):
    ds = VTNManifestDataset(manifest_dir, "val", ids=_small_ids(val_cutsets))
    ex = ds[0]
    assert ex["source_mel"].shape[1] == 80 and ex["target_mel"].shape[1] == 80
    assert torch.isfinite(ex["source_mel"]).all() and torch.isfinite(ex["target_mel"]).all()
    assert isinstance(ex["text"], str) and ex["text"]
    assert isinstance(ex["speaker"], str) and ex["speaker"]


def test_collate_pads_to_batch_max(manifest_dir, val_cutsets):
    ds = VTNManifestDataset(manifest_dir, "val", ids=_small_ids(val_cutsets, 3))
    batch = collate_vtn([ds[i] for i in range(len(ds))])
    B = len(ds)
    assert batch["src_mel"].shape[0] == B and batch["src_mel"].shape[2] == 80
    assert batch["src_mel"].shape[1] == int(batch["src_lens"].max())
    assert batch["tgt_mel"].shape[1] == int(batch["tgt_lens"].max())
    assert len(batch["texts"]) == B


def test_dataloader_feeds_model(manifest_dir, val_cutsets):
    """Real manifest data flows end-to-end through the VTN model."""
    dl = make_dataloader(manifest_dir, "val", batch_size=3, shuffle=False,
                         ids=_small_ids(val_cutsets, 3))
    batch = next(iter(dl))
    model = VTN(VTNConfig(d_model=32, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                          dim_feedforward=64, prenet_dim=32, postnet_channels=32)).eval()
    out = model(batch["src_mel"], batch["tgt_mel"], batch["src_lens"], batch["tgt_lens"])
    Tt, Ts = batch["tgt_mel"].shape[1], batch["src_mel"].shape[1]
    assert out["mel_after"].shape == (3, Tt, 80)
    assert out["attn"].shape == (3, Tt, Ts)
    assert torch.isfinite(out["mel_after"]).all()
