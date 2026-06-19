"""Synthetic shape/contract guards for the VTN model (fast, no data, run everywhere)."""
import torch

from sap.models.vc.vtn.model import VTN, VTNConfig

NMELS = 80


def _tiny():
    return VTN(VTNConfig(n_mels=NMELS, d_model=32, nhead=4, num_encoder_layers=2,
                         num_decoder_layers=2, dim_feedforward=64, prenet_dim=32,
                         postnet_channels=32))


def test_forward_contract_unequal_lengths():
    """T_src != T_tgt on purpose: source/target are not frame-aligned."""
    torch.manual_seed(0)
    m = _tiny()
    B, Ts, Tt = 2, 9, 13
    out = m(torch.randn(B, Ts, NMELS), torch.randn(B, Tt, NMELS))
    assert out["mel_before"].shape == (B, Tt, NMELS)
    assert out["mel_after"].shape == (B, Tt, NMELS)
    assert out["stop_logits"].shape == (B, Tt)
    assert out["attn"].shape == (B, Tt, Ts)
    assert all(torch.isfinite(v).all() for v in out.values())


def test_forward_with_padding_lengths():
    torch.manual_seed(0)
    m = _tiny()
    B, Ts, Tt = 2, 10, 12
    out = m(torch.randn(B, Ts, NMELS), torch.randn(B, Tt, NMELS),
            src_lens=torch.tensor([10, 6]), tgt_lens=torch.tensor([12, 7]))
    assert out["mel_after"].shape == (B, Tt, NMELS)
    assert torch.isfinite(out["mel_after"]).all()


def test_inference_autoregressive():
    """Free-running decode returns [1, T<=max_len, n_mels] and stops/caps cleanly."""
    torch.manual_seed(0)
    m = _tiny()
    out = m.inference(torch.randn(1, 9, NMELS), max_len=20)
    assert out["mel_after"].shape[0] == 1 and out["mel_after"].shape[2] == NMELS
    assert 1 <= out["mel_after"].shape[1] <= 20
    assert out["n_frames"] == out["mel_after"].shape[1]
    assert torch.isfinite(out["mel_after"]).all()


def test_backward_reaches_all_heads():
    torch.manual_seed(0)
    m = _tiny()
    B, Ts, Tt = 2, 8, 10
    out = m(torch.randn(B, Ts, NMELS), torch.randn(B, Tt, NMELS))
    # include every head so all output params get gradients
    loss = (out["mel_before"].pow(2).mean() + out["mel_after"].pow(2).mean()
            + out["stop_logits"].pow(2).mean() + out["attn"].pow(2).mean())
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"params with no grad: {missing}"
