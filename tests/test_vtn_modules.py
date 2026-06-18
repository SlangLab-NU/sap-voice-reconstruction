"""Tests for the Tacotron 2-style VTN components and the guided-attention loss."""
import torch

from sap.models.backbone import TransformerEncoder, TransformerEncoderLayer
from sap.models.vc.vtn.modules import (
    AttnTransformerDecoderLayer,
    Postnet,
    Prenet,
    guided_attention_loss,
)

B, T, NMELS, D, H = 2, 11, 80, 32, 4


def test_prenet_shape():
    pre = Prenet(NMELS, (64, 64))
    assert pre(torch.randn(B, T, NMELS)).shape == (B, T, 64)


def test_prenet_dropout_active_in_eval():
    """Tacotron 2 prenet keeps dropout ON at inference -> two eval passes differ."""
    pre = Prenet(NMELS, (64, 64), dropout=0.5).eval()
    x = torch.randn(B, T, NMELS)
    torch.manual_seed(1); a = pre(x)
    torch.manual_seed(2); b = pre(x)
    assert not torch.allclose(a, b)


def test_postnet_is_residual_shape():
    post = Postnet(NMELS, channels=32, n_layers=5)
    res = post(torch.randn(B, T, NMELS))
    assert res.shape == (B, T, NMELS) and torch.isfinite(res).all()


def test_attn_decoder_layer_captures_attention():
    layer = AttnTransformerDecoderLayer(D, H, dim_feedforward=64, dropout=0.0, batch_first=True)
    tgt, mem = torch.randn(B, 5, D), torch.randn(B, T, D)
    layer(tgt, mem)
    assert layer.last_attn is not None
    assert layer.last_attn.shape == (B, 5, T)  # [B, T_tgt, T_src]


def test_guided_attention_loss_scalar_and_range():
    attn = torch.softmax(torch.randn(B, 9, 7), dim=-1)
    loss = guided_attention_loss(attn)
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert 0.0 <= float(loss) <= 1.0


def test_guided_attention_diagonal_beats_offdiagonal():
    """A diagonal alignment should incur lower guided-attn loss than an anti-diagonal one."""
    To = Ti = 8
    diag = torch.eye(Ti).unsqueeze(0)               # [1, To, Ti] perfectly diagonal
    anti = torch.flip(torch.eye(Ti), dims=[1]).unsqueeze(0)
    assert float(guided_attention_loss(diag)) < float(guided_attention_loss(anti))


def test_guided_attention_masks_lengths():
    attn = torch.softmax(torch.randn(2, 10, 10), dim=-1)
    full = guided_attention_loss(attn)
    masked = guided_attention_loss(attn, ilens=torch.tensor([10, 4]), olens=torch.tensor([10, 5]))
    assert torch.isfinite(masked) and float(masked) != float(full)
