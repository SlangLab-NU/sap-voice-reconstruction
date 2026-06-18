"""Tests for the shared transformer backbone (adopted from VallE @ e65a69b).

Per the project rule, adopted/pulled code is unit-tested too: these confirm the lift
imports cleanly in our env and the modules satisfy their shape/mask contracts.
"""
import torch

from sap.models.backbone import (
    MultiheadAttention,
    SinePositionalEmbedding,
    TokenEmbedding,
    Transpose,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

B, T, D, H = 2, 7, 32, 4


def _enc():
    layer = TransformerEncoderLayer(D, H, dim_feedforward=64, dropout=0.0, batch_first=True)
    return TransformerEncoder(layer, num_layers=2)


def test_encoder_shape():
    x = torch.randn(B, T, D)
    assert _enc()(x).shape == (B, T, D)


def test_encoder_padding_mask_runs():
    x = torch.randn(B, T, D)
    pad = torch.zeros(B, T, dtype=torch.bool)
    pad[1, 4:] = True
    y = _enc()(x, src_key_padding_mask=pad)
    assert y.shape == (B, T, D) and torch.isfinite(y).all()


def test_mha_returns_attention():
    x = torch.randn(B, T, D)
    out, attn = MultiheadAttention(D, H, batch_first=True)(x, x, x, need_weights=True)
    assert out.shape == (B, T, D)
    assert attn.shape == (B, T, T)  # head-averaged


def test_decoder_layer_cross_attn_shape():
    layer = TransformerDecoderLayer(D, H, dim_feedforward=64, dropout=0.0, batch_first=True)
    tgt, mem = torch.randn(B, 5, D), torch.randn(B, T, D)
    causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    out = layer(tgt, mem, tgt_mask=causal)
    assert out.shape == (B, 5, D) and torch.isfinite(out).all()


def test_sine_positional_embedding():
    x = torch.randn(B, T, D)
    assert SinePositionalEmbedding(D)(x).shape == (B, T, D)


def test_token_embedding():
    emb = TokenEmbedding(dim_model=D, vocab_size=10)
    ids = torch.randint(0, 10, (B, T))
    assert emb(ids).shape == (B, T, D)


def test_transpose():
    x = torch.randn(B, T, D)
    assert Transpose()(x).shape == (B, D, T)
