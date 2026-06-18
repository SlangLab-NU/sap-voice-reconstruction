"""Tacotron 2-style mel-decoder components for VTN, reimplemented clean in PyTorch.

Spec reference only: Shen et al. 2018 ("Natural TTS Synthesis…", Tacotron 2) and the
canonical NVIDIA/tacotron2 ``model.py`` for exact Prenet/Postnet shapes. No code vendored.

Also defines a thin VTN-local subclass of the shared backbone
:class:`~sap.models.backbone.TransformerDecoderLayer` that captures cross-attention
weights (the backbone layer runs cross-attn with ``need_weights=False``); VTN needs the
encoder-decoder attention for the guided-attention loss.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sap.models.backbone import TransformerDecoderLayer


class Prenet(nn.Module):
    """Tacotron 2 decoder prenet: stacked FC + ReLU with dropout applied **always**
    (train and eval) — the dropout is the prenet's regularizer/bottleneck at inference."""

    def __init__(self, in_dim: int, sizes=(256, 256), dropout: float = 0.5):
        super().__init__()
        dims = [in_dim] + list(sizes)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout
        self.out_dim = dims[-1]

    def forward(self, x: Tensor) -> Tensor:
        for lin in self.layers:
            x = F.dropout(F.relu(lin(x)), p=self.dropout, training=True)
        return x


class Postnet(nn.Module):
    """Tacotron 2 postnet: ``n_layers`` Conv1d(channels, k=5), tanh on all but the last,
    BatchNorm + dropout each; returns a **residual** to add to the predicted mel.

    Input/output are time-major ``[B, T, n_mels]`` (transposed internally for Conv1d)."""

    def __init__(self, n_mels: int = 80, channels: int = 512, kernel: int = 5,
                 n_layers: int = 5, dropout: float = 0.5):
        super().__init__()
        pad = (kernel - 1) // 2
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            in_ch = n_mels if i == 0 else channels
            out_ch = n_mels if i == n_layers - 1 else channels
            self.convs.append(nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
                                             nn.BatchNorm1d(out_ch)))
        self.n_layers = n_layers
        self.dropout = dropout

    def forward(self, mel: Tensor) -> Tensor:
        x = mel.transpose(1, 2)  # [B, n_mels, T]
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < self.n_layers - 1:
                x = torch.tanh(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x.transpose(1, 2)  # [B, T, n_mels] residual


class AttnTransformerDecoderLayer(TransformerDecoderLayer):
    """Backbone decoder layer that also stores its encoder-decoder attention.

    The shared backbone runs cross-attn with ``need_weights=False`` for speed; VTN needs
    the alignment for guided-attention, so we override only ``_mha_block`` to request and
    cache the (head-averaged) weights in ``self.last_attn`` (``[B, T_tgt, T_src]``)."""

    last_attn: Optional[Tensor] = None

    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask, key_padding_mask) -> Tensor:
        x, attn = self.multihead_attn(
            x, mem, mem, attn_mask=attn_mask,
            key_padding_mask=key_padding_mask, need_weights=True,
        )
        self.last_attn = attn
        return self.dropout2(x)


def guided_attention_loss(attn: Tensor, ilens: Optional[Tensor] = None,
                          olens: Optional[Tensor] = None, sigma: float = 0.4) -> Tensor:
    """Diagonal-encouraging loss on encoder-decoder attention (Tachibana et al. 2018).

    ``attn``: ``[B, T_out, T_in]`` (decoder queries × encoder keys). Penalizes attention
    mass far from the time-diagonal via ``W = 1 - exp(-(o/O - i/I)^2 / 2σ²)``. Masked to
    valid frames when ``ilens``/``olens`` are given. Returns a scalar.
    """
    B, T_out, T_in = attn.shape
    device = attn.device
    if olens is None:
        olens = torch.full((B,), T_out, device=device, dtype=torch.float)
    if ilens is None:
        ilens = torch.full((B,), T_in, device=device, dtype=torch.float)
    olens = olens.to(device=device, dtype=torch.float)
    ilens = ilens.to(device=device, dtype=torch.float)

    o_idx = torch.arange(T_out, device=device).float()
    i_idx = torch.arange(T_in, device=device).float()
    o = o_idx.view(1, T_out, 1) / olens.view(B, 1, 1)
    i = i_idx.view(1, 1, T_in) / ilens.view(B, 1, 1)
    W = 1.0 - torch.exp(-((o - i) ** 2) / (2 * sigma ** 2))

    valid = (o_idx.view(1, T_out, 1) < olens.view(B, 1, 1)) & \
            (i_idx.view(1, 1, T_in) < ilens.view(B, 1, 1))
    weighted = attn * W * valid
    return weighted.sum() / valid.sum().clamp(min=1)
