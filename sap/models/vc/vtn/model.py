"""Minimal VTN-style Transformer-VC model (Issue 7), reimplemented clean in PyTorch.

Direct mel-to-mel voice conversion: an atypical **source** mel is encoded by a Transformer
encoder; a Transformer decoder autoregressively predicts the **target** mel (teacher-forced
in training) using a Tacotron 2 prenet, with a Postnet residual refinement and a stop-token
head. The encoder/decoder/attention come from the shared :mod:`sap.models.backbone`
(reused across VTN and VALL-E); the mel-domain pieces are Tacotron 2-derived.

**Forward contract** (``batch_first``, all mels are log-mel ``[B, T, n_mels]``):

    out = model(src_mel, tgt_mel, src_lens=None, tgt_lens=None)
    out["mel_before"]  # [B, T_tgt, n_mels]  pre-postnet prediction
    out["mel_after"]   # [B, T_tgt, n_mels]  mel_before + postnet residual
    out["stop_logits"] # [B, T_tgt]          end-of-sequence logit per frame
    out["attn"]        # [B, T_tgt, T_src]   last-layer encoder-decoder attention

Source and target are **not** frame-aligned (atypical source is generally longer) — the
encoder-decoder cross-attention learns the mapping; nothing here assumes T_src == T_tgt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from sap.models.backbone import (
    SinePositionalEmbedding,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from sap.models.vc.vtn.modules import AttnTransformerDecoderLayer, Postnet, Prenet


@dataclass
class VTNConfig:
    n_mels: int = 80
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    prenet_dim: int = 256
    prenet_dropout: float = 0.5
    postnet_channels: int = 512
    postnet_layers: int = 5
    norm_first: bool = True


def _pad_mask(lengths: Optional[Tensor], max_len: int, device) -> Optional[Tensor]:
    """``[B, max_len]`` bool key-padding mask (True = padded/ignored), or None."""
    if lengths is None:
        return None
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    return idx >= lengths.to(device).unsqueeze(1)


class VTN(nn.Module):
    def __init__(self, config: VTNConfig = VTNConfig()):
        super().__init__()
        self.config = config
        c = config

        # --- encoder (source mel -> memory) ---
        self.encoder_proj = nn.Linear(c.n_mels, c.d_model)
        self.enc_pos = SinePositionalEmbedding(c.d_model)
        enc_layer = TransformerEncoderLayer(
            c.d_model, c.nhead, c.dim_feedforward, c.dropout,
            batch_first=True, norm_first=c.norm_first,
        )
        self.encoder = TransformerEncoder(enc_layer, c.num_encoder_layers)

        # --- decoder (Tacotron prenet -> Transformer decoder) ---
        self.prenet = Prenet(c.n_mels, (c.prenet_dim, c.prenet_dim), c.prenet_dropout)
        self.dec_proj = nn.Linear(c.prenet_dim, c.d_model)
        self.dec_pos = SinePositionalEmbedding(c.d_model)
        self.dec_layers = nn.ModuleList(
            AttnTransformerDecoderLayer(
                c.d_model, c.nhead, c.dim_feedforward, c.dropout,
                batch_first=True, norm_first=c.norm_first,
            )
            for _ in range(c.num_decoder_layers)
        )

        # --- output heads ---
        self.mel_out = nn.Linear(c.d_model, c.n_mels)
        self.stop_out = nn.Linear(c.d_model, 1)
        self.postnet = Postnet(c.n_mels, c.postnet_channels, n_layers=c.postnet_layers)

    def _shift_right(self, tgt_mel: Tensor) -> Tensor:
        """Prepend an all-zero go-frame and drop the last frame (teacher forcing input)."""
        B, _, n = tgt_mel.shape
        go = tgt_mel.new_zeros(B, 1, n)
        return torch.cat([go, tgt_mel[:, :-1, :]], dim=1)

    def forward(self, src_mel: Tensor, tgt_mel: Tensor,
                src_lens: Optional[Tensor] = None,
                tgt_lens: Optional[Tensor] = None) -> Dict[str, Tensor]:
        device = src_mel.device
        T_src, T_tgt = src_mel.size(1), tgt_mel.size(1)
        src_pad = _pad_mask(src_lens, T_src, device)
        tgt_pad = _pad_mask(tgt_lens, T_tgt, device)

        # encode source
        memory = self.encoder(self.enc_pos(self.encoder_proj(src_mel)),
                              src_key_padding_mask=src_pad)

        # decode (teacher forced) with a causal self-attention mask
        dec_in = self.dec_pos(self.dec_proj(self.prenet(self._shift_right(tgt_mel))))
        # bool causal mask (True = disallowed) — same dtype as the key-padding masks
        causal = torch.triu(torch.ones(T_tgt, T_tgt, dtype=torch.bool, device=device), diagonal=1)
        h = dec_in
        for layer in self.dec_layers:
            h = layer(h, memory, tgt_mask=causal,
                      tgt_key_padding_mask=tgt_pad,
                      memory_key_padding_mask=src_pad)

        mel_before = self.mel_out(h)
        mel_after = mel_before + self.postnet(mel_before)
        stop_logits = self.stop_out(h).squeeze(-1)
        attn = self.dec_layers[-1].last_attn  # [B, T_tgt, T_src]
        return {"mel_before": mel_before, "mel_after": mel_after,
                "stop_logits": stop_logits, "attn": attn}
