"""Minimal VTN-style Transformer-VC model (Issue 7), reimplemented clean in PyTorch.

Direct mel-to-mel voice conversion: an atypical **source** mel is encoded by a Transformer
encoder; a Transformer decoder autoregressively predicts the **target** mel (teacher-forced
in training) using a Tacotron 2 prenet, with a Postnet residual refinement and a stop-token
head. The encoder/decoder/attention come from the shared :mod:`sap.models.backbone`
(reused across VTN and VALL-E); the mel-domain pieces are Tacotron 2-derived.

**Forward contract** (``batch_first``, all mels are log-mel ``[B, T, n_mels]``):

    out = model(src_mel, tgt_mel, src_lens=None, tgt_lens=None)
    out["mel_before"]  # [B, T_tgt, n_mels]      pre-postnet prediction (full resolution)
    out["mel_after"]   # [B, T_tgt, n_mels]      mel_before + postnet residual
    out["stop_logits"] # [B, T_tgt]              end-of-sequence logit per frame
    out["attn"]        # [B, ceil(T_tgt/r), T_src] last-layer enc-dec attn (reduced rate)
    out["reduction"]   # int r

The decoder runs at a **reduced frame rate** (``reduction_factor`` r): each decoder step
predicts r mel frames at once, so the autoregressive sequence is r× shorter — this cuts
error accumulation (exposure bias) that otherwise muffles the back half of free-running
synthesis, and speeds up train/inference. mel/stop are returned at full resolution.

Source and target are **not** frame-aligned (atypical source is generally longer) — the
encoder-decoder cross-attention learns the mapping; nothing here assumes T_src == T_tgt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
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
    reduction_factor: int = 2  # mel frames predicted per decoder step (r=1 => frame-by-frame)


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

        # --- output heads (each reduced-rate step emits r frames) ---
        self.r = c.reduction_factor
        self.mel_out = nn.Linear(c.d_model, c.n_mels * self.r)
        self.stop_out = nn.Linear(c.d_model, self.r)
        self.postnet = Postnet(c.n_mels, c.postnet_channels, n_layers=c.postnet_layers)

    def forward(self, src_mel: Tensor, tgt_mel: Tensor,
                src_lens: Optional[Tensor] = None,
                tgt_lens: Optional[Tensor] = None,
                ss_prob: float = 0.0) -> Dict[str, Tensor]:
        device = src_mel.device
        r, n = self.r, self.config.n_mels
        B, T_src = src_mel.size(0), src_mel.size(1)
        T_tgt = tgt_mel.size(1)
        src_pad = _pad_mask(src_lens, T_src, device)

        # encode source
        memory = self.encoder(self.enc_pos(self.encoder_proj(src_mel)),
                              src_key_padding_mask=src_pad)

        # pad target to a multiple of r, group into reduced-rate frames
        pad = (r - T_tgt % r) % r
        tgt_p = F.pad(tgt_mel, (0, 0, 0, pad)) if pad else tgt_mel
        T_pad = tgt_p.size(1)
        T_dec = T_pad // r
        # decoder input at step t = last GT frame of group t-1 (go-frame for t=0)
        gt_last = tgt_p.view(B, T_dec, r, n)[:, :, -1, :]  # [B, T_dec, n]
        go = tgt_mel.new_zeros(B, 1, n)
        dec_lens = torch.ceil(tgt_lens.to(device).float() / r).long() if tgt_lens is not None else None
        tgt_pad = _pad_mask(dec_lens, T_dec, device)

        gt_dec_in = torch.cat([go, gt_last[:, :-1, :]], dim=1)
        if ss_prob and ss_prob > 0.0:
            # two-pass scheduled sampling (Mihaylova & Martins 2019): pass 1 teacher-forced
            # (no-grad) to harvest the model's own predicted frames, pass 2 feeds them back.
            with torch.no_grad():
                mb1, _, _, _ = self._run_decoder(memory, gt_dec_in, src_pad, tgt_pad, T_dec, T_pad, T_tgt)
            mb1p = F.pad(mb1, (0, 0, 0, pad)) if pad else mb1
            pred_last = mb1p.view(B, T_dec, r, n)[:, :, -1, :].detach()  # feed mel_before, detached
            use_model = torch.rand(B, T_dec, 1, device=device) < ss_prob
            mixed_last = torch.where(use_model, pred_last, gt_last)
            dec_in_frames = torch.cat([go, mixed_last[:, :-1, :]], dim=1)
        else:
            dec_in_frames = gt_dec_in

        mel_before, mel_after, stop_logits, attn = self._run_decoder(
            memory, dec_in_frames, src_pad, tgt_pad, T_dec, T_pad, T_tgt)
        return {"mel_before": mel_before, "mel_after": mel_after,
                "stop_logits": stop_logits, "attn": attn, "reduction": r}

    def _run_decoder(self, memory, dec_in_frames, src_pad, tgt_pad, T_dec, T_pad, T_tgt):
        """Run prenet -> decoder -> heads for given (possibly mixed) decoder input frames."""
        B, n = dec_in_frames.size(0), self.config.n_mels
        device = dec_in_frames.device
        dec_in = self.dec_pos(self.dec_proj(self.prenet(dec_in_frames)))
        causal = torch.triu(torch.ones(T_dec, T_dec, dtype=torch.bool, device=device), diagonal=1)
        h = dec_in
        for layer in self.dec_layers:
            h = layer(h, memory, tgt_mask=causal,
                      tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=src_pad)
        mel_before = self.mel_out(h).view(B, T_pad, n)[:, :T_tgt, :]
        mel_after = mel_before + self.postnet(mel_before)
        stop_logits = self.stop_out(h).reshape(B, T_pad)[:, :T_tgt]
        return mel_before, mel_after, stop_logits, self.dec_layers[-1].last_attn

    @torch.no_grad()
    def inference(self, src_mel: Tensor, max_len: int = 1000,
                  stop_threshold: float = 0.5) -> Dict[str, Tensor]:
        """Free-running synthesis from a single source mel ``[1, T_src, n_mels]``.

        Autoregressive: encode the source once, then predict the target mel frame-by-frame
        (Tacotron-style — feed the predicted pre-postnet frame back through the prenet),
        stopping when the stop-token fires or ``max_len`` is hit. Postnet is applied once to
        the full sequence. Returns ``mel_after [1, T_gen, n_mels]``, ``attn``, ``n_frames``.
        """
        self.eval()
        device = src_mel.device
        r, n = self.r, self.config.n_mels
        B = src_mel.size(0)
        memory = self.encoder(self.enc_pos(self.encoder_proj(src_mel)))
        dec_inputs = src_mel.new_zeros(B, 1, n)  # reduced-rate decoder inputs (go-frame)
        groups = []
        for _ in range(max(1, max_len // r)):
            h = self.dec_pos(self.dec_proj(self.prenet(dec_inputs)))
            T = h.size(1)
            causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
            for layer in self.dec_layers:
                h = layer(h, memory, tgt_mask=causal)
            last = h[:, -1:, :]
            group = self.mel_out(last).view(B, r, n)        # r frames this step
            groups.append(group)
            dec_inputs = torch.cat([dec_inputs, group[:, -1:, :]], dim=1)  # feed last frame back
            if torch.sigmoid(self.stop_out(last)).max().item() > stop_threshold:
                break
        mel_before = torch.cat(groups, dim=1)
        mel_after = mel_before + self.postnet(mel_before)
        return {"mel_after": mel_after, "attn": self.dec_layers[-1].last_attn,
                "n_frames": mel_before.size(1)}
