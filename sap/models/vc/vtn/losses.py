"""VTN training loss — ESPnet Transformer-VC formulation, reimplemented clean.

Spec reference: ESPnet ``TransformerLoss`` (shared by Transformer-TTS / VC) plus the
guided-attention loss. No ESPnet code imported.

    total = L1(before)+L1(after) + MSE(before)+MSE(after) + BCE(stop) + w * guided_attn

All mel/stop terms are masked to valid (non-padded) target frames. ``mel_l1``/``mel_mse``
(on the post-postnet output) and ``guided`` are returned separately so a trainer/test can
assess *learning* (reconstruction collapse + attention diagonalization), not just run a step.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sap.models.vc.vtn.modules import guided_attention_loss


def _frame_mask(lengths: Tensor, max_len: int, device) -> Tensor:
    """``[B, max_len]`` bool, True = valid frame."""
    return torch.arange(max_len, device=device).unsqueeze(0) < lengths.to(device).unsqueeze(1)


class VTNLoss(nn.Module):
    def __init__(self, bce_pos_weight: float = 50.0, lambda_stop: float = 5.0,
                 stop_terminal_window: int = 8, guided_weight: float = 100.0,
                 guided_sigma: float = 0.4):
        super().__init__()
        self.bce_pos_weight = bce_pos_weight
        self.lambda_stop = lambda_stop
        self.stop_terminal_window = stop_terminal_window
        self.guided_weight = guided_weight
        self.guided_sigma = guided_sigma

    def forward(self, out: Dict[str, Tensor], target_mel: Tensor,
                olens: Tensor, ilens: Optional[Tensor] = None
                ) -> Tuple[Tensor, Dict[str, Tensor]]:
        before, after = out["mel_before"], out["mel_after"]
        stop_logits, attn = out["stop_logits"], out.get("attn")
        device = target_mel.device
        T = target_mel.size(1)

        mmask = _frame_mask(olens, T, device).unsqueeze(-1).expand_as(target_mel)
        tgt_v = target_mel.masked_select(mmask)

        def l1(x):
            return F.l1_loss(x.masked_select(mmask), tgt_v)

        def mse(x):
            return F.mse_loss(x.masked_select(mmask), tgt_v)

        l1_loss = l1(before) + l1(after)
        l2_loss = mse(before) + mse(after)

        # stop target: positive over the last `stop_terminal_window` valid frames (denser
        # supervision than a single terminal frame -> reliable stopping), masked to valid frames
        fmask = _frame_mask(olens, stop_logits.size(1), device)
        stop_t = torch.zeros_like(stop_logits)
        K = self.stop_terminal_window
        for b, L in enumerate(olens.tolist()):
            stop_t[b, max(0, L - K):L] = 1.0
        bce = F.binary_cross_entropy_with_logits(
            stop_logits.masked_select(fmask), stop_t.masked_select(fmask),
            pos_weight=torch.tensor(self.bce_pos_weight, device=device),
        )

        if attn is not None:
            # attn is at the decoder's (reduced) frame rate -> scale target lengths by r
            r = out.get("reduction", 1)
            o_red = torch.ceil(olens.to(device).float() / r).long() if r > 1 else olens
            guided = guided_attention_loss(attn, ilens, o_red, self.guided_sigma)
        else:
            guided = torch.zeros((), device=device)

        total = l1_loss + l2_loss + self.lambda_stop * bce + self.guided_weight * guided
        stats = {"l1": l1_loss, "l2": l2_loss, "bce": bce, "guided": guided,
                 "mel_l1": l1(after), "mel_mse": mse(after)}
        return total, stats
