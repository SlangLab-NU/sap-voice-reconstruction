"""Griffin-Lim baseline vocoder (log-mel -> waveform), no checkpoint.

Phase-1 proof-of-life vocoder (cf. Issue 9 "a simple baseline is fine"). Inverts the
:class:`~sap.data.features.MelConfig` log-mel: ``exp`` -> ``InverseMelScale`` -> Griffin-Lim.
Lossy/robotic by nature (80-bin mel + phase reconstruction); use it to compare a predicted
mel against the *same vocoder's* rendering of the target mel, not against real audio. A
neural vocoder (e.g. HiFi-GAN) is a later upgrade.
"""
from __future__ import annotations

import torch
import torchaudio

from sap.data.features import VTN_MEL, MelConfig


class GriffinLimVocoder:
    def __init__(self, config: MelConfig = VTN_MEL, n_iter: int = 60):
        self.config = config
        self.inv_mel = torchaudio.transforms.InverseMelScale(
            n_stft=config.n_fft // 2 + 1,
            n_mels=config.n_mels,
            sample_rate=config.sample_rate,
            f_min=config.f_min,
            f_max=config.f_max,
            norm="slaney",
            mel_scale="slaney",
        )
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            power=config.power,
            n_iter=n_iter,
        )

    @torch.no_grad()
    def __call__(self, log_mel: torch.Tensor) -> torch.Tensor:
        """``log_mel`` ``[T, n_mels]`` -> waveform ``[T_wav]`` at ``config.sample_rate``."""
        mel = torch.exp(log_mel).transpose(-1, -2)  # [n_mels, T], undo log
        spec = self.inv_mel(mel)                     # [n_freq, T]
        return self.griffin_lim(spec)                # [T_wav]
