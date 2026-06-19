"""Vocoders: log-mel -> waveform. A small interface so the baseline (Griffin-Lim) and a
neural vocoder (HiFi-GAN) are interchangeable behind one contract.

The contract a vocoder must honor is its **mel profile**: it consumes log-mel in a specific
:class:`~sap.data.features.MelConfig`. Griffin-Lim is profile-agnostic (it inverts whatever
config it's given); a neural vocoder is trained on one profile and only sounds right on mels
in that profile — so it declares `expected_mel` and callers must feed matching mels.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio

from sap.data.features import VTN_MEL, MelConfig


class Vocoder(ABC):
    """log-mel ``[T, n_mels]`` -> waveform ``[T_wav]`` at ``self.sample_rate``."""

    expected_mel: MelConfig

    @property
    def sample_rate(self) -> int:
        return self.expected_mel.sample_rate

    @abstractmethod
    def __call__(self, log_mel: torch.Tensor) -> torch.Tensor:
        ...


class GriffinLimVocoder(Vocoder):
    """Baseline, no checkpoint. Inverts the given log-mel profile via InverseMelScale +
    Griffin-Lim. Profile-agnostic and lossy/robotic — the proof-of-life vocoder."""

    def __init__(self, config: MelConfig = VTN_MEL, n_iter: int = 60):
        self.expected_mel = config
        self.inv_mel = torchaudio.transforms.InverseMelScale(
            n_stft=config.n_fft // 2 + 1, n_mels=config.n_mels,
            sample_rate=config.sample_rate, f_min=config.f_min, f_max=config.f_max,
            norm="slaney", mel_scale="slaney",
        )
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=config.n_fft, win_length=config.win_length,
            hop_length=config.hop_length, power=config.power, n_iter=n_iter,
        )

    @torch.no_grad()
    def __call__(self, log_mel: torch.Tensor) -> torch.Tensor:
        mel = torch.exp(log_mel).transpose(-1, -2)  # [n_mels, T], undo log
        return self.griffin_lim(self.inv_mel(mel))  # [T_wav]


class HiFiGANVocoder(Vocoder):
    """Neural vocoder scaffold — a drop-in `Vocoder` for higher-quality audio.

    To activate, fill in `_load_generator` / `__call__` with a HiFi-GAN generator and a
    checkpoint trained on `expected_mel`. The hard requirement is **mel-profile match**:
    the checkpoint must be trained on the same MelConfig the VTN emits (default VTN_MEL:
    24k / 80 / n_fft 1024 / hop 256 / fmin 80 / fmax 7600). If only a differently-configured
    checkpoint exists, either source one matching this profile or re-extract mels + retrain
    VTN to the checkpoint's profile (see docs/issue 9). Until then this raises clearly.
    """

    def __init__(self, checkpoint: Union[str, Path], config: MelConfig = VTN_MEL,
                 device: str = "cpu"):
        self.expected_mel = config
        self.checkpoint = str(checkpoint)
        self.device = device
        self._generator = self._load_generator(self.checkpoint)

    def _load_generator(self, checkpoint: str):
        raise NotImplementedError(
            "HiFiGANVocoder is a scaffold. To enable it, implement _load_generator (HiFi-GAN "
            "generator architecture + weights) and __call__ (exp(log_mel) -> generator -> wav), "
            f"with a checkpoint trained on the mel profile {self.expected_mel}. "
            "Use --vocoder griffinlim until a profile-matching checkpoint is available."
        )

    @torch.no_grad()
    def __call__(self, log_mel: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        mel = torch.exp(log_mel).transpose(-1, -2)  # [n_mels, T]
        return self._generator(mel.unsqueeze(0).to(self.device)).squeeze().cpu()


def build_vocoder(name: str = "griffinlim", config: MelConfig = VTN_MEL,
                  checkpoint: Optional[Union[str, Path]] = None, device: str = "cpu",
                  n_iter: int = 60) -> Vocoder:
    """Factory: ``"griffinlim"`` (default) or ``"hifigan"`` (requires ``checkpoint``)."""
    if name == "griffinlim":
        return GriffinLimVocoder(config, n_iter=n_iter)
    if name == "hifigan":
        if checkpoint is None:
            raise ValueError("vocoder 'hifigan' requires a checkpoint path")
        return HiFiGANVocoder(checkpoint, config, device)
    raise ValueError(f"unknown vocoder: {name!r} (use 'griffinlim' or 'hifigan')")
