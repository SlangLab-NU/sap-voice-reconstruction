"""Acoustic feature extraction — VTN-profile mel (minimal slice of the Issue 3 layer).

Clean implementation on torch/torchaudio (no ESPnet). A :class:`MelConfig` is a
"representation profile"; :data:`VTN_MEL` pins the proven VTN Transformer-VC recipe
profile. SAP foundation audio is 16 kHz mono, so extraction resamples to the profile's
sample rate (VTN: 24 kHz) before computing log-mel.

These params are *ours to set* (the recipe values are a reference target, not imported).
The cascaded StyleTTS2 path uses a different profile (24k / n_fft 2048 / win 1200 / hop
300) and is added later as a separate :class:`MelConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchaudio


@dataclass(frozen=True)
class MelConfig:
    """A mel "representation profile". Defaults are the VTN Transformer-VC profile."""

    sample_rate: int = 24000
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    f_min: float = 80.0
    f_max: float = 7600.0
    power: float = 1.0  # magnitude spectrogram
    log_eps: float = 1e-5
    name: str = "vtn"


#: Proven VTN profile (from the espnet jp_dialect/vc1 recipe — ours to set, not imported).
VTN_MEL = MelConfig()


class MelExtractor:
    """Waveform/file -> log-mel ``[T, n_mels]`` (time-major, to match model ``[B, T, n_mels]``)."""

    def __init__(self, config: MelConfig = VTN_MEL):
        self.config = config
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=config.power,
            center=True,
            norm="slaney",
            mel_scale="slaney",
        )

    def _to_logmel(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [n_mels, T] -> log -> [T, n_mels]
        mel = torch.log(torch.clamp(mel, min=self.config.log_eps))
        return mel.transpose(-1, -2).contiguous()

    def from_waveform(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """``wav``: ``[T]`` or ``[C, T]``. Returns log-mel ``[T_mel, n_mels]``."""
        if wav.dim() == 2:  # [C, T] -> mono
            wav = wav.mean(dim=0)
        wav = wav.to(torch.float32)
        if sample_rate != self.config.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.config.sample_rate)
        return self._to_logmel(self._mel(wav))

    def from_wav(self, path: Union[str, Path]) -> torch.Tensor:
        """Read a wav from disk and return log-mel ``[T_mel, n_mels]``."""
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        wav = torch.from_numpy(np.asarray(data))
        if wav.dim() == 2:  # soundfile returns [T, C] -> [C, T]
            wav = wav.transpose(0, 1)
        return self.from_waveform(wav, sr)
