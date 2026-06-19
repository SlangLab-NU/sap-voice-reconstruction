"""Tests for the vocoder interface, Griffin-Lim baseline, and HiFi-GAN scaffold."""
import pytest
import torch

from sap.data.features import MelConfig
from sap.data.vocoder import GriffinLimVocoder, build_vocoder


def test_build_vocoder_griffinlim_default():
    v = build_vocoder("griffinlim")
    assert isinstance(v, GriffinLimVocoder) and v.sample_rate == 24000


def test_build_vocoder_hifigan_requires_checkpoint():
    with pytest.raises(ValueError):
        build_vocoder("hifigan")


def test_hifigan_scaffold_raises_until_implemented():
    with pytest.raises(NotImplementedError):
        build_vocoder("hifigan", checkpoint="/nonexistent.pt")


def test_build_vocoder_unknown_name():
    with pytest.raises(ValueError):
        build_vocoder("nope")


def test_vocoder_logmel_to_waveform_shape():
    cfg = MelConfig()
    voc = GriffinLimVocoder(cfg, n_iter=4)  # few iters -> fast test
    T = 20
    log_mel = torch.randn(T, cfg.n_mels) * 2 - 4  # log-mel-ish range
    wav = voc(log_mel)
    assert wav.dim() == 1 and torch.isfinite(wav).all()
    # ~ hop_length * T samples
    assert abs(wav.numel() - cfg.hop_length * T) <= cfg.win_length


def test_vocoder_real_target_mel(vtn_pairs):
    """Round-trip a real target mel through the vocoder -> finite audio of plausible length."""
    from sap.data.features import MelExtractor

    ext = MelExtractor()
    voc = GriffinLimVocoder(n_iter=8)
    mel = ext.from_wav(vtn_pairs[0]["target"])
    wav = voc(mel)
    assert wav.dim() == 1 and torch.isfinite(wav).all()
    assert wav.numel() > ext.config.hop_length * (mel.shape[0] - 2)
