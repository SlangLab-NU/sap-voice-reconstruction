"""Tests for the VTN-profile mel extractor (sap.data.features)."""
import torch

from sap.data.features import MelConfig, MelExtractor, VTN_MEL


def test_vtn_profile_values():
    assert (VTN_MEL.sample_rate, VTN_MEL.n_mels, VTN_MEL.n_fft, VTN_MEL.hop_length) == (24000, 80, 1024, 256)
    assert VTN_MEL.f_min == 80.0 and VTN_MEL.f_max == 7600.0


def test_from_waveform_shape_and_finite():
    ext = MelExtractor()
    wav = torch.randn(24000)  # 1s @ 24k
    mel = ext.from_waveform(wav, 24000)
    assert mel.dim() == 2 and mel.shape[1] == 80
    assert torch.isfinite(mel).all()
    # ~ T/hop frames
    assert abs(mel.shape[0] - (24000 // 256 + 1)) <= 1


def test_stereo_is_downmixed():
    ext = MelExtractor()
    m = ext.from_waveform(torch.randn(2, 16000), 16000)  # [C, T]
    assert m.shape[1] == 80 and torch.isfinite(m).all()


def test_resample_changes_frame_count():
    ext = MelExtractor()  # target 24k
    n = 16000
    m16 = ext.from_waveform(torch.randn(n), 16000)  # resampled 16k->24k
    m24 = ext.from_waveform(torch.randn(int(n * 24000 / 16000)), 24000)
    assert abs(m16.shape[0] - m24.shape[0]) <= 1  # same duration -> ~same frames


# ---- real-data (skips if fixture audio absent) ----

def test_real_pair_mels(vtn_pairs):
    ext = MelExtractor()
    for p in vtn_pairs:
        src = ext.from_wav(p["source"])  # 16k original
        tgt = ext.from_wav(p["target"])  # 24k synthetic
        assert src.shape[1] == 80 and tgt.shape[1] == 80
        assert torch.isfinite(src).all() and torch.isfinite(tgt).all()
        # atypical source is slower/longer than the synthetic target (non-aligned pairs)
        assert src.shape[0] > tgt.shape[0]
