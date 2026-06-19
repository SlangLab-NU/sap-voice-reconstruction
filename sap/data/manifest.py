"""VTN data path over the SAP Step-6 Lhotse manifests (replaces the speaker_pairs.csv path).

Loads the paired source/target CutSets and joins them 1:1 by shared ``id``
(``<speaker>_<original_stem>``). ``source`` = original atypical audio (16 kHz);
``target`` = StyleTTS2 synthetic (24 kHz). Both are turned into log-mel with the **same**
:class:`~sap.data.features.MelExtractor`, which resamples each to the profile sample rate
(VTN: 24 kHz) — so source/target mels share one config despite the input SR mismatch.

Source and target are **not** frame-aligned (e.g. 8.08 s source vs 2.07 s target); the
loader just returns both mel sequences and their lengths — the seq2seq model/attention learns
the alignment. See ``docs/sap_data_foundation_contract.md`` and the ``vtn-manifest-handoff`` note.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sap.data.features import VTN_MEL, MelConfig, MelExtractor

DEFAULT_MANIFEST_DIR = Path("/projects/aanchan/data/manifests")
SPLITS = ("train", "val", "test")  # val = held out from SAP TRAIN; test = SAP DEV


def load_pair_cutsets(manifest_dir: Union[str, Path], split: str):
    """Return ``(source_cuts, target_cuts)`` CutSets for a split."""
    from lhotse import CutSet, RecordingSet, SupervisionSet

    d = Path(manifest_dir)

    def cuts(role: str):
        return CutSet.from_manifests(
            recordings=RecordingSet.from_file(d / f"sap_recordings_{split}_{role}.jsonl.gz"),
            supervisions=SupervisionSet.from_file(d / f"sap_supervisions_{split}_{role}.jsonl.gz"),
        )

    return cuts("source"), cuts("target")


class VTNManifestDataset(Dataset):
    """Paired (source mel, target mel) examples from the Lhotse manifests.

    ``__getitem__`` returns a dict: ``id, speaker, text, etiology, source_mel [Ts, n_mels],
    target_mel [Tt, n_mels]``. Pass ``ids`` to restrict to a subset (e.g. for quick runs).
    """

    def __init__(self, manifest_dir: Union[str, Path] = DEFAULT_MANIFEST_DIR,
                 split: str = "train", mel: Optional[MelConfig] = None,
                 ids: Optional[Sequence[str]] = None,
                 max_duration: Optional[float] = None):
        src_cuts, tgt_cuts = load_pair_cutsets(manifest_dir, split)
        self._src: Dict[str, object] = {c.id: c for c in src_cuts}
        self._tgt: Dict[str, object] = {c.id: c for c in tgt_cuts}
        if ids is None:
            ids = sorted(self._src.keys() & self._tgt.keys())
        if max_duration is not None:  # drop very long utts (O(T^2) attention -> OOM)
            ids = [i for i in ids
                   if self._src[i].duration <= max_duration
                   and self._tgt[i].duration <= max_duration]
        self.ids: List[str] = list(ids)
        self.mel = MelExtractor(mel or VTN_MEL)

    def __len__(self) -> int:
        return len(self.ids)

    def _mel_of(self, cut) -> torch.Tensor:
        wav = torch.from_numpy(cut.load_audio())  # [C, T] at cut.sampling_rate
        return self.mel.from_waveform(wav, cut.sampling_rate)  # [T, n_mels] at common SR

    def __getitem__(self, i: int) -> Dict:
        uid = self.ids[i]
        scut, tcut = self._src[uid], self._tgt[uid]
        sup = scut.supervisions[0]
        return {
            "id": uid,
            "speaker": sup.speaker,
            "text": sup.text,
            "etiology": (sup.custom or {}).get("etiology"),
            "source_mel": self._mel_of(scut),
            "target_mel": self._mel_of(tcut),
        }


def collate_vtn(batch: List[Dict]) -> Dict:
    """Pad source/target mels to batch max; return tensors + lengths + metadata."""
    src = [b["source_mel"] for b in batch]
    tgt = [b["target_mel"] for b in batch]
    src_lens = torch.tensor([m.shape[0] for m in src])
    tgt_lens = torch.tensor([m.shape[0] for m in tgt])

    def pad(mels, T):
        return torch.stack([F.pad(m, (0, 0, 0, T - m.shape[0])) for m in mels])

    return {
        "src_mel": pad(src, int(src_lens.max())),
        "tgt_mel": pad(tgt, int(tgt_lens.max())),
        "src_lens": src_lens,
        "tgt_lens": tgt_lens,
        "ids": [b["id"] for b in batch],
        "speakers": [b["speaker"] for b in batch],
        "texts": [b["text"] for b in batch],
    }


def make_dataloader(manifest_dir: Union[str, Path] = DEFAULT_MANIFEST_DIR,
                    split: str = "train", batch_size: int = 8, shuffle: bool = True,
                    num_workers: int = 0, mel: Optional[MelConfig] = None,
                    ids: Optional[Sequence[str]] = None,
                    max_duration: Optional[float] = None) -> DataLoader:
    ds = VTNManifestDataset(manifest_dir, split, mel=mel, ids=ids, max_duration=max_duration)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_vtn)
