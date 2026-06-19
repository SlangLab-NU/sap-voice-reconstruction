# Testing approach

Believable, real-data tests over synthetic-only ones. Not captured in the planning repo —
this note records the convention for this implementation repo.

## Principles

- **Adopted/pulled code is tested too.** The shared `sap/models/backbone/` was lifted from
  `SlangLab-NU/VallE`; it still gets unit tests (import/dep sanity after the lift + shape/mask
  contracts). Adopted code is not trusted blindly.
- **Believable over synthetic.** The headline model test is an **overfit-single-batch** on
  *real* SAP pairs with the *real* training loss, asserting the model actually learns — not
  just that a forward/backward step runs without erroring.
- **Real audio is never committed.** SAP data is access-controlled. Fixture audio lives under
  `tests/fixtures/audio/` which is gitignored (`*.wav`); only the manifest
  `tests/fixtures/vtn_pairs.jsonl` is committed. Tests **skip** cleanly when the audio is
  absent (fresh clone / CI); synthetic shape-guard tests run everywhere.

## Direction of the task

VTN learns **atypical source mel → synthetic target mel**: input is the original disordered
SAP recording; the target is the StyleTTS2-synthesized rendering of the same transcript
(styled on the speaker's own voice — intelligible content, preserved identity). The target is
a generated artifact of the data foundation (Step 3), not a matched human "typical" recording
(see `sap_data_foundation_contract.md`). Source and target are **not frame-aligned**.

## Tiers

1. **Synthetic shape guards** (fast, everywhere): forward/contract shapes, unequal
   `T_src != T_tgt`, padded-batch masks, gradient flow.
2. **Real-data forward** (skips if audio absent): real wav → VTN-profile mel → model; shapes
   and finiteness on real mels.
3. **Overfit-single-batch** (skips if audio absent): the believable one — see below.

## What "learning as expected" means (overfit test)

Loss = ESPnet Transformer-VC formulation (`sap/models/vc/vtn/losses.py`):
`L1(before)+L1(after) + MSE(before)+MSE(after) + stop-BCE + guided-attention`, masked to
valid frames. Overfitting a small real batch must show:

1. **Reconstruction collapses** — mel L1/MSE drop to a small fraction of their initial value
   (the independent "it fit the target mel" signal).
2. **Attention diagonalizes** — encoder-decoder attention develops a near-diagonal alignment
   (guided-attention loss decreases); the hallmark a seq2seq VC model learned the mapping.
3. Stop-token predictions become correct.

## Fixture

Three approved short Parkinson's command utterances ("Turn on heat.", "What's the time?",
"Make it warmer."), source 16 kHz / synthetic target 24 kHz. Approved by listening.

## Listenable proof-of-life

Learnability is gated by the `test_vtn_real.py` overfit test (no audio). To actually *hear* a
trained model, `scripts/infer_vtn.py` loads a checkpoint and renders, per source utterance, the
original source, the target mel, a teacher-forced reconstruction, and a free-running synthesis —
all through the **Griffin-Lim baseline vocoder** (`sap/data/vocoder.py`). Compare
`*_synth_free_GL` vs `*_target_GL` (same vocoder) to isolate the model from Griffin-Lim's own
quality; `*_recon_tf_GL` is the teacher-forced best case. A neural vocoder (HiFi-GAN) is a
follow-up upgrade.

## Running

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # CPU torch is enough
.venv/bin/python -m pytest                      # synthetic always; real-data if fixture audio present
.venv/bin/python scripts/vtn_overfit_demo.py    # render listenable overfit audio
```
