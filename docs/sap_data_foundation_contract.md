# SAP Data Foundation — Output Contracts (Issue 2)

**Status:** design note, Phase 1 Issue 2. Describes what `SlangLab-NU/sap-data-preparation`
produces so this repo can consume it as the authoritative **SAP data foundation**
(`sap/data/foundation/`) without reimplementing extraction, synthesis, or manifest logic.

**Sources:** `sap-data-preparation` scripts + its `docs/`, and the live artifacts under
`/projects/aanchan/data/` as of 2026-06-16. Upstream is the source of truth; if a script
changes, re-verify against it. Paths below are the **current local layout** — downstream
code must treat all roots as configurable (see CLAUDE.md cluster path conventions), not
hardcode `/projects/aanchan/...`.

**Provenance.** This contract was verified against `SlangLab-NU/sap-data-preparation`
commit `9212229` (`main`, 2026-05-19, "Add docs folder with prefix mode constraints and
WER proxy explainers"). The two repos stay decoupled — we consume artifacts, not source —
so when Step 6 manifests are generated, record the producing `sap-data-preparation` commit
alongside them (a provenance field/sidecar) and re-verify this note if that commit differs.

---

## Pipeline at a glance

| Step | Script | Produces | Consumed by |
|------|--------|----------|-------------|
| 1 | `extract_sap_data.py` | `extracted/{SPLIT}/{speaker_id}/` → `*_16kHz.wav` + `{speaker_id}.json` | 2,3,4 |
| 2 | `extract_speaker_data_and_ratings.py` | `speaker_ratings_{SPLIT}.csv` | 3,4,5 |
| 3 | `generate_synthetic_speech.py` | `synthetic/{Etiology}/{speaker_id}/*_synthetic.wav` + `speaker_pairs_{SPLIT}.csv` | 6, **our adapters** |
| 4 | `calculate_sap_wer.py` | per-speaker `Average_WER` CSV | 5 |
| 5 | `select_validation_speakers.py` | `val_speakers` CSV (stratified) | 6 |
| 6 | `sap.py` | Lhotse `sap_{recordings,supervisions}_{train,val,test}_{source,target}.jsonl[.gz]` | **our adapters** |

Splits at extraction are **TRAIN** and **DEV** only; there is **no TEST split** upstream.
`sap.py` maps `--test-csv` (the DEV pairs) onto the `test` manifests, and optionally carves
**VAL** out of TRAIN using Step 5's selected speakers.

The two artifacts our adapters actually consume are the **`speaker_pairs_{SPLIT}.csv`** (Step 3)
and the **Lhotse manifests** (Step 6). Steps 1/2/4/5 are upstream plumbing we read only for
metadata (ratings, WER, split assignment).

---

## Canonical keys & the source↔target pairing model

- **`speaker_id`** — UUID v4 (e.g. `74613052-fd59-45c8-ab54-08db7376d336`). The join key across
  every CSV and manifest.
- **Audio filename stem** encodes the utterance: `{speaker_id}_{utterance_id}_{prompt_id}_16kHz`.
  - original: `…_16kHz.wav`; synthetic: `…_16kHz_synthetic.wav`.
- **Lhotse recording/supervision `id`** = `{speaker_id}_{stem}` (so the speaker UUID appears twice).
  Source and target for the same utterance share this `id`, which is how a pair is linked.
- **Pairing = same utterance, two renderings:**
  - **source** = original *atypical* speech (the SAP recording).
  - **target** = *synthetic* speech of the same transcript (StyleTTS2, the reconstruction target).
  - ⚠️ **Not frame-aligned.** Atypical source is generally longer/slower than its synthetic
    target (see upstream `docs/prefix_mode_and_data_loss.md`). Do **not** assume frame-level
    alignment; VC training must map full source context → target, not a per-frame mapping.

---

## Step 2 — `speaker_ratings_{SPLIT}.csv`

Columns (verbatim): `Speaker_ID`, `Etiology`, `Average_Rating`, `Number_of_Ratings`,
`Number_Not_Rated`, `Total_Utterances`.

- `Average_Rating` is the literal string `N/A` when a speaker has no ratings → load with
  `pd.to_numeric(..., errors="coerce")`.
- **Rating scale is inverted:** 1 = most intelligible, 7 = least. Most speakers are largely
  unrated (only a handful of utterances per speaker carry ratings), which is why Step 4/5 use
  WER as an intelligibility proxy to stratify the unrated majority.

## Step 3 — `speaker_pairs_{SPLIT}.csv`  (primary adapter input)

Emitted per etiology under `synthetic/{Etiology}/speaker_pairs_{SPLIT}.csv`, and as a merged
`synthetic/speaker_pairs_{SPLIT}.csv` when run with `--mode all`.

Columns (verbatim): `speaker_id`, `etiology`, `original_audio`, `synthetic_audio`,
`transcript`, `prompt_text`, `category_description`, `status`.

- `original_audio` / `synthetic_audio` are **absolute paths** to the source and target wavs.
- `transcript` = cleaned spoken transcript (used as the synthesis input). `prompt_text` = the
  script shown to the speaker (may differ from what was said).
- `status` ∈ {`success`, `skipped`, `failed: {error}`}. **Filter to `status == "success"`**
  for complete pairs; `failed` rows have `synthetic_audio` null/NaN; re-runs can emit `skipped`
  duplicate rows → **dedup on `(speaker_id, original_audio)`** before use.
- `category_description` (e.g. `Digital Assistant Commands`, `Novel Sentences`,
  `Spontaneous Speech Prompts`) is preserved for downstream stratification.

## Step 6 — Lhotse manifests  (primary adapter input)

Files (per split × role), written `.jsonl.gz` by default (`--json` for plain `.jsonl`):
`sap_recordings_{train,val,test}_{source,target}` and `sap_supervisions_{train,val,test}_{source,target}`.
`val_*` only exist when `sap.py` is given `--val-speakers`.

**Recording** fields: `id`, `sources:[{type:"file", channels:[0], source:<abs wav path>}]`,
`duration`, `sampling_rate` (16000), `num_samples`. Mono, 16 kHz.

**Supervision** fields: `id`, `recording_id` (= `id`), `start` (0.0), `duration`, `channel` (0),
`language` ("English"), `speaker` (= `speaker_id`), `text` (cleaned transcript), and `custom`.

- `custom` = `{prompt_text, category_description, etiology}` on **source**; **target omits
  `etiology`** → always read it as `custom.get("etiology")`, never `custom["etiology"]`.
- `sap.py` excludes pairs whose `status` starts with `failed`, and asserts no speaker overlap
  between TRAIN and VAL after the Step 5 split.

---

## What downstream layers may rely on (contract we build adapters against)

1. **`speaker_id` is the universal join key**; pair identity is the shared `{speaker_id}_{stem}`.
2. **All audio is 16 kHz mono `.wav`**, paths absolute.
3. **Use `transcript` (not `prompt_text`)** as the canonical text — it is what was actually
   spoken and what synthesis/WER align to. `prompt_text` needs email/URL normalization and is
   not synthesis-aligned.
4. **`status == "success"`** is the validity gate for pairs; manifests already apply the
   equivalent filter (drop `failed`).
5. **Source = atypical original, target = synthetic typical**, same transcript, **not time-aligned**.
6. Etiology/category metadata is available for stratified splits and per-condition reporting.

These map onto the planning README's canonical adapter schema (`utt_id`, `speaker_id`, `split`,
`wav_path`, `transcript`/`normalized_text`/`phonemes`, `source_speaker`/`target_speaker`,
`speaker_type`, `severity`, …) — Issue 4 will formalize that schema; this note is its input.

---

## Unresolved questions / ambiguities

These need a decision or verification before/while building Issues 3–6:

1. **RESOLVED — Lhotse manifests are built** (2026-06-17) at `/projects/aanchan/data/manifests/`:
   `sap_{recordings,supervisions}_{train,val,test}_{source,target}.jsonl.gz`. Verified against
   this contract (recording `id` = `<speaker>_<stem>`, text in supervisions, source 16 kHz /
   target 24 kHz, `custom.etiology` source-only). Counts: train 244,727 pairs / 639 spk, val
   41,968 / 108, test 47,781 / 123. The VTN data path now consumes these via
   `sap/data/manifest.py` (join source/target 1:1 by `id`). Pairs-CSV path is retired.
2. **`normalized_text` and `phonemes` are not produced upstream.** The foundation gives raw +
   lightly-cleaned `transcript` only. Text normalization and phonemization are **our** Step-3
   (Issue 3) responsibility — decide the normalizer/phonemizer and whether to reuse VallE's.
3. **RESOLVED — split terminology.** `train`/`val`/`test` manifests where **val = held out from
   SAP TRAIN** and **test = SAP DEV**; UASpeech is a separate zero-shot cross-corpus test (not in
   these manifests). Val speakers are carved from TRAIN with no train/val speaker overlap.
4. **Etiology string vs. directory form.** CSVs/manifests use human form ("Parkinson's Disease");
   directories use slugged form ("Parkinsons_Disease"). Adapters that walk directories must apply
   the same slugging (spaces→`_`, drop apostrophes); don't assume the CSV value matches a path.
5. **Speaker-level control/typical references.** The "target" here is *synthesized* typical
   speech, not a matched human control speaker. If speaker-similarity evaluation wants a real
   reference, that source is undefined — flag for the eval design (Issue 14).
6. **Duration-mismatch handling for VC.** Confirm max-duration filtering / batching policy for
   non-aligned pairs (upstream `docs/prefix_mode_and_data_loss.md` notes OOM at long durations).
7. **Cued-speech / multi-speaker exclusions and bracketed cues** are dropped upstream and never
   reach `text`; quantify the resulting utterance loss when reporting dataset stats.
