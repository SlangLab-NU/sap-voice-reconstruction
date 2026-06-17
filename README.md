# sap-voice-reconstruction

Consolidated **implementation** repository for atypical-speech voice reconstruction
in the Speech Accessibility Project (SAP) setting.

- **Operational guidance for agents/devs:** see [`CLAUDE.md`](./CLAUDE.md) (scope,
  reuse-first policy, execution environment, architecture + terminology rules).
- **Authoritative roadmap & planning docs:** the separate, versioned planning repo
  `planning-doc-voice-reconstruction` (issue plans, paper drafts, evaluation plans).
  It is the single source of truth — those docs are **not** duplicated here.
  `docs/` in this repo holds only **implementation-specific** notes authored as
  issues progress (e.g. the Issue 2 data-foundation contract note).

This scaffold is **Phase 1, Issue 1** (structure only — no model logic yet).

## Design in one paragraph

Treat `SlangLab-NU/sap-data-preparation` as the authoritative **SAP data
foundation**. On top of it, a shared **representation-preparation layer** (text /
phonemes / mel) and a **canonical adapter schema** feed model-facing **adapters**.
Two Phase 1 model families consume those adapters — **VTN-style Transformer-VC**
(direct voice conversion) and **cascaded ASR -> TTS**. A shared **evaluation**
harness (intelligibility, speaker preservation, acoustic distortion, latency /
throughput) compares them. A third family, **VALL-E-style** token generation, is
scaffolded for Phase 2 but is out of Phase 1 scope.

```
SAP foundation -> shared representation prep -> canonical schema
               -> task/model adapters -> train/infer wrappers
               -> evaluation + deployment analysis
```

## Layout

```text
configs/        # YAML configs: data/, vc/, cascaded/, valle/
data/           # foundation/ (sap-data-preparation outputs), derived/, splits/  (artifacts gitignored)
docs/           # impl-specific notes only; roadmap/paper docs stay in the planning repo
scripts/        # runnable entrypoints (sync, prepare, train, infer, eval, benchmark)
sap/            # the Python package
  data/
    foundation/ # wrap sap-data-preparation manifests / Lhotse / synthetic pairs
    adapters/   # vc_adapter, cascaded_adapter, valle_adapter
  models/
    vc/vtn/     # VTN-style Transformer-VC (Phase 1)
    cascaded/   # asr/ + tts/ wrappers + pipeline (Phase 1)
    valle/      # VALL-E-style (Phase 2, scaffold only)
  eval/         # shared evaluation harness
  utils/        # config, io, logging, checkpoints
cluster/        # slurm/ templates + env/ (conda/module setup) for the NU cluster
third_party/    # vendored / referenced upstream code
```

See `CLAUDE.md` for scope guardrails, the reuse-first order of operations, and NU
cluster path conventions (default new artifacts to `/scratch/aa.mohan`).
