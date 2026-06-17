# CLAUDE.md

## Project overview
This is the **implementation** repository for a unified speech reconstruction framework for the Speech Accessibility Project (SAP) setting. The planning and coordination repo is `planning-doc-voice-reconstruction`, whose `docs/issues/phase1_issue_plan.md` is the authoritative roadmap.

The project is intentionally split into two phases:

- **Phase 1**: establish the SAP data foundation, a minimal **VTN-style Transformer-VC** direct voice conversion path, a reproducible **cascaded ASR -> TTS** baseline, shared evaluation, and deployment-oriented analysis.
- **Phase 2**: extend the framework with stronger direct VC variants, **VALL-E-style** token-based modeling, and a broader cross-paradigm benchmark.

This guidance originated in the planning repo but was written to govern coding-agent work in this implementation repo; it now lives here as the operational source of truth for building.

---

## Scope boundaries

### Phase 1 in scope
Phase 1 is intentionally limited to:
- SAP data foundation integration via `SlangLab-NU/sap-data-preparation`
- shared representation-preparation framework for text and acoustic representations
- VTN-style direct VC
- cascaded ASR -> TTS
- shared evaluation
- latency / throughput / chunked-interactive feasibility
- paper-planning assets for a Phase 1 submission

### Phase 1 out of scope
The following are **not** Phase 1 tasks:
- VALL-E model implementation
- neural audio codec token generation
- full token-model benchmarking
- scope expansion beyond the two-paradigm Phase 1 paper story

### Phase 2 in scope
Phase 2 may include:
- pretrained VTN extensions
- speaker conditioning
- neural audio codec token extraction
- prompt-token packaging
- VALL-E-style integration
- unified three-family benchmarking

---

## Execution environment and operational context

Use the following environment assumptions unless a repo-specific implementation doc explicitly overrides them:

- canonical project root: `/projects/van-speech-nlp`
- raw SAP dataset location: `/projects/van-speech-nlp/SAPC-DATASET`
- shared container directory: `/projects/van-speech-nlp/VallE_containers`
- preferred location for newly created data/model artifacts on the NU cluster: `/scratch/aa.mohan`

Known containers:

### SAP data-preparation containers
- `sap_analysis.sif`
- `sap_data_prep.sif`

### VALL-E-related containers
- `valle_container3.sif`
- `valle_h200_container.sif`

If a required container is missing, it may be pulled or built using:

```bash
apptainer build <container_name>.sif docker://jordanwhlewis/<container_name>:latest
```

### Storage guidance
When creating new derived artifacts on the NU cluster, prefer storing them under `/scratch/aa.mohan` unless a task explicitly requires another location.

This includes, where appropriate:
- generated manifests and intermediate derived data
- cached prepared representations
- checkpoints
- experiment outputs
- logs
- evaluation outputs
- temporary run artifacts

Do not assume newly created data or model artifacts should be written under `/projects/van-speech-nlp` unless that is explicitly required for shared canonical resources.

### Initial execution order
Before model-side implementation work, prefer this sequence:
1. verify access to `/projects/van-speech-nlp/SAPC-DATASET`
2. verify required containers exist under `/projects/van-speech-nlp/VallE_containers`
3. run `sap-data-preparation` first
4. inspect and document generated manifests/artifacts/contracts
5. only then proceed to adapters, shared representations, and model-side integration

Treat successful SAP data preparation as the first concrete execution milestone for Phase 1.

---

## Reuse-first policy
Prefer **reusing, adapting, wrapping, or minimally vendoring** proven code from upstream repositories instead of writing fresh implementations.

### Default order of operations
When implementing anything, follow this order:
1. **inspect existing upstream code**
2. **reuse directly if possible**
3. **wrap or adapt if direct reuse is awkward**
4. **copy/minimally vendor only if needed**
5. **rewrite only as a last resort**

### Rewrite only if
Fresh implementation should happen only if:
- the upstream code is fundamentally incompatible with the SAP data contracts,
- the code is too entangled or brittle to reuse safely,
- or maintaining compatibility would be more complex than a clean replacement.

### Avoid
Avoid creating:
- duplicate tokenization pipelines
- parallel data-preparation stacks
- unnecessary rewrites of working trainers/inference scripts
- architecture drift between Phase 1 and Phase 2
- disconnected VALL-E-specific infrastructure that ignores the shared scaffold

---

## Upstream repositories and source-of-truth guidance

### `SlangLab-NU/sap-data-preparation`
Treat this repo as the **authoritative SAP data foundation**.

Preferred usage:
- consume its manifests, outputs, and pairing artifacts
- preserve its working contracts where possible
- build ingestion/validation/adapters on top of it

Do **not** casually reimplement SAP data preparation logic in downstream repos.

### `SlangLab-NU/VallE`
Treat this repo as the primary upstream source for:
- token/representation preparation ideas
- collaters and batching patterns
- transformer utility modules
- trainer/inference structure
- VALL-E-style integration in Phase 2

Prefer selective reuse of proven components over recreating VALL-E-related plumbing from scratch.

### Existing cascaded ASR/TTS pipelines
Treat existing working cascaded systems as wrapper/integration targets.

Preferred usage:
- wrap working inference paths
- preserve proven invocation logic where possible
- consolidate interfaces and evaluation around them

Do **not** rebuild ASR or TTS internals unless there is a compelling reason.

---

## Architecture rules

### Shared representation-preparation framework
Phase 1 should establish a **shared representation-preparation framework** that supports:
- text / normalized text / phonemes
- mel spectrograms or related acoustic representations
- cached artifacts and manifest references
- model-ready adapter outputs for Phase 1 systems

Phase 2 should extend this same framework with:
- neural audio codec token extraction
- prompt-token packaging
- token-ready manifests for VALL-E-style workflows

Do **not** create a disconnected token pipeline for Phase 2 if the same scaffold can be extended.

### Shared scaffold vs model-specific logic
Prefer sharing:
- data contracts
- caching/manifests
- configuration conventions
- transformer building blocks
- evaluation harnesses
- latency/deployment benchmarking plumbing

Keep model-specific:
- VTN mel-output heads and stop-token logic
- VALL-E codec-token heads and AR/NAR generation logic
- system-specific loss definitions
- model-family-specific decoding behavior

### Adapter-first integration
Use adapters and wrappers to connect systems to the shared SAP foundation.

Preferred architecture pattern:
- SAP foundation outputs
- shared representation-preparation layer
- canonical schema
- task/model adapters
- training/inference wrappers
- evaluation + deployment analysis

---

## Terminology rules
Use the following terminology consistently:

- **VTN-style Transformer-VC** for the direct VC Phase 1 model path
- **cascaded ASR -> TTS** for the cascaded baseline
- **VALL-E-style token-based generation** for the Phase 2 token model family
- **shared representation-preparation framework** for the reusable feature/token preparation layer
- **SAP data foundation** for artifacts derived from `SlangLab-NU/sap-data-preparation`

Avoid terminology drift such as:
- calling the Phase 1 direct VC path “Parrotron”
- implying VALL-E is part of Phase 1 implementation
- calling all representation-preparation code a single universal tokenizer

---

## Documentation expectations
When editing planning docs or implementation guidance:
- keep Phase 1 and Phase 2 scope clearly separated
- explicitly note reuse opportunities from upstream repos
- document whether a task should reuse, wrap, vendor, or rewrite code
- note assumptions and unresolved questions clearly
- prefer concise, implementation-aware language over vague research prose

If a task may create ambiguity, state explicitly:
- what is in scope
- what is out of scope
- what should be reused
- what should remain model-family-specific

---

## Agent guidance
If you are a coding agent working from this repository:
- do not expand Phase 1 into VALL-E implementation work unless explicitly instructed
- do not rewrite mature upstream code just for stylistic consistency
- do not introduce hardcoded user-specific or machine-specific paths
- prefer config-driven integration
- preserve compatibility with upstream contracts where practical
- keep new abstractions minimal and justified
- explain reuse decisions in summaries when modifying plans or code
- when creating new derived data or model artifacts on the NU cluster, default to `/scratch/aa.mohan` unless a task explicitly requires a different location

When proposing implementation plans, explicitly identify:
- what code should be reused from upstream
- what code should be wrapped
- what code is missing and must be newly implemented

---

## Anti-goals
The project should avoid:
- greenfield reimplementation of everything
- creating multiple incompatible data pipelines
- mixing Phase 1 and Phase 2 paper stories
- overclaiming novelty in Phase 1
- building a repo structure that blocks future codec-token extension
- producing large amounts of low-value glue code without clear architectural purpose
