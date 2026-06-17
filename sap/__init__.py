"""SAP voice reconstruction: consolidated implementation package.

Layered design (see the repository README and the planning repo
`planning-doc-voice-reconstruction`):

- ``sap.data.foundation`` — wraps ``SlangLab-NU/sap-data-preparation`` outputs
  (manifests, Lhotse recipes, synthetic pairs) as the authoritative SAP data
  foundation. Do not reimplement extraction/synthesis here.
- ``sap.data.adapters`` — model-facing adapters (VC, cascaded, VALL-E) built on
  top of a shared representation-preparation layer and canonical schema.
- ``sap.models`` — model families: ``vc/vtn`` (VTN-style Transformer-VC),
  ``cascaded`` (ASR -> TTS), ``valle`` (Phase 2).
- ``sap.eval`` — shared evaluation harness (intelligibility, speaker
  preservation, acoustic distortion, latency/throughput).
- ``sap.utils`` — config, IO, logging, checkpoint helpers.

Phase 1 scope: VTN-style direct VC and cascaded ASR -> TTS only. VALL-E-style
token generation is Phase 2; its package is scaffolded but intentionally empty.
"""

__version__ = "0.0.0"
