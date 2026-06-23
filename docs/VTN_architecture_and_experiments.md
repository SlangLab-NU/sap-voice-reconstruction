# VTN — Voice Transformer Network for Dysarthric Voice Reconstruction

Direct **mel → mel** voice conversion: encode an *atypical* (dysarthric) source mel, decode a
*clean target* mel (StyleTTS2-synthesized ground truth), vocode to audio. The research problem is
**free-running (autoregressive inference) quality**: teacher-forced reconstruction is crystal
clear, but feeding the decoder its own predictions degrades the output. The cascade ASR→TTS system
is the *comparison baseline*; the VTN is the contribution.

> Repo: `sap-voice-reconstruction`, branch `vtn-tacotron2-lsa-decoder` (nuvan). Trained on the SAP
> TRAIN split, validated on VAL; 2× A4500 DDP via SLURM.

---

## Architecture (runs 6–10: Tacotron2-LSA decoder — current)

```mermaid
flowchart TD
    SRC["Source mel — atypical speaker<br/>[B, T_src, 80]"]

    subgraph ENCODER["Transformer Encoder"]
        EP["Linear 80→256  +  sine pos-emb"]
        EL["4× Transformer encoder layers<br/>d_model=256, heads=4, ff=1024, pre-norm"]
        EP --> EL
    end

    SRC --> EP
    EL --> MEM["memory [B, T_src, 256]"]

    subgraph DECODER["Tacotron2 decoder — recurrent, autoregressive at reduced rate r"]
        PREV["prev mel frame<br/>TF: ground-truth | free: own prediction"]
        PN["Prenet (256,256), dropout 0.5<br/>(stays ON at inference)"]
        AR["Attention LSTMCell (1024)"]
        LSA["Location-Sensitive Attention<br/>loc filters 32, kernel 31, attn dim 128"]
        CTX["context vector"]
        DR["Decoder LSTMCell (1024)"]
        MP["mel_proj → r×80"]
        SP["stop_proj → r  (BCE stop token)"]
        PREV --> PN --> AR --> LSA --> CTX --> DR
        AR --> DR
        DR --> MP
        DR --> SP
    end

    MEM --> LSA
    MP --> MB["mel_before"]
    MB --> POST["Postnet — 512ch × 5 conv"]
    POST --> MA["mel_after = mel_before + residual<br/>[B, T_tgt, 80]"]
    MA --> VOC["Vocoder<br/>HiFi-GAN UNIVERSAL_V1 (hifigan_lj profile) | Griffin-Lim"]
    VOC --> WAV["reconstructed waveform"]

    MA -. "free-running feedback (autoregressive)" .-> PREV
    SP -. "stop > 0.5 → end decode" .-> MA

    DR -. "decoder hidden states" .-> CTC["CTC phoneme head<br/>(runs 9–10) — alignment-free content loss"]
    MEM -. "cross-attn" .-> AUX["Aux ASR decoder<br/>(runs 5–10) — content supervision"]
```

### Earlier architecture (runs 1–5: Transformer decoder)

Same encoder, but the decoder was a content-only cross-attention Transformer instead of the
recurrent LSA stack:

```mermaid
flowchart TD
    SRC["Source mel — atypical<br/>[B, T_src, 80]"]

    subgraph ENCODER["Transformer Encoder (shared across all runs)"]
        EP["Linear 80→256  +  sine pos-emb"]
        EL["4× Transformer encoder layers<br/>d_model=256, heads=4, ff=1024"]
        EP --> EL
    end

    SRC --> EP
    EL --> MEM["memory [B, T_src, 256]"]

    subgraph TDEC["Transformer decoder — content-only cross-attention (runs 1–5)"]
        PREV["prev mel frame<br/>TF: ground-truth | free: own prediction"]
        PN["Prenet (256,256), dropout 0.5"]
        DP["Linear prenet→256  +  sine pos-emb"]
        DL["4× Transformer decoder layers<br/>masked self-attn + cross-attn over memory"]
        MO["mel_out → r×80"]
        SO["stop_out → r"]
        PREV --> PN --> DP --> DL
        DL --> MO
        DL --> SO
    end

    MEM --> DL
    MO --> MB["mel_before"]
    MB --> POST["Postnet — 512ch × 5"]
    POST --> MA["mel_after [B, T_tgt, 80]"]
    MA --> VOC["Vocoder → waveform"]
    MA -. "free-running feedback (autoregressive)" .-> PREV
    MEM -. "cross-attn (run5 only)" .-> AUX["Aux ASR decoder (run5)"]
```

**Why it was abandoned at run6:** content-only cross-attention has no location/monotonicity bias,
so in free-running it *parks* on a fixed source region and the output collapses. That motivated the
switch to the **location-sensitive (LSA) recurrent decoder** (the diagram above), whose attention
convolves the cumulative alignment to bias monotonic advancement. Lineage within this variant:
runs 1–4 had **no** aux ASR head (run5 added it); reduction factor was **r=2** (runs 1–3) then
**r=1** (runs 4–5).

### Training losses
```mermaid
flowchart LR
    L["total loss"] --- MSE["mel reconstruction (mel_before & mel_after)"]
    L --- BCE["stop BCE (pos_weight 5)"]
    L --- GA["guided attention (diagonal prior)"]
    L --- ASRL["aux ASR CE (runs 5+)"]
    L --- CTCL["CTC phoneme (runs 9+, on dec hidden states)"]
```
- **Scheduled sampling (runs 8+):** `ss_prob` ramps up over training (warmup → linear ramp), so
  the decoder is increasingly fed its *own* predictions during teacher-forced training — meant to
  reduce the train/inference (exposure-bias) mismatch.
- **Reduction factor r:** decoder emits `r` mel frames per step (r=2 → shorter autoregressive
  rollout; r=1 → frame-by-frame).

---

## Experiments (run1 → run10)

`free_dtw` = length-normalized DTW mel-L1 between **free-running** output and target (lower =
closer). `tf_dtw` is the teacher-forced equivalent (the "clear" ceiling, ~16–21). The free-running
metric was only added at run8, when exposure bias became the focus.

| Run | Decoder | r | Aux ASR | SS | CTC | final free_dtw | Outcome |
|-----|---------|---|:---:|:---:|:---:|:---:|---------|
| **run1** | Transformer | 2 | – | – | – | (not logged) | initial mel→mel VC baseline |
| **run2** | Transformer | 2 | – | – | – | (n/l) | repeat / stabilize |
| **run3** | Transformer | 2 | – | – | – | (n/l) | batch 6 |
| **run4** | Transformer | 1 | – | – | – | (n/l) | frame-by-frame (r=1) |
| **run5** | Transformer | 1 | ✓ | – | – | (n/l) | + auxiliary ASR head (content supervision) |
| **run6** | **Tacotron2-LSA** | 1 | ✓ | – | – | (n/l) | switch to recurrent location-sensitive decoder |
| **run7** | Tacotron2-LSA | 1 | ✓ | – | – | (n/l) | listening baseline (`infer_run7_best`) |
| **run8** | Tacotron2-LSA | 1 | ✓ | ✓ | – | **44.8** | scheduled sampling → **FAILED**: flat free_dtw + destabilized length |
| **run9** | Tacotron2-LSA | 1 | ✓ | ✓ | ✓ | **41.2** | + CTC head → rescued length (hit_max 0), gap **not** closed |
| **run10** | Tacotron2-LSA | **2** | ✓ | ✓ | ✓ | **35.3** | r=2 + late SS ramp → **best so far; intelligible via HiFi-GAN** |

```mermaid
flowchart LR
    R1["run1–3<br/>Transformer dec, r=2"] --> R4["run4<br/>r=1"]
    R4 --> R5["run5<br/>+ aux ASR"]
    R5 --> R6["run6–7<br/>→ Tacotron2-LSA decoder"]
    R6 --> R8["run8<br/>+ scheduled sampling<br/>❌ flat + length blows up"]
    R8 --> R9["run9<br/>+ CTC head<br/>length fixed, gap stays"]
    R9 --> R10["run10<br/>r=2 + CTC + SS<br/>✅ free_dtw 35, intelligible"]
    R10 --> NEXT["next: attention-anchored stop<br/>+ mel de-noise"]
```

---

## run10 diagnosis (attention + mel dump, step 40000)

Rendered the standard set through HiFi-GAN UNIVERSAL_V1; **words are recoverable** — the best
result yet. Two defects remain, and the attention/mel diagnostic pinned each to a mechanism:

```mermaid
flowchart TD
    OBS["run10 free-running output:<br/>intelligible BUT (a) white-noise floor, (b) some utts run away into noise"]
    OBS --> Q{"attention diagnostic<br/>(free vs teacher-forced)"}
    Q --> A["Alignment is FINE<br/>~monotonic, reaches source end"]
    A --> A1["⟹ NOT an attention-collapse problem<br/>(robust-attention swap ruled OUT)"]
    Q --> B["Runaways: alignment reaches end,<br/>then PARKS — BCE stop never fires"]
    B --> B1["FIX: attention-anchored stop<br/>(end decode when attn hits last source frame)"]
    Q --> C["Free mel is broadband-noisy vs clean TF mel<br/>even when alignment works"]
    C --> C1["FIX: mel de-noise / anti-oversmoothing<br/>(adversarial or refinement postnet)"]
```

**Next steps (priority):**
1. **Attention-anchored stop** — inference-only change to `Tacotron2Decoder.infer()`; kills the
   runaway white-noise. Quick win.
2. **Mel noise-floor** — the real research lever: adversarial/anti-oversmoothing refinement of the
   free-running mels (the broadband hiss the vocoder faithfully renders).
3. Deprioritized: robust/monotonic attention (DCA/forward) and non-AR decoding — the diagnostic
   shows alignment is *not* the bottleneck.
