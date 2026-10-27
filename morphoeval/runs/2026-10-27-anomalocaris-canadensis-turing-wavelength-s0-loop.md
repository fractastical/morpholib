# MorphoEval Full-Loop Telemetry — `2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-anomalocaris-canadensis-turing-wavelength-s0`
- **Primary paper / dataset:** Anomalocaris canadensis Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **664.82 s** (11.08 min) — Tier A: `664.25 s` · Tier B: `0.57 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **30,972** (input: `0`, output: `30,972`, subagent: `0`) — Tier A: `14,026` · Tier B: `16,946`
- **Total artifact tokens produced:** `30,972`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 663.79 | 0.00 | 0 | 1,426 | 0 | **1,426** | 1,426 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 3,375 | 0 | **3,375** | 3,375 | `anomalocaris_canadensis_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 64 | 0 | **64** | 64 | `anomalocaris_canadensis_TMP2023_003_0003_tyrrell.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 1,014 | 0 | **1,014** | 1,014 | `Loop_3113_anomalocaris_canadensis_turing_wavelength_s0.lean`, `delphi10000-anomalocaris-canadensis-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 8,019 | 0 | **8,019** | 8,019 | `2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `anomalocaris_canadensis_turing_wavelength_s0_comparison.png`, `anomalocaris_canadensis_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.09 | 0.00 | 0 | 4,955 | 0 | **4,955** | 4,955 | `2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-audit.yaml`, `2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.10 | 0.00 | 0 | 3,817 | 0 | **3,817** | 3,817 | `2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.09 | 0.00 | 0 | 2,612 | 0 | **2,612** | 2,612 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.10 | 0.00 | 0 | 4,772 | 0 | **4,772** | 4,772 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.10 | 0.00 | 0 | 790 | 0 | **790** | 790 | `2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **664.82** | **0.00** | **0** | **30,972** | **0** | **30,972** | **30,972** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Artifact:** `datasets/anomalocaris_canadensis_turing_wavelength_s0/scout_manifest.json` (5,503 B, `1,426` tok, SHA-256 `f5667708ff38ccfd…`)

### Step 2 — `paper` (Paper)
- **Artifact:** `datasets/anomalocaris_canadensis_turing_wavelength_s0/anomalocaris_canadensis_turing_wavelength_s0_reference.json` (6,030 B, `1,683` tok, SHA-256 `a973d7ad2eb9d962…`)
- **Artifact:** `datasets/anomalocaris_canadensis_turing_wavelength_s0/paper_citations.bib` (2,519 B, `794` tok, SHA-256 `57881589cd1d1e25…`)
- **Artifact:** `datasets/anomalocaris_canadensis_turing_wavelength_s0/README.md` (3,502 B, `898` tok, SHA-256 `408d6c77fa9c97c0…`)

### Step 3 — `source` (Formal source)
- **Artifact:** `datasets/anomalocaris_canadensis_turing_wavelength_s0/anomalocaris_canadensis_TMP2023_003_0003_tyrrell.jpg` (6,921,317 B, `64` tok, SHA-256 `0dcc1da801bddc9b…`)

### Step 4 — `rules` (Pin the rules)
- **Artifact:** `morphoeval/lean/loops/Loop_3113_anomalocaris_canadensis_turing_wavelength_s0.lean` (2,027 B, `64` tok, SHA-256 `d5cdef249b15cd46…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-anomalocaris-canadensis-turing-wavelength-s0.yaml` (3,479 B, `950` tok, SHA-256 `f5c158df2d54b7af…`)

### Step 5 — `script` (Script)
- **Artifact:** `models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py` (28,140 B, `8,019` tok, SHA-256 `f8003ea1555b543c…`)

### Step 6 — `output` (Output & compare)
- **Artifact:** `docs/media/anomalocaris_canadensis_turing_wavelength_s0_comparison.png` (661,168 B, `64` tok, SHA-256 `8c788c0b684071e1…`)
- **Artifact:** `docs/media/anomalocaris_canadensis_turing_wavelength_s0_growth.gif` (3,313,375 B, `64` tok, SHA-256 `fdc6e92b6f8b2b81…`)

### Step 7 — `audit` (Dual-agent audit)
- **Artifact:** `morphoeval/audits/2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-audit.yaml` (8,562 B, `2,181` tok, SHA-256 `8642bff780ecf6fc…`)
- **Artifact:** `morphoeval/audits/2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-audit-v2.yaml` (10,651 B, `2,774` tok, SHA-256 `3ab9372a1d83133b…`)

### Step 8 — `measure` (Measure)
- **Artifact:** `morphoeval/audits/2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-audit.measurements.yaml` (6,809 B, `1,882` tok, SHA-256 `a8772e0985d39928…`)
- **Artifact:** `morphoeval/audits/2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-audit-v2.measurements.yaml` (6,991 B, `1,935` tok, SHA-256 `8eb8563fea5614cb…`)

### Step 9 — `build` (Write as claims)
- **Artifact:** `morphoeval/kg/2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-audit-v2/REPORT.md` (9,694 B, `2,612` tok, SHA-256 `ce02b253ad50a9cd…`)

### Step 11 — `graph` (Public record)
- **Artifact:** `morphoeval/kg/2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-audit/REPORT.md` (8,220 B, `2,160` tok, SHA-256 `10782f170793f55b…`)
- **Artifact:** `morphoeval/kg/2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-audit-v2/REPORT.md` (9,694 B, `2,612` tok, SHA-256 `ce02b253ad50a9cd…`)

### Step 12 — `remed` (Remediate & re-score)
- **Artifact:** `morphoeval/runs/2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-loop-image-fitness.json` (2,623 B, `790` tok, SHA-256 `51038091ad514efe…`)

## Result (slot 3113)

*Anomalocaris canadensis*, Burgess Shale, 508 Ma, Turing wavelength 2.397e-05 m (24 µm detail tier). Chronoseries 1/1. The predecessor bracket is the Emu Bay eye (~515 Ma, 2–3 cm, ≥16,000 lenses; reconstructed facet spacing ~266 µm). The successor bracket is the Fezouata giant (488–472 Ma), where body size changes and this cellular wavenumber is unmeasured.

The audit assessor is Grok 4.7, one assessor. The model names in the telemetry header are the tracker defaults, not a second scoring pass.

Pass 1 overall **74.4**. Pass 2 overall **88.3** (faithful). Pass 2 closes the diffusivity sweep, the equal-diffusion ablation, the cast segmentation, and the dispersion fit (Pearson r = 0.999989, RMSE = 0.00626, KS D = 0.071 versus a zero-growth null of 0.714).

Image fitness on the last GIF frame versus the Tyrrell cast (TMP 2023.003.0003): **91.7** = 0.55×96.0 + 0.30×84 + 0.15×91. GIF is 720×540, 12 frames.

Pass 2 gates: image_fitness 91.7 ≥ 70, cv_sim_similarity 96.0 ≥ 58, frames 12 ≥ 6, min side 540 ≥ 340, non-circular specimen metrics present, Pearson r 0.999989 > 0.98, audit overall 88.3 ≥ 85. Pass 3 was not run.

Signing was not done. `~/.nanopub/profile.yml` is absent and the `nanopub` package is not installed, so `sign_morphoeval.py` produced no trusty URIs. The nanopubs in `morphoeval/kg/` are unsigned. The shared rubric nanopub was not re-emitted; it matches the already published Bos taurus v7 rubric.

Open limits kept in the Pass 2 audit: one museum cast rather than original cuticle, kinetics not measured on this animal, and no second assessor.
