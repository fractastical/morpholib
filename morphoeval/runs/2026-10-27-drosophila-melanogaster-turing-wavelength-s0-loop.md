# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-melanogaster-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-melanogaster-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila melanogaster Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `none-this-run`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.08 s** (0.02 min) — Tier A: `0.53 s` · Tier B: `0.56 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **40,081** (input: `0`, output: `40,081`, subagent: `0`) — Tier A: `17,550` · Tier B: `22,531`
- **Total artifact tokens produced:** `40,081`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,176 | 0 | **1,176** | 1,176 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,592 | 0 | **2,592** | 2,592 | `drosophila_melanogaster_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 64 | 0 | **64** | 64 | `Drosophila_melanogaster_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,124 | 0 | **1,124** | 1,124 | `Loop_3192_drosophila_melanogaster_turing_wavelength_s0.lean`, `delphi10000-drosophila-melanogaster-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 12,466 | 0 | **12,466** | 12,466 | `2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_melanogaster_turing_wavelength_s0_comparison.png`, `drosophila_melanogaster_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 3,964 | 0 | **3,964** | 3,964 | `2026-10-27-drosophila-melanogaster-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 4,258 | 0 | **4,258** | 4,258 | `2026-10-27-drosophila-melanogaster-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-drosophila-melanogaster-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 6,516 | 0 | **6,516** | 6,516 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 950 | 0 | **950** | 950 | `2026-10-27-drosophila-melanogaster-turing-wavelength-s0-loop-image-fitness.json` |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 3,226 | 0 | **3,226** | 3,226 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 3,617 | 0 | **3,617** | 3,617 | `2026-10-27-drosophila-melanogaster-turing-wavelength-s0-audit-v2.yaml` |
| — | **All** | **TOTAL** | — | `12/12` | **1.08** | **0.00** | **0** | **40,081** | **0** | **40,081** | **40,081** | — |

## Result

*Drosophila melanogaster*, Holocene, 0 Ma, detail-scale Turing wavelength **2.736e-06 m** (2.7 µm, tier `3_subcellular_um`). Schnakenberg γ is solved so the linear critical wavelength is 2.736 µm. At that same γ the reconstructed wavelengths are 2.242 µm for the successor *D. miranda* and 3.243 µm for the predecessor *D. mauritiana*.

Pass 2 overall **86.7** (faithful). Dispersion Pearson r = 0.9999999989, RMSE = 4.62e-5. The 2D FFT wavelength is 2.799 µm. The growth GIF is 16 frames at 768×576. CV similarity of the last frame to the Karwath male is 96.0. Image fitness = 0.55×96 + 0.30×82 + 0.15×88 = **90.6**.

Gates after Pass 2, all met, so Pass 3 was not run:

- image_fitness 90.6 ≥ 70
- cv_sim_similarity 96.0 ≥ 58
- GIF frames 16 ≥ 10
- resolution 576 px ≥ 340
- non-circular specimen metrics present (foreground fraction, circularity, FFT wavelength, mask span, noise-order indicator; none equal their reference)
- Pearson r 0.9999999989 > 0.98
- audit overall 86.7 ≥ 85

Mode-seeded 1D saturation keeps miranda < melanogaster < mauritiana. Broadband-noise saturation does not. The mask quantile sweep moves foreground fraction by 0.154 while circularity stays below 0.09.

Signing was not done. `~/.nanopub/profile.yml` is absent, so the ProbKnow graphs were conformance-checked and left unsigned. Lean 4 is not installed on this VM; the five `decide` theorems contain no `sorry` and were not kernel-checked here.

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia scout; no slug collision; OpenAlex/Crossref/Wikimedia scout; no slug collision
- **Artifact:** `datasets/drosophila_melanogaster_turing_wavelength_s0/scout_manifest.json` (3,946 B, `1,176` tok, SHA-256 `2baeb400d5365508…`)

### Step 2 — `paper` (Paper)
- **Notes:** Equations, parameter table, 2.736e-06 m, 0 Ma; Equations, parameter table, 2.736e-06 m, 0 Ma
- **Artifact:** `datasets/drosophila_melanogaster_turing_wavelength_s0/drosophila_melanogaster_turing_wavelength_s0_reference.json` (3,329 B, `1,032` tok, SHA-256 `553df65dcc62824e…`)
- **Artifact:** `datasets/drosophila_melanogaster_turing_wavelength_s0/paper_citations.bib` (2,636 B, `808` tok, SHA-256 `383f4b1239516e41…`)
- **Artifact:** `datasets/drosophila_melanogaster_turing_wavelength_s0/README.md` (2,538 B, `752` tok, SHA-256 `deb3b8334442d5ee…`)

### Step 3 — `source` (Formal source)
- **Notes:** Karwath CC BY-SA 2.5 male, SHA-256 pinned; Karwath CC BY-SA 2.5 male, SHA-256 pinned
- **Artifact:** `datasets/drosophila_melanogaster_turing_wavelength_s0/Drosophila_melanogaster_specimen.jpg` (56,227 B, `64` tok, SHA-256 `2a66a100d8405cc9…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3192, five theorems, zero sorry; Lean Loop_3192, five theorems, zero sorry
- **Artifact:** `morphoeval/lean/loops/Loop_3192_drosophila_melanogaster_turing_wavelength_s0.lean` (1,771 B, `64` tok, SHA-256 `094a57fc0e0ad0ba…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-drosophila-melanogaster-turing-wavelength-s0.yaml` (3,699 B, `1,060` tok, SHA-256 `b64f2fa9790bb807…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg script with --self-test; Schnakenberg script with --self-test
- **Artifact:** `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` (43,227 B, `12,466` tok, SHA-256 `930172a82cc24cd7…`)

### Step 6 — `output` (Output & compare)
- **Notes:** comparison PNG and 16-frame growth GIF; comparison PNG and 16-frame growth GIF
- **Artifact:** `docs/media/drosophila_melanogaster_turing_wavelength_s0_comparison.png` (422,992 B, `64` tok, SHA-256 `a5198e44d8253331…`)
- **Artifact:** `docs/media/drosophila_melanogaster_turing_wavelength_s0_growth.gif` (4,812,017 B, `64` tok, SHA-256 `16db51212fc73ddf…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 audit overall 86.2; Pass 1 audit overall 86.2
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-melanogaster-turing-wavelength-s0-audit.yaml` (15,476 B, `3,964` tok, SHA-256 `dbf2a5cc171c12fc…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs fingerprints and morphometrics; measure_outputs fingerprints and morphometrics
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-melanogaster-turing-wavelength-s0-audit.measurements.yaml` (7,426 B, `2,042` tok, SHA-256 `218eeddfecdcb4f9…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-melanogaster-turing-wavelength-s0-audit-v2.measurements.yaml` (8,153 B, `2,216` tok, SHA-256 `6fec143131383c24…`)

### Step 9 — `build` (Write as claims)
- **Notes:** ProbKnow conformance OK; ProbKnow conformance OK
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-melanogaster-turing-wavelength-s0-audit/REPORT.md` (12,324 B, `3,290` tok, SHA-256 `d338426ffa749429…`)
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-melanogaster-turing-wavelength-s0-audit-v2/REPORT.md` (11,760 B, `3,226` tok, SHA-256 `defdd748ca6df384…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no ~/.nanopub/profile.yml; Signing unavailable: no ~/.nanopub/profile.yml
- **Artifact:** `morphoeval/runs/2026-10-27-drosophila-melanogaster-turing-wavelength-s0-loop-image-fitness.json` (3,205 B, `950` tok, SHA-256 `c07ae4ff7ae52676…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for pass 1 and pass 2; REPORT.md present for pass 1 and pass 2
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-melanogaster-turing-wavelength-s0-audit-v2/REPORT.md` (11,760 B, `3,226` tok, SHA-256 `defdd748ca6df384…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 mask sweep and saturated 1D chronospecies check; overall 86.7; gates pass; Pass 2 mask sweep and saturated 1D chronospecies check; overall 86.7; gates pass
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-melanogaster-turing-wavelength-s0-audit-v2.yaml` (13,831 B, `3,617` tok, SHA-256 `bf13d2b967cc8427…`)
