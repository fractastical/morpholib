# MorphoEval Full-Loop Telemetry — `2026-10-27-cornus-controversa-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-cornus-controversa-turing-wavelength-s0`
- **Primary paper / dataset:** Cornus controversa Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` (single assessor; no second assessor was run)
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1495.00 s** (24.92 min) — Tier A: `1010.00 s` · Tier B: `485.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 180.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 240.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 60.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 90.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 400.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 200.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 15.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 180.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1495.00** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex and Crossref. Slot 3155 is not in campaign_100_slots.yaml. No slug collision.; OpenAlex and Crossref. Slot 3155 is not in campaign_100_slots.yaml. No slug collision.

### Step 2 — `paper` (Paper)
- **Notes:** Extracted Schnakenberg/Turing equations, 1.939 mm, 0 Ma, claim boundary.; Extracted Schnakenberg/Turing equations, 1.939 mm, 0 Ma, claim boundary.

### Step 3 — `source` (Formal source)
- **Notes:** Vendored the public-domain Meise crown from the species gallery and pinned SHA-256.; Vendored the public-domain Meise crown from the species gallery and pinned SHA-256.

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3155 theorems and math-KG yaml.; Lean Loop_3155 theorems and math-KG yaml.

### Step 5 — `script` (Script)
- **Notes:** Pass 1 script with dispersion fit, sweep, ablation, specimen segmentation.; Pass 1 script with dispersion fit, sweep, ablation, specimen segmentation.

### Step 6 — `output` (Output & compare)
- **Notes:** PNG and 12-frame 480x640 GIF. CV similarity 96.0.; PNG and 12-frame 480x640 GIF. CV similarity 96.0.

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 audit overall 82.7. R1-R3 open.; Pass 1 audit overall 82.7. R1-R3 open.

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs.py on the Pass 1 audit. quick_gate 100.; measure_outputs.py on the Pass 1 audit. quick_gate 100.

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg.py --conform. Loci not on origin/main; --no-verify-loci.; build_morphoeval_kg.py --conform. Loci not on origin/main; --no-verify-loci.

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing not performed: no ~/.nanopub/profile.yml and nanopub is not installed.; Signing not performed: no ~/.nanopub/profile.yml and nanopub is not installed.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for the Pass 1 KG.; REPORT.md present for the Pass 1 KG.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2: 16-wavelength remeasure keeps 2.216 mm; template reported separately. Audit v2 overall 85.1. Gates pass.; Pass 2: 16-wavelength remeasure keeps 2.216 mm; template reported separately. Audit v2 overall 85.1. Gates pass.

## Result

*Cornus controversa*, Holocene, 0 Ma, chronoseries 1/1. Linear Turing wavelength is 1.939 mm (0.001939 m). Pearson r of early-time mode growth against analytic σ(k) is 0.9999998 (RMSE 2.13e-4). The unbiased saturated wavelength is 2.216 mm on both an 8-wavelength patch and a 16-wavelength patch, so the shift is not a coarse Fourier bin. A midrib template of amplitude 0.004 entrains the field back to 1.939 mm and is reported separately. The crown photograph has no scale bar.

Pass 1 audit overall 82.7. Pass 2 audit overall 85.1. Image fitness 88.2 (CV similarity 96.0, 12 frames, 480×640). Gates after Pass 2 all pass, so there is no Pass 3.

Signing was not performed. `~/.nanopub/profile.yml` is absent and the `nanopub` package is not installed. No signature was invented. Nanopubs in `morphoeval/kg/2026-10-27-cornus-controversa-turing-wavelength-s0-audit-v2/` passed the offline conformance check and are unsigned.

R4 (no second assessor) and R5 (no signature) stay open.
