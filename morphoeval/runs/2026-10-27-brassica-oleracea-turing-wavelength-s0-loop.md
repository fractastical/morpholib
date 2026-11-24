# MorphoEval Full-Loop Telemetry — `2026-10-27-brassica-oleracea-turing-wavelength-s0-loop`

## Result (Pass 2 gates)

*Brassica oleracea* (Romanesco), Holocene 0 Ma, Turing wavelength **172 µm** (24 waves on a 4128 µm patch). Schnakenberg kinetics, D_v solved for that mode.

| Gate | Result |
|---|---|
| image_fitness | **88.1** (threshold 70) = 0.55×88.4 + 0.30×86 + 0.15×91 |
| cv_sim_similarity | **88.4** (threshold 58) |
| GIF | **12** frames, **640×480** |
| Pearson r (modal σ) | **0.9999999522** (RMSE 0.0036; threshold > 0.98) |
| Nonlinear D_v sweep r | **0.8266** (reported separately; Fourier binning; not substituted for the modal r) |
| Audit overall | Pass 1 **77.1**, Pass 2 **88.8** (threshold 85) |
| Non-circular specimen metrics | curd circularity 0.583 vs circle null 1; foreground 0.399 vs 0.5 |
| Signing | Unavailable. `~/.nanopub/profile.yml` is absent. Nanopubs were not signed. |

Pass 3 was not run. The cell-scale successor (15 µm) has σ = −14.56; the nominal 100 µm tier has σ = +0.406, below the 172 µm peak (σ = +0.551). Equal diffusion quenches the pattern (std 0.00051 vs 1.586).

The per-step wall times below are 1 s placeholders from the telemetry log, not the solver runtime.

- **Target simulation:** `delphi10000-brassica-oleracea-turing-wavelength-s0`
- **Primary paper / dataset:** Brassica oleracea Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `grok-4.7`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **12.00 s** (0.20 min) — Tier A: `6.00 s` · Tier B: `6.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 1.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 1.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 1.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 1.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 1.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 1.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 1.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 1.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 1.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 1.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 1.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 1.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **12.00** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex DOI lookups and Wikimedia specimens; slot 3137 is outside campaign_100

### Step 2 — `paper` (Paper)
- **Notes:** Extracted Schnakenberg Jacobian and sigma(k); scale 172 um, age 0 Ma

### Step 3 — `source` (Formal source)
- **Notes:** Vendored gallery Romanesco JPEG and pinned SHA-256

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3137: 24 waves, nominal tier, cell successor

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg solver, self-test, measure_morphoeval

### Step 6 — `output` (Output & compare)
- **Notes:** comparison PNG and 12-frame 640x480 GIF; CV 88.4

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 77.1 and Pass 2 overall 88.8

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100; modal r 0.9999999522

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg --conform OK for v1 and v2

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable; no ~/.nanopub/profile.yml

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md written for both audit KG directories

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 kept sweep r=0.8266 separate from modal r; gates pass, no Pass 3
