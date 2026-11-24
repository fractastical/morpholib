# MorphoEval Full-Loop Telemetry — `2026-10-27-coturnix-japonica-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-coturnix-japonica-turing-wavelength-s0`
- **Primary paper / dataset:** Coturnix japonica Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `unavailable`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **0.88 s** (0.01 min) — Tier A: `0.44 s` · Tier B: `0.44 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **0.88** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex and Crossref for Haupaix 2018, Prum & Williamson 2002, Schnakenberg 1979, Turing 1952. Slot 3157 is outside campaign_100_slots.yaml. No slug collision.

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, BibTeX, and README state 22.42 µm and Holocene 0 Ma.

### Step 3 — `source` (Formal source)
- **Notes:** Vendored CC BY-SA 2.0 Yehliu Coturnix japonica JPEG; SHA-256 pinned.

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3157: five theorems, 0 sorry, plus math_kg yaml.

### Step 5 — `script` (Script)
- **Notes:** Pass 1 script, then Pass 2 flank ROI, dual wavelength annotation, dispersion-curve Pearson.

### Step 6 — `output` (Output & compare)
- **Notes:** self-test passed; GIF 12 frames 960x540; Pearson r 0.99966; CV 96.0.

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 audit 86.2 and Pass 2 audit 87.9, supersedes chain recorded.

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs.py quick_gate 100/100 on both audits.

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg.py --conform OK for v1 and v2. Unsigned.

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing skipped: ~/.nanopub/profile.yml is absent. No trusty URI was fabricated.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for audit and audit-v2.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** R1 tested and kept as onset-tracking moment; R2 and R3 fixed; R4 left open.

## Pass 2 gates

- image_fitness = 91.8 (threshold 70)
- cv_sim_similarity = 96.0 (threshold 58)
- GIF frames = 12, resolution = 960×540
- Pearson r of the Du-sweep growth wavelength vs the onset law = 0.99966
- Pass 2 audit overall = 87.9
- Non-circular specimen metrics: plumage melanin fraction 0.340 vs simulated midpoint fraction 0.089; flank fraction 0.364 vs full-plumage 0.340; saturated stripe wavelength 33.65 µm vs linear 22.42 µm
- Signing: not done. `~/.nanopub/profile.yml` is absent, and no signature was invented.
- Nanopub conformance: Pass 1 and Pass 2 `--conform` both OK. Pass 1 has no trusty URI, so Pass 2 could not carry `npx:supersedes`; the audit yaml records `supersedes_audit_id`.
