# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-affinis-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-affinis-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila affinis Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-run`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.01 s** (0.02 min) — Tier A: `0.51 s` · Tier B: `0.50 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1.01** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Crossref DOIs and Wikimedia Commons File:Daff-fem.tif; slot 3174 is outside campaign_100.; Crossref DOIs and Wikimedia Commons File:Daff-fem.tif; slot 3174 is outside campaign_100.

### Step 2 — `paper` (Paper)
- **Notes:** Extracted Schnakenberg kinetics, k_c, BCs, and the 2.444 µm / 0 Ma chronoseries.; Extracted Schnakenberg kinetics, k_c, BCs, and the 2.444 µm / 0 Ma chronoseries.

### Step 3 — `source` (Formal source)
- **Notes:** Vendored CC BY-SA 4.0 female habitus and pinned SHA-256.; Vendored CC BY-SA 4.0 female habitus and pinned SHA-256.

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean MathOfBiology.Loop_3174, five theorems, zero sorry, plus math_kg yaml.; Lean MathOfBiology.Loop_3174, five theorems, zero sorry, plus math_kg yaml.

### Step 5 — `script` (Script)
- **Notes:** models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py with --self-test.; models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py with --self-test.

### Step 6 — `output` (Output & compare)
- **Notes:** comparison png and 12-frame 960x720 growth gif; CV similarity 96.0.; comparison png and 12-frame 960x720 growth gif; CV similarity 96.0.

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 82.0; Pass 2 overall 86.2 after Du calibration.; Pass 1 overall 82.0; Pass 2 overall 86.2 after Du calibration.

### Step 8 — `measure` (Measure)
- **Notes:** 14 non-circular measurements; quick_gate 100; Pearson r 0.9991.; 14 non-circular measurements; quick_gate 100; Pearson r 0.9991.

### Step 9 — `build` (Write as claims)
- **Notes:** Both KG directories conformance-OK. Unsigned.; Both KG directories conformance-OK. Unsigned.

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml absent. Not faked.; Signing unavailable: ~/.nanopub/profile.yml absent. Not faked.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for audit and audit-v2.; REPORT.md present for audit and audit-v2.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed the 2.619 µm FFT gap. image_fitness 90.3. Gates pass. No pass 3.; Pass 2 closed the 2.619 µm FFT gap. image_fitness 90.3. Gates pass. No pass 3.

## Result (slot 3174)

*Drosophila affinis*, Holocene, 0 Ma, detail tier 2.4 µm (2.444e-6 m). Schnakenberg reaction-diffusion with Dv/Du = 10. Pass 1 left Du on the determinant condition, so the nonlinear FFT selected 2.619 µm (audit overall 82.0). Pass 2 bisects Du until the fastest mode is 2.443 µm; the FFT peak is then 2.432 µm. Pearson r of continuum versus discrete growth rates is 0.9991. The equal-diffusivity ablation does not grow.

The habitus is Wikimedia Commons File:Daff-fem.tif (MA Han, CC BY-SA 4.0). Computer-vision similarity of the 12-frame 960×720 GIF against that photograph is 96.0. Image fitness is 0.55×96.0 + 0.30×82 + 0.15×86 = 90.3.

Pass 2 gates: image_fitness 90.3 ≥ 70, cv_sim_similarity 96.0 ≥ 58, 12 frames ≥ 6, 720 px ≥ 340, non-circular specimen metrics present, Pearson r 0.9991 > 0.98, audit overall 86.2 ≥ 85. Pass 3 was not required.

Nanopubs were conformance-checked and left unsigned. `~/.nanopub/profile.yml` is not on this machine. The creator IRI on the unsigned nanopubs is the repository URI, not a personal ORCID.

The wall-clock column above is only the time spent closing the telemetry records, not the simulation runtime.
