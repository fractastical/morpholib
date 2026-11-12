# MorphoEval Full-Loop Telemetry — `2026-10-27-gasterosteus-aculeatus-marine-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0`
- **Primary paper / dataset:** Gasterosteus aculeatus_marine Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-run`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **138.31 s** (2.31 min) — Tier A: `131.37 s` · Tier B: `6.95 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **795** (input: `0`, output: `795`, subagent: `0`) — Tier A: `795` · Tier B: `0`
- **Total artifact tokens produced:** `795`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 125.58 | 0.00 | 0 | 795 | 0 | **795** | 795 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 1.15 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 1.16 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 1.16 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 1.16 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 1.16 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 1.16 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 1.16 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 1.16 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 1.17 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 1.17 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 1.14 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **138.31** | **0.00** | **0** | **795** | **0** | **795** | **795** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Slot 3232 is outside campaign_100_slots.yaml; OpenAlex+Wikimedia query used. No s0 path collision. Vendored Holger Krisp CC BY 4.0 G. aculeatus photograph.
- **Artifact:** `datasets/gasterosteus_aculeatus_marine_turing_wavelength_s0/scout_manifest.json` (2,897 B, `795` tok, SHA-256 `eb4dfa3bf549ca14…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON states 52 µm (5.2e-05 m) and 0.015 Ma. Equations, Jacobian, k_c, and the chronoseries map are in the reference JSON and bib.

### Step 3 — `source` (Formal source)
- **Notes:** Vendored Holger Krisp CC BY 4.0 Gasterosteus aculeatus photograph, SHA-256 0c9e2c6dfe107e95adac946fca94b5887561d5229ff7e598385f0664b6cb456c. Gallery spiral was not used.

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3232: h=15500000, k_c^2=0.775, equal-diffusion h<0, 52<78 and 52<104, 887/52=17. No sorry.

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg integrator, sweep, ablation, segmentation, gain fit, --self-test.

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test passed. GIF 12 frames 960x548. Comparison PNG written.

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 72.4. Bare-field radial Pearson 0.262.

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100/100. 12 measurements, values differ from references.

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg --conform OK for audit and audit-v2. Rubric reintroduction allowed because 00-rubric.trig already exists in earlier audits.

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml does not exist. Nanopubs were not signed and trusty URIs were not minted. This is not a signed publication.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for both kg directories. Pass 2 overall 87.3.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 radial Pearson 0.988, RMSE 0.154, KS 0.0395, CV 96.0, image_fitness 91.2, ablation ratio 6.02, sweep monotone 41.6/52/78/104 µm, bar spacing 641 µm != 52. Gates pass. No Pass 3.
