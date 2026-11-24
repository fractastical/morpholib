# MorphoEval Full-Loop Telemetry — `2026-10-27-aulacopleura-koninckii-holaspis22-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0`
- **Primary paper / dataset:** Aulacopleura koninckii_holaspis22 Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.08 s** (0.02 min) — Tier A: `0.60 s` · Tier B: `0.48 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **757** (input: `0`, output: `757`, subagent: `0`) — Tier A: `757` · Tier B: `0`
- **Total artifact tokens produced:** `757`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 757 | 0 | **757** | 757 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.18 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1.08** | **0.00** | **0** | **757** | **0** | **757** | **757** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Slot 3128 is outside campaign_100_slots.yaml; OpenAlex/Wikimedia query used. No collision with the s0 slug.
- **Artifact:** `datasets/aulacopleura_koninckii_holaspis22_turing_wavelength_s0/scout_manifest.json` (2,713 B, `757` tok, SHA-256 `1de0f2b99c20097e…`)

### Step 2 — `paper` (Paper)
- **Notes:** Extracted Schnakenberg Jacobian, k_c, periodic BC, and Hughes 2017 Dyar ratios 1.115/1.103 into the reference JSON.

### Step 3 — `source` (Formal source)
- **Notes:** Pinned gallery holaspis22 JPEG sha256 1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a. Wikimedia CC images scouted but not substituted.

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3128: four theorems, zero sorry. t22 entry is four molts after t18; 1.115^4 exceeds 22/18.

### Step 5 — `script` (Script)
- **Notes:** Pass 1 dispersion self-test r=0.99963. Pass 2 added diffusivity sweep, quadratic ablation, and specimen KS.

### Step 6 — `output` (Output & compare)
- **Notes:** GIF 14 frames 611x720 cv_sim_similarity=96.0. Comparison PNG is the six-panel pass-2 figure.

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 79.9. Pass 2 overall 87.2 after closing R1-R4. Single builder-assessor.

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs.py: 6 metrics on pass 1, 9 on pass 2, quick_gate 100/100. Ring count 23.8889 vs reference 22.

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg.py --conform OK for both audits. Shared rubric not re-emitted (--rubric-from bos-taurus v7). Loci note: tokenized origin URL did not match the checkpoint repository string.

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable. ~/.nanopub/profile.yml missing and the nanopub package is not installed. Nanopubs left unsigned. No signature was fabricated.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for pass 1 (overall 79.9) and pass 2 (overall 87.2).

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed R1-R4. Gates passed (image_fitness 91.5, cv 96.0, r 0.99963, overall 87.2). Pass 3 not required.
