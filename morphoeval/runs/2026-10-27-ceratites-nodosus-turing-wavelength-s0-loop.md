# MorphoEval Full-Loop Telemetry — `2026-10-27-ceratites-nodosus-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-ceratites-nodosus-turing-wavelength-s0`
- **Primary paper / dataset:** Ceratites nodosus Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **348.84 s** (5.81 min) — Tier A: `6.99 s` · Tier B: `341.85 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **16,308** (input: `0`, output: `16,308`, subagent: `0`) — Tier A: `13,495` · Tier B: `2,813`
- **Total artifact tokens produced:** `16,308`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 662 | 0 | **662** | 662 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.07 | 0.00 | 0 | 2,905 | 0 | **2,905** | 2,905 | `ceratites_nodosus_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.07 | 0.00 | 0 | 64 | 0 | **64** | 64 | `ceratites_nodosus_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.07 | 0.00 | 0 | 956 | 0 | **956** | 956 | `Loop_3142_ceratites_nodosus_turing_wavelength_s0.lean`, `delphi10000-ceratites-nodosus-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 6.62 | 0.00 | 0 | 8,780 | 0 | **8,780** | 8,780 | `2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.07 | 0.00 | 0 | 128 | 0 | **128** | 128 | `ceratites_nodosus_turing_wavelength_s0_comparison.png`, `ceratites_nodosus_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 341.47 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.08 | 0.00 | 0 | 2,813 | 0 | **2,813** | 2,813 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **348.84** | **0.00** | **0** | **16,308** | **0** | **16,308** | **16,308** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Wikimedia query path; slot 3142 is outside the 100-slot campaign file
- **Artifact:** `datasets/ceratites_nodosus_turing_wavelength_s0/scout_manifest.json` (2,384 B, `662` tok, SHA-256 `d5783f94b1dd776b…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, bib, README state 26 µm and 240 Ma
- **Artifact:** `datasets/ceratites_nodosus_turing_wavelength_s0/ceratites_nodosus_turing_wavelength_s0_reference.json` (4,528 B, `1,385` tok, SHA-256 `256f3874e2f667a0…`)
- **Artifact:** `datasets/ceratites_nodosus_turing_wavelength_s0/paper_citations.bib` (1,854 B, `590` tok, SHA-256 `c4b089031d26801a…`)
- **Artifact:** `datasets/ceratites_nodosus_turing_wavelength_s0/README.md` (3,680 B, `930` tok, SHA-256 `ae965a155c4d31c3…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored gallery specimen; SHA-256 pinned
- **Artifact:** `datasets/ceratites_nodosus_turing_wavelength_s0/ceratites_nodosus_specimen.jpg` (105,042 B, `64` tok, SHA-256 `3c37b42ed1bd3fde…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3142 and math KG yaml
- **Artifact:** `morphoeval/lean/loops/Loop_3142_ceratites_nodosus_turing_wavelength_s0.lean` (2,010 B, `64` tok, SHA-256 `55704b5f028ac17c…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-ceratites-nodosus-turing-wavelength-s0.yaml` (3,394 B, `892` tok, SHA-256 `1adfa10044f6b3ff…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 1 Schnakenberg k_c model
- **Artifact:** `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` (30,607 B, `8,780` tok, SHA-256 `343880eb1f96a048…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test passed; GIF 16 frames 650x720
- **Artifact:** `docs/media/ceratites_nodosus_turing_wavelength_s0_comparison.png` (490,995 B, `64` tok, SHA-256 `1b02be065e72e6fa…`)
- **Artifact:** `docs/media/ceratites_nodosus_turing_wavelength_s0_growth.gif` (3,153,184 B, `64` tok, SHA-256 `0f28265585029436…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 audit; Pass 1 audit overall 81.1

### Step 8 — `measure` (Measure)
- **Notes:** Pass 1 and Pass 2 measurement sidecars

### Step 9 — `build` (Write as claims)
- **Notes:** Conformance OK for v1 and v2

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no ~/.nanopub/profile.yml. Not faked.; Unsigned. No trusty URI invented.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md written for v1 and v2
- **Artifact:** `morphoeval/kg/2026-10-27-ceratites-nodosus-turing-wavelength-s0-audit-v2/REPORT.md` (10,201 B, `2,813` tok, SHA-256 `545087e898e48dcf…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed R1-R4. R5 and R6 remain. Gates passed; no Pass 3.; image_fitness 89.4, cv 96.0, pearson 0.990, audit 85.8

## Result

*Ceratites nodosus*, Middle Triassic (Ladinian, 240 Ma), cellular Turing wavelength 26 µm (2.56e-05 m). Pass 2 audit overall **85.8**. Pass 1 was 81.1.

Gates after Pass 2, all met, so there was no Pass 3:

- image_fitness 89.4 (>= 70)
- cv_sim_similarity 96.0 (>= 58)
- GIF 16 frames at 650×720 (>= 6 frames, >= 340 px)
- 13 non-circular specimen metrics
- Pearson r 0.990 (> 0.98)
- audit overall 85.8 (>= 85)

The continuous-symbol unstable band is r = 0.976 (second-difference truncation). The discrete Jacobian match on that band is r = 1.000. Signing did not run: `~/.nanopub/profile.yml` is absent, and no trusty URI was invented. Nanopubs in `morphoeval/kg/` are unsigned. Open items that stay open are a single shell photograph, which cannot resolve 26 µm pigment, and the lack of a second assessor.
