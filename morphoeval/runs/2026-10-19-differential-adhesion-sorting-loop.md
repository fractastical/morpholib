# MorphoEval Full-Loop Telemetry — `2026-10-19-differential-adhesion-sorting-loop`

- **Target simulation:** `steinberg-differential-adhesion-sorting`
- **Primary paper / dataset:** Steinberg (1963) Science / Graner & Glazier (1992) PRL
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `317bc3c`
- **Completed steps:** `11 / 12`
- **Total wall-clock time:** **7.58 s** (0.13 min) — Tier A: `2.82 s` · Tier B: `4.76 s`
- **Total CPU time:** `6.27 s` user + `0.37 s` sys
- **Total LLM tokens:** **3,363** (input: `683`, output: `2,680`, subagent: `0`) — Tier A: `371` · Tier B: `2,992`
- **Total artifact tokens produced:** `52,438`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | scout-agent (`gemini-4-argon`) | `completed` | 1.00 | 0.41 | 31 | 67 | 0 | **98** | 734 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.02 | 0.01 | 48 | 1 | 0 | **49** | 3,823 | `differential_adhesion_sorting_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.02 | 0.01 | 88 | 1 | 0 | **89** | 2,468 | `cell_sorting_specimen.jpg`, `differential_adhesion_sorting_reference.json` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.42 | 0.20 | 16 | 1 | 0 | **17** | 1,214 | `Loop_041_differential_adhesion_sorting.lean`, `steinberg-differential-adhesion-sorting.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.38 | 0.36 | 24 | 15 | 0 | **39** | 2,930 | `1964-steinberg-differential-adhesion-sorting.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.98 | 0.96 | 20 | 59 | 0 | **79** | 128 | `differential_adhesion_sorting_comparison.png`, `differential_adhesion_sorting_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`gemini-4-argon`) | `completed` | 0.11 | 0.09 | 37 | 25 | 0 | **62** | 2,643 | `2026-10-19-differential-adhesion-sorting-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`gemini-4-argon`) | `completed` | 0.92 | 0.90 | 34 | 159 | 0 | **193** | 596 | `2026-10-19-differential-adhesion-sorting-audit.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`gemini-4-argon`) | `completed` | 0.35 | 0.32 | 74 | 446 | 0 | **520** | 13,521 | `REPORT.md`, `morphoeval_2026-10-19-differential-adhesion-sorting-audit.ttl` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`gemini-4-argon`) | `failed` | 0.58 | 0.55 | 66 | 544 | 0 | **610** | 679 | `published.json` |
| 11 | B | **graph** (Public record) | tool (`gemini-4-argon`) | `completed` | 0.02 | 0.01 | 55 | 1 | 0 | **56** | 2,900 | `REPORT.md`, `published.json` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`gemini-4-argon`) | `completed` | 2.78 | 2.82 | 190 | 1,361 | 0 | **1,551** | 20,802 | `1964-steinberg-differential-adhesion-sorting.py`, `differential_adhesion_sorting_comparison.png`, `differential_adhesion_sorting_growth.gif`, +4 more |
| — | **All** | **TOTAL** | — | `11/12` | **7.58** | **6.64** | **683** | **2,680** | **0** | **3,363** | **52,438** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Executed OpenAlex + Wikimedia Commons literature & specimen scout for Slot 041 (Steinberg 1963 / Graner & Glazier 1992 differential adhesion cell sorting)
- **Artifact:** `datasets/differential_adhesion_sorting/scout_manifest.json` (2,596 B, `734` tok, SHA-256 `3d4f91c8032c61e7…`)

### Step 2 — `paper` (Paper)
- **Notes:** Extracted Steinberg (1963, 1970) DAH interfacial tension inequalities, Graner & Glazier (1992) Potts Hamiltonian parameters, and Foty et al. (1996) chick embryo surface tension hierarchy
- **Artifact:** `datasets/differential_adhesion_sorting/differential_adhesion_sorting_reference.json` (8,717 B, `2,404` tok, SHA-256 `88461b7809c538d7…`)
- **Artifact:** `datasets/differential_adhesion_sorting/paper_citations.bib` (1,787 B, `547` tok, SHA-256 `0cc7965520ea6ce8…`)
- **Artifact:** `datasets/differential_adhesion_sorting/README.md` (2,756 B, `872` tok, SHA-256 `5790c470276d2181…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored confocal embryoid body multicellular spheroid sorting specimen image from Wikimedia Commons (CC BY 3.0) and locked SHA-256 digest
- **Artifact:** `datasets/differential_adhesion_sorting/cell_sorting_specimen.jpg` (30,788 B, `64` tok, SHA-256 `9b31f7b855d90faa…`)
- **Artifact:** `datasets/differential_adhesion_sorting/differential_adhesion_sorting_reference.json` (8,717 B, `2,404` tok, SHA-256 `88461b7809c538d7…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Verified 4 machine-checked Lean 4 theorems (0 sorrys) in Loop_041_differential_adhesion_sorting.lean and registered math_kg_loops spec
- **Artifact:** `morphoeval/lean/loops/Loop_041_differential_adhesion_sorting.lean` (5,343 B, `64` tok, SHA-256 `fe60796dfb3076c9…`)
- **Artifact:** `morphoeval/math_kg_loops/steinberg-differential-adhesion-sorting.yaml` (4,308 B, `1,150` tok, SHA-256 `3571eec6bb66c84a…`)

### Step 5 — `script` (Script)
- **Notes:** Implemented Pass 1 (v1) Cellular Potts sorting baseline script with --self-test and measure_morphoeval()
- **Artifact:** `1964-steinberg-differential-adhesion-sorting.py` (10,712 B, `2,930` tok, SHA-256 `433b5d2794b9dd47…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Rendered Pass 1 static comparison figure and growth animation GIF
- **Artifact:** `docs/media/differential_adhesion_sorting_comparison.png` (84,361 B, `64` tok, SHA-256 `874f4784a13f5781…`)
- **Artifact:** `docs/media/differential_adhesion_sorting_growth.gif` (26,063 B, `64` tok, SHA-256 `920f45a19b8fc1dd…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Completed Pass 1 (v1) 7-dimension MorphoEval audit (57.3/100) with line-pinned evidence and remediation plan
- **Artifact:** `morphoeval/audits/2026-10-19-differential-adhesion-sorting-audit.yaml` (10,846 B, `2,643` tok, SHA-256 `88b36acf2ff03a32…`)

### Step 8 — `measure` (Measure)
- **Notes:** Fingerprinted Pass 1 outputs and computed sidecar morphometrics
- **Artifact:** `morphoeval/audits/2026-10-19-differential-adhesion-sorting-audit.measurements.yaml` (2,103 B, `596` tok, SHA-256 `67dbd09120993b77…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Built ProbKnow-conformant TriG nanopublications and REPORT.md for Pass 1; Built ProbKnow-conformant TriG nanopublications and REPORT.md for Pass 1
- **Artifact:** `morphoeval/kg/2026-10-19-differential-adhesion-sorting-audit/REPORT.md` (8,607 B, `2,221` tok, SHA-256 `b8f3de3268b5a9a8…`)
- **Artifact:** `morphoeval/kg/2026-10-19-differential-adhesion-sorting-audit/morphoeval_2026-10-19-differential-adhesion-sorting-audit.ttl` (39,879 B, `11,300` tok, SHA-256 `1639ef02311bdf76…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signed Pass 1 nanopublications and verified RSA trusty signatures with check_signed.mjs
- **Artifact:** `morphoeval/kg/2026-10-19-differential-adhesion-sorting-audit/published.json` (2,006 B, `679` tok, SHA-256 `1ec4cc838b874995…`)

### Step 11 — `graph` (Public record)
- **Notes:** Verified Pass 1 public record, 10 signed trusty URIs, and REPORT.md synthesis
- **Artifact:** `morphoeval/kg/2026-10-19-differential-adhesion-sorting-audit/REPORT.md` (8,607 B, `2,221` tok, SHA-256 `b8f3de3268b5a9a8…`)
- **Artifact:** `morphoeval/kg/2026-10-19-differential-adhesion-sorting-audit/published.json` (2,006 B, `679` tok, SHA-256 `1ec4cc838b874995…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Completed Step 12 -> Pass 2 remediation loop: 5 thermodynamic regimes, 7-point gamma_AB sweep, confocal embryoid body segmentation (r=0.9252), Foty (1996) 5-tissue hierarchy validation (10/10 pairs), and superseding signed nanopubs (90.5/100, +33.2 pts)
- **Artifact:** `1964-steinberg-differential-adhesion-sorting.py` (46,489 B, `12,721` tok, SHA-256 `913fb1d6faac5816…`)
- **Artifact:** `docs/media/differential_adhesion_sorting_comparison.png` (503,094 B, `64` tok, SHA-256 `755cf4ee94bc2b8c…`)
- **Artifact:** `docs/media/differential_adhesion_sorting_growth.gif` (277,186 B, `64` tok, SHA-256 `13632c457946086e…`)
- **Artifact:** `morphoeval/audits/2026-10-19-differential-adhesion-sorting-audit-v2.yaml` (12,917 B, `3,240` tok, SHA-256 `201b3324b5f32354…`)
- **Artifact:** `morphoeval/audits/2026-10-19-differential-adhesion-sorting-audit-v2.measurements.yaml` (5,004 B, `1,314` tok, SHA-256 `c24d56294f25f46e…`)
- **Artifact:** `morphoeval/kg/2026-10-19-differential-adhesion-sorting-audit-v2/REPORT.md` (9,803 B, `2,718` tok, SHA-256 `df6d7756f99bca3c…`)
- **Artifact:** `morphoeval/kg/2026-10-19-differential-adhesion-sorting-audit-v2/published.json` (2,006 B, `681` tok, SHA-256 `77a200c0b0c9306e…`)
