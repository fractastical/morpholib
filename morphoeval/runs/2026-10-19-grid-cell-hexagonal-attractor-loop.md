# MorphoEval Full-Loop Telemetry — `2026-10-19-grid-cell-hexagonal-attractor-loop`

- **Target simulation:** `2008-burak-grid-cell-hexagonal-attractor.py`
- **Primary paper / dataset:** Burak & Fiete (2009) PLoS Comput. Biol. 5:e1000291; Hafting et al. (2005) Nature 436:801-806; Stensola et al. (2012) Nature 492:72-78
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `2a34864`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **542.33 s** (9.04 min) — Tier A: `288.35 s` · Tier B: `253.98 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **258,169** (input: `612`, output: `257,557`, subagent: `0`) — Tier A: `162,728` · Tier B: `95,441`
- **Total artifact tokens produced:** `27,690`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 2.24 | 0.00 | 0 | 567 | 0 | **567** | 876 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 85.82 | 0.00 | 188 | 48,703 | 0 | **48,891** | 2,524 | `grid_cell_hexagonal_attractor_reference.json`, `paper_citations.bib` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 1,121 | 0 | **1,121** | 1,121 | `rat_entorhinal_hippocampal_specimen.jpg`, `README.md` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 144.57 | 0.00 | 0 | 62,716 | 0 | **62,716** | 2,550 | `Loop_088_grid_cell_hexagonal_attractor.lean`, `burak-grid-cell-hexagonal-attractor.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 54.91 | 0.00 | 0 | 49,305 | 0 | **49,305** | 1,645 | `2008-burak-grid-cell-hexagonal-attractor.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.72 | 0.00 | 0 | 128 | 0 | **128** | 128 | `grid_cell_hexagonal_attractor_comparison.png`, `grid_cell_hexagonal_attractor_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 13.09 | 0.00 | 0 | 4,071 | 0 | **4,071** | 2,297 | `2026-10-19-grid-cell-hexagonal-attractor-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.47 | 0.00 | 0 | 816 | 0 | **816** | 816 | `2026-10-19-grid-cell-hexagonal-attractor-audit.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.44 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 13.79 | 0.00 | 0 | 6,413 | 0 | **6,413** | 686 | `published.json` |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.09 | 0.00 | 0 | 2,135 | 0 | **2,135** | 2,135 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 226.10 | 0.00 | 424 | 81,582 | 0 | **82,006** | 12,912 | `2008-burak-grid-cell-hexagonal-attractor.py`, `grid_cell_hexagonal_attractor_comparison.png`, `grid_cell_hexagonal_attractor_growth.gif` |
| — | **All** | **TOTAL** | — | `12/12` | **542.33** | **0.00** | **612** | **257,557** | **0** | **258,169** | **27,690** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Artifact:** `datasets/grid_cell_hexagonal_attractor/scout_manifest.json` (3,264 B, `876` tok, SHA-256 `77ac05875318888d…`)

### Step 2 — `paper` (Paper)
- **Artifact:** `datasets/grid_cell_hexagonal_attractor/grid_cell_hexagonal_attractor_reference.json` (7,032 B, `2,073` tok, SHA-256 `951b00ff56331c34…`)
- **Artifact:** `datasets/grid_cell_hexagonal_attractor/paper_citations.bib` (1,493 B, `451` tok, SHA-256 `e1074e8d98b9dc00…`)

### Step 3 — `source` (Formal source)
- **Artifact:** `datasets/grid_cell_hexagonal_attractor/rat_entorhinal_hippocampal_specimen.jpg` (800,089 B, `64` tok, SHA-256 `63b3a392f8d0b25f…`)
- **Artifact:** `datasets/grid_cell_hexagonal_attractor/README.md` (3,340 B, `1,057` tok, SHA-256 `7ee2c39846914fa0…`)

### Step 4 — `rules` (Pin the rules)
- **Artifact:** `morphoeval/lean/loops/Loop_088_grid_cell_hexagonal_attractor.lean` (8,681 B, `64` tok, SHA-256 `3252aee8d739483a…`)
- **Artifact:** `morphoeval/math_kg_loops/burak-grid-cell-hexagonal-attractor.yaml` (8,970 B, `2,486` tok, SHA-256 `5c89e7d6605691aa…`)

### Step 5 — `script` (Script)
- **Artifact:** `2008-burak-grid-cell-hexagonal-attractor.py` (5,825 B, `1,645` tok, SHA-256 `f1c0a712337a7ad6…`)

### Step 6 — `output` (Output & compare)
- **Artifact:** `docs/media/grid_cell_hexagonal_attractor_comparison.png` (479,407 B, `64` tok, SHA-256 `40360ed0659fb8b5…`)
- **Artifact:** `docs/media/grid_cell_hexagonal_attractor_growth.gif` (94,929 B, `64` tok, SHA-256 `7947b7a19400ccc7…`)

### Step 7 — `audit` (Dual-agent audit)
- **Artifact:** `morphoeval/audits/2026-10-19-grid-cell-hexagonal-attractor-audit.yaml` (9,020 B, `2,297` tok, SHA-256 `f6cc7379d8f21565…`)

### Step 8 — `measure` (Measure)
- **Artifact:** `morphoeval/audits/2026-10-19-grid-cell-hexagonal-attractor-audit.measurements.yaml` (2,479 B, `816` tok, SHA-256 `1ced846eca617fe8…`)

### Step 10 — `publish` (Sign & verify)
- **Artifact:** `morphoeval/kg/2026-10-19-grid-cell-hexagonal-attractor-audit/published.json` (1,970 B, `686` tok, SHA-256 `44b041aeba9a00e5…`)

### Step 11 — `graph` (Public record)
- **Artifact:** `morphoeval/kg/2026-10-19-grid-cell-hexagonal-attractor-audit/REPORT.md` (7,944 B, `2,135` tok, SHA-256 `775b072f9aac7c04…`)

### Step 12 — `remed` (Remediate & re-score)
- **Artifact:** `2008-burak-grid-cell-hexagonal-attractor.py` (45,822 B, `12,784` tok, SHA-256 `9a070215b86dda51…`)
- **Artifact:** `docs/media/grid_cell_hexagonal_attractor_comparison.png` (990,359 B, `64` tok, SHA-256 `9e15f015a1a8af31…`)
- **Artifact:** `docs/media/grid_cell_hexagonal_attractor_growth.gif` (957,172 B, `64` tok, SHA-256 `2f1d902ce68724f1…`)
