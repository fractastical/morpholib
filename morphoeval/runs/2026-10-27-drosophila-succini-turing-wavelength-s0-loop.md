# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-succini-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-succini-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila succini Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **860.14 s** (14.34 min) — Tier A: `855.07 s` · Tier B: `5.08 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **34,849** (input: `0`, output: `34,849`, subagent: `0`) — Tier A: `15,231` · Tier B: `19,618`
- **Total artifact tokens produced:** `34,849`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 851.02 | 0.00 | 0 | 1,215 | 0 | **1,215** | 1,215 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.80 | 0.00 | 0 | 3,440 | 0 | **3,440** | 3,440 | `drosophila_succini_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.81 | 0.00 | 0 | 64 | 0 | **64** | 64 | `drosophila_succini_turing_wavelength_s0_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.81 | 0.00 | 0 | 1,191 | 0 | **1,191** | 1,191 | `Loop_3172_drosophila_succini_turing_wavelength_s0.lean`, `delphi10000-drosophila-succini-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.81 | 0.00 | 0 | 9,193 | 0 | **9,193** | 9,193 | `2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.82 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_succini_turing_wavelength_s0_comparison.png`, `drosophila_succini_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.82 | 0.00 | 0 | 2,957 | 0 | **2,957** | 2,957 | `2026-10-27-drosophila-succini-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.83 | 0.00 | 0 | 3,940 | 0 | **3,940** | 3,940 | `2026-10-27-drosophila-succini-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-drosophila-succini-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.83 | 0.00 | 0 | 5,584 | 0 | **5,584** | 5,584 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.86 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.86 | 0.00 | 0 | 2,975 | 0 | **2,975** | 2,975 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.87 | 0.00 | 0 | 4,162 | 0 | **4,162** | 4,162 | `2026-10-27-drosophila-succini-turing-wavelength-s0-audit-v2.yaml`, `2026-10-27-drosophila-succini-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **860.14** | **0.00** | **0** | **34,849** | **0** | **34,849** | **34,849** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Slot 3172 absent from campaign_100; OpenAlex DOI lookup + Wikimedia; Slot 3172 is outside campaign_100. DOI-resolved Turing, Schnakenberg, Kondo, Hennig Zenodo, Grimaldi Zenodo. Commons empty. No s0 collision; 114 µm sibling left untouched.
- **Artifact:** `datasets/drosophila_succini_turing_wavelength_s0/scout_manifest.json` (4,722 B, `1,215` tok, SHA-256 `b2f6581854a7b8e0…`)

### Step 2 — `paper` (Paper)
- **Notes:** Equations, Jacobian, k_c, chronoseries contrast, 2.759e-06 m and 44 Ma in the reference JSON and README.
- **Artifact:** `datasets/drosophila_succini_turing_wavelength_s0/drosophila_succini_turing_wavelength_s0_reference.json` (6,857 B, `2,330` tok, SHA-256 `e227ba771980817d…`)
- **Artifact:** `datasets/drosophila_succini_turing_wavelength_s0/paper_citations.bib` (1,616 B, `485` tok, SHA-256 `77f5cfe7c321181d…`)
- **Artifact:** `datasets/drosophila_succini_turing_wavelength_s0/README.md` (2,209 B, `625` tok, SHA-256 `3a127b4f8387d181…`)

### Step 3 — `source` (Formal source)
- **Notes:** No Commons bitmap. Vendored species-gallery card, SHA-256 1a0de2972916cbfa2e2b0e21ebeb0aed9e2f4f99f11b26993c29f2c9b36464cd.
- **Artifact:** `datasets/drosophila_succini_turing_wavelength_s0/drosophila_succini_turing_wavelength_s0_specimen.jpg` (99,713 B, `64` tok, SHA-256 `1a0de2972916cbfa…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3172: five theorems, zero sorry. Math KG yaml with variables and lean_code.
- **Artifact:** `morphoeval/lean/loops/Loop_3172_drosophila_succini_turing_wavelength_s0.lean` (1,760 B, `64` tok, SHA-256 `168cf64b9c561139…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-drosophila-succini-turing-wavelength-s0.yaml` (4,172 B, `1,127` tok, SHA-256 `ca8a00e0fa6bc60a…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg k_c model with --self-test. Band Pearson r = 0.999999997.
- **Artifact:** `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` (31,885 B, `9,193` tok, SHA-256 `cb576eb5f2159049…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test PASS. GIF 12 frames 960x720. Comparison PNG written.
- **Artifact:** `docs/media/drosophila_succini_turing_wavelength_s0_comparison.png` (448,421 B, `64` tok, SHA-256 `8b3f9e4fcc75332b…`)
- **Artifact:** `docs/media/drosophila_succini_turing_wavelength_s0_growth.gif` (4,958,003 B, `64` tok, SHA-256 `c5d87de63a9f808b…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 76.8. Open items: sweep sidecar, specimen segmentation, band GOF.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-succini-turing-wavelength-s0-audit.yaml` (11,846 B, `2,957` tok, SHA-256 `18cd3a8e77fe16e7…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100/100, 11 measurements, none self-referential.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-succini-turing-wavelength-s0-audit.measurements.yaml` (6,985 B, `1,933` tok, SHA-256 `f9d33a17003cfdba…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-succini-turing-wavelength-s0-audit-v2.measurements.yaml` (7,305 B, `2,007` tok, SHA-256 `99cb85b790a0cd4e…`)

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg --conform OK for pass 1 and pass 2. Rubric nanopub not re-emitted.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-succini-turing-wavelength-s0-audit/REPORT.md` (9,605 B, `2,609` tok, SHA-256 `c8b978a1f6ab6334…`)
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-succini-turing-wavelength-s0-audit-v2/REPORT.md` (10,760 B, `2,975` tok, SHA-256 `9d21428150f32fc2…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing did not run. `morphoeval/sign_morphoeval.py` failed on import because `requests` is not installed, and `~/.nanopub/profile.yml` is absent, so no key exists. Nanopubs were not signed and were not marked as signed.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md pass 2 overall 86.5 faithful. Supersession chain v1 -> v2.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-succini-turing-wavelength-s0-audit-v2/REPORT.md` (10,760 B, `2,975` tok, SHA-256 `9d21428150f32fc2…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closes sweep, ablation, specimen CV segmentation, and band Pearson r > 0.98. image_fitness 83.9, cv 84.9. Gates pass; no pass 3.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-succini-turing-wavelength-s0-audit-v2.yaml` (13,661 B, `3,449` tok, SHA-256 `c1cca457b04dad42…`)
- **Artifact:** `morphoeval/runs/2026-10-27-drosophila-succini-turing-wavelength-s0-loop-image-fitness.json` (2,377 B, `713` tok, SHA-256 `cdf2c62437004892…`)
