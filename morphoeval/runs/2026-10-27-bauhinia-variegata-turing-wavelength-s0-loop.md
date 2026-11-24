# MorphoEval Full-Loop Telemetry — `2026-10-27-bauhinia-variegata-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-bauhinia-variegata-turing-wavelength-s0`
- **Primary paper / dataset:** Bauhinia variegata Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1096.67 s** (18.28 min) — Tier A: `1091.35 s` · Tier B: `5.31 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **32,563** (input: `0`, output: `32,563`, subagent: `0`) — Tier A: `15,391` · Tier B: `17,172`
- **Total artifact tokens produced:** `32,563`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 1087.03 | 0.00 | 0 | 1,605 | 0 | **1,605** | 1,605 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.85 | 0.00 | 0 | 2,984 | 0 | **2,984** | 2,984 | `bauhinia_variegata_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.86 | 0.00 | 0 | 64 | 0 | **64** | 64 | `bauhinia_variegata_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.87 | 0.00 | 0 | 890 | 0 | **890** | 890 | `Loop_3132_bauhinia_variegata_turing_wavelength_s0.lean`, `delphi10000-bauhinia-variegata-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.87 | 0.00 | 0 | 9,720 | 0 | **9,720** | 9,720 | `2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.87 | 0.00 | 0 | 128 | 0 | **128** | 128 | `bauhinia_variegata_turing_wavelength_s0_comparison.png`, `bauhinia_variegata_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.87 | 0.00 | 0 | 5,107 | 0 | **5,107** | 5,107 | `2026-10-27-bauhinia-variegata-turing-wavelength-s0-audit.yaml`, `2026-10-27-bauhinia-variegata-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.88 | 0.00 | 0 | 3,795 | 0 | **3,795** | 3,795 | `2026-10-27-bauhinia-variegata-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-bauhinia-variegata-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.88 | 0.00 | 0 | 4,828 | 0 | **4,828** | 4,828 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.89 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.89 | 0.00 | 0 | 2,646 | 0 | **2,646** | 2,646 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.90 | 0.00 | 0 | 796 | 0 | **796** | 796 | `2026-10-27-bauhinia-variegata-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1096.67** | **0.00** | **0** | **32,563** | **0** | **32,563** | **32,563** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Slot 3132 is not in campaign_100_slots.yaml. OpenAlex and Wikimedia scout written; no name collision with the chiral-pod slot.
- **Artifact:** `datasets/bauhinia_variegata_turing_wavelength_s0/scout_manifest.json` (5,886 B, `1,605` tok, SHA-256 `a3ec9f80b8bf56fd…`)

### Step 2 — `paper` (Paper)
- **Notes:** Equations, 69.12 µm scale, and age 0 Ma recorded in the reference JSON and bibliography.
- **Artifact:** `datasets/bauhinia_variegata_turing_wavelength_s0/bauhinia_variegata_turing_wavelength_s0_reference.json` (5,576 B, `1,694` tok, SHA-256 `65eb5aab368642b9…`)
- **Artifact:** `datasets/bauhinia_variegata_turing_wavelength_s0/paper_citations.bib` (2,431 B, `751` tok, SHA-256 `b2ce9fe72b983b7e…`)
- **Artifact:** `datasets/bauhinia_variegata_turing_wavelength_s0/README.md` (1,876 B, `539` tok, SHA-256 `75c1266133012f1d…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored the CC BY-SA 3.0 MHNT pod photograph. SHA-256 300694d305567ca5b259ab023ea8e40a734061a3f55b78de1a23d30cbd2ffd3a.
- **Artifact:** `datasets/bauhinia_variegata_turing_wavelength_s0/bauhinia_variegata_specimen.jpg` (87,454 B, `64` tok, SHA-256 `300694d305567ca5…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3132: four theorems, zero sorry. Ratio margin, gauge cancellation, 144 wavelengths per centimetre, age 0 Ma.
- **Artifact:** `morphoeval/lean/loops/Loop_3132_bauhinia_variegata_turing_wavelength_s0.lean` (2,147 B, `64` tok, SHA-256 `ecde18740c8cd706…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-bauhinia-variegata-turing-wavelength-s0.yaml` (2,963 B, `826` tok, SHA-256 `b4dd5a5a04321279…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg/Turing script with self-test, ablation, diffusivity sweep, and specimen segmentation.
- **Artifact:** `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` (34,527 B, `9,720` tok, SHA-256 `608d29ac8b110aa1…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test passed. GIF 12 frames 960x623. Pearson r 0.9935. CV similarity 96.0.
- **Artifact:** `docs/media/bauhinia_variegata_turing_wavelength_s0_comparison.png` (485,676 B, `64` tok, SHA-256 `5a38587e481dd1f3…`)
- **Artifact:** `docs/media/bauhinia_variegata_turing_wavelength_s0_growth.gif` (787,507 B, `64` tok, SHA-256 `0068a14bf244c49f…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall about 78.9 with four remediations. Pass 2 overall 87.3 after the sweep, blend, and valve KS.
- **Artifact:** `morphoeval/audits/2026-10-27-bauhinia-variegata-turing-wavelength-s0-audit.yaml` (8,296 B, `2,132` tok, SHA-256 `1df028ef98a6f610…`)
- **Artifact:** `morphoeval/audits/2026-10-27-bauhinia-variegata-turing-wavelength-s0-audit-v2.yaml` (11,004 B, `2,975` tok, SHA-256 `c1cbae6cd12c7ed0…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100/100 on both audits. 14 measurements, values distinct from references where the images differ.
- **Artifact:** `morphoeval/audits/2026-10-27-bauhinia-variegata-turing-wavelength-s0-audit.measurements.yaml` (6,495 B, `1,722` tok, SHA-256 `40816c5e7573cf6c…`)
- **Artifact:** `morphoeval/audits/2026-10-27-bauhinia-variegata-turing-wavelength-s0-audit-v2.measurements.yaml` (7,531 B, `2,073` tok, SHA-256 `587773d7cdb09f73…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Knowledge graphs built with --conform. ProbKnow conformance OK. Rubric nanopub supersedes the bos-taurus v7 publication.
- **Artifact:** `morphoeval/kg/2026-10-27-bauhinia-variegata-turing-wavelength-s0-audit/REPORT.md` (8,021 B, `2,182` tok, SHA-256 `d88be44fce43a32e…`)
- **Artifact:** `morphoeval/kg/2026-10-27-bauhinia-variegata-turing-wavelength-s0-audit-v2/REPORT.md` (9,271 B, `2,646` tok, SHA-256 `7afb8f86e442fe4e…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable. ~/.nanopub/profile.yml is absent and the nanopub package is not installed. No signature was fabricated. Publisher ORCID on the unsigned graph is the repository publisher 0009-0007-4948-9192.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md for pass 2 records overall 87.3, verdict faithful.
- **Artifact:** `morphoeval/kg/2026-10-27-bauhinia-variegata-turing-wavelength-s0-audit-v2/REPORT.md` (9,271 B, `2,646` tok, SHA-256 `7afb8f86e442fe4e…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed the weak ratio sweep, the opaque inset (Pearson 0.9776 to 0.9935), the missing scale bar, and the unseparated KS. Gates pass. No pass 3. Remaining: nonlinear wavelength 76.16 vs 69.12 µm, GIF palette KS, no second assessor.
- **Artifact:** `morphoeval/runs/2026-10-27-bauhinia-variegata-turing-wavelength-s0-loop-image-fitness.json` (2,669 B, `796` tok, SHA-256 `007d9c16b4cc19e1…`)

## Pass 2 gates

Holocene *Bauhinia variegata*, 69.12 µm (6.912e-05 m, log10 = -4.160), age 0 Ma. Schnakenberg kinetics with D_v/D_u = 16.9. Linear wavelength is exact. Saturated radial peak is 76.16 µm and is reported as a nonlinear shift.

| Gate | Result |
|---|---|
| image_fitness | 91.8 (>= 70) |
| cv_sim_similarity | 96.0 (>= 58) |
| GIF frames | 12 (>= 6), 960×623 |
| Pearson r | 0.9935 (> 0.98) |
| Audit overall | 87.3 (>= 85) |
| Non-circular specimen metrics | foreground fraction, circularity, valve KS, coherency, FFT wavelength |

Pass 3 was not run. Nanopub signing was not performed: no profile and no nanopub package. The graphs are unsigned.
