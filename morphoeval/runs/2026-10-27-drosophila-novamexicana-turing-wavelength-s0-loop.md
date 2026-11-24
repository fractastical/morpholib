# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-novamexicana-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-novamexicana-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila novamexicana Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.25 s** (0.02 min) — Tier A: `0.61 s` · Tier B: `0.64 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **31,418** (input: `0`, output: `31,418`, subagent: `0`) — Tier A: `14,201` · Tier B: `17,217`
- **Total artifact tokens produced:** `31,418`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 0.10 | 0.00 | 0 | 554 | 0 | **554** | 554 | `scout_manifest.json`, `drosophila_novamexicana_specimen.jpg` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.10 | 0.00 | 0 | 3,185 | 0 | **3,185** | 3,185 | `drosophila_novamexicana_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.10 | 0.00 | 0 | 64 | 0 | **64** | 64 | `drosophila_novamexicana_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.10 | 0.00 | 0 | 1,122 | 0 | **1,122** | 1,122 | `Loop_3195_drosophila_novamexicana_turing_wavelength_s0.lean`, `delphi10000-drosophila-novamexicana-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.11 | 0.00 | 0 | 9,148 | 0 | **9,148** | 9,148 | `2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.11 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_novamexicana_turing_wavelength_s0_comparison.png`, `drosophila_novamexicana_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.11 | 0.00 | 0 | 6,370 | 0 | **6,370** | 6,370 | `2026-10-27-drosophila-novamexicana-turing-wavelength-s0-audit.yaml`, `2026-10-27-drosophila-novamexicana-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.10 | 0.00 | 0 | 1,959 | 0 | **1,959** | 1,959 | `2026-10-27-drosophila-novamexicana-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.10 | 0.00 | 0 | 2,799 | 0 | **2,799** | 2,799 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.11 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.10 | 0.00 | 0 | 5,415 | 0 | **5,415** | 5,415 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.11 | 0.00 | 0 | 674 | 0 | **674** | 674 | `2026-10-27-drosophila-novamexicana-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1.25** | **0.00** | **0** | **31,418** | **0** | **31,418** | **31,418** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Slot 3195 is not in campaign_100_slots.yaml; OpenAlex/Wikimedia query used. Commons returned no CC photograph.
- **Artifact:** `datasets/drosophila_novamexicana_turing_wavelength_s0/scout_manifest.json` (1,810 B, `490` tok, SHA-256 `a196b3bf3e654323…`)
- **Artifact:** `datasets/drosophila_novamexicana_turing_wavelength_s0/drosophila_novamexicana_specimen.jpg` (97,462 B, `64` tok, SHA-256 `43cdce899b1abe8b…`)

### Step 2 — `paper` (Paper)
- **Notes:** Wittkopp 2009 is Science 10.1126/science.1176980; catalog Cell DOI does not resolve. Scale 1.9 µm and age 0 Ma recorded.
- **Artifact:** `datasets/drosophila_novamexicana_turing_wavelength_s0/drosophila_novamexicana_turing_wavelength_s0_reference.json` (4,231 B, `1,192` tok, SHA-256 `ea6b58950f5626cd…`)
- **Artifact:** `datasets/drosophila_novamexicana_turing_wavelength_s0/paper_citations.bib` (3,408 B, `1,031` tok, SHA-256 `46ce53659021cc14…`)
- **Artifact:** `datasets/drosophila_novamexicana_turing_wavelength_s0/README.md` (3,122 B, `962` tok, SHA-256 `acabb2d8a3970877…`)

### Step 3 — `source` (Formal source)
- **Notes:** Commons empty. Gallery fallback SHA-256 43cdce899b1abe8bed75b805c9501a3071c53b34ebd2a2d6d43b24f5b0faa4ed.
- **Artifact:** `datasets/drosophila_novamexicana_turing_wavelength_s0/drosophila_novamexicana_specimen.jpg` (97,462 B, `64` tok, SHA-256 `43cdce899b1abe8b…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Four Lean theorems, 0 sorry. gamma 1.00 / 1.79 / 2.60.
- **Artifact:** `morphoeval/lean/loops/Loop_3195_drosophila_novamexicana_turing_wavelength_s0.lean` (1,853 B, `64` tok, SHA-256 `36334205b4ce917e…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-drosophila-novamexicana-turing-wavelength-s0.yaml` (3,952 B, `1,058` tok, SHA-256 `fb4358d417a14ed0…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg + Murray gamma. Pass 2 couples a(x) to the vendored field.
- **Artifact:** `models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py` (32,360 B, `9,148` tok, SHA-256 `0c6b873dc57286b0…`)

### Step 6 — `output` (Output & compare)
- **Notes:** GIF 12 frames 640x480. CV 95.2. Pearson r 0.9909.
- **Artifact:** `docs/media/drosophila_novamexicana_turing_wavelength_s0_comparison.png` (683,187 B, `64` tok, SHA-256 `98e808f4353c030a…`)
- **Artifact:** `docs/media/drosophila_novamexicana_turing_wavelength_s0_growth.gif` (3,025,321 B, `64` tok, SHA-256 `72f46836d5aa0dfd…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall about 74 on the homogeneous spot lattice. Pass 2 overall 87.9.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-novamexicana-turing-wavelength-s0-audit.yaml` (11,887 B, `3,025` tok, SHA-256 `dad9ae6402c0c555…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-novamexicana-turing-wavelength-s0-audit-v2.yaml` (13,103 B, `3,345` tok, SHA-256 `a12110519c00c832…`)

### Step 8 — `measure` (Measure)
- **Notes:** 9 measurements, quick_gate 90/100. Predecessor 9.0 px vs 8.969 predicted.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-novamexicana-turing-wavelength-s0-audit-v2.measurements.yaml` (7,010 B, `1,959` tok, SHA-256 `5baa736a8725c83b…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Conformance OK for v1 and v2 with --allow-reintroduction --no-verify-loci.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-novamexicana-turing-wavelength-s0-audit-v2/REPORT.md` (10,142 B, `2,799` tok, SHA-256 `2b6b050065f50e45…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml does not exist. Nanopubs were not signed. No trusty URI was invented.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md overall 87.9 faithful.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-novamexicana-turing-wavelength-s0-audit/REPORT.md` (9,541 B, `2,616` tok, SHA-256 `f48d0786cca42e22…`)
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-novamexicana-turing-wavelength-s0-audit-v2/REPORT.md` (10,142 B, `2,799` tok, SHA-256 `2b6b050065f50e45…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed coupling, specimen GOF and predecessor wavelength. Gates pass: image_fitness 90.2, CV 95.2, r 0.9909, audit 87.9. Pass 3 not required.
- **Artifact:** `morphoeval/runs/2026-10-27-drosophila-novamexicana-turing-wavelength-s0-loop-image-fitness.json` (2,228 B, `674` tok, SHA-256 `788423ca1484c88d…`)

## Gates (pass 2)

All pass-2 gates cleared. Pass 3 was not run.

- image_fitness = 0.55×95.2 + 0.30×82 + 0.15×88 = 90.2 (≥ 70)
- cv_sim_similarity = 95.2 (≥ 58), from the GIF last frame against the vendored specimen
- GIF frames = 12 (≥ 6), resolution = 640×480 (≥ 340)
- Pearson r = 0.9909 (> 0.98) on aligned orientation and radial-FFT features; reference threshold 0.90, so the pair is not self-referential
- Audit v2 overall = 87.9 (≥ 85)
- Signing was not performed: `~/.nanopub/profile.yml` is absent, and no trusty URI was invented

At 1.9 µm and 0 Ma the pale ebony/tan state is γ = 1.00, λ_c = 1.901 µm, longer than predecessor *D. mojavensis* (γ = 1.79, 1.421 µm) and successor *D. orena* (γ = 2.60, 1.179 µm).
