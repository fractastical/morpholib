# MorphoEval Full-Loop Telemetry — `2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-ambystoma-mexicanum-turing-wavelength-s0`
- **Primary paper / dataset:** Ambystoma mexicanum Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `unavailable`
- **Start commit:** `dff4f2b5`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **259.00 s** (4.32 min) — Tier A: `170.00 s` · Tier B: `89.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **37,874** (input: `0`, output: `37,874`, subagent: `0`) — Tier A: `18,058` · Tier B: `19,816`
- **Total artifact tokens produced:** `37,874`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 45.00 | 0.00 | 0 | 2,453 | 0 | **2,453** | 2,453 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 3,217 | 0 | **3,217** | 3,217 | `ambystoma_mexicanum_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 64 | 0 | **64** | 64 | `ambystoma_mexicanum_chromatophores_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 931 | 0 | **931** | 931 | `Loop_3107_ambystoma_mexicanum_turing_wavelength_s0.lean`, `delphi10000-ambystoma-mexicanum-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 8,928 | 0 | **8,928** | 8,928 | `2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 15.00 | 0.00 | 0 | 2,465 | 0 | **2,465** | 2,465 | `ambystoma_mexicanum_turing_wavelength_s0_comparison.png`, `ambystoma_mexicanum_turing_wavelength_s0_growth.gif`, `ambystoma_mexicanum_turing_wavelength_s0_metrics.json` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 25.00 | 0.00 | 0 | 5,936 | 0 | **5,936** | 5,936 | `2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-audit.yaml`, `2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 12.00 | 0.00 | 0 | 4,164 | 0 | **4,164** | 4,164 | `2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 15.00 | 0.00 | 0 | 5,774 | 0 | **5,774** | 5,774 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 2.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 5.00 | 0.00 | 0 | 3,053 | 0 | **3,053** | 3,053 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 30.00 | 0.00 | 0 | 889 | 0 | **889** | 889 | `2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **259.00** | **0.00** | **0** | **37,874** | **0** | **37,874** | **37,874** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia; slot 3107 is outside the 100-slot campaign file; No ambystoma-mexicanum-turing-wavelength-s0 collision. Seed DOI 10.1007/978-3-642-75056-4.
- **Artifact:** `datasets/ambystoma_mexicanum_turing_wavelength_s0/scout_manifest.json` (8,753 B, `2,453` tok, SHA-256 `8cfbad14370ae14b…`)

### Step 2 — `paper` (Paper)
- **Notes:** Scale 0.0001843 m and age 0 Ma stated. 184.3 µm is the slot coordinate, not a scale-bar reading.
- **Artifact:** `datasets/ambystoma_mexicanum_turing_wavelength_s0/ambystoma_mexicanum_turing_wavelength_s0_reference.json` (5,790 B, `1,544` tok, SHA-256 `5047ff540b2f84a0…`)
- **Artifact:** `datasets/ambystoma_mexicanum_turing_wavelength_s0/paper_citations.bib` (2,045 B, `622` tok, SHA-256 `09529c875384269a…`)
- **Artifact:** `datasets/ambystoma_mexicanum_turing_wavelength_s0/README.md` (3,910 B, `1,051` tok, SHA-256 `a3909cfa6f8d6d1e…`)

### Step 3 — `source` (Formal source)
- **Notes:** CC BY-SA 4.0 chromatophore micrograph SHA-256 cc106cdbb5c28a40a47946801d35e6f13ada02665b8c773180123279a6cf22c8. Gallery limb plate not used.
- **Artifact:** `datasets/ambystoma_mexicanum_turing_wavelength_s0/ambystoma_mexicanum_chromatophores_specimen.jpg` (3,536,463 B, `64` tok, SHA-256 `cc106cdbb5c28a40…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Five Lean theorems, zero sorry. lean binary not installed, so proofs were not machine-checked here.
- **Artifact:** `morphoeval/lean/loops/Loop_3107_ambystoma_mexicanum_turing_wavelength_s0.lean` (1,981 B, `64` tok, SHA-256 `45a468b898563695…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-ambystoma-mexicanum-turing-wavelength-s0.yaml` (3,165 B, `867` tok, SHA-256 `30a5fdc6af7b289c…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 2 noise-seeded Schnakenberg. Self-test passed.
- **Artifact:** `models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py` (32,709 B, `8,928` tok, SHA-256 `adab8ca22d21c8de…`)

### Step 6 — `output` (Output & compare)
- **Notes:** GIF 640x640, 12 frames. CV similarity 87.5.
- **Artifact:** `docs/media/ambystoma_mexicanum_turing_wavelength_s0_comparison.png` (515,758 B, `64` tok, SHA-256 `eac1fafcca751364…`)
- **Artifact:** `docs/media/ambystoma_mexicanum_turing_wavelength_s0_growth.gif` (1,612,487 B, `64` tok, SHA-256 `8d6fd506a2a8af42…`)
- **Artifact:** `datasets/ambystoma_mexicanum_turing_wavelength_s0/ambystoma_mexicanum_turing_wavelength_s0_metrics.json` (7,953 B, `2,337` tok, SHA-256 `089f5533e94d2527…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 80.0. Pass 2 overall 85.2. Single assessor; R5 left open.
- **Artifact:** `morphoeval/audits/2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-audit.yaml` (11,411 B, `2,832` tok, SHA-256 `1654eb2b88d8430c…`)
- **Artifact:** `morphoeval/audits/2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-audit-v2.yaml` (12,289 B, `3,104` tok, SHA-256 `e4dce559362e2755…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 2 quick_gate 100/100. Pearson r 0.999595.
- **Artifact:** `morphoeval/audits/2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-audit.measurements.yaml` (7,620 B, `2,016` tok, SHA-256 `96b335fdccb5d61f…`)
- **Artifact:** `morphoeval/audits/2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-audit-v2.measurements.yaml` (8,215 B, `2,148` tok, SHA-256 `a6074e1b2cc77d18…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Both passes conform. Rubric nanopub reused from the Bos taurus v7 publication.
- **Artifact:** `morphoeval/kg/2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-audit/REPORT.md` (10,463 B, `2,721` tok, SHA-256 `56cd5dd56c457fd6…`)
- **Artifact:** `morphoeval/kg/2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-audit-v2/REPORT.md` (11,402 B, `3,053` tok, SHA-256 `13fe923399a87997…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml is absent. Nanopubs were not signed and check_signed.mjs was not claimed as a pass.

### Step 11 — `graph` (Public record)
- **Notes:** Pass 2 REPORT overall 85.2, verdict faithful.
- **Artifact:** `morphoeval/kg/2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-audit-v2/REPORT.md` (11,402 B, `3,053` tok, SHA-256 `13fe923399a87997…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 gates met: image_fitness 84.0, CV 87.5, Pearson r 0.9996, audit 85.2. Pass 3 not required. R5 single assessor remains open.
- **Artifact:** `morphoeval/runs/2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-loop-image-fitness.json` (3,151 B, `889` tok, SHA-256 `e9af80804248f2c9…`)

## Gates after Pass 2

All numeric gates passed on Pass 2. Pass 3 was not run.

- image_fitness 84.0 (>= 70)
- cv_sim_similarity 87.5 (>= 58), 12 frames, 640 px
- Pearson r 0.999595 (> 0.98)
- audit overall 85.2 (>= 85)
- non-circular specimen metrics: dark-area fraction 0.380 versus model 0.314, coherency 0.485 versus 0.904

Signing was not performed. `~/.nanopub/profile.yml` is absent, so the nanopublications are unsigned drafts. `check_signed.mjs` was not claimed as a pass.

The single-assessor limit (Pass 1 remediation R5) is still open. No second provider was invented.
