# MorphoEval Full-Loop Telemetry — `2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-clytia-hemisphaerica-turing-wavelength-s0`
- **Primary paper / dataset:** Clytia hemisphaerica Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **326.00 s** (5.43 min) — Tier A: `240.00 s` · Tier B: `86.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **28,954** (input: `0`, output: `28,954`, subagent: `0`) — Tier A: `14,115` · Tier B: `14,839`
- **Total artifact tokens produced:** `28,954`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 1,534 | 0 | **1,534** | 1,534 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 50.00 | 0.00 | 0 | 2,728 | 0 | **2,728** | 2,728 | `clytia_hemisphaerica_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent (`grok-4.7`) | `completed` | 25.00 | 0.00 | 0 | 64 | 0 | **64** | 64 | `clytia_hemisphaerica_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 1,160 | 0 | **1,160** | 1,160 | `Loop_3150_clytia_hemisphaerica_turing_wavelength_s0.lean`, `delphi10000-clytia-hemisphaerica-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 80.00 | 0.00 | 0 | 8,501 | 0 | **8,501** | 8,501 | `2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | builder-agent (`grok-4.7`) | `completed` | 15.00 | 0.00 | 0 | 128 | 0 | **128** | 128 | `clytia_hemisphaerica_turing_wavelength_s0_growth.gif`, `clytia_hemisphaerica_turing_wavelength_s0_comparison.png` |
| 7 | B | **audit** (Dual-agent audit) | builder-agent (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 4,102 | 0 | **4,102** | 4,102 | `2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-audit.yaml`, `2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | builder-agent (`grok-4.7`) | `completed` | 8.00 | 0.00 | 0 | 2,989 | 0 | **2,989** | 2,989 | `2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | builder-agent (`grok-4.7`) | `completed` | 12.00 | 0.00 | 0 | 4,599 | 0 | **4,599** | 4,599 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | builder-agent (`grok-4.7`) | `completed` | 2.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | builder-agent (`grok-4.7`) | `completed` | 4.00 | 0.00 | 0 | 2,434 | 0 | **2,434** | 2,434 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder-agent (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 715 | 0 | **715** | 715 | `2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **326.00** | **0.00** | **0** | **28,954** | **0** | **28,954** | **28,954** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Wikimedia scout; slot 3150 absent from campaign yaml; no slug collision; OpenAlex/Wikimedia scout; slot 3150 absent from campaign yaml; no slug collision
- **Artifact:** `datasets/clytia_hemisphaerica_turing_wavelength_s0/scout_manifest.json` (5,493 B, `1,534` tok, SHA-256 `014d315226f357c5…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, bib, README; scale 8.64e-6 m and age 0 Ma; Reference JSON, bib, README; scale 8.64e-6 m and age 0 Ma
- **Artifact:** `datasets/clytia_hemisphaerica_turing_wavelength_s0/clytia_hemisphaerica_turing_wavelength_s0_reference.json` (4,243 B, `1,226` tok, SHA-256 `8e95bb9584ac6852…`)
- **Artifact:** `datasets/clytia_hemisphaerica_turing_wavelength_s0/paper_citations.bib` (2,881 B, `908` tok, SHA-256 `045903087c0ddff1…`)
- **Artifact:** `datasets/clytia_hemisphaerica_turing_wavelength_s0/README.md` (2,275 B, `594` tok, SHA-256 `bfb12b8aa4644e1e…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored Momose CC BY-SA day-2 planula; Vendored Momose CC BY-SA day-2 planula
- **Artifact:** `datasets/clytia_hemisphaerica_turing_wavelength_s0/clytia_hemisphaerica_specimen.jpg` (3,109,854 B, `64` tok, SHA-256 `95bffe69b1854823…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean 4 Loop_3150, five theorems, zero sorry; Lean 4 Loop_3150, five theorems, zero sorry
- **Artifact:** `morphoeval/lean/loops/Loop_3150_clytia_hemisphaerica_turing_wavelength_s0.lean` (2,018 B, `64` tok, SHA-256 `02bc5565c5fd67b5…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-clytia-hemisphaerica-turing-wavelength-s0.yaml` (4,000 B, `1,096` tok, SHA-256 `0884884386d14a18…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg script with self-test, probe, ablation, sweep; Schnakenberg script with self-test, probe, ablation, sweep
- **Artifact:** `models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py` (29,775 B, `8,501` tok, SHA-256 `35e280b63ee0c275…`)

### Step 6 — `output` (Output & compare)
- **Notes:** self-test passed; GIF 640px 12 frames; CV 82.1; self-test passed; GIF 640px 12 frames; CV 82.1
- **Artifact:** `docs/media/clytia_hemisphaerica_turing_wavelength_s0_growth.gif` (661,371 B, `64` tok, SHA-256 `0f928ee2c59844b3…`)
- **Artifact:** `docs/media/clytia_hemisphaerica_turing_wavelength_s0_comparison.png` (447,751 B, `64` tok, SHA-256 `20a127f4c9842022…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 audit 75.8; Pass 2 audit 87.4; Pass 1 audit 75.8; Pass 2 audit 87.4
- **Artifact:** `morphoeval/audits/2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-audit.yaml` (7,193 B, `1,822` tok, SHA-256 `471d3fc34f8d5528…`)
- **Artifact:** `morphoeval/audits/2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-audit-v2.yaml` (8,365 B, `2,280` tok, SHA-256 `9c1f1ecbd410359e…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100/100 both passes; measure_outputs quick_gate 100/100 both passes
- **Artifact:** `morphoeval/audits/2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-audit.measurements.yaml` (5,181 B, `1,447` tok, SHA-256 `76eca56eda184d94…`)
- **Artifact:** `morphoeval/audits/2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-audit-v2.measurements.yaml` (5,474 B, `1,542` tok, SHA-256 `a3409cbacda09642…`)

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg --conform OK for v1 and v2; build_morphoeval_kg --conform OK for v1 and v2
- **Artifact:** `morphoeval/kg/2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-audit/REPORT.md` (8,200 B, `2,165` tok, SHA-256 `36a6fe5c0c48a2e0…`)
- **Artifact:** `morphoeval/kg/2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-audit-v2/REPORT.md` (8,724 B, `2,434` tok, SHA-256 `eecba85cfd2e398c…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** signing unavailable: ~/.nanopub/profile.yml missing; nanopubs not signed; signing unavailable: ~/.nanopub/profile.yml missing; nanopubs not signed

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for pass 1 and pass 2; conformance OK; REPORT.md present for pass 1 and pass 2; conformance OK
- **Artifact:** `morphoeval/kg/2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-audit-v2/REPORT.md` (8,724 B, `2,434` tok, SHA-256 `eecba85cfd2e398c…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 eigenvector probe r=0.9993; gates met; pass 3 not required; Pass 2 eigenvector probe r=0.9993; gates met; pass 3 not required
- **Artifact:** `morphoeval/runs/2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-loop-image-fitness.json` (2,434 B, `715` tok, SHA-256 `362614648778ac3d…`)

## Result

Slot 3150 is the Holocene (0 Ma) detail look at 8.64 µm for *Clytia hemisphaerica*. The chronoseries is 1/1, so the age does not change; a reconstructed 250 µm planula holds 28 repeats of that wavelength and a 2 mm bell would hold 231. Pass 1's activator-only growth-rate probe scored Pearson r = 0.506 (audit 75.8). Pass 2 fits the unstable eigenvector: r = 0.999301, RMSE = 0.014 s⁻¹, audit 87.4. The 12-frame 640 px GIF scores cv_sim_similarity 82.1 against Momose's CC BY-SA day-2 planula, and image_fitness is 83.3. All Pass-2 gates hold. Pass 3 was not run.

Nanopublications were built with `build_morphoeval_kg.py --conform` (both reports say conformance OK). They were not signed. `~/.nanopub/profile.yml` is absent and the `nanopub` package is not installed, so `sign_morphoeval.py` cannot run. Nothing was published to a registry.
