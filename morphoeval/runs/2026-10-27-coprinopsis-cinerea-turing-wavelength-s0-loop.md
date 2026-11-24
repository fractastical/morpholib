# MorphoEval Full-Loop Telemetry — `2026-10-27-coprinopsis-cinerea-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-coprinopsis-cinerea-turing-wavelength-s0`
- **Primary paper / dataset:** Coprinopsis cinerea Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-run`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.05 s** (0.02 min) — Tier A: `0.52 s` · Tier B: `0.53 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **26,223** (input: `0`, output: `26,223`, subagent: `0`) — Tier A: `11,921` · Tier B: `14,302`
- **Total artifact tokens produced:** `26,223`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 729 | 0 | **729** | 729 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,640 | 0 | **1,640** | 1,640 | `coprinopsis_cinerea_turing_wavelength_s0_reference.json` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 64 | 0 | **64** | 64 | `coprinopsis_cinerea_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 64 | 0 | **64** | 64 | `Loop_3154_coprinopsis_cinerea_turing_wavelength_s0.lean` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 9,360 | 0 | **9,360** | 9,360 | `2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 64 | 0 | **64** | 64 | `coprinopsis_cinerea_turing_wavelength_s0_comparison.png` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,413 | 0 | **2,413** | 2,413 | `2026-10-27-coprinopsis-cinerea-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,162 | 0 | **2,162** | 2,162 | `2026-10-27-coprinopsis-cinerea-turing-wavelength-s0-audit.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,588 | 0 | **2,588** | 2,588 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 3,170 | 0 | **3,170** | 3,170 | `REPORT.md` |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 3,170 | 0 | **3,170** | 3,170 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 799 | 0 | **799** | 799 | `2026-10-27-coprinopsis-cinerea-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1.05** | **0.00** | **0** | **26,223** | **0** | **26,223** | **26,223** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex full-text search was contaminated by Botrytis; Crossref-verified Turing 1952, Schnakenberg 1979, Kues 2000, Stajich 2010. No slug collision.; OpenAlex full-text search was contaminated by Botrytis; Crossref-verified Turing 1952, Schnakenberg 1979, Kues 2000, Stajich 2010. No slug collision.
- **Artifact:** `datasets/coprinopsis_cinerea_turing_wavelength_s0/scout_manifest.json` (2,686 B, `729` tok, SHA-256 `0d10620c586241bf…`)

### Step 2 — `paper` (Paper)
- **Notes:** Extracted Schnakenberg PDE, Jacobian, k_c, scale 7.744e-05 m, age 0 Ma, chronoseries 1/1.; Extracted Schnakenberg PDE, Jacobian, k_c, scale 7.744e-05 m, age 0 Ma, chronoseries 1/1.
- **Artifact:** `datasets/coprinopsis_cinerea_turing_wavelength_s0/coprinopsis_cinerea_turing_wavelength_s0_reference.json` (5,938 B, `1,640` tok, SHA-256 `d4e14ac29c4347af…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored CC BY-SA 3.0 Mushroom Observer 593470 thumbnail. SHA-256 f5905e9c02019fc55d651c22742ce29a53541673e37355631a29a5076046b06b. Gallery fallback not used.; Vendored CC BY-SA 3.0 Mushroom Observer 593470 thumbnail. SHA-256 f5905e9c02019fc55d651c22742ce29a53541673e37355631a29a5076046b06b. Gallery fallback not used.
- **Artifact:** `datasets/coprinopsis_cinerea_turing_wavelength_s0/coprinopsis_cinerea_specimen.jpg` (223,222 B, `64` tok, SHA-256 `f5905e9c02019fc5…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean namespace MathOfBiology.Loop_3154, five theorems, zero sorry. Math KG yaml registered.; Lean namespace MathOfBiology.Loop_3154, five theorems, zero sorry. Math KG yaml registered.
- **Artifact:** `morphoeval/lean/loops/Loop_3154_coprinopsis_cinerea_turing_wavelength_s0.lean` (2,162 B, `64` tok, SHA-256 `575b5f88ef45f1eb…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 1 integrator plus Pass 2 sweep, ablations, segmentation, and dispersion GOF. --self-test passed.; Pass 1 integrator plus Pass 2 sweep, ablations, segmentation, and dispersion GOF. --self-test passed.
- **Artifact:** `models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` (32,685 B, `9,360` tok, SHA-256 `382bd98167c15e98…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Rendered comparison PNG and 12-frame 640x480 growth GIF. CV similarity 87.2.; Rendered comparison PNG and 12-frame 640x480 growth GIF. CV similarity 87.2.
- **Artifact:** `docs/media/coprinopsis_cinerea_turing_wavelength_s0_comparison.png` (779,099 B, `64` tok, SHA-256 `3a01f0cb3e83a03a…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 audit withholds sweep/ablation/segmentation/GOF credit. Pass 2 audit-v2 accepts them. Overall 86.2.; Pass 1 audit withholds sweep/ablation/segmentation/GOF credit. Pass 2 audit-v2 accepts them. Overall 86.2.
- **Artifact:** `morphoeval/audits/2026-10-27-coprinopsis-cinerea-turing-wavelength-s0-audit.yaml` (9,087 B, `2,413` tok, SHA-256 `a9385bffbb7de0ef…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs.py on both audits. quick_gate 100/100. 14 measurements, none self-referential.; measure_outputs.py on both audits. quick_gate 100/100. 14 measurements, none self-referential.
- **Artifact:** `morphoeval/audits/2026-10-27-coprinopsis-cinerea-turing-wavelength-s0-audit.measurements.yaml` (8,097 B, `2,162` tok, SHA-256 `37e36fbbed0bd839…`)

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg.py --conform --rubric-from the published Bos taurus v7 rubric. Conformance OK, 9 nanopubs each. Rubric not reintroduced.; build_morphoeval_kg.py --conform --rubric-from the published Bos taurus v7 rubric. Conformance OK, 9 nanopubs each. Rubric not reintroduced.
- **Artifact:** `morphoeval/kg/2026-10-27-coprinopsis-cinerea-turing-wavelength-s0-audit/REPORT.md` (9,647 B, `2,588` tok, SHA-256 `76bdcec440dfb92c…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml does not exist. Nanopubs were not signed and check_signed.mjs was not run. No signature was fabricated.; Signing unavailable: ~/.nanopub/profile.yml does not exist. Nanopubs were not signed and check_signed.mjs was not run. No signature was fabricated.
- **Artifact:** `morphoeval/kg/2026-10-27-coprinopsis-cinerea-turing-wavelength-s0-audit-v2/REPORT.md` (11,734 B, `3,170` tok, SHA-256 `9247bb17831e3869…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for the Pass 2 KG. Overall 86.2, verdict faithful.; REPORT.md present for the Pass 2 KG. Overall 86.2, verdict faithful.
- **Artifact:** `morphoeval/kg/2026-10-27-coprinopsis-cinerea-turing-wavelength-s0-audit-v2/REPORT.md` (11,734 B, `3,170` tok, SHA-256 `9247bb17831e3869…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 gates met: image_fitness 83.8, CV 87.2, 12 frames, 480 px, Pearson r 0.9999, audit 86.2. Pass 3 not required.; Pass 2 gates met: image_fitness 83.8, CV 87.2, 12 frames, 480 px, Pearson r 0.9999, audit 86.2. Pass 3 not required.
- **Artifact:** `morphoeval/runs/2026-10-27-coprinopsis-cinerea-turing-wavelength-s0-loop-image-fitness.json` (2,840 B, `799` tok, SHA-256 `a727e132d4925cb6…`)

## Scientific result (Pass 2)

*Coprinopsis cinerea*, Holocene extant, 0 Ma, chronoseries 1/1. Characteristic scale 77 µm (exact 7.744e-05 m, log10 = -4.111), tier `4_cellular_10um` detail variant. There is no predecessor and no successor chronospecies. The contrast with the 10 µm tier floor is a factor of 7.744 in wavelength.

Schnakenberg kinetics on Turing's critical wavenumber, γ solved so the linear wavelength is 77.44 µm. Dispersion Pearson r = 0.9999 (RMSE 0.0155, KS 0.095). γ-sweep Pearson r = 0.9932. On-band mode 8 grows ×2.72; off-band mode 2 decays to ×0.61; equal diffusion decays to ×0.31. The saturated radial peak is 68.84 µm, an 11% nonlinear shift that is reported rather than forced back onto 77.44. Specimen is Mushroom Observer 593470 (Igor, CC BY-SA 3.0). Gill-disc circularity is 0.231 versus a perfect disc of 1.0.

Pass 2 audit overall **86.2**. Image fitness **83.8** = 0.55×87.2 + 0.30×74 + 0.15×91. GIF is 12 frames at 640×480.

Gates: image_fitness ≥ 70, CV ≥ 58, frames ≥ 6, resolution ≥ 340, a non-circular specimen metric, Pearson r > 0.98, audit overall ≥ 85. All passed. Pass 3 was not run.

Signing: `~/.nanopub/profile.yml` is absent. Nanopubs were built and passed conformance. They were not signed.
