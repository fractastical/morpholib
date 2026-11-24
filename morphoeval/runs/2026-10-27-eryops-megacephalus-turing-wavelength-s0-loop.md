# MorphoEval Full-Loop Telemetry — `2026-10-27-eryops-megacephalus-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-eryops-megacephalus-turing-wavelength-s0`
- **Primary paper / dataset:** Eryops megacephalus Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor:** `grok-4.7`. A second assessor was not run.
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **0.99 s** (0.02 min) — Tier A: `0.48 s` · Tier B: `0.50 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **26,430** (input: `0`, output: `26,430`, subagent: `0`) — Tier A: `12,260` · Tier B: `14,170`
- **Total artifact tokens produced:** `26,430`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 1,154 | 0 | **1,154** | 1,154 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 1,213 | 0 | **1,213** | 1,213 | `eryops_megacephalus_turing_wavelength_s0_reference.json` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 64 | 0 | **64** | 64 | `eryops_megacephalus_amnh_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 64 | 0 | **64** | 64 | `Loop_3222_eryops_megacephalus_turing_wavelength_s0.lean` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 9,701 | 0 | **9,701** | 9,701 | `2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 64 | 0 | **64** | 64 | `eryops_megacephalus_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.09 | 0.00 | 0 | 3,454 | 0 | **3,454** | 3,454 | `2026-10-27-eryops-megacephalus-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.09 | 0.00 | 0 | 1,889 | 0 | **1,889** | 1,889 | `2026-10-27-eryops-megacephalus-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.08 | 0.00 | 0 | 2,822 | 0 | **2,822** | 2,822 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.08 | 0.00 | 0 | 2,822 | 0 | **2,822** | 2,822 | `REPORT.md` |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.08 | 0.00 | 0 | 2,541 | 0 | **2,541** | 2,541 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.08 | 0.00 | 0 | 642 | 0 | **642** | 642 | `2026-10-27-eryops-megacephalus-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **0.99** | **0.00** | **0** | **26,430** | **0** | **26,430** | **26,430** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex DOI resolution plus Wikimedia CC BY 4.0 AMNH candidates. No filename collision with eryops-megacephalus-turing-wavelength-s0.; OpenAlex DOI resolution plus Wikimedia CC BY 4.0 AMNH candidates. No filename collision with eryops-megacephalus-turing-wavelength-s0.
- **Artifact:** `datasets/eryops_megacephalus_turing_wavelength_s0/scout_manifest.json` (4,107 B, `1,154` tok, SHA-256 `866bb26bef8ffc37…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, BibTeX, and README state 127 µm and 295 Ma.; Reference JSON, BibTeX, and README state 127 µm and 295 Ma.
- **Artifact:** `datasets/eryops_megacephalus_turing_wavelength_s0/eryops_megacephalus_turing_wavelength_s0_reference.json` (3,884 B, `1,213` tok, SHA-256 `7254f5db446b89b1…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored Vicpeters CC BY 4.0 AMNH mount, SHA-256 pinned.; Vendored Vicpeters CC BY 4.0 AMNH mount, SHA-256 pinned.
- **Artifact:** `datasets/eryops_megacephalus_turing_wavelength_s0/eryops_megacephalus_amnh_specimen.jpg` (287,455 B, `64` tok, SHA-256 `11fc4e4eec64d5f1…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3222, five theorems, zero sorry, plus math KG.; Lean Loop_3222, five theorems, zero sorry, plus math KG.
- **Artifact:** `morphoeval/lean/loops/Loop_3222_eryops_megacephalus_turing_wavelength_s0.lean` (1,771 B, `64` tok, SHA-256 `3dd62c57d1ee8e1d…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg k_c model with self-test, sweep, ablation, and bone-masked render.; Schnakenberg k_c model with self-test, sweep, ablation, and bone-masked render.
- **Artifact:** `models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` (34,510 B, `9,701` tok, SHA-256 `a344166283e334ed…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test passed. Pearson r=0.99999998, masked CV=96.0, GIF 720x540 x 12.; Self-test passed. Pearson r=0.99999998, masked CV=96.0, GIF 720x540 x 12.
- **Artifact:** `docs/media/eryops_megacephalus_turing_wavelength_s0_growth.gif` (2,794,928 B, `64` tok, SHA-256 `b226d43e5bb86317…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 audit 84.3 with open remediations; Pass 2 audit 90.8 closes them.; Pass 1 audit 84.3 with open remediations; Pass 2 audit 90.8 closes them.
- **Artifact:** `morphoeval/audits/2026-10-27-eryops-megacephalus-turing-wavelength-s0-audit-v2.yaml` (13,318 B, `3,454` tok, SHA-256 `d2be5cf90538a596…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100/100 on both audits. Non-circular specimen metrics retained.; measure_outputs quick_gate 100/100 on both audits. Non-circular specimen metrics retained.
- **Artifact:** `morphoeval/audits/2026-10-27-eryops-megacephalus-turing-wavelength-s0-audit-v2.measurements.yaml` (6,685 B, `1,889` tok, SHA-256 `c5e99029ed3258d7…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Knowledge graphs conform. Rubric nanopub reused from the published Bos taurus rubric.; Knowledge graphs conform. Rubric nanopub reused from the published Bos taurus rubric.
- **Artifact:** `morphoeval/kg/2026-10-27-eryops-megacephalus-turing-wavelength-s0-audit-v2/REPORT.md` (10,184 B, `2,822` tok, SHA-256 `76861470a1579f9f…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing skipped honestly: ~/.nanopub/profile.yml does not exist.; Signing skipped honestly: ~/.nanopub/profile.yml does not exist.
- **Artifact:** `morphoeval/kg/2026-10-27-eryops-megacephalus-turing-wavelength-s0-audit-v2/REPORT.md` (10,184 B, `2,822` tok, SHA-256 `76861470a1579f9f…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md overall 90.8 faithful for Pass 2 and the Pass 1 report is present.; REPORT.md overall 90.8 faithful for Pass 2 and the Pass 1 report is present.
- **Artifact:** `morphoeval/kg/2026-10-27-eryops-megacephalus-turing-wavelength-s0-audit/REPORT.md` (9,156 B, `2,541` tok, SHA-256 `a9cb2963da057b38…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** image_fitness 93.0, cv_sim_similarity 96.0, 12 frames, 720x540. Gates passed on Pass 2. No Pass 3.; image_fitness 93.0, cv_sim_similarity 96.0, 12 frames, 720x540. Gates passed on Pass 2. No Pass 3.
- **Artifact:** `morphoeval/runs/2026-10-27-eryops-megacephalus-turing-wavelength-s0-loop-image-fitness.json` (2,104 B, `642` tok, SHA-256 `f49248d12087383c…`)

## Result

*Eryops megacephalus*, Early Permian, 295 Ma, detail-scale Turing wavelength 127 µm. The Schnakenberg critical mode is 12 pixels on a 1.016 mm patch. Dispersion Pearson r on Fourier modes m = 2..16 is 0.99999998 (RMSE 1.14e-4). Equal diffusion and the autocatalysis ablation both decay. The CC BY 4.0 AMNH mount is the specimen. Pass 1 painted the whole frame (cv_sim_similarity 94.4) and scored 84.3. Pass 2 restricts the ocelli to the eroded bone mask (cv_sim_similarity 96.0) and scores 90.8.

Gates after Pass 2: image_fitness 93.0, cv_sim_similarity 96.0, 12 frames, 720×540, non-circular specimen metrics (foreground 0.3208 vs 0.5, aspect 1.3358 vs 1, coherency 0.5334 vs 0.25), Pearson r 0.99999998, audit overall 90.8. Pass 3 was not required.

Nanopubs were not signed. `~/.nanopub/profile.yml` is not on this machine. The telemetry wall-clock above is the logging pass, not the solver time.

The comparison PNG is force-added. `*.png` is gitignored; this file is one of the required loop outputs.
