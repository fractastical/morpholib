# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-willistoni-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-willistoni-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila willistoni Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.05 s** (0.02 min) — Tier A: `0.51 s` · Tier B: `0.54 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **33,307** (input: `0`, output: `33,307`, subagent: `0`) — Tier A: `15,368` · Tier B: `17,939`
- **Total artifact tokens produced:** `33,307`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,142 | 0 | **1,142** | 1,142 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 3,682 | 0 | **3,682** | 3,682 | `drosophila_willistoni_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_willistoni_turing_wavelength_s0_specimen.png`, `drosophila_willistoni_turing_wavelength_s0_specimen_source.tif` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,173 | 0 | **1,173** | 1,173 | `Loop_3211_drosophila_willistoni_turing_wavelength_s0.lean`, `delphi10000-drosophila-willistoni-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 9,115 | 0 | **9,115** | 9,115 | `2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_willistoni_turing_wavelength_s0_comparison.png`, `drosophila_willistoni_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 3,111 | 0 | **3,111** | 3,111 | `2026-10-27-drosophila-willistoni-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,787 | 0 | **1,787** | 1,787 | `2026-10-27-drosophila-willistoni-turing-wavelength-s0-audit.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,819 | 0 | **2,819** | 2,819 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,819 | 0 | **2,819** | 2,819 | `REPORT.md` |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 3,176 | 0 | **3,176** | 3,176 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 4,227 | 0 | **4,227** | 4,227 | `2026-10-27-drosophila-willistoni-turing-wavelength-s0-audit-v2.yaml`, `2026-10-27-drosophila-willistoni-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1.05** | **0.00** | **0** | **33,307** | **0** | **33,307** | **33,307** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia. Slot 3211 is outside campaign_100_slots.yaml. No collision with drosophila-willistoni-turing-wavelength-s0.; OpenAlex/Crossref/Wikimedia. Slot 3211 is outside campaign_100_slots.yaml. No collision with drosophila-willistoni-turing-wavelength-s0.
- **Artifact:** `datasets/drosophila_willistoni_turing_wavelength_s0/scout_manifest.json` (4,403 B, `1,142` tok, SHA-256 `9b5d43aee23e54ee…`)

### Step 2 — `paper` (Paper)
- **Notes:** Extracted Schnakenberg kinetics, onset k_c, fastest-mode wavelength 1.466 µm, age 0 Ma.; Extracted Schnakenberg kinetics, onset k_c, fastest-mode wavelength 1.466 µm, age 0 Ma.
- **Artifact:** `datasets/drosophila_willistoni_turing_wavelength_s0/drosophila_willistoni_turing_wavelength_s0_reference.json` (5,991 B, `1,777` tok, SHA-256 `9669b9e9c9b9346c…`)
- **Artifact:** `datasets/drosophila_willistoni_turing_wavelength_s0/paper_citations.bib` (3,795 B, `1,178` tok, SHA-256 `f28ff9b6126a1f2f…`)
- **Artifact:** `datasets/drosophila_willistoni_turing_wavelength_s0/README.md` (2,822 B, `727` tok, SHA-256 `476aeea4b61b0482…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored MA Hanson CC BY-SA 4.0 File:Dwil_1.tif and the RGB PNG, SHA-256 pinned.; Vendored MA Hanson CC BY-SA 4.0 File:Dwil_1.tif and the RGB PNG, SHA-256 pinned.
- **Artifact:** `datasets/drosophila_willistoni_turing_wavelength_s0/drosophila_willistoni_turing_wavelength_s0_specimen.png` (534,219 B, `64` tok, SHA-256 `6a3af6f8dcdfabbe…`)
- **Artifact:** `datasets/drosophila_willistoni_turing_wavelength_s0/drosophila_willistoni_turing_wavelength_s0_specimen_source.tif` (1,856,066 B, `64` tok, SHA-256 `8e47bb5c8a0f7972…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3211: wavelength order, granule order, onset margin, ablation. No sorry.; Lean Loop_3211: wavelength order, granule order, onset margin, ablation. No sorry.
- **Artifact:** `morphoeval/lean/loops/Loop_3211_drosophila_willistoni_turing_wavelength_s0.lean` (2,406 B, `64` tok, SHA-256 `5a8ca904e9b06573…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-drosophila-willistoni-turing-wavelength-s0.yaml` (3,812 B, `1,109` tok, SHA-256 `b12f3c1ca31101f2…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg solver, sweep, ablation, specimen segmentation, --self-test passed.; Schnakenberg solver, sweep, ablation, specimen segmentation, --self-test passed.
- **Artifact:** `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` (32,659 B, `9,115` tok, SHA-256 `989497e49f7d0872…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Comparison PNG and 12-frame 720x540 growth GIF.; Comparison PNG and 12-frame 720x540 growth GIF.
- **Artifact:** `docs/media/drosophila_willistoni_turing_wavelength_s0_comparison.png` (447,136 B, `64` tok, SHA-256 `f0b879c8a574ac06…`)
- **Artifact:** `docs/media/drosophila_willistoni_turing_wavelength_s0_growth.gif` (1,833,875 B, `64` tok, SHA-256 `481ccd337d573ff3…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 77.1. Remediations: sweep, ablation, specimen CV, goodness-of-fit.; Pass 1 overall 77.1. Remediations: sweep, ablation, specimen CV, goodness-of-fit.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-willistoni-turing-wavelength-s0-audit.yaml` (12,709 B, `3,111` tok, SHA-256 `e124ee58e549aefa…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 1 sidecar quick_gate 100/100. 12 measurements, 2 artifacts.; Pass 1 sidecar quick_gate 100/100. 12 measurements, 2 artifacts.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-willistoni-turing-wavelength-s0-audit.measurements.yaml` (6,651 B, `1,787` tok, SHA-256 `9d36a559869ba7eb…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Pass 1 KG conformance OK. Shared rubric not re-emitted.; Pass 1 KG conformance OK. Shared rubric not re-emitted.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-willistoni-turing-wavelength-s0-audit/REPORT.md` (10,842 B, `2,819` tok, SHA-256 `a262f517f7e8ad30…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no ~/.nanopub/profile.yml. Nanopubs were not signed and signatures were not fabricated.; Signing unavailable: no ~/.nanopub/profile.yml. Nanopubs were not signed and signatures were not fabricated.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-willistoni-turing-wavelength-s0-audit/REPORT.md` (10,842 B, `2,819` tok, SHA-256 `a262f517f7e8ad30…`)

### Step 11 — `graph` (Public record)
- **Notes:** Pass 1 REPORT.md overall 77.1. Pass 2 REPORT.md overall 86.9.; Pass 1 REPORT.md overall 77.1. Pass 2 REPORT.md overall 86.9.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-willistoni-turing-wavelength-s0-audit-v2/REPORT.md` (12,203 B, `3,176` tok, SHA-256 `1459d99f60eb445a…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 resolved sweep, ablation, specimen CV, and GOF. image_fitness 91.2, cv 96.0, Pearson r 1.0, audit 86.9. Pass 3 not required.; Pass 2 resolved sweep, ablation, specimen CV, and GOF. image_fitness 91.2, cv 96.0, Pearson r 1.0, audit 86.9. Pass 3 not required.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-willistoni-turing-wavelength-s0-audit-v2.yaml` (14,739 B, `3,611` tok, SHA-256 `d4cf6d75ebb46049…`)
- **Artifact:** `morphoeval/runs/2026-10-27-drosophila-willistoni-turing-wavelength-s0-loop-image-fitness.json` (2,227 B, `616` tok, SHA-256 `4dfe97b59662055d…`)

## Result

Pass 2 cleared every gate. Pass 3 was not run.

The scored audit assessor is Grok 4.7. The header above still names the telemetry defaults.

- Scale 1.5 µm (1.466e-6 m), age 0 Ma.
- Predecessor *Drosophila virilis*: 1.676 µm, 7.40 granules across the cell.
- Focal *Drosophila willistoni*: 1.466 µm, 5.77 granules.
- Successor *Drosophila yakuba*: 1.509 µm, 6.09 granules.
- Kinetics and diffusivity ratio stay fixed. Absolute diffusivities scale as λ², and λ scales as body length^0.35.
- Wavelength-sweep Pearson r = 1.0 (gate is r > 0.98). RMSE is roundoff. The noise-seeded field measures 1.480 µm against the 1.466 µm reference.
- Abdomen-pixel Pearson r = 0.925 is a separate comparison and is not the law-fit statistic.
- image_fitness = 91.2, cv_sim_similarity = 96.0, GIF 12 frames at 720×540.
- Pass 1 overall 77.1. Pass 2 overall 86.9.
- Nanopub signing was not available (`~/.nanopub/profile.yml` is absent). The graphs were conformance-checked and left unsigned.
