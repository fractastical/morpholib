# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-paulistorum-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-paulistorum-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila paulistorum Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor:** Grok 4.7 scored both passes in this run. The tracker defaults still name `gemini-4-argon` and `claude-sonnet-5-5-high`; those models were not called.
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **920.49 s** (15.34 min) — Tier A: `915.46 s` · Tier B: `5.03 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **36,226** (input: `0`, output: `36,226`, subagent: `0`) — Tier A: `17,159` · Tier B: `19,067`
- **Total artifact tokens produced:** `36,226`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 911.43 | 0.00 | 0 | 1,800 | 0 | **1,800** | 1,800 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.80 | 0.00 | 0 | 2,990 | 0 | **2,990** | 2,990 | `drosophila_paulistorum_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.80 | 0.00 | 0 | 64 | 0 | **64** | 64 | `drosophila_paulistorum_male_haars_2018.png` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.81 | 0.00 | 0 | 1,255 | 0 | **1,255** | 1,255 | `Loop_3197_drosophila_paulistorum_turing_wavelength_s0.lean`, `delphi10000-drosophila-paulistorum-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.81 | 0.00 | 0 | 10,922 | 0 | **10,922** | 10,922 | `2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.81 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_paulistorum_turing_wavelength_s0_comparison.png`, `drosophila_paulistorum_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.82 | 0.00 | 0 | 6,283 | 0 | **6,283** | 6,283 | `2026-10-27-drosophila-paulistorum-turing-wavelength-s0-audit.yaml`, `2026-10-27-drosophila-paulistorum-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.83 | 0.00 | 0 | 3,550 | 0 | **3,550** | 3,550 | `2026-10-27-drosophila-paulistorum-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-drosophila-paulistorum-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.83 | 0.00 | 0 | 2,964 | 0 | **2,964** | 2,964 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.84 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.85 | 0.00 | 0 | 5,604 | 0 | **5,604** | 5,604 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.85 | 0.00 | 0 | 666 | 0 | **666** | 666 | `2026-10-27-drosophila-paulistorum-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **920.49** | **0.00** | **0** | **36,226** | **0** | **36,226** | **36,226** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex DOI resolution plus Wikimedia CC BY-SA male. No s0 filename collision. Slot 3197 is outside campaign_100_slots.yaml.
- **Artifact:** `datasets/drosophila_paulistorum_turing_wavelength_s0/scout_manifest.json` (6,522 B, `1,800` tok, SHA-256 `a859ae40d3f79e52…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON states 2.417e-6 m, log10 -5.617, age 0 Ma, orena and persimilis.
- **Artifact:** `datasets/drosophila_paulistorum_turing_wavelength_s0/drosophila_paulistorum_turing_wavelength_s0_reference.json` (5,118 B, `1,413` tok, SHA-256 `39b5153e4682ffee…`)
- **Artifact:** `datasets/drosophila_paulistorum_turing_wavelength_s0/paper_citations.bib` (2,358 B, `752` tok, SHA-256 `5c00ca3d9a631902…`)
- **Artifact:** `datasets/drosophila_paulistorum_turing_wavelength_s0/README.md` (2,969 B, `825` tok, SHA-256 `b770bcff5173b0a0…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored Jonathan Haars 2018-11-29 CC BY-SA 4.0 male. SHA-256 e01e60e633f99ef12a7474a79f5afaaa56b1502bc76590c4102723cf5cf7d1ae. Gallery spiral not used.
- **Artifact:** `datasets/drosophila_paulistorum_turing_wavelength_s0/drosophila_paulistorum_male_haars_2018.png` (981,339 B, `64` tok, SHA-256 `e01e60e633f99ef1…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Six Lean theorems, no unfinished proofs. Trace, determinant, k numerator, orena packing 37 rem 11, persimilis packing 48 rem 684, six waves tile 14502 nm.
- **Artifact:** `morphoeval/lean/loops/Loop_3197_drosophila_paulistorum_turing_wavelength_s0.lean` (2,307 B, `64` tok, SHA-256 `b38dd186749e9c22…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-drosophila-paulistorum-turing-wavelength-s0.yaml` (4,409 B, `1,191` tok, SHA-256 `dcc9fb94c9a6f629…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg integrator, sigma-peak placement at 2.417 um, Du sweep, equal-diffusion ablation, specimen segmentation, measure_morphoeval.
- **Artifact:** `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` (39,756 B, `10,922` tok, SHA-256 `8e510cc9f27013af…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test passed. GIF 12 frames 640x480. Nominal wavelength 2.417 um. Nonlinear Pearson r 0.989. RMSE 0.064 um.
- **Artifact:** `docs/media/drosophila_paulistorum_turing_wavelength_s0_comparison.png` (435,973 B, `64` tok, SHA-256 `26d4992383a6d270…`)
- **Artifact:** `docs/media/drosophila_paulistorum_turing_wavelength_s0_growth.gif` (2,776,691 B, `64` tok, SHA-256 `ac4e45110afd8d24…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 79.4. Nonlinear r was 0.877 because sigma peaked near 3.5 um. Pass 2 overall 88.3 after diffusivity magnitude 0.478965.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-paulistorum-turing-wavelength-s0-audit.yaml` (11,055 B, `2,889` tok, SHA-256 `4bff1fb2e7a67a7c…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-paulistorum-turing-wavelength-s0-audit-v2.yaml` (12,893 B, `3,394` tok, SHA-256 `1f7cb10432cf67bb…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100/100. Ten non-circular metrics including specimen foreground 0.2995 vs prior 0.22.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-paulistorum-turing-wavelength-s0-audit.measurements.yaml` (6,157 B, `1,821` tok, SHA-256 `79a9d14f0ae142d1…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-paulistorum-turing-wavelength-s0-audit-v2.measurements.yaml` (6,030 B, `1,729` tok, SHA-256 `868352dd8adf9592…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Both audits conform to ProbKnow. Pass 2 REPORT overall 88.3. --supersede-from skipped because Pass 1 was unsigned and has no published.json. Supersession is in the audit id chain.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-paulistorum-turing-wavelength-s0-audit-v2/REPORT.md` (10,464 B, `2,964` tok, SHA-256 `1d6dec87a83107be…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable. ~/.nanopub/profile.yml does not exist. Nanopubs were not signed and no signature was fabricated. publisher URI is https://github.com/fractastical, not an ORCID.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for Pass 1 and Pass 2. Pass 2 overall 88.3, verdict faithful.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-paulistorum-turing-wavelength-s0-audit/REPORT.md` (9,534 B, `2,640` tok, SHA-256 `365915f7615e8d89…`)
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-paulistorum-turing-wavelength-s0-audit-v2/REPORT.md` (10,464 B, `2,964` tok, SHA-256 `1d6dec87a83107be…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 fixed the growth-rate peak onto 2.417 um. Nonlinear r 0.989, CV 85.9, image fitness 85.9, 12 frames, 640x480, four non-circular specimen metrics. All gates passed. No Pass 3.
- **Artifact:** `morphoeval/runs/2026-10-27-drosophila-paulistorum-turing-wavelength-s0-loop-image-fitness.json` (2,141 B, `666` tok, SHA-256 `88e369f7f0b2fa28…`)

## Result

*Drosophila paulistorum*, Holocene, **0 Ma**, detail rung **2.4 µm** (2.417×10⁻⁶ m, log10 = −5.617, `3_subcellular_um`). Catalog predecessor *D. orena* (89.44 µm, 37 packings, remainder 11 nm). Catalog successor *D. persimilis* (116.7 µm, 48 packings, remainder 684 nm). All three are extant; this is not a stratigraphic lineage.

Pass 1 pinned the classical determinant wavenumber at 2.417 µm while σ(k) peaked near 3.5 µm, so the nonlinear field correlation was **r = 0.877**. Pass 2 multiplies both diffusivities by 0.478965 so the maximum growth rate sits at 2.417 µm. No Pass 3.

| Gate | Result |
|---|---|
| image_fitness ≥ 70 | **85.9** (0.55×85.9 + 0.30×84 + 0.15×90) |
| cv_sim_similarity ≥ 58 | **85.9** |
| GIF frames ≥ 6 | **12** |
| resolution ≥ 340 px | **640×480** |
| non-circular specimen metric | **4** (foreground 0.2995 vs 0.22, coherency 0.632 vs 0.45, circularity 0.597 vs 0.72, wavelength 2.41732 vs 2.417) |
| Pearson r > 0.98 | **0.989** (nonlinear FFT vs maximum-growth λ; RMSE 0.064 µm) |
| audit overall ≥ 85 | **88.3** (Pass 2; Pass 1 was 79.4) |

Specimen: Jonathan Haars, 29 Nov 2018, CC BY-SA 4.0, SHA-256 `e01e60e633f99ef12a7474a79f5afaaa56b1502bc76590c4102723cf5cf7d1ae`. Signing: no `~/.nanopub/profile.yml`; nanopubs are unsigned. ProbKnow conformance passed for both audits.
