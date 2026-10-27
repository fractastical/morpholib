# MorphoEval Full-Loop Telemetry — `2026-10-27-eusthenopteron-foordi-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-eusthenopteron-foordi-turing-wavelength-s0`
- **Primary paper / dataset:** Eusthenopteron foordi Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **361.46 s** (6.02 min) — Tier A: `360.86 s` · Tier B: `0.60 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **22,944** (input: `0`, output: `22,944`, subagent: `0`) — Tier A: `12,026` · Tier B: `10,918`
- **Total artifact tokens produced:** `22,944`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 718 | 0 | **718** | 718 | `scout_manifest.json`, `eusthenopteron_foordi_specimen.jpg` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.07 | 0.00 | 0 | 2,606 | 0 | **2,606** | 2,606 | `eusthenopteron_foordi_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.07 | 0.00 | 0 | 64 | 0 | **64** | 64 | `eusthenopteron_foordi_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 1,072 | 0 | **1,072** | 1,072 | `Loop_3225_eusthenopteron_foordi_turing_wavelength_s0.lean`, `delphi10000-eusthenopteron-foordi-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 360.47 | 0.00 | 0 | 7,438 | 0 | **7,438** | 7,438 | `2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 128 | 0 | **128** | 128 | `eusthenopteron_foordi_turing_wavelength_s0_comparison.png`, `eusthenopteron_foordi_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.08 | 0.00 | 0 | 4,127 | 0 | **4,127** | 4,127 | `2026-10-27-eusthenopteron-foordi-turing-wavelength-s0-audit.yaml`, `2026-10-27-eusthenopteron-foordi-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.08 | 0.00 | 0 | 1,657 | 0 | **1,657** | 1,657 | `2026-10-27-eusthenopteron-foordi-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.08 | 0.00 | 0 | 2,324 | 0 | **2,324** | 2,324 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.21 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.08 | 0.00 | 0 | 1,914 | 0 | **1,914** | 1,914 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.08 | 0.00 | 0 | 896 | 0 | **896** | 896 | `2026-10-27-eusthenopteron-foordi-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **361.46** | **0.00** | **0** | **22,944** | **0** | **22,944** | **22,944** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia query; slot 3225 is outside campaign_100
- **Artifact:** `datasets/eusthenopteron_foordi_turing_wavelength_s0/scout_manifest.json` (2,322 B, `654` tok, SHA-256 `823594066b7c6906…`)
- **Artifact:** `datasets/eusthenopteron_foordi_turing_wavelength_s0/eusthenopteron_foordi_specimen.jpg` (377,137 B, `64` tok, SHA-256 `ad5986d4a4f2a605…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, bib, README state 104 µm and 379 Ma
- **Artifact:** `datasets/eusthenopteron_foordi_turing_wavelength_s0/eusthenopteron_foordi_turing_wavelength_s0_reference.json` (4,817 B, `1,261` tok, SHA-256 `aabe83196459146e…`)
- **Artifact:** `datasets/eusthenopteron_foordi_turing_wavelength_s0/paper_citations.bib` (2,308 B, `716` tok, SHA-256 `229b640b1162aa64…`)
- **Artifact:** `datasets/eusthenopteron_foordi_turing_wavelength_s0/README.md` (2,342 B, `629` tok, SHA-256 `af0b0e92319c441e…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored CC BY-SA 3.0 Brussels mount
- **Artifact:** `datasets/eusthenopteron_foordi_turing_wavelength_s0/eusthenopteron_foordi_specimen.jpg` (377,137 B, `64` tok, SHA-256 `ad5986d4a4f2a605…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3225 and math KG
- **Artifact:** `morphoeval/lean/loops/Loop_3225_eusthenopteron_foordi_turing_wavelength_s0.lean` (1,770 B, `64` tok, SHA-256 `53c3f5d9d8505d8e…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-eusthenopteron-foordi-turing-wavelength-s0.yaml` (3,698 B, `1,008` tok, SHA-256 `f22d92fd2274d8f8…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 1+2 Schnakenberg k_c model
- **Artifact:** `models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` (25,571 B, `7,438` tok, SHA-256 `9f9c9772b56b5e59…`)

### Step 6 — `output` (Output & compare)
- **Notes:** self-test passed; GIF 10 frames 974x480
- **Artifact:** `docs/media/eusthenopteron_foordi_turing_wavelength_s0_comparison.png` (631,334 B, `64` tok, SHA-256 `721d8d4672825d00…`)
- **Artifact:** `docs/media/eusthenopteron_foordi_turing_wavelength_s0_growth.gif` (2,695,211 B, `64` tok, SHA-256 `960a6f94384c58dc…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 81.2; Pass 2 overall 86.1
- **Artifact:** `morphoeval/audits/2026-10-27-eusthenopteron-foordi-turing-wavelength-s0-audit.yaml` (7,054 B, `1,777` tok, SHA-256 `a83d5faa0853473c…`)
- **Artifact:** `morphoeval/audits/2026-10-27-eusthenopteron-foordi-turing-wavelength-s0-audit-v2.yaml` (8,780 B, `2,350` tok, SHA-256 `a80e8dd0b0b45ca4…`)

### Step 8 — `measure` (Measure)
- **Artifact:** `morphoeval/audits/2026-10-27-eusthenopteron-foordi-turing-wavelength-s0-audit-v2.measurements.yaml` (6,006 B, `1,657` tok, SHA-256 `82a63869536051f7…`)

### Step 9 — `build` (Write as claims)
- **Notes:** conform OK both passes; rubric trig is a shared reintroduction
- **Artifact:** `morphoeval/kg/2026-10-27-eusthenopteron-foordi-turing-wavelength-s0-audit-v2/REPORT.md` (8,197 B, `2,324` tok, SHA-256 `d6e4e231a1fb376d…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** signing unavailable; No ~/.nanopub/profile.yml; unsigned trig only, not faked

### Step 11 — `graph` (Public record)
- **Artifact:** `morphoeval/kg/2026-10-27-eusthenopteron-foordi-turing-wavelength-s0-audit/REPORT.md` (7,152 B, `1,914` tok, SHA-256 `e246fbb47c286c60…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed sweep, ablation, segmentation, GOF; gates pass; no pass 3
- **Artifact:** `morphoeval/runs/2026-10-27-eusthenopteron-foordi-turing-wavelength-s0-loop-image-fitness.json` (3,032 B, `896` tok, SHA-256 `91235957ab5feed4…`)

## Result

*Eusthenopteron foordi*, Frasnian **379 Ma**, chronoseries 1/1, Turing wavenumber at **104 µm** (0.0001042 m). Schnakenberg kinetics with \(k_c = 2\pi\) per 104.2 µm. Realized FFT wavelength 102.01 µm. Dispersion Pearson r = 0.9999998, RMSE = 0.00261. The Brussels CC BY-SA 3.0 mount is the specimen. At fixed \(k_c\), the registry successor *Panderichthys rhombolepis* (0.165 m) packs about 240 more wavelengths than the 0.140 m *Eusthenopteron* length.

Pass 1 audit overall **81.2**. Pass 2 audit overall **86.1** after crediting the Dv/Du sweep (Spearman 0.90, not strictly monotone), equal-diffusion and autocatalysis ablations, fossil-mask segmentation, and the dispersion / KS goodness-of-fit.

Image fitness (Pass 2): **90.4** = 0.55×95.7 + 0.30×82 + 0.15×88. GIF 10 frames, 974×480. Every gate passed (fitness ≥ 70, CV ≥ 58, frames ≥ 6, resolution ≥ 340, non-circular specimen metrics, Pearson r > 0.98, audit ≥ 85). Pass 3 was not run.

Signing was not performed. `nanopub` is not installed and `~/.nanopub/profile.yml` is absent. The knowledge-graph trig files are unsigned; conformance of the unsigned nanopubs passed. No signature was invented.

The audit assessor is Grok 4.7, a single builder-assessor. The telemetry header names are the track_loop defaults.
