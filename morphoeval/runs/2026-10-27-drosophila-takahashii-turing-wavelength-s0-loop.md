# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-takahashii-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-takahashii-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila takahashii Turing Reaction-Diffusion Wavelength Selection
- **Builder / assessor:** Grok 4.7, single agent. The model names in the step table are track_loop defaults, not a second assessor.
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.12 s** (0.02 min) — Tier A: `0.57 s` · Tier B: `0.55 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **26,977** (input: `0`, output: `26,977`, subagent: `0`) — Tier A: `14,653` · Tier B: `12,324`
- **Total artifact tokens produced:** `26,977`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 0.11 | 0.00 | 0 | 1,124 | 0 | **1,124** | 1,124 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.10 | 0.00 | 0 | 2,772 | 0 | **2,772** | 2,772 | `drosophila_takahashii_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 64 | 0 | **64** | 64 | `Drosophila_takahashii_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 1,060 | 0 | **1,060** | 1,060 | `Loop_3208_drosophila_takahashii_turing_wavelength_s0.lean`, `delphi10000-drosophila-takahashii-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 9,505 | 0 | **9,505** | 9,505 | `2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_takahashii_turing_wavelength_s0_comparison.png`, `drosophila_takahashii_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.09 | 0.00 | 0 | 5,656 | 0 | **5,656** | 5,656 | `2026-10-27-drosophila-takahashii-turing-wavelength-s0-audit.yaml`, `2026-10-27-drosophila-takahashii-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.09 | 0.00 | 0 | 3,959 | 0 | **3,959** | 3,959 | `2026-10-27-drosophila-takahashii-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-drosophila-takahashii-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.09 | 0.00 | 0 | 2,709 | 0 | **2,709** | 2,709 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1.12** | **0.00** | **0** | **26,977** | **0** | **26,977** | **26,977** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex DOIs plus Commons; slot 3208 is outside campaign_100; No s0 collision. Rejected D. pseudotakahashii.
- **Artifact:** `datasets/drosophila_takahashii_turing_wavelength_s0/scout_manifest.json` (4,161 B, `1,124` tok, SHA-256 `9b6681d3a2e6cbf4…`)

### Step 2 — `paper` (Paper)
- **Artifact:** `datasets/drosophila_takahashii_turing_wavelength_s0/drosophila_takahashii_turing_wavelength_s0_reference.json` (5,491 B, `1,416` tok, SHA-256 `c96d7a5d72c78239…`)
- **Artifact:** `datasets/drosophila_takahashii_turing_wavelength_s0/paper_citations.bib` (2,967 B, `891` tok, SHA-256 `101832d102fd3942…`)
- **Artifact:** `datasets/drosophila_takahashii_turing_wavelength_s0/README.md` (1,747 B, `465` tok, SHA-256 `f7b22cc6970b19de…`)

### Step 3 — `source` (Formal source)
- **Notes:** Gallery fallback SHA-256 be067c44d300a455
- **Artifact:** `datasets/drosophila_takahashii_turing_wavelength_s0/Drosophila_takahashii_specimen.jpg` (96,574 B, `64` tok, SHA-256 `be067c44d300a455…`)

### Step 4 — `rules` (Pin the rules)
- **Artifact:** `morphoeval/lean/loops/Loop_3208_drosophila_takahashii_turing_wavelength_s0.lean` (1,684 B, `64` tok, SHA-256 `fe2bb87da79a627d…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-drosophila-takahashii-turing-wavelength-s0.yaml` (3,678 B, `996` tok, SHA-256 `00cf52f4cb3ac28f…`)

### Step 5 — `script` (Script)
- **Artifact:** `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` (34,578 B, `9,505` tok, SHA-256 `04341dcd8be6f91c…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 2 composite GIF 640x480 x12, CV 93.5
- **Artifact:** `docs/media/drosophila_takahashii_turing_wavelength_s0_comparison.png` (473,535 B, `64` tok, SHA-256 `9d1909ffe0bdfb85…`)
- **Artifact:** `docs/media/drosophila_takahashii_turing_wavelength_s0_growth.gif` (3,503,941 B, `64` tok, SHA-256 `8bd6293c55579c69…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 77.0; Pass 2 overall 88.5
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-takahashii-turing-wavelength-s0-audit.yaml` (10,781 B, `2,772` tok, SHA-256 `fd71c8b52c7873cc…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-takahashii-turing-wavelength-s0-audit-v2.yaml` (11,036 B, `2,884` tok, SHA-256 `7b21bf515df0071a…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pearson r 0.9905; quick_gate 100
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-takahashii-turing-wavelength-s0-audit.measurements.yaml` (7,121 B, `1,965` tok, SHA-256 `12afc4ad6a66bb17…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-takahashii-turing-wavelength-s0-audit-v2.measurements.yaml` (7,250 B, `1,994` tok, SHA-256 `97f5f0e76d152c31…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Both audits conform. Rubric nanopub not reintroduced.

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml missing. Nanopubs left unsigned.

### Step 11 — `graph` (Public record)
- **Notes:** Pass 2 REPORT overall 88.5
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-takahashii-turing-wavelength-s0-audit-v2/REPORT.md` (9,820 B, `2,709` tok, SHA-256 `9a68166ebeaad868…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Fixed fragmented Otsu contour and mask-only motion. Pass 3 not required.

## Result (Pass 2)

*Drosophila takahashii*, Holocene 0 Ma, detail scale 2.4 µm (2.362e-06 m). Schnakenberg Turing wavelength on a 28.344 µm chart. Predecessor *D. suzukii* envelope 98.45 µm (41 detail wavelengths). Successor *D. teissieri* envelope 107 µm (8550 nm longer; the detail wavelength is not rescaled).

Pass 1 field render scored 77.0. Raw Otsu circularity was 0.0018 and a mask-only blend did not move the frame. Pass 2 uses a smoothed body component (circularity 0.307) and a full-frame crossfade.

Gates, all met:

- image_fitness 91.3 (threshold 70)
- cv_sim_similarity 93.5 (threshold 58)
- GIF 12 frames, 640×480
- Pearson r 0.9905
- audit overall 88.5
- non-circular specimen metrics include body area 0.053 vs 0.099 and coherency 0.269 vs 0.160

Signing was not done. `~/.nanopub/profile.yml` is absent, so the nanopubs are conformance-checked and unsigned. Pass 3 was not required.
