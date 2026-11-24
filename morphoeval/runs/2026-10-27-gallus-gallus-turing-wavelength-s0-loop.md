# MorphoEval Full-Loop Telemetry — `2026-10-27-gallus-gallus-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-gallus-gallus-turing-wavelength-s0`
- **Primary paper / dataset:** Gallus gallus Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-run`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1097.21 s** (18.29 min) — Tier A: `1092.20 s` · Tier B: `5.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **27,790** (input: `0`, output: `27,790`, subagent: `0`) — Tier A: `13,319` · Tier B: `14,471`
- **Total artifact tokens produced:** `27,790`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 1088.16 | 0.00 | 0 | 1,144 | 0 | **1,144** | 1,144 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.81 | 0.00 | 0 | 2,786 | 0 | **2,786** | 2,786 | `gallus_gallus_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.81 | 0.00 | 0 | 128 | 0 | **128** | 128 | `gallus_gallus_ocellus_specimen.jpg`, `gallus_gallus_red_junglefowl_cc-by-4.0.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.81 | 0.00 | 0 | 1,087 | 0 | **1,087** | 1,087 | `Loop_3228_gallus_gallus_turing_wavelength_s0.lean`, `delphi10000-gallus-gallus-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.81 | 0.00 | 0 | 8,046 | 0 | **8,046** | 8,046 | `2026-delphi10000-gallus-gallus-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.81 | 0.00 | 0 | 128 | 0 | **128** | 128 | `gallus_gallus_turing_wavelength_s0_comparison.png`, `gallus_gallus_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.82 | 0.00 | 0 | 5,347 | 0 | **5,347** | 5,347 | `2026-10-27-gallus-gallus-turing-wavelength-s0-audit.yaml`, `2026-10-27-gallus-gallus-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.83 | 0.00 | 0 | 1,798 | 0 | **1,798** | 1,798 | `2026-10-27-gallus-gallus-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.83 | 0.00 | 0 | 2,524 | 0 | **2,524** | 2,524 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.84 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.84 | 0.00 | 0 | 4,802 | 0 | **4,802** | 4,802 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.85 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1097.21** | **0.00** | **0** | **27,790** | **0** | **27,790** | **27,790** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex plus direct DOI lookup. No collision with gallus-gallus-turing-wavelength-s0.
- **Artifact:** `datasets/gallus_gallus_turing_wavelength_s0/scout_manifest.json` (3,966 B, `1,144` tok, SHA-256 `efa51472c24d4ac2…`)

### Step 2 — `paper` (Paper)
- **Notes:** Extracted Schnakenberg onset law, 319.2 µm, age 0 Ma, chronoseries 1/1.
- **Artifact:** `datasets/gallus_gallus_turing_wavelength_s0/gallus_gallus_turing_wavelength_s0_reference.json` (4,711 B, `1,344` tok, SHA-256 `5589d95bf0b106f7…`)
- **Artifact:** `datasets/gallus_gallus_turing_wavelength_s0/paper_citations.bib` (1,689 B, `499` tok, SHA-256 `a988509e01cc5820…`)
- **Artifact:** `datasets/gallus_gallus_turing_wavelength_s0/README.md` (3,469 B, `943` tok, SHA-256 `1969fd227fc8d84c…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored gallery ocellus plate and CC BY 4.0 junglefowl photo (dhfischer, iNaturalist 477443293). SHA-256 pinned.
- **Artifact:** `datasets/gallus_gallus_turing_wavelength_s0/gallus_gallus_ocellus_specimen.jpg` (99,083 B, `64` tok, SHA-256 `7e7d58db6bb564eb…`)
- **Artifact:** `datasets/gallus_gallus_turing_wavelength_s0/gallus_gallus_red_junglefowl_cc-by-4.0.jpg` (230,711 B, `64` tok, SHA-256 `4089edd4b164a5f8…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3228: onset gap, equal-diffusion ablation, no predecessor, no successor, six-wavelength domain. Zero sorry.
- **Artifact:** `morphoeval/lean/loops/Loop_3228_gallus_gallus_turing_wavelength_s0.lean` (2,546 B, `64` tok, SHA-256 `9b66ab402f829a4b…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-gallus-gallus-turing-wavelength-s0.yaml` (3,656 B, `1,023` tok, SHA-256 `a32c207b31c2efaa…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 1 single harmonic, then pass 2 sweep, ablation, CV segmentation, held-out fit. --self-test passed.
- **Artifact:** `models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py` (28,666 B, `8,046` tok, SHA-256 `6b2d7519a5fbd93f…`)

### Step 6 — `output` (Output & compare)
- **Notes:** GIF 12 frames 800x600. Comparison PNG. Held-out Pearson r=0.9855. CV similarity 90.5.
- **Artifact:** `docs/media/gallus_gallus_turing_wavelength_s0_comparison.png` (406,896 B, `64` tok, SHA-256 `eebe3c4bcde6058f…`)
- **Artifact:** `docs/media/gallus_gallus_turing_wavelength_s0_growth.gif` (5,553,246 B, `64` tok, SHA-256 `a58a55b4d7b5a3f6…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 72.4. Pass 2 overall 86.4, supersedes pass 1.
- **Artifact:** `morphoeval/audits/2026-10-27-gallus-gallus-turing-wavelength-s0-audit.yaml` (9,814 B, `2,582` tok, SHA-256 `1b09c3f54fce730a…`)
- **Artifact:** `morphoeval/audits/2026-10-27-gallus-gallus-turing-wavelength-s0-audit-v2.yaml` (10,251 B, `2,765` tok, SHA-256 `3800ae8234014d68…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 2 quick_gate 100/100. Seven measurements, references differ from values.
- **Artifact:** `morphoeval/audits/2026-10-27-gallus-gallus-turing-wavelength-s0-audit-v2.measurements.yaml` (6,298 B, `1,798` tok, SHA-256 `567368f33ca4415e…`)

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg.py --conform. Pass 2 conformance OK. Rubric nanopub not re-emitted.
- **Artifact:** `morphoeval/kg/2026-10-27-gallus-gallus-turing-wavelength-s0-audit-v2/REPORT.md` (9,036 B, `2,524` tok, SHA-256 `8580c2538cdfd52d…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml is absent and the nanopub package is not installed. Unsigned TriG left in the kg directory. No signatures were fabricated.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for pass 1 and pass 2.
- **Artifact:** `morphoeval/kg/2026-10-27-gallus-gallus-turing-wavelength-s0-audit/REPORT.md` (8,364 B, `2,278` tok, SHA-256 `c22d5e021070fff8…`)
- **Artifact:** `morphoeval/kg/2026-10-27-gallus-gallus-turing-wavelength-s0-audit-v2/REPORT.md` (9,036 B, `2,524` tok, SHA-256 `8580c2538cdfd52d…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed R1-R5. Gates passed, so pass 3 was not run. Residuals: n=1 plate and a single assessor.

## Gate results (pass 2)

Pass 3 was not run. Every gate cleared.

| Gate | Result |
|---|---|
| image_fitness >= 70 | 87.7 |
| cv_sim_similarity >= 58 | 90.5 |
| GIF frames >= 6 | 12 |
| resolution >= 340 px | 800×600 |
| non-circular specimen metric | held-out Pearson 0.9855 vs null 0.7704; coherency 0.9584 vs plate 0.8463; painted pitch 66.4 µm vs λ_c 319.2 µm |
| Pearson r > 0.98 | 0.9855 (odd radii, three-harmonic critical mode) |
| audit overall >= 85 | 86.4 (pass 2; pass 1 was 72.4) |

Signing was not performed. `~/.nanopub/profile.yml` is absent and the `nanopub` package is not installed. The knowledge-graph TriG files are unsigned. No signature was invented.

λ_c = 319.2 µm at 0 Ma. Gallus chronoseries 1/1: no predecessor and no successor chronospecies. The painted gallery bar measures a 66.4 µm ring pitch and is not the biological wavelength.
