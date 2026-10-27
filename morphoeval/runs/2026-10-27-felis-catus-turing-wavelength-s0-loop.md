# MorphoEval Full-Loop Telemetry — `2026-10-27-felis-catus-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-felis-catus-turing-wavelength-s0`
- **Primary paper / dataset:** Felis catus Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor:** `grok-4.7` (single builder-assessor; no second assessor was run)
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **963.45 s** (16.06 min) — Tier A: `958.37 s` · Tier B: `5.08 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **30,499** (input: `0`, output: `30,499`, subagent: `0`) — Tier A: `14,426` · Tier B: `16,073`
- **Total artifact tokens produced:** `30,499`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 954.19 | 0.00 | 0 | 1,467 | 0 | **1,467** | 1,467 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.83 | 0.00 | 0 | 2,726 | 0 | **2,726** | 2,726 | `felis_catus_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.83 | 0.00 | 0 | 64 | 0 | **64** | 64 | `felis_catus_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.84 | 0.00 | 0 | 793 | 0 | **793** | 793 | `Loop_3226_felis_catus_turing_wavelength_s0.lean`, `delphi10000-felis-catus-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.84 | 0.00 | 0 | 9,248 | 0 | **9,248** | 9,248 | `2026-delphi10000-felis-catus-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.84 | 0.00 | 0 | 128 | 0 | **128** | 128 | `felis_catus_turing_wavelength_s0_comparison.png`, `felis_catus_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.84 | 0.00 | 0 | 5,309 | 0 | **5,309** | 5,309 | `2026-10-27-felis-catus-turing-wavelength-s0-audit.yaml`, `2026-10-27-felis-catus-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.85 | 0.00 | 0 | 2,001 | 0 | **2,001** | 2,001 | `2026-10-27-felis-catus-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.84 | 0.00 | 0 | 2,806 | 0 | **2,806** | 2,806 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.85 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.85 | 0.00 | 0 | 5,076 | 0 | **5,076** | 5,076 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.86 | 0.00 | 0 | 881 | 0 | **881** | 881 | `2026-10-27-felis-catus-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **963.45** | **0.00** | **0** | **30,499** | **0** | **30,499** | **30,499** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia. No felis-catus-turing-wavelength-s0 collision. Sibling felis_catus_turing_wavelength left untouched.
- **Artifact:** `datasets/felis_catus_turing_wavelength_s0/scout_manifest.json` (5,026 B, `1,467` tok, SHA-256 `4abea4f5ffa93d84…`)

### Step 2 — `paper` (Paper)
- **Notes:** Equations, 436 µm, 0 Ma, claim boundary.
- **Artifact:** `datasets/felis_catus_turing_wavelength_s0/felis_catus_turing_wavelength_s0_reference.json` (3,985 B, `1,124` tok, SHA-256 `a7694c1c70d27f08…`)
- **Artifact:** `datasets/felis_catus_turing_wavelength_s0/paper_citations.bib` (2,357 B, `728` tok, SHA-256 `94d3acdf830d6401…`)
- **Artifact:** `datasets/felis_catus_turing_wavelength_s0/README.md` (3,229 B, `874` tok, SHA-256 `97e79aaac6633eb5…`)

### Step 3 — `source` (Formal source)
- **Notes:** CC BY-SA 3.0 Felis catus photograph by Von.grzanka, SHA-256 pinned.
- **Artifact:** `datasets/felis_catus_turing_wavelength_s0/felis_catus_specimen.jpg` (216,559 B, `64` tok, SHA-256 `9360be6f1689a1df…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3226, seven theorems, zero sorry.
- **Artifact:** `morphoeval/lean/loops/Loop_3226_felis_catus_turing_wavelength_s0.lean` (1,934 B, `64` tok, SHA-256 `fb2328066ab521a2…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-felis-catus-turing-wavelength-s0.yaml` (2,518 B, `729` tok, SHA-256 `0157596e4eeca5c7…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg spectral solver, pass-2 sweep, ablation, segmentation, dispersion fit.
- **Artifact:** `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` (33,061 B, `9,248` tok, SHA-256 `2e4ccb0c28615f82…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test passed. GIF 12 frames 640x480. CV similarity 96.0.
- **Artifact:** `docs/media/felis_catus_turing_wavelength_s0_comparison.png` (614,649 B, `64` tok, SHA-256 `23f064546c3b9212…`)
- **Artifact:** `docs/media/felis_catus_turing_wavelength_s0_growth.gif` (2,379,367 B, `64` tok, SHA-256 `1a5b6367d29faa8a…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 74.6. Pass 2 overall 87.6.
- **Artifact:** `morphoeval/audits/2026-10-27-felis-catus-turing-wavelength-s0-audit.yaml` (10,390 B, `2,649` tok, SHA-256 `fb98ed693fb7eaf8…`)
- **Artifact:** `morphoeval/audits/2026-10-27-felis-catus-turing-wavelength-s0-audit-v2.yaml` (10,244 B, `2,660` tok, SHA-256 `c142096d822b22d0…`)

### Step 8 — `measure` (Measure)
- **Notes:** 15 measurements, quick_gate 100. Pearson r 0.9901.
- **Artifact:** `morphoeval/audits/2026-10-27-felis-catus-turing-wavelength-s0-audit-v2.measurements.yaml` (7,424 B, `2,001` tok, SHA-256 `3b1a1dbd6b2e8d38…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Conformance OK for v1 and v2. Shared rubric not re-emitted.
- **Artifact:** `morphoeval/kg/2026-10-27-felis-catus-turing-wavelength-s0-audit-v2/REPORT.md` (10,405 B, `2,806` tok, SHA-256 `79f269934fd7bcce…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml is absent. No trusty URI was fabricated.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for pass 1 and pass 2.
- **Artifact:** `morphoeval/kg/2026-10-27-felis-catus-turing-wavelength-s0-audit/REPORT.md` (8,400 B, `2,270` tok, SHA-256 `1b9ae8de7c06152e…`)
- **Artifact:** `morphoeval/kg/2026-10-27-felis-catus-turing-wavelength-s0-audit-v2/REPORT.md` (10,405 B, `2,806` tok, SHA-256 `79f269934fd7bcce…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed sweep, ablation, segmentation, and goodness-of-fit. Gates passed, so pass 3 was not run. image_fitness 90.9, cv 96.0, audit 87.6, Pearson r 0.9901.
- **Artifact:** `morphoeval/runs/2026-10-27-felis-catus-turing-wavelength-s0-loop-image-fitness.json` (2,865 B, `881` tok, SHA-256 `63cecda1886f5c57…`)

## Gates after pass 2

Pass 3 was not run. Every gate below passed.

| Gate | Result |
| --- | --- |
| image_fitness ≥ 70 | 90.9 |
| cv_sim_similarity ≥ 58 | 96.0 |
| GIF frames ≥ 6 | 12 |
| resolution ≥ 340 px (rendered at 480) | 640×480 |
| non-circular specimen metric | yes (circularity 0.146 vs reference 1, and seven other value≠reference rows) |
| Pearson r > 0.98 | 0.9901 |
| audit overall ≥ 85 | 87.6 (pass 2; pass 1 was 74.6) |

Simulated dominant wavelength is 431.3 µm against the slot scale of 436 µm. Age is 0 Ma. The Felis chronoseries in this slot is 1/1, so the comparison is across scale: an 80 µm cellular domain is stable, and a 4360 µm flank holds ten repeats of the same wavenumber.

Signing was not performed. `~/.nanopub/profile.yml` is absent, so no trusty URI was produced. The pass-1 and pass-2 nanopubs did pass the offline Project Delphi conformance check.
