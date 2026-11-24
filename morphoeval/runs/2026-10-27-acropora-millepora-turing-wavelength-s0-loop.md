# MorphoEval Full-Loop Telemetry — `2026-10-27-acropora-millepora-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-acropora-millepora-turing-wavelength-s0`
- **Primary paper / dataset:** Acropora millepora Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `dff4f2b5`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **964.78 s** (16.08 min) — Tier A: `964.27 s` · Tier B: `0.51 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **2,095** (input: `0`, output: `2,095`, subagent: `0`) — Tier A: `2,095` · Tier B: `0`
- **Total artifact tokens produced:** `2,095`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 464.42 | 0.00 | 0 | 1,026 | 0 | **1,026** | 1,026 | `scout_3104_preview.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 205.14 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 64 | 0 | **64** | 64 | `acropora_millepora_gbr_lundgren2013.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 877 | 0 | **877** | 877 | `Loop_3104_acropora_millepora_turing_wavelength_s0.lean`, `delphi10000-acropora-millepora-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 294.46 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `acropora_millepora_turing_wavelength_s0_comparison.png`, `acropora_millepora_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **964.78** | **0.00** | **0** | **2,095** | **0** | **2,095** | **2,095** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex+Wikimedia scout; no slug collision with acropora-millepora-turing-wavelength-s0. Colony photo CC BY 4.0 Lundgren et al. 2013.
- **Artifact:** `/tmp/scout_3104_preview.json` (3,656 B, `1,026` tok, SHA-256 `51232971088cd527…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, bib, README: 385.2 um, 0 Ma, Schnakenberg k_c.

### Step 3 — `source` (Formal source)
- **Notes:** CC BY 4.0 Lundgren GBR photograph already vendored
- **Artifact:** `datasets/acropora_millepora_turing_wavelength_s0/acropora_millepora_gbr_lundgren2013.jpg` (284,626 B, `64` tok, SHA-256 `94c2290198c33f39…`)

### Step 4 — `rules` (Pin the rules)
- **Artifact:** `morphoeval/lean/loops/Loop_3104_acropora_millepora_turing_wavelength_s0.lean` (1,825 B, `64` tok, SHA-256 `554c40903c2f79f6…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-acropora-millepora-turing-wavelength-s0.yaml` (3,015 B, `813` tok, SHA-256 `5b30792a3e985bbe…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 2 script: Schnakenberg, dihedral 6, KS, ablation inset. Self-test passed.

### Step 6 — `output` (Output & compare)
- **Notes:** GIF 720x540, 12 frames, CV 85.9.
- **Artifact:** `docs/media/acropora_millepora_turing_wavelength_s0_comparison.png` (1,107,243 B, `64` tok, SHA-256 `b961f6f44ace9b71…`)
- **Artifact:** `docs/media/acropora_millepora_turing_wavelength_s0_growth.gif` (1,834,818 B, `64` tok, SHA-256 `fb612905014ee0a4…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 82.8; pass 2 overall 87.1. Three pass-1 remediations closed.

### Step 8 — `measure` (Measure)
- **Notes:** quick_gate 100/100. Pearson r 0.999986. Non-circular specimen metrics.

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg.py --conform --allow-reintroduction. Conformance OK. Rubric nanopub reuses the shared rubric.

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable. ~/.nanopub/profile.yml absent and sign_morphoeval.py cannot import requests/nanopub. Unsigned nanopubs only. Not faked.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for audit and audit-v2. Overall 87.1 faithful.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed dihedral gauge-fix, orientation KS, and ablation time-series. Gates passed, so pass 3 was not run.

## Scientific result (slot 3104)

*Acropora millepora*, Holocene, 0 Ma, chronoseries length 1 (no in-series predecessor or successor). Selected Turing wavelength 385.2 µm (3.852×10⁻⁴ m, log10 = −3.414), tier 5 primordium, detail variant. Schnakenberg kinetics a=0.12, b=0.78, D_v/D_u=15.75. D_u is solved so argmax σ(k) sits on that wavelength. The Murray band center is 319.4 µm and is not the selected mode.

The specimen is the CC BY 4.0 Great Barrier Reef colony photograph of Lundgren et al. 2013, not a 385 µm micrograph. The GIF is the gauge-fixed activator field (dihedral 6), 12 frames, 720×540.

Scores are from one builder-assessor (Grok 4.7). The telemetry header names the track_loop default assessor pair; those models did not score this slot.

| Gate | Result |
|---|---|
| image_fitness | 84.4 (threshold 70) |
| cv_sim_similarity | 85.9 (threshold 58); pass 1 was 86.1 |
| GIF frames | 12 |
| min resolution | 540 px |
| non-circular specimen metrics | coherency 0.427 vs 0.722; foreground 0.550 vs 0.558; orientation KS 0.054 vs 0 |
| Pearson r | 0.999986 (RMSE 4.06 µm) |
| audit overall | pass 1: 82.8; pass 2: 87.1 |

Pass 3 was not run. Signing was not performed: `~/.nanopub/profile.yml` is absent, and `sign_morphoeval.py` cannot import `requests` in this environment. The nanopubs under `morphoeval/kg/` are unsigned. Open limits that do not fail the gates: no 385 µm pigment micrograph, and no second assessor.
