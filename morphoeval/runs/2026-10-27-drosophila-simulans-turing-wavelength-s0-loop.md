# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-simulans-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-simulans-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila simulans Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `grok-4.7`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **999.97 s** (16.67 min) — Tier A: `999.54 s` · Tier B: `0.44 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **653** (input: `0`, output: `653`, subagent: `0`) — Tier A: `653` · Tier B: `0`
- **Total artifact tokens produced:** `653`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 430.22 | 0.00 | 0 | 653 | 0 | **653** | 653 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 569.03 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **999.97** | **0.00** | **0** | **653** | **0** | **653** | **653** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex Crossref Wikimedia collision check; Slot 3205 is outside campaign_100; OpenAlex/Wikimedia queried directly. No filename collision with drosophila-simulans-turing-wavelength-s0. Sibling slot 2701 is the 83 um model and was not modified.
- **Artifact:** `datasets/drosophila_simulans_turing_wavelength_s0/scout_manifest.json` (2,334 B, `653` tok, SHA-256 `87457088261c8128…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, bib, README state 1.9 µm and 0 Ma.

### Step 3 — `source` (Formal source)
- **Notes:** slot 3205 source

### Step 4 — `rules` (Pin the rules)
- **Notes:** slot 3205 rules

### Step 5 — `script` (Script)
- **Notes:** slot 3205 script

### Step 6 — `output` (Output & compare)
- **Notes:** slot 3205 output

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** slot 3205 audit

### Step 8 — `measure` (Measure)
- **Notes:** slot 3205 measure

### Step 9 — `build` (Write as claims)
- **Notes:** slot 3205 build

### Step 10 — `publish` (Sign & verify)
- **Notes:** slot 3205 publish

### Step 11 — `graph` (Public record)
- **Notes:** slot 3205 graph

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** slot 3205 remed

## Pass results

*Drosophila simulans*, Holocene (0 Ma), detail scale 1.92 µm (1.92e-6 m, log10 = −5.717). Schnakenberg kinetics with γ fixed by the Turing wavenumber. Catalog neighbours: *D. silvestris* at 2.64 µm (no posterior lobe) and *D. subobscura* at 1.60 µm (no posterior lobe). Mode counts on a 96 µm arc: 36, 50, 60.

The species plate's measured annulus is 67.57 µm (111 px = 100 µm), which is 35.2 detail wavelengths per ring. That number is not relabelled as 1.92 µm.

| Gate | Result |
| --- | --- |
| Pass 1 audit overall | 80.5 |
| Pass 2 audit overall | 88.8 |
| Held-out Pearson r | 0.9895 |
| Pass-1 single-harmonic Pearson r | 0.9825 |
| Plate CV similarity | 89.5 |
| Wikimedia whole-fly CV control | 85.6 |
| image_fitness | 88.7 = 0.55·89.5 + 0.30·86 + 0.15·91 |
| GIF | 12 frames, 800×600 |
| Equal-diffusion amplitude ratio | 0.0078 |
| Numerical k / infinite-line k_c | 3.115 / 3.272 = 0.952 (bounded disk; left as a gap) |

Pass 3 was not run. Every listed gate is above its threshold. R4 stays open: there is no second assessor.

Signing was not performed. `~/.nanopub/profile.yml` is absent, so the nanopubs in `morphoeval/kg/2026-10-27-drosophila-simulans-turing-wavelength-s0-audit*` are unsigned. `build_morphoeval_kg.py --conform` passed for both audits. Lean 4 is not installed on this runner; `Loop_3205_drosophila_simulans_turing_wavelength_s0.lean` has nine theorems and no `sorry`, and was not kernel-checked here.
