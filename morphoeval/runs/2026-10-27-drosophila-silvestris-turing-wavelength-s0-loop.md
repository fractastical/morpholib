# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-silvestris-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-silvestris-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila silvestris Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor:** `grok-4.7` (builder-assessor only; no second model scored this loop)
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **739.82 s** (12.33 min) — Tier A: `649.53 s` · Tier B: `90.30 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **31,919** (input: `0`, output: `31,919`, subagent: `0`) — Tier A: `14,636` · Tier B: `17,283`
- **Total artifact tokens produced:** `31,919`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 329.79 | 0.00 | 0 | 615 | 0 | **615** | 615 | `scout_manifest.json`, `drosophila_silvestris_turing_wavelength_s0_specimen.jpg` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 3,471 | 0 | **3,471** | 3,471 | `drosophila_silvestris_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 64 | 0 | **64** | 64 | `drosophila_silvestris_turing_wavelength_s0_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.12 | 0.00 | 0 | 1,259 | 0 | **1,259** | 1,259 | `Loop_3204_drosophila_silvestris_turing_wavelength_s0.lean`, `delphi10000-drosophila-silvestris-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 178.82 | 0.00 | 0 | 9,099 | 0 | **9,099** | 9,099 | `2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 140.62 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_silvestris_turing_wavelength_s0_comparison.png`, `drosophila_silvestris_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 23.79 | 0.00 | 0 | 5,374 | 0 | **5,374** | 5,374 | `2026-10-27-drosophila-silvestris-turing-wavelength-s0-audit.yaml`, `2026-10-27-drosophila-silvestris-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.14 | 0.00 | 0 | 4,048 | 0 | **4,048** | 4,048 | `2026-10-27-drosophila-silvestris-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-drosophila-silvestris-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 66.06 | 0.00 | 0 | 5,176 | 0 | **5,176** | 5,176 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.10 | 0.00 | 0 | 2,685 | 0 | **2,685** | 2,685 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **739.82** | **0.00** | **0** | **31,919** | **0** | **31,919** | **31,919** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia. No slug collision. CC BY-SA 3.0 Kilohana specimen vendored. Slot 3204 is outside campaign_100_slots.yaml so scout used --query/--taxon.
- **Artifact:** `datasets/drosophila_silvestris_turing_wavelength_s0/scout_manifest.json` (2,001 B, `551` tok, SHA-256 `6516ed21140c060b…`)
- **Artifact:** `datasets/drosophila_silvestris_turing_wavelength_s0/drosophila_silvestris_turing_wavelength_s0_specimen.jpg` (151,406 B, `64` tok, SHA-256 `dd93694218361504…`)

### Step 2 — `paper` (Paper)
- **Artifact:** `datasets/drosophila_silvestris_turing_wavelength_s0/drosophila_silvestris_turing_wavelength_s0_reference.json` (4,903 B, `1,571` tok, SHA-256 `9f4f77b1d9bee8df…`)
- **Artifact:** `datasets/drosophila_silvestris_turing_wavelength_s0/paper_citations.bib` (2,635 B, `837` tok, SHA-256 `4ff27448f5d20b31…`)
- **Artifact:** `datasets/drosophila_silvestris_turing_wavelength_s0/README.md` (4,271 B, `1,063` tok, SHA-256 `7d426a35f7a5d9f6…`)

### Step 3 — `source` (Formal source)
- **Notes:** CC BY-SA 3.0 Kilohana photograph, SHA-256 dd936942183615043876595eb722f77a3c7edbc4fa886bd87ca6bd9eedd4d81c
- **Artifact:** `datasets/drosophila_silvestris_turing_wavelength_s0/drosophila_silvestris_turing_wavelength_s0_specimen.jpg` (151,406 B, `64` tok, SHA-256 `dd93694218361504…`)

### Step 4 — `rules` (Pin the rules)
- **Artifact:** `morphoeval/lean/loops/Loop_3204_drosophila_silvestris_turing_wavelength_s0.lean` (2,384 B, `64` tok, SHA-256 `a1dfe905a4123c67…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-drosophila-silvestris-turing-wavelength-s0.yaml` (4,560 B, `1,195` tok, SHA-256 `aa2b82b6d0071c74…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 1 self-test passed (k r=0.989, lambda r=0.988, CV=90.5, 139 raw components). Pass 2 centroid, wing-band mask, and clean terminal frame landed in the same script before the output render.
- **Artifact:** `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` (32,599 B, `9,099` tok, SHA-256 `cd4de698cd62c614…`)

### Step 6 — `output` (Output & compare)
- **Notes:** 12 frames, 720x576, pearson k 0.998, CV 96.0
- **Artifact:** `docs/media/drosophila_silvestris_turing_wavelength_s0_comparison.png` (594,499 B, `64` tok, SHA-256 `0830930fc3e69be6…`)
- **Artifact:** `docs/media/drosophila_silvestris_turing_wavelength_s0_growth.gif` (1,766,654 B, `64` tok, SHA-256 `d7debf773cd5bad2…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall targeted 81.4 with R1-R4 open. Pass 2 closes them.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-silvestris-turing-wavelength-s0-audit.yaml` (9,297 B, `2,447` tok, SHA-256 `190fd12e387d770a…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-silvestris-turing-wavelength-s0-audit-v2.yaml` (11,313 B, `2,927` tok, SHA-256 `8134d0821f1b731b…`)

### Step 8 — `measure` (Measure)
- **Notes:** quick_gate 90/100. Pearson r 0.9981. Nine non-circular measurements.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-silvestris-turing-wavelength-s0-audit.measurements.yaml` (7,457 B, `1,992` tok, SHA-256 `f4c7fb9a51ee5b2b…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-silvestris-turing-wavelength-s0-audit-v2.measurements.yaml` (7,738 B, `2,056` tok, SHA-256 `969fba8af9ea6138…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Conformance OK with --allow-reintroduction because 00-rubric.trig is already published from earlier audits. Unsigned.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-silvestris-turing-wavelength-s0-audit/REPORT.md` (9,173 B, `2,491` tok, SHA-256 `9bada0613e14299f…`)
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-silvestris-turing-wavelength-s0-audit-v2/REPORT.md` (10,008 B, `2,685` tok, SHA-256 `c562f4ea1e1e24cd…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable. ~/.nanopub/profile.yml does not exist. sign_morphoeval.py and check_signed.mjs were not run. No signatures were fabricated.

### Step 11 — `graph` (Public record)
- **Notes:** Pass 1 overall 81.4. Pass 2 overall 89.6. Supersession chain recorded in REPORT.md.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-silvestris-turing-wavelength-s0-audit-v2/REPORT.md` (10,008 B, `2,685` tok, SHA-256 `c562f4ea1e1e24cd…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** R1-R4 closed. Gates passed on pass 2 (image_fitness 92.1, CV 96.0, Pearson r 0.9981 and 0.9972, audit 89.6). Pass 3 not required.

## Gates after pass 2

All required gates passed. Pass 3 was not run.

| Gate | Result |
|---|---|
| image_fitness ≥ 70 | 92.1 |
| cv_sim_similarity ≥ 58 | 96.0 |
| GIF frames ≥ 6 | 12 (720×576) |
| resolution ≥ 340 px | 576 |
| non-circular specimen metric | foreground 0.571 vs 0.50; wing spots 17 vs 25; coherency 0.518 vs 0.40 |
| Pearson r > 0.98 | wavenumber 0.9981; wavelength 0.9972 |
| audit overall ≥ 85 | pass 2 **89.6** (pass 1 was 81.4) |

Signing was not available: `~/.nanopub/profile.yml` is absent. The nanopubs are unsigned. No signature was fabricated.
