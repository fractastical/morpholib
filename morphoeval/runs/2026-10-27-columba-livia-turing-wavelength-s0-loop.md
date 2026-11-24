# MorphoEval Full-Loop Telemetry — `2026-10-27-columba-livia-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-columba-livia-turing-wavelength-s0`
- **Primary paper / dataset:** Columba livia Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-run`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1355.00 s** (22.58 min) — Tier A: `930.00 s` · Tier B: `425.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **8,280** (input: `0`, output: `8,280`, subagent: `0`) — Tier A: `8,280` · Tier B: `0`
- **Total artifact tokens produced:** `8,280`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | grok-4.7 (`grok-4.7`) | `completed` | 180.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | grok-4.7 (`grok-4.7`) | `completed` | 120.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | grok-4.7 (`grok-4.7`) | `completed` | 90.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | grok-4.7 (`grok-4.7`) | `completed` | 80.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | grok-4.7 (`grok-4.7`) | `completed` | 420.00 | 0.00 | 0 | 8,280 | 0 | **8,280** | 8,280 | `2026-delphi10000-columba-livia-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | grok-4.7 (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | grok-4.7 (`grok-4.7`) | `completed` | 150.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | grok-4.7 (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | grok-4.7 (`grok-4.7`) | `completed` | 25.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | grok-4.7 (`grok-4.7`) | `completed` | 5.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | grok-4.7 (`grok-4.7`) | `completed` | 15.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | grok-4.7 (`grok-4.7`) | `completed` | 200.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1355.00** | **0.00** | **0** | **8,280** | **0** | **8,280** | **8,280** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia; slot 3151 is outside campaign_100; OpenAlex broad query empty; Crossref follow-up resolved six DOIs. No slug collision.

### Step 3 — `source` (Formal source)
- **Notes:** Vendored Alan D. Wilson CC BY-SA 2.5 rock dove. SHA-256 74dfe533e6b396ac7c607cf39ec48ce3b867de1c26b527b027f86f0b61dc357e.

### Step 5 — `script` (Script)
- **Notes:** self-test passed; lambda 256 um; Pearson r 0.999889
- **Artifact:** `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` (29,602 B, `8,280` tok, SHA-256 `5b8d0e339a5eeade…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 83.0; pass 2 overall 88.5. No second-company assessor.

### Step 8 — `measure` (Measure)
- **Notes:** quick_gate 100/100 both passes

### Step 9 — `build` (Write as claims)
- **Notes:** conform OK; rubric reintroduction allowed; loci not verified against this tokenized origin

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: requests missing and ~/.nanopub/profile.yml absent. No trusty URI.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for audit and audit-v2

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 1 CV 57.8 / fitness 63.0. Pass 2 fitted GIF CV 96.0 / fitness 92.7. Gates passed. Pass 3 not run.

## Result

Holocene *Columba livia*, age 0 Ma, chronoseries 1/1. Critical wavelength 256 µm on a 1024 µm vane patch (only unstable Fourier mode n = 4). Dispersion Pearson r = 0.999889. Equal-diffusion ablation ratio = 0.0177.

| Gate | Pass 1 | Pass 2 |
|---|---:|---:|
| image_fitness ≥ 70 | 63.0 | 92.7 |
| cv_sim_similarity ≥ 58 | 57.8 | 96.0 |
| GIF frames ≥ 6 (rendered 12) | 12 | 12 |
| min resolution ≥ 340 (rendered 480) | 480 | 480 |
| non-circular specimen metric | yes | yes |
| Pearson r > 0.98 | 0.999889 | 0.999889 |
| audit overall ≥ 85 | 83.0 | 88.5 |

Pass 3 was not run. Signing did not happen: `requests` is not installed and `~/.nanopub/profile.yml` is absent, so no trusty URI was minted. Nanopubs in the two kg directories are unsigned. No second-company assessor was run.
