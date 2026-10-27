# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-subobscura-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-subobscura-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila subobscura Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **144.00 s** (2.40 min) — Tier A: `100.00 s` · Tier B: `44.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **27,000** (input: `0`, output: `27,000`, subagent: `0`) — Tier A: `14,685` · Tier B: `12,315`
- **Total artifact tokens produced:** `27,000`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 12.00 | 0.00 | 0 | 1,806 | 0 | **1,806** | 1,806 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 20.00 | 0.00 | 0 | 2,897 | 0 | **2,897** | 2,897 | `drosophila_subobscura_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 8.00 | 0.00 | 0 | 64 | 0 | **64** | 64 | `drosophila_subobscura_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 15.00 | 0.00 | 0 | 1,203 | 0 | **1,203** | 1,203 | `Loop_3206_drosophila_subobscura_turing_wavelength_s0.lean`, `delphi10000-drosophila-subobscura-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 40.00 | 0.00 | 0 | 8,587 | 0 | **8,587** | 8,587 | `2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 5.00 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_subobscura_turing_wavelength_s0_comparison.png`, `drosophila_subobscura_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 20.00 | 0.00 | 0 | 4,662 | 0 | **4,662** | 4,662 | `2026-10-27-drosophila-subobscura-turing-wavelength-s0-audit.yaml`, `2026-10-27-drosophila-subobscura-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 6.00 | 0.00 | 0 | 1,935 | 0 | **1,935** | 1,935 | `2026-10-27-drosophila-subobscura-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 4.00 | 0.00 | 0 | 2,503 | 0 | **2,503** | 2,503 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 1.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 1.00 | 0.00 | 0 | 2,503 | 0 | **2,503** | 2,503 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 12.00 | 0.00 | 0 | 712 | 0 | **712** | 712 | `2026-10-27-drosophila-subobscura-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **144.00** | **0.00** | **0** | **27,000** | **0** | **27,000** | **27,000** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia. Slot 3206 is outside campaign_100_slots.yaml. No collision with the 124 µm sibling.
- **Artifact:** `datasets/drosophila_subobscura_turing_wavelength_s0/scout_manifest.json` (6,379 B, `1,806` tok, SHA-256 `b05b53f6082f2ee0…`)

### Step 2 — `paper` (Paper)
- **Notes:** Huey 2000 wing-size cline; Turing 1952; Schnakenberg 1979. Scale 3.0 µm and age 0 Ma recorded.
- **Artifact:** `datasets/drosophila_subobscura_turing_wavelength_s0/drosophila_subobscura_turing_wavelength_s0_reference.json` (5,706 B, `1,609` tok, SHA-256 `0b9b69eda941b675…`)
- **Artifact:** `datasets/drosophila_subobscura_turing_wavelength_s0/paper_citations.bib` (2,141 B, `685` tok, SHA-256 `9ff11f4db4fde2d9…`)
- **Artifact:** `datasets/drosophila_subobscura_turing_wavelength_s0/README.md` (2,348 B, `603` tok, SHA-256 `e13bf39b18cd6bbe…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored Martin Cooper CC BY 2.0 collage, SHA-256 5366640b506b531749072d3e4097e68630b8132fd433df786b50e2e16c89711d.
- **Artifact:** `datasets/drosophila_subobscura_turing_wavelength_s0/drosophila_subobscura_specimen.jpg` (4,792,720 B, `64` tok, SHA-256 `5366640b506b5317…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Four Lean theorems, zero sorry. Wavenumber ratios exceed 27 and 33.
- **Artifact:** `morphoeval/lean/loops/Loop_3206_drosophila_subobscura_turing_wavelength_s0.lean` (1,517 B, `64` tok, SHA-256 `ef462cdbeff5c29e…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-drosophila-subobscura-turing-wavelength-s0.yaml` (4,211 B, `1,139` tok, SHA-256 `91fc38733868fb8e…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 1 path is the bare field. Pass 2 adds the Du sweep, equal-diffusion ablation, and specimen mask.
- **Artifact:** `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` (30,688 B, `8,587` tok, SHA-256 `e97aacd29c32b91d…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 2 self-test passed. Pearson r = 0.999998. GIF 12 frames at 720x480.
- **Artifact:** `docs/media/drosophila_subobscura_turing_wavelength_s0_comparison.png` (876,428 B, `64` tok, SHA-256 `243f026cc7f99290…`)
- **Artifact:** `docs/media/drosophila_subobscura_turing_wavelength_s0_growth.gif` (2,516,450 B, `64` tok, SHA-256 `0040e9b46892f41d…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 74.6 with R1-R4. Pass 2 overall 87.8 closes them.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-subobscura-turing-wavelength-s0-audit.yaml` (7,812 B, `2,016` tok, SHA-256 `f69fc525159b78be…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-subobscura-turing-wavelength-s0-audit-v2.yaml` (9,952 B, `2,646` tok, SHA-256 `6c4ce378fd9d9bf2…`)

### Step 8 — `measure` (Measure)
- **Notes:** Nine measurements, quick_gate 90/100. Otsu fraction 0.4649 != border-contrast 0.7715.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-subobscura-turing-wavelength-s0-audit-v2.measurements.yaml` (7,221 B, `1,935` tok, SHA-256 `8abd5168d3d8edac…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Both audits conform. Shared rubric not re-emitted.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-subobscura-turing-wavelength-s0-audit-v2/REPORT.md` (9,121 B, `2,503` tok, SHA-256 `42817f1c53171177…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Not signed. ~/.nanopub/profile.yml is absent and sign_morphoeval.py cannot import requests. Unsigned nanopubs passed --conform. No trusty URIs were fabricated.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md overall 87.8, verdict faithful.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-subobscura-turing-wavelength-s0-audit-v2/REPORT.md` (9,121 B, `2,503` tok, SHA-256 `42817f1c53171177…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed R1-R4. Gates passed, so Pass 3 was not run. image_fitness 90.9, cv 96.0.
- **Artifact:** `morphoeval/runs/2026-10-27-drosophila-subobscura-turing-wavelength-s0-loop-image-fitness.json` (2,436 B, `712` tok, SHA-256 `6ca29386d24ac7fb…`)

## Result (Pass 2 gates)

*Drosophila subobscura*, Sophophora obscura & willistoni clade, Holocene (0 Ma), detail scale 3.0 µm (2.97e-6 m, log10 = −5.527). The law is the Turing critical wavenumber of Schnakenberg kinetics. Predecessor *Drosophila simulans* (82.56 µm) and successor *Drosophila suzukii* (98.45 µm) are the same geological age; this slot's wavenumber is 27.80× and 33.15× theirs. Huey et al. (2000) supply the wing-size cline, not a measured 3 µm lattice.

Pass 1 (`--pass 1`) left the Du sweep, the equal-diffusion ablation, and the specimen mask unset (audit overall 74.6, remediation R1–R4). Pass 2 closed those four items. Pass 3 was not run.

| Gate | Result |
|---|---|
| image_fitness ≥ 70 | 90.9 |
| cv_sim_similarity ≥ 58 | 96.0 |
| GIF frames ≥ 6 | 12 (720×480) |
| resolution ≥ 340 px | 480 |
| non-circular specimen metric | Otsu foreground 0.4649 vs border-contrast 0.7715 |
| Pearson r > 0.98 | 0.999998 (RMSE 0.000495, KS 0.083) |
| audit overall ≥ 85 | 87.8 (faithful) |

Nanopublications were built and passed the offline conformance check. They were not signed: `~/.nanopub/profile.yml` is absent, and `sign_morphoeval.py` cannot import `requests`. No trusty URI was invented.
