# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-hydei-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-hydei-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila hydei Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `none-single-assessor`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1331.23 s** (22.19 min) — Tier A: `1319.57 s` · Tier B: `11.66 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **31,260** (input: `0`, output: `31,260`, subagent: `0`) — Tier A: `14,901` · Tier B: `16,359`
- **Total artifact tokens produced:** `31,260`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 103.36 | 0.00 | 0 | 203 | 0 | **203** | 203 | `scout_raw.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 607.97 | 0.00 | 0 | 3,530 | 0 | **3,530** | 3,530 | `drosophila_hydei_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 607.97 | 0.00 | 0 | 64 | 0 | **64** | 64 | `drosophila_hydei_turing_wavelength_s0_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,295 | 0 | **1,295** | 1,295 | `Loop_3189_drosophila_hydei_turing_wavelength_s0.lean`, `delphi10000-drosophila-hydei-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 9,681 | 0 | **9,681** | 9,681 | `2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_hydei_turing_wavelength_s0_comparison.png`, `drosophila_hydei_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 6,276 | 0 | **6,276** | 6,276 | `2026-10-27-drosophila-hydei-turing-wavelength-s0-audit.yaml`, `2026-10-27-drosophila-hydei-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 8.31 | 0.00 | 0 | 3,843 | 0 | **3,843** | 3,843 | `2026-10-27-drosophila-hydei-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-drosophila-hydei-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 2.96 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.12 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.09 | 0.00 | 0 | 5,448 | 0 | **5,448** | 5,448 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 792 | 0 | **792** | 792 | `2026-10-27-drosophila-hydei-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1331.23** | **0.00** | **0** | **31,260** | **0** | **31,260** | **31,260** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex + Wikimedia; slot 3189 is not in campaign_100_slots.yaml; Slot 3189 absent from campaign_100_slots.yaml; queried OpenAlex and Wikimedia directly. No filename collision with delphi10000-drosophila-hydei-turing-wavelength-s0. Sibling slot 3001 is the 117 µm primordium model, not this detail scale.
- **Artifact:** `/tmp/hydei_scout/scout_raw.json` (798 B, `203` tok, SHA-256 `db95c09c5720c49e…`)

### Step 2 — `paper` (Paper)
- **Notes:** Scale 2.344e-06 m (2.3 µm, log10=-5.630), age 0 Ma. Schnakenberg–Turing k_c.
- **Artifact:** `datasets/drosophila_hydei_turing_wavelength_s0/drosophila_hydei_turing_wavelength_s0_reference.json` (4,815 B, `1,472` tok, SHA-256 `638a3272e18240dc…`)
- **Artifact:** `datasets/drosophila_hydei_turing_wavelength_s0/paper_citations.bib` (2,634 B, `831` tok, SHA-256 `4266c3fd37318a28…`)
- **Artifact:** `datasets/drosophila_hydei_turing_wavelength_s0/README.md` (4,339 B, `1,227` tok, SHA-256 `258a7865f6ffc435…`)

### Step 3 — `source` (Formal source)
- **Notes:** CC BY 3.0 Wikimedia Drosophila hydei 09.JPG, SHA-256 pinned. Gallery JPEG was a synthetic pattern and was not used.
- **Artifact:** `datasets/drosophila_hydei_turing_wavelength_s0/drosophila_hydei_turing_wavelength_s0_specimen.jpg` (212,098 B, `64` tok, SHA-256 `483d25650491bb68…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Seven Lean theorems, 0 sorry. 50× primordium identity.
- **Artifact:** `morphoeval/lean/loops/Loop_3189_drosophila_hydei_turing_wavelength_s0.lean` (2,093 B, `64` tok, SHA-256 `b82b75f1b0496ea5…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-drosophila-hydei-turing-wavelength-s0.yaml` (4,437 B, `1,231` tok, SHA-256 `21469f136aae94a9…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 2 script: envelope fit, gamma sweep, equal-diffusion ablation, specimen segmentation.
- **Artifact:** `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` (33,031 B, `9,681` tok, SHA-256 `05f8eec68fc62604…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test Pearson r=0.9985. GIF 12 frames 720x540. CV similarity 96.0.
- **Artifact:** `docs/media/drosophila_hydei_turing_wavelength_s0_comparison.png` (338,902 B, `64` tok, SHA-256 `7573be404d095e4c…`)
- **Artifact:** `docs/media/drosophila_hydei_turing_wavelength_s0_growth.gif` (1,502,876 B, `64` tok, SHA-256 `6d2fcf25752ff726…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 82.4. Pass 2 closes R1-R3 and scores 87.5. Single assessor.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-hydei-turing-wavelength-s0-audit.yaml` (11,367 B, `3,006` tok, SHA-256 `acb579c3af18daa7…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-hydei-turing-wavelength-s0-audit-v2.yaml` (12,315 B, `3,270` tok, SHA-256 `bb8dae7d95a65ad5…`)

### Step 8 — `measure` (Measure)
- **Notes:** Fingerprints and non-circular specimen metrics.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-hydei-turing-wavelength-s0-audit.measurements.yaml` (6,791 B, `1,813` tok, SHA-256 `946ffb00b74180e4…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-hydei-turing-wavelength-s0-audit-v2.measurements.yaml` (7,404 B, `2,030` tok, SHA-256 `751dbb67715fb51f…`)

### Step 9 — `build` (Write as claims)
- **Notes:** ProbKnow conformance OK for pass 1 and pass 2. Unsigned: no nanopub profile.

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no ~/.nanopub/profile.yml. Graphs are unsigned. Did not fabricate trusty hashes.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for pass 1 and pass 2.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-hydei-turing-wavelength-s0-audit/REPORT.md` (9,352 B, `2,583` tok, SHA-256 `694a7bd97b4acbcc…`)
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-hydei-turing-wavelength-s0-audit-v2/REPORT.md` (10,211 B, `2,865` tok, SHA-256 `c8624f495b14213e…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 gates met: image_fitness 91.2, CV 96.0, Pearson r 0.9985, audit 87.5. Pass 3 not required.
- **Artifact:** `morphoeval/runs/2026-10-27-drosophila-hydei-turing-wavelength-s0-loop-image-fitness.json` (2,681 B, `792` tok, SHA-256 `82e33b620cdd98ef…`)
