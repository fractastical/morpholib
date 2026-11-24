# MorphoEval Full-Loop Telemetry — `2026-10-27-canis-lupus-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-canis-lupus-turing-wavelength-s0`
- **Primary paper / dataset:** Canis lupus Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **360.00 s** (6.00 min) — Tier A: `180.00 s` · Tier B: `180.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **38,282** (input: `0`, output: `38,282`, subagent: `0`) — Tier A: `15,714` · Tier B: `22,568`
- **Total artifact tokens produced:** `38,282`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 1,783 | 0 | **1,783** | 1,783 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 3,220 | 0 | **3,220** | 3,220 | `canis_lupus_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 64 | 0 | **64** | 64 | `canis_lupus_turing_wavelength_s0_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 943 | 0 | **943** | 943 | `Loop_3139_canis_lupus_turing_wavelength_s0.lean`, `delphi10000-canis-lupus-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 9,576 | 0 | **9,576** | 9,576 | `2026-delphi10000-canis-lupus-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 128 | 0 | **128** | 128 | `canis_lupus_turing_wavelength_s0_comparison.png`, `canis_lupus_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 30.00 | 0.00 | 0 | 6,175 | 0 | **6,175** | 6,175 | `2026-10-27-canis-lupus-turing-wavelength-s0-audit.yaml`, `2026-10-27-canis-lupus-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 30.00 | 0.00 | 0 | 3,625 | 0 | **3,625** | 3,625 | `2026-10-27-canis-lupus-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-canis-lupus-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 30.00 | 0.00 | 0 | 5,777 | 0 | **5,777** | 5,777 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 30.00 | 0.00 | 0 | 3,050 | 0 | **3,050** | 3,050 | `REPORT.md` |
| 11 | B | **graph** (Public record) | tool | `completed` | 30.00 | 0.00 | 0 | 3,050 | 0 | **3,050** | 3,050 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 30.00 | 0.00 | 0 | 891 | 0 | **891** | 891 | `2026-10-27-canis-lupus-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **360.00** | **0.00** | **0** | **38,282** | **0** | **38,282** | **38,282** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Slot 3139 is outside campaign_100. OpenAlex DOI lookup plus Wikimedia scout. No filename collision.; Slot 3139 is outside campaign_100. OpenAlex DOI lookup plus Wikimedia scout. No filename collision.
- **Artifact:** `datasets/canis_lupus_turing_wavelength_s0/scout_manifest.json` (6,368 B, `1,783` tok, SHA-256 `ce8b43ccdb4e68d5…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, BibTeX, README. Scale 1.781e-6 m and age 0 Ma.; Reference JSON, BibTeX, README. Scale 1.781e-6 m and age 0 Ma.
- **Artifact:** `datasets/canis_lupus_turing_wavelength_s0/canis_lupus_turing_wavelength_s0_reference.json` (6,899 B, `1,924` tok, SHA-256 `d16ba21863080c9e…`)
- **Artifact:** `datasets/canis_lupus_turing_wavelength_s0/paper_citations.bib` (2,767 B, `872` tok, SHA-256 `17550cda1707cfc9…`)
- **Artifact:** `datasets/canis_lupus_turing_wavelength_s0/README.md` (1,562 B, `424` tok, SHA-256 `a73125158bb03408…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored gallery specimen Canis_lupus.jpg. SHA-256 8dd4a5b3caa19a6f06509a0acce3b4040493b8b1dbdd37cc608378b858e7a60c.; Vendored gallery specimen Canis_lupus.jpg. SHA-256 8dd4a5b3caa19a6f06509a0acce3b4040493b8b1dbdd37cc608378b858e7a60c.
- **Artifact:** `datasets/canis_lupus_turing_wavelength_s0/canis_lupus_turing_wavelength_s0_specimen.jpg` (369,256 B, `64` tok, SHA-256 `8dd4a5b3caa19a6f…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean namespace MathOfBiology.Loop_3139, four theorems, zero sorry, plus math-kg yaml.; Lean namespace MathOfBiology.Loop_3139, four theorems, zero sorry, plus math-kg yaml.
- **Artifact:** `morphoeval/lean/loops/Loop_3139_canis_lupus_turing_wavelength_s0.lean` (1,852 B, `64` tok, SHA-256 `92a96286eafae5d9…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-canis-lupus-turing-wavelength-s0.yaml` (2,876 B, `879` tok, SHA-256 `69fe9f694339bd48…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 1 script with Schnakenberg k_c, self-test, GIF and PNG. Pass 2 flipped REMEDIATION_PASS to 2.; Pass 1 script with Schnakenberg k_c, self-test, GIF and PNG. Pass 2 flipped REMEDIATION_PASS to 2.
- **Artifact:** `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` (33,512 B, `9,576` tok, SHA-256 `90d8c34d2283b840…`)

### Step 6 — `output` (Output & compare)
- **Notes:** self-test PASS. GIF 16 frames 640x480. CV similarity 96.0. Pearson r 0.99352.; self-test PASS. GIF 16 frames 640x480. CV similarity 96.0. Pearson r 0.99352.
- **Artifact:** `docs/media/canis_lupus_turing_wavelength_s0_comparison.png` (622,920 B, `64` tok, SHA-256 `38a5f6da5be08515…`)
- **Artifact:** `docs/media/canis_lupus_turing_wavelength_s0_growth.gif` (3,945,362 B, `64` tok, SHA-256 `d1a71525f7f75bfc…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 81.6. Pass 2 overall 87.4. R1 R2 R3 R5 closed. R4 second assessor left open.; Pass 1 overall 81.6. Pass 2 overall 87.4. R1 R2 R3 R5 closed. R4 second assessor left open.
- **Artifact:** `morphoeval/audits/2026-10-27-canis-lupus-turing-wavelength-s0-audit.yaml` (12,096 B, `3,032` tok, SHA-256 `dd5666014ef4c8a1…`)
- **Artifact:** `morphoeval/audits/2026-10-27-canis-lupus-turing-wavelength-s0-audit-v2.yaml` (11,994 B, `3,143` tok, SHA-256 `9fbba48d14f6a7e6…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs.py pass 1 (7 metrics) and pass 2 (10 metrics, quick_gate 100).; measure_outputs.py pass 1 (7 metrics) and pass 2 (10 metrics, quick_gate 100).
- **Artifact:** `morphoeval/audits/2026-10-27-canis-lupus-turing-wavelength-s0-audit.measurements.yaml` (5,647 B, `1,631` tok, SHA-256 `4b91bea2dd3b4c0a…`)
- **Artifact:** `morphoeval/audits/2026-10-27-canis-lupus-turing-wavelength-s0-audit-v2.measurements.yaml` (7,014 B, `1,994` tok, SHA-256 `7e920edd76aa22c5…`)

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg.py --conform OK for both audits. Rubric nanopub reused from the published Bos taurus v7 head.; build_morphoeval_kg.py --conform OK for both audits. Rubric nanopub reused from the published Bos taurus v7 head.
- **Artifact:** `morphoeval/kg/2026-10-27-canis-lupus-turing-wavelength-s0-audit/REPORT.md` (10,218 B, `2,727` tok, SHA-256 `395ad5ff7979643b…`)
- **Artifact:** `morphoeval/kg/2026-10-27-canis-lupus-turing-wavelength-s0-audit-v2/REPORT.md` (10,778 B, `3,050` tok, SHA-256 `f53884d725fe0654…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable. ~/.nanopub/profile.yml does not exist. No trusty URI was minted and none was invented.; Signing unavailable. ~/.nanopub/profile.yml does not exist. No trusty URI was minted and none was invented.
- **Artifact:** `morphoeval/kg/2026-10-27-canis-lupus-turing-wavelength-s0-audit-v2/REPORT.md` (10,778 B, `3,050` tok, SHA-256 `f53884d725fe0654…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for pass 1 and pass 2. Pass 2 supersession chain is recorded. Overall 87.4.; REPORT.md present for pass 1 and pass 2. Pass 2 supersession chain is recorded. Overall 87.4.
- **Artifact:** `morphoeval/kg/2026-10-27-canis-lupus-turing-wavelength-s0-audit-v2/REPORT.md` (10,778 B, `3,050` tok, SHA-256 `f53884d725fe0654…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 executed the sweep, equal-diffusion and gamma=0 controls, and the KS test. Gates cleared. Pass 3 not required.; Pass 2 executed the sweep, equal-diffusion and gamma=0 controls, and the KS test. Gates cleared. Pass 3 not required.
- **Artifact:** `morphoeval/runs/2026-10-27-canis-lupus-turing-wavelength-s0-loop-image-fitness.json` (3,018 B, `891` tok, SHA-256 `4fa099d60bd98755…`)

## Result (Pass 2)

Holocene *Canis lupus* (Canine, MDCK), age 0 Ma, detail scale 1.781e-6 m. Schnakenberg Turing wavenumber on an 8-wavelength apical patch. The header's second assessor name is the telemetry default; that assessor was not run (audit item R4, left open).

Gates after Pass 2, all cleared, so Pass 3 was not run:

- image_fitness = 91.2 (threshold 70)
- cv_sim_similarity = 96.0 (threshold 58), from `compute_sim_vs_specimen_cv_similarity` on the 16-frame 640×480 GIF
- Pearson r = 0.99352 (threshold 0.98), RMSE = 1.002, against a reference of 1 and 0
- audit overall = 87.4 (threshold 85); Pass 1 was 81.6
- non-circular metrics include specimen nearest-neighbour spacing 8.94 px (reference 1.781 µm, different quantity) and KS = 0.378 (reference 0)

Signing: `~/.nanopub/profile.yml` is absent. `sign_morphoeval.py` was not run. No trusty URI was minted. The knowledge-graph TriG files are unsigned. Conformance of the unsigned graph passed.
