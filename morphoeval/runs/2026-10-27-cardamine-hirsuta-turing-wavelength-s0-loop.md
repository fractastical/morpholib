# MorphoEval Full-Loop Telemetry — `2026-10-27-cardamine-hirsuta-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-cardamine-hirsuta-turing-wavelength-s0`
- **Primary paper / dataset:** Cardamine hirsuta Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `pending`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **241.00 s** (4.02 min) — Tier A: `172.00 s` · Tier B: `69.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **35,145** (input: `0`, output: `35,145`, subagent: `0`) — Tier A: `15,018` · Tier B: `20,127`
- **Total artifact tokens produced:** `35,145`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 1,777 | 0 | **1,777** | 1,777 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 2,709 | 0 | **2,709** | 2,709 | `cardamine_hirsuta_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 15.00 | 0.00 | 0 | 64 | 0 | **64** | 64 | `cardamine_hirsuta_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 25.00 | 0.00 | 0 | 1,083 | 0 | **1,083** | 1,083 | `Loop_3140_cardamine_hirsuta_turing_wavelength_s0.lean`, `delphi10000-cardamine-hirsuta-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 50.00 | 0.00 | 0 | 9,257 | 0 | **9,257** | 9,257 | `2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 12.00 | 0.00 | 0 | 128 | 0 | **128** | 128 | `cardamine_hirsuta_turing_wavelength_s0_comparison.png`, `cardamine_hirsuta_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 3,065 | 0 | **3,065** | 3,065 | `2026-10-27-cardamine-hirsuta-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 12.00 | 0.00 | 0 | 4,095 | 0 | **4,095** | 4,095 | `2026-10-27-cardamine-hirsuta-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-cardamine-hirsuta-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 8.00 | 0.00 | 0 | 5,884 | 0 | **5,884** | 5,884 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 5.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 4.00 | 0.00 | 0 | 2,997 | 0 | **2,997** | 2,997 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 4,086 | 0 | **4,086** | 4,086 | `2026-10-27-cardamine-hirsuta-turing-wavelength-s0-audit-v2.yaml`, `2026-10-27-cardamine-hirsuta-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **241.00** | **0.00** | **0** | **35,145** | **0** | **35,145** | **35,145** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref scout plus Wikimedia public-domain specimen. No slug collision.; OpenAlex/Crossref scout plus Wikimedia public-domain specimen. No slug collision.
- **Artifact:** `datasets/cardamine_hirsuta_turing_wavelength_s0/scout_manifest.json` (6,440 B, `1,777` tok, SHA-256 `79d1ae5d5b27c95e…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, BibTeX, README. Scale 154.1 um and age 0 Ma.; Reference JSON, BibTeX, README. Scale 154.1 um and age 0 Ma.
- **Artifact:** `datasets/cardamine_hirsuta_turing_wavelength_s0/cardamine_hirsuta_turing_wavelength_s0_reference.json` (4,450 B, `1,242` tok, SHA-256 `c696a1319c853366…`)
- **Artifact:** `datasets/cardamine_hirsuta_turing_wavelength_s0/paper_citations.bib` (3,357 B, `981` tok, SHA-256 `bfcb6bdfc46bcf19…`)
- **Artifact:** `datasets/cardamine_hirsuta_turing_wavelength_s0/README.md` (1,988 B, `486` tok, SHA-256 `03de545ad27ccbf4…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored Commons File:Cardamine hirsuta.jpg (public domain), SHA-256 pinned.; Vendored Commons File:Cardamine hirsuta.jpg (public domain), SHA-256 pinned.
- **Artifact:** `datasets/cardamine_hirsuta_turing_wavelength_s0/cardamine_hirsuta_specimen.jpg` (197,711 B, `64` tok, SHA-256 `1c268fecb3660b31…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3140 (8 theorems, 0 sorry) and math-kg YAML.; Lean Loop_3140 (8 theorems, 0 sorry) and math-kg YAML.
- **Artifact:** `morphoeval/lean/loops/Loop_3140_cardamine_hirsuta_turing_wavelength_s0.lean` (1,917 B, `64` tok, SHA-256 `2e449314ed2577e4…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-cardamine-hirsuta-turing-wavelength-s0.yaml` (3,632 B, `1,019` tok, SHA-256 `98724e7a4a2c0b57…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg script with dispersion fit, sweep, ablation, specimen segmentation.; Schnakenberg script with dispersion fit, sweep, ablation, specimen segmentation.
- **Artifact:** `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` (32,208 B, `9,257` tok, SHA-256 `9026007f563aa491…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test passed. PNG and 12-frame 720x540 GIF. Pearson r=0.999896, CV=96.0.; Self-test passed. PNG and 12-frame 720x540 GIF. Pearson r=0.999896, CV=96.0.
- **Artifact:** `docs/media/cardamine_hirsuta_turing_wavelength_s0_comparison.png` (344,427 B, `64` tok, SHA-256 `42f28ae17a71284a…`)
- **Artifact:** `docs/media/cardamine_hirsuta_turing_wavelength_s0_growth.gif` (775,887 B, `64` tok, SHA-256 `911ab38123183c13…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 audit overall 80.2 with remediation R1-R5.; Pass 1 audit overall 80.2 with remediation R1-R5.
- **Artifact:** `morphoeval/audits/2026-10-27-cardamine-hirsuta-turing-wavelength-s0-audit.yaml` (12,314 B, `3,065` tok, SHA-256 `4590b3c1b43e83ce…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100/100, 11 measurements, values differ from references.; measure_outputs quick_gate 100/100, 11 measurements, values differ from references.
- **Artifact:** `morphoeval/audits/2026-10-27-cardamine-hirsuta-turing-wavelength-s0-audit.measurements.yaml` (7,380 B, `2,049` tok, SHA-256 `06a7a7142900ebc4…`)
- **Artifact:** `morphoeval/audits/2026-10-27-cardamine-hirsuta-turing-wavelength-s0-audit-v2.measurements.yaml` (7,354 B, `2,046` tok, SHA-256 `986ac64e6b4765a5…`)

### Step 9 — `build` (Write as claims)
- **Notes:** ProbKnow conformance OK for Pass 1 and Pass 2 nanopubs.; ProbKnow conformance OK for Pass 1 and Pass 2 nanopubs.
- **Artifact:** `morphoeval/kg/2026-10-27-cardamine-hirsuta-turing-wavelength-s0-audit/REPORT.md` (10,445 B, `2,887` tok, SHA-256 `251634f5dab11c3f…`)
- **Artifact:** `morphoeval/kg/2026-10-27-cardamine-hirsuta-turing-wavelength-s0-audit-v2/REPORT.md` (10,664 B, `2,997` tok, SHA-256 `e23d13ae79147bdd…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no ~/.nanopub/profile.yml and no nanopub module. Not fabricated.; Signing unavailable: no ~/.nanopub/profile.yml and no nanopub module. Not fabricated.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for Pass 1 (80.2) and Pass 2 (88.2).; REPORT.md present for Pass 1 (80.2) and Pass 2 (88.2).
- **Artifact:** `morphoeval/kg/2026-10-27-cardamine-hirsuta-turing-wavelength-s0-audit-v2/REPORT.md` (10,664 B, `2,997` tok, SHA-256 `e23d13ae79147bdd…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed R1-R4. R5 second assessor remains open. Gates pass; no Pass 3. image_fitness 91.8.; Pass 2 closed R1-R4. R5 second assessor remains open. Gates pass; no Pass 3. image_fitness 91.8.
- **Artifact:** `morphoeval/audits/2026-10-27-cardamine-hirsuta-turing-wavelength-s0-audit-v2.yaml` (12,514 B, `3,176` tok, SHA-256 `b2dcac8bfcbdf134…`)
- **Artifact:** `morphoeval/runs/2026-10-27-cardamine-hirsuta-turing-wavelength-s0-loop-image-fitness.json` (3,169 B, `910` tok, SHA-256 `b3d397110b8e7c47…`)

## Result

*Cardamine hirsuta*, Holocene, 0 Ma, chronoseries 1/1. Classical Turing wavelength λ_c = 154.1 µm on a Schnakenberg patch with D_v/D_u = 10.5. A single ~16 µm cell cannot host that mode; the detail patch holds eight classical wavelengths. RCO changes leaflet domain size at a larger scale and does not reset k_c.

Pass 2 gates: image fitness 91.8, CV similarity 96.0, 12 frames at 720×540, dispersion Pearson r = 0.999896, audit overall 88.2. Nonlinear spacing is 173.4 µm and is left different from 154.1 µm. Signing was not done: there is no nanopub profile and the nanopub package is not installed. Lean was not executed here (`lean` is not installed); the module has eight theorems and no `sorry`. A second assessor is still open (R5).
