# MorphoEval Full-Loop Telemetry — `2026-10-27-dugesia-japonica-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-dugesia-japonica-turing-wavelength-s0`
- **Primary paper / dataset:** Dugesia japonica Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `none`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.07 s** (0.02 min) — Tier A: `0.52 s` · Tier B: `0.55 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **31,992** (input: `0`, output: `31,992`, subagent: `0`) — Tier A: `14,336` · Tier B: `17,656`
- **Total artifact tokens produced:** `31,992`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,072 | 0 | **1,072** | 1,072 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 3,323 | 0 | **3,323** | 3,323 | `dugesia_japonica_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `dugesia_japonica_ocellus_pigment_nogi2009.png`, `dugesia_japonica_field_kalininsky_ccby4.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,331 | 0 | **1,331** | 1,331 | `Loop_3213_dugesia_japonica_turing_wavelength_s0.lean`, `delphi10000-dugesia-japonica-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 8,354 | 0 | **8,354** | 8,354 | `2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `dugesia_japonica_turing_wavelength_s0_comparison.png`, `dugesia_japonica_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | primary-assessor (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 5,739 | 0 | **5,739** | 5,739 | `2026-10-27-dugesia-japonica-turing-wavelength-s0-audit.yaml`, `2026-10-27-dugesia-japonica-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 3,684 | 0 | **3,684** | 3,684 | `2026-10-27-dugesia-japonica-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-dugesia-japonica-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 4,907 | 0 | **4,907** | 4,907 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,513 | 0 | **2,513** | 2,513 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 813 | 0 | **813** | 813 | `2026-10-27-dugesia-japonica-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1.07** | **0.00** | **0** | **31,992** | **0** | **31,992** | **31,992** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex and Wikimedia; slot 3213 is outside campaign_100; No s0 collision. Sibling slot 166 left untouched.
- **Artifact:** `datasets/dugesia_japonica_turing_wavelength_s0/scout_manifest.json` (3,939 B, `1,072` tok, SHA-256 `3ed6c6db12573fd9…`)

### Step 2 — `paper` (Paper)
- **Notes:** Scale 6.912 um and age 0 Ma recorded. Real Crossref DOIs.
- **Artifact:** `datasets/dugesia_japonica_turing_wavelength_s0/dugesia_japonica_turing_wavelength_s0_reference.json` (5,588 B, `1,746` tok, SHA-256 `2bc1220e0880d8c0…`)
- **Artifact:** `datasets/dugesia_japonica_turing_wavelength_s0/paper_citations.bib` (3,294 B, `987` tok, SHA-256 `c8f9334af9bf74c6…`)
- **Artifact:** `datasets/dugesia_japonica_turing_wavelength_s0/README.md` (2,207 B, `590` tok, SHA-256 `1a029195f38a450a…`)

### Step 3 — `source` (Formal source)
- **Notes:** CC BY 2.5 Nogi plate and CC BY 4.0 iNaturalist voucher, SHA-256 pinned.
- **Artifact:** `datasets/dugesia_japonica_turing_wavelength_s0/dugesia_japonica_ocellus_pigment_nogi2009.png` (826,011 B, `64` tok, SHA-256 `6abf2ae940aa3115…`)
- **Artifact:** `datasets/dugesia_japonica_turing_wavelength_s0/dugesia_japonica_field_kalininsky_ccby4.jpg` (988,342 B, `64` tok, SHA-256 `e40acb76c9286a8d…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Six Lean theorems, 0 sorry. Lean binary not installed in this VM.
- **Artifact:** `morphoeval/lean/loops/Loop_3213_dugesia_japonica_turing_wavelength_s0.lean` (1,266 B, `64` tok, SHA-256 `ccb873ba73858df3…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-dugesia-japonica-turing-wavelength-s0.yaml` (4,493 B, `1,267` tok, SHA-256 `81d384751536489f…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 1 script: saturated Schnakenberg, self-test, specimen mask render.
- **Artifact:** `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` (29,315 B, `8,354` tok, SHA-256 `ddc5e9107a9662ea…`)

### Step 6 — `output` (Output & compare)
- **Notes:** self-test PASS. Pearson r=0.99998. GIF 16 frames at 640 px.
- **Artifact:** `docs/media/dugesia_japonica_turing_wavelength_s0_comparison.png` (552,219 B, `64` tok, SHA-256 `a9ac178d5b9463ac…`)
- **Artifact:** `docs/media/dugesia_japonica_turing_wavelength_s0_growth.gif` (1,099,037 B, `64` tok, SHA-256 `810e44644592871f…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 then Pass 2; Pass 1 overall 83.2. Pass 2 overall 88.5 after R1-R4. Single assessor.
- **Artifact:** `morphoeval/audits/2026-10-27-dugesia-japonica-turing-wavelength-s0-audit.yaml` (11,591 B, `2,951` tok, SHA-256 `8369291e73a99230…`)
- **Artifact:** `morphoeval/audits/2026-10-27-dugesia-japonica-turing-wavelength-s0-audit-v2.yaml` (10,618 B, `2,788` tok, SHA-256 `48f9c5d38b956318…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 2 quick_gate 100/100, 8 measurements, Pearson r 0.99998.
- **Artifact:** `morphoeval/audits/2026-10-27-dugesia-japonica-turing-wavelength-s0-audit.measurements.yaml` (6,104 B, `1,729` tok, SHA-256 `352dc4386ecaf714…`)
- **Artifact:** `morphoeval/audits/2026-10-27-dugesia-japonica-turing-wavelength-s0-audit-v2.measurements.yaml` (6,890 B, `1,955` tok, SHA-256 `5caeb08469f4795f…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Conform OK. Rubric not re-emitted. Pass 1 had no trusty URI so Pass 2 does not claim npx:supersedes.
- **Artifact:** `morphoeval/kg/2026-10-27-dugesia-japonica-turing-wavelength-s0-audit/REPORT.md` (8,881 B, `2,394` tok, SHA-256 `ffdb2136973e850c…`)
- **Artifact:** `morphoeval/kg/2026-10-27-dugesia-japonica-turing-wavelength-s0-audit-v2/REPORT.md` (9,108 B, `2,513` tok, SHA-256 `1112d2513b7e4685…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml is absent. No trusty URI was created.

### Step 11 — `graph` (Public record)
- **Notes:** Pass 2 REPORT overall 88.5 faithful.
- **Artifact:** `morphoeval/kg/2026-10-27-dugesia-japonica-turing-wavelength-s0-audit-v2/REPORT.md` (9,108 B, `2,513` tok, SHA-256 `1112d2513b7e4685…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** R1-R4 fixed. Gates pass: image_fitness 91.4, CV 94.7, r 0.99998, 16 frames, 640 px. Pass 3 not required. R5 second assessor and R6 mask merge remain.
- **Artifact:** `morphoeval/runs/2026-10-27-dugesia-japonica-turing-wavelength-s0-loop-image-fitness.json` (2,848 B, `813` tok, SHA-256 `ab2dee90695c3190…`)

## Result

Holocene *Dugesia japonica*, age 0 Ma, chronoseries 1/1, detail scale 6.912 µm (log10 = −5.160). Saturated Schnakenberg kinetics with D_v/D_u = 48 select linear λ_c = 6.912 µm. The steady field peaks at 8.064 µm (mode 6). Equal diffusion is stable. Pearson r of linear mode growth against analytic σ(k) is 0.99998.

Pass 2 gates: image_fitness 91.4, CV similarity 94.7, 16 frames, 640 px, five specimen metrics with value ≠ reference, audit overall 88.5. Pass 3 was not required.

Signing was not done. `~/.nanopub/profile.yml` is absent, and no trusty URI was invented. R5 (second assessor) and R6 (merged body mask) stay open.
