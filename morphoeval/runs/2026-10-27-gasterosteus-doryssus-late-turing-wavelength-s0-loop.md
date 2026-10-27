# MorphoEval Full-Loop Telemetry — `2026-10-27-gasterosteus-doryssus-late-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0`
- **Primary paper / dataset:** Gasterosteus doryssus_late Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` (single assessor; a second assessor was not run)
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.09 s** (0.02 min) — Tier A: `0.57 s` · Tier B: `0.52 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | grok-4.7 (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | grok-4.7 (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | grok-4.7 (`grok-4.7`) | `completed` | 0.13 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | grok-4.7 (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | grok-4.7 (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | grok-4.7 (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | grok-4.7 (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | grok-4.7 (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | grok-4.7 (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | grok-4.7 (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | grok-4.7 (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | grok-4.7 (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1.09** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex returned no works for the long query. Crossref resolved Hunt, Bell & Travis 2008 (10.1111/j.1558-5646.2007.00310.x). No filename collision with slot 1209.; OpenAlex returned no works for the long query. Crossref resolved Hunt, Bell & Travis 2008 (10.1111/j.1558-5646.2007.00310.x). No filename collision with slot 1209.

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, bib and README state 27 µm and 9.99 Ma. Sibling 1.7 mm parameters were not copied.; Reference JSON, bib and README state 27 µm and 9.99 Ma. Sibling 1.7 mm parameters were not copied.

### Step 3 — `source` (Formal source)
- **Notes:** Specimen is the species-gallery plate SHA-256 fd5b18be. Wikimedia hits are extant G. aculeatus and were not used as the chronospecies image.; Specimen is the species-gallery plate SHA-256 fd5b18be. Wikimedia hits are extant G. aculeatus and were not used as the chronospecies image.

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean namespace MathOfBiology.Loop_3231, six theorems, 0 sorry. lean binary not installed, so proofs were not kernel-checked.; Lean namespace MathOfBiology.Loop_3231, six theorems, 0 sorry. lean binary not installed, so proofs were not kernel-checked.

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg dispersion, discrete Laplacian, 1D ring, ablation, far-field ocellus GIF.; Schnakenberg dispersion, discrete Laplacian, 1D ring, ablation, far-field ocellus GIF.

### Step 6 — `output` (Output & compare)
- **Notes:** self-test r=0.999999 RMSE=0.309 µm. GIF 12 frames 640x480. PNG comparison written.; self-test r=0.999999 RMSE=0.309 µm. GIF 12 frames 640x480. PNG comparison written.

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 78.5. Remediation R1-R4 opened.; Pass 1 overall 78.5. Remediation R1-R4 opened.

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100/100, 10 measurements, values differ from references.; measure_outputs quick_gate 100/100, 10 measurements, values differ from references.

### Step 9 — `build` (Write as claims)
- **Notes:** KG built with --conform --allow-reintroduction --no-verify-loci. Shared rubric URI already exists in the repo.; KG built with --conform --allow-reintroduction --no-verify-loci. Shared rubric URI already exists in the repo.

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable. ~/.nanopub/profile.yml absent. Did not run nanopub setup --newkeys and did not invent published.json.; Signing unavailable. ~/.nanopub/profile.yml absent. Did not run nanopub setup --newkeys and did not invent published.json.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for the pass-1 audit.; REPORT.md present for the pass-1 audit.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 overall 86.2. CV 82.3. image_fitness 83.7. Pearson r 0.999999. All gates pass. npx:supersedes not emitted because signing never produced published.json.; Pass 2 overall 86.2. CV 82.3. image_fitness 83.7. Pearson r 0.999999. All gates pass. npx:supersedes not emitted because signing never produced published.json.

## Pass 2 gates

| Gate | Result |
|---|---|
| image_fitness >= 70 | 83.7 |
| cv_sim_similarity >= 58 | 82.3 |
| GIF frames >= 6 | 12 |
| resolution >= 340 px | 640×480 |
| non-circular specimen metric | discrete wavelength 26.77 µm vs reference 27.07 µm (and nine other measured values) |
| Pearson r > 0.98 | 0.999999 |
| audit overall >= 85 | 86.2 |

Pass 3 was not required. Nanopubs are unsigned: `~/.nanopub/profile.yml` is absent, and no signing key was created. `--supersede-from` could not attach `npx:supersedes` because that flag reads `published.json`, which only signing writes. The audit YAML chain still records pass 2 as superseding pass 1.
