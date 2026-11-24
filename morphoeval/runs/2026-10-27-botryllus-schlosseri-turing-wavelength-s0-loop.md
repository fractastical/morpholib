# MorphoEval Full-Loop Telemetry — `2026-10-27-botryllus-schlosseri-turing-wavelength-s0-loop`

## Gates (Pass 2)

All gates pass. No Pass 3.

- image_fitness = 81.4 (0.55×83.7 + 0.30×74 + 0.15×88), threshold 70
- cv_sim_similarity = 83.7, threshold 58
- GIF frames = 12, resolution = 640×480
- non-circular specimen metrics: circularity 0.0853 versus circle reference 1; stripe wavelength 44.47 µm versus 38.72 µm
- Pearson r = 0.999998 (onset formula versus argmax σ(k) on the γ sweep), threshold 0.98
- audit overall = 85.3 (`2026-10-27-botryllus-schlosseri-turing-wavelength-s0-audit-v2`)

Pass 1 overall was 82.5. Pass 2 closed the diffusivity-ratio sweep, the u²v knockout, and the saved RMSE.

Signing was not done. `~/.nanopub/profile.yml` is absent, and `sign_morphoeval.py` also failed to import `requests` in this environment. No signature was fabricated. The knowledge-graph build did pass `--conform` for both audits.

The scored assessor is Grok 4.7, a single builder-assessor. The tracker defaults below name other models; they did not score this audit.

- **Target simulation:** `delphi10000-botryllus-schlosseri-turing-wavelength-s0`
- **Primary paper / dataset:** Botryllus schlosseri Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.30 s** (0.02 min) — Tier A: `0.58 s` · Tier B: `0.73 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **32,799** (input: `0`, output: `32,799`, subagent: `0`) — Tier A: `16,238` · Tier B: `16,561`
- **Total artifact tokens produced:** `32,799`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 0.10 | 0.00 | 0 | 1,537 | 0 | **1,537** | 1,537 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 2,578 | 0 | **2,578** | 2,578 | `botryllus_schlosseri_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 64 | 0 | **64** | 64 | `Botryllus_schlosseri_Parent1991_Banyuls.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.12 | 0.00 | 0 | 1,120 | 0 | **1,120** | 1,120 | `Loop_3136_botryllus_schlosseri_turing_wavelength_s0.lean`, `delphi10000-botryllus-schlosseri-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 9,178 | 0 | **9,178** | 9,178 | `2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 1,761 | 0 | **1,761** | 1,761 | `botryllus_schlosseri_turing_wavelength_s0_comparison.png`, `botryllus_schlosseri_turing_wavelength_s0_growth.gif`, `botryllus_schlosseri_turing_wavelength_s0_metrics.json` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.10 | 0.00 | 0 | 5,326 | 0 | **5,326** | 5,326 | `2026-10-27-botryllus-schlosseri-turing-wavelength-s0-audit.yaml`, `2026-10-27-botryllus-schlosseri-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.14 | 0.00 | 0 | 3,327 | 0 | **3,327** | 3,327 | `2026-10-27-botryllus-schlosseri-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-botryllus-schlosseri-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.14 | 0.00 | 0 | 2,457 | 0 | **2,457** | 2,457 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.13 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.13 | 0.00 | 0 | 4,653 | 0 | **4,653** | 4,653 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.09 | 0.00 | 0 | 798 | 0 | **798** | 798 | `2026-10-27-botryllus-schlosseri-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1.30** | **0.00** | **0** | **32,799** | **0** | **32,799** | **32,799** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex and Wikimedia scout. No -s0 path collision. Sibling slot 0167 left untouched.; OpenAlex and Wikimedia scout. No -s0 path collision. Sibling slot 0167 left untouched.
- **Artifact:** `datasets/botryllus_schlosseri_turing_wavelength_s0/scout_manifest.json` (5,663 B, `1,537` tok, SHA-256 `2bfcca9475555d73…`)

### Step 2 — `paper` (Paper)
- **Notes:** Equations, 39 µm, and 0 Ma pinned in the reference JSON and BibTeX. DOIs checked on OpenAlex.; Equations, 39 µm, and 0 Ma pinned in the reference JSON and BibTeX. DOIs checked on OpenAlex.
- **Artifact:** `datasets/botryllus_schlosseri_turing_wavelength_s0/botryllus_schlosseri_turing_wavelength_s0_reference.json` (4,376 B, `1,303` tok, SHA-256 `1ec57fcd143ac8d8…`)
- **Artifact:** `datasets/botryllus_schlosseri_turing_wavelength_s0/paper_citations.bib` (1,849 B, `596` tok, SHA-256 `b6ff531c2322eb99…`)
- **Artifact:** `datasets/botryllus_schlosseri_turing_wavelength_s0/README.md` (2,656 B, `679` tok, SHA-256 `c05a3537622190c0…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored Géry Parent 1991 public-domain colony photograph from Wikimedia Commons. SHA-256 pinned.; Vendored Géry Parent 1991 public-domain colony photograph from Wikimedia Commons. SHA-256 pinned.
- **Artifact:** `datasets/botryllus_schlosseri_turing_wavelength_s0/Botryllus_schlosseri_Parent1991_Banyuls.jpg` (2,365,686 B, `64` tok, SHA-256 `e351975e1a0b4822…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean module Loop_3136 with four theorems and zero sorry, plus the math KG yaml.; Lean module Loop_3136 with four theorems and zero sorry, plus the math KG yaml.
- **Artifact:** `morphoeval/lean/loops/Loop_3136_botryllus_schlosseri_turing_wavelength_s0.lean` (1,758 B, `64` tok, SHA-256 `c5289a4e4c32ba71…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-botryllus-schlosseri-turing-wavelength-s0.yaml` (3,722 B, `1,056` tok, SHA-256 `1544fb2e0dcfa083…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg–Turing script with --self-test. Pass 2 added the ratio sweep and the u^2 v knockout.; Schnakenberg–Turing script with --self-test. Pass 2 added the ratio sweep and the u^2 v knockout.
- **Artifact:** `models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py` (31,402 B, `9,178` tok, SHA-256 `45816bf1016eed16…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Comparison PNG and 12-frame 640x480 GIF. Self-test passed. Pearson r = 0.999998.; Comparison PNG and 12-frame 640x480 GIF. Self-test passed. Pearson r = 0.999998.
- **Artifact:** `docs/media/botryllus_schlosseri_turing_wavelength_s0_comparison.png` (646,369 B, `64` tok, SHA-256 `aaf2c0b9f3b67edc…`)
- **Artifact:** `docs/media/botryllus_schlosseri_turing_wavelength_s0_growth.gif` (2,857,505 B, `64` tok, SHA-256 `55f5711ce0e622c7…`)
- **Artifact:** `datasets/botryllus_schlosseri_turing_wavelength_s0/botryllus_schlosseri_turing_wavelength_s0_metrics.json` (4,972 B, `1,633` tok, SHA-256 `3da7872dbb8d2fa6…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 82.5. Pass 2 overall 85.3 after the three remediations.; Pass 1 overall 82.5. Pass 2 overall 85.3 after the three remediations.
- **Artifact:** `morphoeval/audits/2026-10-27-botryllus-schlosseri-turing-wavelength-s0-audit.yaml` (10,010 B, `2,568` tok, SHA-256 `6f7c715d3de0d2fe…`)
- **Artifact:** `morphoeval/audits/2026-10-27-botryllus-schlosseri-turing-wavelength-s0-audit-v2.yaml` (10,478 B, `2,758` tok, SHA-256 `2eb99a66f3d81239…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs.py quick_gate 100/100 on both audits. Six Pass-2 measurements, none self-referential.; measure_outputs.py quick_gate 100/100 on both audits. Six Pass-2 measurements, none self-referential.
- **Artifact:** `morphoeval/audits/2026-10-27-botryllus-schlosseri-turing-wavelength-s0-audit.measurements.yaml` (5,351 B, `1,570` tok, SHA-256 `05fd91692b94613c…`)
- **Artifact:** `morphoeval/audits/2026-10-27-botryllus-schlosseri-turing-wavelength-s0-audit-v2.measurements.yaml` (6,120 B, `1,757` tok, SHA-256 `9e9f6d922143f879…`)

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg.py --conform OK for v1 and v2. --no-verify-loci because the new files are not on the base commit. --allow-reintroduction because the shared rubric trig already exists in older audits.; build_morphoeval_kg.py --conform OK for v1 and v2. --no-verify-loci because the new files are not on the base commit. --allow-reintroduction because the shared rubric trig already exists in older audits.
- **Artifact:** `morphoeval/kg/2026-10-27-botryllus-schlosseri-turing-wavelength-s0-audit-v2/REPORT.md` (8,974 B, `2,457` tok, SHA-256 `d3e85c6e1538b75b…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable. ~/.nanopub/profile.yml is absent. sign_morphoeval.py was not able to sign. No signature was fabricated.; Signing unavailable. ~/.nanopub/profile.yml is absent. sign_morphoeval.py was not able to sign. No signature was fabricated.

### Step 11 — `graph` (Public record)
- **Notes:** Pass 2 REPORT.md records overall 85.3, verdict faithful.; Pass 2 REPORT.md records overall 85.3, verdict faithful.
- **Artifact:** `morphoeval/kg/2026-10-27-botryllus-schlosseri-turing-wavelength-s0-audit-v2/REPORT.md` (8,974 B, `2,457` tok, SHA-256 `d3e85c6e1538b75b…`)
- **Artifact:** `morphoeval/kg/2026-10-27-botryllus-schlosseri-turing-wavelength-s0-audit/REPORT.md` (8,184 B, `2,196` tok, SHA-256 `a61acdd1ee4e34bf…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed R1 ratio sweep, R2 autocatalysis knockout, and R3 RMSE. Image fitness 81.4, CV 83.7, Pearson r 0.999998. All gates pass, so no Pass 3.; Pass 2 closed R1 ratio sweep, R2 autocatalysis knockout, and R3 RMSE. Image fitness 81.4, CV 83.7, Pearson r 0.999998. All gates pass, so no Pass 3.
- **Artifact:** `morphoeval/runs/2026-10-27-botryllus-schlosseri-turing-wavelength-s0-loop-image-fitness.json` (2,741 B, `798` tok, SHA-256 `5c37320748e4454c…`)
