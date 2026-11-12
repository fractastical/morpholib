# MorphoEval Full-Loop Telemetry — `2026-10-27-cavia-porcellus-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-cavia-porcellus-turing-wavelength-s0`
- **Primary paper / dataset:** Cavia porcellus Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` (no second-company assessor was run)
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.08 s** (0.02 min) — Tier A: `0.53 s` · Tier B: `0.56 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **38,143** (input: `0`, output: `38,143`, subagent: `0`) — Tier A: `15,078` · Tier B: `23,065`
- **Total artifact tokens produced:** `38,143`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,420 | 0 | **1,420** | 1,420 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,258 | 0 | **2,258** | 2,258 | `cavia_porcellus_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 64 | 0 | **64** | 64 | `cavia_porcellus_turing_wavelength_s0_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 894 | 0 | **894** | 894 | `Loop_3141_cavia_porcellus_turing_wavelength_s0.lean`, `delphi10000-cavia-porcellus-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 10,314 | 0 | **10,314** | 10,314 | `2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `cavia_porcellus_turing_wavelength_s0_growth.gif`, `cavia_porcellus_turing_wavelength_s0_comparison.png` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 3,295 | 0 | **3,295** | 3,295 | `2026-10-27-cavia-porcellus-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 5,249 | 0 | **5,249** | 5,249 | `2026-10-27-cavia-porcellus-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-cavia-porcellus-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 6,717 | 0 | **6,717** | 6,717 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 3,398 | 0 | **3,398** | 3,398 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 4,406 | 0 | **4,406** | 4,406 | `2026-10-27-cavia-porcellus-turing-wavelength-s0-audit-v2.yaml`, `2026-10-27-cavia-porcellus-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1.08** | **0.00** | **0** | **38,143** | **0** | **38,143** | **38,143** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Crossref DOIs confirmed; OpenAlex morphogenesis query empty; Crossref DOIs confirmed; OpenAlex morphogenesis query empty
- **Artifact:** `datasets/cavia_porcellus_turing_wavelength_s0/scout_manifest.json` (4,949 B, `1,420` tok, SHA-256 `b953e3741dab2bed…`)

### Step 2 — `paper` (Paper)
- **Notes:** Extracted Schnakenberg k_c and Swift-Hohenberg amplitude equation at 126.7 um, 0 Ma; Extracted Schnakenberg k_c and Swift-Hohenberg amplitude equation at 126.7 um, 0 Ma
- **Artifact:** `datasets/cavia_porcellus_turing_wavelength_s0/cavia_porcellus_turing_wavelength_s0_reference.json` (3,776 B, `1,059` tok, SHA-256 `6cf1e5c9b49da4a2…`)
- **Artifact:** `datasets/cavia_porcellus_turing_wavelength_s0/paper_citations.bib` (2,141 B, `692` tok, SHA-256 `15d320c60c644760…`)
- **Artifact:** `datasets/cavia_porcellus_turing_wavelength_s0/README.md` (1,809 B, `507` tok, SHA-256 `4d8fe856ee52991a…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored gallery specimen and pinned SHA-256; Vendored gallery specimen and pinned SHA-256
- **Artifact:** `datasets/cavia_porcellus_turing_wavelength_s0/cavia_porcellus_turing_wavelength_s0_specimen.jpg` (99,015 B, `64` tok, SHA-256 `42014eaf5b445ec5…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Five Lean theorems, zero sorry; math KG loop YAML; Five Lean theorems, zero sorry; math KG loop YAML
- **Artifact:** `morphoeval/lean/loops/Loop_3141_cavia_porcellus_turing_wavelength_s0.lean` (1,566 B, `64` tok, SHA-256 `db399fef94500295…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-cavia-porcellus-turing-wavelength-s0.yaml` (2,984 B, `830` tok, SHA-256 `79a6e2d2761ee604…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 2 script with sweeps, ablation, specimen segmentation, eigensolver fit; Pass 2 script with sweeps, ablation, specimen segmentation, eigensolver fit
- **Artifact:** `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` (37,036 B, `10,314` tok, SHA-256 `152b1e8b87cbfdc6…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test passed; GIF 12 frames 640x480; comparison PNG; Self-test passed; GIF 12 frames 640x480; comparison PNG
- **Artifact:** `docs/media/cavia_porcellus_turing_wavelength_s0_growth.gif` (2,459,286 B, `64` tok, SHA-256 `0057ed24394e027d…`)
- **Artifact:** `docs/media/cavia_porcellus_turing_wavelength_s0_comparison.png` (567,960 B, `64` tok, SHA-256 `ecf63291fef200a1…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 80.8 with charge, seed-bin, and eigensolver remediations; Pass 1 overall 80.8 with charge, seed-bin, and eigensolver remediations
- **Artifact:** `morphoeval/audits/2026-10-27-cavia-porcellus-turing-wavelength-s0-audit.yaml` (12,771 B, `3,295` tok, SHA-256 `7793c336cf6dc83b…`)

### Step 8 — `measure` (Measure)
- **Notes:** 14 measurements, quick_gate 100; 14 measurements, quick_gate 100
- **Artifact:** `morphoeval/audits/2026-10-27-cavia-porcellus-turing-wavelength-s0-audit.measurements.yaml` (9,617 B, `2,613` tok, SHA-256 `5b2aa4dd85708d18…`)
- **Artifact:** `morphoeval/audits/2026-10-27-cavia-porcellus-turing-wavelength-s0-audit-v2.measurements.yaml` (9,545 B, `2,636` tok, SHA-256 `b911b4978692c20d…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Both audits conform; rubric reintroduction allowed and not signed; Both audits conform; rubric reintroduction allowed and not signed
- **Artifact:** `morphoeval/kg/2026-10-27-cavia-porcellus-turing-wavelength-s0-audit/REPORT.md` (11,605 B, `3,319` tok, SHA-256 `63d79808c01bd5b4…`)
- **Artifact:** `morphoeval/kg/2026-10-27-cavia-porcellus-turing-wavelength-s0-audit-v2/REPORT.md` (11,973 B, `3,398` tok, SHA-256 `91000fc8c5cf6b15…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no ~/.nanopub/profile.yml. Nothing was signed or given a trusty URI.; Signing unavailable: no ~/.nanopub/profile.yml. Nothing was signed or given a trusty URI.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for pass 1 (80.8) and pass 2 (88.3); REPORT.md present for pass 1 (80.8) and pass 2 (88.3)
- **Artifact:** `morphoeval/kg/2026-10-27-cavia-porcellus-turing-wavelength-s0-audit-v2/REPORT.md` (11,973 B, `3,398` tok, SHA-256 `91000fc8c5cf6b15…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 audit 88.3; image_fitness 87.8; cv 88.2; all gates passed. No pass 3.; Pass 2 audit 88.3; image_fitness 87.8; cv 88.2; all gates passed. No pass 3.
- **Artifact:** `morphoeval/audits/2026-10-27-cavia-porcellus-turing-wavelength-s0-audit-v2.yaml` (13,590 B, `3,513` tok, SHA-256 `23d53beae1878330…`)
- **Artifact:** `morphoeval/runs/2026-10-27-cavia-porcellus-turing-wavelength-s0-loop-image-fitness.json` (3,137 B, `893` tok, SHA-256 `e10e07862119c4ad…`)

## Outcome

Pass 2 gates all hold. Audit overall is 88.3 (Pass 1 was 80.8). Image fitness is 87.8, with cv_sim_similarity 88.2, 12 frames, and a 640×480 GIF. Pearson r of the dispersion is 1.0 because the closed-form quadratic matches `numpy.linalg.eigvals` at roundoff (RMSE 0); that is not an independent time-integrator residual. The gallery field's scale-bar wavelength is 78.18 µm against the assigned 126.7 µm, and that difference is reported rather than fitted away.

Nanopub signing was not available: `~/.nanopub/profile.yml` is absent, so `sign_morphoeval.py` was not run and no trusty URI was minted. The shared rubric nanopub was reintroduced with `--allow-reintroduction` and must not be signed as a second introduction. Lean 4 is not installed here, so `Loop_3141` was not executed. No pass 3 was required.
