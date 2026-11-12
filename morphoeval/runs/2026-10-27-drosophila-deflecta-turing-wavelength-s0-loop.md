# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-deflecta-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-deflecta-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila deflecta Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` (no second assessor was run)
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **176.00 s** (2.93 min) — Tier A: `110.00 s` · Tier B: `66.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **29,404** (input: `0`, output: `29,404`, subagent: `0`) — Tier A: `13,358` · Tier B: `16,046`
- **Total artifact tokens produced:** `29,404`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 18.00 | 0.00 | 0 | 1,031 | 0 | **1,031** | 1,031 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 3,211 | 0 | **3,211** | 3,211 | `drosophila_deflecta_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 8.00 | 0.00 | 0 | 64 | 0 | **64** | 64 | `drosophila_deflecta_turing_wavelength_s0_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 12.00 | 0.00 | 0 | 1,023 | 0 | **1,023** | 1,023 | `Loop_3181_drosophila_deflecta_turing_wavelength_s0.lean`, `delphi10000-drosophila-deflecta-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 7,901 | 0 | **7,901** | 7,901 | `2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 12.00 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_deflecta_turing_wavelength_s0_comparison.png`, `drosophila_deflecta_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 15.00 | 0.00 | 0 | 2,310 | 0 | **2,310** | 2,310 | `2026-10-27-drosophila-deflecta-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 8.00 | 0.00 | 0 | 3,591 | 0 | **3,591** | 3,591 | `2026-10-27-drosophila-deflecta-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-drosophila-deflecta-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 12.00 | 0.00 | 0 | 4,618 | 0 | **4,618** | 4,618 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 2.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 4.00 | 0.00 | 0 | 2,303 | 0 | **2,303** | 2,303 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 25.00 | 0.00 | 0 | 3,224 | 0 | **3,224** | 3,224 | `2026-10-27-drosophila-deflecta-turing-wavelength-s0-audit-v2.yaml`, `2026-10-27-drosophila-deflecta-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **176.00** | **0.00** | **0** | **29,404** | **0** | **29,404** | **29,404** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Commons scout. No -s0 collision. Commons had no CC still.; OpenAlex/Crossref/Commons scout. No -s0 collision. Commons had no CC still.
- **Artifact:** `datasets/drosophila_deflecta_turing_wavelength_s0/scout_manifest.json` (3,947 B, `1,031` tok, SHA-256 `5641f1b05fbb39f4…`)

### Step 2 — `paper` (Paper)
- **Notes:** Equations, 2.633e-06 m, 0 Ma, chronoseries bridge.; Equations, 2.633e-06 m, 0 Ma, chronoseries bridge.
- **Artifact:** `datasets/drosophila_deflecta_turing_wavelength_s0/drosophila_deflecta_turing_wavelength_s0_reference.json` (4,775 B, `1,289` tok, SHA-256 `1ab0abd4ee1055ae…`)
- **Artifact:** `datasets/drosophila_deflecta_turing_wavelength_s0/paper_citations.bib` (3,261 B, `1,005` tok, SHA-256 `d7dd8f5429130b0e…`)
- **Artifact:** `datasets/drosophila_deflecta_turing_wavelength_s0/README.md` (3,401 B, `917` tok, SHA-256 `cb48e211410dddc0…`)

### Step 3 — `source` (Formal source)
- **Notes:** Gallery tile vendored and SHA-256 pinned. Commons fallback.; Gallery tile vendored and SHA-256 pinned. Commons fallback.
- **Artifact:** `datasets/drosophila_deflecta_turing_wavelength_s0/drosophila_deflecta_turing_wavelength_s0_specimen.jpg` (98,667 B, `64` tok, SHA-256 `d7c1f825d40ea128…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3181, four theorems, zero sorry. Math KG yaml.; Lean Loop_3181, four theorems, zero sorry. Math KG yaml.
- **Artifact:** `morphoeval/lean/loops/Loop_3181_drosophila_deflecta_turing_wavelength_s0.lean` (1,847 B, `64` tok, SHA-256 `524cff9c66924d84…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-drosophila-deflecta-turing-wavelength-s0.yaml` (3,412 B, `959` tok, SHA-256 `c33bc8c0aada8dcd…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg k_c script with --self-test and measure_morphoeval.; Schnakenberg k_c script with --self-test and measure_morphoeval.
- **Artifact:** `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` (27,890 B, `7,901` tok, SHA-256 `2f99be53032cc33d…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 1 unforced GIF cv 82.9; pass 2 conditioned GIF cv 96.0.; Pass 1 unforced GIF cv 82.9; pass 2 conditioned GIF cv 96.0.
- **Artifact:** `docs/media/drosophila_deflecta_turing_wavelength_s0_comparison.png` (349,834 B, `64` tok, SHA-256 `93f8187057d33430…`)
- **Artifact:** `docs/media/drosophila_deflecta_turing_wavelength_s0_growth.gif` (3,727,178 B, `64` tok, SHA-256 `3b3e8657adf486b7…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 audit overall below the pass-2 gate; remediation listed.; Pass 1 audit overall below the pass-2 gate; remediation listed.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-deflecta-turing-wavelength-s0-audit.yaml` (8,821 B, `2,310` tok, SHA-256 `4f14819f32a882ab…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 1 and pass 2 measurement sidecars. Quick gate 100.; Pass 1 and pass 2 measurement sidecars. Quick gate 100.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-deflecta-turing-wavelength-s0-audit.measurements.yaml` (6,303 B, `1,741` tok, SHA-256 `56b6f496b286a03e…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-deflecta-turing-wavelength-s0-audit-v2.measurements.yaml` (6,793 B, `1,850` tok, SHA-256 `f266b93f28b533a6…`)

### Step 9 — `build` (Write as claims)
- **Notes:** KG conformant for pass 1 and pass 2. Rubric reintroduction allowed because this slot does not own the shared rubric nanopub. Not signed.; KG conformant for pass 1 and pass 2. Rubric reintroduction allowed because this slot does not own the shared rubric nanopub. Not signed.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-deflecta-turing-wavelength-s0-audit/REPORT.md` (8,578 B, `2,315` tok, SHA-256 `f59bd8651e163a29…`)
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-deflecta-turing-wavelength-s0-audit-v2/REPORT.md` (8,420 B, `2,303` tok, SHA-256 `1ff876781acc6c54…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml is absent. No trusty URI was created.; Signing unavailable: ~/.nanopub/profile.yml is absent. No trusty URI was created.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md pass 2 overall 89.5, verdict faithful.; REPORT.md pass 2 overall 89.5, verdict faithful.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-deflecta-turing-wavelength-s0-audit-v2/REPORT.md` (8,420 B, `2,303` tok, SHA-256 `1ff876781acc6c54…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2: forced fit Pearson r 0.983, RMSE 0.059, KS 0.334, cv 96.0, image_fitness 92.9. Gates passed. No pass 3.; Pass 2: forced fit Pearson r 0.983, RMSE 0.059, KS 0.334, cv 96.0, image_fitness 92.9. Gates passed. No pass 3.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-deflecta-turing-wavelength-s0-audit-v2.yaml` (9,331 B, `2,437` tok, SHA-256 `b2f4164ed8df8da3…`)
- **Artifact:** `morphoeval/runs/2026-10-27-drosophila-deflecta-turing-wavelength-s0-loop-image-fitness.json` (2,753 B, `787` tok, SHA-256 `78158a5a844b181d…`)

## Result

*Drosophila deflecta*, Holocene (0 Ma), detail-scale Turing wavelength **2.633 µm**. Predecessor *D. cyrtoloma* 5.024 µm; successor *D. elegans* 2.447 µm. Schnakenberg parameters `(a, b, D_u, D_v) = (0.132, 0.780, 0.061, 3.050)`.

Pass 2 gates: image_fitness **92.9**, cv_sim_similarity **96.0**, 12 frames, 640×480, Pearson r **0.983**, RMSE 0.059, KS 0.334, audit overall **89.5**. No pass 3.

Signing: `~/.nanopub/profile.yml` is absent, so the nanopubs were not signed and no trusty URI was invented. The draft graph uses the repository's existing publisher ORCID `https://orcid.org/0009-0007-4948-9192` (the same URI already in `morphoeval/vendor/templates`). `lean` is not installed on this VM, so the Lean file was not machine-checked.
