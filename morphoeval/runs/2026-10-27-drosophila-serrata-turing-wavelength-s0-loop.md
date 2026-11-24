# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-serrata-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-serrata-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila serrata Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-run`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **0.92 s** (0.02 min) — Tier A: `0.46 s` · Tier B: `0.46 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **30,153** (input: `0`, output: `30,153`, subagent: `0`) — Tier A: `13,271` · Tier B: `16,882`
- **Total artifact tokens produced:** `30,153`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 297 | 0 | **297** | 297 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 3,198 | 0 | **3,198** | 3,198 | `drosophila_serrata_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 64 | 0 | **64** | 64 | `drosophila_serrata_turing_wavelength_s0_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 935 | 0 | **935** | 935 | `Loop_3203_drosophila_serrata_turing_wavelength_s0.lean`, `delphi10000-drosophila-serrata-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 8,649 | 0 | **8,649** | 8,649 | `2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_serrata_turing_wavelength_s0_comparison.png`, `drosophila_serrata_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.07 | 0.00 | 0 | 5,323 | 0 | **5,323** | 5,323 | `2026-10-27-drosophila-serrata-turing-wavelength-s0-audit.yaml`, `2026-10-27-drosophila-serrata-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.08 | 0.00 | 0 | 3,195 | 0 | **3,195** | 3,195 | `2026-10-27-drosophila-serrata-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-drosophila-serrata-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.08 | 0.00 | 0 | 5,001 | 0 | **5,001** | 5,001 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.08 | 0.00 | 0 | 2,420 | 0 | **2,420** | 2,420 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.08 | 0.00 | 0 | 943 | 0 | **943** | 943 | `2026-10-27-drosophila-serrata-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **0.92** | **0.00** | **0** | **30,153** | **0** | **30,153** | **30,153** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Artifact:** `datasets/drosophila_serrata_turing_wavelength_s0/scout_manifest.json` (1,115 B, `297` tok, SHA-256 `c7e0fab1e5c50d1c…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, bib, README with 2.123 um and 0 Ma
- **Artifact:** `datasets/drosophila_serrata_turing_wavelength_s0/drosophila_serrata_turing_wavelength_s0_reference.json` (5,689 B, `1,561` tok, SHA-256 `cf9d347264532138…`)
- **Artifact:** `datasets/drosophila_serrata_turing_wavelength_s0/paper_citations.bib` (2,696 B, `849` tok, SHA-256 `019ebb4919ea13e0…`)
- **Artifact:** `datasets/drosophila_serrata_turing_wavelength_s0/README.md` (2,924 B, `788` tok, SHA-256 `17d97cd94ff686fc…`)

### Step 3 — `source` (Formal source)
- **Notes:** Gallery fallback SHA-256 pinned; no CC single-animal photo on Commons/iNat/GBIF
- **Artifact:** `datasets/drosophila_serrata_turing_wavelength_s0/drosophila_serrata_turing_wavelength_s0_specimen.jpg` (96,676 B, `64` tok, SHA-256 `a7fa07bfd7ede1c0…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3203, 5 theorems, 0 sorry
- **Artifact:** `morphoeval/lean/loops/Loop_3203_drosophila_serrata_turing_wavelength_s0.lean` (2,073 B, `64` tok, SHA-256 `3d073c3a73503f77…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-drosophila-serrata-turing-wavelength-s0.yaml` (3,354 B, `871` tok, SHA-256 `fab2e4a1c20dd399…`)

### Step 5 — `script` (Script)
- **Artifact:** `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` (29,265 B, `8,649` tok, SHA-256 `ff9aaf3de1215a46…`)

### Step 6 — `output` (Output & compare)
- **Artifact:** `docs/media/drosophila_serrata_turing_wavelength_s0_comparison.png` (387,124 B, `64` tok, SHA-256 `69278c46b9775589…`)
- **Artifact:** `docs/media/drosophila_serrata_turing_wavelength_s0_growth.gif` (3,313,138 B, `64` tok, SHA-256 `2fbf667000d2f0d9…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 86.4; Pass 2 overall 89.0
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-serrata-turing-wavelength-s0-audit.yaml` (11,026 B, `2,806` tok, SHA-256 `c7fac511d94bb619…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-serrata-turing-wavelength-s0-audit-v2.yaml` (9,695 B, `2,517` tok, SHA-256 `33c6c05508c5d3ca…`)

### Step 8 — `measure` (Measure)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-serrata-turing-wavelength-s0-audit.measurements.yaml` (5,797 B, `1,541` tok, SHA-256 `1883c167fe618ace…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-serrata-turing-wavelength-s0-audit-v2.measurements.yaml` (6,221 B, `1,654` tok, SHA-256 `6feba6614a9addd0…`)

### Step 9 — `build` (Write as claims)
- **Notes:** conformance OK both passes
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-serrata-turing-wavelength-s0-audit/REPORT.md` (9,555 B, `2,581` tok, SHA-256 `d2888c03151b641d…`)
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-serrata-turing-wavelength-s0-audit-v2/REPORT.md` (8,866 B, `2,420` tok, SHA-256 `083e3235d09ad0d3…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no ~/.nanopub/profile.yml. Graphs were not signed.; Honest skip: nanopub profile absent, no trusty URI minted.

### Step 11 — `graph` (Public record)
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-serrata-turing-wavelength-s0-audit-v2/REPORT.md` (8,866 B, `2,420` tok, SHA-256 `083e3235d09ad0d3…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed R1 and R3. R2 second assessor remains open. Gates passed; no Pass 3.
- **Artifact:** `morphoeval/runs/2026-10-27-drosophila-serrata-turing-wavelength-s0-loop-image-fitness.json` (3,241 B, `943` tok, SHA-256 `2c5d0e3690fac8c1…`)

## Scientific result (slot 3203)

*Drosophila serrata*, Holocene, 0 Ma, detail scale 2.123 µm. Schnakenberg activator diffusivity is fixed so the band-center wavenumber is that wavelength. Raising the inhibitor ratio shortens λ along the chronoseries: *D. sechellia* 2134 nm, *D. serrata* 2123 nm, *D. silvestris* 2102 nm.

Pass 1 audit overall 86.4. Pass 2 overall 89.0 after marking k_max (k_max/k_c = 0.975, nonlinear FFT peak / k_max = 0.976) and stating that the gallery plate's bar reads 100 µm. Pearson r of RK2 versus the analytic dispersion is 0.99998. GIF is 12 frames at 640×520. CV similarity versus the gallery plate is 94.1. Image fitness on Pass 2 is 91.1 = 0.55×94.1 + 0.30×86 + 0.15×90.

## Gates after Pass 2

| Gate | Result |
|---|---|
| image_fitness ≥ 70 | 91.1 pass |
| cv_sim_similarity ≥ 58 | 94.1 pass |
| GIF frames ≥ 6 | 12 pass |
| resolution ≥ 340 px | 640×520 pass |
| non-circular specimen metric | coherency 0.775 vs 0.1177; orientation KS 0.0714 vs 0.5 |
| Pearson r > 0.98 | 0.99998 pass |
| audit overall ≥ 85 | 89.0 pass |

Pass 3 was not required. R2 stays open: no second assessor. Signing was not done: `~/.nanopub/profile.yml` is absent, so no trusty URI was minted.
