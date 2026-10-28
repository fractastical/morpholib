# MorphoEval Full-Loop Telemetry — `2026-10-28-t1917-023-on-concretions-spicules-and-sp-loop`

- **Target simulation:** `thompson1917-t1917-023`
- **Primary paper / dataset:** Thompson, D.W. (1917) On Growth and Form, Ch. IX, pp. 411-487, Figs. 194-235
- **Builder model:** `claude-fable-5-1`
- **Assessor models:** `claude-fable-5-1` + `claude-sonnet-5-5-high`
- **Start commit:** `fd43e5aba`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **3443.73 s** (57.40 min) — Tier A: `1262.89 s` · Tier B: `2180.84 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **85,110** (input: `0`, output: `85,110`, subagent: `0`) — Tier A: `46,399` · Tier B: `38,711`
- **Total artifact tokens produced:** `77,599`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 546.15 | 0.00 | 0 | 5,106 | 0 | **5,106** | 5,106 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 207.75 | 0.00 | 0 | 6,953 | 0 | **6,953** | 6,953 | `t1917_023_on_concretions_spicules_and_sp_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 0.10 | 0.00 | 0 | 320 | 0 | **320** | 320 | `thompson_1917_p421_figs195-196_harting_calcospherites.jpg`, `thompson_1917_p422_fig197_aggregated_calcospherites.jpg`, `thompson_1917_p432_fig208-209_spherocrystal_otoliths.jpg`, +2 more |
| 4 | A | **rules** (Pin the rules) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 170.68 | 0.00 | 0 | 3,192 | 0 | **3,192** | 3,192 | `Thompson1917_T1917_023.lean`, `thompson1917-t1917-023.yaml` |
| 5 | A | **script** (Script) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 338.02 | 0.00 | 0 | 29,373 | 0 | **29,373** | 21,990 | `t1917-023_on-concretions-spicules-and-spicular-skeletons.py`, `result.json` |
| 6 | A | **output** (Output & compare) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 0.19 | 0.00 | 0 | 1,455 | 0 | **1,455** | 1,327 | `t1917_023_on_concretions_spicules_and_sp_comparison.png`, `t1917_023_on_concretions_spicules_and_sp_growth.gif`, `result.json` |
| 7 | B | **audit** (Dual-agent audit) | claude-fable-5-1 | `completed` | 140.07 | 0.00 | 0 | 9,394 | 0 | **9,394** | 9,394 | `2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit.yaml`, `2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | claude-fable-5-1 | `completed` | 62.13 | 0.00 | 0 | 5,460 | 0 | **5,460** | 5,460 | `2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit.measurements.yaml`, `2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | claude-fable-5-1 | `completed` | 103.71 | 0.00 | 0 | 8,026 | 0 | **8,026** | 8,026 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | claude-fable-5-1 | `completed` | 11.01 | 0.00 | 0 | 1,249 | 0 | **1,249** | 1,249 | `published.json`, `published.json` |
| 11 | B | **graph** (Public record) | claude-fable-5-1 | `completed` | 10.01 | 0.00 | 0 | 8,026 | 0 | **8,026** | 8,026 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | claude-fable-5-1 | `completed` | 1853.91 | 0.00 | 0 | 6,556 | 0 | **6,556** | 6,556 | `2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit-v2.yaml`, `result.json` |
| — | **All** | **TOTAL** | — | `12/12` | **3443.73** | **0.00** | **0** | **85,110** | **0** | **85,110** | **77,599** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Verified IA scan ongrowthform1917thom pages 421/422/432 (Harting Figs 195-197, 208, otoliths Fig 209) match claim; analogues: Carmel ooids (PD), Liesegang rings (CC BY-SA 3.0). MATCH, no collision.
- **Artifact:** `datasets/thompson_1917/t1917-023/scout_manifest.json` (19,365 B, `5,106` tok, SHA-256 `89dbc647da5d51c9…`)

### Step 2 — `paper` (Paper)
- **Notes:** Equations E1-E6, state variables, BCs, 5-regime eta table, Liesegang parameters, 8 empirical reference values, 6 textual claims C1-C6 extracted from IA OCR pp. 420-437.
- **Artifact:** `datasets/thompson_1917/t1917-023/t1917_023_on_concretions_spicules_and_sp_reference.json` (12,180 B, `3,525` tok, SHA-256 `23dafa48d0e5c552…`)
- **Artifact:** `datasets/thompson_1917/t1917-023/paper_citations.bib` (6,845 B, `2,070` tok, SHA-256 `7a05770584be0d5e…`)
- **Artifact:** `datasets/thompson_1917/t1917-023/README.md` (5,074 B, `1,358` tok, SHA-256 `c183d6a78fd27191…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored 3 PD IA plates (pp. 421, 422, 432) + 2 CC/PD Commons analogue specimens; SHA-256 pinned in reference JSON vendored_files and scout_manifest.
- **Artifact:** `datasets/thompson_1917/t1917-023/thompson_1917_p421_figs195-196_harting_calcospherites.jpg` (651,734 B, `64` tok, SHA-256 `f05b4d1f9ce006b8…`)
- **Artifact:** `datasets/thompson_1917/t1917-023/thompson_1917_p422_fig197_aggregated_calcospherites.jpg` (740,028 B, `64` tok, SHA-256 `65eabed4733a09dd…`)
- **Artifact:** `datasets/thompson_1917/t1917-023/thompson_1917_p432_fig208-209_spherocrystal_otoliths.jpg` (651,924 B, `64` tok, SHA-256 `cbfffad55d3f82b7…`)
- **Artifact:** `datasets/thompson_1917/t1917-023/carmel_ooids_thin_section_specimen.jpg` (366,785 B, `64` tok, SHA-256 `3f1e500de229723d…`)
- **Artifact:** `datasets/thompson_1917/t1917-023/liesegang_rings_silver_chromate_gelatine_specimen.jpg` (46,771 B, `64` tok, SHA-256 `2f41645ff68f6000…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean 4.35.0 compiles with 0 errors / 0 sorry: 11 theorems across 5 laws (constant lamella + semigroup, parabolic thinning, asphericity non-increasing, impingement plane affine + midpoint, geometric band gap). Math KG yaml registers 5 equations.
- **Artifact:** `morphoeval/lean/loops/Thompson1917_T1917_023.lean` (10,419 B, `64` tok, SHA-256 `40d6f14e64767d7e…`)
- **Artifact:** `morphoeval/math_kg_loops/thompson1917-t1917-023.yaml` (10,495 B, `3,128` tok, SHA-256 `6462c0dac0f84ed3…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 2 v2; Pass 2 v2: eta sweep (5), b0 sweep (4), anisotropic + no-gap controls, CV on Figs 195/196/197/208 + ooid + Liesegang specimen, same-pipeline Voronoi (hard-core centres), r/RMSE/KS/CV stats, 21 metrics, 16-check self-test, result.json verdict
- **Artifact:** `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py` (69,671 B, `20,791` tok, SHA-256 `42a978450aba357e…`)
- **Artifact:** `datasets/thompson_1917/t1917-023/result.json` (4,094 B, `1,199` tok, SHA-256 `21ca4470ecd94e5f…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 2; self-test 16/16 PASS; full run ~55 s; verdict supported 7/7
- **Artifact:** `docs/media/t1917_023_on_concretions_spicules_and_sp_comparison.png` (2,054,366 B, `64` tok, SHA-256 `f4073dac2bba015a…`)
- **Artifact:** `docs/media/t1917_023_on_concretions_spicules_and_sp_growth.gif` (328,103 B, `64` tok, SHA-256 `98bd60bda3486f0f…`)
- **Artifact:** `datasets/thompson_1917/t1917-023/result.json` (4,094 B, `1,199` tok, SHA-256 `21ca4470ecd94e5f…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 2; Pass 2 v2 audit superseding v1: 7 dimensions, 22 line-pinned evidence items, 3 residual remediation items
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit.yaml` (15,717 B, `4,037` tok, SHA-256 `9289f164c1125e7c…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit-v2.yaml` (20,211 B, `5,357` tok, SHA-256 `6a975b0008943f74…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 2; 21 measurements (19 with references), quick_gate 100/100 validated
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit.measurements.yaml` (6,016 B, `1,821` tok, SHA-256 `b317ec6dc4bedb50…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit-v2.measurements.yaml` (12,536 B, `3,639` tok, SHA-256 `bf3eafdb9e01e488…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Pass 2; v2: 9 nanopubs, 889 triples, conformance OK, overall 86.0 (faithful, up from 56.5); rubric not re-emitted (identical to bos-taurus-v7 head); v1 nanopubs unpublished so no npx:supersedes needed, chain recorded via supersedes_audit_id
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit/REPORT.md` (11,516 B, `3,223` tok, SHA-256 `302d3b02442d6d8e…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit-v2/REPORT.md` (16,596 B, `4,803` tok, SHA-256 `f6ca5145d6df9317…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Pass 2; 9 v2 nanopubs signed, @nanopub/sign verification 9 ok / 0 failed, not published
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit/published.json` (1,694 B, `616` tok, SHA-256 `1704dc9c9ed7e281…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit-v2/published.json` (1,694 B, `633` tok, SHA-256 `2a7d0ad8ec572af5…`)

### Step 11 — `graph` (Public record)
- **Notes:** Pass 2; v2 REPORT verified: 86.0 faithful, supersession chain v1 -> v2 recorded, 3 residual remediation items
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit/REPORT.md` (11,516 B, `3,223` tok, SHA-256 `302d3b02442d6d8e…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit-v2/REPORT.md` (16,596 B, `4,803` tok, SHA-256 `f6ca5145d6df9317…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 remediation; Closed loop: 56.5 -> 86.0 (+29.5); verdict SUPPORTED 7/7 written to result.json
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit-v2.yaml` (20,211 B, `5,357` tok, SHA-256 `6a975b0008943f74…`)
- **Artifact:** `datasets/thompson_1917/t1917-023/result.json` (4,094 B, `1,199` tok, SHA-256 `21ca4470ecd94e5f…`)
