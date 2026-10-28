# MorphoEval Full-Loop Telemetry — `2026-10-28-t1917-054-on-the-theory-of-transformatio-loop`

- **Target simulation:** `thompson1917-t1917-054`
- **Primary paper / dataset:** Thompson, D.W. (1917) On Growth and Form, Ch. XVII, pp. 757-759, Figs. 388-393
- **Builder model:** `claude-fable-5-1`
- **Assessor models:** `claude-fable-5-1` + `claude-sonnet-5-5-high`
- **Start commit:** `deb2a5348`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1371.22 s** (22.85 min) — Tier A: `1340.16 s` · Tier B: `31.06 s`
- **Total CPU time:** `79.91 s` user + `1.87 s` sys
- **Total LLM tokens:** **469,200** (input: `392,000`, output: `77,200`, subagent: `0`) — Tier A: `247,500` · Tier B: `221,700`
- **Total artifact tokens produced:** `85,316`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 494.64 | 0.00 | 60,000 | 6,000 | 0 | **66,000** | 3,081 | `scout_manifest.json`, `ongrowthform1917thom_p760_leaf0779_fig393.jpg` |
| 2 | A | **paper** (Paper) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 171.94 | 0.00 | 25,000 | 7,000 | 0 | **32,000** | 5,795 | `t1917_054_on_the_theory_of_transformatio_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 0.10 | 0.00 | 8,000 | 1,500 | 0 | **9,500** | 1,144 | `archaeopteryx_lithographica_berlin_specimen_mfn.jpg`, `apatornis_celer_holotype_synsacrum_marsh1880.jpg`, `dryad_k1v34cf_deeming_mayr_2018_metadata.json` |
| 4 | A | **rules** (Pin the rules) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 255.90 | 0.00 | 40,000 | 9,000 | 0 | **49,000** | 2,863 | `Thompson1917_T1917_054.lean`, `thompson1917-t1917-054.yaml` |
| 5 | A | **script** (Script) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 381.39 | 0.00 | 70,000 | 14,000 | 0 | **84,000** | 11,765 | `t1917-054_on-the-theory-of-transformations.py` |
| 6 | A | **output** (Output & compare) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 36.20 | 44.84 | 6,000 | 1,000 | 0 | **7,000** | 4,751 | `t1917_054_on_the_theory_of_transformatio_comparison.png`, `t1917_054_on_the_theory_of_transformatio_growth.gif`, `result.json` |
| 7 | B | **audit** (Dual-agent audit) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 0.17 | 0.00 | 65,000 | 13,000 | 0 | **78,000** | 9,745 | `2026-10-28-t1917-054-on-the-theory-of-transformatio-audit.yaml`, `2026-10-28-t1917-054-on-the-theory-of-transformatio-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 27.17 | 33.37 | 4,000 | 600 | 0 | **4,600** | 5,054 | `2026-10-28-t1917-054-on-the-theory-of-transformatio-audit.measurements.yaml`, `2026-10-28-t1917-054-on-the-theory-of-transformatio-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 1.88 | 1.84 | 11,000 | 1,500 | 0 | **12,500** | 8,239 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 1.57 | 1.73 | 7,000 | 1,000 | 0 | **8,000** | 1,244 | `published.json`, `published.json` |
| 11 | B | **graph** (Public record) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 0.17 | 0.00 | 6,000 | 600 | 0 | **6,600** | 8,239 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 0.08 | 0.00 | 90,000 | 22,000 | 0 | **112,000** | 23,396 | `t1917-054_on-the-theory-of-transformations.py`, `thompson1917-t1917-054.yaml` |
| — | **All** | **TOTAL** | — | `12/12` | **1371.22** | **81.78** | **392,000** | **77,200** | **0** | **469,200** | **85,316** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Verified pre-scouted Dryad k1v34cf = variable mismatch (1-D pelvic canal width; file behind browser challenge). Verdict data = Thompson's own Figs 388-393 from IA scan ongrowthform1917thom (pp. 757-760 vendored + SHA-256 pinned). Commons: Berlin Archaeopteryx (CC BY-SA 3.0), Marsh 1880 Apatornis synsacrum (PD).
- **Artifact:** `datasets/thompson_1917/t1917-054/scout_manifest.json` (11,273 B, `3,017` tok, SHA-256 `6a8a3e51bdde2a4a…`)
- **Artifact:** `datasets/thompson_1917/t1917-054/ongrowthform1917thom_p760_leaf0779_fig393.jpg` (447,349 B, `64` tok, SHA-256 `0e73dc9e37ed7e64…`)

### Step 2 — `paper` (Paper)
- **Notes:** E1-E7 (grid map, equidistant interpolation, inscribed form, TPS realisation, affine control, linear-morph identity, t* statistic); claims C1-C4 with C4 (biological reality) flagged untestable.
- **Artifact:** `datasets/thompson_1917/t1917-054/t1917_054_on_the_theory_of_transformatio_reference.json` (11,136 B, `3,097` tok, SHA-256 `92bc0f5512fe8f9d…`)
- **Artifact:** `datasets/thompson_1917/t1917-054/paper_citations.bib` (4,048 B, `1,129` tok, SHA-256 `6a29f62000543207…`)
- **Artifact:** `datasets/thompson_1917/t1917-054/README.md` (5,667 B, `1,569` tok, SHA-256 `3b551873e61ee77a…`)

### Step 3 — `source` (Formal source)
- **Notes:** 7 inputs SHA-256 pinned & verified: 4 IA page scans (PD), Berlin Archaeopteryx (CC BY-SA 3.0, derivative), Marsh 1880 Apatornis synsacrum (PD), Dryad metadata (CC0).
- **Artifact:** `datasets/thompson_1917/t1917-054/archaeopteryx_lithographica_berlin_specimen_mfn.jpg` (487,953 B, `64` tok, SHA-256 `96341aed3d81b2b6…`)
- **Artifact:** `datasets/thompson_1917/t1917-054/apatornis_celer_holotype_synsacrum_marsh1880.jpg` (192,503 B, `64` tok, SHA-256 `28213a31e4c88718…`)
- **Artifact:** `datasets/thompson_1917/t1917-054/dryad_k1v34cf_deeming_mayr_2018_metadata.json` (4,198 B, `1,016` tok, SHA-256 `4f2b721ee3bec249…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean 4.35.0: 11 theorems (equidistance, similarity equivariance, no-fold, betweenness, t* soundness, affine det endpoints, shear area, scaling orientation) compile with 0 errors, 0 sorry; 5 KG laws registered.
- **Artifact:** `morphoeval/lean/loops/Thompson1917_T1917_054.lean` (9,162 B, `64` tok, SHA-256 `95fa666846395c1d…`)
- **Artifact:** `morphoeval/math_kg_loops/thompson1917-t1917-054.yaml` (9,299 B, `2,799` tok, SHA-256 `b39f26bfbb533cb0…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 1 v1 (~560 lines): Fig.393 segmentation, auto landmarks (acetabulum + 4 tips), arc-length semilandmarks, GPA, TPS phi + affine control, equidistant interpolation, t*, Procrustes comparisons, figures, result.json, measure_morphoeval, 10-check self-test.
- **Artifact:** `thompson/claims/t1917-054_on-the-theory-of-transformations.py` (40,067 B, `11,765` tok, SHA-256 `215b985683b4e5b9…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 1 render: self-test 10/10 (1.4 s); full run 3.8 s; t*=0.263/0.529/0.824; d(pred,drawn)=0.134/0.121/0.093 vs nearest endpoint 0.214/0.312/0.145; Pass 2 render: self-test 16/16 (2.7 s); full run incl. 24-setting sweep 28 s
- **Artifact:** `docs/media/t1917_054_on_the_theory_of_transformatio_comparison.png` (1,061,469 B, `64` tok, SHA-256 `a30bd5dd44fca703…`)
- **Artifact:** `docs/media/t1917_054_on_the_theory_of_transformatio_growth.gif` (601,044 B, `64` tok, SHA-256 `91bbcf8957c6ce38…`)
- **Artifact:** `datasets/thompson_1917/t1917-054/result.json` (16,321 B, `4,623` tok, SHA-256 `2a03419b94cc0dcb…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 audit: 7 dimensions, 19 line-pinned evidence items (5 supports:false), 5 needs, 6 remediation items; overall 68.0/100; Pass 2 audit v2 (supersedes v1): 7 dimensions, 20 line-pinned evidence items, 4 needs resolved / 2 open, overall 83.2/100 (+14.4)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-054-on-the-theory-of-transformatio-audit.yaml` (17,021 B, `4,354` tok, SHA-256 `e3be3142b6767cc1…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-054-on-the-theory-of-transformatio-audit-v2.yaml` (20,523 B, `5,391` tok, SHA-256 `30466eb974d30f01…`)

### Step 8 — `measure` (Measure)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-054-on-the-theory-of-transformatio-audit.measurements.yaml` (7,005 B, `2,171` tok, SHA-256 `6c92b472c00ee0d2…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-054-on-the-theory-of-transformatio-audit-v2.measurements.yaml` (9,449 B, `2,883` tok, SHA-256 `a94570b7c6431932…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Pass 1 KG: 9 nanopubs + ttl + REPORT.md; rubric reused from bos-taurus v7 (--rubric-from); fixed Clarke DOI URI-encoding and code_locus line-range format; Pass 2 KG: supersedes Pass 1 nanopubs (--supersede-from, --no-resolve-heads), rubric reused, Delphi conformance
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-054-on-the-theory-of-transformatio-audit/REPORT.md` (13,226 B, `3,757` tok, SHA-256 `c1e3025454bfd367…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-054-on-the-theory-of-transformatio-audit-v2/REPORT.md` (15,476 B, `4,482` tok, SHA-256 `cb59c64540651c65…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Pass 1: 9 nanopubs signed locally (sign-only, NOT published) with a loop-local ephemeral RSA key (repo production key unavailable in the cloud VM); node check_signed.mjs verification; Pass 2: 9 superseding nanopubs signed locally (loop-local ephemeral key, sign-only, NOT published) and verified with check_signed.mjs
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-054-on-the-theory-of-transformatio-audit/published.json` (1,694 B, `624` tok, SHA-256 `2656082dbf2f9b15…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-054-on-the-theory-of-transformatio-audit-v2/published.json` (1,694 B, `620` tok, SHA-256 `7cd7f8c63d296ce1…`)

### Step 11 — `graph` (Public record)
- **Notes:** Pass 1 REPORT.md verified: overall 68.8/100 (mostly faithful), 9 signed nanopubs verified 9 ok / 0 failed via check_signed.mjs; Pass 2 REPORT.md verified: overall 83.1/100 (faithful), supersedes Pass 1 (68.8); 9 signed nanopubs verified 9 ok / 0 failed; Pass 1 URIs unpublished so no npx:supersedes triple is emitted (supersession recorded in audit + description)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-054-on-the-theory-of-transformatio-audit/REPORT.md` (13,226 B, `3,757` tok, SHA-256 `c1e3025454bfd367…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-054-on-the-theory-of-transformatio-audit-v2/REPORT.md` (15,476 B, `4,482` tok, SHA-256 `cb59c64540651c65…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 remediation (R2-R6): 6x4 parameter sweep, pairing permutation + correspondence-shift null (exact p), IoU/boundary-RMSE/Pearson r/KS GOF, Jacobian no-fold check (dense TPS folds; 9-anchor coarse grid fold-free), Berlin specimen CV pass (descriptive), 16-check self-test, enlarged 15-panel figure; script 1193 lines
- **Artifact:** `thompson/claims/t1917-054_on-the-theory-of-transformations.py` (72,237 B, `20,507` tok, SHA-256 `af3683214cc66bc8…`)
- **Artifact:** `morphoeval/math_kg_loops/thompson1917-t1917-054.yaml` (9,511 B, `2,889` tok, SHA-256 `ec4b600f21673155…`)
