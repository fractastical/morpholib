# MorphoEval Full-Loop Telemetry — `2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-loop`

- **Target simulation:** `thompson1917-t1917-034`
- **Primary paper / dataset:** Thompson, D.W. (1917) On Growth and Form, Ch. XV, pp. 652-669, Figs. 328-330
- **Builder model:** `claude-fable-5.1`
- **Assessor models:** `claude-fable-5.1` + `claude-fable-5.1 (Pass 2 re-score)`
- **Start commit:** `fd43e5aba`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **3306.77 s** (55.11 min) — Tier A: `1849.38 s` · Tier B: `1457.39 s`
- **Total CPU time:** `132.36 s` user + `1.94 s` sys
- **Total LLM tokens:** **366,693** (input: `935`, output: `365,758`, subagent: `0`) — Tier A: `334,128` · Tier B: `32,565`
- **Total artifact tokens produced:** `373,767`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | cloud-agent (`claude-fable-5.1`) | `completed` | 1119.74 | 0.00 | 0 | 4,220 | 0 | **4,220** | 4,220 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | cloud-agent (`claude-fable-5.1`) | `completed` | 225.81 | 0.00 | 0 | 8,054 | 0 | **8,054** | 8,054 | `t1917_034_on_the_shapes_of_eggs_and_of_c_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | cloud-agent (`claude-fable-5.1`) | `completed` | 0.12 | 0.00 | 0 | 289,585 | 0 | **289,585** | 289,585 | `biggins2018_data_for_comparing_indices.csv`, `biggins2018_C126_uria_aalge_silhouette.jpg`, `mwnh_0481_uria_aalge_eggs.jpg`, +4 more |
| 4 | A | **rules** (Pin the rules) | cloud-agent (`claude-fable-5.1`) | `completed` | 424.67 | 0.00 | 0 | 3,679 | 0 | **3,679** | 3,679 | `Thompson1917_T1917_034.lean`, `thompson1917-t1917-034.yaml` |
| 5 | A | **script** (Script) | cloud-agent (`claude-fable-5.1`) | `completed` | 0.21 | 0.00 | 0 | 27,749 | 0 | **27,749** | 23,073 | `t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py`, `result.json` |
| 6 | A | **output** (Output & compare) | cloud-agent (`claude-fable-5.1`) | `completed` | 78.83 | 88.07 | 150 | 691 | 0 | **841** | 128 | `t1917_034_on_the_shapes_of_eggs_and_of_c_comparison.png`, `t1917_034_on_the_shapes_of_eggs_and_of_c_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | cloud-agent (`claude-fable-5.1`) | `completed` | 0.22 | 0.00 | 0 | 9,697 | 0 | **9,697** | 9,697 | `2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit.yaml`, `2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | cloud-agent (`claude-fable-5.1`) | `completed` | 35.64 | 40.49 | 94 | 902 | 0 | **996** | 5,007 | `2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit.measurements.yaml`, `2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | cloud-agent (`claude-fable-5.1`) | `completed` | 4.25 | 3.18 | 392 | 1,902 | 0 | **2,294** | 7,737 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | cloud-agent (`claude-fable-5.1`) | `completed` | 2.29 | 2.56 | 299 | 2,481 | 0 | **2,780** | 5,789 | `thompson1917-t1917-034-claim.trig`, `thompson1917-t1917-034-claim.trig` |
| 11 | B | **graph** (Public record) | cloud-agent (`claude-fable-5.1`) | `completed` | 0.22 | 0.00 | 0 | 7,737 | 0 | **7,737** | 7,737 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | cloud-agent (`claude-fable-5.1`) | `completed` | 1414.78 | 0.00 | 0 | 9,061 | 0 | **9,061** | 9,061 | `result.json`, `REPORT.md` |
| — | **All** | **TOTAL** | — | `12/12` | **3306.77** | **134.30** | **935** | **365,758** | **0** | **366,693** | **373,767** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Verified both Dryad records for T1917-034 (dryad.q8rp3 Deeming 2018; dryad.8kv2b20 Biggins 2018) via Anubis PoW gate; MATCH. OpenAlex 12 hits; Commons MWNH owl/grebe/plover/guillemot egg photos + CC0 C126 silhouette.
- **Artifact:** `datasets/thompson_1917/t1917-034/scout_manifest.json` (14,569 B, `4,220` tok, SHA-256 `83406c03aa7136ec…`)

### Step 2 — `paper` (Paper)
- **Notes:** Extracted Thompson pp. 652-660 claims C1-C8, Preston (1953) meridian family, Thompson envelope equation, Biggins/Deeming parameter tables.
- **Artifact:** `datasets/thompson_1917/t1917-034/t1917_034_on_the_shapes_of_eggs_and_of_c_reference.json` (16,645 B, `5,303` tok, SHA-256 `fc94bf45d268ef81…`)
- **Artifact:** `datasets/thompson_1917/t1917-034/paper_citations.bib` (4,016 B, `1,284` tok, SHA-256 `45e263a3bef66328…`)
- **Artifact:** `datasets/thompson_1917/t1917-034/README.md` (5,139 B, `1,467` tok, SHA-256 `fb518f6febac4a22…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored CC0 Dryad tables + C126 silhouette and 4 CC BY-SA 3.0 Museum Wiesbaden egg photographs (owl, grebe, lapwing, guillemot); 13 SHA-256 pins written.
- **Artifact:** `datasets/thompson_1917/t1917-034/biggins2018_data_for_comparing_indices.csv` (643,591 B, `281,067` tok, SHA-256 `f0e39a8bfa1f08fe…`)
- **Artifact:** `datasets/thompson_1917/t1917-034/biggins2018_C126_uria_aalge_silhouette.jpg` (221,304 B, `64` tok, SHA-256 `0a5c73b782104608…`)
- **Artifact:** `datasets/thompson_1917/t1917-034/mwnh_0481_uria_aalge_eggs.jpg` (395,245 B, `64` tok, SHA-256 `fcaadba0ae209c4c…`)
- **Artifact:** `datasets/thompson_1917/t1917-034/mwnh_0308_vanellus_vanellus_eggs.jpg` (565,236 B, `64` tok, SHA-256 `426b26bbad6c67d2…`)
- **Artifact:** `datasets/thompson_1917/t1917-034/mwnh_0646_strix_aluco_egg.jpg` (227,126 B, `64` tok, SHA-256 `139f2249abb44e77…`)
- **Artifact:** `datasets/thompson_1917/t1917-034/mwnh_1921_podiceps_cristatus_eggs.jpg` (346,512 B, `64` tok, SHA-256 `866eb96ea90025cc…`)
- **Artifact:** `datasets/thompson_1917/t1917-034/deeming2018_jab_egg_shape_composition.csv` (19,998 B, `8,198` tok, SHA-256 `4d7e86bad43f8d67…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean 4.35 compiles: 5 theorems (pole asymmetry criterion, exact volume + reflection invariance, envelope equation at poles, cone equilibrium, prolate-never-oblate), 0 sorrys, zero imports; alignment + integrity gates pass.
- **Artifact:** `morphoeval/lean/loops/Thompson1917_T1917_034.lean` (10,649 B, `64` tok, SHA-256 `268e2053bb63710f…`)
- **Artifact:** `morphoeval/math_kg_loops/thompson1917-t1917-034.yaml` (11,249 B, `3,615` tok, SHA-256 `95f316c93c17dcf8…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 2; Pass 2 v2 (1033 lines): SHA-256 verification, Thompson closure + pressure path, CV segmentation of 12 eggs in 5 photos, Cartesian-oval/ellipsoid/Baker controls, eta^2/KW/KS/RMSE stats, 17 measurements, result.json, CLI overrides.
- **Artifact:** `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` (62,303 B, `18,391` tok, SHA-256 `884a195e87605ca8…`)
- **Artifact:** `datasets/thompson_1917/t1917-034/result.json` (15,600 B, `4,682` tok, SHA-256 `12f87cbc4cfb2ca6…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 1 render; Pass 2 render
- **Artifact:** `docs/media/t1917_034_on_the_shapes_of_eggs_and_of_c_comparison.png` (974,275 B, `64` tok, SHA-256 `d8c69f3b02b18347…`)
- **Artifact:** `docs/media/t1917_034_on_the_shapes_of_eggs_and_of_c_growth.gif` (784,316 B, `64` tok, SHA-256 `12d3560d4c1cbe3a…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 2; Pass 2 audit 90.1/100 (+34.0); supersedes v1.
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit.yaml` (16,487 B, `4,366` tok, SHA-256 `f4a6751f835f75d2…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit-v2.yaml` (19,821 B, `5,331` tok, SHA-256 `d6e6a2d4f7dd0ec4…`)

### Step 8 — `measure` (Measure)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit.measurements.yaml` (5,963 B, `1,931` tok, SHA-256 `5ab443d32e60a694…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit-v2.measurements.yaml` (10,534 B, `3,076` tok, SHA-256 `a3771dd2e083b9e4…`)

### Step 9 — `build` (Write as claims)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit/REPORT.md` (11,828 B, `3,358` tok, SHA-256 `8cf9035f70d384b6…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit-v2/REPORT.md` (14,557 B, `4,379` tok, SHA-256 `f020764982f673a6…`)

### Step 10 — `publish` (Sign & verify)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit/signed/thompson1917-t1917-034-claim.trig` (9,411 B, `2,836` tok, SHA-256 `a688ea23b8d86269…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit-v2/signed/thompson1917-t1917-034-claim.trig` (10,204 B, `2,953` tok, SHA-256 `4651b6ba3d1beb91…`)

### Step 11 — `graph` (Public record)
- **Notes:** Pass 2; Pass 2 REPORT.md verified: overall 90.1 (faithful), supersession chain v1 -> v2, 9 nanopubs conformant, 9/9 signatures verify.
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit/REPORT.md` (11,828 B, `3,358` tok, SHA-256 `8cf9035f70d384b6…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit-v2/REPORT.md` (14,557 B, `4,379` tok, SHA-256 `f020764982f673a6…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Remediated all 5 Pass 1 items; score 56.1 -> 90.1; verdict supported.
- **Artifact:** `datasets/thompson_1917/t1917-034/result.json` (15,600 B, `4,682` tok, SHA-256 `12f87cbc4cfb2ca6…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit-v2/REPORT.md` (14,557 B, `4,379` tok, SHA-256 `f020764982f673a6…`)
