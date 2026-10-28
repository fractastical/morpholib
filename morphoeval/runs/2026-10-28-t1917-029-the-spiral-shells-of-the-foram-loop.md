# MorphoEval Full-Loop Telemetry — `2026-10-28-t1917-029-the-spiral-shells-of-the-foram-loop`

- **Target simulation:** `thompson1917-t1917-029`
- **Primary paper / dataset:** Thompson, D.W. (1917) On Growth and Form, Ch. XII, pp. 587-611, Figs. 308-317
- **Builder model:** `claude-fable-5-1`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `deb2a5348`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **3092.42 s** (51.54 min) — Tier A: `1913.72 s` · Tier B: `1178.70 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **314,884** (input: `0`, output: `314,884`, subagent: `0`) — Tier A: `255,507` · Tier B: `59,377`
- **Total artifact tokens produced:** `210,414`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | cloud-agent (`claude-fable-5-1`) | `completed` | 891.79 | 0.00 | 0 | 3,240 | 0 | **3,240** | 3,240 | `scout_manifest.json`, `shi2020_dryad_sn02v6x26_Landmarks.dat`, `thompson1917_fig309_nummulina_antiquior_vonmoller_p591.jpg` |
| 2 | A | **paper** (Paper) | cloud-agent (`claude-fable-5-1`) | `completed` | 323.59 | 0.00 | 0 | 5,621 | 0 | **5,621** | 5,621 | `t1917_029_the_spiral_shells_of_the_foram_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | cloud-agent (`claude-fable-5-1`) | `completed` | 0.10 | 0.00 | 0 | 1,413 | 0 | **1,413** | 1,413 | `fusulinid_axial_thin_section_commons_NEON.jpg`, `thompson1917_fig309_nummulina_antiquior_vonmoller_p591.jpg`, `shi2020_dryad_sn02v6x26_Landmarks.dat`, +1 more |
| 4 | A | **rules** (Pin the rules) | cloud-agent (`claude-fable-5-1`) | `completed` | 256.16 | 0.00 | 0 | 2,847 | 0 | **2,847** | 2,847 | `Thompson1917_T1917_029.lean`, `thompson1917-t1917-029.yaml` |
| 5 | A | **script** (Script) | cloud-agent (`claude-fable-5-1`) | `completed` | 433.09 | 0.00 | 0 | 35,846 | 0 | **35,846** | 24,644 | `t1917-029_the-spiral-shells-of-the-foraminifera.py`, `thompson1917-t1917-029.yaml` |
| 6 | A | **output** (Output & compare) | cloud-agent (`claude-fable-5-1`) | `completed` | 8.98 | 0.00 | 0 | 206,540 | 0 | **206,540** | 113,272 | `t1917_029_the_spiral_shells_of_the_foram_comparison.png`, `t1917_029_the_spiral_shells_of_the_foram_growth.gif`, `t1917_029_metrics.json`, +1 more |
| 7 | B | **audit** (Dual-agent audit) | cloud-agent (`claude-fable-5-1`) | `completed` | 0.21 | 0.00 | 0 | 11,824 | 0 | **11,824** | 11,824 | `2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit.yaml`, `2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | cloud-agent (`claude-fable-5-1`) | `completed` | 17.96 | 0.00 | 0 | 5,362 | 0 | **5,362** | 5,362 | `2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit.measurements.yaml`, `2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | cloud-agent (`claude-fable-5-1`) | `completed` | 35.02 | 0.00 | 0 | 15,781 | 0 | **15,781** | 15,781 | `REPORT.md`, `thompson1917-t1917-029-overall.trig`, `REPORT.md`, +1 more |
| 10 | B | **publish** (Sign & verify) | cloud-agent (`claude-fable-5-1`) | `completed` | 41.37 | 0.00 | 0 | 5,261 | 0 | **5,261** | 5,261 | `published.json`, `thompson1917-t1917-029-overall.trig`, `published.json` |
| 11 | B | **graph** (Public record) | cloud-agent (`claude-fable-5-1`) | `completed` | 0.21 | 0.00 | 0 | 8,505 | 0 | **8,505** | 8,505 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | cloud-agent (`claude-fable-5-1`) | `completed` | 1083.92 | 0.00 | 0 | 12,644 | 0 | **12,644** | 12,644 | `2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit-v2.yaml`, `REPORT.md`, `result.json` |
| — | **All** | **TOTAL** | — | `12/12` | **3092.42** | **0.00** | **0** | **314,884** | **0** | **314,884** | **210,414** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Dryad doi:10.5061/dryad.sn02v6x26 verified (SHA-256 match to Dryad digests); taxon+variable MATCH to Thompson's von Moller Fusulina exemplar; Commons API 503 during run, PD Fig. 309 from IA scan vendored
- **Artifact:** `datasets/thompson_1917/t1917-029/scout_manifest.json` (11,608 B, `3,112` tok, SHA-256 `b6ac77cd74801b48…`)
- **Artifact:** `datasets/thompson_1917/t1917-029/shi2020_dryad_sn02v6x26_Landmarks.dat` (1,989,725 B, `64` tok, SHA-256 `f7bb3fe73332d58c…`)
- **Artifact:** `datasets/thompson_1917/t1917-029/thompson1917_fig309_nummulina_antiquior_vonmoller_p591.jpg` (219,228 B, `64` tok, SHA-256 `e7e37b2c3ea0a6f3…`)

### Step 2 — `paper` (Paper)
- **Notes:** Equations E1-E6, von Moller/van Iterson/Rhumbler tables transcribed from IA scan p.593/595; Crossref-verified Shi 2021 Paleobiology 47(1)
- **Artifact:** `datasets/thompson_1917/t1917-029/t1917_029_the_spiral_shells_of_the_foram_reference.json` (11,184 B, `3,353` tok, SHA-256 `cdd3787ee526a2a9…`)
- **Artifact:** `datasets/thompson_1917/t1917-029/paper_citations.bib` (4,019 B, `1,225` tok, SHA-256 `9773df6d18414051…`)
- **Artifact:** `datasets/thompson_1917/t1917-029/README.md` (3,443 B, `1,043` tok, SHA-256 `8e42de28c53fac98…`)

### Step 3 — `source` (Formal source)
- **Notes:** Commons File:Fusulinid.jpg (CC BY-SA 2.5, SHA-1 verified vs Commons) + PD Thompson Fig. 309; Dryad files SHA-256 verified vs Dryad digests; all pinned in reference JSON
- **Artifact:** `datasets/thompson_1917/t1917-029/fusulinid_axial_thin_section_commons_NEON.jpg` (206,203 B, `64` tok, SHA-256 `39f54718bbc8afa2…`)
- **Artifact:** `datasets/thompson_1917/t1917-029/thompson1917_fig309_nummulina_antiquior_vonmoller_p591.jpg` (219,228 B, `64` tok, SHA-256 `e7e37b2c3ea0a6f3…`)
- **Artifact:** `datasets/thompson_1917/t1917-029/shi2020_dryad_sn02v6x26_Landmarks.dat` (1,989,725 B, `64` tok, SHA-256 `f7bb3fe73332d58c…`)
- **Artifact:** `datasets/thompson_1917/t1917-029/shi2020_dryad_sn02v6x26_Table_1.csv` (3,658 B, `1,221` tok, SHA-256 `b5ddc6561b74b26d…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** 5 theorems (gnomonic succession, mass-proportional growth, log-difference constancy, equal-angle periodicity, continuous series limits) compile with lean 4.35.0, 0 sorry
- **Artifact:** `morphoeval/lean/loops/Thompson1917_T1917_029.lean` (8,804 B, `64` tok, SHA-256 `eaaa1ef66a63b06e…`)
- **Artifact:** `morphoeval/math_kg_loops/thompson1917-t1917-029.yaml` (9,693 B, `2,783` tok, SHA-256 `bcaa8ef8c69daa0b…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 2 v2; Pass 2 v2: R1 CV walls (thin section + Fig. 309), R2 (W,d,T) sweep, R3 controls, R4 drift test, R5 MC gap null, R6 KS/bootstrap/RMSE, R7 data-fitted overlay, R8 env stamp + regression asserts, R9 bib URLs
- **Artifact:** `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` (75,997 B, `21,631` tok, SHA-256 `fa04f97826657430…`)
- **Artifact:** `morphoeval/math_kg_loops/thompson1917-t1917-029.yaml` (10,438 B, `3,013` tok, SHA-256 `0f5efbf4f38878c1…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 2; Pass 2 verdict MIXED 3/4: T2 fails (pseudoschwagerine 15% ontogenetic drift), T1/T3/T4 pass; CV thin section W=1.49
- **Artifact:** `docs/media/t1917_029_the_spiral_shells_of_the_foram_comparison.png` (883,361 B, `64` tok, SHA-256 `c266574c8781e151…`)
- **Artifact:** `docs/media/t1917_029_the_spiral_shells_of_the_foram_growth.gif` (872,268 B, `64` tok, SHA-256 `6df0512f746268e7…`)
- **Artifact:** `datasets/thompson_1917/t1917-029/t1917_029_metrics.json` (316,060 B, `110,990` tok, SHA-256 `34d702e99af764e9…`)
- **Artifact:** `datasets/thompson_1917/t1917-029/result.json` (6,977 B, `2,154` tok, SHA-256 `ce66eac681e9affd…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 2; Pass 2 self-audit overall 77.4 (+14.95), supersedes Pass 1; R10-R14 open
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit.yaml` (22,272 B, `5,698` tok, SHA-256 `080a6b9d5351a626…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit-v2.yaml` (23,569 B, `6,126` tok, SHA-256 `cc52eae2f342ecdb…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 2; Pass 2: 3 outputs fingerprinted, 17 measurements, quick_gate 100/100
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit.measurements.yaml` (7,710 B, `2,202` tok, SHA-256 `c72afa776026d964…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit-v2.measurements.yaml` (10,856 B, `3,160` tok, SHA-256 `96187884c6901559…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Pass 2; Pass 2: 9 nanopubs, 718 triples, npx:supersedes Pass 1 URIs, Delphi conformance OK
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit/REPORT.md` (15,142 B, `4,141` tok, SHA-256 `5cd631735c490ed5…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit/thompson1917-t1917-029-overall.trig` (13,125 B, `3,676` tok, SHA-256 `bddc697b825acc34…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit-v2/REPORT.md` (15,558 B, `4,364` tok, SHA-256 `bd848b44738a012d…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit-v2/thompson1917-t1917-029-overall.trig` (12,626 B, `3,600` tok, SHA-256 `1a3d52aef4b4cc3a…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Pass 2 sign-only; Pass 2: 9 nanopubs signed (ephemeral key, sign-only); @nanopub/sign verify 9 ok
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit/published.json` (1,694 B, `620` tok, SHA-256 `35a3cc955533f240…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit/signed/thompson1917-t1917-029-overall.trig` (14,288 B, `4,024` tok, SHA-256 `b1299e40889d75a1…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit-v2/published.json` (1,694 B, `617` tok, SHA-256 `ecd4a3130d3f2217…`)

### Step 11 — `graph` (Public record)
- **Notes:** Pass 2; Pass 2 REPORT.md verified: overall 77.4 mostly faithful, chain v1 -> v2
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit/REPORT.md` (15,142 B, `4,141` tok, SHA-256 `5cd631735c490ed5…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit-v2/REPORT.md` (15,558 B, `4,364` tok, SHA-256 `bd848b44738a012d…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 remediation of R1-R9; Pass 2 closed: R1-R9 remediated, audit v2 77.4 supersedes v1 62.45, verdict MIXED 3/4
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit-v2.yaml` (23,569 B, `6,126` tok, SHA-256 `cc52eae2f342ecdb…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit-v2/REPORT.md` (15,558 B, `4,364` tok, SHA-256 `bd848b44738a012d…`)
- **Artifact:** `datasets/thompson_1917/t1917-029/result.json` (6,977 B, `2,154` tok, SHA-256 `ce66eac681e9affd…`)
