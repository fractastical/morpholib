# MorphoEval Full-Loop Telemetry — `2026-10-28-t1917-024-on-concretions-spicules-and-sp-loop`

- **Target simulation:** `thompson1917-t1917-024`
- **Primary paper / dataset:** Thompson, D.W. (1917) On Growth and Form, Ch. IX, pp. 411-487, Figs. 194-235
- **Builder model:** `claude-fable-5-1`
- **Assessor models:** `claude-fable-5-1` + `claude-sonnet-5-5-high`
- **Start commit:** `fc7454cd8`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1949.39 s** (32.49 min) — Tier A: `1567.20 s` · Tier B: `382.19 s`
- **Total CPU time:** `18.41 s` user + `1.53 s` sys
- **Total LLM tokens:** **451,889** (input: `364,646`, output: `87,243`, subagent: `0`) — Tier A: `312,688` · Tier B: `139,201`
- **Total artifact tokens produced:** `80,701`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 542.70 | 0.00 | 45,000 | 6,000 | 0 | **51,000** | 5,035 | `scout_manifest.json`, `dryad_8mv00jv_dataset_metadata.json` |
| 2 | A | **paper** (Paper) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 202.72 | 0.00 | 30,000 | 7,000 | 0 | **37,000** | 7,064 | `t1917_024_on_concretions_spicules_and_sp_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 0.09 | 0.00 | 8,000 | 1,500 | 0 | **9,500** | 3,591 | `grantia_indica_zootaxa4426_fig53_sem.jpg`, `grantia_spicules_readey_100x.jpg`, `thompson1917_fig214_ia_leaf465_p445.jpg`, +1 more |
| 4 | A | **rules** (Pin the rules) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 336.68 | 0.00 | 40,000 | 9,000 | 0 | **49,000** | 2,847 | `Thompson1917_T1917_024.lean`, `thompson1917-t1917-024.yaml` |
| 5 | A | **script** (Script) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 476.31 | 0.00 | 130,000 | 36,000 | 0 | **166,000** | 17,665 | `t1917-024_on-concretions-spicules-and-spicular-skeletons.py` |
| 6 | A | **output** (Output & compare) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 8.70 | 11.37 | 126 | 62 | 0 | **188** | 9,414 | `t1917_024_on_concretions_spicules_and_sp_comparison.png`, `t1917_024_on_concretions_spicules_and_sp_growth.gif`, `sem_ray_angles.json`, +1 more |
| 7 | B | **audit** (Dual-agent audit) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 137.00 | 0.00 | 65,000 | 14,000 | 0 | **79,000** | 10,039 | `2026-10-28-t1917-024-on-concretions-spicules-and-sp-audit.yaml`, `2026-10-28-t1917-024-on-concretions-spicules-and-sp-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 3.20 | 4.50 | 81 | 426 | 0 | **507** | 2,521 | `2026-10-28-t1917-024-on-concretions-spicules-and-sp-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 2.56 | 2.45 | 264 | 1,222 | 0 | **1,486** | 7,232 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 1.48 | 1.62 | 175 | 1,433 | 0 | **1,608** | 1,236 | `published.json`, `published.json` |
| 11 | B | **graph** (Public record) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 0.22 | 0.00 | 6,000 | 600 | 0 | **6,600** | 7,232 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 237.73 | 0.00 | 40,000 | 10,000 | 0 | **50,000** | 6,825 | `2026-10-28-t1917-024-on-concretions-spicules-and-sp-audit-v2.yaml`, `result.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1949.39** | **19.94** | **364,646** | **87,243** | **0** | **451,889** | **80,701** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Dryad 8mv00jv = Silurian hexactine fossil tomograms (5.8GB): metadata pinned, DATASET_MISMATCH for primary test; sanctioned alternative = Thompson Fig.214 IA leaf465 + CC BY Grantia SEM plate
- **Artifact:** `datasets/thompson_1917/t1917-024/scout_manifest.json` (11,819 B, `3,232` tok, SHA-256 `9651d12265b171b3…`)
- **Artifact:** `datasets/thompson_1917/t1917-024/dryad_8mv00jv_dataset_metadata.json` (6,523 B, `1,803` tok, SHA-256 `8b1273b453ccdb77…`)

### Step 2 — `paper` (Paper)
- **Notes:** Neumann-triangle E1-E10, P1-P6, parameter regime table; 7 DOIs Crossref-verified
- **Artifact:** `datasets/thompson_1917/t1917-024/t1917_024_on_concretions_spicules_and_sp_reference.json` (14,425 B, `3,920` tok, SHA-256 `5c04b8ece13d0dca…`)
- **Artifact:** `datasets/thompson_1917/t1917-024/paper_citations.bib` (6,313 B, `1,740` tok, SHA-256 `e42cdf8f80c265cb…`)
- **Artifact:** `datasets/thompson_1917/t1917-024/README.md` (5,118 B, `1,404` tok, SHA-256 `12cc47ce9cac8416…`)

### Step 3 — `source` (Formal source)
- **Notes:** CC BY 3.0 Grantia SEM plate (primary), CC BY-SA 4.0 Grantia micrograph, PD Thompson p.445 scan; SHA-256 pinned in reference JSON
- **Artifact:** `datasets/thompson_1917/t1917-024/specimens/grantia_indica_zootaxa4426_fig53_sem.jpg` (782,038 B, `64` tok, SHA-256 `70194abe78894820…`)
- **Artifact:** `datasets/thompson_1917/t1917-024/specimens/grantia_spicules_readey_100x.jpg` (394,865 B, `64` tok, SHA-256 `08641621770534e1…`)
- **Artifact:** `datasets/thompson_1917/t1917-024/specimens/thompson1917_fig214_ia_leaf465_p445.jpg` (498,472 B, `64` tok, SHA-256 `019168c1fa4adddc…`)
- **Artifact:** `datasets/thompson_1917/t1917-024/thompson1917_ch9_pp443-449_ocr_excerpt.txt` (14,599 B, `3,399` tok, SHA-256 `52af6bf9d7e46b7d…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean 4.35.0 exit 0, 6 theorems 0 sorry (Plateau 120, sagittal paired equality, equiangular iff c=m factorisation, ordering/T-limit, tetrahedral+orthogonal decide); KG yaml 5 laws
- **Artifact:** `morphoeval/lean/loops/Thompson1917_T1917_024.lean` (11,308 B, `64` tok, SHA-256 `c76af5bd05ebc356…`)
- **Artifact:** `morphoeval/math_kg_loops/thompson1917-t1917-024.yaml` (9,182 B, `2,783` tok, SHA-256 `e7a80ec0307f590d…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 1 v1: Neumann model E1-E9, radial-arc SEM ray measurement, comparison PNG, growth GIF, measure_morphoeval; Pass 2 v2: PCA ray axes (sweep closure r=0.9999), Dirichlet null, rho sweep, crystal/lattice ablations, population micrograph, Fig.214 digitisation, GoF stats, CLI, result.json
- **Artifact:** `thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py` (60,418 B, `17,665` tok, SHA-256 `b126b4637fd9c998…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 1 outputs: self-test OK; 8 three-ray spicules, 3 monaxons; Pass 2 outputs: self-test OK; verdict mixed (P1,P2 supported; P3/alcyonarians untestable; mechanism contested)
- **Artifact:** `docs/media/t1917_024_on_concretions_spicules_and_sp_comparison.png` (587,937 B, `64` tok, SHA-256 `e4526bfdf295d887…`)
- **Artifact:** `docs/media/t1917_024_on_concretions_spicules_and_sp_growth.gif` (198,042 B, `64` tok, SHA-256 `b2b0014db3e72232…`)
- **Artifact:** `datasets/thompson_1917/t1917-024/measurements/sem_ray_angles.json` (25,547 B, `7,980` tok, SHA-256 `87970d1b6e1d337a…`)
- **Artifact:** `datasets/thompson_1917/t1917-024/result.json` (4,186 B, `1,306` tok, SHA-256 `212dad917f94af37…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 audit: 7 dims, overall ~54; R1-R8 remediation; Pass 2 audit v2 supersedes v1: overall 75.9; R1, R9-R13 open
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-024-on-concretions-spicules-and-sp-audit.yaml` (17,833 B, `4,561` tok, SHA-256 `0d4a735b6603db85…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-024-on-concretions-spicules-and-sp-audit-v2.yaml` (20,804 B, `5,478` tok, SHA-256 `d18357fd2c849be1…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 2 measurements
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-024-on-concretions-spicules-and-sp-audit-v2.measurements.yaml` (8,509 B, `2,521` tok, SHA-256 `06a07b1fab9d928f…`)

### Step 9 — `build` (Write as claims)
- **Notes:** rubric nanopub reused from bos-taurus v7 dir; Pass 2 build supersedes v1 kg
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-024-on-concretions-spicules-and-sp-audit/REPORT.md` (12,254 B, `3,253` tok, SHA-256 `6687874262dd743e…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-024-on-concretions-spicules-and-sp-audit-v2/REPORT.md` (14,341 B, `3,979` tok, SHA-256 `2623f6ed9c6d4b9c…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** signed locally with loop-ephemeral key (not the project key), unpublished; Pass 2 signed (ephemeral loop key), verified
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-024-on-concretions-spicules-and-sp-audit/published.json` (1,694 B, `625` tok, SHA-256 `7d2bc5fa9cc5e629…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-024-on-concretions-spicules-and-sp-audit-v2/published.json` (1,694 B, `611` tok, SHA-256 `538532857f03df12…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md verified: overall 54.0; 9 nanopubs signed+verified (ephemeral key); Pass 2 REPORT.md verified: overall 75.9, supersedes v1 nanopubs
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-024-on-concretions-spicules-and-sp-audit/REPORT.md` (12,254 B, `3,253` tok, SHA-256 `6687874262dd743e…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-024-on-concretions-spicules-and-sp-audit-v2/REPORT.md` (14,341 B, `3,979` tok, SHA-256 `2623f6ed9c6d4b9c…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 complete: v2 audit 75.9 supersedes v1 62.3; verdict mixed
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-024-on-concretions-spicules-and-sp-audit-v2.yaml` (20,893 B, `5,519` tok, SHA-256 `10e1335d2fb52247…`)
- **Artifact:** `datasets/thompson_1917/t1917-024/result.json` (4,186 B, `1,306` tok, SHA-256 `212dad917f94af37…`)
