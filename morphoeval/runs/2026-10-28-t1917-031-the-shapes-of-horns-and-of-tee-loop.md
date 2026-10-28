# MorphoEval Full-Loop Telemetry — `2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-loop`

- **Target simulation:** `thompson1917-t1917-031`
- **Primary paper / dataset:** Thompson, D.W. (1917) On Growth and Form, Ch. XIII, pp. 612-634, Figs. 318-324
- **Builder model:** `claude-fable-5-1`
- **Assessor models:** `claude-fable-5-1` + `claude-sonnet-5-5-high`
- **Start commit:** `fc7454cd8`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **2151.64 s** (35.86 min) — Tier A: `1954.52 s` · Tier B: `197.12 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **128,948** (input: `0`, output: `128,948`, subagent: `0`) — Tier A: `90,085` · Tier B: `38,863`
- **Total artifact tokens produced:** `120,840`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | cloud-agent (`claude-fable-5-1`) | `completed` | 835.46 | 0.00 | 0 | 3,593 | 0 | **3,593** | 3,593 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | cloud-agent (`claude-fable-5-1`) | `completed` | 308.04 | 0.00 | 0 | 5,061 | 0 | **5,061** | 5,061 | `t1917_031_the_shapes_of_horns_and_of_tee_reference.json`, `paper_citations.bib` |
| 3 | A | **source** (Formal source) | cloud-agent (`claude-fable-5-1`) | `completed` | 0.09 | 0.00 | 0 | 51,454 | 0 | **51,454** | 51,454 | `ovis_ammon_argali_skull_horns_specimen.jpg`, `thompson_1917_fig320_ovis_ammon_head.jpg`, `thompson_1917_fig319_capra_sinaitica_head.jpg`, +2 more |
| 4 | A | **rules** (Pin the rules) | cloud-agent (`claude-fable-5-1`) | `completed` | 405.32 | 0.00 | 0 | 2,392 | 0 | **2,392** | 2,392 | `Thompson1917_T1917_031.lean`, `thompson1917-t1917-031.yaml` |
| 5 | A | **script** (Script) | cloud-agent (`claude-fable-5-1`) | `completed` | 405.43 | 0.00 | 0 | 22,785 | 0 | **22,785** | 15,404 | `t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` |
| 6 | A | **output** (Output & compare) | cloud-agent (`claude-fable-5-1`) | `completed` | 0.18 | 0.00 | 0 | 4,800 | 0 | **4,800** | 4,073 | `t1917_031_the_shapes_of_horns_and_of_tee_comparison.png`, `t1917_031_the_shapes_of_horns_and_of_tee_growth.gif`, `t1917_031_metrics.json`, +1 more |
| 7 | B | **audit** (Dual-agent audit) | cloud-agent | `completed` | 0.19 | 0.00 | 0 | 8,007 | 0 | **8,007** | 8,007 | `2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit.yaml`, `2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | cloud-agent | `completed` | 8.26 | 0.00 | 0 | 5,326 | 0 | **5,326** | 5,326 | `2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit.measurements.yaml`, `2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | cloud-agent | `completed` | 86.23 | 0.00 | 0 | 7,890 | 0 | **7,890** | 7,890 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | cloud-agent | `completed` | 22.83 | 0.00 | 0 | 1,240 | 0 | **1,240** | 1,240 | `published.json`, `published.json` |
| 11 | B | **graph** (Public record) | cloud-agent | `completed` | 9.24 | 0.00 | 0 | 7,890 | 0 | **7,890** | 7,890 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | cloud-agent | `completed` | 70.36 | 0.00 | 0 | 8,510 | 0 | **8,510** | 8,510 | `2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit-v2.yaml`, `REPORT.md` |
| — | **All** | **TOTAL** | — | `12/12` | **2151.64** | **0.00** | **0** | **128,948** | **0** | **128,948** | **120,840** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Dryad qz612jmb4 verified: species match, shape-variable mismatch (length-mass allometry). Primary shape data = Thompson 1917 Figs 319/320 (IA leaves 636/641) + CC BY-SA 4.0 Argali skull photo.
- **Artifact:** `datasets/thompson_1917/t1917-031/scout_manifest.json` (12,723 B, `3,593` tok, SHA-256 `c6e9d7fbfcf1ea70…`)

### Step 2 — `paper` (Paper)
- **Notes:** E1-E6 extracted from Ch. XIII OCR (lines 29543-30530) + Ch. XI p.558 helico-spiral formula; parameter regime table p.617
- **Artifact:** `datasets/thompson_1917/t1917-031/t1917_031_the_shapes_of_horns_and_of_tee_reference.json` (12,893 B, `3,632` tok, SHA-256 `64826ef38cca003b…`)
- **Artifact:** `datasets/thompson_1917/t1917-031/paper_citations.bib` (4,685 B, `1,429` tok, SHA-256 `eb8c1ca49ca4877d…`)

### Step 3 — `source` (Formal source)
- **Notes:** SHA-256 pinned in reference.json input_dataset_sha256; Dryad CC0 table, IA public-domain figure crops, Commons CC BY-SA 4.0 Argali skull
- **Artifact:** `datasets/thompson_1917/t1917-031/ovis_ammon_argali_skull_horns_specimen.jpg` (500,088 B, `64` tok, SHA-256 `f8c17c09c7ebe13e…`)
- **Artifact:** `datasets/thompson_1917/t1917-031/thompson_1917_fig320_ovis_ammon_head.jpg` (151,998 B, `64` tok, SHA-256 `4d6a6093ce49a70a…`)
- **Artifact:** `datasets/thompson_1917/t1917-031/thompson_1917_fig319_capra_sinaitica_head.jpg` (294,421 B, `64` tok, SHA-256 `55131cbad1f81417…`)
- **Artifact:** `datasets/thompson_1917/t1917-031/thompson_1917_fig318_ram_horns_diagram.jpg` (190,511 B, `64` tok, SHA-256 `18f820edfb2de26c…`)
- **Artifact:** `datasets/thompson_1917/t1917-031/dryad_qz612jmb4_dataset_Tidiereetal2020.txt` (130,031 B, `51,198` tok, SHA-256 `8e13941af7964f2d…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** lean 4.35.0 exit 0, 0 sorry, 16 theorems in MathOfBiology.Loop_225 (E1-E6 + projection non-identifiability)
- **Artifact:** `morphoeval/lean/loops/Thompson1917_T1917_031.lean` (7,546 B, `64` tok, SHA-256 `33c91997720897a6…`)
- **Artifact:** `morphoeval/math_kg_loops/thompson1917-t1917-031.yaml` (7,997 B, `2,328` tok, SHA-256 `9c695b9a4114bace…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 2; Pass 2 v2.0.0-pass2: HSV segmentation of real Argali horn, Capra Fig.319, two-pass trimming, gnomonic ray test, bootstrap/KS, controls, ablation, sensitivity sectors, verdict logic a-d, Dryad analogue
- **Artifact:** `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` (54,387 B, `15,404` tok, SHA-256 `6ae58c8e31a28248…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 2; Pass 2: self-test OK; Argali alpha_proj 74.7 [73.6,75.6] vs 75; Fig.320 84.2 (beta_implied 22); gnomonic fails Fig.320; verdict mixed
- **Artifact:** `docs/media/t1917_031_the_shapes_of_horns_and_of_tee_comparison.png` (971,711 B, `64` tok, SHA-256 `832658753c97c6a0…`)
- **Artifact:** `docs/media/t1917_031_the_shapes_of_horns_and_of_tee_growth.gif` (214,879 B, `64` tok, SHA-256 `6a22c063cbd8c00c…`)
- **Artifact:** `datasets/thompson_1917/t1917-031/t1917_031_metrics.json` (5,953 B, `1,979` tok, SHA-256 `faf646bab18c562a…`)
- **Artifact:** `datasets/thompson_1917/t1917-031/result.json` (6,709 B, `1,966` tok, SHA-256 `b497cdf5262491b9…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 2; Pass 2 superseding audit v2: overall 67.6 (from 53.5); R1 open; new R8-R11
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit.yaml` (14,938 B, `3,907` tok, SHA-256 `6d190b72acfe8289…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit-v2.yaml` (15,350 B, `4,100` tok, SHA-256 `c46ce9988d781197…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 2; 23 measurements, 5 outputs fingerprinted, quick_gate 100/100
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit.measurements.yaml` (5,975 B, `2,039` tok, SHA-256 `4e0f8834a4c9734e…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit-v2.measurements.yaml` (10,497 B, `3,287` tok, SHA-256 `3e320a9f0ad7ded2…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Pass 2; 9 nanopubs, 803 triples, Delphi conformance OK; --supersede-from Pass 1 (sign-only URIs => no npx:supersedes emitted by design), chain via supersedes_audit_id
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit/REPORT.md` (12,108 B, `3,480` tok, SHA-256 `dc886f10d6ee4dc2…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit-v2/REPORT.md` (15,045 B, `4,410` tok, SHA-256 `67f8b7d40cb8c802…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Pass 2; 9 signed (ephemeral loop key, sign-only), @nanopub/sign verify 9 ok 0 failed
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit/published.json` (1,694 B, `626` tok, SHA-256 `e3294330a6f1de79…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit-v2/published.json` (1,694 B, `614` tok, SHA-256 `6159a18ac0e02a20…`)

### Step 11 — `graph` (Public record)
- **Notes:** Pass 2; REPORT.md verified: overall 67.6 (mostly faithful), verdict mixed
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit/REPORT.md` (12,108 B, `3,480` tok, SHA-256 `dc886f10d6ee4dc2…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit-v2/REPORT.md` (15,045 B, `4,410` tok, SHA-256 `67f8b7d40cb8c802…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Closed R2-R7 from Pass 1; verdict mixed; R1, R8-R11 remain
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit-v2.yaml` (15,350 B, `4,100` tok, SHA-256 `c46ce9988d781197…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit-v2/REPORT.md` (15,045 B, `4,410` tok, SHA-256 `67f8b7d40cb8c802…`)
