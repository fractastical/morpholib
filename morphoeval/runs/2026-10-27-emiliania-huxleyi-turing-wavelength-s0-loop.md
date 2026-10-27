# MorphoEval Full-Loop Telemetry — `2026-10-27-emiliania-huxleyi-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-emiliania-huxleyi-turing-wavelength-s0`
- **Primary paper / dataset:** Emiliania huxleyi Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **537.55 s** (8.96 min) — Tier A: `537.01 s` · Tier B: `0.54 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **30,003** (input: `0`, output: `30,003`, subagent: `0`) — Tier A: `15,269` · Tier B: `14,734`
- **Total artifact tokens produced:** `30,003`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 536.57 | 0.00 | 0 | 1,362 | 0 | **1,362** | 1,362 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 3,688 | 0 | **3,688** | 3,688 | `emiliania_huxleyi_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 64 | 0 | **64** | 64 | `emiliania_huxleyi_coccosphere_coccolith.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 919 | 0 | **919** | 919 | `Loop_3217_emiliania_huxleyi_turing_wavelength_s0.lean`, `delphi10000-emiliania-huxleyi-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 9,108 | 0 | **9,108** | 9,108 | `2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `emiliania_huxleyi_turing_wavelength_s0_comparison.png`, `emiliania_huxleyi_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 4,778 | 0 | **4,778** | 4,778 | `2026-10-27-emiliania-huxleyi-turing-wavelength-s0-audit.yaml`, `2026-10-27-emiliania-huxleyi-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,722 | 0 | **1,722** | 1,722 | `2026-10-27-emiliania-huxleyi-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,501 | 0 | **2,501** | 2,501 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 602 | 0 | **602** | 602 | `2026-10-27-emiliania-huxleyi-turing-wavelength-s0-loop-image-fitness.json` |
| 11 | B | **graph** (Public record) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,501 | 0 | **2,501** | 2,501 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,630 | 0 | **2,630** | 2,630 | `2026-10-27-emiliania-huxleyi-turing-wavelength-s0-audit-v2.yaml` |
| — | **All** | **TOTAL** | — | `12/12` | **537.55** | **0.00** | **0** | **30,003** | **0** | **30,003** | **30,003** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Slot 3217 is outside campaign_100_slots.yaml; OpenAlex/Crossref/Wikimedia queried directly.; Campaign slot missing; OpenAlex/Crossref/Wikimedia queried directly. No collision.
- **Artifact:** `datasets/emiliania_huxleyi_turing_wavelength_s0/scout_manifest.json` (4,719 B, `1,362` tok, SHA-256 `0244c20ac7f18492…`)

### Step 2 — `paper` (Paper)
- **Notes:** slot 3217
- **Artifact:** `datasets/emiliania_huxleyi_turing_wavelength_s0/emiliania_huxleyi_turing_wavelength_s0_reference.json` (6,039 B, `1,715` tok, SHA-256 `7eca0ef524dbb598…`)
- **Artifact:** `datasets/emiliania_huxleyi_turing_wavelength_s0/paper_citations.bib` (4,343 B, `1,344` tok, SHA-256 `4e3f4a300348bb17…`)
- **Artifact:** `datasets/emiliania_huxleyi_turing_wavelength_s0/README.md` (2,267 B, `629` tok, SHA-256 `ae4028cfa7b9be23…`)

### Step 3 — `source` (Formal source)
- **Notes:** slot 3217
- **Artifact:** `datasets/emiliania_huxleyi_turing_wavelength_s0/emiliania_huxleyi_coccosphere_coccolith.jpg` (164,686 B, `64` tok, SHA-256 `94d4b431c4912727…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** slot 3217
- **Artifact:** `morphoeval/lean/loops/Loop_3217_emiliania_huxleyi_turing_wavelength_s0.lean` (2,188 B, `64` tok, SHA-256 `ce059296ce7b9a0e…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-emiliania-huxleyi-turing-wavelength-s0.yaml` (3,131 B, `855` tok, SHA-256 `5893b3a35f989d4b…`)

### Step 5 — `script` (Script)
- **Notes:** slot 3217
- **Artifact:** `models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py` (29,895 B, `9,108` tok, SHA-256 `3cada4118d1fc2fb…`)

### Step 6 — `output` (Output & compare)
- **Notes:** slot 3217
- **Artifact:** `docs/media/emiliania_huxleyi_turing_wavelength_s0_comparison.png` (659,755 B, `64` tok, SHA-256 `daeb9a91e4300345…`)
- **Artifact:** `docs/media/emiliania_huxleyi_turing_wavelength_s0_growth.gif` (3,694,038 B, `64` tok, SHA-256 `c90f6524d06e8f07…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** slot 3217
- **Artifact:** `morphoeval/audits/2026-10-27-emiliania-huxleyi-turing-wavelength-s0-audit.yaml` (8,375 B, `2,148` tok, SHA-256 `19eb9db830aa9834…`)
- **Artifact:** `morphoeval/audits/2026-10-27-emiliania-huxleyi-turing-wavelength-s0-audit-v2.yaml` (9,873 B, `2,630` tok, SHA-256 `f56254aaefb36a10…`)

### Step 8 — `measure` (Measure)
- **Notes:** slot 3217
- **Artifact:** `morphoeval/audits/2026-10-27-emiliania-huxleyi-turing-wavelength-s0-audit-v2.measurements.yaml` (5,999 B, `1,722` tok, SHA-256 `a6561085d659c08c…`)

### Step 9 — `build` (Write as claims)
- **Notes:** slot 3217
- **Artifact:** `morphoeval/kg/2026-10-27-emiliania-huxleyi-turing-wavelength-s0-audit-v2/REPORT.md` (9,064 B, `2,501` tok, SHA-256 `38c7b08630e5cc9b…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** slot 3217
- **Artifact:** `morphoeval/runs/2026-10-27-emiliania-huxleyi-turing-wavelength-s0-loop-image-fitness.json` (2,121 B, `602` tok, SHA-256 `1511cc53a1a324e5…`)

### Step 11 — `graph` (Public record)
- **Notes:** slot 3217
- **Artifact:** `morphoeval/kg/2026-10-27-emiliania-huxleyi-turing-wavelength-s0-audit-v2/REPORT.md` (9,064 B, `2,501` tok, SHA-256 `38c7b08630e5cc9b…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** slot 3217
- **Artifact:** `morphoeval/audits/2026-10-27-emiliania-huxleyi-turing-wavelength-s0-audit-v2.yaml` (9,873 B, `2,630` tok, SHA-256 `f56254aaefb36a10…`)

## Result

Holocene *Emiliania huxleyi* (0 Ma) at 5.808 nm. Schnakenberg kinetics with the Turing peak fixed on an eight-wavelength, 46.464 nm patch. The reconstructed predecessor is six wavelengths on that same patch (7.744 nm). There is no successor.

Pass 1 audit overall **82.7**. The intensity annulus was a weak spoke estimator (score 0.22) and the damped wing was not reported. Pass 2 replaced the annulus with a tangential-Sobel measurement (harmonic 30, spoke score 5.20) and recorded wing RMSE 0.063 at dt 0.006. Pass 2 overall **89.1**. No pass 3.

Gates after pass 2:

| Gate | Result |
|---|---|
| image_fitness ≥ 70 | 90.8 |
| cv_sim_similarity ≥ 58 | 92.6 |
| GIF frames ≥ 6 | 12 |
| resolution ≥ 340 px | 720×540 |
| non-circular specimen metric | foreground 0.675 vs 0.435, circularity 0.043 vs 0.329, coherency 0.587 vs 0.743 |
| Pearson r > 0.98 | 0.9944 (reference 1.0) |
| audit overall ≥ 85 | 89.1 |

Signing was not done. `~/.nanopub/profile.yml` is not on this machine, so `sign_morphoeval.py` was not run and no trusty URI was invented. The knowledge graphs were still built and passed `build_morphoeval_kg.py --conform`. The publisher ORCID on those unsigned nanopubs is the repository publisher already used in this corpus (`https://orcid.org/0009-0007-4948-9192`), not a signature. Lean 4 is not installed here; the six theorems have no `sorry` and were not machine-checked.
