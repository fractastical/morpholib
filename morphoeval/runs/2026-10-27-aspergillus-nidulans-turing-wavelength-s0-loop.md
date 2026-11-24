# MorphoEval Full-Loop Telemetry — `2026-10-27-aspergillus-nidulans-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-aspergillus-nidulans-turing-wavelength-s0`
- **Primary paper / dataset:** Aspergillus nidulans Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1331.05 s** (22.18 min) — Tier A: `1326.08 s` · Tier B: `4.97 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **29,667** (input: `0`, output: `29,667`, subagent: `0`) — Tier A: `15,014` · Tier B: `14,653`
- **Total artifact tokens produced:** `29,667`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 1322.06 | 0.00 | 0 | 2,226 | 0 | **2,226** | 2,226 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.80 | 0.00 | 0 | 2,367 | 0 | **2,367** | 2,367 | `aspergillus_nidulans_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.80 | 0.00 | 0 | 64 | 0 | **64** | 64 | `aspergillus_nidulans_turing_wavelength_s0_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.80 | 0.00 | 0 | 902 | 0 | **902** | 902 | `Loop_3122_aspergillus_nidulans_turing_wavelength_s0.lean`, `delphi10000-aspergillus-nidulans-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.81 | 0.00 | 0 | 9,327 | 0 | **9,327** | 9,327 | `2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.81 | 0.00 | 0 | 128 | 0 | **128** | 128 | `aspergillus_nidulans_turing_wavelength_s0_growth.gif`, `aspergillus_nidulans_turing_wavelength_s0_comparison.png` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.82 | 0.00 | 0 | 6,194 | 0 | **6,194** | 6,194 | `2026-10-27-aspergillus-nidulans-turing-wavelength-s0-audit.yaml`, `2026-10-27-aspergillus-nidulans-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.82 | 0.00 | 0 | 2,016 | 0 | **2,016** | 2,016 | `2026-10-27-aspergillus-nidulans-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.83 | 0.00 | 0 | 2,878 | 0 | **2,878** | 2,878 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.83 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.83 | 0.00 | 0 | 2,878 | 0 | **2,878** | 2,878 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.84 | 0.00 | 0 | 687 | 0 | **687** | 687 | `2026-10-27-aspergillus-nidulans-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1331.05** | **0.00** | **0** | **29,667** | **0** | **29,667** | **29,667** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia; slot 3122 not in campaign_100; Slot 3122 is not in campaign_100. OpenAlex keyword search was off-target; seed DOIs resolved directly. No collision with slot 182.
- **Artifact:** `datasets/aspergillus_nidulans_turing_wavelength_s0/scout_manifest.json` (7,714 B, `2,226` tok, SHA-256 `19fc61a95e05aacb…`)

### Step 2 — `paper` (Paper)
- **Notes:** Schnakenberg onset law, scale 1.284e-05 m, age 0 Ma.
- **Artifact:** `datasets/aspergillus_nidulans_turing_wavelength_s0/aspergillus_nidulans_turing_wavelength_s0_reference.json` (3,154 B, `930` tok, SHA-256 `6b16f2c6f118230e…`)
- **Artifact:** `datasets/aspergillus_nidulans_turing_wavelength_s0/paper_citations.bib` (3,047 B, `953` tok, SHA-256 `ac5604136866cf10…`)
- **Artifact:** `datasets/aspergillus_nidulans_turing_wavelength_s0/README.md` (1,832 B, `484` tok, SHA-256 `a73a96f1084597c5…`)

### Step 3 — `source` (Formal source)
- **Notes:** Public-domain Wikimedia colony macrograph, SHA-256 pinned.
- **Artifact:** `datasets/aspergillus_nidulans_turing_wavelength_s0/aspergillus_nidulans_turing_wavelength_s0_specimen.jpg` (367,116 B, `64` tok, SHA-256 `282a45a621880a69…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3122, five theorems, zero sorry.
- **Artifact:** `morphoeval/lean/loops/Loop_3122_aspergillus_nidulans_turing_wavelength_s0.lean` (1,902 B, `64` tok, SHA-256 `760f255f86303568…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-aspergillus-nidulans-turing-wavelength-s0.yaml` (3,068 B, `838` tok, SHA-256 `ddc4e8de8aec4588…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 2 script with sweep, ablation, Sobel L1, three-seed CV.
- **Artifact:** `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` (32,399 B, `9,327` tok, SHA-256 `20fe487babc0c05d…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test passed. GIF 16 frames, 640x480.
- **Artifact:** `docs/media/aspergillus_nidulans_turing_wavelength_s0_growth.gif` (1,443,952 B, `64` tok, SHA-256 `743d23d1aec9c7fb…`)
- **Artifact:** `docs/media/aspergillus_nidulans_turing_wavelength_s0_comparison.png` (275,906 B, `64` tok, SHA-256 `44bf1efce2b3c4b5…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 83.8; pass 2 overall 88.1.
- **Artifact:** `morphoeval/audits/2026-10-27-aspergillus-nidulans-turing-wavelength-s0-audit.yaml` (12,142 B, `3,102` tok, SHA-256 `4e529e47d93484d1…`)
- **Artifact:** `morphoeval/audits/2026-10-27-aspergillus-nidulans-turing-wavelength-s0-audit-v2.yaml` (11,633 B, `3,092` tok, SHA-256 `af392e594ea1994d…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 2 sidecar: dispersion r=0.99773, Sobel L1=0.01424, seed CV=0.
- **Artifact:** `morphoeval/audits/2026-10-27-aspergillus-nidulans-turing-wavelength-s0-audit-v2.measurements.yaml` (7,311 B, `2,016` tok, SHA-256 `cf4a2315f313d3ba…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Both audits conform. Rubric nanopub reused from the existing Bos taurus graph, not reintroduced.
- **Artifact:** `morphoeval/kg/2026-10-27-aspergillus-nidulans-turing-wavelength-s0-audit-v2/REPORT.md` (10,414 B, `2,878` tok, SHA-256 `8a95017fd7666355…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml does not exist. Nanopubs were not signed and were not published.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md overall 88.1 for pass 2.
- **Artifact:** `morphoeval/kg/2026-10-27-aspergillus-nidulans-turing-wavelength-s0-audit-v2/REPORT.md` (10,414 B, `2,878` tok, SHA-256 `8a95017fd7666355…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed multi-seed CV, Sobel orientation L1, and alpha-sweep export. Gates pass; no pass 3.
- **Artifact:** `morphoeval/runs/2026-10-27-aspergillus-nidulans-turing-wavelength-s0-loop-image-fitness.json` (2,285 B, `687` tok, SHA-256 `361494d5fe0f77a9…`)
