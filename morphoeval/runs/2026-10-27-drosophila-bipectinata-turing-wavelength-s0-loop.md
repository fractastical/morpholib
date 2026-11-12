# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-bipectinata-turing-wavelength-s0-loop`

## Result (slot 3178)

*Drosophila bipectinata*, Holocene 0 Ma, detail rung 1.92 µm. Schnakenberg kinetics with gamma solved so the continuous Turing wavelength is 1.92 µm. At the same age, *D. biarmipes* (D_u × 25/16) is 2.40 µm and *D. buzzatii* (D_u × 16/25) is 1.536 µm.

Pass 1 audit overall **76.4**. Pass 2 closed the sweep, finite-amplitude ablation, specimen segmentation, and dispersion GoF. Pass 2 overall **85.3**. Even-mode Pearson r **0.99478**. CV similarity **93.0**. Image fitness **88.4**. GIF 640×480, 13 frames. Pass 3 was not required.

Signing was not performed: `~/.nanopub/profile.yml` is absent, and `sign_morphoeval.py` also failed to import `requests`. The knowledge graphs passed `build_morphoeval_kg.py --conform` unsigned. The publisher URI on those graphs is `https://w3id.org/delphi/agent/loop-3178-unsigned`, not a personal ORCID.

The catalog DOI `10.1111/j.1525-142X.2005.05044.x` was rejected; Crossref assigns it to a zebrafish myostatin paper. The morphology anchor is Kopp & True 2002, `10.1046/j.1525-142x.2002.02017.x`. No CC photograph was found; the specimen file is the gallery fallback.

Tracker wall-clock below is bookkeeping only. The solver self-test itself takes a few seconds.

- **Target simulation:** `delphi10000-drosophila-bipectinata-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila bipectinata Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `grok-4.7`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.07 s** (0.02 min) — Tier A: `0.54 s` · Tier B: `0.53 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1.07** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Commons/GBIF/iNat; slot 3178 not in campaign_100 yaml; No s0 collision. Catalog DOI 10.1111/j.1525-142X.2005.05044.x resolves to Kerr 2005 myostatin, not Kopp. Real anchor Kopp & True 2002 10.1046/j.1525-142x.2002.02017.x. No CC photo on GBIF/iNat/Commons; gallery fallback.

### Step 2 — `paper` (Paper)
- **Notes:** loop 3178; completed; see loop report

### Step 3 — `source` (Formal source)
- **Notes:** loop 3178; completed; see loop report

### Step 4 — `rules` (Pin the rules)
- **Notes:** loop 3178; completed; see loop report

### Step 5 — `script` (Script)
- **Notes:** loop 3178; completed; see loop report

### Step 6 — `output` (Output & compare)
- **Notes:** loop 3178; completed; see loop report

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** loop 3178; completed; see loop report

### Step 8 — `measure` (Measure)
- **Notes:** loop 3178; completed; see loop report

### Step 9 — `build` (Write as claims)
- **Notes:** loop 3178; completed; see loop report

### Step 10 — `publish` (Sign & verify)
- **Notes:** loop 3178; completed; see loop report

### Step 11 — `graph` (Public record)
- **Notes:** loop 3178; completed; see loop report

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** loop 3178; completed; see loop report
