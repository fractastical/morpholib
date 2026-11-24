# MorphoEval Full-Loop Telemetry — `2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-chamaeleo-calyptratus-turing-wavelength-s0`
- **Primary paper / dataset:** Chamaeleo calyptratus Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **310.00 s** (5.17 min) — Tier A: `185.00 s` · Tier B: `125.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **31,233** (input: `0`, output: `31,233`, subagent: `0`) — Tier A: `15,646` · Tier B: `15,587`
- **Total artifact tokens produced:** `31,233`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 1,964 | 0 | **1,964** | 1,964 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 2,572 | 0 | **2,572** | 2,572 | `chamaeleo_calyptratus_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 64 | 0 | **64** | 64 | `Chamaeleo_calyptratus_HZell_CC-BY-SA-3.0.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 25.00 | 0.00 | 0 | 851 | 0 | **851** | 851 | `Loop_3144_chamaeleo_calyptratus_turing_wavelength_s0.lean`, `delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 50.00 | 0.00 | 0 | 10,067 | 0 | **10,067** | 10,067 | `2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 128 | 0 | **128** | 128 | `chamaeleo_calyptratus_turing_wavelength_s0_comparison.png`, `chamaeleo_calyptratus_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 35.00 | 0.00 | 0 | 4,902 | 0 | **4,902** | 4,902 | `2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-audit.yaml`, `2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 15.00 | 0.00 | 0 | 2,056 | 0 | **2,056** | 2,056 | `2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 20.00 | 0.00 | 0 | 2,724 | 0 | **2,724** | 2,724 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 5.00 | 0.00 | 0 | 654 | 0 | **654** | 654 | `2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-loop-image-fitness.json` |
| 11 | B | **graph** (Public record) | tool | `completed` | 10.00 | 0.00 | 0 | 2,724 | 0 | **2,724** | 2,724 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 40.00 | 0.00 | 0 | 2,527 | 0 | **2,527** | 2,527 | `2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-audit-v2.yaml` |
| — | **All** | **TOTAL** | — | `12/12` | **310.00** | **0.00** | **0** | **31,233** | **0** | **31,233** | **31,233** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia; slot 3144 is outside the 100-slot campaign file. No s0 collision.; OpenAlex/Crossref/Wikimedia; slot 3144 is outside the 100-slot campaign file. No s0 collision.
- **Artifact:** `datasets/chamaeleo_calyptratus_turing_wavelength_s0/scout_manifest.json` (6,878 B, `1,964` tok, SHA-256 `cbfa3fc2b6d381da…`)

### Step 2 — `paper` (Paper)
- **Notes:** Equations, 260.4 um, 0 Ma, transfer boundary versus Teyssier 2015.; Equations, 260.4 um, 0 Ma, transfer boundary versus Teyssier 2015.
- **Artifact:** `datasets/chamaeleo_calyptratus_turing_wavelength_s0/chamaeleo_calyptratus_turing_wavelength_s0_reference.json` (4,019 B, `1,245` tok, SHA-256 `da3690c545638865…`)
- **Artifact:** `datasets/chamaeleo_calyptratus_turing_wavelength_s0/paper_citations.bib` (1,754 B, `530` tok, SHA-256 `356decc154f563d0…`)
- **Artifact:** `datasets/chamaeleo_calyptratus_turing_wavelength_s0/README.md` (2,732 B, `797` tok, SHA-256 `351e8fadee99f50d…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored H. Zell CC BY-SA 3.0 photograph; SHA-256 pinned.; Vendored H. Zell CC BY-SA 3.0 photograph; SHA-256 pinned.
- **Artifact:** `datasets/chamaeleo_calyptratus_turing_wavelength_s0/Chamaeleo_calyptratus_HZell_CC-BY-SA-3.0.jpg` (2,000,401 B, `64` tok, SHA-256 `b37709d121c1b455…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3144, six theorems, zero sorry, plus the math KG yaml.; Lean Loop_3144, six theorems, zero sorry, plus the math KG yaml.
- **Artifact:** `morphoeval/lean/loops/Loop_3144_chamaeleo_calyptratus_turing_wavelength_s0.lean` (1,929 B, `64` tok, SHA-256 `4226fb707af17d56…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.yaml` (2,804 B, `787` tok, SHA-256 `7967a591c2365e92…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg IMEX, sweep, ablation, flank ROI, self-test.; Schnakenberg IMEX, sweep, ablation, flank ROI, self-test.
- **Artifact:** `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` (34,422 B, `10,067` tok, SHA-256 `7189e56ffd8fd84e…`)

### Step 6 — `output` (Output & compare)
- **Notes:** comparison PNG and 12-frame 720x540 GIF.; comparison PNG and 12-frame 720x540 GIF.
- **Artifact:** `docs/media/chamaeleo_calyptratus_turing_wavelength_s0_comparison.png` (1,092,561 B, `64` tok, SHA-256 `9f12527c529ae712…`)
- **Artifact:** `docs/media/chamaeleo_calyptratus_turing_wavelength_s0_growth.gif` (3,264,774 B, `64` tok, SHA-256 `2df160546a80217a…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 81.4; Pass 2 overall 86.3.; Pass 1 overall 81.4; Pass 2 overall 86.3.
- **Artifact:** `morphoeval/audits/2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-audit.yaml` (9,217 B, `2,375` tok, SHA-256 `f6e0475e87e4b110…`)
- **Artifact:** `morphoeval/audits/2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-audit-v2.yaml` (9,724 B, `2,527` tok, SHA-256 `866e2f823325636a…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100; Pearson r 0.999.; measure_outputs quick_gate 100; Pearson r 0.999.
- **Artifact:** `morphoeval/audits/2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-audit-v2.measurements.yaml` (7,832 B, `2,056` tok, SHA-256 `cb23ce97ce73253d…`)

### Step 9 — `build` (Write as claims)
- **Notes:** ProbKnow conformance OK. Signing unavailable.; ProbKnow conformance OK. Signing unavailable.
- **Artifact:** `morphoeval/kg/2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-audit-v2/REPORT.md` (10,037 B, `2,724` tok, SHA-256 `22e286bd671b533f…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** No ~/.nanopub/profile.yml. Unsigned. Placeholder ORCID is not an identity.; No ~/.nanopub/profile.yml. Unsigned. Placeholder ORCID is not an identity.
- **Artifact:** `morphoeval/runs/2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-loop-image-fitness.json` (2,241 B, `654` tok, SHA-256 `119599a9e2ef0977…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md overall 86.3 faithful.; REPORT.md overall 86.3 faithful.
- **Artifact:** `morphoeval/kg/2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-audit-v2/REPORT.md` (10,037 B, `2,724` tok, SHA-256 `22e286bd671b533f…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 flank composite, flank ROI, ASCII caption. Gates passed; no pass 3.; Pass 2 flank composite, flank ROI, ASCII caption. Gates passed; no pass 3.
- **Artifact:** `morphoeval/audits/2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-audit-v2.yaml` (9,724 B, `2,527` tok, SHA-256 `866e2f823325636a…`)
