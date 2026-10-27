# MorphoEval Full-Loop Telemetry — `2026-10-27-cooksonia-pertoni-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-cooksonia-pertoni-turing-wavelength-s0`
- **Primary paper / dataset:** Cooksonia pertoni Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` only. No second assessor ran. The default second-assessor name was not used.
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.01 s** (0.02 min) — Tier A: `0.52 s` · Tier B: `0.49 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **42,762** (input: `0`, output: `42,762`, subagent: `0`) — Tier A: `16,380` · Tier B: `26,382`
- **Total artifact tokens produced:** `42,762`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,084 | 0 | **1,084** | 1,084 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,691 | 0 | **2,691** | 2,691 | `cooksonia_pertoni_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 64 | 0 | **64** | 64 | `cooksonia_pertoni_amgueddfa_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 957 | 0 | **957** | 957 | `Loop_3153_cooksonia_pertoni_turing_wavelength_s0.lean`, `delphi10000-cooksonia-pertoni-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 11,456 | 0 | **11,456** | 11,456 | `2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 128 | 0 | **128** | 128 | `cooksonia_pertoni_turing_wavelength_s0_comparison.png`, `cooksonia_pertoni_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 6,049 | 0 | **6,049** | 6,049 | `2026-10-27-cooksonia-pertoni-turing-wavelength-s0-audit.yaml`, `2026-10-27-cooksonia-pertoni-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,101 | 0 | **2,101** | 2,101 | `2026-10-27-cooksonia-pertoni-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,898 | 0 | **2,898** | 2,898 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 980 | 0 | **980** | 980 | `2026-10-27-cooksonia-pertoni-turing-wavelength-s0-loop-image-fitness.json` |
| 11 | B | **graph** (Public record) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,898 | 0 | **2,898** | 2,898 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 11,456 | 0 | **11,456** | 11,456 | `2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` |
| — | **All** | **TOTAL** | — | `12/12` | **1.01** | **0.00** | **0** | **42,762** | **0** | **42,762** | **42,762** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia. No slug collision with the 73 µm sibling.; OpenAlex/Crossref/Wikimedia. No slug collision with the 73 µm sibling.
- **Artifact:** `datasets/cooksonia_pertoni_turing_wavelength_s0/scout_manifest.json` (3,758 B, `1,084` tok, SHA-256 `f458deaa22521915…`)

### Step 2 — `paper` (Paper)
- **Notes:** Equations, Jacobian, 1.3 µm and 425 Ma written into the reference.; Equations, Jacobian, 1.3 µm and 425 Ma written into the reference.
- **Artifact:** `datasets/cooksonia_pertoni_turing_wavelength_s0/cooksonia_pertoni_turing_wavelength_s0_reference.json` (5,207 B, `1,476` tok, SHA-256 `93a1a0f57016c745…`)
- **Artifact:** `datasets/cooksonia_pertoni_turing_wavelength_s0/paper_citations.bib` (1,686 B, `567` tok, SHA-256 `e7e2f5f78fe35466…`)
- **Artifact:** `datasets/cooksonia_pertoni_turing_wavelength_s0/README.md` (2,391 B, `648` tok, SHA-256 `df25909b58621b07…`)

### Step 3 — `source` (Formal source)
- **Notes:** CC BY-SA Amgueddfa Cymru compression vendored and SHA-256 pinned.; CC BY-SA Amgueddfa Cymru compression vendored and SHA-256 pinned.
- **Artifact:** `datasets/cooksonia_pertoni_turing_wavelength_s0/cooksonia_pertoni_amgueddfa_specimen.jpg` (3,097,667 B, `64` tok, SHA-256 `aa48bf1c9b50b41d…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Six Lean theorems, 0 sorry, plus the math-kg loop yaml.; Six Lean theorems, 0 sorry, plus the math-kg loop yaml.
- **Artifact:** `morphoeval/lean/loops/Loop_3153_cooksonia_pertoni_turing_wavelength_s0.lean` (1,203 B, `64` tok, SHA-256 `8671fec364c53fcb…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-cooksonia-pertoni-turing-wavelength-s0.yaml` (3,222 B, `893` tok, SHA-256 `3e83686867da5e83…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg cylinder PDE, --self-test passed.; Schnakenberg cylinder PDE, --self-test passed.
- **Artifact:** `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` (39,491 B, `11,456` tok, SHA-256 `7544cc8e4fcf0882…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Comparison PNG and 12-frame 720x540 growth GIF.; Comparison PNG and 12-frame 720x540 growth GIF.
- **Artifact:** `docs/media/cooksonia_pertoni_turing_wavelength_s0_comparison.png` (624,221 B, `64` tok, SHA-256 `93e88f9e5d0b1916…`)
- **Artifact:** `docs/media/cooksonia_pertoni_turing_wavelength_s0_growth.gif` (5,887,468 B, `64` tok, SHA-256 `cf9f5da25f5ea446…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 85.8. Pass 2 overall 86.6.; Pass 1 overall 85.8. Pass 2 overall 86.6.
- **Artifact:** `morphoeval/audits/2026-10-27-cooksonia-pertoni-turing-wavelength-s0-audit.yaml` (11,878 B, `3,042` tok, SHA-256 `778e72ecc22000e4…`)
- **Artifact:** `morphoeval/audits/2026-10-27-cooksonia-pertoni-turing-wavelength-s0-audit-v2.yaml` (11,436 B, `3,007` tok, SHA-256 `fe9663f6be615e94…`)

### Step 8 — `measure` (Measure)
- **Notes:** 12 measurements, quick_gate 100.; 12 measurements, quick_gate 100.
- **Artifact:** `morphoeval/audits/2026-10-27-cooksonia-pertoni-turing-wavelength-s0-audit-v2.measurements.yaml` (7,429 B, `2,101` tok, SHA-256 `1552912ccf46bf43…`)

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg.py --conform --allow-reintroduction. Conformance OK.; build_morphoeval_kg.py --conform --allow-reintroduction. Conformance OK.
- **Artifact:** `morphoeval/kg/2026-10-27-cooksonia-pertoni-turing-wavelength-s0-audit-v2/REPORT.md` (10,285 B, `2,898` tok, SHA-256 `675381b4964e49c1…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no nanopub profile and requests is not installed.; Signing unavailable: no nanopub profile and requests is not installed.
- **Artifact:** `morphoeval/runs/2026-10-27-cooksonia-pertoni-turing-wavelength-s0-loop-image-fitness.json` (3,267 B, `980` tok, SHA-256 `e1e6fe9bf8d0722b…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md written for audit v2 with supersession chain.; REPORT.md written for audit v2 with supersession chain.
- **Artifact:** `morphoeval/kg/2026-10-27-cooksonia-pertoni-turing-wavelength-s0-audit-v2/REPORT.md` (10,285 B, `2,898` tok, SHA-256 `675381b4964e49c1…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 restored axial bands and refined the branch-angle template. Gates passed. No pass 3.; Pass 2 restored axial bands and refined the branch-angle template. Gates passed. No pass 3.
- **Artifact:** `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` (39,491 B, `11,456` tok, SHA-256 `7544cc8e4fcf0882…`)

## Scientific result

*Cooksonia pertoni*, Late Silurian (Wenlock–Ludlow, 425 Ma), detail scale 1.3 µm (1.339e-6 m). Schnakenberg kinetics with D_v/D_u = 10. Selected wavenumber k_c = 2.356, wavelength 1.339 µm. Equal diffusion (pre-vascular predecessor) has σ_max = −0.086. *Rhynia gwynne-vaughanii* at 407 Ma, with a longer activator diffusion length, moves the wavelength to 1.695 µm.

The fitness target is the Amgueddfa Cymru compression (CC BY-SA 4.0), not the synthetic gallery image. Pass 2 image fitness is 91.2 (CV similarity 95.8, specimen-match 84, morphometric validation 89). Dispersion Pearson r is 1.000. Specimen radial Pearson r is 0.988. The GIF is 12 frames at 720×540. Audit v2 overall is 86.6. Pass 3 was not required.

Two pass-1 items stay open: there is no second assessor, and the nanopubs are unsigned because `~/.nanopub/profile.yml` is missing and `sign_morphoeval.py` cannot import `requests`. The 1.3 µm length is the assigned detail scale. Edwards, Davies & Axe (1992) showed annular to spiral tracheids; they did not publish this wavelength, and the museum photograph does not resolve it. The bands on the GIF are a magnified display of the axial Turing cut.

The step table above is the telemetry backfill. Each `--self-test` run of the PDE took about 14 seconds.
