# MorphoEval Full-Loop Telemetry — `2026-10-27-cloudina-hartmannae-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-cloudina-hartmannae-turing-wavelength-s0`
- **Primary paper / dataset:** Cloudina hartmannae Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-independently-run`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **0.94 s** (0.02 min) — Tier A: `0.46 s` · Tier B: `0.48 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **35,840** (input: `0`, output: `35,840`, subagent: `0`) — Tier A: `15,701` · Tier B: `20,139`
- **Total artifact tokens produced:** `35,840`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | grok-4.7 (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 1,372 | 0 | **1,372** | 1,372 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | grok-4.7 (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 3,790 | 0 | **3,790** | 3,790 | `cloudina_hartmannae_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | grok-4.7 (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 192 | 0 | **192** | 192 | `cloudina_hartmannae_specimen.jpg`, `cloudina_hartmannae_wikimedia_cc_by_sa.jpg`, `cloudina_corumba_cc_by_2.jpg` |
| 4 | A | **rules** (Pin the rules) | grok-4.7 (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 1,398 | 0 | **1,398** | 1,398 | `Loop_3149_cloudina_hartmannae_turing_wavelength_s0.lean`, `delphi10000-cloudina-hartmannae-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | grok-4.7 (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 8,388 | 0 | **8,388** | 8,388 | `2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | grok-4.7 (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 561 | 0 | **561** | 561 | `cloudina_hartmannae_turing_wavelength_s0_comparison.png`, `cloudina_hartmannae_turing_wavelength_s0_growth.gif`, `cloudina_hartmannae_turing_wavelength_s0_metrics.json` |
| 7 | B | **audit** (Dual-agent audit) | grok-4.7 (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,141 | 0 | **2,141** | 2,141 | `2026-10-27-cloudina-hartmannae-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | grok-4.7 (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 1,907 | 0 | **1,907** | 1,907 | `2026-10-27-cloudina-hartmannae-turing-wavelength-s0-audit.measurements.yaml` |
| 9 | B | **build** (Write as claims) | grok-4.7 (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,273 | 0 | **2,273** | 2,273 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | grok-4.7 (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,273 | 0 | **2,273** | 2,273 | `REPORT.md` |
| 11 | B | **graph** (Public record) | grok-4.7 (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,273 | 0 | **2,273** | 2,273 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | grok-4.7 (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 9,272 | 0 | **9,272** | 9,272 | `2026-10-27-cloudina-hartmannae-turing-wavelength-s0-audit-v2.yaml`, `2026-10-27-cloudina-hartmannae-turing-wavelength-s0-audit-v2.measurements.yaml`, `REPORT.md`, +1 more |
| — | **All** | **TOTAL** | — | `12/12` | **0.94** | **0.00** | **0** | **35,840** | **0** | **35,840** | **35,840** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia. Slot 3149 is outside the 100-slot campaign file. No s0 filename collision. Sibling 104.2 µm slot 1252 excluded.; OpenAlex/Crossref/Wikimedia. Slot 3149 is outside the 100-slot campaign file. No s0 filename collision. Sibling 104.2 µm slot 1252 excluded.
- **Artifact:** `datasets/cloudina_hartmannae_turing_wavelength_s0/scout_manifest.json` (4,827 B, `1,372` tok, SHA-256 `b2c1cc8dc9098033…`)

### Step 2 — `paper` (Paper)
- **Notes:** Equations, 2.397e-6 m, 549 Ma, claim boundary.; Equations, 2.397e-6 m, 549 Ma, claim boundary.
- **Artifact:** `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_hartmannae_turing_wavelength_s0_reference.json` (6,305 B, `1,763` tok, SHA-256 `dcd1cbc35eecb656…`)
- **Artifact:** `datasets/cloudina_hartmannae_turing_wavelength_s0/paper_citations.bib` (3,146 B, `988` tok, SHA-256 `2aa1757e3ab334ef…`)
- **Artifact:** `datasets/cloudina_hartmannae_turing_wavelength_s0/README.md` (3,703 B, `1,039` tok, SHA-256 `9108101cdfe57f58…`)

### Step 3 — `source` (Formal source)
- **Notes:** Species card copied from the gallery; CC BY-SA life restoration and CC BY 2.0 Corumbá fossil vendored and SHA-256 pinned.; Species card copied from the gallery; CC BY-SA life restoration and CC BY 2.0 Corumbá fossil vendored and SHA-256 pinned.
- **Artifact:** `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_hartmannae_specimen.jpg` (97,334 B, `64` tok, SHA-256 `cbe0233dbf275f4e…`)
- **Artifact:** `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_hartmannae_wikimedia_cc_by_sa.jpg` (249,757 B, `64` tok, SHA-256 `f4c59996a8dc38dd…`)
- **Artifact:** `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_corumba_cc_by_2.jpg` (271,203 B, `64` tok, SHA-256 `f20328a5dcb570d1…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean 4 standalone, 5 theorems, 0 sorry. lean binary not installed, so proofs were not executed.; Lean 4 standalone, 5 theorems, 0 sorry. lean binary not installed, so proofs were not executed.
- **Artifact:** `morphoeval/lean/loops/Loop_3149_cloudina_hartmannae_turing_wavelength_s0.lean` (2,805 B, `64` tok, SHA-256 `6695dc82dc941188…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-cloudina-hartmannae-turing-wavelength-s0.yaml` (4,840 B, `1,334` tok, SHA-256 `19d5fe5edaeda8e2…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 1 integrator plus Pass 2 sweep, ablation, segmentation, and dispersion fit.; Pass 1 integrator plus Pass 2 sweep, ablation, segmentation, and dispersion fit.
- **Artifact:** `models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` (28,909 B, `8,388` tok, SHA-256 `bfff41e739e79850…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test passed. spacing 2.397 µm, r=0.99997, sweep r=0.9999, CV=91.5, 12 frames, 640x480.; Self-test passed. spacing 2.397 µm, r=0.99997, sweep r=0.9999, CV=91.5, 12 frames, 640x480.
- **Artifact:** `docs/media/cloudina_hartmannae_turing_wavelength_s0_comparison.png` (379,283 B, `64` tok, SHA-256 `bb3134888de8b71d…`)
- **Artifact:** `docs/media/cloudina_hartmannae_turing_wavelength_s0_growth.gif` (1,884,937 B, `64` tok, SHA-256 `e890de0a8d59dd17…`)
- **Artifact:** `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_hartmannae_turing_wavelength_s0_metrics.json` (1,175 B, `433` tok, SHA-256 `34a24c4d8ec75173…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 78.0. Remediations: sweep, controls, segmentation, goodness-of-fit.; Pass 1 overall 78.0. Remediations: sweep, controls, segmentation, goodness-of-fit.
- **Artifact:** `morphoeval/audits/2026-10-27-cloudina-hartmannae-turing-wavelength-s0-audit.yaml` (8,468 B, `2,141` tok, SHA-256 `61e193188e1b0e3c…`)

### Step 8 — `measure` (Measure)
- **Notes:** quick_gate 100/100. 12 measurements, values differ from references.; quick_gate 100/100. 12 measurements, values differ from references.
- **Artifact:** `morphoeval/audits/2026-10-27-cloudina-hartmannae-turing-wavelength-s0-audit.measurements.yaml` (6,866 B, `1,907` tok, SHA-256 `5313a72c865cbfb1…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Conformance OK. Unsigned. Placeholder ORCID. --allow-reintroduction because the shared rubric trig is already published elsewhere. --no-verify-loci because new files are not on the parent commit.; Conformance OK. Unsigned. Placeholder ORCID. --allow-reintroduction because the shared rubric trig is already published elsewhere. --no-verify-loci because new files are not on the parent commit.
- **Artifact:** `morphoeval/kg/2026-10-27-cloudina-hartmannae-turing-wavelength-s0-audit/REPORT.md` (8,654 B, `2,273` tok, SHA-256 `6ad2df9fb3b73e70…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable. ~/.nanopub/profile.yml absent. Did not run sign_morphoeval.py or check_signed.mjs. Did not invent a trusty URI.; Signing unavailable. ~/.nanopub/profile.yml absent. Did not run sign_morphoeval.py or check_signed.mjs. Did not invent a trusty URI.
- **Artifact:** `morphoeval/kg/2026-10-27-cloudina-hartmannae-turing-wavelength-s0-audit/REPORT.md` (8,654 B, `2,273` tok, SHA-256 `6ad2df9fb3b73e70…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md overall 78.0 on Pass 1.; REPORT.md overall 78.0 on Pass 1.
- **Artifact:** `morphoeval/kg/2026-10-27-cloudina-hartmannae-turing-wavelength-s0-audit/REPORT.md` (8,654 B, `2,273` tok, SHA-256 `6ad2df9fb3b73e70…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 overall 88.1. CV re-scored 91.5. image_fitness 88.7. Gates passed, so Pass 3 was not run. npx:supersedes not emitted (no published.json).; Pass 2 overall 88.1. CV re-scored 91.5. image_fitness 88.7. Gates passed, so Pass 3 was not run. npx:supersedes not emitted (no published.json).
- **Artifact:** `morphoeval/audits/2026-10-27-cloudina-hartmannae-turing-wavelength-s0-audit-v2.yaml` (12,108 B, `3,196` tok, SHA-256 `3a3943ca12a25226…`)
- **Artifact:** `morphoeval/audits/2026-10-27-cloudina-hartmannae-turing-wavelength-s0-audit-v2.measurements.yaml` (7,743 B, `2,174` tok, SHA-256 `8179b6379e0b585b…`)
- **Artifact:** `morphoeval/kg/2026-10-27-cloudina-hartmannae-turing-wavelength-s0-audit-v2/REPORT.md` (11,178 B, `3,041` tok, SHA-256 `d974e997ae4d4e21…`)
- **Artifact:** `morphoeval/runs/2026-10-27-cloudina-hartmannae-turing-wavelength-s0-loop-image-fitness.json` (2,882 B, `861` tok, SHA-256 `da408db6f75fdac8…`)

## Result (slot 3149)

*Cloudina hartmannae*, terminal Ediacaran Nama Group, 549 Ma, detail scale 2.397 µm. The predecessor rung is the 104.2 µm funnel scale (slot 1252); this rung is the wall-normal organic lamella (Yang et al. 2020 band 0.5–5 µm). Eight lamellae span 19.176 µm, inside Grant's 8–50 µm shell layer. Schnakenberg parameters are a=0.0642, b=0.9915, Dv/Du=8.65, with Du and Dv solved so λ_c=2.397 µm. They are not the textbook pair and not slot 1252's stub.

Pass 1 audit overall **78.0**. Pass 2 audit overall **88.1** after the diffusivity sweep (r=0.9999), equal-diffusion control (σmax=-0.124), autocatalysis ablation, species-card versus Corumbá segmentation, and real-eigenvalue dispersion fit (r=0.99997, RMSE=0.0312). The KS test against the band-wide normal rejects (statistic 0.623, p=0.0016): the field locks to one wavelength. Spacing standard deviation is 0 because the domain is exactly eight critical wavelengths.

Image fitness on Pass 2 is **88.7** = 0.55×91.5 + 0.30×84 + 0.15×88. GIF is 12 frames at 640×480. Pearson r is 0.99997. Every gate passed, so Pass 3 was not run.

Signing was not available: `~/.nanopub/profile.yml` is absent. Nanopubs are unsigned. The ORCID on the audits is the setup-guide placeholder, not a person. `lean` is not installed, so the five theorems were not executed. No second assessor scored the audits.
