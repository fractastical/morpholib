# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-pseudoobscura-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-pseudoobscura-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila pseudoobscura Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **0.00 s** (0.00 min) — Tier A: `0.00 s` · Tier B: `0.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **0.00** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

## Outcome (Pass 2 — gates)

The tracker table above records step completion only. Wall-clock cells are 0.00 s because `track_loop.py start` was called with a flag this build rejects, so each step was closed without a timed interval. The assessor for the audits is Grok 4.7, not the default names printed in the actor column.

*Drosophila pseudoobscura*, detail rung 2.0 µm (1.961e-6 m, log10 = -5.708), Holocene 0 Ma. Predecessor *D. planitibia* is the 197.2 µm primordium rung; successor *D. santomea* is the 102 µm primordium rung. Age does not change (Δ = 0 Ma). The Turing fastest mode is 1.961 µm. Diffusivity sweep Pearson r = 0.99988. Cubic saturation plateaus at activator rms 0.258. Last-frame CV similarity to the CC BY-SA male (MA Hanson, 2019) is 83.1.

| Gate | Result |
| --- | --- |
| image_fitness >= 70 | 84.9 |
| cv_sim_similarity >= 58 | 83.1 |
| GIF frames >= 6 | 12 |
| resolution >= 340 px | 640×480 |
| non-circular specimen metric | circularity 0.1052 vs reference 1.0; centroid 1.9571 vs 1.961 |
| Pearson r > 0.98 | 0.99988 |
| audit overall >= 85 | 89.3 (pass 2); pass 1 was 86.1 |

Pass 3 was not run. No gate failed.

Signing was not performed. `~/.nanopub/profile.yml` is absent, and `morphoeval/sign_morphoeval.py` cannot import the `nanopub` package in this environment. The TriG files under `morphoeval/kg/2026-10-27-drosophila-pseudoobscura-turing-wavelength-s0-audit-v2/` are unsigned. Nothing was published to a registry.

