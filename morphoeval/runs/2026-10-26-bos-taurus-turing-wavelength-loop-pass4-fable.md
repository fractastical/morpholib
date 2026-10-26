# Loop Pass 4: Fable Second-Assessor Round (Blinded)

**Assessor:** [Fable second assessor](b1da8f01-150b-4ef1-ba26-23efb0b395fd) (`claude-fable-5-1`, Anthropic) — fresh agent, instructed *not* to read the builder's audits, run records, or scoring tables; given the rubric, the script, and the dataset. It independently re-derived the linear theory, re-ran the script (metrics JSON reproduced byte-for-byte), segmented/inspected the outputs, and verified all DOIs via Crossref.
**Audit:** `morphoeval/audits/2026-10-26-bos-taurus-turing-wavelength-second-assessor-audit.yaml` (validates against rubric 0.1.0). This closes R1 from passes 1–3.

## Panel

| Dimension (weight) | Builder v3 (Muse) | Fable (blinded) | Δ |
|---|---:|---:|---:|
| model-fidelity (20) | 70 | 62 | −8 |
| specimen-match (25) | 52 | 35 | −17 |
| parameter-expressivity (10) | 82 | 58 | −24 |
| morphometric-validation (15) | 72 | 40 | −32 |
| source-traceability (10) | 85 | 70 | −15 |
| code-traceability (8) | 68 | 60 | −8 |
| repro-package (12) | 72 | 80 | **+8** |
| **Overall** | **68.6** | **54.4** | **−14.2** |

## What Fable found that the builder missed

1. **The Pass-1 headline verdict is partly a polarity artefact.** "142 spots vs 1 patch" depends on counting high-activator as pigment; with the opposite polarity *any* spot field is one connected patch. The robust discriminator is the enclave distribution (cow CV 0.99 / roughness 11.9 vs sim 0.46 / 5.0) — which still separates them, but the verdict text overclaimed.
2. **The Pass-2/3 fit is tautological on its two heaviest terms.** Coverage equals the chosen threshold quantile by construction, and largest-patch share ≈ 1 is guaranteed by the connected background. A static γ=0.1 field with the same polarity already scores 1 patch / share 1.0 / 11 enclaves. The growing/heterogeneous machinery is not what produces the headline match.
3. **The fit leaderboard is inside seed noise** (winning config: 0.88–2.17 across 5 seeds; top-5 span 1.21–1.57), so "growing-het > growing > static" is unresolved.
4. **ROI artefacts:** the rectangular flank ROI merges background above the back, ground/shadow, and udder/legs into the "coat"; the cow's roughness is inflated by JPEG speckle at 1112 px relative to the 160-px lattice.
5. **Attribution errors (fixed after this round):** the kinetics are Schnakenberg (1979), not Murray's own coat-marking kinetics (now cited, docstring corrected); the MITF paper is Edea et al., *Genes & Genomics* 39(3):285–293 — not "Bacha et al." (docstring + `.bib` corrected, Crossref-verified); the roughness disc constant is ≈4.6 for the 4-neighbour perimeter, not 3.54 (comment corrected).
6. Minor: self-test covers no Pass-2/3 code; `--no-show` is a no-op; ROI fractions and growth stages hard-coded in three places; `animate()` duplicates `simulate_growing()`.

Fable also *confirmed*: the dispersion mathematics, the equal-diffusion control, DOI resolution, lockfile-vs-venv match, and bit-level reproducibility of the metrics JSON.

## Carried forward (Fable R1–R11, merged with earlier items)

Remove/disclose the tautological fit terms and report multi-seed distances before claiming any model class wins · reframe the Pass-1 verdict around polarity · body/coat mask instead of the rectangle (R3) · ≥10 specimens (R2) · melanoblast positive control (R5) — now the best-motivated next step, since enclave geometry is the statistic that actually discriminates · extend self-test to Pass-2/3 code · commit everything so the checkpoint pins the cited loci.
