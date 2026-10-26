# Loop Pass 7: Second Fable Round (Blinded, on the Pass-6 Tree)

**Assessor:** [Fable re-audit](da855146-5c25-4cfb-9d95-95878fa6b8c2) (`claude-fable-5-1`), blinded as in Pass 4. It re-derived the Turing maths, reproduced the committed metrics/PNG/GIF **byte-identically** in an isolated copy, verified all 10 specimen SHAs and spot-checked licences, and re-segmented every ROI.
**Audit:** `morphoeval/audits/2026-10-26-bos-taurus-turing-wavelength-second-assessor-audit-v2.yaml` (valid; note: checkpoint commit is local-only, not pushed).

## Panel

| Dimension | Builder v5 | Fable round 2 | (Fable round 1 → builder then) |
|---|---:|---:|---|
| model-fidelity | 72 | 64 | |
| specimen-match | 58 | 38 | |
| parameter-expressivity | 80 | 60 | |
| morphometric-validation | 85 | 50 | |
| source-traceability | 90 | 68 | |
| code-traceability | 72 | 66 | |
| repro-package | 78 | **82** | |
| **Overall** | **73.8** | **57.7** | (54.4 vs 68.6 in round 1) |

## New findings (all carried into the next remediation)

1. **ROI contamination in ≥6/10 specimens** (head in 08, canopy in 09, grass/shadow in 03/05/06/10; specimen 07's lone "enclave" is sky+coat). Held-out stats partly measure crop geometry — full coat mask (R3b) is now unavoidable, not optional.
2. **Roughness is resolution-dependent:** cow-01 enclave roughness 9.7 → 6.3 when downsampled to the 160-px lattice — much closer to RD (5.3) than Eden (8.9). Eden's pixel-scale fuzz is being credited as cow-like jaggedness. Needs scale-matched or scale-free shape statistics.
3. **The Eden > RD ranking is not significant:** paired tests p≈0.2 (both passes), Eden's Pass-5 mean pulled by one seed. No uncertainties reported anywhere; the figure title states the ranking flatly. Needs ≥20 seeds + paired tests/bootstrap CIs before the claim is made in those terms.
4. **Aspect-ratio bias:** diameters normalised by width only; ROI aspect varies (03 is 2.8:1 vs lattice 4:3), biasing diameter terms. Use √area normalisation.
5. **Model nitpicks with teeth:** the γ-heterogeneity field is re-drawn each growth stage (not tissue-attached); growth has no dilution term; the Eden control cites no source (Eden 1961 / Mort et al. 2016).
6. **Attribution, round 2:** the "Crossref-verified" Edea author list was still wrong in initials — now fixed from Crossref directly (Zewdu Edea, Hailu Dadi, Tadelle Dessie, Il-Hoi Kim, Kwan-Suk Kim); CC creator credits for all 9 images added to the reference JSON (both fixed after this audit); README updated from n=1.

**Fixed immediately after this round:** Edea authors (script + `.bib`), CC creator credits (reference JSON), README headline. **Not yet fixed:** R3b coat mask, scale-free roughness, significance testing + 20 seeds, √area normalisation, tissue-attached heterogeneity/dilution, Eden citations, self-test Pass 6, pushing the checkpoint commits.
