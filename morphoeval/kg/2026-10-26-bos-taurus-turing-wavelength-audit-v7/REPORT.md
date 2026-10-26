# MorphoEval report — audit `2026-10-26-bos-taurus-turing-wavelength-audit-v7`

Rubric **0.1.0** · checkpoint `2e5574cb673de7d817650ef02de8f6b9e9e9394f` · assessor `pkr:model/muse-wonder-1.0-high` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-26-bos-taurus-turing-wavelength-audit -> 2026-10-26-bos-taurus-turing-wavelength-audit-v2 -> 2026-10-26-bos-taurus-turing-wavelength-audit-v3 -> 2026-10-26-bos-taurus-turing-wavelength-audit-v4 -> 2026-10-26-bos-taurus-turing-wavelength-audit-v5 -> 2026-10-26-bos-taurus-turing-wavelength-audit-v6 -> 2026-10-26-bos-taurus-turing-wavelength-audit-v7`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Turing RD claim for Holstein coat spotting (tested negative hypothesis) (`models/1981-murray-bos-taurus-turing-wavelength.py`) | 74 | 60 | 80 | 85 | 90 | 74 | 76 | 76.0 | 69.4 | 80.1 | 74.6 | **74.6** | mostly faithful |

## Findings and remediation

### Turing RD claim for Holstein coat spotting (tested negative hypothesis)  —  overall 74.6 (mostly faithful)

TESTED NEGATIVE HYPOTHESIS, published as such. Claim tested: the black-and-white spotting of Holstein-Friesian cattle is laid down by a Turing reaction-diffusion pre-pattern with a characteristic wavelength (Murray 1981). Verdict: not supported. A static single-wavelength Turing pattern is rejected on 10 coat-masked specimens (held-out morphometric distance 6.46 vs ~3.3 for body-scale alternatives; the cows carry a few body-scale patches, specimen 01: 3 patches, 99.2% of pigment in one, where the Turing field makes ~125 uniform spots at lambda_c = 10.26 px). A growing-domain RD variant cannot be distinguished from a stochastic melanoblast-colonisation (Eden) control once all masks are measured at the same scale (held-out RD growing-het 3.29 vs Eden 3.43; Eden wins 4/9; paired Wilcoxon p = 0.91; bootstrap 95% CI of the mean difference -0.80..1.14), so adult patch statistics are not evidence for a Turing mechanism in this system. The negative result agrees with the causal genetics: KIT/MITF regulatory variants cause Holstein spotting (Jivanji et al. 2019, 2025) acting through melanoblast survival, proliferation and migration, modelled quantitatively as stochastic colonisation (Mort et al. 2016); no activator-inhibitor morphogen pair is known in cattle skin. Scores below (overall 74.6) rate the simulation package's fidelity and reproducibility, never the truth of the tested claim.

- **Model fidelity to source: 74** — Tissue-attached heterogeneity and a stated no-dilution limitation improve the growing-domain model's honesty; Eden remains a minimal Eden process without a literature-pinned migration law.  
  evidence: `models/1981-murray-bos-taurus-turing-wavelength.py` L233–290
- **Morphological realism / specimen match: 60** — Coat-masked specimen panels now track real patch shapes with background excluded (montage/full animation); model panels unchanged in character -- RD smooth ovals, Eden jagged networks, neither matching patch placement.  
  evidence: `datasets/bos_taurus_turing_wavelength/bos_taurus_turing_wavelength_full_animation.gif` L0–0
- **Parameter expressivity: 80** — Unchanged sweeps; 12 seeds now used in the held-out comparison.  
  evidence: `models/1981-murray-bos-taurus-turing-wavelength.py` L513–560
- **Morphometric validation: 85** — Coat-domain, scale-matched, aspect-corrected morphometrics with paired significance testing and bootstrap CIs; the pipeline now overturns the builder's own earlier conclusion, which is what validation is for. Remaining: coat segmentation is saturation-heuristic (fails where coloured background is achromatic), n=9 held-out specimens limits power, Eden founder density was still selected on cow 01.  
  evidence: `models/1981-murray-bos-taurus-turing-wavelength.py` L442–512; `models/1981-murray-bos-taurus-turing-wavelength.py` L513–560
- **Source and data traceability: 90** — 10 pinned, credited, licensed specimens; all papers Crossref-verified, including the 2026-10-26 additions (Mort 2016; Jivanji 2019, 2025) that carry the established melanoblast account.  
  evidence: `datasets/bos_taurus_turing_wavelength/paper_citations.bib` L35–64
- **Code-to-model traceability: 74** — Pass-8/9 changes carry in-situ Fable-round citations and the CLAIM UNDER TEST block states the verdict in the script header; animate paths still duplicate some constants.  
  evidence: `models/1981-murray-bos-taurus-turing-wavelength.py` L57–80
- **Reproducibility package: 76** — Self-test green at the publication checkpoint; deterministic; but the full pipeline is ~100 s and the animation generators duplicate the growing-domain loop (Fable nit, still open).  
  evidence: `datasets/bos_taurus_turing_wavelength/README.md` L7–21

Remediation:
- {'id': 'R9', 'item': 'Cite Eden (1961) / a melanoblast migration source for the control; consider a migration-wave (neural-crest front) variant, whose enclaves invade from edges like real Holstein white'}
- {'id': 'R11', 'item': 'Independent re-score of the Pass-8+ tree before any Eden-vs-RD claim is quoted (the published negative verdict on static Turing does not depend on it; the RD-vs-colonisation null does)'}

## Environment at checkpoint


Pass-9 publication checkpoint: claim framing, verdict and claim boundaries committed; KG built, signed and nanopub-published from this audit.
