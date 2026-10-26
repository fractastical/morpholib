# Loop Pass 9: Publish the tested negative hypothesis

**Framing.** The loop is re-issued as an explicit claim with an assessment, not as a model showcase:

> **Claim:** "The black-and-white spotting of Holstein-Friesian cattle is laid down by a Turing reaction–diffusion pre-pattern (Murray 1981) with a characteristic wavelength."
>
> **Assessment (negative):** static Turing rejected (held-out distance 6.46 vs ~3.3); growing-domain RD indistinguishable from stochastic melanoblast colonisation at matched measurement scale (3.29 vs 3.43, Wilcoxon p = 0.91, Pass 8). Adult patch statistics are not evidence for a Turing mechanism in this system.

**Why publish a negative:** the claim is famous (Murray 1981 is the canonical citation for "Turing explains animal coats"), the test is quantitative and held-out (10 specimens, coat-masked, scale-matched, paired statistics), and the result agrees with the causal genetics (KIT/MITF regulatory variants, Jivanji et al. 2019/2025; stochastic colonisation modelling, Mort et al. 2016 — all Crossref-verified 2026-10-26 and added to `paper_citations.bib`). Two blinded independent (Fable) assessments scored the package 54.4 and 57.7 and overturned two builder conclusions en route; that adversarial record is part of the published evidence, not an embarrassment to bury.

**Changes:** README rewritten around the claim + verdict + claim boundaries; script docstring carries the CLAIM UNDER TEST block; `.bib` gains Mort 2016, Jivanji 2019, Jivanji 2025; audit v7 (supersedes v6) re-issues the scores at the publication checkpoint with an explicit `claim_boundary` (reported / reconstructed / transfer / does-not-claim).

**Publication (MorphoEval steps 9–11):** KG built with `--conform` → signed → published to the test registry → verified → published to production; trusty URIs recorded in `morphoeval/kg/2026-10-26-bos-taurus-turing-wavelength-audit-v7/published.json`. The rubric score (74.6) is the weight-of-evidence for the *simulation package's faithfulness*, and the claim nanopub's boundary text states the negative biological verdict so the score cannot be misread as support for Turing patterning in cows.

**Still open:** R9 migration-wave colonisation variant; R11 independent re-score of the Pass 8+ tree (the published assessment is builder-scored, disclosed as such in the claim boundary).
