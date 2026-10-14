# MorphoEval — faithfulness checkpoints for the morphology simulations

Sibling of **BioEval** / **NeuroEval** (Zenodo). Those score *papers* on a
versioned, weighted rubric; MorphoEval scores *our own simulation scripts* for
faithfulness to the organism and to the source model, and records each scoring
as a knowledge graph so a simulation can be **checkpointed and re-scored** after
every change.

```
morphoeval/
  rubric.yaml                      versioned rubric (7 weighted dimensions, 0-100)
  audits/<date>-<name>.yaml        one audit = scores + evidence + references + checkpoint
  build_morphoeval_kg.py           audit -> ProbKnow nanopubs (.trig) + merged .ttl + REPORT.md
  vendor/probknow-1.1.ttl          vendored ontology for the offline --conform check
  kg/<audit-id>/                   generated graph for that audit (commit it: it IS the checkpoint)
```

## Rubric 0.1.0

| group | dimension | weight |
|---|---|---:|
| faithfulness | model-fidelity — code implements the published model as stated | 20 |
| faithfulness | **specimen-match** — output resembles real specimens / authors' figures | **25** |
| faithfulness | parameter-expressivity — each morphospace parameter visibly controls its feature | 10 |
| faithfulness | morphometric-validation — measured quantities compared to specimens/literature | 15 |
| reproducibility | source-traceability — DOI/PMID/repo/data resolvable | 10 |
| reproducibility | code-traceability — code steps map to source equations/sections | 8 |
| reproducibility | repro-package — pins, seeds, documented run, sane environment | 12 |

Overall = weighted mean. Rule carried over from the siblings: **score only what
the evidence supports** — every score needs at least one code locus, a
not-applicable dimension sits at the neutral baseline (50) and is flagged, and
scripts that were not inspected are listed `unevaluated` with no scores.

## Workflow

```bash
# 1. validate an audit against the rubric (weights, ranges, evidence files, refs)
python3 morphoeval/build_morphoeval_kg.py morphoeval/audits/2026-10-13-morphology-audit.yaml --check

# 2. build the graph, check ProbKnow conformance, run a demo SPARQL ranking
python3 morphoeval/build_morphoeval_kg.py morphoeval/audits/2026-10-13-morphology-audit.yaml --conform --query

# 3. after fixing a simulation: copy the audit file to a new date, re-score the
#    affected dimensions with new line evidence, bump `checkpoint.commit`, rebuild.
#    Diff kg/<old>/REPORT.md against kg/<new>/REPORT.md to see the delta.
```

## What is in the graph (ProbKnow ontology v1.1)

No MorphoEval-specific namespace. Everything is expressed in the
[ProbKnow / Project Delphi ontology](https://github.com/fractastical/probknow-ontology)
(`pk:` = `https://w3id.org/probknow/ontology/`, individuals under
`pkr:` = `https://w3id.org/probknow/resource/`), vendored in
`vendor/probknow-1.1.ttl` for an offline conformance check (`--conform`). This
puts simulation checkpoints in the same probabilistic knowledge graph as the
Levin-lab claims, so Science Oracle queries work unchanged.

| MorphoEval notion | ProbKnow term |
|---|---|
| "script S faithfully reproduces organism O under source model P" | `pkr:claim/morphoeval-<sim>` a **`pk:Claim`** (a *belief*, not a fact) · `pk:derivedFromPaper <doi>` · `pk:context` |
| one rubric score | `pkr:assessment/<audit>-<sim>-<dim>` a **`pk:LLMClaimAssessment`** · `pk:assessedBy pkr:model/<assessor>` · `pk:assessmentType "morphoeval/0.1.0/<dim>"` · `pk:weightOfEvidence` = score/100 · `pk:hasVerdict` band · `dct:description` rationale |
| code/artefact locus behind a score | `pkr:evidence/…` a **`pk:EvidenceItem`** · `prov:wasDerivedFrom` commit-pinned GitHub `#L<a>-L<b>` (or `prov:wasGeneratedBy` the script, for rendered outputs) |
| checkpoint | `…-overall` **`pk:LLMClaimAssessment`** · `prov:wasDerivedFrom` the 7 dimension assessments + the commit · environment in description · remediation as `pk:evidenceAgainst` literals |
| the rubric | `pkr:concept/morphoeval-rubric-0.1.0` a `pk:Concept` (`skos:narrower` one `pk:Concept` per dimension) |
| source papers | `https://doi.org/…` a `pk:Paper` (real DOI IRIs as subjects) |
| the scorer | `pkr:model/claude-fable-5-1` a `pk:Model` (`prov:actedOnBehalfOf` the publisher's ORCID) |

Verdict bands on the 0–100 scale: ≥80 *faithful* · 60–79 *mostly faithful* ·
40–59 *partially faithful* · 20–39 *unfaithful* · <20 *not faithful*.

Each nanopub is one `.trig` on the nanopub temp base
`http://purl.org/nanopub/temp/np001/` (`sub:Head|assertion|provenance|pubinfo`),
the same shape as `probknow-ontology/nanopubs/src/*.trig`, with pubinfo
`dct:creator <ORCID>` · CC BY 4.0 · `npx:hasNanopubType` · `npx:introduces` ·
`pk:domain pkr:domain/<slug>` (`morphogenesis`; `levin-lab` for the planarian
script). Signing with `nanopub-*.jar sign` (or `@nanopub/nanopub-js`) mints the
trusty `https://w3id.org/np/RA…` URI. Scripts marked `unevaluated` produce no
nanopubs (nothing has been assessed). `morphoeval_<audit-id>.ttl` is the flat
merge of all assertion graphs for local SPARQL.

Publishing: sign → **test registry first** → production only after review.
Published nanopubs are immutable; a re-score is a new assessment nanopub, not an
edit.

## Current checkpoint (2026-10-13, commit f3fe478)

Tracked in [morpholib#1](https://github.com/fractastical/morpholib/issues/1).

| simulation | faithfulness | reproducibility | overall |
|---|---:|---:|---:|
| PlanformDB planarian animation | 5.4 | 36.3 | **14.7** |
| Raup shell coiling | 28.6 | 57.7 | **37.3** |
| Gray–Scott (Turing) | 54.3 | 49.0 | **52.7** |
| Chindapol coral flow | 49.3 | 70.0 | **55.5** |
| Hoffmann–Donoughe wing venation | 81.1 | 70.0 | **77.8** |
| Petroff stromatolite / Kahng coral / Cervera–Levin–Mafe | — | — | *unevaluated* |
