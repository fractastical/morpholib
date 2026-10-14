# MorphoEval — faithfulness checkpoints for the morphology simulations

Sibling of **BioEval** (Bioinformatics Paper Evaluator, Zenodo concept DOI
[10.5281/zenodo.20567719](https://doi.org/10.5281/zenodo.20567719)) and
**NeuroEval** (Zenodo concept DOI
[10.5281/zenodo.20622878](https://doi.org/10.5281/zenodo.20622878)). Those score *papers* on a
versioned, weighted rubric; MorphoEval scores *our own simulation scripts* for
faithfulness to the organism and to the source model, and records each scoring
as a knowledge graph so a simulation can be **checkpointed and re-scored** after
every change.

```
morphoeval/
  rubric.yaml                      versioned rubric (7 weighted dimensions, 0-100)
  audits/<date>-<name>.yaml        one audit = scores + evidence + references + checkpoint
  measure_outputs.py               morphology snapshot: sha256 of rendered outputs + morphometrics
                                   -> audits/<audit>.measurements.yaml (merged into the nanopubs)
  build_morphoeval_kg.py           audit -> ProbKnow nanopubs (.trig) + merged .ttl + REPORT.md
  sign_morphoeval.py               sign / publish (test -> prod) -> kg/<audit-id>/signed/ + published.json
  check_signed.mjs                 independent trusty+signature verification (@nanopub/sign, Rust/WASM)
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

# 2. snapshot the morphology: fingerprint the rendered outputs and measure them
#    (re-runs the wing script; ~2 min). Outputs not on disk are reported, never invented.
python3 morphoeval/measure_outputs.py morphoeval/audits/2026-10-13-morphology-audit.yaml

# 3. build the graph, check ProbKnow conformance, run a demo SPARQL ranking
#    (--supersede: changed nanopubs get npx:supersedes <URI in kg/<audit>/published.json>)
python3 morphoeval/build_morphoeval_kg.py morphoeval/audits/2026-10-13-morphology-audit.yaml --conform --query --supersede

# 4. after fixing a simulation: copy the audit file to a new date, re-score the
#    affected dimensions with new line evidence, bump `checkpoint.commit`, and
#    build with --supersede-from <previous kg dir>: same-named nanopubs get
#    npx:supersedes <the URI published from that dir>; nanopubs whose assertion
#    and provenance are unchanged (e.g. the rubric) are not re-emitted.
#    Diff kg/<old>/REPORT.md against kg/<new>/REPORT.md to see the delta.
python3 morphoeval/build_morphoeval_kg.py morphoeval/audits/2026-10-14-morphology-audit.yaml \
    --conform --query --supersede-from morphoeval/kg/2026-10-13-morphology-audit
```

## What is in the graph (ProbKnow ontology v1.1)

No MorphoEval-specific namespace. Everything is expressed in the
[ProbKnow / Project Delphi ontology](https://github.com/fractastical/probknow-ontology)
(`pk:` = `https://w3id.org/probknow/ontology/`, individuals under
`pkr:` = `https://w3id.org/probknow/resource/`), vendored in
`vendor/probknow-1.1.ttl` for an offline conformance check (`--conform`). This
puts simulation checkpoints in the same probabilistic knowledge graph as the
Levin-lab claims, so Science Oracle queries work unchanged.

The IRI bases are not hard-wired: if the ontology project renames (e.g. to a
ScienceOracle namespace), set in the audit YAML

```yaml
audit:
  ontology:
    base: https://w3id.org/scienceoracle/ontology/
    resource_base: https://w3id.org/scienceoracle/resource/
```

re-vendor the renamed `.ttl`, and rebuild; `--conform` compares terms by local
name so it works across the rename. Until the rename exists upstream the
default (`probknow`) is the only spec the output can conform to.

Domains: `pk:domain pkr:domain/<slug>` + `rdfs:label` in pubinfo is all a
domain is (no registry); `claims-by-domain.rq` and the Oracle's domain list
simply count them, so `morphogenesis` appears as a new bucket once published.

| MorphoEval notion | ProbKnow term |
|---|---|
| "script S faithfully reproduces organism O under source model P" | `pkr:claim/morphoeval-<sim>` a **`pk:Claim`** (a *belief*, not a fact) · `pk:derivedFromPaper <doi>` · `pk:context` |
| one rubric score | `pkr:assessment/<audit>-<sim>-<dim>` a **`pk:LLMClaimAssessment`** · `pk:assessedBy pkr:model/<assessor>` · `pk:assessmentType "morphoeval/0.1.0/<dim>"` · `pk:weightOfEvidence` = score/100 · `pk:hasVerdict` band · `dct:description` rationale |
| the code under assessment | `<repo>/blob/<commit>/<script>` a **`schema:SoftwareSourceCode`**, `prov:Entity` · `dct:creator` / `schema:author` / `prov:wasAttributedTo` the code's author(s) (`audit.code_creators`, per-sim `code_creators` / `code_contributors`) · `schema:version` = commit · linked from the claim by `dct:subject` |
| code/artefact locus behind a score | `pkr:evidence/…` a **`pk:EvidenceItem`** · `prov:wasDerivedFrom` commit-pinned GitHub `#L<a>-L<b>` (or the fingerprinted output, for rendered artefacts) |
| the rendered output that was judged | `pkr:evidence/<audit>-<sim>-output-<file>` a **`schema:ImageObject`**, `pk:EvidenceItem` · `schema:sha256` · `schema:contentSize` · `schema:encodingFormat` · width/height · `prov:wasDerivedFrom` the script. The binary stays out of git and out of the nanopub; the hash pins it |
| the form itself (morphometrics) | `pkr:evidence/<audit>-<sim>-morphometric-m<i>` a **`schema:PropertyValue`**, `pk:EvidenceItem` on the `morphometric-validation` assessment · `schema:name` / `schema:value` / `schema:unitText` / `schema:measurementTechnique` · `schema:valueReference` → reference `PropertyValue` with `dct:source` (only where a reference is actually known) |
| checkpoint | `…-overall` **`pk:LLMClaimAssessment`** · `prov:wasDerivedFrom` the 7 dimension assessments + the commit · environment in description · remediation as `pk:evidenceAgainst` literals |
| the rubric | `pkr:concept/morphoeval-rubric-0.1.0` a `pk:Concept` (`skos:narrower` one `pk:Concept` per dimension) |
| source papers | `https://doi.org/…` a `pk:Paper` (real DOI IRIs as subjects) |
| the scorer | `pkr:model/claude-fable-5-1` a `pk:Model` (`prov:actedOnBehalfOf` the publisher's ORCID) |

Three distinct roles are stamped and must not be conflated: the **code
creator** (`dct:creator` on the script entity), the **assessor** (`pk:assessedBy`
a `pk:Model`), and the **nanopub publisher** (`dct:creator` in pubinfo, the
ORCID the model `prov:actedOnBehalfOf`). `--conform` rejects a claim whose
script has no creator stamp.

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

## Signing and publishing

```bash
# sign with ~/.nanopub/profile.yml -> kg/<audit-id>/signed/*.trig + kg/<audit-id>/published.json
python3 morphoeval/sign_morphoeval.py morphoeval/kg/2026-10-13-morphology-audit
# verify independently (Rust/WASM nanopub toolkit; `npm install @nanopub/sign` once, outside the repo
# or anywhere above it; or set NANOPUB_SIGN_DIR)
node morphoeval/check_signed.mjs morphoeval/kg/2026-10-13-morphology-audit/signed/*.trig
# publish: test registry first, production only after review (the tool refuses prod before test)
python3 morphoeval/sign_morphoeval.py morphoeval/kg/2026-10-13-morphology-audit --publish test
python3 morphoeval/sign_morphoeval.py morphoeval/kg/2026-10-13-morphology-audit --publish prod
```

Signing is deterministic (fixed `dct:created`), so sign-only, test and prod
yield the same `https://w3id.org/np/RA…` URIs: what was verified on the test
registry is byte-for-byte what goes to production. `published.json` records
each nanopub's URI and furthest stage (`null` / `test` / `prod`); the signer
refuses to re-sign a file whose content no longer matches its recorded URI.
Published nanopubs are immutable; a re-score is a new assessment nanopub
(`npx:supersedes` the old one), not an edit.

Tooling notes: nanopub-py 2.1.0 on Python 3.14 cannot read `profile.yml`
(the signer builds the `Profile` directly) and its `has_valid_trusty` verifier
is broken (it rejects nanopub-java-signed published nanopubs too), which is why
verification goes through `@nanopub/sign`. `nanopubs/np_utils.py check` is
broken against nanopub-py 2.1 for the same reason.

**Status (2026-10-13):** all 46 nanopubs of audit `2026-10-13-morphology-audit`
are **published on the production nanopub network** (`https://w3id.org/np/<RA…>`,
URIs in `kg/<audit>/published.json`), after the same bytes were verified on the
test registry. Earlier test-only iterations (first versions without the
morphology snapshot, and their superseding second versions) are recorded in
`kg/<audit>/published.test-scratch.json`; they exist only on the test registry.
Rubric: https://w3id.org/np/RAjd9dIu9uNVgvT4npxJgWwL6cbp8GCsy3gK86Qkj-iH4

**Re-score (2026-10-14):** the planarian simulation was rewritten
(`2013-lobo-planform-morphology.py`, decoding the Planform region graph
exactly) and re-audited as `2026-10-14-morphology-audit` at commit `161f269`.
Its 9 new nanopubs (claim, 7 dimensions, overall) are on production and each
carries `npx:supersedes` pointing at the 2026-10-13 nanopub it replaces; the
rubric nanopub was unchanged and was not re-issued. The other simulations were
not touched and their 2026-10-13 nanopubs stand. Both audits' measurements
sidecars (`audits/*.measurements.yaml`) pin the rendered outputs by sha256.

## Current checkpoint

Tracked in [morpholib#1](https://github.com/fractastical/morpholib/issues/1).

| simulation | checkpoint | faithfulness | reproducibility | overall |
|---|---|---:|---:|---:|
| PlanformDB planarian morphology (`2013-lobo-planform-morphology.py`) | 161f269, audit 2026-10-14 | 66.1 | 67.3 | **66.5** |
| ~~PlanformDB planarian animation (`1900-planformDB_parser.py`)~~ superseded | f3fe478, audit 2026-10-13 | 5.4 | 36.3 | ~~14.7~~ |
| Raup shell coiling | f3fe478, audit 2026-10-13 | 28.6 | 57.7 | **37.3** |
| Gray–Scott (Turing) | f3fe478, audit 2026-10-13 | 54.3 | 49.0 | **52.7** |
| Chindapol coral flow | f3fe478, audit 2026-10-13 | 49.3 | 70.0 | **55.5** |
| Hoffmann–Donoughe wing venation | f3fe478, audit 2026-10-13 | 81.1 | 70.0 | **77.8** |
| Petroff stromatolite / Kahng coral / Cervera–Levin–Mafe | — | — | — | *unevaluated* |
