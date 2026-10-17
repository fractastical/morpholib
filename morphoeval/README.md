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
  vendor/templates/*.trig          the ProbKnow Space's governed template nanopubs (shape check)
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

Since 2026-10-15 every nanopub is shaped by the **governed templates of the
ProbKnow Space** (https://w3id.org/spaces/probknow) and links them from pubinfo
(`nt:wasCreatedFromTemplate` / `…ProvenanceTemplate` / `…PubinfoTemplate`), so
Nanodash renders each one with its form and the Space counts it as conformant.
The template nanopubs are vendored in `vendor/templates/` and `--conform` checks
every assertion, provenance and pubinfo graph against the statements of the
template it links (placeholder prefixes, datatypes, the 0–1 regex on
probabilities, required/repeatable statements). Earlier nanopubs (2026-10-13 /
2026-10-14) used a freer shape and are being re-issued with `npx:supersedes`.

| MorphoEval notion | ProbKnow term (template) |
|---|---|
| "script S faithfully reproduces organism O under source model P" | `pkr:claim/morphoeval-<sim>` a **`pk:Claim`** (claim template): `rdfs:label`, `dct:description`, `pk:context` (scripts, output, organism, commit, environment, non-DOI sources) · `pk:derivedFromPaper <doi>` · `pk:hasAssessment` → the 8 assessments · `pk:weightOfEvidence` = overall/100 |
| one rubric score | `pkr:assessment/<audit>-<sim>-<dim>` a **`pk:LLMClaimAssessment`** (LLM-assessment template) · `pk:evaluatesHypothesis pkr:hypothesis/morphoeval-<sim>-<dim>` ("<sim> satisfies the criterion <dim>", `dct:identifier "morphoeval/0.1.0/<dim>"`) · `pk:assessedBy pkr:model/<assessor>` a `pk:Model` · `pk:hasProbability` = score/100 · `pk:hasVerdict` band · `rdfs:comment` = rationale + evidence loci (commit-pinned GitHub `#L<a>-L<b>`, sha256-fingerprinted outputs, morphometrics with their reference values) · `pk:evidenceFor` / `pk:evidenceAgainst` when an evidence entry is marked `supports: true|false` |
| checkpoint | `…-overall`, the same template, evaluating `pkr:hypothesis/morphoeval-<sim>` (the claim proposition); remediation items as `pk:evidenceAgainst` |
| the rubric | `pkr:concept/morphoeval-rubric-0.1.0` a `pk:Concept` (`skos:narrower` one `pk:Concept` per dimension); no governed template exists for a rubric, so it carries only the provenance and pubinfo templates |
| provenance | "Extracted from a paper by ProbKnow" template: `prov:wasAttributedTo` the model and the publisher, `prov:wasDerivedFrom` each DOI with `dct:title` / `dct:date`; the rubric uses the generic "Attributed to" template |
| the scorer | `pkr:model/claude-fable-5-1` a `pk:Model` (`prov:actedOnBehalfOf` the publisher's ORCID, in pubinfo) |

The templates carry evidence as literals, so the code loci, output fingerprints
and morphometrics that used to be separate `pk:EvidenceItem` / `schema:` nodes
are now written into the assessment's `rdfs:comment` (and the REPORT.md); the
audit YAML and measurements sidecars are unchanged.

Verdict bands on the 0–100 scale: ≥80 *faithful* · 60–79 *mostly faithful* ·
40–59 *partially faithful* · 20–39 *unfaithful* · <20 *not faithful*.

Each nanopub is one `.trig` on the nanopub temp base
`http://purl.org/nanopub/temp/np001/` (`sub:Head|assertion|provenance|pubinfo`),
with pubinfo `rdfs:label` · `dct:created` (full UTC timestamp; set
`MORPHOEVAL_CREATED` to pin it for a reproducible rebuild) · `dct:creator <ORCID>`
· CC BY 4.0 · `npx:hasNanopubType` · `npx:introduces` · `pk:domain
pkr:domain/<slug>` (`morphogenesis`; `levin-lab` for the planarian script) · the
template links. Before a rebuilt nanopub supersedes a published URI the builder
resolves that URI's current head with Nanopub Query's `get-latest-version-of-np`
(`--no-resolve-heads` skips it), so version chains never fork. Signing mints the
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

Signing is deterministic for a given build (the `dct:created` stamp is in the
file), so sign-only, test and prod yield the same `https://w3id.org/np/RA…`
URIs: what was verified on the test registry is byte-for-byte what goes to
production. `published.json` records each nanopub's URI and furthest stage
(`null` / `test` / `prod`); the signer refuses to re-sign a file whose content
no longer matches its recorded URI, and after each POST it fetches the nanopub
back from the registry it posted to. Published nanopubs are immutable; a
re-score is a new assessment nanopub (`npx:supersedes` the old one), not an edit.

Tooling notes: nanopub-py 2.1.0 on Python 3.14 cannot read `profile.yml`
(the signer builds the `Profile` directly) and its `has_valid_trusty` verifier
is broken (it rejects nanopub-java-signed published nanopubs too), which is why
verification goes through `@nanopub/sign`. `nanopubs/np_utils.py check` is
broken against nanopub-py 2.1 for the same reason. nanopub-py's `publish()` is
not used either: it re-serialises through rdflib, which turns the plain-decimal
`xsd:double` literals the hash was computed over into exponent form, so the
signer posts the signed text verbatim. The registry URL is chosen explicitly in
the signer (nanopub-py only switches to the test registry on its own copy of
the config; reading `conf.use_server` from the outside posts to production,
which is how the 10 Limbform nanopubs of 2026-10-15 reached production
directly — content already verified, but the test step was skipped).

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

**Odonata audit (2026-10-14):** `2026-eshghi-odonata-wing-cells.py` — a data
comparison rather than a generative model, so `parameter-expressivity` is
marked `not_applicable` (neutral 50) — audited as `2026-10-14-odonata-audit`
at commit `fbbcaf6`: overall 76.0. Its 9 nanopubs are on production; the
rubric nanopub was again reused, not re-issued.

**Template conformance (2026-10-15):** following Tobias Kuhn's review on
[project-delphi-ontology#1](https://github.com/fractastical/project-delphi-ontology/issues/1)
(the 64 nanopubs above used declared terms but linked no governed template and
carried `00:00:00` timestamps), the builder now emits the ProbKnow Space's
template shapes with template links and full timestamps (see above). The first
audit published this way is **Limbform** (`2014-lobo-limbform-morphology.py`,
`2026-10-14-limbform-audit` at commit `747c57f`): overall 71.3, 10 nanopubs on
production, and its rubric nanopub
(https://w3id.org/np/RAhBMChdpmgS_yBfy8cfYXpfKktqlUD72A9bxZzvWDkYM) supersedes
the original rubric (whose text is unchanged; only the shape and timestamp
differ). The second is the **C. elegans** audit
(`2015-santella-wormguides-celegans.py`, `2026-10-15-celegans-audit` at commit
`2864784`): overall 67.7, 9 nanopubs (rubric reused), published to the test
registry first with the fixed signer, verified with `check_signed.mjs` and
fetched back from both registries before going to production; overall claim
https://w3id.org/np/RASUT7taw502ymt3A17lN_bXMn4rC1HY3FNs1ai8YkZSQ. It was
re-scored the next day (`2026-10-16-celegans-audit` at commit `303a87e`,
overall 70.2) after the script's calibration was moved from the nuclear cloud
to the atlas's own eggshell mesh; only model-fidelity (75 → 80), specimen-match
(60 → 62) and morphometric-validation (65 → 72) changed, the other four
dimensions keep their rationales with updated line numbers. The 9 new nanopubs
`npx:supersedes` the 2026-10-15 ones (built with `--supersede-from
morphoeval/kg/2026-10-15-celegans-audit --supersede-from
morphoeval/kg/2026-10-14-limbform-audit`, so the shared rubric is not
re-emitted); overall claim
https://w3id.org/np/RAqDf8b5w5WNe3Chr8OWp5uUyU8U1Bz5BrbGdamyd4BOU.

The third is the **Thompson** audit
(`thompson/1917-thompson-scarus-pomacanthus.py`, `2026-10-16-thompson-audit`
at commit `3cf7f8c`): overall 70.2, 9 nanopubs (rubric reused via
`--supersede-from morphoeval/kg/2026-10-14-limbform-audit`), test registry →
`check_signed.mjs` 9/9 → fetched back 9/9 → production; overall claim
https://w3id.org/np/RAjJx_KHiAAaPSxjsoYOJaFSU1JYmDUswVaRLmka2NIM0. It is the
first audit of a data analysis that tests a historical claim rather than of a
model that generates form: Thompson's *Scarus* → *Pomacanthus* "system of
coaxial circles" (1917, p. 749) is read as a Möbius map, fitted by least
squares to 17 homologous landmarks on 22 *Scarus* and 9 *Pomacanthus* species
(Claverie & Wainwright 2014 via Friedman et al. 2019, CC0, vendored with a
SHA-256 pin), and found to explain 1 % (isogonal) to 32 % (oblique) of the
Procrustes distance, against 31 % for a plain affine map, with a residual twice
the within-genus spread. The rubric is read as for the other data-analysis
audits (the "model" is Thompson's family of grids, the "output" the fish it
carries); parameter expressivity is scored as applicable (55) because the family
has constants whose effect is shown, unlike the decoders where it is neutral.
Building this audit exposed a builder bug: `plain_doubles()`, which rewrites
rdflib's exponent-form doubles to plain decimals, also rewrote `1e-17` inside
a description string; it and the matching conformance check now skip string
literals (no previously built checkpoint changes).

**Re-issue wave (2026-10-16):** the three earlier audits were rebuilt with
`--supersede` (`2026-10-14-morphology-audit` and `2026-10-14-odonata-audit`
also with `--supersede-from morphoeval/kg/2026-10-14-limbform-audit`, so the
rubric is recognised as already re-issued and not emitted again). Of the 64
pre-template nanopubs, 10 were no longer the heads of their chains (the
original rubric, superseded by RAhBMChd…, and the 9 planarian-animation
nanopubs of 2026-10-13, superseded by the 2026-10-14 re-score); the builder
skipped those and removed their stale unsigned files. The other 54 were
re-emitted in template shape with scores, evidence and measurements unchanged,
each `npx:supersedes` its predecessor (heads confirmed on Nanopub Query first),
published to the test registry, verified (`check_signed.mjs` 64/64 incl. the
retained files), fetched back, then published to production; Nanopub Query now
resolves every old URI to its re-issue. `published.json` keeps the old URI
under `history`. The signer's production gate now checks only the files it is
about to post (manifest entries whose file was superseded elsewhere no longer
block it).

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
| Odonata wing cells, fossil vs extant (`2026-eshghi-odonata-wing-cells.py`) | fbbcaf6, audit 2026-10-14-odonata | 75.7 | 76.7 | **76.0** |
| LimbformDB limb morphologies (`2014-lobo-limbform-morphology.py`) | 747c57f, audit 2026-10-14-limbform | 70.0 | 74.3 | **71.3** |
| WormGUIDES C. elegans embryo atlas + connectome (`2015-santella-wormguides-celegans.py`) | 303a87e, audit 2026-10-16-celegans (re-score of 2026-10-15: 67.7) | 67.6 | 76.3 | **70.2** |
| Thompson Scarus -> Pomacanthus on real landmarks (`thompson/1917-thompson-scarus-pomacanthus.py`) | 3cf7f8c, audit 2026-10-16-thompson | 65.6 | 80.8 | **70.2** |
| Petroff stromatolite / Kahng coral / Cervera–Levin–Mafe | — | — | — | *unevaluated* |
