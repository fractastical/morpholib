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

| group | sub-axis | dimension | weight |
|---|---|---|---:|
| faithfulness | model-implementation (30) | model-fidelity — code implements the published model as stated | 20 |
| faithfulness | biological-validation (40) | **specimen-match** — output resembles real specimens / authors' figures | **25** |
| faithfulness | model-implementation (30) | parameter-expressivity — each morphospace parameter visibly controls its feature | 10 |
| faithfulness | biological-validation (40) | morphometric-validation — measured quantities compared to specimens/literature | 15 |
| reproducibility | reproducibility (30) | source-traceability — DOI/PMID/repo/data resolvable | 10 |
| reproducibility | reproducibility (30) | code-traceability — code steps map to source equations/sections | 8 |
| reproducibility | reproducibility (30) | repro-package — pins, seeds, documented run, sane environment | 12 |

To avoid conflating faithful code with valid biology, `faithfulness` (weight 70) is decomposed and reported along two separate sub-axes: **model-implementation fidelity** (`impl.`, weight 30 = `model-fidelity` 20 + `parameter-expressivity` 10) and **biological/empirical validation** (`bio.`, weight 40 = `specimen-match` 25 + `morphometric-validation` 15), alongside **reproducibility** (`repro.`, weight 30).

**Score semantics (`[0, 100]` uncalibrated ratings vs probabilities):** MorphoEval scores are uncalibrated, criterion-referenced ratings on `0–100`, not probabilities ($s = 76$ does not mean $P(\text{faithful}) = 0.76$). When serialized into the governed ProbKnow Space templates (`RAku3DIN` and `RAIVOEuh`), `score / 100` is written to `pk:hasProbability` and `overall / 100` to `pk:weightOfEvidence` strictly as a unit-interval rescaling for template conformance (or omitted via `--omit-probability`).

**Non-applicable (`not_applicable`) dimensions and baseline-50 shrinkage:** Rule carried over from the siblings: **score only what the evidence supports** — every score needs at least one code locus, and scripts that were not inspected are listed `unevaluated` with no scores. When a dimension is structurally inapplicable (`not_applicable: true`), substituting `50` into the 100-weight sum produces $\bar{s}_{\text{pad}} = (1 - \alpha)\bar{s}_{\text{app}} + 50\alpha$, where $\alpha = w_{\text{NA}} / 100$. Mathematically, substituting `50` is **not** neutral: it shrinks scores toward `50` (penalising scripts with $\bar{s}_{\text{app}} > 50$ and rewarding scripts with $\bar{s}_{\text{app}} < 50$). Only weight renormalisation over applicable dimensions ($\bar{s}_{\text{app}}$) is mathematically neutral; MorphoEval therefore reports **both** the applicable-only renormalised score (`app.`) and the baseline-50 padded score (`pad.`).

## The cycle

The forms in question are the shapes of bodies — how they arise in development and come back in regeneration — and most of the recent papers here share one idea about it: that beside genes and chemical morphogens, a tissue stores its pattern bioelectrically, as a map of membrane voltages shared through gap junctions, so that changing the voltage changes what regrows (the two-headed planarian of the Planform database and the Cervera–Levin–Mafe model). The top-level README opens with a short synopsis. Tier A is how such a form gets into the repository: it starts with a person scouting — choosing a
paper, dataset or claim worth the work and pointing the agent at it — and ends in one of three kinds of script
(reproduce the paper's model from its equations, decode the authors' formal encoding into form, or test a
published claim against measurements). Between the tiers sits what the scripts produce — box 6, the
fifteen animations, each labelled with its paper — which is what tier B scores; a person reads the scores
and decides what to fix or where to go next; the builder agent then re-works the script — its fidelity to
the paper, its sources, its checks — and the fix runs back through box 6 into the script (👤 marks the two
places people decide). Two agent roles are meant to be distinct: the **builder** that writes and re-works the
code and the **assessor** that scores it. For the first nine audits the same model filled both (`claude-fable-5-1`, Anthropic); on 2026-10-17 the Cervera–Levin–Mafe script was
scored across two rounds by a model from another company (`gpt-5.6-sol`, OpenAI), given the rubric, the script and the
fixtures and no opinions, and it found defects the builder had not — an unused morphospace axis, a CSV
that doubles the simulated displacement, a wrong PMID (44.9), and after remediation (55.6) two further threshold/legend bugs the builder had
introduced and overlooked. In addition, a complete 10-script second-assessor pass (`2026-10-17-blinded-sample-audit.yaml`, 80 signed nanopubs) was scored by `gemini-4-argon` (Google) across all 10 active evaluated scripts so that **no script relies solely on its builder grading itself**, showing strong rank-order concordance on overall scores across all 10 scripts (Spearman $\rho = 0.891$, Kendall $\tau = 0.778$, Pearson $r = 0.920$ on padded scores; $\rho = 0.903$ on applicable-only scores; and exact $\rho = 1.00$, $\tau = 1.00$ on the initial 5-script strictly blinded subset; dimension-level $\rho = 0.796$, $r = 0.813$ across all 70 pairs, $\text{MAD} = 10.4$ points) while surfacing concrete defects the builder had missed (e.g. a 65-character SHA-256 typo in `2014-lobo-limbform-morphology.py`). The graph takes any assessor's nanopubs on the same
hypotheses; further remediation is tracked in
[morpholib#1](https://github.com/fractastical/morpholib/issues/1). Each pass pins a commit; scores are 0–100 uncalibrated ratings per dimension with a weighted mean overall; published
nanopubs are immutable and re-scores supersede them. The same figure as an image (for the PDF report
and anything that cannot render Mermaid) is `docs/media/morphoeval_pipeline.png`, drawn by
`morphoeval/report/draw_pipeline.py`; in it box 6 is the reproductions' own animations in
`docs/media/`, each labelled with its paper (author, year, venue, data source). `draw_pipeline.py --gif` writes `morphoeval_pipeline.gif`, the same figure with
the thumbnails playing (used at the top of the root README); the PDF report uses the still ([`morphoeval_pipeline.png`](../docs/media/morphoeval_pipeline.png)).

![From paper to reproduction to signed evaluation — the MorphoEval pipeline](../docs/media/morphoeval_pipeline.gif)

```mermaid
flowchart TB
    classDef build fill:#dbeafe,stroke:#334155,color:#1e293b
    classDef eval fill:#fde68a,stroke:#334155,color:#1e293b
    classDef pub fill:#bbf7d0,stroke:#334155,color:#1e293b
    classDef loop fill:#fecaca,stroke:#334155,color:#1e293b

    subgraph A["A &nbsp;Making a reproduction — reproduce the paper's model · decode the authors' encoding · test a published claim"]
        direction LR
        SC["👤 1 Scout<br/>a person finds a model of form worth the work —<br/>a classic paper, an open dataset, an untested claim,<br/>a score that is too low — and points the agent at it"]:::build
        P["2 Paper<br/>the model as the authors published it: a set of<br/>equations, a database of drawings, or a claim"]:::build
        S["3 Formal source<br/>the paper's own numbers and records, fetched or<br/>copied in and locked to a fingerprint (SHA-256),<br/>so a changed file is refused"]:::build
        R["4 Pin the rules<br/>prose → exact rule, tested on every record;<br/>the authors' code, where it exists, is read to<br/>settle what they meant, never copied"]:::build
        X["🤖 builder · 5 Script<br/>one program per paper: reproduces, decodes or tests,<br/>then draws, measures and checks itself<br/><small>&lt;year&gt;-&lt;author&gt;-&lt;model&gt;.py</small>"]:::build
        SC --> P --> S --> R --> X
    end

    O["6 Output &amp; compare<br/>what each script produces — its own figure and animation<br/>(the fifteen in docs/media/) — set beside the authors' figures<br/>or real specimens; distinct forms must come out distinct.<br/>This is what gets scored."]:::build
    X -- writes --> O

    subgraph B["B &nbsp;Auditing it — the MorphoEval cycle"]
        direction LR
        AU["🤖🤖 assessors · 7 Dual-agent audit<br/>two assessor agents from different companies —<br/>not just the builder scoring itself — independently score<br/>7 questions (0–100) with line-pinned evidence at one commit<br/><small>audits/&lt;date&gt;-&lt;name&gt;.yaml</small>"]:::eval
        M["8 Measure<br/>outputs hashed and measured independently<br/>against specimens or the paper:<br/>numbers, not opinions<br/><small>measure_outputs.py</small>"]:::eval
        BG["9 Write as claims<br/>each score becomes one small machine-readable<br/>statement: 'this script reproduces X' — what rubric score,<br/>says who, on what evidence, under what licence<br/><small>build_morphoeval_kg.py (RDF, ProbKnow templates)</small>"]:::pub
        PU["10 Sign &amp; publish<br/>each statement is signed and put on a public<br/>network of servers under a permanent address;<br/>test network first, then for real<br/><small>sign_morphoeval.py</small>"]:::pub
        G[("11 Public record<br/>anyone can look a score up, cite it, check its<br/>signature or query across all audits<br/><small>Nanopub Query · kg/&lt;audit&gt;/REPORT.md</small>")]:::pub
        T["👤 12 Remediate<br/>what the audit found wrong becomes a to-do list;<br/>a person picks, the builder agent re-works the<br/>script, and the touched scores are judged again<br/><small>GitHub issue #1</small>"]:::loop
        AU --> M --> BG --> PU --> G --> T
    end

    O -- scored, with evidence --> AU
    T -- "👤 a person picks what to fix; 🤖 the builder re-works the script<br/>(fidelity, sources, checks) → outputs re-made → 🤖 the assessor judges the touched scores again and published anew;<br/>the old statements stay on record, marked as replaced (nothing published is edited)" --> X
    G -. other assessors (models, people) may publish their own<br/>assessment of the same hypotheses .-> BG
```

*What a nanopublication is:* the smallest publishable unit of science — one statement (here "this script's reproduction of Raup 1966 scores 64.1/100 on the MorphoEval rubric (mostly faithful)") packaged with who asserted it, when, on what evidence and under what licence, cryptographically signed, and stored on a public network of servers under a permanent address anyone can cite or fetch. Once published it cannot be edited; a re-score publishes a new statement that points at the one it replaces, so the history stays visible, and other assessors can publish their own assessment of the same hypothesis next to ours.

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
| "script S faithfully reproduces organism O under source model P" | `pkr:claim/morphoeval-<sim>` a **`pk:Claim`** (claim template): `rdfs:label`, `dct:description`, `pk:context` (scripts, output, organism, commit, environment, non-DOI sources) · `pk:derivedFromPaper <doi>` · `pk:hasAssessment` → the 8 assessments · `pk:weightOfEvidence` = overall/100 (unit-interval rescaled uncalibrated rubric score, optional via `--omit-probability`) |
| one rubric score | `pkr:assessment/<audit>-<sim>-<dim>` a **`pk:LLMClaimAssessment`** (LLM-assessment template) · `pk:evaluatesHypothesis pkr:hypothesis/morphoeval-<sim>-<dim>` ("<sim> satisfies the criterion <dim>", `dct:identifier "morphoeval/0.1.0/<dim>"`) · `pk:assessedBy pkr:model/<assessor>` a `pk:Model` · `pk:hasProbability` = score/100 (unit-interval rescaled uncalibrated rubric score, not $P(\text{faithful})$; optional via `--omit-probability`) · `pk:hasVerdict` band · `rdfs:comment` = rationale + evidence loci (commit-pinned GitHub `#L<a>-L<b>`, sha256-fingerprinted outputs, morphometrics with their reference values) · `pk:evidenceFor` / `pk:evidenceAgainst` when an evidence entry is marked `supports: true|false` |
| checkpoint | `…-overall`, the same template, evaluating `pkr:hypothesis/morphoeval-<sim>` (the claim proposition); remediation items as `pk:evidenceAgainst` |
| the rubric | `pkr:concept/morphoeval-rubric-0.1.0` a `pk:Concept` (`skos:narrower` one `pk:Concept` per dimension); no governed template exists for a rubric, so it carries only the provenance and pubinfo templates |
| provenance | "Extracted from a paper by ProbKnow" template: `prov:wasAttributedTo` the model and the publisher, `prov:wasDerivedFrom` each DOI with `dct:title` / `dct:date`; the rubric uses the generic "Attributed to" template |
| the scorer | `pkr:model/<assessor-id>` (`claude-fable-5-1`, `gpt-5.6-sol`, or `gemini-4-argon`) a `pk:Model` (`prov:actedOnBehalfOf` the publisher's ORCID, in pubinfo) |

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

**Thompson re-score (2026-10-17-thompson, 70.2 → 76.6):** the audit's second
remediation item — test the outline Thompson deformed, not 17 points — was
done at commits `9852075`/`232e8f8`. The script fetches two CC-licensed
Wikimedia Commons photographs (a collected *Scarus ghobban*, CC BY 4.0; a
living *Pomacanthus semicirculatus*, CC BY 2.0; SHA-256-pinned, cached, not
vendored), segments their silhouettes, registers each to its species' landmarks
by two anchors the silhouette gives, fits the five families to the species'
landmarks and carries the parrotfish silhouette with each. No landmark-fitted
deformation beats the no-deformation baseline on the outline (IoU 0.662 vs
affine 0.652, Möbius 0.645, oblique coaxial 0.618, thin-plate spline 0.629);
the sidecar re-derives the IoU with a different rasteriser (agreement 0.0001).
Model fidelity 72 → 78, specimen match 55 → 72, morphometric validation
82 → 86, source traceability 90 → 92, code traceability 78 → 80; parameter
expressivity and repro package unchanged. Built with `--supersede-from` the
2026-10-16 Thompson, Limbform and Raup dirs (rubric reused), test registry →
`check_signed.mjs` 9/9 → fetched back → production; overall claim
https://w3id.org/np/RAYFKAfyOnqIKXWzdEzCN_3zyX2KX_ijCXNX-ZVtBPeP8.

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

**Cross-cutting remediation (2026-10-17):** three items that every audit's
repro-package or source-traceability rationale had carried were closed in the
working tree (they will show up in the scores only when the scripts are next
re-audited; published nanopubs are not edited): `requirements.txt` now lists
`scikit-image`, `requests` and the morphoeval/nanopub dependencies, and
`requirements.lock.txt` pins the exact versions of the audit environment
(Python 3.14.7; verified to install into a fresh venv and to reproduce the
Thompson numbers); `2013-lobo-planform-morphology.py` records the SHA-256 and
size of the `planformDB_2.5.0.edb` it was verified on
(`datasets/planform/planformDB.sha256`) and checks the file it is given, with
`--strict-db` to refuse any other; and the Odonata audit's "report #7 and
#23/#24 to the authors" item has a draft message in
`datasets/odonata/NOTE_TO_AUTHORS.md`, to be sent by the repository owner.

**Raup re-score (2026-10-17-raup, 37.3 → 64.1):** the first audit's
remediation item ("make the translation scale with the whorls") was done at
commit `ba6fd20`, and doing it exposed two further departures from Raup's
model: the generating curve was drawn in the horizontal plane (every shell a
zero-thickness ribbon) and the axis offset made the rendered inner/outer
margin ratio (1+D)/(3−D) rather than D. The cause sat in the formal source:
`datasets/raup/variables.json` had been written in 2025 without the paper,
and its definition of T was not Raup's. The script now implements the
definitions as restated with equations by Noshita (2014) and Okabe &
Yoshimura (2017), says so, recovers W, D and T from its own rendered mesh
(`recover_parameters`, to 5e-15) and refuses to draw unless they match and
the spire grows monotonically with T; the six per-taxon parameter sets were
replaced by illustrative points labelled "cf." the shells they resemble, with
the old values kept under `empirical_shells_provenance`. Nine nanopubs
supersede the 2026-10-13 ones (rubric reused: the builder was given both
`--supersede-from` dirs so the unchanged rubric was not re-emitted). Still
open: read Raup 1966 and pin the definitions to it; put a measured specimen
beside at least one panel.

## Report and Zenodo deposit

BioEval and NeuroEval each have a Zenodo record bundling an arXiv-style report
PDF with a source archive; MorphoEval follows the same pattern.

- `report/report.md` is the prose; `report/build_report.py` fills its tables
  from the audit YAMLs and `rubric.yaml` (so they cannot drift from the
  published scores), draws the cycle figure (`report/draw_pipeline.py`),
  resolves the head count on Nanopub Query, and renders the PDF with pandoc +
  xelatex into the git-ignored `report/build/`.
- `report/zenodo_deposit.py --build-only` builds the PDF, a `git archive` of
  HEAD (tracked files only: no caches, outputs or local databases) and the
  metadata from `.zenodo.json`; without `--build-only` it creates a *draft*
  deposition (sandbox with `--sandbox`/`ZENODO_SANDBOX_TOKEN`, production with
  `ZENODO_TOKEN`), uploads both files and sets the metadata. Publishing — which
  mints the DOI and cannot be undone — is a separate `--publish --deposition
  <id>` run that asks for a typed `yes`. `--new-version-of <record>` makes a
  new version under the same concept DOI for later checkpoints.
- `CITATION.cff` and `.zenodo.json` at the root also feed GitHub's citation
  box and the GitHub–Zenodo release integration, if that is enabled instead.

## Current checkpoint

Tracked in [morpholib#1](https://github.com/fractastical/morpholib/issues/1). Faithfulness (weight 70) is reported along its two sub-axes — **model-implementation fidelity** (`impl.`, weight 30) and **biological/empirical validation** (`bio.`, weight 40) — alongside **reproducibility** (`repro.`, weight 30). Where a dimension is `not_applicable` (`*`), both the weight-renormalised applicable-only mean (`app.`) and the baseline-50 padded score (`pad.`) are reported.

### Primary checkpoint chain (`role: primary`)

For a complete, step-by-step walkthrough of how a 12-step MorphoEval + Lean 4 run works from Step 1 (`scout`) through Step 7 (`audit`) to Step 12 (`remed` Pass 2 loop), see **[`runs/2026-10-18-runions-venation-loop.md`](./runs/2026-10-18-runions-venation-loop.md)** (`40.9 → 91.3`, `+50.4` pts) and **[`runs/2026-10-18-vogel-phyllotaxis-loop.md`](./runs/2026-10-18-vogel-phyllotaxis-loop.md)** (`64.3 → 85.9`, `+21.6` pts).

| Simulation | checkpoint | assessor | impl. (30) | bio. (40) | repro. (30) | overall (app.) | overall (pad.) |
|---|---|---|---:|---:|---:|---:|---:|
| Runions leaf venation (`2005-runions-leaf-venation.py`) ([sample run](./runs/2026-10-18-runions-venation-loop.md)) | 365a3cd, audit 2026-10-18-venation-v2 (was 40.9) | Gemini 4 Argon + Claude Sonnet 5.5 | 90.7 | 89.8 | 93.9 | 91.3 | **91.3** |
| Wolfram / Meinhardt *Conus textile* Rule 30 (`1984-wolfram-conus-textile-rule30.py`) | fb234bc, audit 2026-10-18-conus | Gemini 4 Argon | 87.7 | 85.5 | 89.3 | 87.3 | **87.3** |
| Vogel / Douady–Couder golden-angle phyllotaxis (`1979-vogel-phyllotaxis.py`) | c89d18d, audit 2026-10-18-phyllotaxis-v2 (was 64.3) | Gemini 4 Argon + Claude Sonnet 5.5 | 84.7 | 84.8 | 88.7 | 85.9 | **85.9** |
| Hoffmann–Donoughe wing venation (`2018-hoffmann-wing-venation.py`) | f3fe478, audit 2026-10-13 | Claude Fable 5.1 | 81.7 | 80.0 | 71.3 | 77.8 | **77.8** |
| Thompson *Scarus* → *Pomacanthus* (`thompson/1917-thompson-scarus-pomacanthus.py`) | 232e8f8, audit 2026-10-17-thompson (was 70.2) | Claude Fable 5.1 | 70.3 | 77.3 | 82.0 | 76.6 | **76.6** |
| Odonata wing cells, fossil vs extant (`2026-eshghi-odonata-wing-cells.py`) | fbbcaf6, audit 2026-10-14-odonata | Claude Fable 5.1 | 63.3 (`70.0` app.) | 85.0 | 78.0 | 78.9 | **76.0** |
| LimbformDB limb morphologies (`2014-lobo-limbform-morphology.py`) | 747c57f, audit 2026-10-14-limbform | Claude Fable 5.1 | 81.7 | 61.3 | 74.3 | 71.3 | **71.3** |
| WormGUIDES *C. elegans* embryo atlas (`2015-santella-wormguides-celegans.py`) | f39a409, audit 2026-10-16-celegans (was 67.7) | Claude Fable 5.1 | 70.0 (`80.0` app.) | 64.4 | 78.0 | 72.4 | **70.2** |
| PlanformDB planarian morphology (`2013-lobo-planform-morphology.py`) | 161f269, audit 2026-10-14 | Claude Fable 5.1 | 81.7 | 55.0 | 66.7 | 66.5 | **66.5** |
| Raup shell coiling (`create_raup_animations.py`) | ba6fd20, audit 2026-10-17-raup (was 37.3) | Claude Fable 5.1 | 76.7 | 45.0 | 77.0 | 64.1 | **64.1** |
| Cervera–Levin–Mafe bioelectric synthesis (`2021-Cervera–Levin–Mafe.py`) | a0d6848, audit 2026-10-17-cervera-v2 (was 44.9 at e71ca55) | **GPT‑5.6 Sol (OpenAI)** | 49.3 | 44.3 | 76.9 | 55.6 | **55.6** |
| Chindapol coral flow (`2013-chindapol-coral-flow.py`) | f3fe478, audit 2026-10-13 | Claude Fable 5.1 | 53.3 | 43.8 | 70.0 | 55.5 | **55.5** |
| Gray–Scott (Turing) (`1952-turing-morpho.py`) | f3fe478, audit 2026-10-13 | Claude Fable 5.1 | 75.0 | 38.8 (`20.0` app.) | 49.0 | 53.6 | **52.7** |
| ~~PlanformDB planarian animation (`1900-planformDB_parser.py`)~~ superseded | f3fe478, audit 2026-10-13 | Claude Fable 5.1 | 9.3 | 2.5 | 36.3 | 14.7 | ~~14.7~~ |
| Bressloff V1 form constants (`2001-bressloff-v1-form-constants.py`), Petroff stromatolite (`2010-petroff-stromatolite.py`), Kahng coral (`2024-kahng-coral-geometry.py`), Zhang sorting morphogenesis (`sorting-morphogenesis/`), Thompson demo (`thompson/1917-thompson.py`) | — | — | — | — | — | — | *unevaluated* |

### Complete 10-script dual-agent second-assessor pass (`2026-10-17-blinded-sample-audit.yaml`, `role: second_assessor`, `gemini-4-argon`, Google)

All 10 active evaluated scripts were independently scored by `gemini-4-argon` (Google) at commit `5659d5e` so that every evaluated script is dual-scored across two model providers (9 paired with `claude-fable-5-1`, Anthropic; 1 paired with `gpt-5.6-sol`, OpenAI). Across all 10 scripts, rank-order agreement on overall scores is strong (Spearman $\rho = 0.891$, Kendall $\tau = 0.778$, Pearson $r = 0.920$ on padded scores; $\rho = 0.903$ on applicable-only scores; and exact $\rho = 1.00$, $\tau = 1.00$ on the initial 5-script strictly blinded subset; dimension-level Spearman $\rho = 0.796$, Pearson $r = 0.813$ across all 70 dimension pairs, mean signed offset $+6.5$ pts, $\text{MAD} = 10.4$ pts). The 80 signed `pk:LLMClaimAssessment` nanopublications are stored in `morphoeval/kg/2026-10-17-blinded-sample-audit/signed/`.

| simulation | primary (`claude-fable-5-1` / `gpt-5.6-sol`†) app. / **pad.** | 2nd assessor (`gemini-4-argon`) impl. (30) | 2nd bio. (40) | 2nd repro. (30) | 2nd assessor app. / **pad.** | $\Delta$ (pad.) | panel mean **pad.** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2018-hoffmann-wing-venation.py` | 77.8 / **77.8** | 77.3 | 84.6 | 80.7 | 81.3 / **81.3** | +3.5 | **79.5** |
| `thompson/1917-thompson-scarus-pomacanthus.py` | 76.6 / **76.6** | 90.0 | 89.8 | 93.0 | 90.8 / **90.8** | +14.2 | **83.7** |
| `2026-eshghi-odonata-wing-cells.py` | 78.9 / **76.0** | 75.3 (`88.0` app.) | 93.5 | 91.0 | 91.4 / **87.3** | +11.3 | **81.7** |
| `2014-lobo-limbform-morphology.py` | 71.3 / **71.3** | 86.7 | 77.0 | 89.3 | 83.6 / **83.6** | +12.3 | **77.5** |
| `2015-santella-wormguides-celegans.py` | 72.4 / **70.2** | 72.7 (`84.0` app.) | 77.0 | 86.9 | 81.9 / **78.7** | +8.5 | **74.4** |
| `2013-lobo-planform-morphology.py` | 66.5 / **66.5** | 84.0 | 67.3 | 81.4 | 76.5 / **76.5** | +10.1 | **71.5** |
| `create_raup_animations.py` | 64.1 / **64.1** | 86.0 | 68.5 | 86.4 | 79.3 / **79.3** | +15.2 | **71.7** |
| `2021-Cervera–Levin–Mafe.py`† | 55.6 / **55.6** | 54.0 | 50.2 | 81.6 | 60.7 / **60.7** | +5.1 | **58.1** |
| `2013-chindapol-coral-flow.py` | 55.5 / **55.5** | 49.3 | 44.3 | 78.3 | 56.0 / **56.0** | +0.5 | **55.7** |
| `1952-turing-morpho.py` | 53.6 / **52.7** | 57.0 | 33.1 (`5.0` app.) | 35.0 | 37.8 / **40.9** | −11.9 | **46.8** |
