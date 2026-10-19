---
title: "MorphoEval: A Pilot Case Study in Auditing Morphology Reproductions and Publishing Criterion-Referenced Scores as Nanopublications"
author:
  - "Joel Dietz (ORCID 0009-0007-4948-9192)"
date: "{{DATE}} — repository revision `{{COMMIT}}`"
abstract: |
  Reproductions of published models of biological form — reaction–diffusion
  patterns, shell-coiling morphospaces, graph-encoded planarian and limb
  morphologies, developmental atlases, and D'Arcy Thompson's coordinate
  transformations — are easy to write and hard to trust: a script can produce a
  plausible-looking animation while implementing the wrong equation, drawing
  the wrong thing, or never being compared with a specimen. MorphoEval is a
  seven-dimension, weighted rubric (model fidelity, specimen match, parameter
  expressivity, morphometric validation, source traceability, code
  traceability, reproducibility package) applied as a pilot case study to the
  reproductions in the `morpholib` repository. To prevent conflating faithful
  code with valid biology, the faithfulness dimensions are decomposed and
  reported along two separate axes — **model-implementation fidelity** (weight
  30: `model-fidelity` + `parameter-expressivity`) and **biological/empirical
  validation** (weight 40: `specimen-match` + `morphometric-validation`) —
  alongside **reproducibility** (weight 30). Scores on `[0, 100]` are explicitly
  defined as **uncalibrated criterion-referenced ratings**, not probabilities of
  faithfulness; when serialized into governed ProbKnow nanopublication
  templates (`pk:LLMClaimAssessment` and `pk:Claim`), `score / 100` is a
  unit-interval rescaling for template conformance rather than a calibrated
  posterior probability. Every score is backed by line-pinned evidence at a
  recorded commit and by independent morphometric measurements on the rendered
  output.   Across the {{N_TOTAL_AUDITS}} audit files ({{N_TOTAL_PASSES}}
  script-level evaluation passes covering {{N_SCRIPTS_ACTIVE}} active scripts
  plus 1 superseded script), {{N_PRIMARY_AUDITS}} primary checkpoint audits
  ({{N_PRIMARY_PASSES}} passes) are backed by {{N_NANOPUBS}} production
  nanopublications ({{N_HEADS}} current heads, plus {{N_PRIMARY_UNPUB_NANOPUBS}}
  signed v1/v2 loop nanopublications), and all {{N_SCRIPTS_ACTIVE}} active
  evaluated scripts are dual-scored across two model providers — combining
  {{N_BLINDED_AUDITS}} standalone second-assessor replication files
  ({{N_BLINDED_PASSES}} passes: 10 by `gemini-4-argon`, Google, in
  `2026-10-17-blinded-sample-audit.yaml`, and 1 blinded pass by
  `claude-fable-5-1`, Anthropic, in `2026-10-19-conus-second-assessor-audit.yaml`,
  totaling {{N_BLINDED_NANOPUBS}} signed second-assessor nanopublications), 2
  independent passes on Cervera–Levin–Mafe by `gpt-5.6-sol` (OpenAI), and 2
  closed-loop dual-agent passes (`gemini-4-argon` + `claude-sonnet-5-5-high`,
  Anthropic) on Runions leaf venation and Vogel phyllotaxis — so that no script
  relies solely on its builder grading itself. Across the 10-script
  `gemini-4-argon` second-assessor pass, rank-order agreement on overall scores
  is strong (Spearman $\rho = 0.891$, Kendall $\tau = 0.778$, Pearson $r =
  0.920$ on padded scores; $\rho = 0.903$ on applicable-only scores; and exact
  $\rho = 1.00$, $\tau = 1.00$ on the initial 5-script blinded subset;
  dimension-level $\rho = 0.796$, $r = 0.813$ across 70 pairs, $\text{MAD} =
  10.4$ points), while the blinded *Conus textile* replication (`50.5` vs `87.3`,
  $\Delta = -36.8$ points; 11-script / 77-pair provider-paired $\rho = 0.678$,
  $\text{MAD} = 13.8$ points) and the closed-loop audits of leaf venation
  (`40.9 → 91.3`) and phyllotaxis (`64.3 → 85.9`) demonstrate that independent
  cross-company assessors catch concrete mathematical and code defects that a
  builder-assessor misses.
---

# Introduction

The `morpholib` repository contains {{N_SCRIPTS_TOTAL}} active, self-contained
Python implementations of published models and datasets of biological form (plus
one superseded historical parser, `1900-planformDB_parser.py`, replaced by
`2013-lobo-planform-morphology.py`, and a minimal companion script `1966-raup.py`
audited jointly with `create_raup_animations.py`):

- **Thirteen active evaluated implementations (all 13 dual-scored across two model providers)**: Thompson's 1917 fish transformation
  grids tested across 791 species (`thompson/1917-thompson-scarus-pomacanthus.py`),
  Turing's 1952 / Pearson's 1993 Gray–Scott reaction–diffusion patterns
  (`1952-turing-morpho.py`), Raup's 1966 shell-coiling morphospace
  (`create_raup_animations.py` + `1966-raup.py`), Vogel's 1979 / Douady &
  Couder's 1992 golden-angle *Helianthus* phyllotaxis (`1979-vogel-phyllotaxis.py`),
  Wolfram's 1984 Rule 30 & Meinhardt–Klingler's 1987 *Conus textile* shell
  pigmentation (`1984-wolfram-conus-textile-rule30.py`), Runions et al.'s 2005
  auxin space-colonization leaf venation & Murray's law
  (`2005-runions-leaf-venation.py`), Chindapol et al.'s 2013
  flow-driven coral morphogenesis (`2013-chindapol-coral-flow.py`), Lobo,
  Malone & Levin's 2013 Planform planarian graph decoder
  (`2013-lobo-planform-morphology.py`), Lobo et al.'s 2014 Limbform limb graph
  decoder (`2014-lobo-limbform-morphology.py`), Santella et al.'s 2015
  WormGUIDES *C. elegans* embryogenesis atlas (`2015-santella-wormguides-celegans.py`),
  Hoffmann et al.'s 2018 insect wing venation model
  (`2018-hoffmann-wing-venation.py`), Cervera, Levin & Mafe's 2021 bioelectric
  planarian synthesis (`2021-Cervera–Levin–Mafe.py`), and Eshghi, Buesch &
  Gorb's 2026 Odonata wing-cell deep-time scaling study
  (`2026-eshghi-odonata-wing-cells.py`).
- **Five unevaluated implementations** (recorded with `status: unevaluated`
  and carrying no scores): Bressloff et al.'s 2001/2002 V1 Euclidean
  symmetry-breaking & retino-cortical form constants
  (`2001-bressloff-v1-form-constants.py`), Petroff et al.'s 2010 stromatolite
  spacing model (`2010-petroff-stromatolite.py`), Kahng, Odle & Wakeman's 2024
  coral colony geometry model (`2024-kahng-coral-geometry.py`), Zhang,
  Goldstein & Levin's 2025 autonomous cell-view sorting morphogenesis &
  chimeric arrays (`sorting-morphogenesis/`), and the standalone Thompson grid
  demo (`thompson/1917-thompson.py`).

Each script writes a figure and an animation. The question MorphoEval asks is:
*how faithful is each implementation to its cited source formalism and to
empirical biological data, and can an independent party reproduce and audit that
judgement?* This report presents MorphoEval as a **pilot case study** in
evidence-pinned, nanopublication-backed software auditing — not yet a
calibrated, general-purpose evaluator — and documents its rubric, score
semantics, multi-assessor concordance, and biological versus implementation
findings.

The project started from a failure. An initial audit of the repository's early
animations (October 2026) found that two of them did not depict biologically
plausible forms, in both cases for a code or geometry reason rather than a
styling one: the original planarian script (`1900-planformDB_parser.py`) drew
the convex hull of sparse organ landmarks instead of decoding the body outline,
so a double-headed, a headless and a wild-type worm were indistinguishable; and
the Raup shell script applied its axial translation term to coiling angle rather
than whorl radius, so every shell collapsed toward a flat planispiral. A
legitimate-looking GIF had hidden both. The remedy was not just to fix the
scripts but to build an evaluation layer that scores each reproduction against
its source model and against biological specimens, records line-pinned and
morphometric evidence, and makes regressions and assessor disagreements
visible.

MorphoEval is the third of a family. BioEval (Zenodo
10.5281/zenodo.20567719) scores computational-biology *papers* for
reproducibility and information-theoretic rigour; NeuroEval
(10.5281/zenodo.20622878) does the same for computational and biohybrid
neuroscience. MorphoEval turns the same machinery inward and scores *our own
simulation and analysis code* across versioned checkpoints.

# The rubric (version 0.1.0) and score semantics

Seven dimensions in two top-level groups, decomposed into three orthogonal
sub-axes whose weights sum to 100 (`30 + 40 + 30`):

{{RUBRIC_TABLE}}

## Decomposition into three sub-axes

Reporting a single weighted average can obscure whether a script succeeds as a
piece of software, as a faithful transcription of a paper's equations, or as an
empirically validated model of an organism. MorphoEval therefore decomposes its
seven dimensions into three sub-axes reported side by side in every table:

1. **Model-implementation fidelity (`impl.`, weight 30 = `model-fidelity` 20 +
   `parameter-expressivity` 10):** Does the code implement the equations,
   algorithm, or formal encoding of the cited source paper as published, and do
   the model's parameters visibly and monotonically control the features they
   are defined to control?
2. **Biological and empirical validation (`bio.`, weight 40 = `specimen-match`
   25 + `morphometric-validation` 15):** Does the rendered output visually match
   real biological specimens (or, where clearly distinguished, the source
   authors' reference figures), and are quantitative morphometrics computed on
   the output and compared against specimen measurements or published reference
   values?
3. **Reproducibility and traceability (`repro.`, weight 30 =
   `source-traceability` 10 + `code-traceability` 8 + `repro-package` 12):** Are
   papers, datasets, and external code linked by persistent identifiers and
   SHA-256 digests, can modelling steps be traced to specific equations or
   sections, and is the execution deterministic under a pinned environment?

## Uncalibrated rubric scores versus probabilities

A MorphoEval score $s \in [0, 100]$ is an **uncalibrated, criterion-referenced
ordinal/interval rating**, not a frequentist or Bayesian probability. A score of
$76/100$ on `eshghi-odonata-wing-cells` means that the script satisfies the
rubric's stated criteria to degree $76$ on the $0\text{--}100$ rubric scale; it
does **not** establish a $76\%$ probability that the reproduction is faithful
($P(\text{faithful}) \neq 0.76$), nor can $s/100$ be combined across assessors
via log-odds or Bayes-factor pooling (`pk:BayesianAssessment`) without empirical
calibration against a ground-truth reference set.

Why, then, do the published RDF nanopublications write `score / 100` into
`pk:hasProbability` (on `pk:LLMClaimAssessment`) and `overall / 100` into
`pk:weightOfEvidence` (on `pk:Claim`)? Because the governed ProbKnow Space
templates (`RAku3DIN` for LLM assessments and `RAIVOEuh` for claims) provide
only `pk:hasProbability` and `pk:weightOfEvidence` (constrained to `xsd:double`
in $[0, 1]$) as numeric literal slots read by Nanodash and Space queries. Within
MorphoEval nanopublications, those two predicates must therefore be read
strictly as **unit-interval rescalings ($s / 100$) of uncalibrated rubric
scores** used for template conformance. Every emitted assessment links to the
rubric via `dct:conformsTo` (`pkr:concept/morphoeval-rubric-0.1.0`), and the
overall assessment's `rdfs:comment` carries an explicit disclaimer (verified by
`build_morphoeval_kg.py --check`) stating that the score is an uncalibrated
criterion-referenced rating rather than a Bayesian probability. Because
statement `st09` (`pk:hasProbability`) in template `RAku3DIN` and statement
`st08` (`pk:weightOfEvidence`) in `RAIVOEuh` are declared
`nt:OptionalStatement`, `build_morphoeval_kg.py` also provides
`--omit-probability` (carrying the score solely in `rdfs:label`,
`pk:hasVerdict`, and `rdfs:comment`) when strict separation from probabilistic
predicates is preferred over dashboard sorting. Upstream in the ProbKnow
ontology (v1.2, [`project-delphi-ontology#12`](https://github.com/fractastical/project-delphi-ontology/issues/12)),
dedicated predicates `pk:hasScore` (`[0, 100]`) and `pk:conformsToRubric` and a
Rubric Assessment Nanopublication template (`https://w3id.org/np/RAOAK6Fe...`)
have been defined so future waves can decouple rubric ratings from Bayesian
credences natively.

## Handling non-applicable (`not_applicable`) dimensions

Two dimensions can be structurally inapplicable to a valid script:
`parameter-expressivity` (weight 10) for an observational data analysis that has
no generative morphospace parameters to sweep (`2026-eshghi-odonata-wing-cells.py`
and `2015-santella-wormguides-celegans.py`), and `specimen-match` (weight 25) for a
generic chemical pattern generator not tied to a specific organism
(`1952-turing-morpho.py`).

Earlier drafts stated that assigning a non-applicable dimension the baseline
value $s_{\text{NA}} = 50$ "neither rewards nor penalises the overall." That
statement is **mathematically incorrect**. If a script achieves a weighted mean
$\bar{s}_{\text{app}}$ across its applicable dimensions (total weight
$100(1 - \alpha)$) and has weight fraction $\alpha = w_{\text{NA}} / 100$ on
non-applicable dimensions, substituting $s_{\text{NA}} = 50$ yields:
$$\bar{s}_{\text{pad}} = (1 - \alpha)\,\bar{s}_{\text{app}} + 50\,\alpha = \bar{s}_{\text{app}} - \alpha\,\bigl(\bar{s}_{\text{app}} - 50\bigr).$$
Padding with $50$ is therefore a **linear shrinkage toward 50**: it penalises
every script whose applicable mean exceeds $50$ (pulling
`2026-eshghi-odonata-wing-cells.py` down by $2.9$ points from
$\bar{s}_{\text{app}} = 78.9$ to $\bar{s}_{\text{pad}} = 76.0$, and
`2015-santella-wormguides-celegans.py` down by $2.2$ points from $72.4$ to $70.2$) and
rewards any script whose applicable mean lies below $50$ (pulling the blinded
assessment of `1952-turing-morpho.py` up by $3.1$ points from
$\bar{s}_{\text{app}} = 37.8$ to $\bar{s}_{\text{pad}} = 40.9$).

Only **weight renormalisation over applicable dimensions**
($\bar{s}_{\text{app}} = \sum_{d \in \text{app}} w_d s_d / \sum_{d \in \text{app}} w_d$)
is mathematically neutral. Substituting $50$ acts instead as a deliberate
shrinkage penalty when a rubric criterion has no empirical bearer (for example,
preventing an organism-free pattern generator from outranking an
organism-validated simulation on overall score). To keep both interpretations
transparent, all tables in this report present **both** the weight-renormalised
applicable-only mean (`app.`) and the baseline-50 padded score (`pad.`).

# Method

![How a body gets its form, and how a form gets into the repository. Tier A, making a reproduction: a person scouts a paper, dataset, or claim and points the builder agent at it; the model as published (equations, graph encoding, or empirical claim); the formal source (parameter values, database or atlas records, measurement tables), fetched or vendored with a pinned SHA-256 digest; the rules pinned down and verified across records; and the script, which reproduces, decodes, or tests. Box 6, between the tiers: a frame of each of the 16 implementations' own animations, labelled with its source paper and colour-coded by epistemic role (`reproduce`, `decode`, `test`). Tier B, the MorphoEval cycle: dual-agent audit with line-pinned evidence at a recorded commit, independent morphometric measurement of the rendered outputs (`measure_outputs.py`), each score serialized as a machine-readable claim assessment (`build_morphoeval_kg.py`), cryptographic signing and publication (`sign_morphoeval.py`), the public record, and remediation (`GitHub issue #1`). Purple figures mark human decisions; robot icons are colour-coded by model provider read directly from the audit files — every evaluated script is dual-scored across two companies' models (14/31 passes by Anthropic `claude-fable-5-1`, 15/31 by Google `gemini-4-argon`, and 2/31 by OpenAI `gpt-5.6-sol`, plus 2 closed-loop passes with Anthropic `claude-sonnet-5-5-high`).](pipeline.png){height=70%}

## Audits and version pinning

An audit is a YAML file (`morphoeval/audits/<date>-<name>-audit.yaml`) that
records the rubric version, the assessor (`id`, `label`, `homepage`) and its
role (`primary` checkpoint vs `second_assessor` blinded replication), the
checkpoint (repository URL, exact git commit hash, Python and library versions,
input SHA-256 digests), the bibliographic references (DOIs or URLs), and, per
audited script, a summary, a score and rationale for each dimension, a list of
evidence loci, and remediation items. Every evidence locus is a repository file
and line range at the checkpoint commit, a fingerprinted rendered output (line
range `0–0`), or a line range in the measurement code. The builder
(`build_morphoeval_kg.py --check`) validates every audit against the rubric
before any graph is emitted.

To keep the release self-consistent, we distinguish four counts throughout this
report:

- **Repository implementations (`scripts`):** **{{N_SCRIPTS_TOTAL}} active model
  scripts** ({{N_SCRIPTS_ACTIVE}} evaluated + {{N_SCRIPTS_UNEVALUATED}}
  unevaluated), plus **1 superseded script** (`1900-planformDB_parser.py`,
  replaced by `2013-lobo-planform-morphology.py`, giving {{N_SCRIPTS_EVALUATED}}
  distinct scripts that have ever been audited).
- **Primary checkpoint audit files (`primary audits`):** **{{N_PRIMARY_AUDITS}}
  YAML files** (`2026-10-13-morphology-audit` through
  `2026-10-18-venation-audit-v2`), containing **{{N_PRIMARY_PASSES}} script-level
  evaluation passes** (5 scripts in the initial 2026-10-13 batch plus 15
  single-script audits or re-scores: 9 files / 13 passes by `claude-fable-5-1`,
  2 files / 2 passes by `gpt-5.6-sol`, and 5 files / 5 passes by
  `gemini-4-argon`).
- **Dual-agent second-assessor replication (`role: second_assessor`):**
  **{{N_BLINDED_AUDITS}} audit files** containing **{{N_BLINDED_PASSES}}
  script-level evaluation passes**: `2026-10-17-blinded-sample-audit.yaml` (10
  passes by `gemini-4-argon`, Google, across the first 10 active evaluated
  scripts: 9 paired with `claude-fable-5-1` and 1 paired with `gpt-5.6-sol`) and
  `2026-10-19-conus-second-assessor-audit.yaml` (1 blinded pass by
  `claude-fable-5-1`, Anthropic, paired with `gemini-4-argon` on
  `1984-wolfram-conus-textile-rule30.py`). In addition, the two closed-loop
  remediations (`2005-runions-leaf-venation.py` and `1979-vogel-phyllotaxis.py`)
  were dual-assessed inside their 12-step loops (`morphoeval/runs/2026-10-18-*.yaml`)
  by `gemini-4-argon` (Google) and `claude-sonnet-5-5-high` (Anthropic).
- **Total audit corpus:** **{{N_TOTAL_AUDITS}} audit files** and
  **{{N_TOTAL_PASSES}} script-level evaluation passes**, backed by
  **{{N_NANOPUBS}} production nanopublications** (**{{N_HEADS}} current heads**),
  **{{N_BLINDED_NANOPUBS}} signed second-assessor nanopublications**, and
  **{{N_PRIMARY_UNPUB_NANOPUBS}} signed loop nanopublications**.

## Measurements

Scores are judgements; the numbers behind them are not. A separate program,
`measure_outputs.py`, re-runs or re-reads each audited script's own functions
and takes morphometrics from the actual geometry or pixels of the outputs —
silhouette aspect ratios and intersections-over-union against the source
authors' figures or specimen photographs, vein-domain counts and circularities,
Murray's-law hydraulic exponent $\alpha_{\text{OLS}}$, Fibonacci parastichy pair
counts $(34, 55)$ and divergence-angle RMSE, Procrustes residuals, allometric
exponents, and dominant 2D FFT wavelengths. Each measurement carries its method,
unit, reference value and source, and the output file's SHA-256 digest in
`<audit>.measurements.yaml`.

## Assessors and dual-agent protocol

Models from three AI providers have contributed audits so that every active
evaluated script ({{N_SCRIPTS_ACTIVE}} of {{N_SCRIPTS_ACTIVE}}) is scored by two
independent companies' models:

1. **Anthropic assessors (`claude-fable-5-1` and `claude-sonnet-5-5-high`):**
   `claude-fable-5-1` scored 9 primary audit files (13 script-level passes
   across 9 active scripts plus the superseded `1900-planformDB_parser.py`) and
   performed a strictly blinded `role: second_assessor` audit on
   `1984-wolfram-conus-textile-rule30.py` (`2026-10-19-conus-second-assessor-audit.yaml`,
   scoring `50.5` vs the builder's `87.3`). In addition, `claude-sonnet-5-5-high`
   served as the independent cross-company assessor in the 12-step remediation
   loops for `2005-runions-leaf-venation.py` (`40.9 → 91.3`) and
   `1979-vogel-phyllotaxis.py` (`64.3 → 85.9`).
2. **OpenAI assessor (`gpt-5.6-sol`):** Scored `2021-Cervera–Levin–Mafe.py`
   across two rounds (`2026-10-17-cervera-audit` at commit `e71ca55`, overall
   `44.9`; and `2026-10-17-cervera-audit-v2` at commit `a0d6848`, overall
   `55.6`), catching seven substantive modelling and calibration defects in
   round 1 and two newly introduced threshold/legend bugs in round 2
   (subsequently fixed at commit `0154143`).
3. **Google assessor (`gemini-4-argon`):** Scored all 10 initial active
   evaluated scripts as a standalone `role: second_assessor`
   (`2026-10-17-blinded-sample-audit.yaml`, 80 signed nanopublications),
   spanning all three epistemic roles (`test`, `decode`, `reproduce`) and both
   `not_applicable` cases — with an initial 5-script subset (`thompson`,
   `odonata`, `limbform`, `raup`, `turing`) executed in strictly isolated
   subagent sessions followed by full 10-script completion across `hoffmann`,
   `celegans`, `planform`, `cervera`, and `chindapol` — and also authored the
   primary checkpoint audits for `1984-wolfram-conus-textile-rule30.py`,
   `2005-runions-leaf-venation.py` (`v1` & `v2`), and
   `1979-vogel-phyllotaxis.py` (`v1` & `v2`).

# The knowledge graph

Each audit is built into nanopublications expressed in the ProbKnow / Project
Delphi ontology (v1.1) and shaped by the governed templates of the ProbKnow
Space:

- per script (primary chain), a **`pk:Claim`** (`pkr:claim/morphoeval-<script>`)
  with label, description, context (scripts, output, organism, commit, non-DOI
  sources), `pk:derivedFromPaper` for each DOI, `pk:hasAssessment` to its eight
  assessments, and `pk:weightOfEvidence` = overall / 100 (unit-interval rescaled
  uncalibrated rubric score);
- per dimension, a **`pk:LLMClaimAssessment`**
  (`pkr:assessment/<audit>-<script>-<dimension>`) evaluating the shared
  hypothesis `pkr:hypothesis/morphoeval-<script>-<dimension>`, with
  `pk:assessedBy pkr:model/<assessor>`, `pk:hasProbability` = score / 100
  (unit-interval rescaled rubric score, optional under `RAku3DIN`), a verdict
  band, and `rdfs:comment` containing the rationale, commit-pinned GitHub line
  ranges, fingerprinted outputs, and morphometric measurements;
- an **overall** `pk:LLMClaimAssessment` evaluating `pkr:hypothesis/morphoeval-<script>`,
  with remediation items recorded as `pk:evidenceAgainst`;
- the **rubric** as a `pk:Concept` with one `skos:narrower` concept per
  dimension (shared by all audits and emitted once in
  `2026-10-13-morphology-audit`).

Because hypothesis URIs (`pkr:hypothesis/morphoeval-<script>-<dimension>`) are
keyed to the script and rubric dimension whereas assessment URIs
(`pkr:assessment/<audit>-<script>-<dimension>`) are keyed to the audit, multiple
assessors evaluate the **exact same hypothesis node** in RDF without overwriting
one another. In the current graph, all 11 scripts with standalone
`role: second_assessor` audits carry parallel `pk:LLMClaimAssessment`s from two
different companies' models across their 88 hypothesis nodes (`10 scripts × 8` in
`2026-10-17-blinded-sample-audit` plus `1 script × 8` in
`2026-10-19-conus-second-assessor-audit`), and the remaining 2 active scripts
(`runions-leaf-venation` and `vogel-phyllotaxis`) carry signed v1/v2 loop
assessments (`40` nanopublications) paired with `claude-sonnet-5-5-high`
in `morphoeval/runs/2026-10-18-*.yaml`.
Two extensions that the graph schema supports remain open for future work: human
expert assessments alongside the LLM assessments, and ProbKnow claims representing
the *source papers' biological hypotheses* themselves (linking "does the script
reproduce the paper?" to "is the paper's biological claim supported by
evidence?").

# Results

## Current primary-chain scores

{{SCORE_TABLE}}

Five active implementations in the repository (`2001-bressloff-v1-form-constants.py`,
`2010-petroff-stromatolite.py`, `2024-kahng-coral-geometry.py`,
`sorting-morphogenesis/`, and `thompson/1917-thompson.py`) have not yet
been audited and carry no scores. For `2018-hoffmann-wing-venation.py` (audited
at `f3fe478`), `2013-chindapol-coral-flow.py` (`f3fe478`), and
`2021-Cervera–Levin–Mafe.py` (audited at `a0d6848`; post-audit threshold/legend
fix at `0154143`), later commits in the repository touched those files after
their most recent published audit; their scores remain pinned to the audited
checkpoint commits shown in the table until a re-audit is published.

## Implementation fidelity versus biological and mechanistic validity

A central methodological lesson of this pilot is that **implementation fidelity
and biological validity are distinct dimensions that must not be collapsed into
a single rank**:

1. **A faithful implementation can reproduce — or refute — a scientifically
   inadequate model.** On `thompson/1917-thompson-scarus-pomacanthus.py`, both
   assessors assign high scores (`76.6` primary, `90.8` blinded) because the
   code rigorously formalises Thompson's transformation grids, pins all data by
   SHA-256, and cross-validates five nested coordinate families across 791
   species and two segmented fish photographs. Yet the scientific conclusion of
   that script is that **Thompson's coaxial-circle hypothesis fails
   empirically**: oblique coaxial circles explain only $32\%$ of the landmark
   Procrustes distance (virtually identical to a 6-parameter affine map at
   $31\%$), fail leave-one-landmark-out cross-validation ($0.0371$ vs $0.0312$
   for no deformation), and reduce full-silhouette overlap (`IoU = 0.618`) below
   the undeformed similarity baseline (`IoU = 0.662`). Conversely,
   `1952-turing-morpho.py` (`impl. = 75.0`, `bio. = 38.8` padded / `20.0`
   applicable-only) and `create_raup_animations.py` (`impl. = 76.7`, `bio. =
   45.0`) faithfully integrate their published mathematical equations while
   providing zero comparison against biological specimens (`bio. <= 45`).
2. **Matching or painting a static silhouette or cartoon does not validate a
   generative developmental mechanism.** Across the corpus, scripts fall into
   three distinct epistemic roles (`reproduce`, `decode`, `test`):
   - **Static graph/atlas decoders (`decode`):** `2013-lobo-planform-morphology.py`
     (`impl. = 81.7`, `bio. = 55.0`) and `2014-lobo-limbform-morphology.py`
     (`impl. = 81.7`, `bio. = 61.3`) decode human-curated relational graph
     tables (`RegionParam`/`RegionsLink` and `Link`/`JointParam`) into 2D
     outlines and compare them against the source papers' software cartoons
     (mean silhouette IoU $0.78$ for Planform, $0.41\text{--}0.74$ for
     Limbform). That agreement validates the **geometric decoding rule** of the
     database formalism; it does **not** simulate a biophysical mechanism that
     generates a planarian or a limb, nor does it compare against photographs of
     real animals.      Similarly, `2015-santella-wormguides-celegans.py` (`impl. = 80.0`
     applicable-only, `bio. = 64.4`) replays measured 4D nuclear coordinates
     from WormGUIDES rather than solving embryonic mechanics.
   - **Silhouette masking and cartoon switching in `reproduce` scripts:**
     `2021-Cervera–Levin–Mafe.py` (`impl. = 49.3`, `bio. = 44.3`) projects a 1D
     reaction–diffusion wavefront $x^*(f_2)$ onto a static segmented photograph
     of *Schmidtea mediterranea* and switches categorically between two
     pre-drawn Planform cartoons (`#1 Wild type` vs `#2 Double head`) when a
     threshold is crossed. Painting a 1D coordinate onto a real worm silhouette
     does not mean the 1D equations generated the worm's 2D shape. Likewise,
     `2018-hoffmann-wing-venation.py` (`impl. = 81.7`, `bio. = 80.0`) seeds its
     Lloyd relaxation centres from local maxima of the real wing's own primary-vein
     distance transform, so macro-compartment geometry is inherited from the
     specimen boundary rather than generated de novo.

The table below separates every active script along these three orthogonal axes
(`impl.`, `bio.`, `repro.`), alongside its epistemic role, its empirical
validation target, and what its score does and does not establish about
biological mechanism:

{{DECOMPOSITION_TABLE}}

## Cross-company dual-agent scoring and inter-rater analysis

To eliminate single-agent self-evaluation across the entire active benchmark, we
compared the primary checkpoint scores against independent cross-company
second-assessor passes across all 11 scripts with standalone `role:
second_assessor` audits (`2026-10-17-blinded-sample-audit.yaml`, 10 passes by
`gemini-4-argon`, Google; and `2026-10-19-conus-second-assessor-audit.yaml`, 1
blinded pass by `claude-fable-5-1`, Anthropic on
`1984-wolfram-conus-textile-rule30.py`), alongside the 2 closed-loop dual-agent
remediations (`2005-runions-leaf-venation.py` and `1979-vogel-phyllotaxis.py`,
jointly audited by `gemini-4-argon` and `claude-sonnet-5-5-high`, Anthropic):

{{BLINDED_TABLE}}

Four empirical findings emerge from the cross-company comparison:

1. **Strong rank-order concordance across the 10-script `gemini-4-argon` pass and
   exact rank-order agreement on the initial 5-script blinded subset.** Across
   the 10 initial active scripts scored by `gemini-4-argon`, overall scores
   correlate strongly with the primary assessor (Spearman $\rho = 0.891$, $p =
   5.4 \times 10^{-4}$; Kendall $\tau = 0.778$, $p = 9.5 \times 10^{-4}$;
   Pearson $r = 0.920$, $p = 1.7 \times 10^{-4}$ on baseline-50 padded scores;
   Spearman $\rho = 0.903$, $p = 3.4 \times 10^{-4}$ on applicable-only
   renormalised scores). On the initial five-script strictly blinded sample
   (`Thompson`, `Odonata`, `Limbform`, `Raup`, `Turing`), both assessors
   independently rank the five scripts in the exact same order:
   $$\text{Thompson} > \text{Odonata} > \text{Limbform} > \text{Raup} > \text{Turing}$$
   (Spearman $\rho = 1.00$, Kendall $\tau = 1.00$, $p = 0.0167$; Pearson
   $r = 0.947$, $p = 0.0146$). Across all $10 \times 7 = 70$
   script-by-dimension score pairs in that batch, Spearman $\rho = 0.796$ ($p =
   1.7 \times 10^{-16}$) and Pearson $r = 0.813$ ($p = 1.3 \times 10^{-17}$).
2. **Bidirectional self-grading audit on *Conus textile* (`87.3` builder vs `50.5`
   blinded 2nd assessor, $\Delta = -36.8$).** Whereas `claude-fable-5-1` built
   the initial 9 scripts and `gemini-4-argon` served as second assessor, on
   `1984-wolfram-conus-textile-rule30.py` the roles were reversed:
   `gemini-4-argon` both built the script and authored its primary audit (`87.3`
   in `2026-10-18-conus-audit.yaml`), and `claude-fable-5-1` performed a
   strictly blinded second-assessor audit (`50.5` in
   `2026-10-19-conus-second-assessor-audit.yaml`). The blinded assessor flagged
   four severe gaps the builder-assessor had overlooked: (i) Eq. 1 in the
   Meinhardt–Klingler (1987) solver (`lines 122–134`) omits the resting
   activator production term $\rho_0$ (`rho_0`), alters the denominator from
   $(1 + \kappa a^2)$ to `1.0 + kappa * a`, and substitutes a non-Biological
   Gaussian spatial filter for diffusion (`model-fidelity` `48` vs `88`); (ii)
   no parameter sweeps over $D_a, D_b, \rho_0, \kappa$ or other ECA rules are
   executed (`parameter-expressivity` `35` vs `85`); (iii) the 3D shell overlay
   uses hardcoded cylinder-like parameters (`W=1.8, D=0.12, T=2.8`) rather than
   digitized *Conus textile* morphometrics and wraps a synthetic pattern rather
   than comparing against the segmented specimen patches (`specimen-match` `52`
   vs `88`); and (iv) all metrics are computed on single fixed RNG seeds with no
   distributional tests (`morphometric-validation` `45` vs `90`). Including
   *Conus* across all 11 scripts with standalone `role: second_assessor` audits
   ($11 \times 7 = 77$ dimension pairs), pooled provider-paired Spearman $\rho =
   0.678$ ($p = 1.3 \times 10^{-11}$) with $\text{MAD} = 13.8$ points.
3. **Uncalibrated LLM assessors exhibit systematic scale/anchoring offsets.**
   Across the 70 dimension pairs in the 10-script batch, the mean signed
   difference (`2nd - primary`) is $+6.46$ points and the mean absolute
   difference ($\text{MAD}$) is $10.40$ points (on the 5-script subset, $+7.06$
   and $\text{MAD} = 13.74$ points). Crucially, the offset is **not** uniform
   leniency:
   - On the untouched, unseeded, uncited `1952-turing-morpho.py` script (at
     commit `5659d5e`), `gemini-4-argon` was **11.9 points stricter** (`40.9` vs
     `52.7` padded; `37.8` vs `53.6` applicable-only), penalising the absence of
     literature DOIs (`source-traceability` 15 vs 55), the absence of wavelength
     measurements (`morphometric-validation` 5 vs 20), and the misleading
     reaction-kinetics comments (`code-traceability` 30 vs 55).
   - On `2013-chindapol-coral-flow.py` (`56.0` vs `55.5`, $\Delta = +0.5$),
     `2018-hoffmann-wing-venation.py` (`81.3` vs `77.8`, $\Delta = +3.5$), and
     `2021-Cervera–Levin–Mafe.py` (`60.7` vs `55.6`, $\Delta = +5.1$, where
     `gemini-4-argon` inspected commit `5659d5e` after commit `0154143` fixed
     the two threshold/legend bugs noted by `gpt-5.6-sol` at `a0d6848`), the two
     assessors agree within $0.5\text{--}5.1$ points.
   - On the remaining refactored scripts (`Thompson`, `Odonata`, `Limbform`,
     `Raup`, `Planform`, `C. elegans`), the second assessor scored **8.5 to 15.2
     points higher**. Part of this shift reflects real repository state: the
     repository-wide dependency lockfiles (`requirements.txt`,
     `requirements.lock.txt`, added in commit `9ca5457`) and the PlanformDB
     SHA-256 check (`datasets/planform/planformDB.sha256`) were present at
     commit `5659d5e`, raising `repro-package` and `source-traceability`. The
     remainder reflects **rubric-anchor ambiguity on `specimen-match`** when no
     real biological specimen is present: on `raup-shell-coiling` (`76` vs `45`),
     `planform-morphology-animation` (`74` vs `60`), and
     `lobo-limbform-morphology` (`86` vs `65`), the second assessor weighted
     visual fidelity to the named shell archetypes and published software
     cartoons more heavily while deducting 14–26 points for the absence of
     biological specimen photographs, whereas the primary assessor treated
     comparison to real specimens as a hard ceiling at `45–65`. This divergence
     reinforces both why raw rubric scores must not be treated as calibrated
     probabilities and why biological specimen comparison must be reported
     separately from formal model decoding.
4. **Independent assessors consistently catch concrete code and data defects
   that a builder-assessor misses.** Across `gpt-5.6-sol` (on
   `2021-Cervera–Levin–Mafe.py`), `gemini-4-argon` (across the 10-script batch),
   `claude-fable-5-1` (on `1984-wolfram-conus-textile-rule30.py`), and
   `claude-sonnet-5-5-high` (in the leaf-venation and phyllotaxis loops),
   independent passes uncovered verifiable bugs in scripts the builder had
   already audited:
   - *`1984-wolfram-conus-textile-rule30.py` (lines 122–134):* omits basal
     activator production $\rho_0$ and replaces $(1 + \kappa a^2)$ with a linear
     denominator in the Meinhardt–Klingler (1987) PDE.
   - *`2014-lobo-limbform-morphology.py` (line 92):* `DB_SHA256` contains **65
     hexadecimal characters** (an extra trailing `'0'` appended to the true
     64-character SHA-256 digest of `limbformDB_2.1.0.ldb`), causing
     `resolve_db_path()` to emit a false-positive hash mismatch warning on every
     run; in addition, `Link.side`, `Link.reversed`, `Link.inverted`, and
     `Digit.reversed` are parsed into dataclasses (lines 146–148, 165) but never
     applied in `layout()`.
   - *`1952-turing-morpho.py` (lines 36–40, 75):* inline comments misdescribe
     the Gray–Scott reaction by calling the consumed substrate `U` an
     "inhibitor" (`# U inhibits V's growth quadratically`), and the script writes
     `turing_patterns_black_neon.gif` (black-neon palette, axes hidden), which
     does not match the committed `docs/media/turing_gray_scott.gif` (Viridis
     palette with $0\text{--}200$ axes).
   - *`create_raup_animations.py` (lines 289–344):*
     `create_parameter_space_visualization()` zips three values of $W$ with
     three values of $D$ (and $W$ with $T$) along a 1D diagonal rather than
     sampling a 2D morphospace grid, and fixes $S = 1.0$ across all presets.
   - *`2026-eshghi-odonata-wing-cells.py` (lines 255–349, 447–452):*
     `fossil_drawing()` (used for display outlines) thresholds at `< 128` and
     keeps only the largest connected component, whereas `_read_drawing()` (used
     for cell segmentation) thresholds at `< 160` and keeps all components $\ge
     150\text{ px}$; running the script also overwrites the tracked file
     `datasets/odonata/fossil_wing_resegmentation.csv` in place.

## Score history across all audits

{{HISTORY_TABLE}}

## What the audits found, script by script

**Runions auxin space-colonization leaf venation (`2005-runions-leaf-venation.py`, pad. 40.9 → 91.3 in 12-step dual-agent loop; impl. 90.7, bio. 89.8, repro. 93.9).**
Evaluated through a 12-step closed-loop remediation audited jointly by
`gemini-4-argon` and `claude-sonnet-5-5-high` (`morphoeval/runs/2026-10-18-runions-leaf-venation-loop.yaml`):
the baseline script (`40.9`) lacked relative neighborhood graph (RNG)
anastomosis, parameter sweeps, and cleared-leaf comparison. After remediation
(`91.3`), the script implements open/closed space colonization (`--reticulate`,
forming 56 closed loops with HLD $\approx 0.145\,\text{mm}^{-2}$), verifies
Murray's pipe-model law via OLS regression ($\alpha = 3.0000$, $R^2 = 1.0000$),
sweeps $r_k/r_a$ across 4 regimes, and compares against a cleared *Acer*
dicot leaf skeleton (`datasets/venation/acer_leaf_skeleton_specimen.jpg`).

**Wolfram Rule 30 & Meinhardt–Klingler *Conus textile* pigmentation (`1984-wolfram-conus-textile-rule30.py`, pad. 87.3 primary builder / 50.5 blinded 2nd assessor; impl. 87.7 [47.3 2nd], bio. 85.5 [38.8 2nd], repro. 89.3 [69.2 2nd]).**
Contrasts discrete Wolfram Rule 30 ECA against continuous 1D activator–inhibitor
PDE reaction–diffusion against two SHA-256-pinned *Conus textile* specimen
photographs (measuring triangle aspect ratio, horizontal autocorrelation length,
and vertical persistence). As detailed in the inter-rater analysis above, its
primary audit (`87.3`) was written by its builder (`gemini-4-argon`), whereas a
strictly blinded cross-company second-assessor pass (`claude-fable-5-1`, `50.5`)
caught a simplified Meinhardt–Klingler denominator (`1.0 + kappa * a` instead of
`1 + kappa * a**2`), a missing basal activator term $\rho_0$, an ad-hoc Gaussian
blur in place of explicit finite-difference diffusion, and lack of parameter
sweeps or multi-seed distributional tests.

**Vogel golden-angle *Helianthus* phyllotaxis (`1979-vogel-phyllotaxis.py`, pad. 64.3 → 85.9 in 12-step dual-agent loop; impl. 84.7, bio. 84.8, repro. 88.7).**
Evaluated through a 12-step closed-loop remediation (`morphoeval/runs/2026-10-18-vogel-phyllotaxis-loop.yaml`,
`gemini-4-argon` + `claude-sonnet-5-5-high`): remediated from `64.3` to `85.9` by
coupling Vogel's kinematic model ($\theta_n = n\phi$, $r_n = c\sqrt{n}$) to a
dynamical Douady & Couder (1992) inhibitory-potential ODE solver that converges
de novo onto the golden angle ($137.48^\circ$, RMSE $0.68^\circ$), sweeping 5
divergence angles and plastochron ratios, and overlaying model primordia against
a *Helianthus annuus* capitulum specimen with $(34, 55)$ Fibonacci parastichy
counts ($R^2 = 0.9999$).

**Hoffmann–Donoughe wing venation (`2018-hoffmann-wing-venation.py`, pad. 77.8 primary / 81.3 2nd assessor; impl. 81.7, bio. 80.0, repro. 71.3).**
The highest-scoring `claude-fable-5-1`-audited simulation on the primary chain: the developmental model's
Voronoi / Lloyd relaxation and secondary-vein interpolation steps are
implemented as published, the model wing is placed beside a segmented real
*Perithemis tenera* hindwing, and vein-domain counts and circularities are
measured on both. Two caveats: (1) mechanistically, Voronoi sites are seeded
from distance-transform peaks inside the real wing's vein compartments rather
than generated de novo; (2) at checkpoint `f3fe478`, `scikit-image` was
undeclared, the environment unpinned, and the result was silently corrupted
under `numpy 2.2.6` on Python 3.14 (wing mask collapsed from $1{,}905{,}316$ px
to $25{,}146$ px) until `numpy` was upgraded to `2.5.4`.

**Thompson's *Scarus* → *Pomacanthus* on real species (`thompson/1917-thompson-scarus-pomacanthus.py`, pad. 70.2 → 76.6 primary / 90.8 blinded; impl. 70.3, bio. 77.3, repro. 82.0).**
Discussed in detail in the case study below: a high-scoring empirical **test**
whose finding is negative for Thompson's biological claim.

**Odonata wing cells through deep time (`2026-eshghi-odonata-wing-cells.py`, app. 78.9 / pad. 76.0 primary; app. 91.4 / pad. 87.3 blinded; impl. 63.3 [70.0 app.], bio. 85.0, repro. 78.0).**
An empirical scaling test (`kind: test`) rather than a generative simulation:
cell-size allometry in 69 fossil versus 394 extant Odonata wings. Strong on
biological validation (`bio. = 85.0` primary, `93.5` blinded) because every
fossil venation drawing in the authors' Zenodo archive is re-segmented from the
pixels and compared row by row with Table S1 ($0.0\%$ median cell-count error,
$3.4\%$ median cell-length difference), exposing two curatorial errors in the
published table. Because `parameter-expressivity` (weight 10) is flagged
`not_applicable`, baseline-50 padding shrinks the primary overall score from
`78.9` (`app.`) to `76.0` (`pad.`).

**Limbform limb morphologies (`2014-lobo-limbform-morphology.py`, pad. 71.3 primary / 83.6 blinded; impl. 81.7, bio. 61.3, repro. 74.3).**
Decodes the Lobo–Levin graph formalism (`kind: decode`) for 555 geometric limb
morphologies in LimbformDB 2.1.0 (verifying the joint-radius rule on $3{,}137$ of
$3{,}146$ joints) and renders wild-type limbs beside Fig. 1 of Lobo et al.
(2014b) (bounding-box-aligned silhouette IoU $0.41\text{--}0.74$). Model
implementation is high (`impl. = 81.7`), while biological validation (`bio. =
61.3`) is capped because comparison is to the source paper's software cartoons
rather than real limb specimens. The blinded second-assessor pass caught a
65-character typo in `DB_SHA256` on line 92.

**WormGUIDES *C. elegans* embryo atlas (`2015-santella-wormguides-celegans.py`, app. 69.7 → 72.4 / pad. 67.7 → 70.2 primary; app. 81.9 / pad. 78.7 2nd assessor; impl. 70.0 [80.0 app.], bio. 64.4, repro. 78.0).**
Decodes the 360-frame nuclear-position atlas (`kind: decode`), verifies the
Sulston lineage grammar on all $1{,}330$ cell names, re-derives the lineage tree
and places the OpenWorm adult connectome beside it. The re-score (`f39a409`)
anchored the spatial scale to the atlas's own eggshell meshes ($29.2\,\mu\text{m}$
unfitted width vs the $\sim 30\,\mu\text{m}$ egg). As an atlas replay,
`parameter-expressivity` is `not_applicable` (`app. = 72.4`, shrunk by
baseline-50 padding to `pad. = 70.2`).

**Planform planarian morphologies (`1900-planformDB_parser.py` 14.7 → `2013-lobo-planform-morphology.py` 66.5 primary / 76.5 2nd assessor; impl. 81.7, bio. 55.0, repro. 66.7).**
The superseded parser (`1900-planformDB_parser.py`, `14.7`) drew a convex hull of
organ landmarks, making every phenotype silhouette virtually identical (`IoU =
0.99`). The rewrite (`2013-lobo-planform-morphology.py`, `kind: decode`) decodes
the Lobo–Malone–Levin region graph across all $1{,}181$ regions in PlanformDB
2.5.0 (`impl. = 81.7`), yielding distinct silhouettes (`IoU = 0.89` wild-type vs
double-head, `0.61` vs headless; mean `IoU = 0.78` against the paper's Fig. 3 & 5
cartoons). Biological validation (`bio. = 55.0`) reflects that the targets are
Planform's cartoons rather than real planarian photographs.

**Raup shell coiling (`create_raup_animations.py`, pad. 37.3 → 64.1 primary / 79.3 2nd assessor; impl. 76.7, bio. 45.0, repro. 77.0).**
The initial audit (`37.3`) discovered that axial translation was scaled by
angle $\theta$ rather than whorl radius $r_c$. The remediated script (`ba6fd20`)
implements the Noshita (2014) / Okabe & Yoshimura (2017) parametric equations
and verifies exact parameter recovery ($W, D, T, S$ to $5 \times 10^{-15}$) in
`self_test()`, raising `impl.` to `76.7` and `repro.` to `77.0`. Biological
validation remains low (`bio. = 45.0` primary) because the six rendered panels
use illustrative `"cf."` parameter presets rather than measurements of real
mollusc shells.

**Cervera–Levin–Mafe bioelectric synthesis (`2021-Cervera–Levin–Mafe.py`, pad. 44.9 → 55.6 by `gpt-5.6-sol` / 60.7 by `gemini-4-argon`; impl. 49.3, bio. 44.3, repro. 76.9).**
Audited twice by an independent OpenAI assessor (`gpt-5.6-sol`) and once by
Google's `gemini-4-argon` (at commit `5659d5e`, following commit `0154143`).
Reproducibility and provenance are strong (`repro. = 76.9` primary, `81.6` 2nd
assessor), while model-implementation fidelity (`impl. = 49.3` primary, `54.0`
2nd) and biological validation (`bio. = 44.3` primary, `50.2` 2nd) reflect its
scope: the script is a qualitative 1D reaction–diffusion and 2-node bioelectric
analogue built without the full source paper text, its 1-parameter crop-position
fit has a $23.5$-percentage-point held-out RMSE on the RNAi series, and its
visual output paints 1D morphogen domains onto a static *S. mediterranea*
silhouette while switching categorically between two pre-drawn Planform cartoons.

**Chindapol flow-driven coral growth (`2013-chindapol-coral-flow.py`, pad. 55.5 primary / 56.0 2nd assessor; impl. 53.3, bio. 43.8, repro. 70.0).**
A documented heuristic stand-in (`impl. = 53.3`) that replaces the paper's
coupled Lattice–Boltzmann Navier–Stokes and advection–diffusion solver with
biased 2D diffusion-limited aggregation (DLA), and compares symmetry trends
qualitatively without colony specimen overlays (`bio. = 43.8`).

**Gray–Scott reaction–diffusion (`1952-turing-morpho.py`, app. 53.6 / pad. 52.7 primary; app. 37.8 / pad. 40.9 blinded; impl. 75.0, bio. 38.8 [20.0 app.], repro. 49.0).**
At the audited commit `f3fe478`, the script integrates the canonical Gray–Scott
PDE (`impl. = 75.0` primary), but performs no morphometric wavelength analysis
(`morphometric-validation = 20` primary, `5` blinded), has no biological
specimen (`specimen-match = 50*`, `not_applicable`), and lacks RNG seeding and
citations (`repro. = 49.0` primary, `35.0` blinded).

## Case study: Thompson's coaxial circles against 791 species

Thompson's method of transformations (*On Growth and Form*, 1917, ch. XVII)
was published as hand-drawn grids with no equations and no fitting procedure.
One of its claims can be tested with open data: that deforming the Cartesian
grid of a *Scarus* outline "into a system of (approximately) coaxial circles"
yields "a very good outline" of *Pomacanthus* (p. 749, Figs. 375–376). A
Cartesian grid whose lines become coaxial circles is the image of the plane
under a Möbius map $z \mapsto (az + b)/(cz + d)$, so the claim is a six-constant
hypothesis; because Thompson's drawn pencils meet obliquely, an eight/nine-constant
oblique coaxial family is fitted too.

Using the 17 lateral-view landmarks of Claverie & Wainwright (2014) for 791
Indo-Pacific shore fishes (released CC0 with Friedman et al. 2019), the
*Scarus* genus mean (22 species) was fitted to the *Pomacanthus* mean (9
species). The genus means are $0.0312$ Procrustes units apart. A plain affine
map explains $31\%$ of that distance; the conformal Möbius family explains
$0.8\%$ (a conformal map cannot change the depth/length ratio, which goes from
$0.38$ to $0.59$); the oblique coaxial family explains $32\%$, hardly more than
its affine part. The residual ($0.0256$) is twice the spread of the
*Pomacanthus* species about their own mean ($0.0132$); fitted to 16 landmarks,
the family predicts the 17th worse than no deformation at all ($0.0371$ LOO-CV
RMS); and among all $3{,}782$ ordered pairs of genus means in the dataset the
pair sits at the 31st percentile for how well coaxial circles describe it.

The re-score (`76.6` primary, `90.8` blinded) followed a second test on real
silhouettes: two openly licensed Wikimedia Commons photographs — a collected
*Scarus ghobban* on white (CC BY 4.0) and a living *Pomacanthus semicirculatus*
over sand (CC BY 2.0), SHA-256-pinned — are segmented, each silhouette is
registered to its own species' landmarks by two anatomical anchors (snout tip
and caudal-peduncle minimum width), and the parrotfish silhouette is carried by
each transformation family fitted to the landmarks. With no deformation at all
the two silhouettes overlap with $\text{IoU} = 0.662$; every landmark-fitted
family leaves the outline overlap lower — affine $0.652$, Möbius $0.645$,
Thompson's oblique coaxial circles $0.618$, and even the exact interpolating
thin-plate spline $0.629$. What the body landmarks ask of the trunk, the dorsal
and anal fin lobes do not follow. This case study illustrates why **the rubric's
implementation and validation scores measure the rigour of the computational test,
not the truth of the historical biological theory**.

## Dataset observations surfaced by auditing

Auditing a reproduction against its source data turns out to audit the data
too:

- *Eshghi, Buesch & Gorb (2026), Table S1.* Row #7 (*Euarchistigma
  peterknobli*, 24.7 mm, 282 cells) does not describe the drawing of that name
  (a 61 mm wing with 829 cells); rows #23 and #24 (*Paracordulagomphus
  aberrans*) appear swapped relative to their images.
- *WormGUIDES atlas.* One static eggshell mesh is reused for frames 1–~250 and
  the nuclei overrun it at the anterior pole by up to 19 units; per-frame
  segmentations begin around frame 280.
- *PlanformDB 2.5.0 & LimbformDB 2.1.0.* The region-shape encoding rule in
  PlanformDB (polar radii determined by link count) holds for all $1{,}181$
  regions, and the Limbform joint-radius rule holds for $3{,}137$ of $3{,}146$
  joints when ectopic sub-joints count their host segment twice.

# Limitations

**A pilot case study, not a calibrated general-purpose evaluator.** Most primary
audit files were written by the LLM agent that also wrote or refactored the
corresponding script (`claude-fable-5-1` on 9 initial scripts; `gemini-4-argon`
on `1984-wolfram-conus-textile-rule30.py`, `2005-runions-leaf-venation.py`, and
`1979-vogel-phyllotaxis.py`). Completing dual-agent scoring across all
{{N_SCRIPTS_ACTIVE}} active evaluated scripts — combining two independent passes
by `gpt-5.6-sol` (OpenAI) on `2021-Cervera–Levin–Mafe.py`, a 10-script
second-assessor pass by `gemini-4-argon` (Google), a strictly blinded
second-assessor pass by `claude-fable-5-1` (Anthropic) on
`1984-wolfram-conus-textile-rule30.py`, and 2 closed-loop dual-agent passes
(`gemini-4-argon` + `claude-sonnet-5-5-high`) on leaf venation and phyllotaxis —
ensures that no script relies solely on its builder grading itself, demonstrates
strong rank-order concordance on the 10-script batch ($\rho = 0.891$ across all
10 scripts, $\rho = 1.00$ on the initial 5-script blinded subset, and $\rho =
0.796$ across 70 dimension pairs; $\rho = 0.678$ across all 77 provider-paired
dimension pairs including *Conus*), and shows that independent assessors catch
genuine mathematical and code defects missed by the builder (most starkly the
$-36.8$-point blinded correction on `1984-wolfram-conus-textile-rule30.py`).
However, it also reveals a $10.4\text{--}13.8$-point mean absolute calibration
gap between uncalibrated zero-shot assessors. Turning MorphoEval into a
calibrated evaluator will require multi-rater human benchmarks and explicit
anchor examples for each score band.

**Uncalibrated scores in a probabilistic ontology slot.** While this report and
the rubric explicitly define MorphoEval ratings as uncalibrated $0\text{--}100$
scores and separate them from probabilities, the governed ProbKnow templates
(`RAku3DIN` and `RAIVOEuh`) store their unit-interval rescaling in
`pk:hasProbability` and `pk:weightOfEvidence`. Downstream consumers of the RDF
graph must inspect `dct:conformsTo` (`pkr:concept/morphoeval-rubric-0.1.0`) and
treat those literals as rescaled rubric scores rather than calibrated
probabilities (or build with `--omit-probability`), pending migration to the
ProbKnow v1.2 `pk:hasScore` (`[0, 100]`) / `RAOAK6Fe...` rubric-assessment
template ([`project-delphi-ontology#12`](https://github.com/fractastical/project-delphi-ontology/issues/12)).

**Rubric weights and N/A shrinkage.** Specimen match at 25 and morphometric
validation at 15 give biological validation 40% of the total weight. Where a
dimension is `not_applicable`, substituting 50 shrinks scores above 50 downward
(by $2.9$ points on Odonata and $2.2$ points on *C. elegans*); both the
applicable-only renormalised mean (`app.`) and the padded mean (`pad.`) should
therefore be consulted together.

**Incomplete coverage and post-audit commits.** Five repository scripts remain
unevaluated (`2001-bressloff-v1-form-constants.py`, `2010-petroff-stromatolite.py`,
`2024-kahng-coral-geometry.py`, `sorting-morphogenesis/`, and
`thompson/1917-thompson.py`), the `v1`/`v2` loop nanopublications for
`2005-runions-leaf-venation.py` and `1979-vogel-phyllotaxis.py` ({{N_PRIMARY_UNPUB_NANOPUBS}}
signed files) are signed locally awaiting production server minting, and later
commits on Hoffmann, Chindapol, Planform, Turing, and Cervera (`0154143`) have
not yet been re-audited into new production nanopublications.

# Reproducibility and availability

- **Code and audits:** https://github.com/fractastical/morpholib (this
  deposition archives the repository at revision `{{COMMIT}}`). Audits are
  under `morphoeval/audits/`, built graphs and signed nanopublications under
  `morphoeval/kg/`, and the rubric at `morphoeval/rubric.yaml` (with 20
  fine-grained sub-axes documented in `morphoeval/schema.yaml`).
- **Environment:** `requirements.lock.txt` pins the exact versions used across
  the audits (Python 3.14.7, numpy 2.5.4, scipy 1.16.3, matplotlib 3.10.7,
  scikit-image 0.26.0, pandas 2.3.3, Pillow 12.0.0).
- **Data:** External inputs are verified against recorded SHA-256 digests on
  load (`datasets/<domain>/README.md`).
- **Nanopublications:** {{N_NANOPUBS}} published on the production network
  ({{N_HEADS}} current heads; rubric nanopublication
  https://w3id.org/np/RAhBMChdpmgS_yBfy8cfYXpfKktqlUD72A9bxZzvWDkYM), plus
  {{N_BLINDED_NANOPUBS}} signed second-assessor nanopublications in
  `morphoeval/kg/2026-10-17-blinded-sample-audit/signed/` and
  `morphoeval/kg/2026-10-19-conus-second-assessor-audit/signed/` and
  {{N_PRIMARY_UNPUB_NANOPUBS}} signed loop nanopublications in
  `morphoeval/kg/2026-10-18-{phyllotaxis,venation}-audit*/signed/`; all URIs are
  recorded in `morphoeval/kg/<audit>/published.json`.
- **Tracking:** Remediation items are tracked at
  https://github.com/fractastical/morpholib/issues/1.

# References

- Bookstein FL (1978). *The Measurement of Biological Shape and Shape Change.* Lecture Notes in Biomathematics 24. Springer. doi:10.1007/978-3-642-93093-5
- Bookstein FL (1989). Principal warps: thin-plate splines and the decomposition of deformations. *IEEE Trans. Pattern Anal. Mach. Intell.* 11:567–585. doi:10.1109/34.24792
- Cervera J, Levin M, Mafe S (2021). Bioelectrical coupling of morphogen gradients: a computational study of planarian regeneration. *Bioelectrochemistry* 141:107886. doi:10.1016/j.bioelechem.2021.107886
- Chindapol N, Kaandorp JA, Cronemberger C, Mass T, Genin A (2013). Modelling growth and form of the scleractinian coral *Pocillopora verrucosa* and the influence of hydrodynamics. *PLoS Comput. Biol.* 9:e1002849. doi:10.1371/journal.pcbi.1002849
- Claverie T, Wainwright PC (2014). A morphospace for reef fishes: elongation is the dominant axis of body shape evolution. *PLoS ONE* 9:e112732. doi:10.1371/journal.pone.0112732
- Cook SJ, et al. (2019). Whole-animal connectomes of both *Caenorhabditis elegans* sexes. *Nature* 571:63–71. doi:10.1038/s41586-019-1352-7
- Dietz J (2026). BioEval — Bioinformatics Paper Evaluator. Zenodo. doi:10.5281/zenodo.20567719
- Dietz J (2026). NeuroEval: Evaluating Data Transparency, Reproducibility, and Neural-Coding Rigor in Computational and Biohybrid/Wetware Neuroscience Papers. Zenodo. doi:10.5281/zenodo.20622878
- Douady S, Couder Y (1992). Phyllotaxis as a physical self-organized growth process. *Phys. Rev. Lett.* 68:2098–2101. doi:10.1103/PhysRevLett.68.2098
- Eshghi S, Buesch A, Gorb S (2026). Supplementary material: Critical crack length as an evolutionary constant in Odonata wing venation: lessons from fossils. Zenodo. doi:10.5281/zenodo.19899648
- Friedman ST, Martinez CM, Price SA, Wainwright PC (2019). The influence of size on body shape diversification across Indo-Pacific shore fishes. *Evolution* 73:1873–1884. doi:10.1111/evo.13755; data: Dryad doi:10.5061/dryad.fn457gr
- Gray P, Scott SK (1984). Autocatalytic reactions in the isothermal, continuous stirred tank reactor. *Chem. Eng. Sci.* 39:1087–1097.
- Hoffmann J, Donoughe S, Li K, Salcedo MK, Rycroft CH (2018). A simple developmental model recapitulates complex insect wing venation patterns. *Proc. Natl Acad. Sci. USA* 115:9905–9910. doi:10.1073/pnas.1721248115
- Kuhn T, et al. (2016). Decentralized provenance-aware publishing with nanopublications. *PeerJ Comput. Sci.* 2:e78. doi:10.7717/peerj-cs.78
- Lobo D, Malone TJ, Levin M (2013). Planform: an application and database of graph-encoded planarian regenerative experiments. *Bioinformatics* 29:1098–1100. doi:10.1093/bioinformatics/btt088
- Lobo D, Feldman EB, Shah M, Malone TJ, Levin M (2014a). Limbform: a functional ontology-based database of limb regeneration experiments. *Bioinformatics* 30:3598–3600. doi:10.1093/bioinformatics/btu582
- Lobo D, Feldman EB, Shah M, Malone TJ, Levin M (2014b). A bioinformatics expert system linking functional data to anatomical outcomes in limb regeneration. *Regeneration* 1:37–56. doi:10.1002/reg2.13
- Meinhardt H, Klingler M (1987). A model for pattern formation on the shells of molluscs. *J. Theor. Biol.* 126:63–89. doi:10.1016/S0022-5193(87)80101-7
- Murray CD (1926). The physiological principle of minimum work. I. The vascular system and the cost of blood volume. *Proc. Natl Acad. Sci. USA* 12:207–214. doi:10.1073/pnas.12.3.207
- Noshita K (2014). Quantification and geometric analysis of coiling patterns in gastropod shells based on 3D and 2D image data. *J. Theor. Biol.* 363:93–104. doi:10.1016/j.jtbi.2014.08.010
- Okabe T, Yoshimura J (2017). Optimal designs of mollusk shells from bivalves to snails. *Sci. Rep.* 7:42445. doi:10.1038/srep42445
- Pearson JE (1993). Complex patterns in a simple system. *Science* 261:189–192. doi:10.1126/science.261.5118.189
- Price CA, Wing S, Weitz JS (2012). Scaling and structure of dicotyledonous leaf venation networks. *Ecol. Lett.* 15:87–95. doi:10.1111/j.1461-0248.2011.01712.x
- Raup DM (1966). Geometric analysis of shell coiling: general problems. *J. Paleontol.* 40:1178–1190. doi:10.2307/1301992
- Runions A, Fuhrer M, Lane B, Federl P, Rolland-Lagan A-G, Prusinkiewicz P (2005). Modeling and visualization of leaf venation patterns. *ACM Trans. Graph.* 24:702–711. doi:10.1145/1073204.1073251
- Santella A, et al. (2015). WormGUIDES: an interactive single cell developmental atlas and tool for collaborative multidimensional data exploration. *BMC Bioinformatics* 16:189. doi:10.1186/s12859-015-0627-8
- Sarma GP, et al. (2018). OpenWorm: overview and recent advances in integrative biological simulation of *Caenorhabditis elegans*. *Phil. Trans. R. Soc. B* 373:20170382. doi:10.1098/rstb.2017.0382
- Sulston JE, Schierenberg E, White JG, Thomson JN (1983). The embryonic cell lineage of the nematode *Caenorhabditis elegans*. *Dev. Biol.* 100:64–119. doi:10.1016/0012-1606(83)90201-4
- Thompson DW (1917). *On Growth and Form.* Cambridge University Press. Scan: https://archive.org/details/ongrowthform1917thom
- Turing AM (1952). The chemical basis of morphogenesis. *Phil. Trans. R. Soc. Lond. B* 237:37–72. doi:10.1098/rstb.1952.0012
- Vogel H (1979). A better way to construct the sunflower head. *Math. Biosci.* 44:179–189. doi:10.1016/0025-5564(79)90080-4
- Wang P, Berkovich A, Marom N, Buehler MJ (2026). Rethinking scientific publishing for the agentic era: Artifact-first living records via ScienceClaw and Infinite. *MRS Bull.* doi:10.1557/s43577-026-01180-1
- Wolfram S (1984). Cellular automata as models of complexity. *Nature* 311:419–424. doi:10.1038/311419a0
