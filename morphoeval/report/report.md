---
title: "MorphoEval: Scoring the Faithfulness of Morphology Reproductions and Publishing the Scores as Nanopublications"
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
  traceability, reproducibility package) applied to each reproduction in the
  `morpholib` repository. Every score is backed by line-pinned evidence at a
  recorded commit and by measurements taken on the rendered output, and every
  audit is published as a set of ProbKnow-template-conformant
  nanopublications — one `pk:LLMClaimAssessment` per dimension and one for the
  overall verdict — signed, independently verified, and placed on the
  production nanopublication network, where re-scores supersede earlier ones.
  This report describes the rubric and workflow, presents the {{N_EVALUATED}}
  audits published to date ({{N_NANOPUBS}} nanopublications, {{N_HEADS}} current
  heads), and discusses what the audits found: two of the original animations
  were geometrically wrong and have been rewritten; auditing surfaced errors in
  two external datasets; and a direct test of Thompson's *Scarus* →
  *Pomacanthus* "coaxial circles" on 791 real species shows the transformation
  explains at most a third of the shape difference. MorphoEval is the
  simulation-faithfulness sibling of BioEval and NeuroEval. Its chief
  limitation is that all scores so far come from a single LLM assessor that
  also wrote most of the audited code; the graph is designed so that further
  assessors can add their own assessments of the same hypotheses.
---

# Introduction

The `morpholib` repository collects small, self-contained Python reproductions
of published models of biological form, from Thompson's 1917 transformations
and Turing's 1952 reaction–diffusion through Raup's shell morphospace (1966) to
graph-encoded planarian (Planform, 2013) and limb (Limbform, 2014)
morphologies, the WormGUIDES *C. elegans* embryo atlas (2015), a developmental
model of insect wing venation (2018) and a 2026 study of Odonata wing cells in
fossils. Each script writes a figure and an animation. The question MorphoEval
asks is simple: *how faithful is each of these to its source and to the
organism?* — and it insists that the answer be evidence-backed, versioned and
published in a form that other agents and people can inspect, dispute and
extend.

The project started from a failure. An audit of the three original animations
(October 2026) found that two of them did not depict biologically plausible
forms, in both cases for a code or geometry reason rather than a styling one:
the planarian animation drew the convex hull of sparse organ landmarks instead
of the body, so a double-headed, a headless and a wild-type worm were
indistinguishable; and the Raup shell animation had a translation term that
never took effect, so every shell was a flat planispiral. A legitimate-looking
GIF had hidden both. The remedy was not just to fix the scripts but to build an
evaluation layer that scores each reproduction against its source and against
specimens, records why, and makes regressions visible.

MorphoEval is the third of a family. BioEval (Zenodo
10.5281/zenodo.20567719) scores computational-biology *papers* for
reproducibility and information-theoretic rigour; NeuroEval
(10.5281/zenodo.20622878) does the same for computational and biohybrid
neuroscience. MorphoEval turns the same machinery inward and scores *our own
simulation code* for faithfulness, so that each script can be checkpointed and
re-scored after changes. The rules carried over from the siblings are: score
only what the evidence supports; a dimension that does not apply sits at the
neutral baseline and is flagged; and items that were not inspected are listed
as unevaluated with no scores at all.

# The rubric (version 0.1.0)

Seven dimensions in two groups, each scored 0–100; the overall score is the
weighted mean.

{{RUBRIC_TABLE}}

*Faithfulness* asks whether the code implements the published model as
published (not a loose analogue), whether the rendered output resembles real
specimens or the authors' own figures when placed side by side, whether each
morphospace parameter visibly and monotonically controls the feature it is
supposed to control, and whether quantitative morphometrics are computed on the
output and compared with values measured on specimens or reported in the
literature. *Reproducibility* asks whether the source paper, code and data are
resolvable identifiers with recorded provenance, whether each modelling step in
the code can be traced to a specific equation, section or figure of the source,
and whether the run is pinned, seeded, documented and free of
environment-induced corruption.

Specimen match carries the largest weight because it is the dimension the
original failures would have caught. Its consequence is that a model with no
organism — Gray–Scott at a single (f, k) point, for instance — cannot score
above the neutral 50 on a quarter of the rubric, which is intended: a pattern
generator that is not compared with any pattern in nature is, for this
purpose, unverified.

# Method

![How a form gets into the repository and how it is scored. Tier A, making a reproduction: the model as published (equations, an encoding, or a claim); the paper's own formal source (parameter values, database or atlas records, measurement tables), fetched or vendored with a pinned hash; the rules pinned down — inferred from prose, verified on every record, with the authors' code read to settle semantics where it exists; the script, which simulates, decodes or tests; and the comparison with the authors' figures or specimens. Tier B, the MorphoEval cycle: audit with line-pinned evidence at a commit, measurement of the rendered outputs, graph built against the ProbKnow templates, signing and publication through the test registry to production, the knowledge graph, and remediation, whose fixes produce a new commit and a re-audit of the touched dimensions that supersedes the earlier nanopublications.](pipeline.png){width=100%}

## Audits

An audit is a YAML file (`morphoeval/audits/<date>-<name>-audit.yaml`) that
records the rubric version, the assessor, the checkpoint (repository, commit,
Python and library versions, input provenance), the references used (DOIs or
URLs), and, per audited script, a summary, a score and a rationale for each
dimension, a list of evidence loci, and remediation items. Every evidence
locus is a file and a line range at the checkpoint commit, or a rendered output
(line range 0–0), or a line range in the measurement code. The builder
validates each audit against the rubric: weights, ranges, evidence files that
exist, references that resolve to entries.

## Measurements

Scores are judgements; the numbers behind them are not. A separate program,
`measure_outputs.py`, re-runs or re-reads each audited script's own functions
and takes morphometrics from the actual geometry or pixels of the outputs —
silhouette aspect ratios and intersections-over-union against the source
authors' figures, vein-domain counts and circularities, Procrustes residuals,
allometric exponents, dominant wavelengths. Each measurement carries its
method, unit, the reference value and source it is compared with, and the
output file's SHA-256. Outputs not on disk are reported as missing, never
invented. The sidecar (`<audit>.measurements.yaml`) is attached to the
nanopublications as `schema:PropertyValue` evidence.

## Scoring rules

- Every score needs at least one evidence locus.
- A dimension that does not apply (parameter expressivity for a data decoder
  with no morphospace parameters; specimen match for a pattern generator with
  no organism) is scored 50 and flagged `not_applicable`, so that it neither
  rewards nor penalises the overall.
- Scripts that were not inspected are listed as unevaluated and carry no
  nanopublications.
- Published nanopublications are immutable. A re-score after a code change is
  a new audit file with a new checkpoint commit; only the dimensions the change
  touched are re-scored, the others keep their rationale with updated line
  numbers, and each new nanopublication `npx:supersedes` the one it replaces
  after the current head of that version chain has been resolved on Nanopub
  Query.

## The assessor

All audits to date carry `assessor: claude-fable-5-1` — Claude Fable 5.1
operating as a Cursor agent, reading the code, running the scripts, inspecting
the outputs and writing the scores and rationales. The same agent wrote most
of the code it audited (everything from the planarian rewrite onward). Both
facts are recorded in the graph (`pk:assessedBy pkr:model/claude-fable-5-1`
on every assessment) and both are limitations, discussed below.

# The knowledge graph

Each audit is built into nanopublications expressed in the ProbKnow / Project
Delphi ontology (v1.1) and shaped by the governed templates of the ProbKnow
Space, so that every nanopublication links the template it was created from
and renders in Nanodash with that form:

- per script, a **`pk:Claim`** (`pkr:claim/morphoeval-<script>`) with label,
  description, context (scripts, output, organism, commit, non-DOI sources),
  `pk:derivedFromPaper` for each DOI, `pk:hasAssessment` to the eight
  assessments, and `pk:weightOfEvidence` = overall / 100;
- per dimension, a **`pk:LLMClaimAssessment`** evaluating the hypothesis
  "*script* satisfies the criterion *dimension*", with `pk:hasProbability` =
  score / 100, a verdict band, the rationale and evidence loci as
  commit-pinned GitHub line ranges, and `pk:evidenceFor` / `pk:evidenceAgainst`
  where the audit marks evidence as supporting or not;
- an **overall** assessment of the claim proposition, with the remediation
  items as `pk:evidenceAgainst`;
- the **rubric** as a `pk:Concept` with one `skos:narrower` concept per
  dimension (shared by all audits and emitted once).

Provenance follows the Space's "extracted from a paper by ProbKnow" template;
publication info records the creator, the licence (CC BY 4.0 for the audit
content), the template links and full timestamps. The pipeline is: build
(`build_morphoeval_kg.py`, with an offline conformance check against the
vendored ontology and templates) → sign and publish to the **test** registry
(`sign_morphoeval.py`) → verify every signed file independently with a second
implementation (`check_signed.mjs`, nanopub-js) → fetch each nanopublication
back from the registry → publish to **production**. Each step must pass before
the next runs, and the URIs and version histories are kept in
`morphoeval/kg/<audit>/published.json`.

Two things the graph is designed to carry that it does not yet: assessments
of the same hypotheses by other assessors (the assessor is an ordinary
`pk:Model` resource, so a second model or a human reviewer can publish
`pk:LLMClaimAssessment`s or human assessments of `pkr:hypothesis/morphoeval-<script>-<dimension>`
and the graph will hold both), and assessments of the *source* papers' claims
in the same Space, which would connect "is the reproduction faithful?" to "is
the model right?".

# Results

## Current scores

{{SCORE_TABLE}}

Scores are the current head of each script's version chain. Three scripts in
the repository (Petroff's stromatolite morphospace, Kahng's coral geometry,
the Cervera–Levin–Mafe bioelectric morphospace) and the schematic Thompson toy
have not been audited and carry no scores.

## Score history

{{HISTORY_TABLE}}

## What the audits found, script by script

**Hoffmann–Donoughe wing venation (77.8).** The strongest reproduction: the
developmental model's steps are implemented as published, the model wing is
placed beside a segmented real *Arigomphus villosipes* hindwing, and
vein-domain counts and circularities are measured on both. Held down by the
reproducibility package: `scikit-image` was undeclared, the environment
unpinned, and the result was silently wrong under numpy 2.2.6 on Python 3.14
until numpy was upgraded during the audit — the clearest case in the corpus of
environment-induced corruption. The dependency and pinning items have since
been remedied for the repository as a whole but not yet re-scored.

**Odonata wing cells through deep time (76.0).** A data analysis rather than a
simulation: cell-size scaling in fossil versus extant Odonata wings. Strong on
specimen match and morphometric validation because every fossil venation
drawing in the authors' Zenodo archive is re-segmented from the pixels and
compared row by row with their Table S1 — which is how two table errors were
found (below). Parameter expressivity is not applicable.

**Limbform limb morphologies (71.3).** Decodes the Lobo–Levin graph formalism
for limbs from LimbformDB and renders wild-type limbs for five species beside
the paper's own Fig. 1 panels, with silhouette IoU and aspect ratio measured
for each. Fidelity is high (the joint-radius rule is checked on every joint in the
database at load, 3137 of 3146 obey it and the nine under-parameterised
joints are padded with a stated rule); morphometric validation is the weak dimension because the
comparison is to the authors' drawings, not to specimens.

**WormGUIDES *C. elegans* embryo atlas (67.7 → 70.2).** Decodes the 360-frame
nuclear-position atlas, verifies the Sulston lineage grammar on all 1330
names, re-derives the lineage chart and places the OpenWorm connectome beside
it. The re-score followed the remediation item "verify the 50 µm calibration
against something in the atlas": the scale is now anchored to the atlas's own
eggshell meshes rather than to the nuclear cloud, with the shell's unfitted
width (29.2 µm against the ~30 µm egg) as a check. Open: no photograph or
WormGUIDES panel beside the output; a ~27-minute clock offset against
Sulston's schedule that is reported but not explained.

**Thompson's *Scarus* → *Pomacanthus* on real species (70.2).** Discussed as
a case study below.

**Planform planarian morphologies (14.7 → 66.5).** The original animation
scored 14.7: a convex hull of organ landmarks, every morphology identical. The
rewrite decodes the Lobo–Malone–Levin region graph exactly — polar radii from
link counts, centres from link distances and angles, borders from ratios — and
checks the decoding rule against all 1181 regions in PlanformDB 2.5.0 at load
time, then renders in Planform's own cartoon convention. Distinct morphologies
are now distinct (silhouette IoU of the wild type against the double-headed
form 0.89, against the headless form 0.61, where the old animation gave
0.99 between any two frames). The database file is not redistributable and is now identified by
SHA-256 at load.

**Chindapol flow-driven coral growth (55.5).** An honest, documented
stand-in rather than the published model: it does not solve the flow or the
nutrient field, and its Péclet number is a drift bias rather than *uL/D*
(model fidelity 40). The 2-D lattice aggregates branch with an upstream bias
but are not *Pocillopora*-like in form, and no colony is placed beside them.
Parameter expressivity is the strength: the flow parameter visibly changes
the branching.

**Gray–Scott reaction–diffusion (52.7).** The update is canonical and the
(f, k) point is in Pearson's spotted regime (model fidelity 85), but nothing is
measured (no wavelength, no placement on the phase diagram), the RNG is
unseeded so every run differs, and there is no organism, so specimen match sits
at the neutral 50. A low score for a correct demo — by design, since an
unchecked pattern generator is unverified.

**Raup shell coiling (37.3).** The translation parameter was applied to the
angle instead of the radius, so every shell was planispiral regardless of T;
the measured effective translation rate was an order of magnitude below
nominal for every named panel (0.0005–0.19 against nominal 0.1–1.2) because
translation was linear in the angle while the radius grew exponentially. Fixing
the geometry and comparing each panel to a photograph of its taxon are open
remediation items; the audit has not been re-scored.

## Case study: Thompson's coaxial circles against 791 species

Thompson's method of transformations (*On Growth and Form*, 1917, ch. XVII)
was published as hand-drawn grids with no equations and no fitting procedure.
One of its claims can be tested with open data: that deforming the Cartesian
grid of a *Scarus* outline "into a system of (approximately) coaxial circles"
yields "a very good outline" of *Pomacanthus* (p. 749, Figs. 375–376). A
Cartesian grid whose lines become coaxial circles is the image of the plane
under a Möbius map z → (az + b)/(cz + d), so the claim is a six-constant
hypothesis; because Thompson's drawn pencils meet obliquely, an eight-constant
family (an affine map followed by a Möbius map) is fitted too.

Using the 17 lateral-view landmarks of Claverie & Wainwright (2014) for 791
Indo-Pacific shore fishes (released CC0 with Friedman et al. 2019), the
*Scarus* genus mean (22 species) was fitted to the *Pomacanthus* mean (9
species). The genus means are 0.0312 Procrustes units apart. A plain affine map
explains 31 % of that distance; the conformal Möbius family explains 0.8 %
(a conformal map cannot change the depth/length ratio, which goes from 0.38 to
0.59); the oblique coaxial family explains 32 %, hardly more than its affine
part. The residual (0.0256) is twice the spread of the *Pomacanthus* species
about their own mean (0.0132); fitted to 16 landmarks, the family predicts the
17th worse than no deformation at all; and among all 3782 ordered pairs of
genus means in the dataset the pair sits at the 31st percentile for how well
coaxial circles describe it. What remains after the affine part is local — the
anal-fin end and origin, the upper-jaw tip, the dorsal-fin end — which is the
kind of change the method was built to exclude. Thompson's "very good outline"
holds at the resolution of his drawing, not of the species. The audit scores
the *test* (70.2: strong on morphometric validation and traceability, weaker
on specimen match because the fish are 17 landmarks rather than the outline
Thompson deformed), not Thompson.

## Dataset observations surfaced by auditing

Auditing a reproduction against its source data turns out to audit the data
too. Three observations, each recorded in the relevant audit and dataset
README:

- *Eshghi, Buesch & Gorb (2026), Table S1.* Row #7 (*Euarchistigma
  peterknobli*, 24.7 mm, 282 cells) does not describe the drawing of that name
  (a 61 mm wing with 829 cells); rows #23 and #24 (*Paracordulagomphus
  aberrans*) appear swapped relative to their images. A note to the authors is
  drafted.
- *WormGUIDES atlas.* One static eggshell mesh is reused for frames 1–~250 and
  the nuclei overrun it at the anterior pole by up to 19 units; it is not the
  shell of this egg. Per-frame segmentations begin around frame 280.
- *PlanformDB 2.5.0.* The region-shape encoding rule (polar radii determined
  by link count) is not stated in the Planform paper in that form; it was
  inferred and holds for all 1181 regions.

# Limitations

**One assessor, and a conflicted one.** Every score comes from a single LLM
assessor, so there is no inter-rater agreement to report, and that assessor
wrote most of the code it scored. The rationales and the line-pinned evidence
are there so that a reader can check each judgement; the measurement sidecars
are computed, not judged. The graph is built to receive assessments of the same
hypotheses from other models and from human reviewers, and doing so is the
most important next step.

**Rubric weights are a choice.** Specimen match at 25 encodes a view about
what matters for morphology reproductions. A different weighting would reorder
the middle of the table (Chindapol, Turing) but not its ends.

**Not all scripts are audited, and not all audits are current.** Three
scripts are unevaluated. The Raup script's geometry bug is diagnosed but not
fixed; the cross-cutting remediation (pinned environment, declared
dependencies) is in the repository but will reach the scores only when each
script is re-audited.

**Specimen comparisons are mostly to authors' figures.** Limbform and
Planform are compared with the source papers' drawings, *C. elegans* with the
atlas's own segmentation, Thompson with landmarks. Only the two wing scripts
are compared with photographs or drawings of real specimens.

# Reproducibility and availability

- **Code and audits:** https://github.com/fractastical/morpholib (this
  deposition archives the repository at revision `{{COMMIT}}`). Audits are
  under `morphoeval/audits/`, built graphs and signed nanopublications under
  `morphoeval/kg/`, the rubric at `morphoeval/rubric.yaml`.
- **Environment:** `requirements.lock.txt` pins the versions the audits were
  run with (Python 3.14.7, numpy 2.5.4, scipy 1.16.3, matplotlib 3.10.7,
  scikit-image 0.26.0); it installs into a fresh virtual environment and
  reproduces the Thompson numbers.
- **Data:** every external input is fetched by the script that uses it and
  verified against a recorded SHA-256 (WormGUIDES: 429 files; Eshghi archive;
  Thompson's scanned page), or vendored with its hash when the licence permits
  (the CC0 landmark table); the non-redistributable PlanformDB is identified by
  hash at load.
- **Nanopublications:** {{N_NANOPUBS}} on the production network, {{N_HEADS}}
  current heads; rubric nanopublication
  https://w3id.org/np/RAhBMChdpmgS_yBfy8cfYXpfKktqlUD72A9bxZzvWDkYM; all URIs
  in `morphoeval/kg/<audit>/published.json`.
- **Tracking:** remediation items are tracked at
  https://github.com/fractastical/morpholib/issues/1.

# References

- Bookstein FL (1978). *The Measurement of Biological Shape and Shape Change.* Lecture Notes in Biomathematics 24. Springer. doi:10.1007/978-3-642-93093-5
- Bookstein FL (1989). Principal warps: thin-plate splines and the decomposition of deformations. *IEEE Trans. Pattern Anal. Mach. Intell.* 11:567–585. doi:10.1109/34.24792
- Chindapol N, Kaandorp JA, Cronemberger C, Mass T, Genin A (2013). Modelling growth and form of the scleractinian coral *Pocillopora verrucosa* and the influence of hydrodynamics. *PLoS Comput. Biol.* 9:e1002849.
- Claverie T, Wainwright PC (2014). A morphospace for reef fishes: elongation is the dominant axis of body shape evolution. *PLoS ONE* 9:e112732. doi:10.1371/journal.pone.0112732
- Cook SJ, et al. (2019). Whole-animal connectomes of both *Caenorhabditis elegans* sexes. *Nature* 571:63–71. doi:10.1038/s41586-019-1352-7
- Dietz J (2026). BioEval — Bioinformatics Paper Evaluator. Zenodo. doi:10.5281/zenodo.20567719
- Dietz J (2026). NeuroEval: Evaluating Data Transparency, Reproducibility, and Neural-Coding Rigor in Computational and Biohybrid/Wetware Neuroscience Papers. Zenodo. doi:10.5281/zenodo.20622878
- Eshghi S, Buesch A, Gorb S (2026). Supplementary material: Critical crack length as an evolutionary constant in Odonata wing venation: lessons from fossils. Zenodo. doi:10.5281/zenodo.19899648
- Friedman ST, Martinez CM, Price SA, Wainwright PC (2019). The influence of size on body shape diversification across Indo-Pacific shore fishes. *Evolution* 73:1873–1884. doi:10.1111/evo.13755; data: Dryad doi:10.5061/dryad.fn457gr
- Gray P, Scott SK (1984). Autocatalytic reactions in the isothermal, continuous stirred tank reactor. *Chem. Eng. Sci.* 39:1087–1097.
- Hoffmann J, Donoughe S, Li K, Salcedo MK, Rycroft CH (2018). A simple developmental model recapitulates complex insect wing venation patterns. *Proc. Natl Acad. Sci. USA* 115:9905–9910. doi:10.1073/pnas.1721248115
- Kuhn T, et al. (2016). Decentralized provenance-aware publishing with nanopublications. *PeerJ Comput. Sci.* 2:e78. doi:10.7717/peerj-cs.78
- Lobo D, Malone TJ, Levin M (2013). Planform: an application and database of graph-encoded planarian regenerative experiments. *Bioinformatics* 29:1098–1100. doi:10.1093/bioinformatics/btt088
- Lobo D, Feldman EB, Shah M, Malone TJ, Levin M (2014). Limbform: a functional ontology-based database of limb regeneration experiments. *Bioinformatics* 30:3598–3600. doi:10.1093/bioinformatics/btu582
- Pearson JE (1993). Complex patterns in a simple system. *Science* 261:189–192. doi:10.1126/science.261.5118.189
- Raup DM (1966). Geometric analysis of shell coiling: general problems. *J. Paleontol.* 40:1178–1190.
- Santella A, et al. (2015). WormGUIDES: an interactive single cell developmental atlas and tool for collaborative multidimensional data exploration. *BMC Bioinformatics* 16:189. doi:10.1186/s12859-015-0627-8
- Sarma GP, et al. (2018). OpenWorm: overview and recent advances in integrative biological simulation of *Caenorhabditis elegans*. *Phil. Trans. R. Soc. B* 373:20170382. doi:10.1098/rstb.2017.0382
- Sulston JE, Schierenberg E, White JG, Thomson JN (1983). The embryonic cell lineage of the nematode *Caenorhabditis elegans*. *Dev. Biol.* 100:64–119. doi:10.1016/0012-1606(83)90201-4
- Thompson DW (1917). *On Growth and Form.* Cambridge University Press. Scan: https://archive.org/details/ongrowthform1917thom
- Turing AM (1952). The chemical basis of morphogenesis. *Phil. Trans. R. Soc. Lond. B* 237:37–72. doi:10.1098/rstb.1952.0012
