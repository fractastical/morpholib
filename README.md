# Infinite Morphospace

A chronological collection of small, self-contained reproductions of published models of biological form — from D'Arcy Thompson's transformations to present-day developmental and palaeobiological datasets — plus an evaluation layer ([MorphoEval](#morphoeval-how-faithful-are-these-reproductions)) that scores how faithfully each reproduction follows its source and publishes the scores as nanopublications.

Each script is run directly (`python3 <script>.py`); rendered outputs (PNG/GIF) are written to the working directory and are git-ignored except for the three legacy figures embedded below. Setup: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt` (some scripts need `scikit-image`).

## Chronological examples

| year | model | script | MorphoEval |
|---|---|---|---|
| 1917 | Thompson, transformations | [`thompson/1917-thompson.py`](./thompson/1917-thompson.py) | — |
| 1952 | Turing / Gray–Scott reaction–diffusion | [`1952-turing-morpho.py`](./1952-turing-morpho.py) | 52.7 |
| 1966 | Raup, shell coiling | [`1966-raup.py`](./1966-raup.py), [`create_raup_animations.py`](./create_raup_animations.py) | 37.3 |
| 2010 | Petroff et al., conical stromatolites | [`2010-petroff-stromatolite.py`](./2010-petroff-stromatolite.py) | unevaluated |
| 2013 | Chindapol et al., flow-driven coral growth | [`2013-chindapol-coral-flow.py`](./2013-chindapol-coral-flow.py) | 55.5 |
| 2013 | Lobo, Malone & Levin, Planform planarian morphologies | [`2013-lobo-planform-morphology.py`](./2013-lobo-planform-morphology.py), [`1900-planformDB_parser.py`](./1900-planformDB_parser.py) | 66.5 |
| 2018 | Hoffmann et al., insect wing venation | [`2018-hoffmann-wing-venation.py`](./2018-hoffmann-wing-venation.py) | 77.8 |
| 2021 | Cervera, Levin & Mafe, bioelectric morphogen antagonism | [`2021-Cervera–Levin–Mafe.py`](./2021-Cervera–Levin–Mafe.py) | unevaluated |
| 2024 | Kahng et al., coral colony geometry | [`2024-kahng-coral-geometry.py`](./2024-kahng-coral-geometry.py) | unevaluated |
| 2026 | Eshghi et al., fossil vs living Odonata wing cells | [`2026-eshghi-odonata-wing-cells.py`](./2026-eshghi-odonata-wing-cells.py) | not yet audited |

### [thompson/1917-thompson.py](./thompson/1917-thompson.py)

A demonstration of D'Arcy Thompson's theory of transformations from *On Growth and Form* (1917), showing how one shape can be deformed into another through mathematical transformations. This classic work established the foundation for mathematical biology and the study of biological form.

Thompson-focused notes and dataset curation are in:
- [`thompson/DARCY_THOMPSON_FOCUSED_OPTIONS.md`](./thompson/DARCY_THOMPSON_FOCUSED_OPTIONS.md)
- [`thompson/MODERN_DATASETS_THOMPSON_ALIGNED.md`](./thompson/MODERN_DATASETS_THOMPSON_ALIGNED.md)
- [`thompson/generate_tps_transformation_panels.py`](./thompson/generate_tps_transformation_panels.py) and [`thompson/ingest_dryad_morphometrics.py`](./thompson/ingest_dryad_morphometrics.py) for thin-plate-spline panels from modern landmark data

### [1952-turing-morpho.py](./1952-turing-morpho.py)

A simulation of the Gray-Scott reaction-diffusion model, implementing Alan Turing's theory from "The Chemical Basis of Morphogenesis" (1952). This model generates classic Turing patterns like spots and stripes, demonstrating how simple chemical reactions can produce complex biological patterns.

![Turing patterns (Gray–Scott)](turing_patterns.gif)

Audit note (2026-10-13, 52.7): the integrator follows Pearson (1993) correctly, but the run is not placed on Pearson's (f, k) phase diagram and no pattern wavelength is measured, so the output is not tied to a specific published pattern class.

### [1966-raup.py](./1966-raup.py)

Implements David Raup's classic model of shell coiling from "Geometric Analysis of Shell Coiling" (1966). This parametric model generates a variety of 3D shell forms by controlling whorl expansion rate (W), distance from coiling axis (D), translation rate (T), and generating curve shape (S). The model demonstrates how three core parameters can generate the vast diversity of mollusk shell morphologies observed in nature.

![Raup shell coiling animation](raup_shell_animation.gif)

Run `python create_raup_animations.py` to generate:
- Animated GIF showing shells morphing through parameter space
- Parameter space visualization exploring W, D, and T
- Comparison with empirical shell forms (Nautilus, Turritella, Ammonite, etc.), using the published parameter values collected in [`datasets/raup/variables.json`](./datasets/raup/variables.json) (sources in [`datasets/raup/variable_extraction_log.md`](./datasets/raup/variable_extraction_log.md))

Audit note (2026-10-13, 37.3): the translation parameter T currently has no visible effect — the generating curve's radius grows exponentially but its axial translation is linear, so Turritella (T = 0.80) renders as the same planispiral disc as Nautilus (T = 0.10). Open remediation on [morpholib#1](https://github.com/fractastical/morpholib/issues/1).

### [2010-petroff-stromatolite.py](./2010-petroff-stromatolite.py)

Petroff et al.'s (2010) biophysics of conical stromatolites: (1) the spacing law ℓ = α√(Dτ), which makes a ~1 cm spacing between neighbouring cones a fingerprint of a ~daylength photosynthetic rhythm and packs cones onto a hexagonal lattice, and (2) a dome-vs-cone morphospace from the deterministic KPZ growth law ∂h/∂t = v + λ(1 + ½(∂h/∂r)²), where β = v/λ sets the aspect ratio. Output: `petroff_stromatolite_morphospace.png`.

**Reference:** Petroff AP, Sim MS, Maslov A, Krupenin M, Rothman DH, Bosak T (2010). Biophysical basis for the geometry of conical stromatolites. *PNAS* 107(22):9956–9961. [DOI](https://doi.org/10.1073/pnas.1001973107)

### [2013-chindapol-coral-flow.py](./2013-chindapol-coral-flow.py)

Chindapol et al.'s (2013) flow-driven growth of *Pocillopora verrucosa*. The paper couples polyp-based accretive growth to Navier–Stokes + advection–diffusion in 3D (COMSOL); this script makes a declared reduction to 2D diffusion-limited aggregation with drift, where the drift/diffusion ratio plays the role of the Péclet number, and recovers the paper's transition from symmetric branching (low Pe) to upstream-biased, compact colonies (high Pe). Outputs: `chindapol_coral_flow_morphospace.png` and `chindapol_coral_flow_morphospace_morphometrics.png` (branch-direction rose plots and surface/volume vs Pe).

Audit note (2026-10-13, 55.5): the reduction is explicit, and the paper's symmetry magnitude and branch alignment are computed, but they are not compared numerically with the paper's Figs 9–12.

**Reference:** Chindapol N, Kaandorp JA, Cronemberger C, Mass T, Genin A (2013). Modelling growth and form of the scleractinian coral *Pocillopora verrucosa* and the influence of hydrodynamics. *PLoS Comput Biol* 9(1):e1002849. [DOI](https://doi.org/10.1371/journal.pcbi.1002849)

### [2013-lobo-planform-morphology.py](./2013-lobo-planform-morphology.py)

Renders every planarian morphology in PlanformDB by decoding the Lobo–Malone–Levin graph formalism exactly. Region shapes are reconstructed from their encoded polar radii (0 links → right/anterior/left/posterior; 1 link → +90/+180/+270° from the edge; k ≥ 2 links → one radius per bisector of consecutive edges), region centres from the `RegionsLink` distance/angle labels, borders from `Ratio·Dist`, and eyes, brain lobes, pharynges and ventral nerve cords from the organ table. The decoding rule is checked at load time against all 1181 regions in planformDB 2.5.0. Rendering follows Planform's own cartoon convention (Fig. 1D of the Biology Open paper): control radii are joined by superellipse arcs that are rounded between free radii and near-square next to a border, giving one straight-sided body with flat region borders and rounded head/tail caps; head and tail colours fade towards the trunk, eyes are ovals with pupils, brain lobes and nerve cords are green, the pharynx is a tube. The formalism stores only three radii per head, so finer species-specific features (auricles, a triangular head) are not in the data and are not drawn.

```bash
export PLANFORM_DB_PATH=/path/to/planformDB_2.5.0.edb
python3 2013-lobo-planform-morphology.py                     # gallery PNG + morphospace timeline GIF
python3 2013-lobo-planform-morphology.py --ids 1 2 3 24 20   # specific morphologies
python3 2013-lobo-planform-morphology.py --csv morphospace.csv --no-anim
```

Outputs (git-ignored): `planform_morphology_gallery.png` (16 canonical phenotypes: wild type, double head, headless, cyclops, side heads, polarity reversed, …), `planform_morphospace_timeline.gif` (year-by-year first reports of each morphology against the cumulative catalogue, ending with all 359 dated morphologies as a morphospace grid), and an optional CSV of silhouette morphometrics (AP length, ML width, aspect, organ/region counts). The decoded wild type measures 557.6 × 119.8 Planform units (aspect 4.65) against 560 × 120 encoded in the database; 369 of the 372 morphology rows have geometry, all 369 are distinct encodings, and 167 have distinct body silhouettes (the rest differ only in organs).

Audit note (2026-10-14, 66.5, up from 14.7 for the earlier convex-hull animation): remaining gaps are the absence of a specimen-level comparison and of a recorded hash/version for the external database.

**References:**
- Lobo D, Malone TJ, Levin M (2013). Towards a bioinformatics of patterning: a computational approach to understanding regulative morphogenesis. *Biology Open* 2(2):156-169. [DOI](https://doi.org/10.1242/bio.20123400)
- Lobo D, Malone TJ, Levin M (2013). Planform: an application and database of graph-encoded planarian regenerative experiments. *Bioinformatics* 29(8):1098-1100. [DOI](https://doi.org/10.1093/bioinformatics/btt088)
- Lobo D, Vico FJ, Dassow J (2011). Graph grammars with string-regulated rewriting. *Theoretical Computer Science* 412(45):6101-6111. [DOI](https://doi.org/10.1016/j.tcs.2011.07.004)

See [`datasets/planform/paper_citations.bib`](./datasets/planform/paper_citations.bib) for BibTeX citations.

#### [1900-planformDB_parser.py](./1900-planformDB_parser.py)

The original PlanformDB reader. Reads a local PlanformDB SQLite/EDB file (`PLANFORM_DB_PATH` or a positional argument) and extracts yearly time series — experiments per year (via `Experiment → Publication.Year`), publications per year, and distinct morphologies observed per year — into a gap-filled `pandas.DataFrame` and the timeline plot below. It also produces `planform_morphology_animation.gif`, which approximates each morphology as a convex hull of organ landmarks; that rendering was audited at 14.7/100 and has been superseded by `2013-lobo-planform-morphology.py` above, so use the parser for the time series and the 2013 script for morphology. [`explore_planform_schema.py`](./explore_planform_schema.py) dumps the database schema.

![PlanformDB timeline: experiments, publications, and cumulative morphologies](parser_output.png)

### [2018-hoffmann-wing-venation.py](./2018-hoffmann-wing-venation.py)

Hoffmann, Donoughe, Li, Salcedo & Rycroft's (2018) developmental model of insect wing venation, run the way the authors ran it and on their own example wing (a hindwing of the dragonfly *Arigomphus villosipes*, downloaded from their repository and cached). The real wing is segmented into its silhouette and vein network; the primary vein scaffold is kept; the local cell size is read from the distance transform of the real veins; and the secondary veins are regenerated by the paper's mechanism — inhibitory centres scattered at the real local density, equilibrated by Lloyd's algorithm, then Voronoi-tessellated. Real and model wings are compared by domain circularity (c = 4πA/P²) and area distributions. Output: `hoffmann_wing_recapitulation.png`.

Audit note (2026-10-13, 77.8, the highest in the repo): 923 real vein domains reproduced; circularity real 0.712 vs model 0.867 (hexagonal ceiling 0.907). Needs `scikit-image` and `Pillow`.

**Reference:** Hoffmann J, Donoughe S, Li K, Salcedo MK, Rycroft CH (2018). A simple developmental model recapitulates complex insect wing venation patterns. *PNAS* 115(40):9905–9910. [DOI](https://doi.org/10.1073/pnas.1721248115). Code: [hoffmannjordan/insect-wing-venation-patterns](https://github.com/hoffmannjordan/insect-wing-venation-patterns).

### [2021-Cervera–Levin–Mafe.py](./2021-Cervera–Levin–Mafe.py)

A reaction-diffusion demo inspired by Cervera–Levin–Mafe (2021), exploring morphogen antagonism and its effect on pattern formation. The model includes:
- Antagonistic morphogens (m1, m2) on an antero-posterior axis with mutual annihilation
- Independent morphogen (m3) on a lateral axis
- Gap-junction blocking effects modeled through diffusion reduction
- 3D morphospace visualization of mean expressions

### [2024-kahng-coral-geometry.py](./2024-kahng-coral-geometry.py)

Kahng, Odle & Wakeman's (2024) geometry of reef-coral colonies. A colony lays down skeleton at a per-area calcification rate G, so volume grows as V(t+1) = V(t) + G·S(t)/ρ; whether linear extension and surface-area productivity are size-dependent is then decided entirely by how the surface-to-volume ratio changes with size. The script reconstructs these relationships for the paper's four canonical shapes (hemispheroid, fixed-aspect cone, flat disc of fixed thickness, cylinder of fixed branch diameter). Output: `kahng_coral_geometry_morphospace.png`.

**Reference:** Kahng SE, Odle E, Wakeman KC (2024). Coral geometry and why it matters. *PeerJ* 12:e17037. [DOI](https://doi.org/10.7717/peerj.17037). Code: [ericodle/coral_geometry_modeling](https://github.com/ericodle/coral_geometry_modeling).

### [2026-eshghi-odonata-wing-cells.py](./2026-eshghi-odonata-wing-cells.py)

Fossil and living dragonfly/damselfly wings compared on the one trait our venation model ([`2018-hoffmann-wing-venation.py`](./2018-hoffmann-wing-venation.py)) actually predicts: the size of the vein-bounded cells. Uses Eshghi, Buesch & Gorb's (2026) measurements of 69 fossil wings (22 extinct species, Solnhofen 147 Ma → Crato 116 Ma → Eocene 52 Ma) and 394 extant wings (204 species), with ages from the Paleobiology Database and Hoffmann et al.'s (2018) segmented North American wings as the extant counterpart (the two groups' independent segmentations agree to 3% on 42 shared species). Two families — Aeshnidae and Gomphidae — occur as Eocene fossils and as living species in the same tables.

```bash
python3 2026-eshghi-odonata-wing-cells.py --no-anim      # statistics, scaling figure, fossil re-segmentation
python3 2026-eshghi-odonata-wing-cells.py                # also the deep-time GIF (downloads ~100 MB once into .wing_cache/, SHA-256 checked)
python3 2026-eshghi-odonata-wing-cells.py --no-anim --no-resegment --no-crosscheck   # tables only, no downloads
```

The fossil half is not taken on trust: every one of the 69 fossil venation drawings in the archive is re-segmented from the pixels (vein ink skeletonised, enclosed regions = cells, scale read from each drawing's own scale bar) and compared wing by wing with Table S1. Cell counts agree to a median 0.0 % (90 % of wings within 2 %), median cell length to 3.4 %, largest cell to 0.9 %; the comparison also turns up one table row that does not match its drawing (#7 *Euarchistigma peterknobli*) and one swapped pair (#23/#24). Per-wing results are in [`datasets/odonata/fossil_wing_resegmentation.csv`](./datasets/odonata/fossil_wing_resegmentation.csv) and `odonata_fossil_resegmentation.png`.

Findings as printed by the script: the median cell length per wing stays in a narrow band across 150 Myr and a 6× range of wing length (fossils 0.78 mm, IQR 0.67–0.94; extant 0.90–0.99 mm) and rises only as L^0.2–0.3, while the largest cells scale almost with the wing (L^0.8); cell counts scale as N ∝ L^1.8 in fossils but N ∝ L^1.0–1.1 in extant wings, versus L² for a strictly constant cell size. For the model this means the inhibition length is approximately, but not strictly, conserved. Outputs (git-ignored): `odonata_wing_cells_scaling.png`, `odonata_fossil_resegmentation.png` and `odonata_wing_cells_deep_time.gif` (each fossil species' venation drawing at true scale against the PBDB Odonatoptera record, ending with the extant Aeshnidae/Gomphidae). Data and provenance: [`datasets/odonata/`](./datasets/odonata/).

**References:**
- Eshghi S, Buesch A, Gorb S (2026). Critical Crack Length as an Evolutionary Constant in Odonata Wing Venation: Lessons from Fossils — Supplementary Material. Zenodo. [DOI](https://doi.org/10.5281/zenodo.19899648) (CC BY 4.0)
- Hoffmann J, Donoughe S, Li K, Salcedo MK, Rycroft CH (2018). A simple developmental model recapitulates complex insect wing venation patterns. *PNAS* 115(40):9905–9910. [DOI](https://doi.org/10.1073/pnas.1721248115)
- Paleobiology Database, Odonatoptera occurrences (accessed 2026-10-14). https://paleobiodb.org

## MorphoEval: how faithful are these reproductions?

[`morphoeval/`](./morphoeval/) is a seven-dimension rubric (model fidelity, specimen match, parameter expressivity, morphometric validation, source traceability, code traceability, reproducibility package; weights sum to 100) applied to the scripts above, with every score backed by line-pinned evidence at a recorded commit and by measurements taken on the rendered outputs. Audits are YAML files under `morphoeval/audits/`, built into ProbKnow-conformant nanopublications (`pk:LLMClaimAssessment` per dimension, `schema:PropertyValue` measurement evidence), signed, verified independently, and published to the production nanopub network; re-scores are new nanopubs that `npx:supersedes` the old ones.

| simulation | checkpoint | faithfulness | reproducibility | overall |
|---|---|---:|---:|---:|
| Hoffmann–Donoughe wing venation | f3fe478, audit 2026-10-13 | 81.1 | 70.0 | **77.8** |
| Planform planarian morphology (`2013-lobo-planform-morphology.py`) | 161f269, audit 2026-10-14 | 66.1 | 67.3 | **66.5** |
| Chindapol coral flow | f3fe478, audit 2026-10-13 | 49.3 | 70.0 | **55.5** |
| Gray–Scott (Turing) | f3fe478, audit 2026-10-13 | 54.3 | 49.0 | **52.7** |
| Raup shell coiling | f3fe478, audit 2026-10-13 | 28.6 | 57.7 | **37.3** |
| Planform animation (`1900-planformDB_parser.py`), superseded | f3fe478, audit 2026-10-13 | 5.4 | 36.3 | ~~14.7~~ |
| Petroff stromatolite, Kahng coral geometry, Cervera–Levin–Mafe, Odonata wing cells | — | — | — | *unevaluated* |

55 nanopubs are on production (46 from 2026-10-13, 9 superseding ones from 2026-10-14); URIs are in `morphoeval/kg/<audit>/published.json`, the rubric nanopub is https://w3id.org/np/RAjd9dIu9uNVgvT4npxJgWwL6cbp8GCsy3gK86Qkj-iH4. Remediation is tracked in [morpholib#1](https://github.com/fractastical/morpholib/issues/1); the data-format review for the ProbKnow template is [project-delphi-ontology#12](https://github.com/fractastical/project-delphi-ontology/issues/12). MorphoEval is the simulation-faithfulness sibling of BioEval ([10.5281/zenodo.20567719](https://doi.org/10.5281/zenodo.20567719)) and NeuroEval ([10.5281/zenodo.20622878](https://doi.org/10.5281/zenodo.20622878)). Full workflow, tooling notes and the rubric text: [`morphoeval/README.md`](./morphoeval/README.md).

## Datasets

- [`datasets/odonata/`](./datasets/odonata/): Eshghi et al. 2026 wing-cell measurements (463 wings, with GBIF families and PBDB ages), the 22 fossil taxa's PBDB age records, and 1762 PBDB Odonatoptera occurrences; README with query URLs, licences and retrieval date.
- [`datasets/raup/`](./datasets/raup/): published Raup parameter values for empirical shells (`variables.json`), the extraction log, paper chronology and citations.
- [`datasets/planform/`](./datasets/planform/): BibTeX for the Planform papers. The PlanformDB `.edb` itself is not vendored; it ships with Planform.
- [`datasets/frog_embryo_data_processed/`](./datasets/frog_embryo_data_processed/): per-embryo masks (`*_mask.npz`, one folder per imaging zoom) used by the wave-vector pipeline's detection summary below.

## Analysis tools

### [Ca²⁺ Wave Vector Analysis](./wave-vector-analysis/)

A complete pipeline for detecting and tracking Ca²⁺ signaling waves in time-lapse microscopy images of embryos. Processes multi-page TIFF images to detect bright spark events, track them across frames, segment embryos, and generate comprehensive analysis outputs. Includes visualization tools and supports 16-bit scientific imaging data.

**See [`wave-vector-analysis/README.md`](./wave-vector-analysis/README.md) for full documentation.**

#### Visualization Example

Flow field visualization showing spatial vector fields of Ca²⁺ wave propagation:

![Ca²⁺ Wave Flow Field](wave-vector-analysis/analysis_results/artistic/flow_field_aurora.png)

*Spatial vector field showing wave propagation directions and speeds. This visualization complements standard analysis tools by providing spatial flow patterns not available in direction distribution plots.*

### [Nanopublications](./nanopubs/)

Nanopublication-related files including wave files (.trig format), hypothesis mappings, and scripts for processing and canonicalizing nanopublications. These tools enable structured representation and sharing of scientific findings in a machine-readable format. The MorphoEval nanopubs use their own signer and verifier (see [`morphoeval/README.md`](./morphoeval/README.md)) because nanopub-py 2.1 does not work on Python 3.14.

**See [`nanopubs/README.md`](./nanopubs/README.md) for details.**
