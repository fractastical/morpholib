# Infinite Morphospace

This repository presents a chronological exploration of morphogenesis and theoretical biology, from early mathematical models to modern computational approaches. The scripts demonstrate key concepts in reaction-diffusion systems, procedural shape generation, and classic models from the field.

## Chronological Examples

### [1900-planformDB_parser.py](./1900-planformDB_parser.py)

Utilities to read a local PlanformDB SQLite/EDB file and extract yearly time series:
- Experiments per year (via `Experiment → Publication.Year`)
- Publications per year
- Distinct morphologies observed per year (frequency > 0)
- Morphology shape extraction and visualization

Produces a merged, gap-filled `pandas.DataFrame` and a quick `matplotlib` plot. Set `PLANFORM_DB_PATH` environment variable to your `.edb` file path and run the script to print summary rows and show the timeline.

The parser produces:
- Time series plot showing experiments, publications, and cumulative morphologies over time
- Animated GIF showing new morphologies documented each year, highlighted and superimposed on the wild-type planarian base

![PlanformDB timeline: experiments, publications, and cumulative morphologies](parser_output.png)

**References:**
- Lobo et al. (2011). Graph grammars with string-regulated rewriting. *Theoretical Computer Science* 412(45):6101-6111. [DOI](https://www.sciencedirect.com/science/article/pii/S0304397511005925)
- Lobo et al. (2013). Planform: an application and database of graph-encoded planarian regenerative experiments. *Bioinformatics* 29(8):1098-1100. [DOI](https://doi.org/10.1093/bioinformatics/btt088)

See [`datasets/planform/paper_citations.bib`](./datasets/planform/paper_citations.bib) for BibTeX citations.

### [2013-lobo-planform-morphology.py](./2013-lobo-planform-morphology.py)

A fresh take on the PlanformDB morphology animation that decodes the Lobo–Malone–Levin graph formalism exactly instead of approximating each phenotype as a convex hull of organ landmarks (which made every morphology look alike). Region shapes are reconstructed from their encoded polar radii (0 links → right/anterior/left/posterior; 1 link → +90/+180/+270° from the edge; k ≥ 2 links → one radius per bisector of consecutive edges), region centres from the `RegionsLink` distance/angle labels, borders from `Ratio·Dist`, and eyes, brain lobes, pharynges and ventral nerve cords from the organ table. The decoding rule is checked at load time against all 1181 regions in planformDB 2.5.0. Rendering follows Planform's own cartoon convention (Fig. 1D of the Biology Open paper): control radii are joined by superellipse arcs that are rounded between free radii and near-square next to a border, which gives one straight-sided body with flat region borders and rounded head/tail caps; head and tail colours fade towards the trunk, eyes are ovals with pupils, brain lobes and nerve cords are green, the pharynx is a tube. Note the formalism stores only three radii per head, so finer species-specific features (auricles, a triangular head) are not in the data and are not drawn.

```bash
export PLANFORM_DB_PATH=/path/to/planformDB_2.5.0.edb
python3 2013-lobo-planform-morphology.py                     # gallery PNG + morphospace timeline GIF
python3 2013-lobo-planform-morphology.py --ids 1 2 3 24 20   # specific morphologies
python3 2013-lobo-planform-morphology.py --csv morphospace.csv --no-anim
```

Outputs (git-ignored): `planform_morphology_gallery.png` (16 canonical phenotypes: wild type, double head, headless, cyclops, side heads, polarity reversed, …), `planform_morphospace_timeline.gif` (year-by-year first reports of each morphology against the cumulative catalogue, ending with all 359 dated morphologies as a morphospace grid), and an optional CSV of silhouette morphometrics (AP length, ML width, aspect, organ/region counts; WT = 558 × 122 units, aspect 4.58).

**References:**
- Lobo D, Malone TJ, Levin M (2013). Towards a bioinformatics of patterning: a computational approach to understanding regulative morphogenesis. *Biology Open* 2(2):156-169. [DOI](https://doi.org/10.1242/bio.20123400)
- Lobo D, Malone TJ, Levin M (2013). Planform: an application and database of graph-encoded planarian regenerative experiments. *Bioinformatics* 29(8):1098-1100. [DOI](https://doi.org/10.1093/bioinformatics/btt088)

### [thompson/1917-thompson.py](./thompson/1917-thompson.py)

A demonstration of D'Arcy Thompson's theory of transformations from *On Growth and Form* (1917), showing how one shape can be deformed into another through mathematical transformations. This classic work established the foundation for mathematical biology and the study of biological form.

Thompson-focused notes and dataset curation are in:
- [`thompson/DARCY_THOMPSON_FOCUSED_OPTIONS.md`](./thompson/DARCY_THOMPSON_FOCUSED_OPTIONS.md)
- [`thompson/MODERN_DATASETS_THOMPSON_ALIGNED.md`](./thompson/MODERN_DATASETS_THOMPSON_ALIGNED.md)

### [1952-turing-morpho.py](./1952-turing-morpho.py)

A simulation of the Gray-Scott reaction-diffusion model, implementing Alan Turing's groundbreaking theory from "The Chemical Basis of Morphogenesis" (1952). This model generates classic Turing patterns like spots and stripes, demonstrating how simple chemical reactions can produce complex biological patterns.

![Turing patterns (Gray–Scott)](turing_patterns.gif)

### [1966-raup.py](./1966-raup.py)

Implements David Raup's classic model of shell coiling from "Geometric Analysis of Shell Coiling" (1966). This parametric model generates a variety of 3D shell forms by controlling whorl expansion rate (W), distance from coiling axis (D), translation rate (T), and generating curve shape (S). The model demonstrates how three core parameters can generate the vast diversity of mollusk shell morphologies observed in nature.

![Raup shell coiling animation](raup_shell_animation.gif)

Run `python create_raup_animations.py` to generate:
- Animated GIF showing shells morphing through parameter space
- Parameter space visualization exploring W, D, and T
- Comparison with empirical shell forms (Nautilus, Turritella, Ammonite, etc.)

### [2021-Cervera–Levin–Mafe.py](./2021-Cervera–Levin–Mafe.py)

A reaction-diffusion demo inspired by Cervera–Levin–Mafe (2021), exploring morphogen antagonism and its effect on pattern formation. The model includes:
- Antagonistic morphogens (m1, m2) on an antero-posterior axis with mutual annihilation
- Independent morphogen (m3) on a lateral axis
- Gap-junction blocking effects modeled through diffusion reduction
- 3D morphospace visualization of mean expressions

### [2026-eshghi-odonata-wing-cells.py](./2026-eshghi-odonata-wing-cells.py)

Fossil and living dragonfly/damselfly wings compared on the one trait our venation model ([`2018-hoffmann-wing-venation.py`](./2018-hoffmann-wing-venation.py)) actually predicts: the size of the vein-bounded cells. Uses Eshghi, Buesch & Gorb's (2026) measurements of 69 fossil wings (22 extinct species, Solnhofen 147 Ma → Crato 116 Ma → Eocene 52 Ma) and 394 extant wings (204 species), with ages from the Paleobiology Database and Hoffmann et al.'s (2018) segmented North American wings as the extant counterpart (the two groups' independent segmentations agree to 3% on 42 shared species). Two families — Aeshnidae and Gomphidae — occur as Eocene fossils and as living species in the same tables.

```bash
python3 2026-eshghi-odonata-wing-cells.py --no-anim      # statistics + figure from datasets/odonata (no downloads)
python3 2026-eshghi-odonata-wing-cells.py                # also the deep-time GIF (downloads ~100 MB once into .wing_cache/)
```

Findings as printed by the script: the median cell length per wing stays in a narrow band across 150 Myr and a 6× range of wing length (fossils 0.78 mm, IQR 0.67–0.94; extant 0.90–0.99 mm) and rises only as L^0.2–0.3, while the largest cells scale almost with the wing (L^0.8); cell counts scale as N ∝ L^1.8 in fossils but N ∝ L^1.0–1.1 in extant wings, versus L² for a strictly constant cell size. For the model this means the inhibition length is approximately, but not strictly, conserved. Outputs (git-ignored): `odonata_wing_cells_scaling.png` and `odonata_wing_cells_deep_time.gif` (each fossil species' venation drawing at true scale against the PBDB Odonatoptera record, ending with the extant Aeshnidae/Gomphidae). Data and provenance: [`datasets/odonata/`](./datasets/odonata/).

**References:**
- Eshghi S, Buesch A, Gorb S (2026). Critical Crack Length as an Evolutionary Constant in Odonata Wing Venation: Lessons from Fossils — Supplementary Material. Zenodo. [DOI](https://doi.org/10.5281/zenodo.19899648) (CC BY 4.0)
- Hoffmann J, Donoughe S, Li K, Salcedo MK, Rycroft CH (2018). A simple developmental model recapitulates complex insect wing venation patterns. *PNAS* 115(40):9905–9910. [DOI](https://doi.org/10.1073/pnas.1721248115)
- Paleobiology Database, Odonatoptera occurrences (accessed 2026-10-14). https://paleobiodb.org

## Analysis Tools

### [Ca²⁺ Wave Vector Analysis](./wave-vector-analysis/)

A complete pipeline for detecting and tracking Ca²⁺ signaling waves in time-lapse microscopy images of embryos. Processes multi-page TIFF images to detect bright spark events, track them across frames, segment embryos, and generate comprehensive analysis outputs. Includes visualization tools and supports 16-bit scientific imaging data.

**See [`wave-vector-analysis/README.md`](./wave-vector-analysis/README.md) for full documentation.**

#### Visualization Example

Flow field visualization showing spatial vector fields of Ca²⁺ wave propagation:

![Ca²⁺ Wave Flow Field](wave-vector-analysis/analysis_results/artistic/flow_field_aurora.png)

*Spatial vector field showing wave propagation directions and speeds. This visualization complements standard analysis tools by providing spatial flow patterns not available in direction distribution plots.*

### [Nanopublications](./nanopubs/)

Nanopublication-related files including wave files (.trig format), hypothesis mappings, and scripts for processing and canonicalizing nanopublications. These tools enable structured representation and sharing of scientific findings in a machine-readable format.

**See [`nanopubs/README.md`](./nanopubs/README.md) for details.**
