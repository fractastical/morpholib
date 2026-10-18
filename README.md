# Infinite Morphospace

A collection of small, self-contained reproductions of published models of biological form — from D'Arcy Thompson's transformations to present-day developmental and palaeobiological datasets — plus an evaluation layer ([MorphoEval](#morphoeval-how-faithful-are-these-reproductions)) that scores how faithfully each reproduction follows its source and publishes the scores as nanopublications.

## How a body gets its form

Morphogenesis is how a living thing gets its shape, and regeneration is how it gets it back: cells divide, move and signal until a body plan appears, or reappears after a cut. The classic models of form reproduced here are geometric ([Thompson's transformations](#thompson1917-thompsonpy), [Raup's coiling shells](#1966-rauppy)), chemical ([Turing's reacting and diffusing morphogens](#1952-turing-morphopy)) or physical ([coral](#2013-chindapol-coral-flowpy) and [stromatolite](#2010-petroff-stromatolitepy) growth shaped by flow and light). The newer idea most of the rest turn on is that the pattern is also stored and read out *bioelectrically*: every cell holds a resting voltage across its membrane, neighbouring cells share it through gap junctions, and the resulting voltage map across a tissue tells it where a head, a tail or a limb belongs. Change the voltage and the form changes — block the junctions and a planarian regrows two heads. That is the experiment the [Planform](#2013-lobo-planform-morphologypy) and [Limbform](#2014-lobo-limbform-morphologypy) databases record, hundreds of variants at a time, as formal graphs of what was cut, what was blocked and what grew back, and that the [Cervera–Levin–Mafe](#2021-cervera–levin–mafepy) model puts into equations. The repository turns such papers into running programs whose output *is* the form; MorphoEval then scores how faithfully each program follows its paper.

![The reproductions, with their MorphoEval scores](./docs/media/all_forms_montage.gif)

*Every animated form in the repository, 1917–2026 (the Thompson Scarus → Pomacanthus fit is a still and is represented by its score), each animation playing once per loop; the last tile is the MorphoEval overall score. Made by [`docs/media/make_montage.py`](./docs/media/make_montage.py) from the per-script animations below.*

![How a reproduction is made and how MorphoEval scores it](./docs/media/morphoeval_pipeline.gif)

*From paper to reproduction to signed evaluation — the MorphoEval pipeline ([animated GIF](./docs/media/morphoeval_pipeline.gif) · [high-res PNG](./docs/media/morphoeval_pipeline.png) · [section below](#morphoeval-how-faithful-are-these-reproductions)). Tier A shows how each of the 14 reproductions is built from its primary paper and formal data; Box 6 plays the 14 reproductions' own animations side by side; Tier B shows the dual-agent MorphoEval audit cycle across Anthropic (`claude-fable-5-1`), Google (`gemini-4-argon`), and OpenAI (`gpt-5.6-sol`).*

Each script is run directly (`python3 <script>.py`); rendered outputs (PNG/GIF) are written to the working directory and are git-ignored. Every entry below has a small animation of the morphology it produces (in [`docs/media/`](./docs/media/), downsized copies of the scripts' own GIF output; [`docs/media/README.md`](./docs/media/README.md) records how each was made). Setup: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt` (lower bounds; `requirements.lock.txt` has the exact versions the MorphoEval audits were run with — Python 3.14.7, numpy 2.5.4, scipy 1.16.3, matplotlib 3.10.7, scikit-image 0.26.0 — and installs cleanly into a fresh venv).

## Index, by year

| year | model | script | MorphoEval (primary / 2nd assessor) |
|---|---|---|---|
| 1917 | Thompson, transformations: *Scarus* → *Pomacanthus* on real landmarks and outlines | [`thompson/1917-thompson-scarus-pomacanthus.py`](./thompson/1917-thompson-scarus-pomacanthus.py) | 76.6 / 90.8 |
| 1917 | Thompson, transformations on a real fish | [`thompson/1917-thompson.py`](./thompson/1917-thompson.py) | — |
| 1952 | Turing / Gray–Scott reaction–diffusion | [`1952-turing-morpho.py`](./1952-turing-morpho.py) | 52.7 / 40.9 |
| 1966 | Raup, shell coiling | [`1966-raup.py`](./1966-raup.py), [`create_raup_animations.py`](./create_raup_animations.py) | 64.1 / 79.3 |
| 2001 | Bressloff et al., V1 Euclidean symmetry-breaking & retino-cortical form constants | [`2001-bressloff-v1-form-constants.py`](./2001-bressloff-v1-form-constants.py) | unevaluated |
| 2010 | Petroff et al., conical stromatolites | [`2010-petroff-stromatolite.py`](./2010-petroff-stromatolite.py) | unevaluated |
| 2013 | Chindapol et al., flow-driven coral growth | [`2013-chindapol-coral-flow.py`](./2013-chindapol-coral-flow.py) | 55.5 / 56.0 |
| 2013 | Lobo, Malone & Levin, Planform planarian morphologies | [`2013-lobo-planform-morphology.py`](./2013-lobo-planform-morphology.py), [`1900-planformDB_parser.py`](./1900-planformDB_parser.py) | 66.5 / 76.5 |
| 2014 | Lobo et al., Limbform limb morphologies (salamander, frog, crustacean, insect, arachnid) | [`2014-lobo-limbform-morphology.py`](./2014-lobo-limbform-morphology.py) | 71.3 / 83.6 |
| 2015 | Santella et al., WormGUIDES *C. elegans* embryo atlas + OpenWorm connectome | [`2015-santella-wormguides-celegans.py`](./2015-santella-wormguides-celegans.py) | 70.2 / 78.7 |
| 2018 | Hoffmann et al., insect wing venation | [`2018-hoffmann-wing-venation.py`](./2018-hoffmann-wing-venation.py) | 77.8 / 81.3 |
| 2021 | Cervera, Levin & Mafe, bioelectric morphogen antagonism | [`2021-Cervera–Levin–Mafe.py`](./2021-Cervera–Levin–Mafe.py) | 55.6 / 60.7 |
| 2024 | Kahng et al., coral colony geometry | [`2024-kahng-coral-geometry.py`](./2024-kahng-coral-geometry.py) | unevaluated |
| 2025 | Zhang, Goldstein & Levin, autonomous cell-view sorting morphogenesis & chimeric arrays | [`sorting-morphogenesis/`](./sorting-morphogenesis/) | unevaluated |
| 2026 | Eshghi et al., fossil vs living Odonata wing cells | [`2026-eshghi-odonata-wing-cells.py`](./2026-eshghi-odonata-wing-cells.py) | 76.0 / 87.3 |

## The reproductions, best-scoring first

Sections are ordered by MorphoEval overall score (the table under [MorphoEval](#morphoeval-how-faithful-are-these-reproductions) gives the breakdown), with the six unevaluated scripts after them in year order. A high score means the reproduction is close to its source and checked against it, not that the underlying model is better biology.

### [2018-hoffmann-wing-venation.py](./2018-hoffmann-wing-venation.py)

Hoffmann, Donoughe, Li, Salcedo & Rycroft's (2018) developmental model of insect wing venation, run the way the authors ran it and on their own example wing (a hindwing of the dragonfly *Arigomphus villosipes*, downloaded from their repository and cached). The real wing is segmented into its silhouette and vein network; the primary vein scaffold is kept; the local cell size is read from the distance transform of the real veins; and the secondary veins are regenerated by the paper's mechanism — inhibitory centres scattered at the real local density, equilibrated by Lloyd's algorithm, then Voronoi-tessellated. Real and model wings are compared by domain circularity (c = 4πA/P²) and area distributions. Output: `hoffmann_wing_recapitulation.png`; `--gif PATH` adds the animation below.

![Hoffmann wing venation: inhibitory centres equilibrating by Lloyd's algorithm on the real wing](docs/media/hoffmann_wing_lloyd.gif)

*What you see:* top, the real *Arigomphus villosipes* hindwing with its veins segmented; bottom, the model built on the same wing — the real primary veins are kept, and the secondary veins are the boundaries between inhibitory centres that start scattered at the real local density (iteration 0, ragged cells) and move to the centroid of their own cell at each Lloyd iteration until the pattern settles into the rounded, roughly hexagonal cells of a real wing (cell colour = circularity).

Audit note (2026-10-13, 77.8, the highest in the repo): 923 real vein domains reproduced; circularity real 0.712 vs model 0.867 (hexagonal ceiling 0.907). Needs `scikit-image` and `Pillow`.

**Reference:** Hoffmann J, Donoughe S, Li K, Salcedo MK, Rycroft CH (2018). A simple developmental model recapitulates complex insect wing venation patterns. *PNAS* 115(40):9905–9910. [DOI](https://doi.org/10.1073/pnas.1721248115). Code: [hoffmannjordan/insect-wing-venation-patterns](https://github.com/hoffmannjordan/insect-wing-venation-patterns).

### [2026-eshghi-odonata-wing-cells.py](./2026-eshghi-odonata-wing-cells.py)

Fossil and living dragonfly/damselfly wings compared on the one trait our venation model ([`2018-hoffmann-wing-venation.py`](./2018-hoffmann-wing-venation.py)) actually predicts: the size of the vein-bounded cells. Uses Eshghi, Buesch & Gorb's (2026) measurements of 69 fossil wings (22 extinct species, Solnhofen 147 Ma → Crato 116 Ma → Eocene 52 Ma) and 394 extant wings (204 species), with ages from the Paleobiology Database and Hoffmann et al.'s (2018) segmented North American wings as the extant counterpart (the two groups' independent segmentations agree to 3% on 42 shared species). Two families — Aeshnidae and Gomphidae — occur as Eocene fossils and as living species in the same tables.

```bash
python3 2026-eshghi-odonata-wing-cells.py --no-anim      # statistics, scaling figure, fossil re-segmentation
python3 2026-eshghi-odonata-wing-cells.py                # also the deep-time GIF (downloads ~100 MB once into .wing_cache/, SHA-256 checked)
python3 2026-eshghi-odonata-wing-cells.py --no-anim --no-resegment --no-crosscheck   # tables only, no downloads
```

The fossil half is not taken on trust: every one of the 69 fossil venation drawings in the archive is re-segmented from the pixels (vein ink skeletonised, enclosed regions = cells, scale read from each drawing's own scale bar) and compared wing by wing with Table S1. Cell counts agree to a median 0.0 % (90 % of wings within 2 %), median cell length to 3.4 %, largest cell to 0.9 %; the comparison also turns up one table row that does not match its drawing (#7 *Euarchistigma peterknobli*) and one swapped pair (#23/#24). Per-wing results are in [`datasets/odonata/fossil_wing_resegmentation.csv`](./datasets/odonata/fossil_wing_resegmentation.csv) and `odonata_fossil_resegmentation.png`.

Audit note (2026-10-14, 76.0): parameter expressivity is not applicable to a data analysis and is scored neutral; the strength is morphometric validation (85), the gaps are that the Hoffmann model is tested but not run here and scikit-image is not in `requirements.txt`.

![Odonata wing cells through deep time: each fossil species' venation drawing at true scale against the PBDB record](docs/media/odonata_wing_cells_deep_time.gif)

*What you see:* one fossil species per frame, oldest first (Solnhofen 147 Ma → Crato 116 Ma → Eocene 52 Ma): top, the Odonatoptera fossil record from the Paleobiology Database with the species' age marked; middle, its venation drawing from Eshghi et al.'s archive at true scale; bottom, the measured cell lengths of its wing against the 0.5–1.5 mm band that almost all cells, fossil or living, fall into. The closing frames show the living Aeshnidae and Gomphidae for comparison.

Findings as printed by the script: the median cell length per wing stays in a narrow band across 150 Myr and a 6× range of wing length (fossils 0.78 mm, IQR 0.67–0.94; extant 0.90–0.99 mm) and rises only as L^0.2–0.3, while the largest cells scale almost with the wing (L^0.8); cell counts scale as N ∝ L^1.8 in fossils but N ∝ L^1.0–1.1 in extant wings, versus L² for a strictly constant cell size. For the model this means the inhibition length is approximately, but not strictly, conserved. Outputs (git-ignored): `odonata_wing_cells_scaling.png`, `odonata_fossil_resegmentation.png` and `odonata_wing_cells_deep_time.gif` (each fossil species' venation drawing at true scale against the PBDB Odonatoptera record, ending with the extant Aeshnidae/Gomphidae). Data and provenance: [`datasets/odonata/`](./datasets/odonata/).

**References:**
- Eshghi S, Buesch A, Gorb S (2026). Critical Crack Length as an Evolutionary Constant in Odonata Wing Venation: Lessons from Fossils — Supplementary Material. Zenodo. [DOI](https://doi.org/10.5281/zenodo.19899648) (CC BY 4.0)
- Hoffmann J, Donoughe S, Li K, Salcedo MK, Rycroft CH (2018). A simple developmental model recapitulates complex insect wing venation patterns. *PNAS* 115(40):9905–9910. [DOI](https://doi.org/10.1073/pnas.1721248115)
- Paleobiology Database, Odonatoptera occurrences (accessed 2026-10-14). https://paleobiodb.org

### [2014-lobo-limbform-morphology.py](./2014-lobo-limbform-morphology.py)

The limb sibling of the Planform decoder ([`2013-lobo-planform-morphology.py`](#2013-lobo-planform-morphologypy), below). Limbform (Lobo, Feldman, Shah, Malone & Levin 2014) is the same lab's formal database of limb-regeneration experiments: 565 limb morphologies of salamanders (axolotl, newts), frogs (*Xenopus*), crustaceans (fiddler crab, crayfish, *Cancer* crabs), insects (cockroaches, cricket) and a spider, each encoded as a labelled graph, and Figure 1 of the *Regeneration* paper shows photograph, graph and software-drawn cartoon side by side for one species per category. This script reproduces the cartoon column for every morphology from the database alone. In Limbform's formalism limb segments are the graph *edges* (`Link`: type, length `Dist`, direction `Ang`, number of long bones), joints are the *vertices* carrying border radii (one per bisector of consecutive segments, or three at +90/+180/+270° for a terminal joint — the same rule as Planform's regions, satisfied by 3137 of the 3146 joints in LimbformDB 2.1.0 and checked at load time), sub-joints sit on a segment for grafts and ectopic limbs, and digits hang off a hand or foot segment with their own length, direction and phalanx count. The decoder lays the graph out from the labels, walks once around the limb tree collecting the joint radii in order into one closed smoothed outline, and colours it as in the paper's cartoons (proximal red → distal blue; arthropod coxa green and basis/trochanter yellow), drawing bones and digits for the tetrapods. 555 of the 565 rows have geometry; the other ten are outcome labels ("Lethal", "No limb", "Molted"). Non-tree encodings fall back to flat per-segment polygons.

The LimbformDB file (SQLite, 897 024 bytes, SHA-256 `9c26bf39…099f0`) is not vendored; `--download` fetches it from the Lobo lab into the git-ignored `.limbform_cache/` and checks the hash.

```bash
python3 2014-lobo-limbform-morphology.py --download               # gallery PNG + morphospace timeline GIF
python3 2014-lobo-limbform-morphology.py --ids 1 280 314 490 491  # the five Fig. 1 wild types
python3 2014-lobo-limbform-morphology.py --csv limb_morphospace.csv --no-anim
```

Outputs (git-ignored): `limbform_wildtype_gallery.png` (the Fig. 1 row — axolotl forelimb, *Xenopus* hindlimb, fiddler-crab walking leg, cricket T3 leg, huntsman-spider leg — plus ten further wild types: salamander hindlimb and stage-41 larva, frog forelimb and stage-51 hindlimb, *Cancer* claw, crayfish walking leg, fiddler-crab claw, cockroach legs, spider pedipalp), `limbform_morphospace_timeline.gif` (year-by-year first reports 1948–2013 of the 498 morphologies that appear as experimental results, ending with one morphospace grid per category), and an optional CSV of per-morphology measurements (segment and digit counts, proximodistal path length, bounding box, silhouette area, first-report year and species).

![Limbform morphospace timeline: limb morphologies decoded from LimbformDB by year of first report](docs/media/limbform_morphospace_timeline.gif)

*What you see:* each frame shows the limb morphologies first reported in one year (1948–2013), drawn from the database's graph encoding: one smoothed outline per limb coloured proximal red → distal blue, with the arthropod coxa in green and basis/trochanter in yellow, long bones in black and digits in blue for the tetrapods. Normal limbs and experimental outcomes (double limbs, grafts, missing segments) appear as they were published; the closing frames arrange every category's morphologies as grids.

MorphoEval audit `2026-10-14-limbform-audit` at commit `747c57f`: **71.3** (model-fidelity 85, specimen-match 65, parameter-expressivity 75, morphometric-validation 55, source-traceability 85, code-traceability 75, repro-package 65). Against the paper's own Fig. 1 cartoons the five wild types have identical segment and digit counts and pose, but bounding-box-aligned silhouette IoU is only 0.41–0.74 (the frog thigh–shank angle differs most), so either the database was revised after the figure or an angle convention is not captured; that and a comparison with the photographs rather than the cartoons are the open remediation items.

**References:**
- Lobo D, Feldman EB, Shah M, Malone TJ, Levin M (2014). Limbform: a functional ontology-based database of limb regeneration experiments. *Bioinformatics* 30(24):3598-3600. [DOI](https://doi.org/10.1093/bioinformatics/btu582)
- Lobo D, Feldman EB, Shah M, Malone TJ, Levin M (2014). A bioinformatics expert system linking functional data to anatomical outcomes in limb regeneration. *Regeneration* 1(2):37-56. [DOI](https://doi.org/10.1002/reg2.13)

### [2015-santella-wormguides-celegans.py](./2015-santella-wormguides-celegans.py)

The nematode member of the four-views family (photograph, schematic, formal description, software model — the layout the Planform and Limbform decoders below use). For *C. elegans* the photograph/EM layer (WormAtlas), the schematic (Sulston's lineage chart), a formal layer and a running software model (OpenWorm/Sibernetic) all exist, but in separate projects, and the formal layer that is usually meant — the NeuroML connectome — is connectivity, not anatomy. The one formal description of embryonic *geometry* is the WormGUIDES atlas (Santella et al. 2015): the position, diameter and lineage name of every nucleus at every minute of a diSPIM recording (image series DISPIM_061416, strain JIM113, 360 frames), plus segmented single-cell meshes for a subset of cells. This script downloads those files from a pinned commit of the WormGUIDES repository (SHA-256 manifest in [`datasets/celegans/`](./datasets/celegans/)), decodes the StarryNite tables, checks that the 1330 names form the Sulston tree (all parse; every cell appears after its parent), re-derives the lineage chart from the names alone, draws the nuclei to scale at six stages and the cell meshes at one, and puts the OpenWorm c302 connectome (Cook et al. 2019 edge list, 300 neurons) beside them as the connectivity layer. The atlas coordinates carry no unit, so they are calibrated by setting the atlas's own eggshell outline mesh to the 50 µm egg length (Gönczy et al. 1999) — one assumption, stated on the figure; the shell's width (29 µm, not fitted) and the spread of the per-frame shell meshes (±3 %) are reported as checks.

```bash
python3 2015-santella-wormguides-celegans.py --download        # ~430 small files into the git-ignored .celegans_cache/
python3 2015-santella-wormguides-celegans.py --csv celegans_embryo_timeseries.csv
```

![C. elegans embryo: every nucleus of the WormGUIDES atlas, coloured by founder lineage, over 360 minutes](docs/media/celegans_embryo_timeline.gif)

*What you see:* left, the embryo seen from above inside the atlas's eggshell outline, with every nucleus drawn at its recorded position and diameter (scale: the shell is 50 µm long) and coloured by the founder cell it descends from — AB blue, MS orange, E green (the future gut, which sinks inward at gastrulation), C purple, D yellow, P4 red (the germ line); right, the running cell count stacked by lineage, from 4 cells at minute 20 to ~580 at minute 379, with the red line marking the frame shown.

Outputs (git-ignored): `celegans_four_views.png` (lineage chart; nuclei by founder lineage at 4, 13, 28, 87, 360 and 580 cells; 51 cell meshes at frame 330; connectome adjacency matrix), `celegans_lineage_chart.png`, `celegans_embryo_timeline.gif` (120 frames: nuclei plus the cell count stacked by founder), and an optional per-frame CSV. Against Sulston's schedule as quoted by WormAtlas the cell-stage anchors agree (113 listed deaths, all in the movie; 562–571 nuclei at 330–350 min against ~550; gastrulation when the E cells ingress at 26 cells against the ~30-cell stage) but the clock does not: the atlas reaches gastrulation at movie-minute 73 against 100 min after first cleavage, a discrepancy the script reports and does not resolve. Details and sources: [`datasets/celegans/README.md`](./datasets/celegans/README.md).

MorphoEval audit `2026-10-16-celegans-audit` at commit `303a87e`: **70.2** (model-fidelity 80, specimen-match 62, parameter-expressivity 50 — not applicable to a data decoder, scored neutral —, morphometric-validation 72, source-traceability 85, code-traceability 75, repro-package 70), superseding the 2026-10-15 audit (67.7 at `2864784`). The re-score follows the calibration change: the scale is now anchored to the atlas's own eggshell mesh (0.1649 µm per unit; the nuclear-cloud estimate was 0.1617) with the shell width and the per-frame mesh spread as unfitted checks, and the audit records a dataset observation — the atlas reuses one static shell mesh for frames 1–~250 that the nuclei overrun by up to 19 units. Still open: no photograph or WormGUIDES panel beside the output, the 50 µm length itself (the acquisition voxel size is not in the repository), and the clock offset against Sulston's schedule. Overall claim: https://w3id.org/np/RAqDf8b5w5WNe3Chr8OWp5uUyU8U1Bz5BrbGdamyd4BOU (supersedes RASUT7ta…).

**References:**
- Santella A, Catena R, Kovacevic I, Shah P, Yu Z, Marquina-Solis J, Kumar A, Wu Y, Schaff J, Colón-Ramos D, Shroff H, Mohler WA, Bao Z (2015). WormGUIDES: an interactive single cell developmental atlas and tool for collaborative multidimensional data exploration. *BMC Bioinformatics* 16:189. [DOI](https://doi.org/10.1186/s12859-015-0627-8). Data: [zhirongbaolab/WormGUIDES](https://github.com/zhirongbaolab/WormGUIDES) (GPL-3.0).
- Sarma GP, et al. (2018). OpenWorm: overview and recent advances in integrative biological simulation of *Caenorhabditis elegans*. *Phil Trans R Soc B* 373:20170382. [DOI](https://doi.org/10.1098/rstb.2017.0382). Connectome: [openworm/c302](https://github.com/openworm/c302) (MIT), from Cook SJ, et al. (2019) *Nature* 571:63–71.
- Sulston JE, Schierenberg E, White JG, Thomson JN (1983). The embryonic cell lineage of the nematode *Caenorhabditis elegans*. *Dev Biol* 100:64–119. [DOI](https://doi.org/10.1016/0012-1606(83)90201-4)

### [thompson/1917-thompson-scarus-pomacanthus.py](./thompson/1917-thompson-scarus-pomacanthus.py)

Thompson's *Scarus* → *Pomacanthus* transformation (*On Growth and Form* 1917, p. 749, Figs. 375–376: "deform its rectilinear co-ordinates into a system of (approximately) coaxial circles … we obtain a very good outline of an allied fish") tested on real species rather than on his drawings. Thompson drew the grids by hand and left no way to fit them; a Cartesian grid whose lines become two pencils of coaxial circles is the image of the plane under a Möbius map, so his claim for this pair is a six-constant hypothesis that can be fitted by least squares to homologous landmarks. The script uses the 17 lateral-view landmarks of Claverie & Wainwright (2014) for 791 Indo-Pacific shore fishes (Friedman et al. 2019, Dryad, CC0, vendored and hash-checked in [`datasets/thompson/`](./datasets/thompson/)) — 22 *Scarus*, 9 *Pomacanthus* and 54 *Chaetodon* species — Procrustes-superimposes them, and fits the *Scarus* genus mean to the *Pomacanthus* genus mean with a similarity, an affine map, the Möbius map (isogonal coaxial circles), a Möbius∘affine map (oblique coaxial circles, which is what Fig. 376 is as drawn) and the exact thin-plate spline, with the spread of the species about their own genus means as the noise floor and all 3782 ordered pairs of genus means as context. Thompson's own figure is cropped from the public-domain Internet Archive scan (page 749 as the original JPEG-2000 leaf, hash-pinned) and placed beside the data.

```bash
python3 thompson/1917-thompson-scarus-pomacanthus.py                 # figures + GIF (fetches the 1917 page and two photographs on first run)
python3 thompson/1917-thompson-scarus-pomacanthus.py --no-scan --no-photos --csv fits.csv   # offline: landmarks only
```

![Scarus to Pomacanthus on real landmarks: Thompson's coaxial-circle grid fitted to the genus means beside the thin-plate-spline grid the data demand](docs/media/thompson_scarus_pomacanthus.gif)

*What you see:* the *Scarus* genus mean (17 landmarks joined into a body polygon, snout at the left) carried towards the *Pomacanthus* mean. Left, the best grid of Thompson's family — Cartesian lines bending into coaxial circles — and the fish it carries; at t = 1 the real *Pomacanthus* mean is drawn in dark red over it and the gap between them is what his transformation leaves unexplained. Right, the thin-plate spline, which fits the landmarks exactly: the grid the data actually demand, with the posterior body stretched dorso-ventrally and the fins displaced locally.

The result goes against the quotation. The genus means are 0.0312 Procrustes units apart; an affine map explains 31 % of that distance, the Möbius map 1 % (a conformal map cannot deepen the body, and *Pomacanthus* is 56 % deeper relative to its length by landmark extents, 0.38 → 0.59; the affine fit stretches by a singular-value ratio of 1.27), oblique coaxial circles 32 % — hardly more than the affine part — and the residual of 0.0256 is twice the spread of the *Pomacanthus* species about their own mean (0.0132). Fitted to 16 landmarks, the 8-parameter family predicts the 17th worse than a plain similarity does (0.0371 vs 0.0312). Among all genus pairs in the dataset, *Scarus* → *Pomacanthus* is at the 31st percentile for how much coaxial circles explain, so it is not even a favourable case; for *Scarus* → the 54 *Chaetodon* species the family explains a median 35 %. What remains is local — the anal-fin end, snout, dorsal-fin end and anal-fin origin carry the largest residuals — which is the kind of change Thompson's method was built to exclude. Caveats in [`datasets/thompson/README.md`](./datasets/thompson/README.md): fin-insertion landmarks rather than his body outline, genus means for his unnamed species, and two readings of "approximately coaxial circles". Of his four fish pairs this is the only one the dataset covers; *Diodon* is there but *Mola* is not.

![Scarus to Pomacanthus on real outlines: the silhouette of a Scarus ghobban photograph carried by each landmark-fitted map onto the silhouette of a Pomacanthus semicirculatus photograph](docs/media/thompson_scarus_pomacanthus_outlines.png)

*On real outlines (added 2026-10-17):* Thompson deformed an outline, not landmarks, so the script also fetches two Wikimedia Commons photographs — a collected *Scarus ghobban* on white (J. Huet, IFREMER, CC BY 4.0) and a living *Pomacanthus semicirculatus* over sand (E. Levy, CC BY 2.0), hash-pinned, not vendored — segments their silhouettes, registers each to its own species' 17 landmarks by two anchors the silhouette gives (snout, caudal peduncle), fits the five maps to the landmarks of *S. ghobban* → *P. semicirculatus* and carries the parrotfish silhouette with each (row F). With no deformation at all the two silhouettes overlap with IoU 0.662; every map read off the landmarks leaves the outline no better — affine 0.652, Möbius 0.645, Thompson's oblique coaxial circles 0.618, and even the thin-plate spline that matches the 17 landmarks exactly 0.629 (mean boundary distance 5.6 % of body length for the similarity, 7.0 % for the coaxial circles). What the landmarks ask of the body the fins do not follow: the angelfish's soft dorsal and anal lobes trail behind the peduncle, which no smooth map of the parrotfish produces. One individual per species, a specimen against a living fish, and a segmentation inside a declared box are the caveats; `--no-photos` skips this part. Numbers in `thompson_outline_metrics.csv`.

MorphoEval audit `2026-10-17-thompson-audit` at commit `232e8f8`: **76.6** (model-fidelity 78, specimen-match 72, parameter-expressivity 55, morphometric-validation 86, source-traceability 92, code-traceability 80, repro-package 75), a re-score of `2026-10-16-thompson-audit` (70.2 at `3cf7f8c`: 72 / 55 / 55 / 82 / 90 / 78 / 75) after the real-outline test was added. The score is for the test, not for Thompson: the fitters are checked on synthetic maps before the real fit, every number is taken on real specimens with a noise floor and a held-out check, the inputs are hash-pinned, and the outline Thompson actually deformed is now carried and compared with a real one against an explicit no-deformation baseline. It is held down because there is one photographed individual per species in different postures (a collected specimen against a living fish), because the field silhouette depends on a declared box and shadow cut and the landmarks are placed on the photographs by two anchors rather than digitised on them, because his drawn grid is shown beside the fitted one rather than digitised and compared constant by constant, and because the constants are not swept one at a time. Overall claim: https://w3id.org/np/RAYFKAfyOnqIKXWzdEzCN_3zyX2KX_ijCXNX-ZVtBPeP8 (supersedes https://w3id.org/np/RAjJx_KHiAAaPSxjsoYOJaFSU1JYmDUswVaRLmka2NIM0).

**References:**
- Thompson DW (1917). *On Growth and Form*. Cambridge University Press, ch. XVII, pp. 748–751. Scan: [archive.org/details/ongrowthform1917thom](https://archive.org/details/ongrowthform1917thom) (public domain).
- Friedman ST, Martinez CM, Price SA, Wainwright PC (2019). The influence of size on body shape diversification across Indo-Pacific shore fishes. *Evolution* 73:1873–1884. [DOI](https://doi.org/10.1111/evo.13755). Data: Dryad [10.5061/dryad.fn457gr](https://doi.org/10.5061/dryad.fn457gr) (CC0).
- Photographs: Huet J / IFREMER, [*Scarus ghobban*](https://commons.wikimedia.org/wiki/File:Perroquet_barbe_bleue_(Scarus_ghobban)_(Ifremer_00762-87435).jpg) (CC BY 4.0); Levy E, [*Pomacanthus semicirculatus*](https://commons.wikimedia.org/wiki/File:Semicircle_angelfish_(Pomacanthus_semicirculatus).jpg) (CC BY 2.0); both via Wikimedia Commons, SHA-256 in [`datasets/thompson/README.md`](./datasets/thompson/README.md).
- Claverie T, Wainwright PC (2014). A morphospace for reef fishes: elongation is the dominant axis of body shape evolution. *PLoS ONE* 9(11):e112732. [DOI](https://doi.org/10.1371/journal.pone.0112732) (landmark definitions, Fig. 1).
- Bookstein FL (1989). Principal warps: thin-plate splines and the decomposition of deformations. *IEEE Trans. Pattern Anal. Mach. Intell.* 11:567–585. [DOI](https://doi.org/10.1109/34.24792)

### [2013-lobo-planform-morphology.py](./2013-lobo-planform-morphology.py)

Renders every planarian morphology in PlanformDB by decoding the Lobo–Malone–Levin graph formalism exactly. Region shapes are reconstructed from their encoded polar radii (0 links → right/anterior/left/posterior; 1 link → +90/+180/+270° from the edge; k ≥ 2 links → one radius per bisector of consecutive edges), region centres from the `RegionsLink` distance/angle labels, borders from `Ratio·Dist`, and eyes, brain lobes, pharynges and ventral nerve cords from the organ table. The decoding rule is checked at load time against all 1181 regions in planformDB 2.5.0. Rendering follows Planform's own cartoon convention (Fig. 1D of the Biology Open paper): control radii are joined by superellipse arcs that are rounded between free radii and near-square next to a border, giving one straight-sided body with flat region borders and rounded head/tail caps; head and tail colours fade towards the trunk, eyes are ovals with pupils, brain lobes and nerve cords are green, the pharynx is a tube. The formalism stores only three radii per head, so finer species-specific features (auricles, a triangular head) are not in the data and are not drawn.

```bash
export PLANFORM_DB_PATH=/path/to/planformDB_2.5.0.edb
python3 2013-lobo-planform-morphology.py                     # gallery PNG + morphospace timeline GIF
python3 2013-lobo-planform-morphology.py --ids 1 2 3 24 20   # specific morphologies
python3 2013-lobo-planform-morphology.py --csv morphospace.csv --no-anim
```

Outputs (git-ignored): `planform_morphology_gallery.png` (16 canonical phenotypes: wild type, double head, headless, cyclops, side heads, polarity reversed, …), `planform_morphospace_timeline.gif` (year-by-year first reports of each morphology against the cumulative catalogue, ending with all 359 dated morphologies as a morphospace grid), and an optional CSV of silhouette morphometrics (AP length, ML width, aspect, organ/region counts). The decoded wild type measures 557.6 × 119.8 Planform units (aspect 4.65) against 560 × 120 encoded in the database; 369 of the 372 morphology rows have geometry, all 369 are distinct encodings, and 167 have distinct body silhouettes (the rest differ only in organs).

![Planform morphospace timeline: planarian morphologies decoded from PlanformDB by year of first report](docs/media/planform_morphospace_timeline.gif)

*What you see:* the left panel counts the morphologies in PlanformDB by the year each was first reported (1898–2013); the right panels draw the morphologies first reported in that year, decoded from the database alone — body regions as smoothed outlines with head (red) and tail (blue) colours, eyes, brain lobes, nerve cords (green) and pharynges. The last frames lay out all 359 dated morphologies as one grid.

Audit note (2026-10-14, 66.5, up from 14.7 for the earlier convex-hull animation): remaining gaps are the absence of a specimen-level comparison and of a recorded hash/version for the external database.

**References:**
- Lobo D, Malone TJ, Levin M (2013). Towards a bioinformatics of patterning: a computational approach to understanding regulative morphogenesis. *Biology Open* 2(2):156-169. [DOI](https://doi.org/10.1242/bio.20123400)
- Lobo D, Malone TJ, Levin M (2013). Planform: an application and database of graph-encoded planarian regenerative experiments. *Bioinformatics* 29(8):1098-1100. [DOI](https://doi.org/10.1093/bioinformatics/btt088)
- Lobo D, Vico FJ, Dassow J (2011). Graph grammars with string-regulated rewriting. *Theoretical Computer Science* 412(45):6101-6111. [DOI](https://doi.org/10.1016/j.tcs.2011.07.004)

See [`datasets/planform/paper_citations.bib`](./datasets/planform/paper_citations.bib) for BibTeX citations.

#### [1900-planformDB_parser.py](./1900-planformDB_parser.py)

The original PlanformDB reader. Reads a local PlanformDB SQLite/EDB file (`PLANFORM_DB_PATH` or a positional argument) and extracts yearly time series — experiments per year (via `Experiment → Publication.Year`), publications per year, and distinct morphologies observed per year — into a gap-filled `pandas.DataFrame` and the timeline plot below. It also produces `planform_morphology_animation.gif`, which approximates each morphology as a convex hull of organ landmarks; that rendering was audited at 14.7/100 and has been superseded by `2013-lobo-planform-morphology.py` (this section), so use the parser for the time series and the 2013 script for morphology. [`explore_planform_schema.py`](./explore_planform_schema.py) dumps the database schema.

![PlanformDB timeline: experiments, publications, and cumulative morphologies](parser_output.png)

### [2013-chindapol-coral-flow.py](./2013-chindapol-coral-flow.py)

Chindapol et al.'s (2013) flow-driven growth of *Pocillopora verrucosa*. The paper couples polyp-based accretive growth to Navier–Stokes + advection–diffusion in 3D (COMSOL); this script makes a declared reduction to 2D diffusion-limited aggregation with drift, where the drift/diffusion ratio plays the role of the Péclet number, and recovers the paper's transition from symmetric branching (low Pe) to upstream-biased, compact colonies (high Pe). Outputs: `chindapol_coral_flow_morphospace.png` and `chindapol_coral_flow_morphospace_morphometrics.png` (branch-direction rose plots and surface/volume vs Pe); `--gif PATH` adds the growth replay below.

![Chindapol coral growth replayed polyp by polyp under no, moderate and strong flow](docs/media/chindapol_coral_flow_growth.gif)

*What you see:* three colonies growing polyp by polyp from one seed, viewed from above, under no flow, moderate flow and strong flow (flow runs left to right; colour = age, dark old to pale new). Each new polyp is placed where a nutrient random-walker first touches the colony, so growth follows nutrient flux; with flow the walkers arrive from upstream and the colony leans into the current and compacts, which is the paper's result.

Audit note (2026-10-13, 55.5): the reduction is explicit, and the paper's symmetry magnitude and branch alignment are computed, but they are not compared numerically with the paper's Figs 9–12.

**Reference:** Chindapol N, Kaandorp JA, Cronemberger C, Mass T, Genin A (2013). Modelling growth and form of the scleractinian coral *Pocillopora verrucosa* and the influence of hydrodynamics. *PLoS Comput Biol* 9(1):e1002849. [DOI](https://doi.org/10.1371/journal.pcbi.1002849)

### [1952-turing-morpho.py](./1952-turing-morpho.py)

A simulation of the Gray-Scott reaction-diffusion model, implementing Alan Turing's theory from "The Chemical Basis of Morphogenesis" (1952). This model generates classic Turing patterns like spots and stripes, demonstrating how simple chemical reactions can produce complex biological patterns.

![Turing patterns (Gray–Scott)](docs/media/turing_gray_scott.gif)

*What you see:* the concentration of one of the two Gray–Scott species on a periodic 2-D grid, from a seeded square to a self-organising spot/stripe pattern (the full-size animation is `turing_patterns.gif`).

Audit note (2026-10-13, 52.7): the integrator follows Pearson (1993) correctly, but the run is not placed on Pearson's (f, k) phase diagram and no pattern wavelength is measured, so the output is not tied to a specific published pattern class.

### [1966-raup.py](./1966-raup.py)

Raup's model of shell coiling (Raup & Michelson 1965, *Science* 147:1294, doi:10.1126/science.147.3663.1294; Raup 1966, *J. Paleontol.* 40:1178–1190, doi:10.2307/1301992). A generating curve (the aperture) is swept around a fixed axis while every dimension grows as a logarithmic spiral: W is the whorl expansion rate (each whorl is the previous one magnified by W), D the inner/outer-margin ratio of the generating curve (its relative distance from the axis), T the axial movement of the curve's centre per unit of its radial movement (T = 0 is planispiral), S the curve's shape. `1966-raup.py` draws a 3 × 3 slice (W against T) of the morphospace; `create_raup_animations.py` renders six illustrative forms and the parameter-space sheets.

![Raup shell coiling animation](docs/media/raup_shell_coiling.gif)

*What you see:* six illustrative points in the morphospace — planispiral (cf. *Nautilus*), turreted (cf. *Turritella*), an evolute disc (cf. an ammonite), conical (cf. *Conus*), high-spired (cf. *Cerithium*), globose (cf. *Helix*) — as the viewpoint rotates from side to overhead. The parameter values are chosen so that each name is geometrically true; they are not Raup's measurements of those taxa (see [`datasets/raup/README.md`](./datasets/raup/README.md)).

Run `python create_raup_animations.py` to generate the animation, `raup_parameter_space.png` (W × D and W × T sheets), `raup_empirical_comparison.png` (the six panels as stills) and `raup_shell_metrics.csv` (W, D, T recovered from each rendered mesh beside the inputs, with height/width and spire/width). The script first recovers W, D, T from meshes on a 3 × 3 × 4 grid and refuses to draw unless they match to 1e-9 and the spire grows monotonically with T.

Audit note (2026-10-13, 37.3 → remediated 2026-10-17): the earlier version translated linearly in angle (`zc = T*theta`) while the whorls grew exponentially, so T had no visible effect and every panel was a flat disc; it also drew the generating curve in the horizontal plane (zero axial thickness) and scaled the axis offset so the rendered inner/outer ratio was (1+D)/(3−D) rather than D. The definitions in `datasets/raup/variables.json` that led to this ("vertical rise per radian") had been written without the paper and are replaced, with their provenance stated, by Raup's definitions as restated by Noshita (2014) and Okabe & Yoshimura (2017). Re-scored in audit 2026-10-17-raup (see the table below); [morpholib#1](https://github.com/fractastical/morpholib/issues/1).

### [2010-petroff-stromatolite.py](./2010-petroff-stromatolite.py)

Petroff et al.'s (2010) biophysics of conical stromatolites: (1) the spacing law ℓ = α√(Dτ), which makes a ~1 cm spacing between neighbouring cones a fingerprint of a ~daylength photosynthetic rhythm and packs cones onto a hexagonal lattice, and (2) a dome-vs-cone morphospace from the deterministic KPZ growth law ∂h/∂t = v + λ(1 + ½(∂h/∂r)²), where β = v/λ sets the aspect ratio. Output: `petroff_stromatolite_morphospace.png`; `--gif PATH` adds the animation below.

![Petroff stromatolite: dome-to-cone sweep of beta = v/lambda](docs/media/petroff_stromatolite_dome_to_cone.gif)

*What you see:* a stromatolite cross-section grown lamina by lamina from a small seed under velocity = v·ẑ + λ·n̂ (vertical phototactic growth plus surface-normal mineral accretion), while the ratio β = v/λ sweeps 0.2 → 6 along the bar below: low β rounds the seed into a dome, high β extrudes a sharp cone. The curves are successive laminae, as in a vertical thin-section.

**Reference:** Petroff AP, Sim MS, Maslov A, Krupenin M, Rothman DH, Bosak T (2010). Biophysical basis for the geometry of conical stromatolites. *PNAS* 107(22):9956–9961. [DOI](https://doi.org/10.1073/pnas.1001973107)

### [2021-Cervera–Levin–Mafe.py](./2021-Cervera–Levin–Mafe.py)

A reaction-diffusion demo inspired by Cervera–Levin–Mafe (2021), exploring morphogen antagonism and its effect on pattern formation. The model includes:
- Antagonistic morphogens (m1, m2) on an antero-posterior axis with mutual annihilation
- Independent morphogen (m3) on a lateral axis
- Gap-junction blocking effects modeled through diffusion reduction
- 3D morphospace visualization of mean expressions

![Cervera–Levin–Mafe-inspired antagonism: the m1/m2 boundary moves as the gap-junction block on m2 deepens, drawn on the silhouette of a real Schmidtea mediterranea](docs/media/cervera_levin_mafe_gj_block.gif)

*What you see:* top, the steady-state profiles of the two antagonistic morphogens along the anterior–posterior axis (m1 from an anterior source, m2 from a posterior one; they annihilate where they meet) and the wavefront x* where they are equal; bottom, the same as territories on a real planarian — the silhouette of a dorsal photograph of *Schmidtea mediterranea* (Stevenson & Beane 2010, PLoS ONE 5(12): e15310, via Wikimedia Commons, CC BY 2.5; fetched once into `.cervera_cache/`, SHA-256 pinned in the script; red = m1 dominates, blue = m2, brightness = the lateral morphogen m3 measured across the local body width). Until 2026-10-17 the body was an ellipse; it still is if the photograph cannot be fetched, and the panel title says so. The photograph is cropped at the ventral margin near mid-body, which is why the silhouette has a short straight edge there. As the gap-junction block cuts m2's diffusion from 1.0× to 0.4× of normal, the boundary shifts posteriorly, the model's reading of how bioelectric coupling moves positional-information boundaries. `python3 "2021-Cervera–Levin–Mafe.py" --gif out.gif`; without `--gif` the three interactive plots open.

MorphoEval audit `2026-10-17-cervera-audit` at commit `e71ca55`: **44.9** (model-fidelity 38, specimen-match 38, parameter-expressivity 35, morphometric-validation 25, source-traceability 78, code-traceability 42, repro-package 78). This is the first audit in the repository scored by a model from a different company than the one that wrote the code — `assessor: gpt-5.6-sol` (OpenAI), run as an independent agent that was given the rubric, the script and the fixtures and no opinions. Its findings, each pinned to lines: the advertised (f2, f3) morphospace is constant along f3 because the loop variable is never used, and its f2 = 0.58 phenotype boundary is drawn rather than derived; the CSV reconstructs each crop's wavefront as 0.50 + 0.26·shift after the shift was normalised by 0.13, so the reported displacement is exactly twice the simulated one; the crop‑specific f2 values, wound gains, time rescaling and the 0.55 head‑readout threshold that picks the double‑head cartoon are hand‑set without a cited calibration; the PlanformDB frequencies are plotted beside model indices with no fit or error; the real planarian is a backdrop, not measured; `--self-test`'s "trace parity" checks a port against another program's synthetic traces (tolerance 0.035, endpoints only), not biology; the full 2021 and 2020 papers were not accessible, so equation‑level fidelity is not asserted; and the PMID the script cites (34418508) is a different paper — the article is 34411690. Runs were byte‑identical. The six remediation items are in the audit; this score is the baseline.

*Fix pass (2026-10-17, after the audit):* the script now says in its docstring and figure titles that it is a qualitative analogue of the paper's antagonistic-morphogen idea plus a port of the rusty-matter circuit, not an implementation of the 2021 equations. The f3 axis is real: m3 is solved on the mediolateral axis with D3 = f3·D3,0 and summarised as the centre/edge ratio λ(f3) (0.20 → 0.01 over f3 = 1.0 → 0.3), and panel E shows the two independent curves rather than a surface. The bipolar criterion is derived instead of drawn: x*_b = 0.583 is the posterior end of the wild-type pharynx read from the PlanformDB organ table over the wild-type silhouette (the old code had hard-coded 0.58 without saying where it came from). The crop → f2 mapping is one fitted parameter, f2 = 1 − a·p with p the crop position (0, ⅓, ⅔, 1; the tail crop has no trunk cut and is held at f2 = 1), trained on Oviedo 2010's octanol series (a = 0.58, RMSE 4.2 points) and tested on the held-out innexin RNAi series (RMSE 23.5 points, r = 0.94) — both numbers printed on panel C and in the CSV. The CSV writes the solver's x* directly (0.500 / 0.520 / 0.546 / 0.583 / 0.500 for head / pre-pharyngeal / pharyngeal / post-pharyngeal / tail crops), and `--self-test` checks that it equals the solver's value. The photograph is measured, not decorative: the anterior share of the real body's area is computed per frame and plotted against x*. Port parity now compares every sampled frame (11 frames × 4 fields, max error 3.3e-2 at tolerance 0.035) and is labelled "port parity … not a biological validation". The PMID is corrected to 34411690. The 24 h mapping in panel D is still a rescaling and the panel says so. 

Re-score `2026-10-17-cervera-audit-v2` at commit `a0d6848`, same independent assessor (GPT‑5.6 Sol, OpenAI), superseding the first audit's nine nanopubs: **55.6** (model-fidelity 45, specimen-match 42, parameter-expressivity 58, morphometric-validation 48, source-traceability 82, code-traceability 60, repro-package 84; faithfulness 46.4, reproducibility 76.9). It confirmed as genuine fixes the analogue labelling, the active f3 axis, the direct CSV wavefronts, the measured silhouette, the full-frame parity and its labelling, the PMID and the expanded deterministic self-test (two runs byte-identical). It did not give much for what remains: the "derived" criterion still combines the Planform pharynx centre with the renderer's hard-coded 100-unit pharynx length, and the rule that an m1/m2 boundary past the pharynx means a double head has no cited source or test; the fit is one parameter on four points with a 23.5-point held-out error and no uncertainty; m3 is a signal axis, not a mapped anatomy; the outcomes are still two Planform cartoons; the 24 h mapping is illustrative. And it found two new defects the builder had missed: `f2_for_shift` bisected on a clipped index that can never exceed its target, so the printed threshold was f2 = 0.20 where the real crossing is 0.42 (visible in the figure the builder had looked at and passed), and panel D's legend carried hard-coded 85.8 % / 20.6 % readouts that the current simulation does not produce (91.8 % / 23.5 %). Both, plus the solver docstring that still attributed the equations to the paper, are fixed in the commit after `a0d6848` and `--self-test` now asserts the inverse; that commit has not been re-scored.

### [2024-kahng-coral-geometry.py](./2024-kahng-coral-geometry.py)

Kahng, Odle & Wakeman's (2024) geometry of reef-coral colonies. A colony lays down skeleton at a per-area calcification rate G, so volume grows as V(t+1) = V(t) + G·S(t)/ρ; whether linear extension and surface-area productivity are size-dependent is then decided entirely by how the surface-to-volume ratio changes with size. The script reconstructs these relationships for the paper's four canonical shapes (hemispheroid, fixed-aspect cone, flat disc of fixed thickness, cylinder of fixed branch diameter). Output: `kahng_coral_geometry_morphospace.png`; `--gif PATH` adds the animation below.

![Kahng coral geometry: the master rule iterated year by year for four colony shapes](docs/media/kahng_coral_geometry_growth.gif)

*What you see:* the growth rule V(t+1) = V(t) + G·S/ρ iterated year by year with the same G and ρ for every shape, so only geometry differs. Left, the hemispheroid and the cone drawn to scale: their surface-to-volume ratio falls as they grow, so their extension plateaus and their surface-area productivity (SAP) drops. Right, surface area against time on a log axis: the flat disc and the fixed-diameter cylinder keep S:V constant and grow exponentially — the paper's size-independent SAP — while the two massive shapes level off.

**Reference:** Kahng SE, Odle E, Wakeman KC (2024). Coral geometry and why it matters. *PeerJ* 12:e17037. [DOI](https://doi.org/10.7717/peerj.17037). Code: [ericodle/coral_geometry_modeling](https://github.com/ericodle/coral_geometry_modeling).

### [2001-bressloff-v1-form-constants.py](./2001-bressloff-v1-form-constants.py)

Bressloff, Cowan, Golubitsky, Thomas & Wiener's (2001/2002) Euclidean symmetry-breaking model of primary visual cortex (V1) and Klüver's four geometric hallucination form constants (tunnels/funnels, spirals, cobwebs, honeycombs), paired with Rule, Stoffregen & Ermentrout's (2011) flicker-phosphene Floquet resonance tongues (`1:1` vs `2:1` stripes vs hex/spots). Reproduces all five `stability_reports` in [`datasets/bressloff/paper-calibration.json`](./datasets/bressloff/paper-calibration.json) to machine precision ($\le 6.7 \times 10^{-16}$). Output: `bressloff_v1_form_constants_morphospace.png` and `bressloff_v1_metrics.csv`; `--gif PATH` adds the animation below.

![Bressloff V1 form constants: cortical planforms and retinal log-polar pullbacks across Klüver's four families](docs/media/bressloff_v1_form_constants.gif)

*What you see:* Klüver's four form-constant families stepped through one at a time (1/4 Tunnel/Rings, 2/4 Spiral/Funnel, 3/4 Cobweb/Square, 4/4 Honeycomb/Hex): left, the V1 cortical wave lattice as its spatial phase drifts over one cycle; right, its inverse retino-cortical log-polar pullback into visual-field coordinates.

**References:**
- Bressloff PC, Cowan JD, Golubitsky M, Thomas PJ, Wiener MC (2001). Geometric visual hallucinations, Euclidean symmetry and the functional architecture of striate cortex. *Phil. Trans. R. Soc. Lond. B* 356(1407):299–330. [DOI](https://doi.org/10.1098/rstb.2000.0769)
- Rule M, Stoffregen M, Ermentrout B (2011). A model for the origin and properties of flicker-induced geometric phosphenes. *PLoS Comput. Biol.* 7(9):e1002158. [DOI](https://doi.org/10.1371/journal.pcbi.1002158)

### [sorting-morphogenesis/](./sorting-morphogenesis/)

Zhang, Goldstein & Levin's (2025) model of autonomous cell-view sorting morphogenesis: 1D arrays of cells that sort themselves locally by Bubble, Insertion or Selection rules without a central controller, navigating around passive (`movable`) and stuck (`immovable`) frozen defects, exhibiting Delayed Gratification (temporary increases in monotonicity error followed by larger reductions), and spontaneously segregating by algotype in chimeric arrays. Includes deterministic seed-locked paired A/B lane comparisons (`--compare-ab`) and a 4-tier claim-boundary catalog ([`sorting-morphogenesis/catalog.json`](./sorting-morphogenesis/catalog.json)).

```bash
python3 sorting-morphogenesis/sorting_morphogenesis.py                  # self-tests + quick replication summary
python3 sorting-morphogenesis/run_experiments.py --n 100 --reps 15      # full 8-figure suite in sorting-morphogenesis/figures/
python3 sorting-morphogenesis/run_experiments.py --compare-ab --n 100   # seed-locked paired A/B comparison + CSV
python3 sorting-morphogenesis/run_experiments.py --gif zhang.gif --gif-only
```

![Zhang, Goldstein & Levin (2025): autonomous cell-view sorting around immovable frozen defects, Delayed Gratification, and chimeric algotype clustering](docs/media/zhang_sorting_morphogenesis.gif)

*What you see:* top-left, autonomous Selection vs Bubble cell arrays ($n = 36$) sorting around two immovable frozen defects (red outlines) — Selection jumps across the defects to near-zero error while Bubble is blocked locally; bottom-left, a 50/50 Bubble + Selection chimera with duplicate values $1..9$ where equal-valued cells spontaneously segregate by algotype (blue = Bubble, orange = Selection); right, the monotonicity-error trajectory with Delayed Gratification excursions and the algotype aggregation curve rising from the `0.50` random baseline to `~0.72`.

*Replication findings ($n = 100$, [`sorting-morphogenesis/README.md`](./sorting-morphogenesis/README.md), reported upstream in [Zhangtaining/cell_research#2](https://github.com/Zhangtaining/cell_research/issues/2)):* all quantitative claims in Sections 4.1–4.4 reproduce — Bubble and Insertion perform the same swap count as top-down ($\binom{n}{2}/2 \approx 2,482$) with **1.55×** and **2.04×** fewer total comparisons+swaps (paper: `1.50×`, `2.03×`); Selection performs **11.05×** more swaps (`1,046.9` vs `94.7`, paper: `~11×`) when `MultiThreadCell.swap()` resets `ideal_position` to `0` on each swap, and **1.18×** more total operations (paper: `1.17×`) when `ideal_position` persists; Selection achieves lower final error on immovable frozen defects (`1.00, 1.95, 2.95` vs `1.80, 3.75, 5.10` for $f=1..3$, Fig. 5C) because its `ideal_position` pointer swaps non-locally across barriers; Insertion avoids halting behind a frozen defect via an explicit prefix-reset rule (`InsertionSortCell.is_enable_to_move` L68–82); Delayed Gratification in Bubble rises monotonically with $f=0..3$ (`0.38 → 0.51` vs traditional `< 0.23`, Fig. 7A); chimeric arrays (`B+S`, `I+S`, `B+I`) peak at `0.635–0.640` algotype aggregation on unique values (Fig. 8A) and stay permanently segregated (`0.698` and `0.661`, Fig. 8D) on duplicate values `1..10`; and opposite-direction chimeras follow the paper's `Bubble > Selection > Insertion` sortedness hierarchy (`B+I = 75.6%`, `B+S = 47.3%`, `I+S = 40.6%`, Figs. 9–10).

**Reference:** Zhang T, Goldstein A, Levin M (2025). Classical sorting algorithms as a model of morphogenesis: Self-sorting arrays reveal unexpected competencies in a minimal model of basal intelligence. *Adaptive Behavior* 33(1):25–54. [DOI](https://doi.org/10.1177/10597123241269740). Code: [Zhangtaining/cell_research](https://github.com/Zhangtaining/cell_research) (pinned commit `1fd2bd59`; replication notes in [issue #2](https://github.com/Zhangtaining/cell_research/issues/2)).

### [thompson/1917-thompson.py](./thompson/1917-thompson.py)

A demonstration of D'Arcy Thompson's theory of transformations from *On Growth and Form* (1917), showing how one shape can be deformed into another through mathematical transformations. This classic work established the foundation for mathematical biology and the study of biological form.

![Thompson transformations: the silhouette of a real Scarus ghobban on a Cartesian grid carried through a shear, coaxial circles (with a real Pomacanthus as the dashed target), a fan of coordinates and a radial expansion](docs/media/thompson_transformations.gif)

*What you see:* a real fish — the silhouette of a *Scarus ghobban* photograph (Wikimedia Commons, Huet/IFREMER, CC BY 4.0; segmented by the script above, which pins the file's SHA-256) — drawn on a square grid; the grid is deformed by four of Thompson's transformation families (the oblique shear of *Argyropelecus* → *Sternoptyx*, the coaxial circles of *Scarus* → *Pomacanthus*, a fan of coordinates as in *Polyprion* → *Pseudopriacanthus*, a radial expansion as in *Diodon* → the sunfish) and the fish follows the grid. In the coaxial panel the silhouette of a real *Pomacanthus semicirculatus* (Levy, CC BY 2.0) is drawn dashed as the target, placed where the deformed parrotfish lands. The deformation strengths are Thompson's illustrative ones, not fitted; the fitted maps, scored on landmarks and outlines, are the script above. Until 2026-10-17 the fish here was a hand-drawn ellipse with a tail and fin. `python3 thompson/1917-thompson.py --gif out.gif`; without `--gif` a two-panel toy applies the coaxial-circle map to the same silhouette (until 2026-10-16 that toy applied the same polynomial to both coordinates, which collapses every shape onto a line; fixed). If the photographs cannot be fetched the script falls back to the stylised fish and says so in the title.

Thompson-focused notes and dataset curation are in:
- [`thompson/DARCY_THOMPSON_FOCUSED_OPTIONS.md`](./thompson/DARCY_THOMPSON_FOCUSED_OPTIONS.md)
- [`thompson/MODERN_DATASETS_THOMPSON_ALIGNED.md`](./thompson/MODERN_DATASETS_THOMPSON_ALIGNED.md)
- [`thompson/generate_tps_transformation_panels.py`](./thompson/generate_tps_transformation_panels.py) and [`thompson/ingest_dryad_morphometrics.py`](./thompson/ingest_dryad_morphometrics.py) for thin-plate-spline panels from modern landmark data

## MorphoEval: how faithful are these reproductions?

[`morphoeval/`](./morphoeval/) is a seven-dimension rubric (model fidelity, specimen match, parameter expressivity, morphometric validation, source traceability, code traceability, reproducibility package; weights sum to 100) applied as a pilot case study to the scripts above, with every score backed by line-pinned evidence at a recorded commit and by measurements taken on the rendered outputs. To prevent conflating faithful code with valid biology, faithfulness (weight 70) is decomposed and reported along two separate axes — **model-implementation fidelity** (`impl.`, weight 30: `model-fidelity` + `parameter-expressivity`) and **biological/empirical validation** (`bio.`, weight 40: `specimen-match` + `morphometric-validation`) — alongside **reproducibility** (`repro.`, weight 30). Scores on `[0, 100]` are **uncalibrated criterion-referenced ratings**, not probabilities ($s = 76$ does not mean $P(\text{faithful}) = 0.76$); when serialized into the governed ProbKnow templates (`pk:LLMClaimAssessment` and `pk:Claim`), `score / 100` in `pk:hasProbability` and `overall / 100` in `pk:weightOfEvidence` are unit-interval rescalings for template conformance. Audits are YAML files under `morphoeval/audits/`. Each score is published as a *nanopublication*: the smallest publishable unit of science — one statement ("this script's reproduction of Raup 1966 scores 64.1/100 on the MorphoEval rubric (mostly faithful)") packaged with who asserted it, when, on what evidence and under what licence, cryptographically signed, and stored on a public network of servers under a permanent address that anyone can cite, fetch or query. Published statements are never edited; a re-score publishes new ones that point at those they replace, so the history stays visible, and other assessors (other models, or people) can publish their own assessment of the same hypothesis next to ours.

How a body gets its form, and how a form gets into the repository. Two lines under the title say what morphogenesis is and that the newer papers hold the pattern to be stored bioelectrically. Tier A: a person scouts — picks a paper, dataset or claim worth the work and points the agent at it — then paper → its formal source → rules pinned and verified on the data → script. Box 6, between the tiers, is what the scripts produce: the reproductions' own animations (playing), each labelled with its paper and colour-coded by what the script does with it (reproduce the paper's model from its equations, decode the authors' formal encoding, or test a published claim), set beside the authors' figures or specimens — this is what gets scored. Tier B: the audit cycle, from the reviewer's scores to signed public statements, to a person who reads them and decides what to fix next; the builder agent then re-works the script — its fidelity to the paper, its sources, its checks — and the fix runs back up through the outputs into the script. Purple figures mark where people decide; the robots are the builder agent (writes and re-works the code) and the assessor agents (score it); a robot's colour is the company whose model it is, read from the audits — box 7 ("7 Dual-agent audit") shows Anthropic's robot (`claude-fable-5-1`), Google's (`gemini-4-argon`), and OpenAI's (`gpt-5.6-sol`), and **every active evaluated script (10/10) is dual-scored across two companies' models** (13/25 passes by Anthropic, 10/25 by Google, 2/25 by OpenAI) so that no script relies solely on its builder grading itself (the Mermaid source is in [`morphoeval/README.md`](./morphoeval/README.md#the-cycle)):

![How a reproduction is made and how MorphoEval scores it](./docs/media/morphoeval_pipeline.gif)

Where a dimension is structurally non-applicable (`not_applicable`), substituting baseline `50` shrinks scores linearly toward `50` ($\bar{s}_{\text{pad}} = (1-\alpha)\bar{s}_{\text{app}} + 50\alpha$); the table therefore reports both the weight-renormalised applicable-only mean (`app.`) and the baseline-50 padded score (`pad.`), plus the second-assessor score (`gemini-4-argon`, Google) and two-assessor panel mean across all 10 active evaluated scripts:

| simulation | checkpoint | primary assessor | impl. (30) | bio. (40) | repro. (30) | primary app. / **pad.** | 2nd assessor (`gemini-4-argon`) app. / **pad.** | panel mean **pad.** |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Hoffmann–Donoughe wing venation (`2018-hoffmann-wing-venation.py`) | f3fe478, audit 2026-10-13 | Claude Fable 5.1 | 81.7 | 80.0 | 71.3 | 77.8 / **77.8** | 81.3 / **81.3** | **79.5** |
| Thompson *Scarus* → *Pomacanthus* on real landmarks and outlines (`thompson/1917-thompson-scarus-pomacanthus.py`) | 232e8f8, audit 2026-10-17-thompson (re-score of 2026-10-16: 70.2) | Claude Fable 5.1 | 70.3 | 77.3 | 82.0 | 76.6 / **76.6** | 90.8 / **90.8** | **83.7** |
| Odonata wing cells, fossil vs extant (`2026-eshghi-odonata-wing-cells.py`) | fbbcaf6, audit 2026-10-14-odonata | Claude Fable 5.1 | 63.3 (`70.0` app.) | 85.0 | 78.0 | 78.9 / **76.0** | 91.4 / **87.3** | **81.7** |
| Limbform limb morphologies (`2014-lobo-limbform-morphology.py`) | 747c57f, audit 2026-10-14-limbform | Claude Fable 5.1 | 81.7 | 61.3 | 74.3 | 71.3 / **71.3** | 83.6 / **83.6** | **77.5** |
| WormGUIDES *C. elegans* embryo atlas + connectome (`2015-santella-wormguides-celegans.py`) | f39a409, audit 2026-10-16-celegans (re-score of 2026-10-15: 67.7) | Claude Fable 5.1 | 70.0 (`80.0` app.) | 64.4 | 78.0 | 72.4 / **70.2** | 81.9 / **78.7** | **74.4** |
| Planform planarian morphology (`2013-lobo-planform-morphology.py`) | 161f269, audit 2026-10-14 | Claude Fable 5.1 | 81.7 | 55.0 | 66.7 | 66.5 / **66.5** | 76.5 / **76.5** | **71.5** |
| Raup shell coiling (`create_raup_animations.py`) | ba6fd20, audit 2026-10-17-raup (re-score of 2026-10-13: 37.3) | Claude Fable 5.1 | 76.7 | 45.0 | 77.0 | 64.1 / **64.1** | 79.3 / **79.3** | **71.7** |
| Cervera–Levin–Mafe bioelectric synthesis (`2021-Cervera–Levin–Mafe.py`) | a0d6848, audit 2026-10-17-cervera-v2 (was 44.9 at e71ca55) | **GPT‑5.6 Sol (OpenAI)** | 49.3 | 44.3 | 76.9 | 55.6 / **55.6** | 60.7 / **60.7** | **58.1** |
| Chindapol coral flow (`2013-chindapol-coral-flow.py`) | f3fe478, audit 2026-10-13 | Claude Fable 5.1 | 53.3 | 43.8 | 70.0 | 55.5 / **55.5** | 56.0 / **56.0** | **55.7** |
| Gray–Scott (Turing) (`1952-turing-morpho.py`) | f3fe478, audit 2026-10-13 | Claude Fable 5.1 | 75.0 | 38.8 (`20.0` app.) | 49.0 | 53.6 / **52.7** | 37.8 / **40.9** | **46.8** |
| Planform animation (`1900-planformDB_parser.py`), superseded | f3fe478, audit 2026-10-13 | Claude Fable 5.1 | 9.3 | 2.5 | 36.3 | 14.7 / ~~14.7~~ | — | — |
| Bressloff V1 form constants (`2001-bressloff-v1-form-constants.py`), Petroff stromatolite (`2010-petroff-stromatolite.py`), Kahng coral (`2024-kahng-coral-geometry.py`), Zhang sorting morphogenesis (`sorting-morphogenesis/`), Thompson demo (`thompson/1917-thompson.py`) | — | — | — | — | — | — | — | *unevaluated* |

Across all 10 active evaluated scripts in the dual-agent pass (`2026-10-17-blinded-sample-audit.yaml`, `gemini-4-argon`, Google), rank-order agreement with the primary assessors (`claude-fable-5-1`, Anthropic on 9 scripts; `gpt-5.6-sol`, OpenAI on `2021-Cervera–Levin–Mafe.py`) is strong (Spearman $\rho = 0.891$, Kendall $\tau = 0.778$, Pearson $r = 0.920$ on padded scores; $\rho = 0.903$ on applicable-only scores; and exact $\rho = 1.00$, $\tau = 1.00$ on the initial 5-script strictly blinded subset; dimension-level $\rho = 0.796$, $r = 0.813$ across all 70 dimension pairs, mean absolute difference $10.4$ points), while the independent passes surfaced concrete defects the builder had missed (including a 65-character SHA-256 typo in `2014-lobo-limbform-morphology.py`).

191 nanopubs are on the production network, of which 91 are the current heads of their primary version chains (46 from 2026-10-13, 9 from the 2026-10-14 planarian re-score, 9 from the 2026-10-14 Odonata audit, 10 from the Limbform audit, 9 from the 2026-10-16 C. elegans re-score, 9 from the 2026-10-17 Thompson re-score, 9 from the 2026-10-17 Raup re-score, 9 from the 2026-10-17 Cervera–Levin–Mafe re-score, which supersedes the 9 of the first Cervera audit the same day — both by OpenAI's GPT‑5.6 Sol — and 54 conformant re-issues of 2026-10-16 that `npx:supersedes` the pre-template versions), plus 80 signed second-assessor nanopubs in `morphoeval/kg/2026-10-17-blinded-sample-audit/signed/`; URIs and version histories are in `morphoeval/kg/<audit>/published.json`. Since 2026-10-15 the nanopubs follow the governed templates of the [ProbKnow Space](https://w3id.org/spaces/probknow) (template links in pubinfo, full timestamps); the current rubric nanopub is https://w3id.org/np/RAhBMChdpmgS_yBfy8cfYXpfKktqlUD72A9bxZzvWDkYM (superseding the original RAjd9dIu…), and every nanopub that was still the head of its chain on 2026-10-16 has been re-issued in that shape with unchanged scores and evidence. Remediation is tracked in [morpholib#1](https://github.com/fractastical/morpholib/issues/1); the data-format review for the ProbKnow template is [project-delphi-ontology#12](https://github.com/fractastical/project-delphi-ontology/issues/12). MorphoEval is the simulation-faithfulness sibling of BioEval ([10.5281/zenodo.20567719](https://doi.org/10.5281/zenodo.20567719)) and NeuroEval ([10.5281/zenodo.20622878](https://doi.org/10.5281/zenodo.20622878)). Full workflow, tooling notes and the rubric text: [`morphoeval/README.md`](./morphoeval/README.md).

## Datasets

- [`datasets/odonata/`](./datasets/odonata/): Eshghi et al. 2026 wing-cell measurements (463 wings, with GBIF families and PBDB ages), the 22 fossil taxa's PBDB age records, and 1762 PBDB Odonatoptera occurrences; README with query URLs, licences and retrieval date.
- [`datasets/raup/`](./datasets/raup/): Raup's parameter definitions (restated from Noshita 2014 and Okabe & Yoshimura 2017, with provenance), six illustrative parameter sets (`variables.json`), the superseded 2025 extraction log, paper chronology and citations
- [`datasets/planform/`](./datasets/planform/): BibTeX for the Planform papers and `planformDB.sha256`, the SHA-256 of the `planformDB_2.5.0.edb` the decoder was verified on (744,448 bytes); the `.edb` itself is not vendored (it ships with Planform), and `2013-lobo-planform-morphology.py` checks the hash of whatever file it is given, warning on a mismatch or refusing it with `--strict-db`.
- LimbformDB 2.1.0 (`.ldb`) is likewise not vendored; `2014-lobo-limbform-morphology.py --download` fetches it from the Lobo lab into the git-ignored `.limbform_cache/` and verifies its SHA-256.
- [`datasets/celegans/`](./datasets/celegans/): SHA-256 manifest of the 416 WormGUIDES / c302 files (pinned commits) that `2015-santella-wormguides-celegans.py --download` fetches into the git-ignored `.celegans_cache/`, with a README on sources, licences, the calibration assumption and the literature anchors.
- [`datasets/bressloff/`](./datasets/bressloff/): SHA-256-pinned reference calibration fixtures (`paper-calibration.json`, `rule-2011-floquet.json`, `driven-neural-fields-registry.json`) and BibTeX citations for Bressloff et al. (2001/2002) and Rule, Stoffregen & Ermentrout (2011).
- [`sorting-morphogenesis/catalog.json`](./sorting-morphogenesis/catalog.json): 4-tier claim-boundary catalog (`reported`, `reconstruction`, `transfer_boundary`, `does_not_claim`) and paired A/B lane definitions for Zhang, Goldstein & Levin (2025).
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
