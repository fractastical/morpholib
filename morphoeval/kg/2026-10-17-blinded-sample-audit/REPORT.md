# MorphoEval report — audit `2026-10-17-blinded-sample-audit`

Rubric **0.1.0** · checkpoint `5659d5e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasProbability` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for ProbKnow template conformance). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson's Scarus -> Pomacanthus transformation fitted to real species (`thompson/1917-thompson-scarus-pomacanthus.py`) | 94 | 86 | 82 | 96 | 97 | 93 | 90 | 90.0 | 89.8 | 93.1 | 90.8 | **90.8** | faithful |
| Odonata wing cells through deep time (`2026-eshghi-odonata-wing-cells.py`) | 88 | 92 | 50* | 96 | 95 | 88 | 90 | 75.3 | 93.5 | 91.1 | 91.5 | **87.3** | faithful |
| LimbformDB limb morphologies (`2014-lobo-limbform-morphology.py`) | 88 | 86 | 84 | 62 | 94 | 86 | 88 | 86.7 | 77.0 | 89.5 | 83.6 | **83.6** | faithful |
| Raup shell coiling morphospace (`create_raup_animations.py`) | 90 | 76 | 78 | 58 | 82 | 78 | 95 | 86.0 | 69.2 | 86.1 | 79.3 | **79.3** | mostly faithful |
| Gray-Scott reaction-diffusion patterns (`1952-turing-morpho.py`) | 68 | 50* | 35 | 5 | 15 | 30 | 55 | 57.0 | 33.1 | 35.0 | 37.8 | **40.9** | partially faithful |
| Hoffmann-Donoughe insect wing venation (`2018-hoffmann-wing-venation.py`) | 82 | 85 | 68 | 84 | 84 | 78 | 80 | 77.3 | 84.6 | 80.8 | 81.3 | **81.3** | faithful |
| WormGUIDES C. elegans embryo atlas with the OpenWorm connectome (`2015-santella-wormguides-celegans.py`) | 84 | 74 | 50* | 82 | 92 | 82 | 86 | 72.7 | 77.0 | 86.9 | 81.9 | **78.7** | mostly faithful |
| PlanformDB planarian morphology animation (`2013-lobo-planform-morphology.py`) | 86 | 74 | 80 | 56 | 85 | 82 | 78 | 84.0 | 67.2 | 81.4 | 76.5 | **76.5** | mostly faithful |
| Cervera–Levin–Mafe gap-junction reaction–diffusion and bioelectric synthesis (`2021-Cervera–Levin–Mafe.py`) | 50 | 48 | 62 | 54 | 84 | 68 | 88 | 54.0 | 50.2 | 81.3 | 60.7 | **60.7** | mostly faithful |
| Chindapol flow-driven coral growth (`2013-chindapol-coral-flow.py`) | 38 | 42 | 72 | 48 | 82 | 68 | 82 | 49.3 | 44.2 | 78.3 | 56.0 | **56.0** | partially faithful |

## Findings and remediation

### Thompson's Scarus -> Pomacanthus transformation fitted to real species  —  overall 90.8 (faithful)

Blinded second-assessor pass (gemini-4-argon). Operationalises D'Arcy Thompson's freehand transformation grids (1917, Figs. 373-382) as a nested hierarchy of conformal and curvilinear coordinate families (similarity, affine, Moebius, coaxial circles via exact complex log-ratio and power-shear, and exact Bookstein 1989 thin-plate spline) fitted to 17 homologous landmarks across 791 Indo-Pacific fish species and tested on segmented museum/field specimen silhouettes.

- **Model fidelity to source: 94** — Operationalises Thompson's freehand coordinate transformations (1917, pp. 749-751, Figs. 375-376) into an explicit nested hierarchy of five analytical families (similarity, affine, projective/Moebius, oblique coaxial circles via complex log-ratio and power-shear, and Bookstein 1989 exact thin-plate spline). Fits all parametric models via multi-start non-linear least squares (fit_family) and verifies exact round-trip invertibility of the bipolar map (inv_coaxial) to machine precision in self_test(). Deducts 6 points only because Thompson's original grids were drawn freehand without algebraic formulas, so the 9-parameter bipolar power-shear equation is a modern formalisation rather than an equation printed in 1917, and Thompson's printed grid lines are not digitised as numerical coordinates.  
  evidence: `thompson/1917-thompson-scarus-pomacanthus.py` L256–360; `thompson/1917-thompson-scarus-pomacanthus.py` L443–463
- **Morphological realism / specimen match: 86** — Evaluates transformations directly against both Procrustes-aligned landmark means of 48 Scarus and 6 Pomacanthus species (plus 791 species across four Thompson fish pairs) and real biological specimen photographs (Scarus ghobban and Pomacanthus semicirculatus), overlayed alongside Thompson's 1917 original scan (Figs. 375-376) in Panel A. Deducts 14 points because silhouette registration uses a 2-anchor similarity transform (snout tip to LM2 and caudal peduncle minimum width to midpoint(LM5, LM6)) rather than 17 landmarks digitised directly on those two photographed individuals, and compares a preserved specimen with collapsed fins against a live swimming fish with extended dorsal/anal filaments.  
  evidence: `thompson/1917-thompson-scarus-pomacanthus.py` L524–650; `thompson_scarus_pomacanthus.png` L0–0; `thompson_scarus_pomacanthus_outlines.png` L0–0
- **Parameter expressivity: 82** — Systematically explores the parameter/complexity hierarchy (k = 4, 6, 8, 9, 34 degrees of freedom), fits every one of the 48 x 6 = 288 Scarus x Pomacanthus species pairs to generate an empirical distribution of residual errors, and tests generalisability across all four of Thompson's teleost pairs (Scarus -> Pomacanthus, Argyropelecus -> Sternoptyx, Pseudopriacanthus -> Scorpaena, Diodon -> Mola). Deducts 18 points because the script fits discrete transformation families to empirical targets rather than exposing interactive CLI parameters to sweep continuous grid parameters (pole spacing a, power alpha, shear beta).  
  evidence: `thompson/1917-thompson-scarus-pomacanthus.py` L186–200; `thompson/1917-thompson-scarus-pomacanthus.py` L414–439; `thompson_scarus_pomacanthus.png` L0–0
- **Morphometric validation: 96** — Computes per-landmark and RMS Procrustes residuals, leave-one-landmark-out cross-validation (LOO-CV), AIC-like parameter penalty, Bookstein's thin-plate spline bending energy (J = 0.258) and partial-warp spectrum (14 non-affine principal warps), plus rasterised silhouette Intersection-over-Union (IoU) and mean/95th-percentile symmetric boundary distances exported to thompson_outline_metrics.csv. Demonstrates quantitatively that Thompson's coaxial-circle family fails LOO-CV (0.0970 vs 0.0568 for similarity) and leaves silhouette IoU (0.6182) below baseline similarity (0.6615).  
  evidence: `thompson/1917-thompson-scarus-pomacanthus.py` L389–412; `thompson/1917-thompson-scarus-pomacanthus.py` L663–718; `thompson_outline_metrics.csv` L1–7
- **Source and data traceability: 97** — Module docstring and dataset README cite Thompson (1917) with Internet Archive leaf/page identifiers, Bookstein (1978, 1989) with DOIs, and Friedman et al. (2019) / Claverie & Wainwright (2014) with Dryad DOI (10.5061/dryad.fn457gr). Every local and remote asset (landmark CSV, 1917 page scan, and both Wikimedia Commons photographs) is pinned by exact byte size and SHA-256 digest and verified on every run.  
  evidence: `thompson/1917-thompson-scarus-pomacanthus.py` L1–75; `thompson/1917-thompson-scarus-pomacanthus.py` L92–134; `datasets/thompson/README.md` L1–61
- **Code-to-model traceability: 93** — Functions map cleanly to the mathematical formulations in Bookstein (1989) and Thompson (1917), with explicit docstrings specifying complex-plane equations, TPS kernel U(r) = r^2 log(r^2), partitioned system matrix L, and bending-energy eigen-decomposition, plus a 17-entry ANATOMICAL_LABELS dictionary mapping TPS indices to anatomical structures.  
  evidence: `thompson/1917-thompson-scarus-pomacanthus.py` L152–184; `thompson/1917-thompson-scarus-pomacanthus.py` L288–359
- **Reproducibility package: 90** — Deterministic optimization seeds (np.random.default_rng(1917)), CLI flags (--no-scan, --no-photos, --out, --outline-out, --metrics-out), self_test() assertion gate, and pinned dependencies in requirements.txt / requirements.lock.txt. Deducts 10 points because the outline comparison depends on runtime HTTP fetches to Wikimedia Commons when .thompson_cache/ is cold.  
  evidence: `thompson/1917-thompson-scarus-pomacanthus.py` L305–319; `thompson/1917-thompson-scarus-pomacanthus.py` L983–1025; `requirements.lock.txt` L1–35

### Odonata wing cells through deep time  —  overall 87.3 (faithful)

Blinded second-assessor pass (gemini-4-argon). Tests the macroevolutionary prediction of Hoffmann et al.'s (2018) inhibition-zone venation model (conserved inhibition length -> cell count N ~ L^2 and cell length ~ L^0) across 69 Mesozoic-Cenozoic fossil wings and 394 extant Odonata wings from Eshghi et al. (2026), and independently re-segments the 69 fossil wing line drawings from pixel scale bars to verify Table S1.

- **Model fidelity to source: 88** — Operationalises the scaling prediction of Hoffmann et al. (2018) (a fixed developmental inhibition radius r_in yields N ~ L^2 and median cell length ~ L^0) via log-log OLS regressions (ols_loglog) and replicates Eshghi et al.'s (2026) morphometric pipeline from raw fossil TIFFs by reading embedded scale bars, skeletonising veins, labelling connected interior regions, and measuring cell area, perimeter, circularity, and exact convex-hull maximum Feret diameter (_max_feret). Deducts 12 points because fossil_drawing() (used for rendering outlines, threshold < 128, largest connected component) and _read_drawing() (used for cell segmentation, threshold < 160, all wing components >= 150 px) use slightly different binarisation/component rules, and the statistical test is an observational scaling analysis rather than a mechanistic simulation.  
  evidence: `2026-eshghi-odonata-wing-cells.py` L137–143; `2026-eshghi-odonata-wing-cells.py` L352–419; `2026-eshghi-odonata-wing-cells.py` L255–349
- **Morphological realism / specimen match: 92** — Renders real fossil wing line drawings from Eshghi et al. (2026) at true millimetre scale colour-coded by per-cell Feret diameter alongside extant segmented wings from Hoffmann et al. (2018) (Arigomphus villosipes and Aeshna cyanea). Across the 67 non-swapped fossil wings, the independent re-segmentation matches Table S1 cell counts to a median signed difference of 0.0% (IQR -0.6% to +0.2%; 60/67 within 2%) and median cell length to +3.4%, while identifying two concrete curatorial errors in the published table (15_Cymatophlebia_longialata and the swapped 58/59 pair).  
  evidence: `2026-eshghi-odonata-wing-cells.py` L422–464; `datasets/odonata/fossil_wing_resegmentation.csv` L1–70; `odonata_fossil_resegmentation.png` L0–0
- **Parameter expressivity: 50** — Empirical macroevolutionary scaling and image-resegmentation pipeline operating on fixed museum and extant specimen archives rather than a parameterised generative model; marked not_applicable per the rubric.  
  evidence: `2026-eshghi-odonata-wing-cells.py` L1–45
- **Morphometric validation: 96** — Quantifies scaling slopes, R^2, p-values, and median/IQR cell lengths across fossil and extant suborders (N ~ L^1.78, R^2 = 0.87 for fossils; N ~ L^1.45, R^2 = 0.65 for extant; median cell length 0.78-0.99 mm across 150 Myr and a 6x wing-length range) and validates per-cell morphometrics (count, median, mean, IQR, 95th percentile Feret diameter, area, circularity) across all 69 fossil wings in datasets/odonata/fossil_wing_resegmentation.csv.  
  evidence: `2026-eshghi-odonata-wing-cells.py` L613–667; `2026-eshghi-odonata-wing-cells.py` L689–732; `datasets/odonata/fossil_wing_resegmentation.csv` L1–70
- **Source and data traceability: 95** — Cites Eshghi et al. (2026) Zenodo archive (DOI 10.5281/zenodo.19899648), Hoffmann et al. (2018) PNAS (DOI 10.1073/pnas.1721248115), and the Paleobiology Database Odonatoptera query URL; verifies SHA-256 digests of both remote archives (ESH_ZIP_SHA256 and HOFF_TAR_SHA256) in _fetch() and documents every vendored CSV in datasets/odonata/README.md.  
  evidence: `2026-eshghi-odonata-wing-cells.py` L66–86; `datasets/odonata/README.md` L1–108
- **Code-to-model traceability: 88** — Clear modular pipeline separating archive verification (_fetch), PBDB age lookup (pbdb_ages, AGE_OVERRIDES), raster scale-bar parsing (_read_drawing), morphological segmentation (segment_fossil_cells), convex-hull Feret caliper measurement (_max_feret), and figure generation.  
  evidence: `2026-eshghi-odonata-wing-cells.py` L149–224; `2026-eshghi-odonata-wing-cells.py` L305–419
- **Reproducibility package: 90** — Fully deterministic pipeline (no stochastic steps), SHA-256-verified auto-download and local caching (.odonata_cache/), vendored summary tables in datasets/odonata/, and pinned environment in requirements.txt / requirements.lock.txt. Deducts 10 points because running the script requires downloading ~95 MB of external archives on a cold cache and overwrites the tracked file datasets/odonata/fossil_wing_resegmentation.csv in place without CLI flags.  
  evidence: `2026-eshghi-odonata-wing-cells.py` L75–86; `2026-eshghi-odonata-wing-cells.py` L447–452; `requirements.lock.txt` L1–35

### LimbformDB limb morphologies  —  overall 83.6 (faithful)

Blinded second-assessor pass (gemini-4-argon). Decodes the LimbformDB 2.1.0 tree-graph schema (Morphology, Link, JointParam, LinkSubJoint, Digit) across all 565 database rows (555 with non-empty geometry) and renders smooth closed limb outlines, internal skeletal elements, and digit phalanges across 14 taxa. Independently caught a 65-hex-character typo in DB_SHA256 (line 92) that causes resolve_db_path() to emit a false-positive SHA-256 mismatch warning.

- **Model fidelity to source: 88** — Faithfully decodes LimbformDB 2.1.0's five-table relational graph (Morphology, Link, JointParam, LinkSubJoint, Digit) using recursive vector accumulation, cyclic half-angle bisectors at branching joints, +90/+180/+270 deg terminal caps, double-counted host segments at ectopic sub-joints, and periodic Catmull-Rom splines, satisfying the joint-radius count rule on 3,137 of 3,146 joints (99.71%). Deducts 12 points because Link.side, Link.reversed, Link.inverted, and Digit.reversed are parsed into dataclasses (lines 146-148, 165) but never used in layout() or draw_morphology(), and because 9 mutant joints require mean-radius padding.  
  evidence: `2014-lobo-limbform-morphology.py` L140–178; `2014-lobo-limbform-morphology.py` L275–421; `2014-lobo-limbform-morphology.py` L606–620
- **Morphological realism / specimen match: 86** — Reconstructs the five wild-type benchmark limbs from Lobo et al. (2014b) Fig. 1 A''-E'' (Ambystoma mexicanum, Xenopus laevis, Uca pugilator, Gryllus bimaculatus, Heteropoda venatoria), all 14 wild-type species in LimbformDB 2.1.0, and 15 experimental surgical/transgenic phenotypes (ectopic grafts, supernumerary hands, hip-rotation duplications, and retinoic-acid proximalisation) with matching segment topology, digit counts, and colour coding. Note: comparison is to the source paper's formalism cartoons rather than photographs of biological limbs.  
  evidence: `2014-lobo-limbform-morphology.py` L627–738; `limbform_wildtype_gallery.png` L0–0; `docs/media/limbform_morphospace_timeline.gif` L0–0
- **Parameter expressivity: 84** — Exercises the full discrete and continuous variation stored in LimbformDB 2.1.0 across 555 geometric morphologies spanning 6 taxonomic classes, 14 species, 12 anatomical segment types, multi-joint branching, ectopic LinkSubJoint grafts, and 0-12 digits. Deducts 16 points because the unused Link.reversed / Link.inverted / Digit.reversed flags leave surgical handedness inversions unexpressed in the rendered geometry.  
  evidence: `2014-lobo-limbform-morphology.py` L222–268; `2014-lobo-limbform-morphology.py` L656–676
- **Morphometric validation: 62** — Performs an exhaustive structural audit of the database encoding in check_encoding() (verifying 555/565 geometric morphologies and 3,137/3,146 joints against the degree-radius rule), but the script itself does not compute silhouette IoU, segment-angle residuals, or landmark errors against Lobo et al. (2014b) Fig. 1 or real limb morphometrics.  
  evidence: `2014-lobo-limbform-morphology.py` L606–620; `2014-lobo-limbform-morphology.py` L745–761
- **Source and data traceability: 94** — Cites both primary papers (Lobo et al. 2014a, DOI 10.1093/bioinformatics/btu582; Lobo et al. 2014b, DOI 10.1002/reg2.13) and the upstream LimbformDB 2.1.0 URL, and links every rendered panel to its database Morphology.Id and Experiment record. Deducts 6 points because DB_SHA256 on line 92 contains a 65-character hex string (an extra trailing '0' appended to the true 64-character SHA-256 digest), so resolve_db_path() always prints a false-positive hash mismatch warning.  
  evidence: `2014-lobo-limbform-morphology.py` L1–56; `2014-lobo-limbform-morphology.py` L91–134
- **Code-to-model traceability: 86** — Cleanly maps SQLite tables (Morphology, Link, JointParam, LinkSubJoint, Digit) to Python dataclasses and geometric primitives, with detailed ASCII diagrams and docstrings explaining the cyclic bisector rule and sub-joint double-counting convention.  
  evidence: `2014-lobo-limbform-morphology.py` L26–49; `2014-lobo-limbform-morphology.py` L305–355
- **Reproducibility package: 88** — Fully deterministic geometry decoding and rendering, automatic fallback download of LimbformDB 2.1.0 into .limbform_cache/, and pinned dependencies in requirements.txt / requirements.lock.txt. Deducts 12 points for the 65-character DB_SHA256 typo in resolve_db_path() and lack of CLI flags for output paths.  
  evidence: `2014-lobo-limbform-morphology.py` L91–134; `requirements.lock.txt` L1–35

### Raup shell coiling morphospace  —  overall 79.3 (mostly faithful)

Blinded second-assessor pass (gemini-4-argon). Implements Raup's (1962, 1966) four-parameter logarithmic-spiral shell model (W, D, T, S) using the parametric surface formulation of Noshita (2014) and Okabe & Yoshimura (2017), with an automated self_test() that recovers (W, D, T, S) from the 3D mesh to machine precision (~5e-15).

- **Model fidelity to source: 90** — Implements the logarithmic spiral growth law a(theta) = r0 * W^(theta / 2pi), generating-curve centre distance rc(theta) = a(theta)*(1+D)/(1-D) (yielding inner/outer margin ratio (rc - a)/(rc + a) = D), axial translation zc(theta) = T * rc(theta) (so dzc/drc = T Scales with whorl radius), and aperture shape S in the meridional plane through the coiling axis, verified to ~5e-15 by recover_parameters() in self_test(). Deducts 10 points because the standalone companion script 1966-raup.py hardcodes S = 1.0 and uses an unnormalised offset rc = r*(1+D) without (1-D) scaling.  
  evidence: `create_raup_animations.py` L85–131; `create_raup_animations.py` L134–211; `1966-raup.py` L10–29
- **Morphological realism / specimen match: 76** — Renders six distinct shell archetypes (planispiral involute cf. Nautilus, high-spired turreted cf. Turritella, evolute planispiral cf. Dactylioceras, low-spired obconical cf. Conus, medium-spired cf. Cerithium, helicoid globose cf. Helix) whose 3D geometries and spire-height/width ratios (0.00 to 2.26) match their morphological descriptions. Deducts 24 points because no real mollusc shell photograph, CT scan, or empirical specimen dataset (such as Raup 1967) is overlaid or compared, and the six parameter sets are explicitly declared as illustrative "cf." points rather than measurements of real specimens.  
  evidence: `datasets/raup/variables.json` L18–86; `raup_empirical_comparison.png` L0–0; `raup_shell_metrics.csv` L1–8
- **Parameter expressivity: 78** — Covers wide ranges of W (1.6 to 10.0), D (0.02 to 0.55), and T (0.0 to 4.2) across the 6-panel comparison and 120-frame morphing animation. Deducts 22 points because S is held constant at 1.0 in every rendered figure and JSON preset, and create_parameter_space_visualization() zips three values of W with three values of D (and W with T) along a 1D diagonal (3 panels per row) rather than sampling a 2D (W, D) or (W, T) morphospace grid.  
  evidence: `create_raup_animations.py` L289–344; `datasets/raup/variables.json` L30–85
- **Morphometric validation: 58** — Recovers (W, D, T, S) and aspect ratios (spire_over_width, outer_radius_ratio) directly from the generated 3D vertices in recover_parameters() and exports them to raup_shell_metrics.csv (max parameter recovery error 5.33e-15). However, this validates internal mathematical self-consistency rather than comparing model outputs against empirical shell measurements from biological specimens.  
  evidence: `create_raup_animations.py` L134–211; `raup_shell_metrics.csv` L1–8
- **Source and data traceability: 82** — Cites Raup (1962, 1965, 1966, 1967) and the two secondary papers from which the exact parametric equations were taken (Noshita 2014, DOI 10.1016/j.jtbi.2014.08.010; Okabe & Yoshimura 2017, DOI 10.1038/srep42445), and transparently documents in both the script and datasets/raup/variables.json that Raup (1966) was not consulted directly and that earlier AI-fabricated page citations were retracted.  
  evidence: `create_raup_animations.py` L1–35; `datasets/raup/variables.json` L1–29
- **Code-to-model traceability: 78** — create_raup_animations.py clearly maps variables (theta, phi, a, rc, zc, r, z) to Noshita (2014) and Okabe & Yoshimura (2017). Deducts 22 points because the companion script 1966-raup.py lacks citations, docstrings, and uses a divergent formula for rc without explanation.  
  evidence: `create_raup_animations.py` L85–131; `1966-raup.py` L1–56
- **Reproducibility package: 95** — Completely deterministic mathematical surface generation, zero external network dependencies, automated self_test() gate executed before rendering, and pinned dependencies in requirements.txt / requirements.lock.txt.  
  evidence: `create_raup_animations.py` L443–459; `requirements.lock.txt` L1–35

### Gray-Scott reaction-diffusion patterns  —  overall 40.9 (partially faithful)

Blinded second-assessor pass (gemini-4-argon). Minimal 76-line forward-Euler Gray-Scott reaction-diffusion demo on a 200x200 periodic grid at (f=0.055, k=0.062). Independently identified that inline comments mislabel the depleted substrate U as an inhibitor, that RNG seeding and literature citations are absent, and that the script's output file (turing_patterns_black_neon.gif, custom black-neon palette without axes) does not match docs/media/turing_gray_scott.gif.

- **Model fidelity to source: 68** — Implements the discrete 2D Gray-Scott reaction-diffusion system dU/dt = Du*Laplacian(U) - U*V^2 + f*(1-U), dV/dt = Dv*Laplacian(V) + U*V^2 - (f+k)*V with periodic boundary conditions via np.roll and parameters Du=0.16, Dv=0.08, f=0.055, k=0.062, dt=1.0. Deducts 32 points because (1) the 5-point finite-difference Laplacian omits the -4*Z centre weight inside laplacian() and absorbs it into Du*(Lu - 4*U), which is algebraically equivalent only for unit grid spacing dx=1, (2) inline comments misstate the reaction mechanism by claiming '# U inhibits V's growth quadratically' when U is the consumed substrate in autocatalytic production +U*V^2, and (3) the script conflates Turing's (1952) linear morphogen model with the cubic Gray-Scott (1984) / Pearson (1993) system.  
  evidence: `1952-turing-morpho.py` L6–13; `1952-turing-morpho.py` L25–40
- **Morphological realism / specimen match: 50** — Generic mathematical pattern generator not tied to a specific biological organism; marked not_applicable per the rubric. Note: the script writes turing_patterns_black_neon.gif (custom black-to-neon-blue colormap, axes off), which does not match the committed docs/media/turing_gray_scott.gif (Viridis-like colormap with visible 0-200 axis ticks).  
  evidence: `1952-turing-morpho.py` L42–75; `turing_patterns.gif` L0–0
- **Parameter expressivity: 35** — Runs only a single hardcoded parameter point (f=0.055, k=0.062) without CLI arguments or a multi-regime panel across Pearson's (1993) (f, k) phase space; only a brief inline comment mentions f=0.035, k=0.065.  
  evidence: `1952-turing-morpho.py` L8–9
- **Morphometric validation: 5** — Zero quantitative analysis: no spatial wavelength, structure factor / FFT power spectrum, spot-count, or phase-diagram comparison is computed.  
  evidence: `1952-turing-morpho.py` L1–76
- **Source and data traceability: 15** — No module docstring, no bibliographic references, no DOIs for Turing (1952), Gray & Scott (1984), or Pearson (1993), and no provenance for the chosen parameter values; only the filename and plot title ('Turing Patterns (Gray-Scott Model)') name the model.  
  evidence: `1952-turing-morpho.py` L1–13; `1952-turing-morpho.py` L74–75
- **Code-to-model traceability: 30** — Variable names (Du, Dv, f, k, U, V) are recognisable, but there are no docstrings or equation numbers and the inline comments on lines 36-39 misdescribe the Gray-Scott reaction kinetics.  
  evidence: `1952-turing-morpho.py` L31–40
- **Reproducibility package: 55** — Self-contained numpy + matplotlib script with pinned repository lockfile (requirements.lock.txt), but np.random.random((size, size)) on lines 19-20 is unseeded so every execution yields a different pattern trajectory, no CLI arguments are supported, and the output filename (turing_patterns_black_neon.gif) differs from docs/media/turing_gray_scott.gif.  
  evidence: `1952-turing-morpho.py` L19–20; `1952-turing-morpho.py` L75–75

### Hoffmann-Donoughe insect wing venation  —  overall 81.3 (faithful)

Blinded second-assessor pass (gemini-4-argon). Implements Hoffmann et al.'s (2018) SI 3.3-3.4 secondary-vein model (density-weighted seeding, damped centroidal-Voronoi / Lloyd equilibration, and Voronoi bisectors) on the segmented primary-vein scaffold of Arigomphus villosipes (923 vein domains), comparing circularity, area, and proximal-distal trends against the real wing and the regular-hexagon ceiling.

- **Model fidelity to source: 82** — Implements the core SI 3.3-3.4 pipeline on the real wing boundary and primary-vein scaffold: local density weighting proportional to 1/cell^2, damped Lloyd equilibration (step 0.55) inside the primary-vein compartments, and secondary veins drawn as Voronoi bisectors between sites sharing the same primary compartment. Deducts 18 points because local cell size is derived from the Euclidean distance transform of the real wing's all-vein mask (dt) rather than from primary-vein thickness as in the paper, and Lloyd relaxation terminates after a fixed n_iter=18 iterations rather than the paper's 0.1%-of-wing-length displacement threshold.  
  evidence: `2018-hoffmann-wing-venation.py` L111–133; `2018-hoffmann-wing-venation.py` L164–195
- **Morphological realism / specimen match: 85** — Places the segmented real Arigomphus villosipes hindwing (panel a) directly beside the model-regenerated secondary venation on the same outline and primary scaffold (panel b), reproducing domain count (923) and proximal-to-distal cell-size gradients. Deducts 15 points because model domains are noticeably more isotropic/hexagonal than the elongated rectangular cell rows between parallel longitudinal veins in the real wing (mean circularity 0.867 vs 0.712).  
  evidence: `2018-hoffmann-wing-venation.py` L255–297; `hoffmann_wing_recapitulation.png` L0–0
- **Parameter expressivity: 68** — Exposes n_iter, seed, and downsample parameters and animates the full iteration-by-iteration equilibration trajectory (iterations 0..18) along with a live circularity trajectory in make_gif() (docs/media/hoffmann_wing_lloyd.gif), though the static figure make_figure() plots only the final n_iter=18 state and does not vary inhibition-density scaling parameters.  
  evidence: `2018-hoffmann-wing-venation.py` L271–271; `2018-hoffmann-wing-venation.py` L346–380; `docs/media/hoffmann_wing_lloyd.gif` L0–0
- **Morphometric validation: 84** — Computes domain circularity c = 4*pi*A/P^2 (using Crofton perimeter), normalised cell area, and proximal-distal circularity profiles across all 923 real and model vein domains and benchmarks both against the regular-hexagon theoretical ceiling pi*sqrt(3)/6 = 0.9069. Deducts 16 points because internal polygon vertex angles and edge-count distributions (pentagons/hexagons/heptagons) analyzed in Hoffmann et al. (2018) are not computed.  
  evidence: `2018-hoffmann-wing-venation.py` L69–69; `2018-hoffmann-wing-venation.py` L200–214; `2018-hoffmann-wing-venation.py` L298–334
- **Source and data traceability: 84** — Cites Hoffmann et al. (2018) with DOI (10.1073/pnas.1721248115), PMID (30224459), and GitHub repository URL, and fetches the official example wing Arigomphus_villosipes_LHW.png from PNAS_Paper_Code.zip. Deducts 16 points because fetch_wing_png() does not verify a pinned SHA-256 digest on the downloaded ZIP archive or extracted PNG.  
  evidence: `2018-hoffmann-wing-venation.py` L5–10; `2018-hoffmann-wing-venation.py` L65–88
- **Code-to-model traceability: 78** — Module docstring clearly maps the 5-step pipeline to SI sections 3.3-3.4 and explains where the implementation simplifies the paper's thickness-based density estimate, though individual helper functions do not cite equation or figure numbers.  
  evidence: `2018-hoffmann-wing-venation.py` L12–34; `2018-hoffmann-wing-venation.py` L164–195
- **Reproducibility package: 80** — Deterministic RNG seeding (np.random.default_rng(seed) with seed=0), CLI flags (--output, --downsample, --gif, --no-show), automatic caching in .wing_cache/, and pinned dependencies (including scikit-image and Pillow) in requirements.txt and requirements.lock.txt. Deducts 20 points because fetch_wing_png() lacks a SHA-256 check on the remote archive and the module docstring on lines 39-40 still states that scikit-image and Pillow are not in requirements.txt.  
  evidence: `2018-hoffmann-wing-venation.py` L39–40; `2018-hoffmann-wing-venation.py` L383–399; `requirements.lock.txt` L1–35

### WormGUIDES C. elegans embryo atlas with the OpenWorm connectome  —  overall 78.7 (mostly faithful)

Blinded second-assessor pass (gemini-4-argon). Decodes 360 frames of 3D nuclear coordinates and diameters from the WormGUIDES C. elegans embryogenesis atlas (Santella et al. 2015), verifies Sulston lineage naming and parent-before-child ordering across all 1,330 cells, calibrates spatial units against the frame-330 eggshell OBJ mesh (0.1649 um/unit), documents the static early shell mesh overrun, and pairs the developmental atlas with the OpenWorm c302 hermaphrodite connectome (300 neurons).

- **Model fidelity to source: 84** — Accurately decodes StarryNite/AceTree nuclei tables following AceTreeTableLineageDataLoader column conventions, parses time offsets from ProductionInfo.csv, verifies the Sulston naming grammar on all 1,330 cell names and parent-before-child ordering on 1,326/1,326 cells, reconstructs the lineage bifurcation tree directly from cell names, and calibrates spatial coordinates using the re-segmented frame-330 eggshell mesh (50 um / 303.1 units = 0.1649 um/unit). Deducts 16 points because the 3D embryo and 51 single-cell OBJ meshes are rendered as 2D orthographic projections rather than full 3D surfaces, and the c302 connectome is collapsed into a single summed adjacency matrix without separating chemical synapses from gap junctions.  
  evidence: `2015-santella-wormguides-celegans.py` L254–319; `2015-santella-wormguides-celegans.py` L335–368; `2015-santella-wormguides-celegans.py` L564–595
- **Morphological realism / specimen match: 74** — Renders empirical 4D nuclear positions scaled to true micrometre dimensions inside the segmented eggshell outline across six canonical embryonic stages, 51 3D cell-surface meshes at frame 330, and the complete Sulston lineage tree. Deducts 26 points because no raw diSPIM fluorescence volume or DIC microscopy image is shown alongside the reconstructed nuclei, and the early frames (1-~250) in the source atlas share a static eggshell mesh that the anterior nuclei overrun by up to 19.0 atlas units.  
  evidence: `2015-santella-wormguides-celegans.py` L433–541; `celegans_four_views.png` L0–0; `docs/media/celegans_embryo_timeline.gif` L0–0
- **Parameter expressivity: 50** — Data decoder and developmental atlas replay with no generative morphospace parameters to sweep; marked not_applicable per the rubric.  
  evidence: `2015-santella-wormguides-celegans.py` L1–23
- **Morphometric validation: 82** — Validates decoded atlas metrics against Sulston et al. (1983) / WormAtlas and Cook et al. (2019) benchmarks: 1,330/1,330 valid Sulston names, 1,326/1,326 parent-before-child appearances, 113/113 programmed cell deaths present, unfitted eggshell width of 29.24 um against the 30 um reference (and 13-mesh scale bracket 0.1649-0.1750 um/unit), 562-571 nuclei at 330-350 min (vs ~550), E-cell gastrulation ingression at 26 cells (vs ~30-cell stage), and 300/302 hermaphrodite neurons (with CANL/CANR explained by lack of synapses). Also quantifies and reports the 19.0-unit static-shell overrun and the ~27-minute clock offset at gastrulation (73 min vs 100 min).  
  evidence: `2015-santella-wormguides-celegans.py` L348–423; `celegans_four_views.png` L0–0
- **Source and data traceability: 92** — Pins both upstream repositories (zhirongbaolab/WormGUIDES at commit 17be611f and openworm/c302 at commit f54d92d7), verifies all 429 fetched files against SHA-256 digests in datasets/celegans/pinned_inputs.sha256 on every run, and cites DOIs for Santella et al. (2015), Sarma et al. (2018), Cook et al. (2019), Sulston et al. (1983), and Gönczy et al. (1999).  
  evidence: `2015-santella-wormguides-celegans.py` L24–49; `2015-santella-wormguides-celegans.py` L90–199; `datasets/celegans/pinned_inputs.sha256` L1–30
- **Code-to-model traceability: 82** — Explicitly documents the WormGUIDES Java loader class (AceTreeTableLineageDataLoader), ProductionInfo.csv and CellShapesConfig.csv fields, Sulston regex grammar, and literature check constants inline where they are used.  
  evidence: `2015-santella-wormguides-celegans.py` L113–124; `2015-santella-wormguides-celegans.py` L254–275
- **Reproducibility package: 86** — Fully deterministic execution, SHA-256-verified input caching in .celegans_cache/, CLI options (--out-dir, --csv, --no-anim, --step, --offline), and exact dependency lockfile in requirements.lock.txt; requires network access only on first run when .celegans_cache/ is unpopulated.  
  evidence: `2015-santella-wormguides-celegans.py` L674–723; `requirements.lock.txt` L1–35

### PlanformDB planarian morphology animation  —  overall 76.5 (mostly faithful)

Blinded second-assessor pass (gemini-4-argon). Decodes the Lobo, Malone & Levin (2013) region-graph formalism across all 369 geometric planarian morphologies in PlanformDB 2.5.0 (verifying the 0->4 / 1->3 / k->k radii rule on all 1,181 regions), renders 16 canonical phenotypes and a 63-frame historical morphospace timeline, and pins the database by SHA-256.

- **Model fidelity to source: 86** — Decodes PlanformDB 2.5.0's relational tables (Morphology, Region, RegionParam, RegionsLink, Organ, OrganPoint) using the formalism's exact link-degree radius rule (0 links -> 4 cardinal radii; 1 link -> +90/+180/+270 deg; k >= 2 links -> k bisector radii), places region centres from Dist/Ang and shared borders at Ratio*Dist, and verifies in check_encoding() that 1,181/1,181 regions obey the rule. Deducts 14 points because consecutive control radii are connected by a heuristic superellipse arc (N_FREE=2.2, N_BORDER=7.0) tuned to resemble Fig. 1D rather than a closed-form curve equation published in the paper.  
  evidence: `2013-lobo-planform-morphology.py` L279–391; `2013-lobo-planform-morphology.py` L733–747
- **Morphological realism / specimen match: 74** — Produces clean, immediately distinguishable Planform cartoon renderings across wild type (#1), double-head (#2, #24), headless (#20), cyclops (#14), lateral/ectopic heads (#71, #133), and multi-head phenotypes, matching the colour gradients and internal organ symbols (eyes, brain lobes, ventral nerve cords, pharynx) of Lobo et al. (2013) Fig. 1D. Deducts 26 points because the database encodes only coarse polar radii (3 per head), so species-specific triangular cephalic lobes, auricles, and pigmentation of real S. mediterranea or D. japonica specimens are absent and no worm photographs are compared in this script.  
  evidence: `2013-lobo-planform-morphology.py` L492–542; `planform_morphology_gallery.png` L0–0; `planform_morphospace_timeline.gif` L0–0
- **Parameter expressivity: 80** — Exercises all 369 geometric morphologies in PlanformDB 2.5.0 (369 unique region+link+organ encodings and 167 distinct body silhouettes) at a shared spatial scale within each gallery/timeline page so body size, aspect ratio, and topological branching vary visibly with the encoded parameters.  
  evidence: `2013-lobo-planform-morphology.py` L557–713
- **Morphometric validation: 56** — Computes rasterised silhouette morphometrics (ap_length, ml_width, aspect, area, head/tail/eye/pharynx counts, and rel_wt_length) in morphometrics(), verifies wild-type dimensions against the database radii (557.6 x 119.8 vs 560 x 120 Planform units), and exports a full table via write_csv(). Held below 60 because no quantitative IoU or landmark comparison against real planarian photographs or published software screenshots is performed inside the script at commit 5659d5e.  
  evidence: `2013-lobo-planform-morphology.py` L418–444; `2013-lobo-planform-morphology.py` L716–731
- **Source and data traceability: 85** — Cites both primary papers (Lobo, Malone & Levin 2013 Biology Open, DOI 10.1242/bio.20123400; Bioinformatics, DOI 10.1093/bioinformatics/btt088) and pins planformDB_2.5.0.edb by exact SHA-256 digest (PLANFORM_DB_SHA256) in both the script and datasets/planform/planformDB.sha256. Deducts 15 points because the proprietary .edb file cannot be redistributed in the repository or auto-downloaded without user setup.  
  evidence: `2013-lobo-planform-morphology.py` L10–17; `2013-lobo-planform-morphology.py` L147–177
- **Code-to-model traceability: 82** — Module docstring and function comments clearly document the database schema, the +y anterior coordinate convention, the link-degree radius rule, and the superellipse boundary construction.  
  evidence: `2013-lobo-planform-morphology.py` L18–37; `2013-lobo-planform-morphology.py` L353–385
- **Reproducibility package: 78** — Deterministic execution, SHA-256 verification with --strict-db flag, comprehensive CLI options (--out-dir, --ids, --csv, --no-anim, --no-gallery), and pinned dependencies in requirements.lock.txt; limited only by the requirement to supply an external PLANFORM_DB_PATH file.  
  evidence: `2013-lobo-planform-morphology.py` L750–787; `requirements.lock.txt` L1–35

### Cervera–Levin–Mafe gap-junction reaction–diffusion and bioelectric synthesis  —  overall 60.7 (mostly faithful)

Blinded second-assessor pass (gemini-4-argon) at commit 5659d5e (including the commit 0154143 fixes for f2_for_shift() and Panel D legend strings). Combines a local 1D reaction-diffusion analogue (AP m1/m2 + orthogonal ML m3), a deterministic Python port of the rusty-matter 2-node bioelectric circuit, a 1-parameter crop-position fit to Oviedo et al. (2010) octanol data tested on held-out innexin-RNAi data, and morphogen territory projection onto a segmented S. mediterranea photograph alongside two decoded Planform cartoons.

- **Model fidelity to source: 50** — Explicitly documents in its header scope statement that the full texts of Cervera, Levin & Mafe (2021) and Cervera et al. (2020) were not accessible and that the 1D reaction-diffusion solver (rd_antagonistic_1d and rd_linear_1d) and rusty-matter 2-node ODE port are qualitative analogues. At commit 5659d5e, f2 and f3 both drive distinct 1D boundary-value solvers, and the inverse threshold finder f2_for_shift() bisects on the unclipped wavefront x*(f2) = x*_b (recovering f2_b = 0.422 where x*(f2_b) = 0.5833). Held at 50 because the equations and constants are local constructions rather than verified transcriptions of the 2021/2020 papers, rd_antagonistic_1d's docstring (line 231) still claims to implement the paper's equations, and m3 (f3) is not coupled to any morphological boundary.  
  evidence: `2021-Cervera–Levin–Mafe.py` L27–47; `2021-Cervera–Levin–Mafe.py` L213–359
- **Morphological realism / specimen match: 48** — Segments a real Wikimedia Commons photograph of Schmidtea mediterranea (Stevenson & Beane 2010, SHA-256 verified) to measure the body-area fraction anterior to x*(f2) and renders decoded Planform #1 Wild Type and #2 Double Head cartoons. Held below 50 because the 1D signals are merely painted onto a static wild-type photograph whose outline never deforms into a double-headed worm, and the morphology readout switches categorically between two pre-drawn Planform cartoons at an assumed pharynx-posterior threshold x*_b = 0.5833.  
  evidence: `2021-Cervera–Levin–Mafe.py` L150–207; `2021-Cervera–Levin–Mafe.py` L795–869; `cervera_levin_mafe_bioelectric_morphospace.png` L0–0
- **Parameter expressivity: 62** — Sweeps f2 across [0.25, 1.0] (monotonically shifting the AP wavefront x*(f2) from 0.500 to 0.648 and anterior silhouette area from 0.483 to 0.639) and sweeps f3 across [0.30, 1.0] (reducing the ML centre/edge ratio lambda(f3) from 0.197 to 0.011), and compares 5 rusty-matter bioelectric scenarios across 3 crop positions. Deducts 38 points because f3 has no effect on the rendered 2D worm morphology and the phenotype output is a binary switch rather than a continuous morphogenetic deformation.  
  evidence: `2021-Cervera–Levin–Mafe.py` L302–359; `2021-Cervera–Levin–Mafe.py` L965–978
- **Morphometric validation: 54** — Fits a 1-parameter linear mapping f2(p) = 1 - a*p to Oviedo et al. (2010) octanol double-head frequencies across 4 axial crop positions (training RMSE = 4.19 percentage points) and evaluates it without refitting on held-out triple-innexin RNAi crops (test RMSE = 23.52 percentage points, r = 0.943), while verifying rusty-matter trace parity across all 5 scenarios (max |delta| = 0.0332) and computing Planform #1 vs #2 silhouette IoU (0.9478). Held at 54 because the 4-point empirical sample is tiny, the 23.5-point held-out error is large, and no morphometric dimensions of real regenerated double-head specimens are measured.  
  evidence: `2021-Cervera–Levin–Mafe.py` L997–1026; `cervera_levin_mafe_metrics.csv` L1–55
- **Source and data traceability: 84** — Cites DOIs and corrected PMIDs for Cervera et al. (2020, 2021), Oviedo et al. (2010), Durant et al. (2019), and Lobo et al. (2013), pins all three vendored JSON fixtures and the Wikimedia Commons photograph by SHA-256, and transparently documents which papers were consulted only via abstract/presentation.  
  evidence: `2021-Cervera–Levin–Mafe.py` L6–47; `2021-Cervera–Levin–Mafe.py` L98–130
- **Code-to-model traceability: 68** — At commit 5659d5e (following commit 0154143), model constants are centralized, f2_for_shift() properly inverts wavefront_for_f2() without clipping, and Panel D dynamically formats pulse bounds (1.0-16.0 h) and posterior head-readout percentages (91.8% vs 23.5%) from simulation state. Deducts 32 points because the 1D RD equations and the pharynx-posterior phenotype threshold (x*_b = 0.5833, which includes 20-unit plotting padding and a 100-unit cartoon pharynx convention) do not map to published equations in Cervera et al. (2021).  
  evidence: `2021-Cervera–Levin–Mafe.py` L320–359; `2021-Cervera–Levin–Mafe.py` L1271–1318
- **Reproducibility package: 88** — Fully deterministic solver and rendering pipeline, SHA-256 input verification, comprehensive --self-test gate (which now asserts exact inverse recovery of f2_b as well as symmetry, monotonicity, f3 sensitivity, and trace parity), headless CLI flags, and pinned dependencies in requirements.lock.txt.  
  evidence: `2021-Cervera–Levin–Mafe.py` L1412–1456; `2021-Cervera–Levin–Mafe.py` L1545–1605

### Chindapol flow-driven coral growth  —  overall 56.0 (partially faithful)

Blinded second-assessor pass (gemini-4-argon). Replaces Chindapol et al.'s (2013) 3D COMSOL Navier-Stokes + advection-diffusion accretive growth model of Pocillopora verrucosa with a declared 2D lattice diffusion-limited aggregation (DLA) random walk with horizontal drift, reproducing qualitative upstream-bending and compactification trends across five drift values.

- **Model fidelity to source: 38** — Honestly declares in the module docstring and in Panel (f) that the paper's coupled 3D laminar Navier-Stokes flow solver, nutrient advection-diffusion boundary layer, and normal-flux surface extension law are replaced by a 2D square-lattice biased random walk (DLA with drift + basal nutrient-shadowing survival probability). While the substitution is transparent, a stochastic particle-sticking walk on a 2D grid is a fundamentally different physical mechanism from continuum hydrodynamic advection-diffusion around a 3D branching colony, and the Peclet number Pe = u_0*L/D is not computed.  
  evidence: `2013-chindapol-coral-flow.py` L11–36; `2013-chindapol-coral-flow.py` L65–137
- **Morphological realism / specimen match: 42** — Renders five 2D dendritic lattice clusters (700 particles each) that transition from radially symmetric branching at drift=0.00 to upstream-skewed, compactified forms at drift=0.22. However, the 1-pixel-wide jagged 2D DLA trees lack the smooth, thick, club-shaped 3D branches of Pocillopora verrucosa and no CT scans or figures from Chindapol et al. (2013) are overlaid.  
  evidence: `2013-chindapol-coral-flow.py` L186–231; `docs/media/chindapol_coral_flow_growth.gif` L0–0
- **Parameter expressivity: 72** — Sweeps five drift values (0.00, 0.04, 0.09, 0.15, 0.22) in the static morphospace figure and three regimes in the growth GIF, producing clear, monotonic shifts in upstream centre-of-mass displacement (cx - mean_x), radius of gyration (r_gyr), symmetry magnitude (smmag), and mean resultant branch vector length (Rbar).  
  evidence: `2013-chindapol-coral-flow.py` L186–203
- **Morphometric validation: 48** — Computes 2D analogues of the paper's angular branch-tip orientation metrics (hangle = atan2(dx, dy), smmag = |mean(sign(hangle))|, and circular resultant length Rbar) on local distance-transform maxima in morphometrics(). Held below 50 because tip detection operates on noisy 2D lattice pixels rather than 3D skeletal medial axes, and no numerical values from Chindapol et al. (2013) Figs. 9-12 or CT-scanned coral colonies are compared.  
  evidence: `2013-chindapol-coral-flow.py` L150–179
- **Source and data traceability: 82** — Cites Chindapol et al. (2013) PLoS Comput. Biol. with full authors, title, venue, and DOI (10.1371/journal.pcbi.1002849) and requires no external dataset files.  
  evidence: `2013-chindapol-coral-flow.py` L5–10
- **Code-to-model traceability: 68** — Both the header docstring and Figure panel (f) ('From the paper to this demo') explicitly map each concept in Chindapol et al. (2013) to its 2D DLA counterpart, though individual functions do not cite specific equation numbers from the paper's Methods.  
  evidence: `2013-chindapol-coral-flow.py` L11–36; `2013-chindapol-coral-flow.py` L252–273
- **Reproducibility package: 82** — Deterministic per-regime RNG seeds (seed = 100 + i), self-contained numpy/scipy/matplotlib implementation with zero network dependencies, CLI arguments (--output, --particles, --gif, --no-show), and pinned dependencies in requirements.lock.txt.  
  evidence: `2013-chindapol-coral-flow.py` L192–193; `2013-chindapol-coral-flow.py` L403–419; `requirements.lock.txt` L1–35

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson's Scarus -> Pomacanthus transformation fitted to real species
- `thompson_scarus_pomacanthus.png` — sha256 `d243f51e2b90b1a2…`, 739,620 B, 2520×1540 px
- `thompson_scarus_pomacanthus_outlines.png` — sha256 `df6479f9f47f28fc…`, 903,004 B, 2520×1470 px

| measurement | value | reference | method |
|---|---:|---|---|
| species in the landmark table | 791 count | 791 | rows of trimmed_tps_Claverie2014.csv (sha256 checked) |
| Scarus species | 22 count | — | rows whose genus is Scarus |
| Pomacanthus species | 9 count | — | rows whose genus is Pomacanthus |
| Procrustes distance, Scarus mean to Pomacanthus mean | 0.0312 Procrustes units (RMS per landmark) | — | RMS landmark distance after optimal similarity superimposition of the genus means (unit centroid size) |
| affine residual | 0.0259 Procrustes units | — | RMS after a least-squares affine map (6 parameters) *(explains 30.9 % of the Procrustes distance)* |
| Moebius residual (isogonal coaxial circles) | 0.0311 Procrustes units | — | RMS after a least-squares Moebius map z -> (az+b)/(cz+1), the conformal map that turns the Cartesian grid into two orthogonal pencils of coaxial circles *(explains 0.8 %: a conformal map cannot change the depth/length ratio)* |
| oblique coaxial-circle residual | 0.0256 Procrustes units | — | RMS after an affine map followed by z -> z/(cz+1) (8 parameters): coaxial circles whose two pencils need not be orthogonal, the family Thompson's Fig. 376 belongs to as drawn *(explains 32.4 %)* |
| thin-plate spline bending energy | 0.6778 dimensionless | — | bending energy of the exact TPS from the Scarus mean to the Pomacanthus mean (Bookstein 1989) |
| within-Pomacanthus RMS (noise floor) | 0.0132 Procrustes units | — | RMS of the Pomacanthus species about their own Procrustes mean *(species range 0.0090-0.0180; within-Scarus 0.0099)* |
| leave-one-landmark-out RMS of the oblique coaxial family | 0.0371 Procrustes units | — | each landmark predicted by the 8-parameter map fitted to the other 16 *(compare with the Procrustes distance: the family has no predictive value for held-out landmarks if it is not smaller)* |
| depth/length, Scarus mean | 0.3772 ratio | — | landmark extent in y over extent in x |
| depth/length, Pomacanthus mean | 0.5873 ratio | — | landmark extent in y over extent in x |
| percentile of Scarus -> Pomacanthus among all genus pairs (oblique coaxial, explained share) | 31.1475 percent | — | share of the 3782 ordered pairs of genus means (genera with >= 3 spp.) explained less well *(median pair 43 %)* |
| Scarus -> Chaetodon spp., median explained by oblique coaxial circles | 35.12 percent | — | median over the 54 Chaetodon species *(range 12-61 %)* |
| silhouette area, Scarus ghobban | 214862 pixels | — | segmented mask at 0.25 scale; photograph SHA-256 732f1e68da7e... checked by the script |
| caudal-peduncle anchor, Scarus ghobban | 0.7919 fraction of silhouette length | — | x of the peduncle anchor between the anterior-most and posterior-most mask columns (fish facing left) *(plausible range for a perciform fish ~0.75-0.90; the anchor fixes the registration of the landmarks to the photograph)* |
| silhouette area, Pomacanthus semicirculatus | 92699 pixels | — | segmented mask at 0.25 scale; photograph SHA-256 5622e415149a... checked by the script |
| caudal-peduncle anchor, Pomacanthus semicirculatus | 0.8395 fraction of silhouette length | — | x of the peduncle anchor between the anterior-most and posterior-most mask columns (fish facing left) *(plausible range for a perciform fish ~0.75-0.90; the anchor fixes the registration of the landmarks to the photograph)* |
| silhouette IoU, similarity | 0.662 ratio | 0.6619 | Scarus ghobban silhouette carried by the map fitted to the 17 landmarks of S. ghobban -> P. semicirculatus, against the P. semicirculatus silhouette; point-in-polygon on a 900 x 900 grid (independent of the script's rasteriser) *(boundary distance 0.0404 = 5.6 % of body length; landmark RMS 0.0388; the no-deformation baseline)* |
| silhouette IoU, affine | 0.6515 ratio | 0.6515 | Scarus ghobban silhouette carried by the map fitted to the 17 landmarks of S. ghobban -> P. semicirculatus, against the P. semicirculatus silhouette; point-in-polygon on a 900 x 900 grid (independent of the script's rasteriser) *(boundary distance 0.0466 = 6.5 % of body length; landmark RMS 0.0322)* |
| silhouette IoU, moebius | 0.6449 ratio | 0.6448 | Scarus ghobban silhouette carried by the map fitted to the 17 landmarks of S. ghobban -> P. semicirculatus, against the P. semicirculatus silhouette; point-in-polygon on a 900 x 900 grid (independent of the script's rasteriser) *(boundary distance 0.0427 = 5.9 % of body length; landmark RMS 0.0381)* |
| silhouette IoU, coaxial | 0.6185 ratio | 0.6184 | Scarus ghobban silhouette carried by the map fitted to the 17 landmarks of S. ghobban -> P. semicirculatus, against the P. semicirculatus silhouette; point-in-polygon on a 900 x 900 grid (independent of the script's rasteriser) *(boundary distance 0.0509 = 7.0 % of body length; landmark RMS 0.0303)* |
| silhouette IoU, tps | 0.6287 ratio | 0.6287 | Scarus ghobban silhouette carried by the map fitted to the 17 landmarks of S. ghobban -> P. semicirculatus, against the P. semicirculatus silhouette; point-in-polygon on a 900 x 900 grid (independent of the script's rasteriser) *(boundary distance 0.0477 = 6.6 % of body length; landmark RMS 0.0000)* |

### Odonata wing cells through deep time
- `odonata_fossil_resegmentation.png` — sha256 `ab6f3b9f89fbcff0…`, 286,190 B, 1663×1323 px

| measurement | value | reference | method |
|---|---:|---|---|
| cell count ∝ wing length^b, fossil | 1.7811 exponent | 2.0 | OLS slope of log10 cell count on log10 wing length, n=69 wings (Eshghi et al. 2026 tables) *(r = 0.739)* |
| median cell length ∝ wing length^b, fossil | -0.1673 exponent | 0.0 | OLS slope of log10 median cell length on log10 wing length |
| median cell length over wings, fossil | 0.7837 mm | — | median over wings of the per-wing median cell length (Eshghi et al. 2026 tables) *(IQR 0.67–0.94 mm)* |
| cell count ∝ wing length^b, extant_dragonfly | 1.111 exponent | 2.0 | OLS slope of log10 cell count on log10 wing length, n=290 wings (Eshghi et al. 2026 tables) *(r = 0.816)* |
| median cell length ∝ wing length^b, extant_dragonfly | 0.2171 exponent | 0.0 | OLS slope of log10 median cell length on log10 wing length |
| median cell length over wings, extant_dragonfly | 0.9885 mm | — | median over wings of the per-wing median cell length (Eshghi et al. 2026 tables) *(IQR 0.87–1.09 mm)* |
| cell count ∝ wing length^b, extant_damselfly | 1.0078 exponent | 2.0 | OLS slope of log10 cell count on log10 wing length, n=104 wings (Eshghi et al. 2026 tables) *(r = 0.872)* |
| median cell length ∝ wing length^b, extant_damselfly | 0.2765 exponent | 0.0 | OLS slope of log10 median cell length on log10 wing length |
| median cell length over wings, extant_damselfly | 0.8991 mm | — | median over wings of the per-wing median cell length (Eshghi et al. 2026 tables) *(IQR 0.83–0.99 mm)* |
| fossil drawings re-segmented | 69 count | 69 | fossil venation drawings in the Zenodo archive segmented by the script |
| scale bar label read from the drawing | 65 count | 69 | drawings whose scale-bar label was read from the glyphs and found consistent with the tabulated wing length *(the rest use the bar with a value inferred from the tabulated wing length, or that length alone)* |
| cell count, median |ΔN|/N vs Table S1 | 0.0 ratio | 0.0 | per-wing relative difference between the script's cell count and the published count, median over 69 wings *(90% of wings within 2%; max 194% (#7, drawing and table row disagree))* |
| median cell length, median |Δ| vs Table S1 | 0.0339 ratio | 0.0 | per-wing relative difference of the median cell length (maximum Feret diameter), median over wings *(signed mean +3.5% (cells bounded by vein mid-lines))* |
| largest cell length, median |Δ| vs Table S1 | 0.0094 ratio | 0.0 | as above for the largest cell of each wing |
| fossil median cell length, re-segmented | 0.8275 mm | 0.7837 | median over wings of the script's own per-wing median cell length |

### LimbformDB limb morphologies
- `docs/media/limbform_morphospace_timeline.gif` — sha256 `1997a7a304d4f6f3…`, 2,172,060 B, 720×405 px, 66 frames
- `limbform_wildtype_gallery.png` — sha256 `d561f4aab1075697…`, 620,043 B, 2400×1875 px

| measurement | value | reference | method |
|---|---:|---|---|
| morphologies decoded | 555 count | 565 | Morphology rows with at least one Link (segment) row *(the rest are outcome labels without geometry (Lethal, No limb, ...))* |
| joints obeying the radius rule | 3137 count | 3146 | joints whose JointParam count equals the number of incident segments (3 for a terminal joint; a sub-joint counts its host segment twice) |
| segments, #1 Wild type, right forelimb | 3 count | 3 | Link rows of the morphology |
| digits, #1 Wild type, right forelimb | 4 count | 4 | Digit rows of the morphology |
| segments, #280 Wild type, right hindlimb | 3 count | 3 | Link rows of the morphology |
| digits, #280 Wild type, right hindlimb | 6 count | 6 | Digit rows of the morphology |
| segments, #314 Wild type, right walking leg, fiddler crab | 7 count | 7 | Link rows of the morphology |
| segments, #490 Wild type, right metathoracic (T3) leg, cricket nymph | 8 count | 8 | Link rows of the morphology |
| segments, #491 Wild type, right leg I, huntsman spider | 7 count | 7 | Link rows of the morphology |
| silhouette IoU vs Fig. 1 A'', #1 Wild type, right forelimb | 0.6448 ratio | 1.0 | largest ink component of the cartoon panel (holes filled) and the decoded silhouette, each cropped to its bounding box and scaled to a 200-px canvas (aspect kept); intersection over union, best of 8 rotations/reflections (rot180+mirror) *(bones, digits and gradient are inside the outline, so this compares outer shape only; the cartoon is a 560-px JPEG, which bounds the achievable IoU below 1)* |
| bounding-box aspect, #1 Wild type, right forelimb | 1.5738 ratio | 1.463 | longer / shorter side of the decoded silhouette's bounding box |
| silhouette IoU vs Fig. 1 B'', #280 Wild type, right hindlimb | 0.4105 ratio | 1.0 | largest ink component of the cartoon panel (holes filled) and the decoded silhouette, each cropped to its bounding box and scaled to a 200-px canvas (aspect kept); intersection over union, best of 8 rotations/reflections (rot0+mirror) *(bones, digits and gradient are inside the outline, so this compares outer shape only; the cartoon is a 560-px JPEG, which bounds the achievable IoU below 1)* |
| bounding-box aspect, #280 Wild type, right hindlimb | 1.4696 ratio | 1.832 | longer / shorter side of the decoded silhouette's bounding box |
| silhouette IoU vs Fig. 1 C'', #314 Wild type, right walking leg, fiddler crab | 0.4313 ratio | 1.0 | largest ink component of the cartoon panel (holes filled) and the decoded silhouette, each cropped to its bounding box and scaled to a 200-px canvas (aspect kept); intersection over union, best of 8 rotations/reflections (rot180+mirror) *(bones, digits and gradient are inside the outline, so this compares outer shape only; the cartoon is a 560-px JPEG, which bounds the achievable IoU below 1)* |
| bounding-box aspect, #314 Wild type, right walking leg, fiddler crab | 1.2414 ratio | 1.051 | longer / shorter side of the decoded silhouette's bounding box |
| silhouette IoU vs Fig. 1 D'', #490 Wild type, right metathoracic (T3) leg, cricket nymph | 0.7432 ratio | 1.0 | largest ink component of the cartoon panel (holes filled) and the decoded silhouette, each cropped to its bounding box and scaled to a 200-px canvas (aspect kept); intersection over union, best of 8 rotations/reflections (rot180+mirror) *(bones, digits and gradient are inside the outline, so this compares outer shape only; the cartoon is a 560-px JPEG, which bounds the achievable IoU below 1)* |
| bounding-box aspect, #490 Wild type, right metathoracic (T3) leg, cricket nymph | 1.1346 ratio | 1.084 | longer / shorter side of the decoded silhouette's bounding box |
| silhouette IoU vs Fig. 1 E'', #491 Wild type, right leg I, huntsman spider | 0.4851 ratio | 1.0 | largest ink component of the cartoon panel (holes filled) and the decoded silhouette, each cropped to its bounding box and scaled to a 200-px canvas (aspect kept); intersection over union, best of 8 rotations/reflections (rot180+mirror) *(bones, digits and gradient are inside the outline, so this compares outer shape only; the cartoon is a 560-px JPEG, which bounds the achievable IoU below 1)* |
| bounding-box aspect, #491 Wild type, right leg I, huntsman spider | 1.5391 ratio | 1.455 | longer / shorter side of the decoded silhouette's bounding box |

### Raup shell coiling morphospace
- `raup_empirical_comparison.png` — sha256 `c50dbee57d0a6495…`, 562,658 B, 1630×1070 px
- `raup_shell_animation.gif` — sha256 `986555a0032be189…`, 15,189,434 B, 1200×800 px, 100 frames

| measurement | value | reference | method |
|---|---:|---|---|
| recovered translation rate T, 'Planispiral' (W=3.0, D=0.0, T=0.0) | 0.0 ratio | 0.0 | Δz_c / Δr_c of the generating-curve centre over the last revolution of raup_shell() *(height/width = 0.629, spire/width = 0.000)* |
| recovered distance from axis D, 'Planispiral' (W=3.0, D=0.0, T=0.0) | 0.0003 ratio | 0.0 | inner margin / outer margin of the aperture (min/max radial distance of the last generating curve) |
| recovered whorl expansion W, 'Planispiral' (W=3.0, D=0.0, T=0.0) | 3.0 ratio | 3.0 | centre distance from the axis at the aperture / one revolution earlier |
| recovered translation rate T, 'Turreted' (W=1.3, D=0.05, T=3.5) | 3.5 ratio | 3.5 | Δz_c / Δr_c of the generating-curve centre over the last revolution of raup_shell() *(height/width = 1.142, spire/width = 0.858)* |
| recovered distance from axis D, 'Turreted' (W=1.3, D=0.05, T=3.5) | 0.0502 ratio | 0.05 | inner margin / outer margin of the aperture (min/max radial distance of the last generating curve) |
| recovered whorl expansion W, 'Turreted' (W=1.3, D=0.05, T=3.5) | 1.3 ratio | 1.3 | centre distance from the axis at the aperture / one revolution earlier |
| recovered translation rate T, 'Evolute disc' (W=2.0, D=0.45, T=0.0) | 0.0 ratio | 0.0 | Δz_c / Δr_c of the generating-curve centre over the last revolution of raup_shell() *(height/width = 0.321, spire/width = 0.000)* |
| recovered distance from axis D, 'Evolute disc' (W=2.0, D=0.45, T=0.0) | 0.4501 ratio | 0.45 | inner margin / outer margin of the aperture (min/max radial distance of the last generating curve) |
| recovered whorl expansion W, 'Evolute disc' (W=2.0, D=0.45, T=0.0) | 2.0 ratio | 2.0 | centre distance from the axis at the aperture / one revolution earlier |
| recovered translation rate T, 'Conical' (W=3.5, D=0.0, T=1.5) | 1.5 ratio | 1.5 | Δz_c / Δr_c of the generating-curve centre over the last revolution of raup_shell() *(height/width = 0.805, spire/width = 0.481)* |
| recovered distance from axis D, 'Conical' (W=3.5, D=0.0, T=1.5) | 0.0003 ratio | 0.0 | inner margin / outer margin of the aperture (min/max radial distance of the last generating curve) |
| recovered whorl expansion W, 'Conical' (W=3.5, D=0.0, T=1.5) | 3.5 ratio | 3.5 | centre distance from the axis at the aperture / one revolution earlier |
| recovered translation rate T, 'High-spired' (W=1.6, D=0.1, T=2.5) | 2.5 ratio | 2.5 | Δz_c / Δr_c of the generating-curve centre over the last revolution of raup_shell() *(height/width = 0.998, spire/width = 0.738)* |
| recovered distance from axis D, 'High-spired' (W=1.6, D=0.1, T=2.5) | 0.1002 ratio | 0.1 | inner margin / outer margin of the aperture (min/max radial distance of the last generating curve) |
| recovered whorl expansion W, 'High-spired' (W=1.6, D=0.1, T=2.5) | 1.6 ratio | 1.6 | centre distance from the axis at the aperture / one revolution earlier |
| recovered translation rate T, 'Globose' (W=3.0, D=0.1, T=1.0) | 1.0 ratio | 1.0 | Δz_c / Δr_c of the generating-curve centre over the last revolution of raup_shell() *(height/width = 0.628, spire/width = 0.341)* |
| recovered distance from axis D, 'Globose' (W=3.0, D=0.1, T=1.0) | 0.1002 ratio | 0.1 | inner margin / outer margin of the aperture (min/max radial distance of the last generating curve) |
| recovered whorl expansion W, 'Globose' (W=3.0, D=0.1, T=1.0) | 3.0 ratio | 3.0 | centre distance from the axis at the aperture / one revolution earlier |

### Gray-Scott reaction-diffusion patterns
- `turing_patterns.gif` — sha256 `0ddf48cbca6cf3df…`, 14,075,347 B, 1280×960 px, 500 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dominant wavelength (last frame) | 14.2857 grid cells | — | peak of the radial power spectrum of the axes-box image, scaled to the N=200 grid *(Gray–Scott f=0.055,k=0.062 labyrinth; no specimen or literature wavelength recorded)* |
| pattern coverage (last frame) | 0.2477 fraction | — | fraction of axes box in the high-V (yellow) band |
| connected high-V components (last frame) | 1 count | — | scipy.ndimage.label on the yellow mask |
| fraction of pattern pixels changed, first→last frame | 0.252 fraction | >0 for an evolving simulation | XOR of yellow masks of frame 0 and the last frame *(≈0 means the GIF shows a frozen pattern)* |
| Pearson coral labyrinth (F=0.055, k=0.062) spectral ring wavenumber k* | 11.0 cycles / domain | >0 (Turing band-pass ring) | peak of 2D radial PSD P(k) on N=160x160 grid after 4000 steps *(λ* = 14.55 grid cells, ring sharpness = 718.1)* |
| Pearson mitosis spots (F=0.028, k=0.062) connected components vs coral labyrinth | 129 count | 14 | connected high-activator components in self-replicating spot regime *(spots remain disconnected (>100 components) whereas coral labyrinth percolates)* |
| equal-diffusion negative control (Du=Dv=0.10) spectral wavenumber k* | 0.0 cycles / domain | 0.0 | dominant wavenumber when differential diffusivity is removed (Du = Dv = 0.10 at F=0.055, k=0.062) *(spatial σ_V = 0.00e+00, ring sharpness = 0.0)* |

### Hoffmann-Donoughe insect wing venation
- `docs/media/hoffmann_wing_lloyd.gif` — sha256 `e53e2133c7bebee1…`, 901,133 B, 720×403 px, 19 frames
- `hoffmann_wing_recapitulation.png` — sha256 `43b99ec8ae8205b3…`, 179,027 B, 1730×1206 px

| measurement | value | reference | method |
|---|---:|---|---|
| mean domain circularity, model wing | 0.867 ratio (4πA/P²) | 0.712 | script's domain_stats() over Voronoi domains after Lloyd equilibration (seed 0) *(regular hexagon = 0.907; model n=923, real n=923)* |
| vein-domain count, model wing | 923 count | 923 | domains large enough to measure, model *(equal by construction: the model is seeded with the real wing's domain count)* |
| vein-domain count, real wing (all) | 923 count | — | peaks of the distance transform inside the segmented real wing *(wing area 1905316 px)* |

### WormGUIDES C. elegans embryo atlas with the OpenWorm connectome
- `celegans_four_views.png` — sha256 `eb4b60086280c9a1…`, 424,197 B, 2240×1680 px
- `docs/media/celegans_embryo_timeline.gif` — sha256 `e8d760778e3dbd34…`, 2,445,902 B, 720×374 px, 60 frames

| measurement | value | reference | method |
|---|---:|---|---|
| lineage names parsing as Sulston names | 1330 count | 1330 | distinct names in the 360 nuclei files matching ^(AB|MS|E|C|D|EMS|P[0-4]|Z[23])[aplrdv]*$ |
| cells first seen after their parent | 1326 count | 1326 | non-founder cells whose parent name appears in an earlier frame |
| calibration | 0.1649 µm per atlas unit | — | x extent of the atlas's eggshell outline mesh at frame 330 (obj_files/embryo_t330.obj, 'Embryo Outline' in CellShapesConfig.csv; 303.1 units) set to the 50 µm egg length *(assumption: Gönczy et al. 1999 JCB 144:927 ('50-µm-long cell'); the atlas coordinates are unit-less (ProductionInfo X/Y/Z_SCALE = 1); the earlier nuclear-cloud estimate gives 0.1617; the sampled shell meshes give 0.1649-0.1750)* |
| eggshell width under the calibration | 29.238 µm | 30.0 | y extent of the eggshell outline mesh at frame 330 (z extent 28.9 µm) *(a consistency check of the calibration: the width is not fitted)* |
| eggshell length spread across sampled shell meshes | 17.3733 atlas units | — | max - min x extent of the eggshell meshes at frames [1, 30, 60, 100, 200, 250, 280, 300, 320, 330, 340, 350, 359] (285.8-303.1) *(one static mesh is reused for frames 1-~250, re-segmented per frame from ~280; the spread bounds the scale uncertainty (~6 %))* |
| nuclear overrun of the shell in x | 19.0129 atlas units | — | largest distance by which a nucleus (centroid ± radius) lies outside the shell mesh's x range at the sampled frames *(1.8 units at frame 330; the overrun is against the static early mesh, which is why the calibration uses the re-segmented shell)* |
| nuclear cloud width at the last frame | 27.28 µm | — | y span of nuclear centroids + one mean diameter under the calibration *(nuclei do not reach the eggshell; compare the eggshell width above)* |
| nuclei at the last frame | 580 count | 558 | valid rows of t360-nuclei (movie-minute 379) *(the movie ends before hatching; 553 of the 580 nuclei carry a parts-list identity)* |
| maximum nucleus count | 585 count | 550 | maximum over the 360 frames |
| deaths listed by the atlas | 113 count | 113 | names in CellDeaths.csv |
| listed dying cells that occur in the movie | 113 count | 113 | CellDeaths.csv names present in at least one nuclei file |
| gastrulation onset, cell count | 26 count | 30 | first frame from which the E cells sit at < 0.8 of the median normalised radius for 5 frames |
| gastrulation onset, movie-minute | 73 min | 100 | same frame, frame index + movie start offset *(discrepancy: the atlas clock runs ~27 min ahead of the Sulston schedule at the same cell stage; not resolved by the script (temperature or time origin))* |
| nuclei at movie-minute 330-350 | 562-571 count | 550 | nucleus count over that window |
| neurons in the connectome edge list | 300 count | 302 | upper-case node names of c302 herm_full_edgelist.csv after DA01->DA1 canonicalisation *(CANL/CANR are 'no synapses' in the WormGUIDES parts list and so cannot appear in an edge list)* |

### PlanformDB planarian morphology animation
- `planform_morphology_gallery.png` — sha256 `77a7dfa087bf8944…`, 443,915 B, 1920×2160 px
- `planform_morphospace_timeline.gif` — sha256 `f7c8cadf3c06c835…`, 3,265,519 B, 1500×780 px, 63 frames

| measurement | value | reference | method |
|---|---:|---|---|
| wild-type silhouette AP length | 557.6 Planform units | 560 | union of decoded region outlines of Morphology 1, rasterised; bbox extent along the AP axis |
| wild-type silhouette ML width | 119.8 Planform units | 120 | as above, across the ML axis |
| wild-type AP/ML aspect | 4.653 ratio | 4.667 | AP length / ML width of the decoded silhouette *(S. mediterranea in Lobo et al. 2013 Fig. 1A is of this order; the formalism, not a specimen, is the reference)* |
| morphologies rendered | 369 count | — | Morphology rows with at least one encoded region *(372 rows in the database; 3 have no regions (Lethal, Midbody Buldge, Double head, side tail))* |
| distinct body silhouettes | 167 count | 369 | unique (AP length, ML width, area to 100 units, #heads, #tails) tuples over all rendered morphologies *(115 morphologies share the wild-type body and differ only in organs (ectopic eyes, doubled CNS, pharynx position, ...), which is how the database encodes them)* |
| distinct encodings (regions + links + organs) | 369 count | 369 | unique tuples of every encoded number read from the database |
| IoU wild type vs double head (#24) | 0.8884 ratio | < 1; the two bodies differ only at the posterior cap (tail -> second head), the rest of the difference is organs | intersection-over-union of the two decoded silhouettes on a shared canvas (region centres aligned by layout) |
| IoU wild type vs headless (#20) | 0.6116 ratio | ≈ (trunk+tail area)/(whole-body area) if the head is simply absent | as above |

### Cervera–Levin–Mafe gap-junction reaction–diffusion and bioelectric synthesis
- `cervera_levin_mafe_bioelectric_morphospace.png` — sha256 `3f521835636f3530…`, 547,708 B, 2330×1781 px
- `cervera_levin_mafe_gj_block.gif` — sha256 `214238f0d75fdcfa…`, 1,398,777 B, 560×378 px, 46 frames

| measurement | value | reference | method |
|---|---:|---|---|
| rusty-matter trace parity, max |Δ| | 0.0332 state units | 0.0 | maximum absolute difference across all 5 scenarios, 11 sampled steps and 4 state averages against rusty_matter_planarian_traces.json |
| symmetric RD wavefront x*(f2=1) | 0.5 AP fraction | 0.5 | zero crossing of m1(x) - m2(x) when D2 = D2_0 |
| pharynx-boundary criterion x*_b | 0.5833 AP fraction | — | posterior end of the Planform #1 wild-type pharynx on the AP extent *(crossed at f2_b = 0.4197 (where x*(f2_b) = 0.5833))* |
| anterior body-area fraction on S. mediterranea silhouette (f2=1 -> f2=0.25) | 0.1561 fraction of body area | — | increase in real-silhouette mask area anterior to x*(f2) from f2=1.0 (0.483) to f2=0.25 (0.639) |
| ML centre/edge ratio λ(f3), f3=1.0 vs f3=0.3 | 0.1855 ratio difference | >0 (monotonic decrease under lateral gap-junction blockade) | m3(L_z/2)/m3(0) drop from f3=1.0 (0.197) to f3=0.3 (0.011) |
| Oviedo 2010 octanol crop-gradient training RMSE | 4.1871 percentage points | 0.0 | 1-parameter fit f2(p) = 1 - 0.58*p against Oviedo et al. 2010 octanol double-head frequencies (Exps 415-418, n=4 crops) |
| Oviedo 2010 triple-innexin RNAi held-out RMSE | 23.5215 percentage points | 0.0 | held-out prediction error of the octanol-fitted curve against Dj-Inx-12 + Dj-Inx-5/13 RNAi double-head frequencies (Exps 446-449, n=4 crops) *(Pearson r = 0.943)* |
| circuit posterior head readout, memory ON vs memory OFF control | 0.9185 readout [0,1] | 0.23491399812436778 | final posterior head-identity readout R_head,post under TransientDepolarizationMemory *(hysteresis memory sustains depolarised head fate after transient pulse ends)* |
| Planform #1 Wild Type vs #2 Double Head silhouette IoU | 0.9478 ratio | <1.0 (posterior tail replaced by ectopic head cap) | 256x256 shared-canvas intersection-over-union of decoded Lobo et al. (2013) graph silhouettes |

### Chindapol flow-driven coral growth
- `chindapol_coral_flow_morphospace.png` — sha256 `c431f9a5ae368f84…`, 235,376 B, 1726×1231 px
- `docs/media/chindapol_coral_flow_growth.gif` — sha256 `30eaa02efc49579c…`, 737,329 B, 720×288 px, 48 frames

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.5.4
- scipy: 1.16.3
- pandas: 2.3.3
- scikit-image: 0.26.0
- matplotlib: 3.10.7
- Pillow: 12.0.0

All ten active evaluated scripts are inspected at commit 5659d5e (HEAD). Eight of the ten target scripts at 5659d5e are byte-for-byte identical to their primary-audit commits (232e8f8 for thompson-scarus-pomacanthus, fbbcaf6 for eshghi-odonata-wing-cells, 747c57f for lobo-limbform-morphology, ba6fd20 for raup-shell-coiling, f3fe478 for turing-gray-scott, hoffmann-wing-venation, and chindapol-coral-flow, and 303a87e for santella-wormguides-celegans); 2013-lobo-planform-morphology.py includes the SHA-256 database pin added after 161f269, and 2021-Cervera–Levin–Mafe.py includes the commit 0154143 fix for the f2_for_shift() threshold inversion and dynamic Panel D legend after a0d6848. Repository-wide dependency lockfiles (requirements.txt, requirements.lock.txt) added in commit 9ca5457 are present.
