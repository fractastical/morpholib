# MorphoEval report — audit `2026-10-13-morphology-audit`

Rubric **0.1.0** · checkpoint `f3fe478` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are 0–100 (emitted as `pk:weightOfEvidence` = score/100); Overall is the rubric-weighted mean. Unevaluated scripts carry a claim but no assessments.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Faithfulness | Reproducibility | **Overall** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PlanformDB planarian morphology animation (`1900-planformDB_parser.py`) | 10 | 5 | 5 | 0 | 55 | 15 | 35 | 5.4 | 36.3 | **14.7** | not faithful |
| Raup shell coiling morphospace (`create_raup_animations.py`) | 35 | 30 | 25 | 20 | 65 | 45 | 60 | 28.6 | 57.7 | **37.3** | unfaithful |
| Hoffmann-Donoughe insect wing venation (`2018-hoffmann-wing-venation.py`) | 85 | 85 | 65 | 80 | 90 | 75 | 50 | 81.1 | 70.0 | **77.8** | mostly faithful |
| Chindapol flow-driven coral growth (`2013-chindapol-coral-flow.py`) | 40 | 45 | 70 | 55 | 80 | 65 | 65 | 49.3 | 70.0 | **55.5** | partially faithful |
| Gray-Scott reaction-diffusion patterns (`1952-turing-morpho.py`) | 85 | 50 | 55 | 20 | 55 | 55 | 40 | 54.3 | 49.0 | **52.7** | partially faithful |
| Petroff stromatolite morphospace (`2010-petroff-stromatolite.py`) | — | — | — | — | — | — | — | — | — | *unevaluated* | — |
| Kahng coral geometry morphospace (`2024-kahng-coral-geometry.py`) | — | — | — | — | — | — | — | — | — | *unevaluated* | — |
| Cervera-Levin-Mafe bioelectric morphospace (`2021-Cervera–Levin–Mafe.py`) | — | — | — | — | — | — | — | — | — | *unevaluated* | — |

## Findings and remediation

### PlanformDB planarian morphology animation  —  overall 14.7 (not faithful)

Never resembles a planarian. Every frame is the same rounded blob with two lateral flaps, changing only colour; double-head, headless and wild-type morphologies are visually indistinguishable.

- **Model fidelity to source: 10** — PlanformDB encodes a morphology as a region graph (Region + RegionParam shape parameters, joined by RegionsLink Dist/Ratio/Ang). The script queries all of it and never uses it to draw; the body is instead a convex hull of organ landmark points.  
  evidence: `1900-planformDB_parser.py` L570–586; `1900-planformDB_parser.py` L617–617; `1900-planformDB_parser.py` L304–327
- **Morphological realism / specimen match: 5** — No frame is recognisable as a planarian; the lateral 'flaps' are the eye/brain vertices of the organ hull. Reference: Planform's own rendering of each Morphology.Id and wild-type S. mediterranea.  
  evidence: `planform_morphology_animation.gif` L0–0; `1900-planformDB_parser.py` L670–687
- **Parameter expressivity: 5** — All ~360 morphologies are normalised into the same unit box and drawn at the same centre and scale, erasing the size/proportion differences that distinguish them.  
  evidence: `1900-planformDB_parser.py` L846–859; `1900-planformDB_parser.py` L300–314
- **Morphometric validation: 0** — No morphometrics are computed or compared.  
  evidence: `1900-planformDB_parser.py` L768–887
- **Source and data traceability: 55** — Input database is a resolvable Planform artefact and the schema is followed, but the script cites no DOI for Planform and requires an external PLANFORM_DB_PATH with no recorded version/hash.  
  evidence: `1900-planformDB_parser.py` L15–16
- **Code-to-model traceability: 15** — No modelling step is tied to the Planform paper; name-based fallback shapes are hard-coded polygons with no biological basis, and a dead code path references an undefined variable.  
  evidence: `1900-planformDB_parser.py` L428–494; `1900-planformDB_parser.py` L498–498
- **Reproducibility package: 35** — Runs deterministically from the DB but depends on an unversioned external file, uses pandas/scipy without pins, and VNC rendering sorts two bilateral cords by Y into one polyline.  
  evidence: `1900-planformDB_parser.py` L758–765

Remediation:
- Render from the region graph (Region + RegionParam + RegionsLink), not an organ hull.
- Use one shared scale (wild-type trunk = 1) and fix the double division.
- Lay morphologies out as small multiples instead of superimposing at the origin.
- Add a --compare mode rendering a Morphology.Id beside Planform's reference drawing.

### Raup shell coiling morphospace  —  overall 37.3 (unfaithful)

The translation parameter T has no visible effect: Turritella (T=0.80), Cerithium (T=1.20) and Conus render as the same flat planispiral disc as Nautilus (T=0.10).

- **Model fidelity to source: 35** — W (whorl expansion) and D (distance from axis) are implemented, but axial translation is zc = T*theta (linear in angle) while the radius grows as exp(b*theta). Over 6 turns at W=2.5 the radius reaches ~244 and Z only ~30, so T is swamped. In Raup's model translation scales with whorl size (z proportional to T*r).  
  evidence: `create_raup_animations.py` L41–41; `create_raup_animations.py` L46–46; `1966-raup.py` L39–39
- **Morphological realism / specimen match: 30** — Planispiral forms are plausible; all high-spired / conical reference taxa are wrong because the spire never develops.  
  evidence: `raup_shell_animation.gif` L0–0
- **Parameter expressivity: 25** — W and D act as expected; T does not act at all.  
  evidence: `create_raup_animations.py` L46–46
- **Morphometric validation: 20** — Reference taxa are named per panel but no measured shape quantity (e.g. spire height / aperture width, apical angle) is computed or compared against specimens or Raup's figures.  
  evidence: `create_raup_animations.py` L160–164
- **Source and data traceability: 65** — Raup 1966 is the acknowledged source and datasets/raup/variables.json names the empirical targets, but the script carries no DOI/JSTOR identifier.  
  evidence: `create_raup_animations.py` L21–35
- **Code-to-model traceability: 45** — Parameters are named after Raup's but the translation equation departs from the source without comment.  
  evidence: `create_raup_animations.py` L21–58
- **Reproducibility package: 60** — Deterministic, no external data; extra 50% padding leaves shells at ~1/3 of panel.  
  evidence: `create_raup_animations.py` L148–148

Remediation:
- Change zc = T*theta to zc = T*r (with the same (1+offset) scaling) in both scripts.
- Add a regression check that z-extent/xy-extent increases monotonically with T.
- Compare each named panel to a reference photograph of its taxon.

### Hoffmann-Donoughe insect wing venation  —  overall 77.8 (mostly faithful)

Follows the authors' own procedure on their own example wing: real silhouette and primary-vein scaffold, secondary veins regenerated by Lloyd equilibration + Voronoi at the real local density. 923 real vein domains reproduced; circularity real 0.712 vs model 0.867 (hexagon ceiling 0.907).

- **Model fidelity to source: 85** — Implements the paper's SI 3.3-3.4 model (inhibitory centres, Lloyd/centroidal-Voronoi equilibration, Voronoi secondary veins) on the real primary scaffold. Density is taken from the real wing's distance transform rather than from primary-vein thickness as in the paper; the equilibration stop criterion is a fixed 18 iterations rather than the paper's 0.1%-of-wing-length motion threshold.  
  evidence: `2018-hoffmann-wing-venation.py` L150–185; `2018-hoffmann-wing-venation.py` L111–133
- **Morphological realism / specimen match: 85** — Output is built on the real specimen outline and real primary veins; panel (b) is recognisably the same wing as panel (a).  
  evidence: `hoffmann_wing_recapitulation.png` L0–0
- **Parameter expressivity: 65** — n_iter and seed density are exposed, but no sweep figure shows the paper's random-to-equilibrated transition in circularity.  
  evidence: `2018-hoffmann-wing-venation.py` L150–150
- **Morphometric validation: 80** — Circularity (Crofton perimeter), normalised area and proximal-distal circularity are computed for real and model domains and compared against the regular-hexagon ceiling, matching the paper's analysis.  
  evidence: `2018-hoffmann-wing-venation.py` L188–201; `2018-hoffmann-wing-venation.py` L69–69
- **Source and data traceability: 90** — DOI, PMID and code repository cited; input image fetched from the authors' PNAS_Paper_Code.zip by URL and cached.  
  evidence: `2018-hoffmann-wing-venation.py` L5–10; `2018-hoffmann-wing-venation.py` L65–67
- **Code-to-model traceability: 75** — Docstring maps steps 1-5 to the paper's procedure; individual functions are not annotated with SI section numbers.  
  evidence: `2018-hoffmann-wing-venation.py` L12–32
- **Reproducibility package: 50** — Deterministic (seed=0) and documented, but scikit-image/Pillow are not in requirements.txt, dependencies are unpinned, and the result was silently wrong under numpy 2.2.6 + Python 3.14 until numpy was upgraded during this audit.  
  evidence: `2018-hoffmann-wing-venation.py` L39–40

Remediation:
- Add scikit-image and Pillow to requirements.txt and pin numpy >= 2.5 for Python 3.14.
- Add a circularity-vs-iteration sweep panel to show the equilibration transition.
- Optionally switch the Lloyd stop rule to the paper's 0.1%-of-wing-length threshold.

### Chindapol flow-driven coral growth  —  overall 55.5 (partially faithful)

Declared reduction: the paper's 3-D Navier-Stokes + advection-diffusion accretive growth (solved in COMSOL) is replaced by 2-D DLA-with-drift, with drift bias standing in for the Peclet number. Qualitative symmetric-to-upstream transition and compactification are recovered.

- **Model fidelity to source: 40** — Honest, documented stand-in, but it does not solve the flow or the nutrient field; the Peclet number is a drift bias, not u*L/D. The physics is reproducible with open-source solvers in 2-D.  
  evidence: `2013-chindapol-coral-flow.py` L27–36; `2013-chindapol-coral-flow.py` L65–122
- **Morphological realism / specimen match: 45** — Top-down 2-D lattice aggregates; branching and upstream bias are plausible but not Pocillopora-like in 3-D form.  
  evidence: `2013-chindapol-coral-flow.py` L205–231
- **Parameter expressivity: 70** — Drift (Pe proxy) sweep produces the monotonic upstream-shift and compaction trends.  
  evidence: `2013-chindapol-coral-flow.py` L186–196
- **Morphometric validation: 55** — Implements the paper's symmetry magnitude (smmag) and branch alignment after its morphometric method, but no numeric comparison with the paper's Figs 9-12 values is made.  
  evidence: `2013-chindapol-coral-flow.py` L164–179
- **Source and data traceability: 80** — DOI cited; no external data needed.  
  evidence: `2013-chindapol-coral-flow.py` L5–9
- **Code-to-model traceability: 65** — Figure panel (f) maps paper concepts to demo components; equations are not cited by section.  
  evidence: `2013-chindapol-coral-flow.py` L252–273
- **Reproducibility package: 65** — Seeded (seed=100+i), self-contained, documented CLI; unpinned deps.  
  evidence: `2013-chindapol-coral-flow.py` L192–193

Remediation:
- Replace DLA-with-drift with a real 2-D flow + advection-diffusion + accretive-growth solver.
- Compare smmag / alignment curves numerically against the paper's reported values.

### Gray-Scott reaction-diffusion patterns  —  overall 52.7 (partially faithful)

Standard Gray-Scott output; forms and dynamics are as expected.

- **Model fidelity to source: 85** — Canonical Gray-Scott update with feed/kill parameters in Pearson's spotted regime.  
  evidence: `1952-turing-morpho.py` L9–9; `1952-turing-morpho.py` L31–40
- **Morphological realism / specimen match: 50** — Not tied to a specific organism; neutral baseline (not applicable).  
  evidence: `turing_patterns.gif` L0–0
- **Parameter expressivity: 55** — Single (f, k) point rendered; alternative regimes mentioned only in a comment.  
  evidence: `1952-turing-morpho.py` L9–9
- **Morphometric validation: 20** — No wavelength / spot-spacing measurement against Pearson's phase diagram.  
  evidence: `1952-turing-morpho.py` L31–75
- **Source and data traceability: 55** — Model named in the title; no DOI for Turing 1952 or Pearson 1993 in the script.  
  evidence: `1952-turing-morpho.py` L74–74
- **Code-to-model traceability: 55** — Update equations are commented inline; no reference to the source equations.  
  evidence: `1952-turing-morpho.py` L37–39
- **Reproducibility package: 40** — RNG is not seeded, so each run produces a different pattern; output filename hard-coded.  
  evidence: `1952-turing-morpho.py` L19–20; `1952-turing-morpho.py` L75–75

Remediation:
- Seed the RNG and expose (f, k) and output path on the CLI.
- Measure the pattern wavelength and place the run on Pearson's (f, k) phase diagram.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (`schema:PropertyValue` evidence on the morphometric-validation assessment).

### PlanformDB planarian morphology animation
- `planform_morphology_animation.gif` — sha256 `27256acdd26eee57…`, 802,794 B, 3200×2000 px, 34 frames

| measurement | value | reference | method |
|---|---:|---|---|
| silhouette width/height (bbox, mean over frames) | 1.9923 ratio | ≈0.15–0.3 (adult S. mediterranea is ~3–7× longer than wide, long axis = AP) | largest non-white connected component in the central panel of each GIF frame; bbox W/H *(min 1.972, max 2.014; rendered body is wider than long)* |
| silhouette IoU between frames (mean) | 0.9908 ratio | distinct morphologies should give low IoU; one body plan over time should be near 1 | pairwise intersection-over-union of the per-frame silhouettes on the fixed canvas *(min 0.979; frames differ mainly by added translucent flaps, not by body form)* |
| frames with a silhouette | 34 count | — | frames where a component was found *(of 34)* |

### Raup shell coiling morphospace
- `raup_shell_animation.gif` — sha256 `aea20e93a9a79a94…`, 24,248,756 B, 3600×2400 px, 161 frames

| measurement | value | reference | method |
|---|---:|---|---|
| effective translation rate T_eff, 'Planispiral' (nominal T=0.1) | 0.0093 ratio | 0.1 | Δz over the last whorl ÷ generating-curve centre radius, from raup_shell(W,D,T,turns=6) *(shell height/width = 0.028)* |
| effective translation rate T_eff, 'Turreted' (nominal T=0.8) | 0.0175 ratio | 0.8 | Δz over the last whorl ÷ generating-curve centre radius, from raup_shell(W,D,T,turns=6) *(shell height/width = 0.052)* |
| effective translation rate T_eff, 'Disc-like' (nominal T=0.2) | 0.0005 ratio | 0.2 | Δz over the last whorl ÷ generating-curve centre radius, from raup_shell(W,D,T,turns=6) *(shell height/width = 0.001)* |
| effective translation rate T_eff, 'Conical' (nominal T=0.5) | 0.0249 ratio | 0.5 | Δz over the last whorl ÷ generating-curve centre radius, from raup_shell(W,D,T,turns=6) *(shell height/width = 0.074)* |
| effective translation rate T_eff, 'High-spired' (nominal T=1.2) | 0.1945 ratio | 1.2 | Δz over the last whorl ÷ generating-curve centre radius, from raup_shell(W,D,T,turns=6) *(shell height/width = 0.583)* |
| effective translation rate T_eff, 'Wide' (nominal T=0.3) | 0.0019 ratio | 0.3 | Δz over the last whorl ÷ generating-curve centre radius, from raup_shell(W,D,T,turns=6) *(shell height/width = 0.006)* |

### Hoffmann-Donoughe insect wing venation
- `hoffmann_wing_recapitulation.png` — sha256 `43b99ec8ae8205b3…`, 179,027 B, 1730×1206 px

| measurement | value | reference | method |
|---|---:|---|---|
| mean domain circularity, model wing | 0.867 ratio (4πA/P²) | 0.712 | script's domain_stats() over Voronoi domains after Lloyd equilibration (seed 0) *(regular hexagon = 0.907; model n=923, real n=923)* |
| vein-domain count, model wing | 923 count | 923 | domains large enough to measure, model *(equal by construction: the model is seeded with the real wing's domain count)* |
| vein-domain count, real wing (all) | 923 count | — | peaks of the distance transform inside the segmented real wing *(wing area 1905316 px)* |

### Chindapol flow-driven coral growth
- `chindapol_coral_flow_morphospace.png` — **not on disk at snapshot time** (no hash, no measurements)

### Gray-Scott reaction-diffusion patterns
- `turing_patterns.gif` — sha256 `0ddf48cbca6cf3df…`, 14,075,347 B, 1280×960 px, 500 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dominant wavelength (last frame) | 14.2857 grid cells | — | peak of the radial power spectrum of the axes-box image, scaled to the N=200 grid *(Gray–Scott f=0.055,k=0.062 labyrinth; no specimen or literature wavelength recorded)* |
| pattern coverage (last frame) | 0.2477 fraction | — | fraction of axes box in the high-V (yellow) band |
| connected high-V components (last frame) | 1 count | — | scipy.ndimage.label on the yellow mask |
| fraction of pattern pixels changed, first→last frame | 0.252 fraction | >0 for an evolving simulation | XOR of yellow masks of frame 0 and the last frame *(≈0 means the GIF shows a frozen pattern)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.5.4
- scipy: 1.16.3
- scikit-image: 0.26.0

numpy was upgraded 2.2.6 -> 2.5.4 during this audit after 2.2.6 on Python 3.14 was found to silently corrupt large boolean-array results (wing mask collapsed from 1,905,316 px to 25,146 px, deterministically).
