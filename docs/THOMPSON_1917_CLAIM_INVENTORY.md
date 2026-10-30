# Thompson (1917) — Every Claim, Cited, Then Assessed for Validation

**Source for every row below (state the date whenever you cite one):**
Thompson, D'Arcy Wentworth (**1917**). *On Growth and Form*. 1st edition, Cambridge University Press.
Internet Archive item [`ongrowthform1917thom`](https://archive.org/details/ongrowthform1917thom) (public domain);
full-text SHA-256 `1ba78d3708fba8c8cf1b819cf951dad72ccb0ac34fb3dd13e5c23745b68f9ff0`; printed page = IA leaf − 20.
Not the 1942 2nd edition and not the 1961 abridgement. Machine-readable registry (the source of truth for this file):
[`thompson/thompson_1917_claims.json`](../thompson/thompson_1917_claims.json), regenerated/checked by
[`thompson/build_1917_claim_inventory.py`](../thompson/build_1917_claim_inventory.py).

## How to read this (the two passes)

- **Pass 1 — cite it.** One row per distinct testable assertion or figure-demonstration Thompson made in **1917**,
  cited by chapter, printed pages and figure numbers. Where Thompson repeats one assertion across several named
  example pairs (Ch. XVII), each pair is its own row because each can pass or fail separately. Verbatim quotes are
  included only where already pinned in repo code; completing verbatim-quote extraction for every row is Pass-1
  extension item E1 below (OCR-checked, ~3–5 days). Rhetorical/methodological passages are rows T1917-001/002/059
  and must not be scored as if they were measurements.
- **Pass 2 — can we validate it, and what would it cost?** Each row carries: status against this repo's actual
  artifacts, the verdict where one exists, what remains, the method and data that would settle it, and effort in
  person-days (S ≤2d, M 3–10d, L 2–6 wk, XL >6 wk; includes the audit write-up, reusing repo tooling).

**Count:** 59 claims — DONE_REPO 32, PARTIAL_REPO 2, DONE_ELSEWHERE 13, UNTESTED 9, NOT_TESTABLE_AS_STATED 3.
**Remaining work if we did everything left:** 68–141 person-days summed row-by-row; materially less in practice
because rows bundle (one epithelium script covers T1917-018+019; one spiral fitter covers T1917-027+029+030+031+032;
one Ch. XVII GM protocol covers T1917-051+055+056+058). Realistic bundled total: **~120–240 person-days**, of which
the P0 set below is **~30–55 days**. DONE_ELSEWHERE rows need no repo work unless we want a reproduction.

## Headline verdicts so far (all claims dated 1917)

1. **The fish claim he is famous for is, literally read, refuted — and bounded, not buried.** *Scarus* → *Pomacanthus*
   by (conformal) coaxial circles (1917, p. 749, Figs. 375–376): Möbius explains ~1% — a conformal map cannot deepen a
   fish (depth/length 0.377 → 0.587); the generous oblique reading explains 32%, fails leave-one-landmark-out, and on
   real outlines is worse than doing nothing (IoU 0.618 vs 0.662). See T1917-046.
2. **The general transformation thesis survives only inside a horizon:** low-parameter warps reach the biological noise
   floor within genera and within ~45–50 Myr, and fail between families (0.3% reach the floor). T1917-039/050.
3. **Arthropod figure pairs (copepods, crabs, amphipods, 1917, pp. 743–746) fare better than the fish pair** on his own
   drawings (e.g. copepods 84.2% vs 20.4% for a box stretch). T1917-042/043/044.
4. **Most of the rest of the book has never been tested by us at all** — 21 UNTESTED rows, concentrated in exactly the
   chapters people cite most after Ch. XVII: logarithmic spirals/forams (XI–XII), horns (XIII), eggs (XV), bone
   trajectories (XVI), and the non-fish Ch. XVII skull series (crocodiles, horse, primates).

## P0 — do these first (cheap, central, or both)

| ID | Claim (1917) | Status | Effort | Why first |
|---|---|---|---|---|
| T1917-008 | Ch. III, pp. 50-155, Figs. 18-22 — Parts of an organism grow at different but definite relative rates, so form changes systematically with size; … | DONE_REPO | 4–8d (M) | RESOLVED 2026-10-28 (overnight loop, composer-2.5, PR 297): verdict supported. Repo supports differential-growth description in an… |
| T1917-014 | Ch. V, pp. 201-276, Figs. 61-67, 72-96 — The forms of single cells and cell-like organisms (Plateau's oil globules; Lagena and other forams/protists) a… | DONE_REPO | 5–10d (M) | RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 302): verdict mixed. Never fitted organism-by-organism to Thompson's fig… |
| T1917-018 | Ch. VII, pp. 293-345, Figs. 100-135 — Cells in contact, like soap films, meet three-at-a-time at ~120 degrees and in other respects obey Plateau's l… | DONE_REPO | 3–6d (M) | RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 304): verdict supported. Angle distribution in a real epithelium has nev… |
| T1917-019 | Ch. VII, pp. 293-345, Figs. 120-127 — Tissue cells tessellate approximately as hexagons / as a froth, the configuration that partitions area with le… | DONE_REPO | 3–6d (M) | RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 303): verdict supported. Same gap as T1917-018: polygon-class distributi… |
| T1917-027 | Ch. XI, pp. 493-586, Figs. 237-303 — Spiral shells (and other gnomonic-growth structures) are logarithmic (equiangular) spirals: the wharves enlarg… | DONE_REPO | 3–6d (M) | RESOLVED 2026-10-28 (overnight loop, composer-2.5, PR 298): verdict supported. Repo explores the Raup/log-spiral space but has not… |
| T1917-029 | Ch. XII, pp. 587-611, Figs. 308-317 — Foraminiferal shells are generated by chambers added in logarithmic-spiral succession with (near-)constant gro… | DONE_REPO | 4–8d (M) | RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 308): verdict mixed. Chamber-by-chamber ratio test is cheap and has, to … |
| T1917-039 | Ch. XVII, pp. 719-776, Figs. 353-408 — GENERAL THESIS: the form of a related species/genus can be obtained from another by a simple, continuous defor… | PARTIAL_REPO | 2–4d (S) | Verdict so far is scale-dependent, and that IS the repo's central result: global low-parameter warps capture most variance within … |
| T1917-041 | Ch. XVII, pp. ~732, Figs. 359-360 — Lanceolate, ovate and cordate leaves are transforms of one another under radial (polar) coordinates about an a… | DONE_REPO | 3–6d (M) | RESOLVED 2026-10-28 (overnight loop, composer-2.5, PR 299): verdict mixed. The law has been borrowed and tested on animals; Thomps… |
| T1917-047 | Ch. XVII (fish II, text), pp. 749, Figs. 375-376 (text) — The 'still more bizarre outlines of other fishes of the same family of Chaetodonts' correspond to small variat… | DONE_REPO | 1–2d (S) | The explained-variance half is tested (median 35%: weak-to-moderate, not 'very slight variations' producing bizarre outlines). The… |
| T1917-051 | Ch. XVII, pp. 752-753, Figs. 383 — The skulls of crocodiles (C. porosus, C. americanus, Notosuchus terrestris) are mutually transformable by coor… | DONE_REPO | 5–10d (M) | RESOLVED 2026-10-28 (overnight loop, composer-2.5, PR 296): verdict mixed. Never tested. The recommended Dryad dataset makes this … |
| T1917-056 | Ch. XVII, pp. 765-767, Figs. 401-402 — The skull of Eohippus (Hyracotherium) is transformed into that of Equus by coordinate deformation, with actual… | UNTESTED | 5–10d (M) | Untested, and one of the most iconic untested pairs. Horse skull GM data are public; intermediates make it falsifiable (do real me… |
| T1917-058 | Ch. XVII, pp. 770-773, Figs. 404-408 — The human skull, inscribed in Cartesian coordinates and deformed, gives the skulls of chimpanzee, baboon and d… | DONE_REPO | 5–10d (M) | RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 305): verdict supported. Never tested on primates. Primate skull GM data… |

## Full inventory (Pass 1 citation + Pass 2 assessment)

### Chapter I — Introductory (1917)

#### T1917-001 — Thompson (1917), Ch. I, pp. 1-15, Figs. -

**Claim (made 1917):** The form of an organism is a 'diagram of forces': form can and should be described physically and mathematically, in terms of the forces that produced it.

**Pass 2 — status:** `NOT_TESTABLE_AS_STATED` · **effort to settle:** 1–2 person-days (S) · **priority:** P2

**What remains / verdict:** Operationalise per-chapter (that is what rows T1917-003..058 do). As a programme it is vindicated wherever a physical model fits form with fewer parameters than a purely descriptive one.

**Method:** Score each operational descendant row; report the fraction supported. · **Data needed:** This registry · **Blockers:** None; do not score the slogan, score its descendants

#### T1917-002 — Thompson (1917), Ch. I + Epilogue, pp. 1-15; Epilogue, Figs. -

**Claim (made 1917):** Many forms are explained by physical forces directly and do not require (and are not well explained by) adaptation/heredity acting part-by-part; related forms differ by whole-form transformation rather than by independent change of parts.

**Pass 2 — status:** `PARTIAL_REPO` · **effort to settle:** 5–15 person-days (M) · **priority:** P1

**What remains / verdict:** Repo tests the morphospace half (global growth fields vs local modules) in fishes/ants, not the anti-adaptationist half, which needs an explicit adaptationist null per system.

**Method:** Per system: compare global-field model vs part-wise/adaptive model by AICc and out-of-sample error. · **Data needed:** Per-system (see descendant rows) · **Blockers:** Needs a fair, pre-specified adaptationist competitor model or the comparison is rhetorical

**Repo evidence:** `thompson/run_morphospace_vs_competition_analysis.py (G-vs-R decomposition, thompson_morphospace_vs_competition_summary.json)`; `thompson/run_three_dataset_validation.py (10,303 spp.)`

### Chapter II — On Magnitude (1917)

#### T1917-003 — Thompson (1917), Ch. II, pp. 16-49, Figs. 1-2

**Claim (made 1917):** Surface grows as L^2 and volume/weight as L^3, so surface:volume falls as 1/L; this alone bounds organism size and forces disproportionate change of parts with size.

**Pass 2 — status:** `DONE_ELSEWHERE` · **effort to settle:** 1–2 person-days (S) · **priority:** P2

**What remains / verdict:** Textbook scaling (supported with well-known exceptions via folding/fractal surfaces). A repo check is a teaching plot, not new science.

**Method:** Log-log surface and volume vs length across taxa; test slopes 2 and 3. · **Data needed:** Any size/shape compilation · **Blockers:** None

#### T1917-004 — Thompson (1917), Ch. II, pp. 16-49, Figs. -

**Claim (made 1917):** Strength of a support scales with cross-sectional area (L^2) while load scales with weight (L^3), so larger land animals must be relatively thicker-boned / differently proportioned (Galilean scaling).

**Pass 2 — status:** `DONE_ELSEWHERE` · **effort to settle:** 3–7 person-days (M) · **priority:** P2

**What remains / verdict:** Modern verdict is mixed: elastic-similarity and stress-similarity scalings fit some clades, geometric L^3/L^2 does not fit all. Worth one clade-level repo test only if bundled with T1917-040.

**Method:** Bone diameter/length and body mass across a clade; compare geometric, elastic and empirical exponents. · **Data needed:** Public limb-bone compilations (e.g. museum CT measurements) · **Blockers:** Phylogenetic correction required

#### T1917-005 — Thompson (1917), Ch. II, pp. 16-49, Figs. 2

**Claim (made 1917):** Different physics dominates at different magnitudes: surface tension and viscosity rule small organisms, gravity and inertia rule large ones; there is a least size of organism set by molecular/thermal scales.

**Pass 2 — status:** `DONE_ELSEWHERE` · **effort to settle:** 2–4 person-days (S) · **priority:** P2

**What remains / verdict:** Qualitatively settled (low-Reynolds-number biology). No repo score against Thompson's stated examples.

**Method:** Regime map (Re, Bond, Peclet numbers) for Thompson's named organisms. · **Data needed:** Sizes/densities for named taxa (literature) · **Blockers:** Choosing his exact examples from OCR text

**Repo evidence:** `datasets/ scout corpus: many *_vertex_packing / *_turing_wavelength organism x mechanism folders (scouted, not Thompson-scored)`

#### T1917-006 — Thompson (1917), Ch. II, pp. 16-49, Figs. 1

**Claim (made 1917):** Cell (e.g. nerve-cell) size is roughly independent of the size of the animal; larger animals have more cells, not proportionally larger cells.

**Pass 2 — status:** `DONE_ELSEWHERE` · **effort to settle:** 2–5 person-days (S) · **priority:** P2

**What remains / verdict:** Broadly supported with clade-specific exceptions (e.g. some neurons, eggs, polyploid cells).

**Method:** Cell size vs body size regression within a tissue type. · **Data needed:** Histology compilations · **Blockers:** Tissue must be held constant

### Chapter III — The Rate of Growth (1917)

#### T1917-007 — Thompson (1917), Ch. III, pp. 50-155, Figs. 3-40

**Claim (made 1917):** Growth in size/weight over time follows a definite, plottable curve: rapid (near-compound-interest) early growth with a steadily diminishing specific rate of growth.

**Pass 2 — status:** `DONE_ELSEWHERE` · **effort to settle:** 3–6 person-days (M) · **priority:** P1

**What remains / verdict:** Supported as a family (von Bertalanffy/logistic/Gompertz fits); no single universal curve. Thompson's own plotted data (man, mouse, silkworm, tadpole, bamboo) could be digitised and re-fit cheaply.

**Method:** Digitise Figs. 3-17 data, fit modern growth models, compare AICc. · **Data needed:** Thompson's cited sources (Quetelet, Bowditch, Ostwald) via his figures · **Blockers:** Figure digitisation error must be estimated

#### T1917-008 — Thompson (1917), Ch. III, pp. 50-155, Figs. 18-22

**Claim (made 1917):** Parts of an organism grow at different but definite relative rates, so form changes systematically with size; the changing proportions of organs (brain, heart, body; plaice weight-length) are growth-rate phenomena.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 4–8 person-days (M) · **priority:** P0

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, composer-2.5, PR 297): verdict supported. Repo supports differential-growth description in ants and several clades, but has not re-tested Thompson's own organ-ratio examples (Figs. 18-22) nor fitted the log-log relative-growth (allometric) law to his data. Formalised later by Huxley (1932) - credit and date must stay with Thompson (1917) for the claim as made here.

**Method:** log(part) vs log(body) slopes with phylogenetic control, on his systems and on repo clades. · **Data needed:** Organ-size series (literature) + existing portfolio datasets · **Blockers:** None major; data are public

**Repo evidence:** `thompson/run_antscan_caste_allometry_tests.py + thompson_ant_caste_allometry_summary.json (ant caste head allometry: 87.0% Thompsonian global vs 43.9% box stretch)`; `thompson/portfolio/ clade-GM outcomes (allometric axes in wings, skulls, shells)`; `datasets/thompson_1917/t1917-008/result.json`; `thompson/claims/t1917-008_the-rate-of-growth.py`

#### T1917-009 — Thompson (1917), Ch. III, pp. 50-155, Figs. 25-32

**Claim (made 1917):** Rate of growth varies lawfully with temperature and shows daily/seasonal/annual periodicity, recorded e.g. in tree growth (Douglass) and plant/root growth curves.

**Pass 2 — status:** `DONE_ELSEWHERE` · **effort to settle:** 2–4 person-days (S) · **priority:** P2

**What remains / verdict:** Settled in dendrochronology and growth physiology; not a repo gap worth new work unless used as a null for other claims.

**Method:** Re-fit his published temperature/periodicity series. · **Data needed:** Cited historical series · **Blockers:** None

#### T1917-010 — Thompson (1917), Ch. III, pp. 50-155, Figs. 33-40

**Claim (made 1917):** Regeneration restores form at a measurable rate related to the amount removed (Durbin/Ellis tail-regeneration curves), i.e. restoration of form is itself a lawful growth process.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 5–10 person-days (M) · **priority:** P2

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 311): verdict supported. No test on his (amphibian tail) system; modern regeneration morphometrics could test rate-vs-amount directly.

**Method:** Digitise Figs. 35-40; compare against a modern regeneration time-series dataset. · **Data needed:** Modern urodele/planarian regeneration morphometrics (public) · **Blockers:** Matching his endpoint definitions

**Repo evidence:** `archive/legacy_planarian_baselines/ (planarian baseline assets, unrelated system, no Thompson scoring)`; `datasets/thompson_1917/t1917-010/result.json`; `thompson/claims/t1917-010_the-rate-of-growth.py`

### Chapter IV — On the Internal Form and Structure of the Cell (1917)

#### T1917-011 — Thompson (1917), Ch. IV, pp. 156-200, Figs. 41-59

**Claim (made 1917):** The mitotic figure behaves as a system of forces between two poles: spindle and aster configurations, and chromosome movements, follow the lines of a bipolar field of force (with Peddie's magnetic-field diagrams as the physical analogue).

**Pass 2 — status:** `DONE_ELSEWHERE` · **effort to settle:** 10–20 person-days (L) · **priority:** P2

**What remains / verdict:** Superseded mechanistically (microtubule dynamics, motor proteins), though the field/polarity language survives in modern spindle mechanics. A repo 'test' would be history-of-mechanism, not a fit.

**Method:** Not recommended as a shape fit; if scored, compare spindle geometry to dipole-field vs microtubule models on imaging data. · **Data needed:** Spindle imaging (public cell atlases) · **Blockers:** Claim is mechanistic, not merely geometric; geometry alone cannot decide it

#### T1917-012 — Thompson (1917), Ch. IV, pp. 156-200, Figs. 58-59

**Claim (made 1917):** Early cleavage/segmentation patterns of eggs conform to the same bipolar (two-like-pole) field configuration.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 7–14 person-days (L) · **priority:** P2

**What remains / verdict:** RESOLVED 2026-10-30 (overnight loop, composer-2.5, PR 318): verdict supported. Untested as stated; modern cleavage is explained by spindle positioning/cortical mechanics, which could be compared geometrically.

**Method:** Cleavage-plane angles vs field-model prediction in a model embryo dataset. · **Data needed:** Public embryo imaging/landmarks · **Blockers:** Operationalising 'field configuration' fairly

**Repo evidence:** `embryo-blind/ and wave-vector-analysis/ are unrelated Ca2+ pipelines, not cleavage-geometry tests`; `datasets/thompson_1917/t1917-012/result.json`; `thompson/claims/t1917-012_on-the-internal-form-and-structure-of-the-cell.py`

### Chapter V — The Forms of Cells (1917)

#### T1917-013 — Thompson (1917), Ch. V, pp. 201-276, Figs. 60-68

**Claim (made 1917):** A free, isolated cell (like a liquid drop) tends to the spherical/minimal-area form under surface tension, departing from it only under identifiable constraints.

**Pass 2 — status:** `DONE_ELSEWHERE` · **effort to settle:** 3–6 person-days (M) · **priority:** P1

**What remains / verdict:** Supported as a null model; modern cell mechanics adds cortex/active stress. Repo has scouted material but no Thompson-scored test.

**Method:** Roundness/area-minimality of isolated cells vs active-mechanics null. · **Data needed:** Cell-shape imaging compilations · **Blockers:** Defining the fair active-cell null

**Repo evidence:** `datasets/ scout corpus: *_vertex_packing and surface-tension organism folders (scouted, not scored to Thompson)`

#### T1917-014 — Thompson (1917), Ch. V, pp. 201-276, Figs. 61-67, 72-96

**Claim (made 1917):** The forms of single cells and cell-like organisms (Plateau's oil globules; Lagena and other forams/protists) are surfaces of revolution from the minimal-surface catalogue: sphere, cylinder, unduloid, nodoid, catenoid.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 5–10 person-days (M) · **priority:** P0

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 302): verdict mixed. Never fitted organism-by-organism to Thompson's figured genera with an area-minimality score. This is one of the cheapest untouched wins: outlines are in his figures and in modern images.

**Method:** Fit each figured outline to the Delaunay surface family; report residual vs free-form spline. · **Data needed:** Thompson's figures (IA scan) + modern protist/foram images (Commons/literature) · **Blockers:** Species identification from 1917 drawings

**Repo evidence:** `models/ Plateau/minimal-surface simulations elsewhere in repo (see models/ catalogue); datasets/*_vertex_packing scout folders`; `datasets/thompson_1917/t1917-014/result.json`; `thompson/claims/t1917-014_the-forms-of-cells.py`

#### T1917-015 — Thompson (1917), Ch. V, pp. 201-276, Figs. 68-71

**Claim (made 1917):** A liquid cylinder breaks up into a row of spheres (Plateau-Rayleigh instability), and the same segmentation is seen in protoplasmic threads/cells.

**Pass 2 — status:** `DONE_ELSEWHERE` · **effort to settle:** 2–4 person-days (S) · **priority:** P2

**What remains / verdict:** Instability is settled physics (wavelength ~ circumference criterion); biological instances are case-by-case.

**Method:** Compare observed bead spacing/diameter to Rayleigh criterion. · **Data needed:** Images of his examples (Spirogyra/protoplasmic strands) · **Blockers:** Finding measurable instances

#### T1917-016 — Thompson (1917), Ch. V, pp. 201-276, Figs. 94-96

**Claim (made 1917):** Cell form changes lawfully with the density/osmotic strength of the medium (Koltzoff's decapod sperm cells in saline solutions), as a surface-tension/osmotic system should.

**Pass 2 — status:** `UNTESTED` · **effort to settle:** 4–8 person-days (M) · **priority:** P2

**What remains / verdict:** No repo test; modern osmotic cell-mechanics could replicate the exact experiment in silico/on public data.

**Method:** Cell volume/shape vs osmolarity curve vs osmotic model. · **Data needed:** Public osmolarity cell-shape data or a small simulation · **Blockers:** His endpoint is drawing-based

### Chapter VI — A Note on Adsorption (1917)

#### T1917-017 — Thompson (1917), Ch. VI, pp. 277-292, Figs. 97-98

**Claim (made 1917):** Adsorption concentrates dissolved substances at surfaces/interfaces, and this physical concentration helps explain localisation and pattern within the cell/organism.

**Pass 2 — status:** `NOT_TESTABLE_AS_STATED` · **effort to settle:** 1–1 person-days (S) · **priority:** P2

**What remains / verdict:** Too general to falsify as written; specific descendants (surface catalysis, membrane localisation) are settled elsewhere.

**Method:** Do not score as one row; split into specific modern claims if ever pursued. · **Data needed:** - · **Blockers:** Vagueness is the blocker

**Repo evidence:** `models/ reaction-diffusion and bioelectric models in repo are mechanistic cousins, not adsorption tests`

### Chapter VII — The Forms of Tissues, or Cell-Aggregates (1917)

#### T1917-018 — Thompson (1917), Ch. VII, pp. 293-345, Figs. 100-135

**Claim (made 1917):** Cells in contact, like soap films, meet three-at-a-time at ~120 degrees and in other respects obey Plateau's laws of film configuration.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 3–6 person-days (M) · **priority:** P0

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 304): verdict supported. Angle distribution in a real epithelium has never been scored here against the 120-degree prediction with a stated tolerance. Modern literature broadly supports Plateau statistics with active-mechanics deviations.

**Method:** Junction-angle histograms from segmented epithelia vs 120-degree/Plateau null. · **Data needed:** Public segmented epithelium (e.g. Drosophila wing disc) · **Blockers:** None major; data are public

**Repo evidence:** `datasets/*_vertex_packing scout folders (organism x vertex-packing, unscored to Thompson)`; `morphoeval/ vertex-model simulations in the wider corpus`; `datasets/thompson_1917/t1917-018/result.json`; `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py`

#### T1917-019 — Thompson (1917), Ch. VII, pp. 293-345, Figs. 120-127

**Claim (made 1917):** Tissue cells tessellate approximately as hexagons / as a froth, the configuration that partitions area with least perimeter.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 3–6 person-days (M) · **priority:** P0

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 303): verdict supported. Same gap as T1917-018: polygon-class distributions (pentagon/hexagon/heptagon fractions) unscored. Known modern result: distributions are froth-like but not perfectly hexagonal.

**Method:** Side-number distribution + Lewis/Desch laws on segmented tissue. · **Data needed:** As T1917-018 · **Blockers:** Bundle with T1917-018 (same data, one script)

**Repo evidence:** `datasets/*_vertex_packing scout folders`; `datasets/thompson_1917/t1917-019/result.json`; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py`

#### T1917-020 — Thompson (1917), Ch. VII, pp. 293-345, Figs. 136-175 (with Ch. VIII)

**Claim (made 1917):** Sachs's rules describe cell division in plant tissues: daughter cells tend to equal volume, new walls meet old walls at right angles and follow minimal-area configurations.

**Pass 2 — status:** `DONE_ELSEWHERE` · **effort to settle:** 5–10 person-days (M) · **priority:** P2

**What remains / verdict:** Tested extensively in modern plant morphogenesis (Errera's rule etc.); supported in refined form. No repo reproduction.

**Method:** Division-plane statistics in a plant tissue dataset vs Sachs/Errera predictions. · **Data needed:** Public plant meristem imaging · **Blockers:** None, but low novelty

### Chapter VIII — The Forms of Tissues (continued) (1917)

#### T1917-021 — Thompson (1917), Ch. VIII, pp. 346-410, Figs. 136-193

**Claim (made 1917):** The bee's honeycomb - hexagonal prisms closed by three rhombi meeting at the Maraldi angle (~109 deg 28 min) - is the configuration using least wax for the capacity (the isoperimetric claim).

**Pass 2 — status:** `DONE_ELSEWHERE` · **effort to settle:** 1–3 person-days (S) · **priority:** P1

**What remains / verdict:** Split verdict in modern literature, and the split matters: the hexagonal tiling IS optimal (Hales 2001), but Fejes Toth showed a slightly better base than the bees' rhombic trihedron, so 'least wax' is false in detail for the end-caps. Any repo treatment must state both halves.

**Method:** Analytic/numerical wax-volume comparison; no new data needed. · **Data needed:** Geometry only (+ comb measurements for what bees actually build) · **Blockers:** None; mainly a careful write-up

#### T1917-022 — Thompson (1917), Ch. VIII, pp. 346-410, Figs. 176-193

**Claim (made 1917):** Isoperimetric/minimal-area principles also govern other hollow and aggregate structures he treats in this chapter (e.g. radiolarian-like and echinoderm larval skeletal arrangements, vessel and tissue partitions).

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 5–15 person-days (M) · **priority:** P2

**What remains / verdict:** RESOLVED 2026-10-30 (overnight loop, composer-2.5, PR 319): verdict mixed. A catch-all row: split per structure before testing. Keep as a parent row only.

**Method:** Split, then per-structure minimality score as in T1917-014/018. · **Data needed:** Per structure · **Blockers:** Must be split; do not score the catch-all

**Repo evidence:** `datasets/*_biomineral_accretion and *_vertex_packing scout folders (unscored)`; `datasets/thompson_1917/t1917-022/result.json`; `thompson/claims/t1917-022_the-forms-of-tissues-continued.py`

### Chapter IX — On Concretions, Spicules, and Spicular Skeletons (1917)

#### T1917-023 — Thompson (1917), Ch. IX, pp. 411-487, Figs. 194-235

**Claim (made 1917):** Concretions and calcospherites grow by radial deposition about a centre (Harting's artificial concretions in egg albumen mimic natural ones), giving concentric/spherical forms by physical deposition alone.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 4–8 person-days (M) · **priority:** P1

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 301): verdict supported. Repo has scouted/simulated accretion widely but never scored radial-deposition geometry against Thompson's/Harting's figured spherites.

**Method:** Radial symmetry + concentric banding metrics on spherite images/simulations. · **Data needed:** Harting figures (IA) + modern spherite/concretion images · **Blockers:** Image sourcing

**Repo evidence:** `datasets/*_biomineral_accretion scout folders (organism x accretion mechanism, unscored)`; `models/ biomineral accretion simulations in repo catalogue`; `datasets/thompson_1917/t1917-023/result.json`; `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py`

#### T1917-024 — Thompson (1917), Ch. IX, pp. 411-487, Figs. 194-235

**Claim (made 1917):** Sponge and alcyonarian spicule forms (needles, tri-/tetraradiates and their aggregates) are the forms physical deposition/crystallisation takes within the cell/field that secretes them.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 7–14 person-days (L) · **priority:** P1

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 312): verdict mixed. Angles between rays (e.g. tri-radiate ~120 deg, tetraxial tetrahedral angles) are directly measurable and untested here.

**Method:** Ray-angle and symmetry statistics on spicule image sets vs deposition-model predictions. · **Data needed:** Spicule SEM/image compilations (literature/GBIF images) · **Blockers:** Homology/species sampling frame

**Repo evidence:** `datasets/ sponge/spicule-adjacent scout folders (unscored)`; `datasets/thompson_1917/t1917-024/result.json`; `thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py`

#### T1917-025 — Thompson (1917), Ch. IX-X, pp. 411-492, Figs. 221-235, 236-259

**Claim (made 1917):** Radiolarian and similar siliceous skeletons approximate geodesic/minimal frameworks on a sphere (latticed shells as systems of shortest lines and near-equal partitions).

**Pass 2 — status:** `UNTESTED` · **effort to settle:** 7–14 person-days (L) · **priority:** P1

**What remains / verdict:** Haeckel's figured species are ideal for a geodesic-framework score; never done in repo.

**Method:** Extract lattice from Haeckel/modern radiolarian images; test geodesic and valence statistics. · **Data needed:** Haeckel plates (public domain) + modern SEM · **Blockers:** 3D lattice from 2D plates

**Repo evidence:** `datasets/acanthometra_* and radiolarian-adjacent scout folders (unscored)`

### Chapter X — A Parenthetic Note on Geodetics (1917)

#### T1917-026 — Thompson (1917), Ch. X, pp. 488-492, Figs. 236-259

**Claim (made 1917):** Geodesic (shortest-line) trajectories on a curved surface explain the courses of spicules, shell sculpture and other skeletal lines laid down on that surface.

**Pass 2 — status:** `UNTESTED` · **effort to settle:** 7–14 person-days (L) · **priority:** P2

**What remains / verdict:** General principle, testable per instance (bundle with T1917-024/025/040).

**Method:** Fit observed surface curves to surface geodesics; residual vs non-geodesic null. · **Data needed:** 3D surface + curve traces for one chosen system · **Blockers:** Needs a 3D system; pick one, not all

### Chapter XI — The Logarithmic Spiral (1917)

#### T1917-027 — Thompson (1917), Ch. XI, pp. 493-586, Figs. 237-303

**Claim (made 1917):** Spiral shells (and other gnomonic-growth structures) are logarithmic (equiangular) spirals: the wharves enlarge by a constant ratio, so the form is unchanged by growth (r proportional to e^{k theta}).

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 3–6 person-days (M) · **priority:** P0

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, composer-2.5, PR 298): verdict supported. Repo explores the Raup/log-spiral space but has not fitted Thompson's own figured shells and reported log-vs-Archimedean discrimination with error bars. Modern literature: broadly supported, with real deviations.

**Method:** Fit log vs Archimedean vs power spirals to shell centrelines; AICc + residuals. · **Data needed:** Shell images/outlines (Commons, museum) for his genera (Nautilus, gastropods) · **Blockers:** None major

**Repo evidence:** `models/1966-raup.py + models/create_raup_animations.py (Raup parameter exploration, the direct descendant formalism)`; `conus_* and nacre/spiral model outputs in repo root/models`; `datasets/thompson_1917/t1917-027/result.json`; `thompson/claims/t1917-027_the-logarithmic-spiral.py`

#### T1917-028 — Thompson (1917), Ch. XI, pp. 493-586, Figs. 304-307

**Claim (made 1917):** In Nautilus the shell spiral and the septa bear the stated geometric relations (septa meeting the wall, spiral as evolute/involute relations he diagrams).

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 3–5 person-days (S) · **priority:** P1

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, composer-2.5, PR 300): verdict mixed. Directly measurable on a Nautilus section/radiograph; never scored.

**Method:** Landmark a Nautilus section; test his septum/spiral relations numerically. · **Data needed:** Nautilus shell section image (public) · **Blockers:** None

**Repo evidence:** `.thompson_cache/ radiograph-style Nautilus material referenced in Ch. X figures, not scored`; `datasets/thompson_1917/t1917-028/result.json`; `thompson/claims/t1917-028_the-logarithmic-spiral.py`

### Chapter XII — The Spiral Shells of the Foraminifera (1917)

#### T1917-029 — Thompson (1917), Ch. XII, pp. 587-611, Figs. 308-317

**Claim (made 1917):** Foraminiferal shells are generated by chambers added in logarithmic-spiral succession with (near-)constant growth ratios, and the genera form a continuous geometric series transformable into one another.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 4–8 person-days (M) · **priority:** P0

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 308): verdict mixed. Chamber-by-chamber ratio test is cheap and has, to our knowledge, no repo or clean modern score on his Brady-figured genera.

**Method:** Measure successive chamber sizes/angles in Brady plates/modern forams; test constancy of ratio and series continuity. · **Data needed:** Brady Challenger plates (public domain) + modern foram images · **Blockers:** Species ID from plates

**Repo evidence:** `datasets/ foram-adjacent scout folders (unscored)`; `datasets/thompson_1917/t1917-029/result.json`; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py`

#### T1917-030 — Thompson (1917), Ch. XII (with V), pp. 587-611; cf. 201-276, Figs. 84-89, 308-317

**Claim (made 1917):** Single-chambered foraminiferal tests (Lagena etc.) take the surface-tension forms of Ch. V (flask/unduloid-like surfaces of revolution), chamber form preceding spiral coiling.

**Pass 2 — status:** `UNTESTED` · **effort to settle:** 2–4 person-days (S) · **priority:** P1

**What remains / verdict:** Bundle with T1917-014 (same fitter, same surface catalogue).

**Method:** Delaunay-surface fit to Lagena outlines. · **Data needed:** Brady plates / modern images · **Blockers:** Bundle with T1917-014

### Chapter XIII — The Shapes of Horns, and of Teeth or Tusks, with a Note on Torsion (1917)

#### T1917-031 — Thompson (1917), Ch. XIII, pp. 612-634, Figs. 318-324

**Claim (made 1917):** Horns (ram, goat, Ovis ammon) are logarithmic spirals wound on a cone (a conical helicoid generated by gnomonic growth), differing between species by the spiral constants and torsion.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 7–14 person-days (L) · **priority:** P1

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 310): verdict mixed. No repo fit to horn centrelines in 3D; modern CT/museum horn data exist. This is the horn analogue of the shell test and should share code with T1917-027.

**Method:** Extract 3D horn centreline; fit conical log-helicoid vs alternatives. · **Data needed:** Museum horn photos/CT (MorphoSource, public specimens) · **Blockers:** 3D data access per species

**Repo evidence:** `datasets/thompson_1917/t1917-031/result.json`; `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py`

#### T1917-032 — Thompson (1917), Ch. XIII, pp. 612-634, Figs. 318-324

**Claim (made 1917):** Teeth and tusks grow by the same spiral/torsion geometry (with St Venant torsion curves describing the twisted prismatic forms).

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 5–10 person-days (M) · **priority:** P2

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 309): verdict mixed. Untested; bundle with T1917-031.

**Method:** As T1917-031 on tusk/tooth centrelines and cross-sections. · **Data needed:** Tusk/tooth images or CT · **Blockers:** Data sourcing

**Repo evidence:** `datasets/thompson_1917/t1917-032/result.json`; `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py`

### Chapter XIV — On Leaf-Arrangement, or Phyllotaxis (1917)

#### T1917-033 — Thompson (1917), Ch. XIV, pp. 635-651, Figs. 325-327

**Claim (made 1917):** Leaf/floret divergence approximates the golden (Fibonacci) angle (~137.5 deg), with rational approximations 1/2, 1/3, 2/5, 3/8... and Fibonacci numbers of parastichies.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 3–6 person-days (M) · **priority:** P1

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, composer-2.5, PR 314): verdict supported. Repo simulates the mechanism but has not scored divergence-angle distributions in real plants against Thompson's fractions. Modern literature: strongly supported with documented exceptions/transitions.

**Method:** Divergence-angle measurement on capitula/stems; compare golden-angle vs alternative packing models. · **Data needed:** Sunflower/ capitulum images (public) - measurement only · **Blockers:** None major

**Repo evidence:** `models/1979-vogel-phyllotaxis.py`; `models/1992-douady-magnetic-drop-phyllotaxis.py (Douady-Couder physical mechanism, a direct modern descendant)`; `datasets/thompson_1917/t1917-033/result.json`; `thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py`

### Chapter XV — On the Shapes of Eggs, and of Certain Other Hollow Structures (1917)

#### T1917-034 — Thompson (1917), Ch. XV, pp. 652-669, Figs. 328-330

**Claim (made 1917):** Birds' (and other hard-shelled) eggs are solids of revolution whose meridian curves form a definite, species-varying family of ovoids, comparable geometrically across species.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 3–6 person-days (M) · **priority:** P1

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 306): verdict supported. Modern egg-shape literature (e.g. universal egg formula work, 2021) postdates and partly supersedes him; a repo test on museum egg outlines is straightforward and has not been done.

**Method:** Fit egg outlines to Thompson-style ovoid parameters vs modern universal formula; compare residuals across species. · **Data needed:** Egg outline images (museum/Commons) · **Blockers:** None major

**Repo evidence:** `datasets/thompson_1917/t1917-034/result.json`; `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py`

#### T1917-035 — Thompson (1917), Ch. XV, pp. 652-669, Figs. 328

**Claim (made 1917):** Certain hollow structures (e.g. sea-urchin tests diagrammed in Fig. 328, skulls and other shells) take forms explicable as membranes/shells under pressure and tension.

**Pass 2 — status:** `UNTESTED` · **effort to settle:** 5–10 person-days (M) · **priority:** P2

**What remains / verdict:** Too broad as one row; the urchin-test instance is measurable (membrane under uniform pressure predicts a specific meridian).

**Method:** Pick the urchin instance; fit meridian to pressure-membrane vs descriptive curves. · **Data needed:** Urchin test profiles · **Blockers:** Split before testing

### Chapter XVI — On Form and Mechanical Efficiency (1917)

#### T1917-036 — Thompson (1917), Ch. XVI, pp. 670-718, Figs. 331-352

**Claim (made 1917):** In a bent beam there is a neutral zone/axis free of tension and compression, and animal/plant structures that bend (bones, stems, trunks) are disposed accordingly (material where the stress is).

**Pass 2 — status:** `DONE_ELSEWHERE` · **effort to settle:** 1–2 person-days (S) · **priority:** P2

**What remains / verdict:** Settled engineering; biological allocation is clade-specific. Low novelty for repo.

**Method:** Only as part of T1917-037/038 bundles. · **Data needed:** - · **Blockers:** -

#### T1917-037 — Thompson (1917), Ch. XVI, pp. 670-718, Figs. 335

**Claim (made 1917):** Trabecular bone is arranged along the principal stress trajectories of the bone, as in Culmann's crane-head diagram applied to the human femur (the trajectorial theory).

**Pass 2 — status:** `UNTESTED` · **effort to settle:** 14–28 person-days (L) · **priority:** P1

**What remains / verdict:** A classic claim with a mixed modern verdict (alignment is real but partial and load-history dependent). A quantitative alignment score on public femur CT would be a genuine repo contribution.

**Method:** Trabecular orientation (fabric tensor) vs finite-element principal-stress directions in femur CT; alignment distribution vs random null. · **Data needed:** Public femur micro-CT / CT (MorphoSource etc.) · **Blockers:** FE model choices must be pre-specified; licensing of CT

**Repo evidence:** `thompson/generate_aristotle_1913_illustrations.py reproduces the Fig. 335 comparison as an illustration only (no measurement)`

#### T1917-038 — Thompson (1917), Ch. XVI, pp. 670-718, Figs. 331-334, 336-352

**Claim (made 1917):** Hollow cylindrical construction (long bones, stems, quills) gives near-maximal strength/stiffness for weight, and branching systems (blood vessels, Fig. 329-330 after Hess; trees) branch at angles/ratios that economise work/material.

**Pass 2 — status:** `DONE_ELSEWHERE` · **effort to settle:** 4–8 person-days (M) · **priority:** P2

**What remains / verdict:** Vessel branching is the Murray's-law literature (1926, after Thompson); broadly supported with exceptions. Bone hollowness ratios are measurable cheaply if bundled.

**Method:** Branch angle/diameter ratios vs Murray optimum on a public vascular/tree dataset. · **Data needed:** Public vascular or tree-branch measurements · **Blockers:** Attribution care: Murray (1926) is not Thompson (1917)

### Chapter XVII — On the Theory of Transformations, or the Comparison of Related Forms (1917)

#### T1917-039 — Thompson (1917), Ch. XVII, pp. 719-776, Figs. 353-408

**Claim (made 1917):** GENERAL THESIS: the form of a related species/genus can be obtained from another by a simple, continuous deformation of the whole coordinate system (a transformation of the whole form), rather than by unrelated changes of separate parts.

**Pass 2 — status:** `PARTIAL_REPO` · **effort to settle:** 2–4 person-days (S) · **priority:** P0

**What remains / verdict:** Verdict so far is scale-dependent, and that IS the repo's central result: global low-parameter warps capture most variance within genera / within ~45-50 Myr, fail to reach noise floor between families, and lose to local modular change in fins/jaws. The thesis as a universal claim is therefore REFUTED in its strong form and SUPPORTED in a bounded (taxonomic/temporal horizon) form. Remaining: the non-fish, non-arthropod figure pairs (T1917-053..061).

**Method:** Per-pair low-parameter warp vs similarity/affine/box nulls, LOO-CV, noise-floor criterion (repo standard). · **Data needed:** Per pair · **Blockers:** None for the synthesis row (roll-up of descendants)

**Repo evidence:** `All Ch. XVII descendant rows T1917-040..061`; `thompson_1917_all_arthropods_summary.json, thompson_four_families_summary.json, thompson_silhouette_area_summary.json, thompson_box_vs_warping_audit.json`

#### T1917-040 — Thompson (1917), Ch. XVII, pp. ~724-735, Figs. 353-358

**Claim (made 1917):** Human (and animal) heads/faces and proportions, in the Durer/Camper tradition, are mutually transformable by coordinate deformation; differences of facial angle/proportion are whole-grid deformations.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 7–14 person-days (L) · **priority:** P2

**What remains / verdict:** RESOLVED 2026-10-30 (overnight loop, composer-2.5, PR 315): verdict supported. His Durer-derived human proportion grids have never been landmark-tested here (the ape/human skull instance is T1917-060).

**Method:** Landmark Durer/Thompson grids and modern head profiles; warp-family fit as per repo standard. · **Data needed:** Profile landmarks (public anthropometric data) · **Blockers:** Historical grid digitisation

**Repo evidence:** `datasets/thompson_1917/t1917-040/result.json`; `thompson/claims/t1917-040_on-the-theory-of-transformations.py`

#### T1917-041 — Thompson (1917), Ch. XVII, pp. ~732, Figs. 359-360

**Claim (made 1917):** Lanceolate, ovate and cordate leaves are transforms of one another under radial (polar) coordinates about an arrest/base point; the Begonia leaf is an instance of the same radial system.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 3–6 person-days (M) · **priority:** P0

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, composer-2.5, PR 299): verdict mixed. The law has been borrowed and tested on animals; Thompson's actual leaves have never been fitted. Cheap, direct, undone.

**Method:** Fit leaf outlines (lanceolate/ovate/cordate series) with the 8p radial-cordate family vs box/affine nulls. · **Data needed:** Leaf outline images (public herbaria/Commons) · **Blockers:** Choosing a defensible leaf series

**Repo evidence:** `thompson/run_antscan_caste_allometry_tests.py docstring formalises 'Thompson (1917, Fig. 359, p. 732) Radial-Tangential Cordate Law' and tests it on ant heads (not leaves)`; `thompson/portfolio/results/antillean_flowers.json (corolla GM extension)`; `datasets/thompson_1917/t1917-041/result.json`; `thompson/claims/t1917-041_on-the-theory-of-transformations.py`

#### T1917-042 — Thompson (1917), Ch. XVII, pp. 743, Figs. 367-368

**Claim (made 1917):** The copepods Oithona nana and Sapphirina are related by a (logarithmic/coordinate) transformation of the whole form.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 4–8 person-days (M) · **priority:** P1

**What remains / verdict:** Supported on Thompson's drawings (77 grid landmarks). Remains: real-specimen outlines, as was done for fish.

**Method:** As fish: specimen photos + 600-pt outline fit. · **Data needed:** Copepod specimen images · **Blockers:** Specimen image sourcing

**Repo evidence:** `thompson/run_all_1917_arthropods_audit.py`; `thompson_1917_all_arthropods_summary.json: copepods_figs_367_368 - box 4p R2 20.41%, Thompson 8p log-quad R2 84.21%, gain +63.8 pts, min Jacobian det 0.177`

#### T1917-043 — Thompson (1917), Ch. XVII, pp. 744, Figs. 369

**Claim (made 1917):** The carapaces of crabs (Geryon, Corystes, Scyramathia/Rochinia, Paralomis/Lithodes, Lupa/Portunus, Chorinus etc.) are mutually transformable by rectilinear-to-curvilinear coordinate transformation.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 5–10 person-days (M) · **priority:** P1

**What remains / verdict:** Supported on drawings and partly on real 3D-scan landmarks; a full 600-pt real-specimen outline test remains (the fish treatment).

**Method:** Real carapace outlines + direct outline fitting. · **Data needed:** Crab carapace images/scans (partly in .thompson_cache/crabs) · **Blockers:** Scan access for all genera

**Repo evidence:** `thompson/run_crab_carapace_paradox_audit.py`; `thompson_crab_carapace_audit_summary.json: on his Fig. 369 drawings, 8p Thompson R2 78.7-99.4% vs 4p box 19.4-97.1%; vs Scholtz et al. (2020) 3D-scan landmarks, Thompson 14-LM lateral-margin RMS improves accuracy 1.7-9.2x`

#### T1917-044 — Thompson (1917), Ch. XVII, pp. 746, Figs. 370

**Claim (made 1917):** The amphipods Harpinia, Stegocephalus and Hyperia galba form a transformation series under coordinate deformation.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 4–8 person-days (M) · **priority:** P2

**What remains / verdict:** Tested on the 1917 plate within the all-arthropods audit; real-specimen outline test remains.

**Method:** Specimen outlines as per fish/copepod rows. · **Data needed:** Amphipod specimen images · **Blockers:** Image sourcing

**Repo evidence:** `thompson/run_all_1917_arthropods_audit.py (Panel E, Fig. 370 plate)`; `thompson_1917_all_arthropods_summary.json`

#### T1917-045 — Thompson (1917), Ch. XVII (fish I), pp. 748, Figs. 373-374

**Claim (made 1917):** Argyropelecus olfersii is transformed into Sternoptyx diaphana by an oblique (shear) coordinate system whose axes are inclined at 70 degrees.

> an oblique coordinate system whose axes are inclined at an angle of 70 deg (paraphrase pinned in thompson/1917-thompson-four-families-atlas.py docstring; verify verbatim in Pass-1 extension)

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 2–4 person-days (S) · **priority:** P1

**What remains / verdict:** Supported as a family at intrageneric/intrafamily scale; the exact pair is landmarked from archival plates. 600-pt exact-pair outline fit is the remaining upgrade.

**Method:** Direct outline fitting on the exact pair (repo silhouette pipeline). · **Data needed:** In repo (.thompson_cache archival plates) · **Blockers:** None

**Repo evidence:** `thompson/1917-thompson-four-families-atlas.py (Family I, 5p oblique shear; all 4,505 taxonomic + 4,334 dated pairs; exact-1917-species landmarking in datasets/thompson/thompson_1917_exact_species_landmarks.json)`; `thompson_four_families_summary.json`

#### T1917-046 — Thompson (1917), Ch. XVII (fish II), pp. 749, Figs. 375-376

**Claim (made 1917):** Deforming the rectilinear coordinates of a typical Scarus into (approximately) coaxial circles gives 'a very good outline' of Pomacanthus.

> Fig. 375 is an outline diagram of a typical Scaroid fish. Let us deform its rectilinear co-ordinates into a system of (approximately) coaxial circles, as in Fig. 376, and then filling into the new system, space by space and point by point, our former diagram of Scarus, we obtain a very good outline of an allied fish, belonging to a neighbouring family, of the genus Pomacanthus. (Thompson 1917, p. 749; quoted verbatim in thompson/1917-thompson-scarus-pomacanthus.py L5-13)

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 5–10 person-days (M) · **priority:** P1

**What remains / verdict:** REFUTED in its literal (conformal/Mobius) reading: Mobius explains 0.8-1% (a conformal map cannot change depth/length 0.377->0.587); the generous oblique-coaxial reading explains 32% at genus-mean landmarks (31st percentile of all genus pairs), LOO-CV fails (0.0371 > 0.0312), and on real outlines every deformation is WORSE than no deformation (IoU 0.618 coaxial vs 0.662 similarity). Remediation already listed in the audit: digitise his drawn grid, fit maps to outlines directly, more individuals.

**Method:** Audit remediation items 1-3 (grid digitisation, outline-fitted maps, multi-individual). · **Data needed:** In repo + more Commons/Bishop photos · **Blockers:** Photo posture confound (declared in audit)

**Repo evidence:** `thompson/1917-thompson-scarus-pomacanthus.py (MorphoEval 2026-10-17 audit, 76.6/100)`; `datasets/thompson/README.md results tables`; `morphoeval/kg/2026-10-17-thompson-audit/REPORT.md`

#### T1917-047 — Thompson (1917), Ch. XVII (fish II, text), pp. 749, Figs. 375-376 (text)

**Claim (made 1917):** The 'still more bizarre outlines of other fishes of the same family of Chaetodonts' correspond to small variations in the values of the constants of the coaxial curves.

> the still more bizarre outlines of other fishes of the same family of Chaetodonts will be found to correspond to very slight modifications of similar co-ordinates; in other words, to small variations in the values of the constants of the coaxial curves. (Thompson 1917, p. 749; quoted in thompson/1917-thompson-scarus-pomacanthus.py L10-13)

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 1–2 person-days (S) · **priority:** P0

**What remains / verdict:** The explained-variance half is tested (median 35%: weak-to-moderate, not 'very slight variations' producing bizarre outlines). The literal constants half - are the fitted constants actually SMALL variations? - is still unreported; it is a one-figure job on existing fits.

**Method:** Report fitted c (and affine part) for all 54 Chaetodon fits; test dispersion vs shape distance. · **Data needed:** Existing fits (re-run with constants saved) · **Blockers:** None. Cheapest open Thompson item in the repo.

**Repo evidence:** `thompson/1917-thompson-scarus-pomacanthus.py item 4: Scarus -> 54 Chaetodon species, oblique-coaxial explains median 35% (range 12-61%)`; `morphoeval/kg/2026-10-17-thompson-audit/REPORT.md (remediation: report the fitted constants per Chaetodon species)`

#### T1917-048 — Thompson (1917), Ch. XVII (fish III), pp. 750, Figs. 377-380

**Claim (made 1917):** Polyprion, Pseudopriacanthus (Pristigenys) altus, Scorpaena and Antigonia capros are related by a fan/divergent-convergent coordinate transformation.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 2–4 person-days (S) · **priority:** P1

**What remains / verdict:** Tested at landmark level across the atlas and on exact 1917 species; 600-pt exact-pair outlines remain.

**Method:** Direct outline fitting (repo silhouette pipeline). · **Data needed:** In repo · **Blockers:** None

**Repo evidence:** `thompson/1917-thompson-four-families-atlas.py (Family III, 8p bilinear fan; exact species landmarked: Polyprion americanus -> Pristigenys alta -> Scorpaena scrofa -> Antigonia capros)`; `thompson_four_families_summary.json`

#### T1917-049 — Thompson (1917), Ch. XVII (fish IV), pp. 751, Figs. 381-382

**Claim (made 1917):** Diodon is transformed into Orthagoriscus (Mola mola) by deforming verticals into concentric circles and horizontals into approximately hyperbolic curves.

> verticals are deformed into a system of concentric circles, and the horizontals into a system of curves approximately hyperbolic (paraphrase pinned in thompson/1917-thompson-four-families-atlas.py docstring; verify verbatim in Pass-1 extension)

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 2–4 person-days (S) · **priority:** P1

**What remains / verdict:** Landmark-level test done; Mola is absent from the 791-spp table so the exact pair rests on archival landmarking - outline-level test is the upgrade.

**Method:** Direct outline fitting on archival/NOAA specimens. · **Data needed:** In repo (.thompson_cache Mola plates) · **Blockers:** None

**Repo evidence:** `thompson/1917-thompson-four-families-atlas.py (Family IV, 8p hyperbolic/radial expansion; Diodon hystrix -> Mola mola exact-species landmarking)`; `thompson_four_families_summary.json`

#### T1917-050 — Thompson (1917), Ch. XVII (fish, generalisation), pp. 748-751, Figs. 373-382

**Claim (made 1917):** IMPLIED GENERALISATION tested by this repo (not a sentence of Thompson's): the four fish families above describe related-fish form difference generally, across taxa and evolutionary time.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 0–0 person-days (S) · **priority:** P2

**What remains / verdict:** This is the repo's most complete Thompson result and should be cited as an extension, never as 'Thompson claimed this in 1917'. Remaining: none for fish at landmark level; outline-level for exact pairs (T1917-045/048/049).

**Method:** Complete for fish; replicate protocol per new clade. · **Data needed:** - · **Blockers:** Citation discipline (1917 claim vs 2026 extension)

**Repo evidence:** `thompson/1917-thompson-four-families-atlas.py: 4,505 taxonomic pairs, 4,334 dated (Rabosky 2018) pairs; intrageneric median 48.0% explained / 54.2% reach noise floor; interfamily median 51.3% but only 0.3% reach noise floor; ~45-50 Myr horizon`; `thompson/1917-thompson-silhouette-area-paradox.py: 600-pt direct fitting cuts non-overlap 56.2% -> 28.5%; 9/17 head-landmark seesaw`; `thompson/run_box_vs_warping_audit.py: warp beats 4p box on homologous fin landmarks (+11.0 to +48.0%), not on unlabeled shadows`; `thompson_blind_test_results.json + docs/THOMPSON_BLIND_TEST_RESULTS.md (pre-registered FishShapes v1 blind test)`

#### T1917-051 — Thompson (1917), Ch. XVII, pp. 752-753, Figs. 383

**Claim (made 1917):** The skulls of crocodiles (C. porosus, C. americanus, Notosuchus terrestris) are mutually transformable by coordinate deformation.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 5–10 person-days (M) · **priority:** P0

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, composer-2.5, PR 296): verdict mixed. Never tested. The recommended Dryad dataset makes this a standard repo-protocol job.

**Method:** Repo GM protocol: landmark warp families vs box/affine/TPS, LOO-CV, noise floor. · **Data needed:** Dryad 10.5061/dryad.14fn1 (cited in thompson/MODERN_DATASETS_THOMPSON_ALIGNED.md) · **Blockers:** Dataset ingestion (script exists: thompson/ingest_dryad_morphometrics.py)

**Repo evidence:** `thompson/portfolio/ has no crocodilian GM clade; thompson/MODERN_DATASETS_THOMPSON_ALIGNED.md recommends Dryad 10.5061/dryad.14fn1 (crocodylian cranial ontogeny) for exactly this`; `datasets/thompson_1917/t1917-051/result.json`; `thompson/claims/t1917-051_on-the-theory-of-transformations.py`

#### T1917-052 — Thompson (1917), Ch. XVII, pp. 754-755, Figs. 384-386

**Claim (made 1917):** The pelvic/shoulder girdles of dinosaurs and plesiosaurs (Stegosaurus, Brontosaurus; Cryptocleidus etc.) are mutually transformable by coordinate deformation.

**Pass 2 — status:** `UNTESTED` · **effort to settle:** 10–20 person-days (L) · **priority:** P2

**What remains / verdict:** Untested; fossil girdle landmark data would need assembling from literature figures/specimens.

**Method:** Landmark his figures + published fossil reconstructions; warp fit with reconstruction-uncertainty caveat. · **Data needed:** Literature fossil figures (public domain for 1917 sources) · **Blockers:** Fossil reconstruction uncertainty dominates

#### T1917-053 — Thompson (1917), Ch. XVII, pp. 756, Figs. 387

**Claim (made 1917):** The skulls of Dimorphodon and Pteranodon are related by coordinate transformation.

**Pass 2 — status:** `UNTESTED` · **effort to settle:** 7–14 person-days (L) · **priority:** P2

**What remains / verdict:** Untested; two-taxon case, so only a descriptive warp (no noise floor) unless embedded in a pterosaur skull dataset.

**Method:** Descriptive warp + embedding in broader pterosaur GM dataset if available. · **Data needed:** Pterosaur skull landmarks (literature) · **Blockers:** n=2 as stated; needs a clade frame to be a test

#### T1917-054 — Thompson (1917), Ch. XVII, pp. 757-759, Figs. 388-393

**Claim (made 1917):** The pelves of Archaeopteryx and Apatornis (after Heilmann) and intermediate configurations can be found by interpolation between coordinate-transformed forms - i.e. interpolated grids generate biologically real intermediate forms.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 10–20 person-days (L) · **priority:** P2

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 307): verdict mixed. The interpolation-generates-real-forms claim is validated in principle (fish ancestors/fossils) but never on Heilmann's bird pelves.

**Method:** Landmark bird pelves; test whether grid-interpolated intermediates match real intermediate taxa better than chance. · **Data needed:** Bird/fossil pelvis landmarks (literature) · **Blockers:** Fossil pelvis data assembly

**Repo evidence:** `thompson/run_phylogenetic_prediction_engine.py: interpolation/ancestral prediction in Thompson tangent space, tested by hold-out and Monte Bolca fossils (thompson_phylogenetic_prediction_engine.json) - in fishes, not bird pelves`; `datasets/thompson_1917/t1917-054/result.json`; `thompson/claims/t1917-054_on-the-theory-of-transformations.py`

#### T1917-055 — Thompson (1917), Ch. XVII, pp. 760-764, Figs. 394-400

**Claim (made 1917):** The skulls of rhinoceroses and their allies (Hyrachyus, Aceratherium, Titanotherium, tapir, horse, rabbit comparisons after Osborn) differ by coordinate transformation of a common skull diagram.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 5–10 person-days (M) · **priority:** P1

**What remains / verdict:** RESOLVED 2026-10-30 (overnight loop, composer-2.5, PR 317): verdict mixed. Clade-analogue tested (Leporidae); Thompson's actual ungulate series untested.

**Method:** Landmark ungulate skull series (public GM datasets exist); repo protocol. · **Data needed:** Public perissodactyl skull GM data · **Blockers:** Dataset selection

**Repo evidence:** `thompson/portfolio/results/leporidae_skulls.json (rabbit-family skull GM - a modern clade analogue, not Thompson's perissodactyl series)`; `datasets/thompson_1917/t1917-055/result.json`; `thompson/claims/t1917-055_on-the-theory-of-transformations.py`

#### T1917-056 — Thompson (1917), Ch. XVII, pp. 765-767, Figs. 401-402

**Claim (made 1917):** The skull of Eohippus (Hyracotherium) is transformed into that of Equus by coordinate deformation, with actual and hypothetical intermediates (after Heilmann) lying on the transformation.

**Pass 2 — status:** `UNTESTED` · **effort to settle:** 5–10 person-days (M) · **priority:** P0

**What remains / verdict:** Untested, and one of the most iconic untested pairs. Horse skull GM data are public; intermediates make it falsifiable (do real mesohippus-stage skulls lie on the warp path?).

**Method:** Landmark equid skull series; test fossil intermediates against warp-path prediction vs straight Procrustes interpolation. · **Data needed:** Equid skull landmarks (literature/public) · **Blockers:** Assembling the fossil series landmarks

#### T1917-057 — Thompson (1917), Ch. XVII, pp. 769, Figs. 403

**Claim (made 1917):** Human scapulae of differing form (after Dwight) are comparable/transformable by the coordinate method.

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 4–8 person-days (M) · **priority:** P2

**What remains / verdict:** RESOLVED 2026-10-30 (overnight loop, composer-2.5, PR 316): verdict mixed. Untested; low priority (variation within one species, weak claim).

**Method:** Scapula outline warp, descriptive. · **Data needed:** Scapula images/landmarks · **Blockers:** Weak claim, low payoff

**Repo evidence:** `datasets/thompson_1917/t1917-057/result.json`; `thompson/claims/t1917-057_on-the-theory-of-transformations.py`

#### T1917-058 — Thompson (1917), Ch. XVII, pp. 770-773, Figs. 404-408

**Claim (made 1917):** The human skull, inscribed in Cartesian coordinates and deformed, gives the skulls of chimpanzee, baboon and dog: ape/human skull difference is a whole-diagram transformation (notably of the facial/cranial proportions).

**Pass 2 — status:** `DONE_REPO` · **effort to settle:** 5–10 person-days (M) · **priority:** P0

**What remains / verdict:** RESOLVED 2026-10-28 (overnight loop, claude-fable-5-1, PR 305): verdict supported. Never tested on primates. Primate skull GM datasets are abundant and public - this is a high-visibility, feasible test.

**Method:** Repo protocol on a public primate skull landmark set (human/chimp/baboon/dog as outgroup comparison). · **Data needed:** Public primate cranial GM dataset · **Blockers:** Dataset licence/selection

**Repo evidence:** `thompson/portfolio/results/leporidae_skulls.json and other mammalian skull GM in portfolio (method analogue only)`; `datasets/thompson_1917/t1917-058/result.json`; `thompson/claims/t1917-058_on-the-theory-of-transformations.py`

### Epilogue (1917)

#### T1917-059 — Thompson (1917), Ch. Epilogue, pp. ~777-788, Figs. -

**Claim (made 1917):** (Restatement) Form is not exhaustively explained by natural selection on independent parts; physical/mathematical law constrains the forms available to evolution.

**Pass 2 — status:** `NOT_TESTABLE_AS_STATED` · **effort to settle:** 1–1 person-days (S) · **priority:** P2

**What remains / verdict:** Score only via descendants (T1917-002/039/050). Modern framing: developmental/physical constraint on morphospace occupation - the repo's horizon result is one quantified instance.

**Method:** Roll-up of descendant rows. · **Data needed:** - · **Blockers:** Do not score the restatement directly

**Repo evidence:** `thompson/run_morphospace_vs_competition_analysis.py is the repo's operational descendant (G vs R, sympatric competition amplification 3.26-9.27x in jaw modules)`

## Pass-1 extensions (completing the citation pass itself)

- **E1. Verbatim quote for every row** (OCR-located, hand-checked against the pinned scan leaf): 3–5 days. Until then,
  rows without a `quote` are cited by chapter/page/figure, which is a complete bibliographic citation but not a quotation.
- **E2. Sentence-level census:** splitting compound rows (esp. Chs. III, V, VIII, IX) into single-sentence claims would
  take the registry from 59 to an estimated 200+ rows: 5–10 days, low marginal value unless a funder/journal requires it.
- **E3. Figure-by-figure census of Ch. XVII** (Figs. 353–408 individually, incl. every Heilmann/Osborn intermediate): 2–4 days;
  several are already rows T1917-051..058.

## Rules for citing from this registry

1. Always give the year: **Thompson (1917)**, 1st edition — never cite these rows as 'Thompson (1942)'.
2. T1917-050 and all portfolio/blind-test results are **2026 extensions**, not 1917 claims; cite them as such.
3. A `DONE_ELSEWHERE` verdict (e.g. honeycomb, square–cube) must name the modern source when used, not just 'Thompson was right'.
4. `NOT_TESTABLE_AS_STATED` rows (001/002/017/059) may be quoted as programme, never scored as predictions.
