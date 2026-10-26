#!/usr/bin/env python3
"""
Literature counterclaim / literature-test layer for the Thompson (1917) registry.

For every claim in thompson/thompson_1917_claims.json (all claims made in
**1917**, 1st edition), this records:
  * has anyone in the published literature tested it, and when?
  * if there is a COUNTERCLAIM, what exactly did it say -- 'Thompson was
    wrong', 'Thompson was right but this method is better by X%', or
    'Thompson's mechanism is not the mechanism'?
  * the quantitative margin where the literature gives one
    (e.g. Fejes Toth 1964: a better honeycomb base saves <0.35%;
     Weaire & Phelan 1994: a better foam cell uses 0.3% less area;
     Skedros et al. 2007: femoral trabeculae meet at ~70 deg, not 90 deg).

Relation taxonomy (field `relation`):
  REFUTED             literature says the claim, as stated, is wrong
  PARTIALLY_REFUTED   a named part fails, the rest stands
  SUPPORTED           literature test confirms it (no counterclaim)
  MIXED               literature both confirms and refutes parts
  SUPERSEDED          not 'wrong', but a named later method does the job
                      better -- margin stated where known
  QUALIFIED           stands only with named restrictions
  NO_DIRECT_TEST      no literature test of *this* claim on *this* material;
                      closest literature is named and the gap stated -- do
                      not let a nearby result masquerade as a test
  NOT_TESTABLE        philosophical as stated; literature debate is noted
                      but cannot settle it

Repo (2026) results are NOT literature. They are reported separately in the
generated document from the registry's pass2 block, labelled REPO 2026, so a
reader never mistakes our own fit for a published verdict.

Outputs:
  thompson/thompson_1917_counterclaims.json
  docs/THOMPSON_1917_COUNTERCLAIMS.md

Usage: venv/bin/python thompson/build_1917_counterclaims.py [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "thompson" / "thompson_1917_claims.json"
OUT_JSON = ROOT / "thompson" / "thompson_1917_counterclaims.json"
OUT_MD = ROOT / "docs" / "THOMPSON_1917_COUNTERCLAIMS.md"

# claim_id -> literature record. `who_year` must state WHEN the counterclaim
# or test was made; every Thompson claim it answers was made in 1917.
LIT: dict[str, dict] = {
    "T1917-001": {"relation": "NOT_TESTABLE", "who_year": "Gould (1971); Medawar and later commentators", "citation": "Gould, S.J. (1971) 'D'Arcy Thompson and the science of form' (essay; repr. in Gould 1977, Ever Since Darwin)", "counterclaim": "No counterclaim is possible against a programme. Gould's influential qualification: physical forces are real but form is also shaped by evolutionary history and contingency -- 'form as diagram of forces' is incomplete, not false.", "margin": None},
    "T1917-002": {"relation": "QUALIFIED", "who_year": "Gould (1971, and from the 1980s)", "citation": "Gould (1971); Gould & Lewontin (1979) Proc. R. Soc. B 205:581-598 (spandrels paper)", "counterclaim": "Thompson's anti-adaptationist, whole-form programme was first celebrated (Gould 1971, which revived him), then qualified by Gould himself: from the 1980s he refuted Thompson's mechanical determinism in favour of historical contingency and 'tinkering'. Modern synthesis: constraints (Thompson's half) and selection both operate; neither alone explains form.", "margin": None},
    "T1917-003": {"relation": "PARTIALLY_REFUTED", "who_year": "Kleiber (1932); Glazier (2022)", "citation": "Kleiber, M. (1932) Hilgardia 6:315-353; Glazier, D.S. (2022) Proc. R. Soc. B 289:20221605", "counterclaim": "The surface-law application (metabolism proportional to surface, M^2/3) was countered by Kleiber (1932): basal metabolic rate scales as ~M^3/4, not M^2/3. Glazier (2022) counters both: exponents are variable and context-dependent -- there is no universal 2/3 or 3/4. The pure geometry (area L^2, volume L^3) is not in dispute; its use as a metabolic law is.", "margin": "Exponent dispute: 0.67 (surface law) vs 0.73-0.75 (Kleiber 1932) vs 'no universal exponent' (Glazier 2022)"},
    "T1917-004": {"relation": "QUALIFIED", "who_year": "McMahon (1973); Schmidt-Nielsen (1984) and later allometry literature", "citation": "McMahon, T.A. (1973) Science 179:1201-1204 (elastic similarity); Schmidt-Nielsen, K. (1984) Scaling: Why Is Animal Size So Important?", "counterclaim": "Pure geometric similarity (strength L^2 vs weight L^3) does not describe real skeletons; McMahon's elastic-similarity (buckling) criterion predicts different, often better-fitting proportions. Thompson's scaling dilemma is real, his implied geometric solution is not the one animals use in general.", "margin": "Elastic similarity predicts diameter proportional to length^1.5 (vs geometric proportionality); fits many trees/limbs better, neither fits all"},
    "T1917-005": {"relation": "SUPPORTED", "who_year": "Vogel (1994) and the low-Reynolds-number literature", "citation": "Vogel, S. (1994) Life in Moving Fluids, 2nd ed.; Purcell (1977) Am. J. Phys. 45:3-11", "counterclaim": "No substantive counterclaim. Purcell (1977) made Thompson's small-scale regime famous ('life at low Reynolds number'): viscosity, not inertia, rules the small -- exactly Thompson's magnitude argument, quantified.", "margin": None},
    "T1917-006": {"relation": "QUALIFIED", "who_year": "Modern cell-size literature (e.g. Marshall et al. 2012)", "citation": "Marshall, W.F. et al. (2012) Curr. Biol. 22:R256-R260 ('What determines cell size?')", "counterclaim": "Broadly right, with named exceptions Thompson did not treat: some neurons, eggs, syncytia and polyploid cells scale with organism/ploidy. Cell size is actively regulated, not a passive constant.", "margin": None},
    "T1917-007": {"relation": "SUPERSEDED", "who_year": "von Bertalanffy (1957)", "citation": "von Bertalanffy, L. (1957) Q. Rev. Biol. 32:217-231", "counterclaim": "Thompson's plotted growth curves are descriptive. Von Bertalanffy replaced curve-description with a mechanistic balance law (dW/dt = anabolism - catabolism) that predicts the curve family. Later qualification: the 'universal' 2/3 exponent in that law is not universal -- optimally fitted exponents vary widely; the von Bertalanffy function is 'weakly universal' for fish, not for non-fish (PeerJ 2018, 'On the exponent in the Von Bertalanffy growth model').", "margin": "Superseded as method, not refuted as data: no single growth curve/exponent fits all taxa"},
    "T1917-008": {"relation": "MIXED", "who_year": "Huxley (1932) formalised it; Royal Society discussion (1950) and Gould (1966) qualified it", "citation": "Huxley, J.S. (1932) Problems of Relative Growth; Needham, Zuckerman et al. (1950) Proc. R. Soc. B 137:465-525; Gould, S.J. (1966) Biol. Rev. 41:587-640", "counterclaim": "SUPPORTING: Huxley (1932) turned Thompson's differential-growth claim into the allometric law y = b x^k. COUNTER (1950): the allometry equation cannot simultaneously hold between two parts and the whole, and often only fits as piecewise straight lines -- so its constants are 'a rough and ready shorthand of description', not physiological entities. Fitting the law does not prove a growth-rate mechanism.", "margin": "Huxley's k = ratio of specific growth rates; the 1950 objection is logical (inconsistent between part-part and part-whole fits), not a percentage"},
    "T1917-009": {"relation": "SUPPORTED", "who_year": "Douglass's dendrochronology and its successors (from 1919)", "citation": "Douglass, A.E. (1919+) dendrochronology; standard growth-physiology literature thereafter", "counterclaim": "No counterclaim; this became dendrochronology and quantitative growth physiology. Included for completeness: a Thompson claim the literature absorbed rather than fought.", "margin": None},
    "T1917-010": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "Modern regeneration literature exists (urodele limb, planarian) but does not re-test Thompson/Durbin/Ellis's amount-removed vs time-restored relation on his system", "counterclaim": "No published counterclaim found. Absence of a test is the finding: this 1917 claim is still open.", "margin": None},
    "T1917-011": {"relation": "SUPERSEDED", "who_year": "Mitchison & Kirschner (1984); McIntosh (2012)", "citation": "Mitchison, T. & Kirschner, M. (1984) Nature 312:237-242 (dynamic instability / search-and-capture); McIntosh, J.R. (2012) 'Motors or dynamics: what really moves chromosomes?'", "counterclaim": "Thompson/Peddie's bipolar field-of-force mechanism is not the mechanism. The spindle is a self-organising microtubule machine: dynamic instability (1984) lets microtubules search-and-capture chromosomes, and depolymerisation itself generates poleward force with no motor and no ATP (shown in vitro from 1988). The bipolar *geometry* Thompson drew was right; the force-field physics was replaced.", "margin": "Mechanism replaced wholesale (1984-2012); geometry retained"},
    "T1917-012": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "Cleavage is now explained by spindle positioning and cortical mechanics (modern cell-biology literature)", "counterclaim": "No direct literature test of Thompson's two-like-pole field reading of cleavage patterns. The modern mechanism (spindle position -> cleavage furrow) makes the field picture unnecessary, which is a supersession by neglect, not a refutation.", "margin": None},
    "T1917-013": {"relation": "PARTIALLY_REFUTED", "who_year": "Heer et al. / Lecuit-group reviews (2017); Fischer-Friedrich et al. (2014)", "citation": "'Tension, contraction and tissue morphogenesis' (2017, PMC5769629), Box 1 'A note about surface tension'", "counterclaim": "Directly addressed to Thompson (1917): 'While D'Arcy Thompson dismissed the connection between surface tension and contractility, we now understand that this connection is paramount... the tension at cell surfaces is NOT driven by surface tension, as Thompson understood it, but by tension in the cortical actin layer.' Membrane tension's contribution to effective surface tension is often negligible. The sphere-as-null survives; Thompson's passive-liquid mechanism does not.", "margin": "Mechanism refuted; spherical null shape retained as approximation"},
    "T1917-014": {"relation": "QUALIFIED", "who_year": "Modern cell-mechanics comparisons (e.g. PLOS Comput. Biol. 2023, cell shape vs minimal surfaces)", "citation": "'Modelling cell shape in 3D structured environments' (2023) PLOS Comput. Biol. 19:e1011412", "counterclaim": "Constant-mean-curvature (Plateau/Delaunay) shapes are a good model for suspended cells and a measurably wrong one for adherent/structured cells: observed-vs-minimised shape discrepancy is large in L- and V-shaped micropatterns (spectral amplitude difference ~0.15-0.18) and small in triangular patterns (~0.03). Thompson's catalogue is a limiting case, not the general rule.", "margin": "Discrepancy ~0.15-0.18 (L/V patterns) vs ~0.03 (triangular): context decides, contra a universal surface-tension law"},
    "T1917-015": {"relation": "SUPPORTED", "who_year": "Rayleigh (1878, pre-Thompson physics Thompson applied); confirmed in modern fluid mechanics", "citation": "Lord Rayleigh (1878) on liquid-jet/cylinder instability; standard texts thereafter", "counterclaim": "No counterclaim to the physics (Thompson was applying Rayleigh/Plateau, not originating it). Biological instances remain case-by-case demonstrations, not a tested general law of protoplasm.", "margin": "Instability criterion (disturbances with wavelength exceeding the cylinder circumference grow) is textbook"},
    "T1917-016": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "Modern osmotic cell mechanics exists but is not a replication of Koltzoff's decapod-sperm experiment as Thompson cited it", "counterclaim": "No published counterclaim or direct replication found. Open.", "margin": None},
    "T1917-017": {"relation": "NOT_TESTABLE", "who_year": "-", "citation": "-", "counterclaim": "Too general to have attracted a testable counterclaim. Its specific descendants (surface catalysis, membrane localisation) were settled case by case.", "margin": None},
    "T1917-018": {"relation": "QUALIFIED", "who_year": "Farhadifar et al. (2007); vertex-model literature thereafter", "citation": "Farhadifar, R. et al. (2007) Curr. Biol. 17:2095-2104 (Drosophila wing epithelium vertex model)", "counterclaim": "Plateau-style junction laws hold as the energy-minimising limit of vertex models, but real epithelia are active: measured junction angles and cell shapes deviate systematically, and the deviations carry the biology (proliferation, contractility). Thompson's 120-degree rule is the passive limit, not the tissue's operating point.", "margin": None},
    "T1917-019": {"relation": "QUALIFIED", "who_year": "Gibson et al. (2006); Farhadifar et al. (2007)", "citation": "Gibson, M.C. et al. (2006) Nature 442:1038-1041; Farhadifar et al. (2007)", "counterclaim": "Epithelial polygon distributions are froth-like but *not* simply 'hexagons tile best': proliferating epithelia have a characteristic polygon spectrum (substantial pentagon/heptagon fractions) explained by division topology, not by perimeter minimisation alone. Thompson's froth analogy survives, his implied hexagonal optimum does not describe growing tissue.", "margin": "Real epithelia: hexagons are only the modal class (~40-50% in the wing disc), with large 5- and 7-sided fractions"},
    "T1917-020": {"relation": "SUPPORTED", "who_year": "Modern plant morphogenesis (Errera's-rule literature, 2000s-)", "citation": "Errera (1888) origin; modern tests in meristem imaging literature", "counterclaim": "No substantive counterclaim; Sachs/Errera rules survive in refined, mechanically informed form. Thompson was reporting and endorsing Sachs, and the literature kept the rule, not the refutation.", "margin": None},
    "T1917-021": {"relation": "PARTIALLY_REFUTED", "who_year": "Fejes Toth (1964); Hales (2001)", "citation": "Fejes Toth, L. (1964) Bull. AMS 70:468-481, 'What the bees know and what they do not know'; Hales, T.C. (2001) Ann. of Math. 153 (honeycomb conjecture proved)", "counterclaim": "SPLIT VERDICT, and the split is the famous one. (a) The hexagonal tiling IS optimal -- proved by Hales (2001): Thompson/Pappus right. (b) The bees' three-rhombi cell base is NOT the wax-minimum: Fejes Toth (1964) built a better base (two hexagons + two rhombi). Thompson's 'least wax' claim is therefore wrong in its 3D detail. Fejes Toth himself adds that the saving is practically illusory.", "margin": "Better base saves <0.35% of the area of a cell opening (much less of total cell surface) -- Fejes Toth (1964); a 2025 re-optimisation puts the total available saving at ~0.50%"},
    "T1917-022": {"relation": "NO_DIRECT_TEST", "who_year": "Weaire & Phelan (1994) for the foam instance", "citation": "Weaire, D. & Phelan, R. (1994) Phil. Mag. Lett. 69:107-110", "counterclaim": "Catch-all row: no single literature verdict. For its foam instance there IS a precise counterclaim: Kelvin's supposedly minimal equal-volume cell (the structure Thompson's generation took as the minimal partition) is beaten by the Weaire-Phelan structure. Split this row per structure before citing any verdict.", "margin": "Weaire-Phelan uses ~0.3% less surface area than Kelvin's cell (1994)"},
    "T1917-023": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "Harting's artificial-spherite analogy is cited historically; no modern quantitative re-test of radial-deposition geometry on Thompson's examples was found", "counterclaim": "No published counterclaim found. Open.", "margin": None},
    "T1917-024": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "Modern spicule literature is biomineralisation/genetic (silicateins, etc.), not a test of Thompson's deposition-geometry claim", "counterclaim": "No direct test found. The modern mechanism literature (protein-guided silica deposition) supersedes the purely physical picture without ever scoring his ray-angle geometry.", "margin": None},
    "T1917-025": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "Haeckel's radiolarian geometry is much reproduced, rarely measured against geodesic/minimal predictions", "counterclaim": "No published counterclaim or quantitative test found. Open -- and, unusually, still open after a century.", "margin": None},
    "T1917-026": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "-", "counterclaim": "A principle, not a tested claim: no literature verdict exists until it is instantiated in one system (trabeculae are the instance with a verdict -- see T1917-037).", "margin": None},
    "T1917-027": {"relation": "QUALIFIED", "who_year": "Raup (1966)", "citation": "Raup, D.M. (1966) J. Paleontol. 40:1178-1190; Raup & Michelson (1965) Science 147", "counterclaim": "Raup did not refute the logarithmic spiral; he demoted it from law to coordinate system. Real shells need three parameters (W whorl expansion, D, T) plus a generating-curve shape, parameters can drift during growth, and large regions of the possible morphospace are empty -- which a pure log-spiral law does not explain. Modern spiral-fitting (2021, 'Refined methods for estimating spirals', doi:10.26879/1071) needs 12 spiral types and treats 'spiral deviation' as the informative signal: exact log-spiral growth is the null, not the finding.", "margin": "3+ Raup parameters replace Thompson's single spiral constant; deviations from log growth are systematic, not noise"},
    "T1917-028": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "Nautilus buoyancy/CT literature (e.g. Dryad 10.5061/dryad.fc4vf, chambered-shell CT) measures function, not Thompson's septum/spiral geometry", "counterclaim": "No direct test of his stated septal relations found. Open (and cheap to close -- see dataset audit).", "margin": None},
    "T1917-029": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "Foraminiferal morphometrics exists (e.g. fusulinid heterochrony, Dryad 10.5061/dryad.sn02v6x26) but does not test constant chamber-ratio generation on Brady's genera", "counterclaim": "No published counterclaim found. Open.", "margin": None},
    "T1917-030": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "As T1917-014 (single-cell surface-tension forms), but never fitted to Lagena specifically", "counterclaim": "No direct test found. The general qualification of T1917-014 (active cortex, context-dependent fit) would apply if the geometry fit at all.", "margin": None},
    "T1917-031": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "Horn literature is investment/sexual-selection focused (e.g. Dryad 10.5061/dryad.fxpnvx16w), not conical-log-spiral fitting", "counterclaim": "No published test of horns as conical logarithmic helicoids, Thompson-style, was found. Open.", "margin": None},
    "T1917-032": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "-", "counterclaim": "No direct test found. Open.", "margin": None},
    "T1917-033": {"relation": "SUPPORTED", "who_year": "Douady & Couder (1992)", "citation": "Douady, S. & Couder, Y. (1992) Phys. Rev. Lett. 68:2098-2101", "counterclaim": "No counterclaim -- the opposite. Douady & Couder produced Fibonacci phyllotaxis and the golden angle (~137.5 deg) in a purely physical system (repelling ferrofluid drops), with no botanical machinery: the pattern is what a self-organising repulsive system does. This supports Thompson's physicalist reading and refutes only the mystical/numerological readings Thompson himself disliked.", "margin": "Golden angle 137.5 deg and Fibonacci parastichies emerge from one control parameter (drop periodicity) in vitro (1992)"},
    "T1917-034": {"relation": "SUPERSEDED", "who_year": "Stoddard et al. (2017); Birkhead et al. (2019); Narushin et al. (2021)", "citation": "Stoddard, M.C. et al. (2017) Science 356:1249-1254; Birkhead, T.R. et al. (2019) Ibis 161; Narushin, V.G. et al. (2021) Ann. N.Y. Acad. Sci. (universal egg formula)", "counterclaim": "Thompson's descriptive ovoid family was superseded twice over. (1) Mechanism: Stoddard et al. (2017), 49,000 eggs / 1,400 species -- egg shape is set by the membrane (pressure/stiffness), and correlates with flight ability; but flight explains little, and Birkhead et al. (2019) counter-counterclaim that incubation site/posture explains 70-85% in alcids/penguins. (2) Description: a 2021 'universal' egg formula with 4 parameters now does the describing. Nobody's ovoid curve, Thompson's included, is the standard anymore.", "margin": "Flight (hand-wing index) explains only ~4% of egg-shape variation (Stoddard 2017, as re-analysed); incubation site >60%, models 70-85% (Birkhead 2019) -- Thompson's geometry-only account has no explanatory variable at all"},
    "T1917-035": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "-", "counterclaim": "Too broad as stated to have a literature verdict. Split first (urchin test is the measurable instance).", "margin": None},
    "T1917-036": {"relation": "SUPPORTED", "who_year": "Standard beam theory (engineering consensus, before and after 1917)", "citation": "Standard mechanics of materials (beam theory), before and after 1917", "counterclaim": "None: neutral-axis beam mechanics was settled engineering when Thompson wrote and remains so. The biological allocation claim bundled with it is T1917-037/038.", "margin": None},
    "T1917-037": {"relation": "MIXED", "who_year": "Venieratos et al. (1987) FOR; Skedros & Baucom (2007) AGAINST the strict form", "citation": "Venieratos et al. (1987), human femur (PubMed 3631524); Skedros, J.G. & Baucom, S.L. (2007) J. Theor. Biol. 244 (doi:10.1016/j.jtbi.2006.06.029)", "counterclaim": "The trajectorial theory (Wolff/Culmann, endorsed by Thompson, Fig. 335) has a split quantitative verdict. FOR: in frontal sections of human femora, trabecular direction converges on principal stress direction with a mean divergence of only +/-7 deg, better where stress is higher (1987). AGAINST the strict form: femoral-neck trabecular systems are systematically NON-orthogonal (~70 deg, not the 90 deg orthogonal trajectories the theory requires), and non-orthogonal (~60 deg) alignment cuts shear coupling by 33-75% under multidirectional loading -- i.e. real bone beats Thompson/Wolff's predicted arrangement at the real task.", "margin": "+/-7 deg alignment (support, 1987) vs ~70 deg instead of 90 deg intersections and 33-75% shear-coupling advantage of the non-Thompson arrangement (2007)"},
    "T1917-038": {"relation": "SUPERSEDED", "who_year": "Murray (1926)", "citation": "Murray, C.D. (1926) Proc. Natl. Acad. Sci. USA 12:207-214 and J. Gen. Physiol. 9:835-841", "counterclaim": "Thompson (following Hess) stated branching-angle rules qualitatively. Murray (1926) replaced them with a derived optimum: minimum-work branching requires parent radius cubed = sum of daughter radii cubed (r0^3 = r1^3 + r2^3), with angles following from the radii -- and showed Hess's angle rule (Thompson's source) was incomplete because it ignored the continuing parent vessel. Not a refutation of Thompson's economy idea; its replacement by a computable law, nine years later.", "margin": "Murray's law r0^3 = r1^3 + r2^3 (1926); minimum theoretical bifurcation angle 75 deg for vessels"},
    "T1917-039": {"relation": "PARTIALLY_REFUTED", "who_year": "Bookstein (1977, 1978, 1989, and again 2023); Scholtz, Knotel & Baum (2020)", "citation": "Bookstein, F.L. (1977) Math. Biosci. 34:177-219; (1978) Comput. Biol. Med.; (1989) IEEE TPAMI 11:567-585; (2023) Evol. Biol.; Scholtz et al. (2020) Zoomorphology 139 (doi:10.1007/s00435-020-00494-1)", "counterclaim": "The most consequential counterclaim in this registry, in three instalments. (1) Bookstein (1977): as *method*, Thompson's grids are 'as intractable as they seemed in 1917' -- hand-drawn, unfitted, generating no offspring; Thompson 'rests with the possibility of a single system of forces... he was content to exemplify'. (2) Bookstein (1989) supplied the missing method (thin-plate spline: exact, least-bending, decomposable), which is why modern grids are computed, not drawn. (3) Bookstein (2023) then countered his own tradition: exact TPS/Procrustes grids can mislead, and residual-leaving polynomial trend grids often interpret better -- so 'exact fit' is not the final court either. Separately, Scholtz et al. (2020) tested Thompson's own crab example in 3D and found only simple transformations congruent (see T1917-043).", "margin": "Method superseded (1989 TPS is exact at landmarks by construction, vs Thompson's unfitted hand grid); but exactness != predictive superiority (2023 qualification). General-thesis verdict must be per-pair -- see repo 2026 horizon result (registry T1917-050)"},
    "T1917-040": {"relation": "NO_DIRECT_TEST", "who_year": "Bookstein (1977) on the grid method generally", "citation": "Bookstein (1977) Math. Biosci. 34:177-219 covers Durer/Camper-style grids as part of the general intractability critique", "counterclaim": "No test of Thompson's Durer-derived human proportion grids as such. The general method critique (T1917-039) applies.", "margin": None},
    "T1917-041": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "-", "counterclaim": "No literature test of lanceolate->ovate->cordate leaf transformation under radial coordinates was found. (The repo has tested Thompson's Fig. 359 radial-cordate *formula* on ant heads -- an analogue, not this claim; see registry.) Open.", "margin": None},
    "T1917-042": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "-", "counterclaim": "No published test of the Oithona->Sapphirina pair was found. The repo's 2026 test (84.2% vs 20.4% box, on his drawings) is REPO, not literature -- see the generated document's repo column.", "margin": None},
    "T1917-043": {"relation": "PARTIALLY_REFUTED", "who_year": "Scholtz, Knotel & Baum (2020)", "citation": "Scholtz, G., Knotel, D. & Baum, D. (2020) Zoomorphology (doi:10.1007/s00435-020-00494-1), 'D'Arcy W. Thompson's Cartesian transformations: a critical evaluation'", "counterclaim": "THE direct literature counterclaim to a Thompson figure. Using Thompson's drawings AND laser-scanned real carapaces (50 micrometre surface scans, projected to silhouettes, 8 landmarks): 'Only the simple transformations show a reasonable form of congruence. In particular, the transformations to majoid spider crabs reveal a complicated transformation of grids with partly crossing lines.' Accurate matching needs increasingly complex, non-Cartesian transformations -- undermining Thompson's simple explanatory model. Curious exception: the lithodid (convergent hermit-crab descendant) transforms easily. Note Thompson also silently included Paralomis, which is not a true crab at all.", "margin": "Qualitative-geometric verdict (crossing grid lines = failed diffeomorphism in the majoid cases), on both his drawings and real 3D-scanned carapaces (2020)"},
    "T1917-044": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "-", "counterclaim": "No published test of the amphipod series found. (Scholtz et al. 2020 tested the adjacent crab figure, not this one.) Open.", "margin": None},
    "T1917-045": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "-", "counterclaim": "No literature test of Argyropelecus->Sternoptyx at 70 degrees was found (deep-sea pair, no landmark dataset until the repo's archival landmarking). Open in the literature; repo-tested 2026.", "margin": None},
    "T1917-046": {"relation": "NO_DIRECT_TEST", "who_year": "Bookstein (1978) on this figure's method", "citation": "Bookstein, F.L. (1978), as discussed in the repo's validating script: Thompson 'drew the grids by hand and left no instructions for fitting them'", "counterclaim": "No published *quantitative* test of Scarus->Pomacanthus existed before this repo's (the audit states: 'there is still no published value to compare against -- nobody has fitted this pair'). The literature counterclaim is methodological only: an unfitted hand grid is not evidence. The quantitative refutation (Mobius ~1%, outline IoU worse than no deformation) is REPO 2026, not literature -- keep the two apart.", "margin": "Literature: none (no numbers). Repo 2026: see pass2 -- do not cite the repo numbers as 'the literature'"},
    "T1917-047": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "-", "counterclaim": "No literature test of the 'small variations in the constants' sentence was found. Repo 2026 tested the variance half (median 35% over 54 Chaetodon spp.); the constants half was still open at registry time and is the cheapest item in the whole programme.", "margin": None},
    "T1917-048": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "-", "counterclaim": "No literature test of the Polyprion->Antigonia fan series was found. Open in the literature; repo-tested 2026 at landmark level.", "margin": None},
    "T1917-049": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "-", "counterclaim": "No literature test of Diodon->Mola by concentric circles/hyperbolae was found. Open in the literature; repo-tested 2026 at landmark level.", "margin": None},
    "T1917-050": {"relation": "QUALIFIED", "who_year": "This row is a 2026 repo generalisation, not a 1917 claim and not literature", "citation": "Repo: thompson/1917-thompson-four-families-atlas.py and companions (2026)", "counterclaim": "Not applicable to the literature: Thompson never made this generalisation in this form. Its nearest literature relatives are Bookstein (1977) (method intractable) and Scholtz et al. (2020) (simple transformations only). Cite it only as 'Dietz/repo (2026)'.", "margin": "Repo 2026: intrageneric median 48.0% explained, 54.2% reach noise floor; interfamily only 0.3% reach the floor; horizon ~45-50 Myr"},
    "T1917-051": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "Crocodylian cranial GM is an active literature (e.g. Dryad 10.5061/dryad.14fn1) but is not framed as a test of Thompson's Fig. 383", "counterclaim": "No published test of Thompson's crocodile-skull transformation was found. The data to do it have existed since at least 2013 (dryad.14fn1) -- the gap is framing, not data.", "margin": None},
    "T1917-052": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "-", "counterclaim": "No literature test found. Fossil girdle transformations remain illustrations.", "margin": None},
    "T1917-053": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "-", "counterclaim": "No literature test found (n=2 as stated; needs a clade frame to become testable).", "margin": None},
    "T1917-054": {"relation": "QUALIFIED", "who_year": "Bookstein (1989) and the TPS/ancestral-reconstruction literature", "citation": "Bookstein (1989) IEEE TPAMI 11:567-585; Rohlf's tps software literature (from 1990)", "counterclaim": "Heilmann's interpolation idea became standard machinery (TPS interpolation between landmark configurations is exactly how morphometricians now generate intermediates), so the *method* was vindicated by adoption. The *biological* claim -- that grid-interpolated forms are the real intermediate forms -- is only testable against fossils, and the literature does not test it on Heilmann's bird pelves.", "margin": None},
    "T1917-055": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "Osborn's rhinoceros/horse skull series are historically famous; no Thompson-transformation test of them was found", "counterclaim": "No direct test found. Open.", "margin": None},
    "T1917-056": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "Equid skull evolution has a modern GM literature (e.g. Dryad 10.5061/dryad.crjdfn3g3, 'horses deviate from a pervasive pattern of skull shape evolution') -- its very title signals that horse skulls do NOT follow the simple pervasive pattern, but it is not a test of Thompson's Eohippus->Equus grid", "counterclaim": "No direct Thompson test found. Closest literature is suggestive-counter (horses as pattern-breakers), not a verdict on Fig. 401-402.", "margin": None},
    "T1917-057": {"relation": "NO_DIRECT_TEST", "who_year": "-", "citation": "-", "counterclaim": "No literature test found. Weak claim, low payoff either way.", "margin": None},
    "T1917-058": {"relation": "NO_DIRECT_TEST", "who_year": "Bookstein (1977) method critique applies", "citation": "Bookstein (1977) Math. Biosci. 34:177-219 (general); primate cranial GM literature is large but not Thompson-framed", "counterclaim": "No published test of human->chimpanzee/baboon/dog skull transformation as Thompson drew it (Figs. 404-408) was found, despite abundant primate landmark data. Open -- and the most visible untested skull claim in the book.", "margin": None},
    "T1917-059": {"relation": "NOT_TESTABLE", "who_year": "Gould (1971; 1980s) -- see T1917-002", "citation": "Gould (1971); Gould & Lewontin (1979)", "counterclaim": "Restatement of T1917-002: the literature's verdict is the constraint/selection synthesis, not a win for either 1917 or a purely adaptationist reading.", "margin": None},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    reg = json.loads(REGISTRY.read_text())
    claims = reg["claims"]
    missing = [c["id"] for c in claims if c["id"] not in LIT]
    extra = [k for k in LIT if k not in {c["id"] for c in claims}]
    if missing or extra:
        print(f"FAIL coverage: missing={missing} extra={extra}")
        return 1
    relations = {"REFUTED", "PARTIALLY_REFUTED", "SUPPORTED", "MIXED", "SUPERSEDED", "QUALIFIED", "NO_DIRECT_TEST", "NOT_TESTABLE"}
    for cid, rec in LIT.items():
        if rec["relation"] not in relations:
            print(f"FAIL {cid}: bad relation {rec['relation']}")
            return 1
        if rec["relation"] not in ("NO_DIRECT_TEST", "NOT_TESTABLE") and rec["who_year"].strip() == "-":
            print(f"FAIL {cid}: a literature verdict requires who/when")
            return 1
    print(f"PASS {len(LIT)} claims covered, all Thompson claims dated 1917")
    if args.check:
        return 0

    from collections import Counter

    cnt = Counter(v["relation"] for v in LIT.values())
    out = {"source_claims": "Thompson (1917) On Growth and Form, 1st edition -- every claim dated 1917",
           "registry": "thompson/thompson_1917_claims.json",
           "relation_taxonomy": "REFUTED | PARTIALLY_REFUTED | SUPPORTED | MIXED | SUPERSEDED | QUALIFIED | NO_DIRECT_TEST | NOT_TESTABLE",
           "counts": dict(cnt), "claims": {}}
    lines = [
        "# Thompson (1917) — Who Tested It, Who Said He Was Wrong, and by How Much",
        "",
        "Every Thompson claim below was made in **1917** (*On Growth and Form*, 1st edition). This document adds the",
        "layer the claim inventory deliberately left out: **the published literature's verdict** -- who tested the",
        "claim, **when**, what the counterclaim actually said, and the **quantitative margin** where one exists",
        "('Thompson was wrong', 'right but beaten by X%', 'right geometry, wrong mechanism').",
        "Repo (2026) results are *not* literature and are labelled **REPO 2026** wherever they appear.",
        "Generated by [`thompson/build_1917_counterclaims.py`](../thompson/build_1917_counterclaims.py);",
        "machine-readable: [`thompson/thompson_1917_counterclaims.json`](../thompson/thompson_1917_counterclaims.json).",
        "",
        f"**Literature verdicts across {len(claims)} claims (1917):** " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())),
        "",
        "## The counterclaims worth quoting (with the numbers)",
        "",
        "| Thompson (1917) | Counterclaim / test (when) | What it said | Margin |",
        "|---|---|---|---|",
        "| Honeycomb = least wax (Ch. VIII) | Fejes Toth (**1964**) | Bees' 3-rhombi base is NOT optimal; a 2-hexagon+2-rhombi base uses less wax. Hexagonal tiling itself IS optimal (Hales, **2001**) | Better base saves **<0.35%** of a cell opening |",
        "| Kelvin/minimal foam cells (Ch. VIII) | Weaire & Phelan (**1994**) | A two-cell-type foam beats Kelvin's 'minimal' cell | **0.3% less** surface area |",
        "| Trabeculae follow stress trajectories (Ch. XVI, Fig. 335) | Venieratos et al. (**1987**) FOR; Skedros & Baucom (**2007**) AGAINST | Femur trabeculae track stress to ±7°, BUT meet at ~70° not the predicted 90°, and the non-orthogonal arrangement is mechanically better | **±7°** vs **~70°≠90°**; **33–75%** less shear coupling for the non-Thompson arrangement |",
        "| Cells = passive surface-tension drops (Ch. V) | Cell-mechanics reviews (**2017**) | 'The tension at cell surfaces is NOT driven by surface tension, as Thompson understood it, but by cortical actin' | Mechanism refuted; shape null retained |",
        "| Mitotic figure = bipolar force field (Ch. IV) | Mitchison & Kirschner (**1984**) | No field: microtubule dynamic instability + search-and-capture; depolymerisation alone moves chromosomes | Mechanism replaced wholesale |",
        "| Transformation grids as method (Ch. XVII) | Bookstein (**1977, 1989, 2023**) | 1977: grids 'as intractable as in 1917', unfitted, no offspring. 1989: TPS replaces them (exact). 2023: exact TPS itself can mislead; residual polynomial grids often interpret better | Method superseded, then the replacement qualified |",
        "| Crab carapaces transform simply (Fig. 369) | Scholtz, Knotel & Baum (**2020**) | On his drawings AND 50 µm laser scans of real crabs: only simple transformations congruent; majoid (spider-crab) grids cross and fail | Grid-crossing = failed transformation (geometric verdict) |",
        "| Surface law of metabolism (Ch. II) | Kleiber (**1932**); Glazier (**2022**) | Metabolism scales ~M^3/4 not M^2/3 (1932); in fact no universal exponent exists (2022) | **0.67 vs 0.75 vs variable** |",
        "| Eggs as descriptive ovoids (Ch. XV) | Stoddard et al. (**2017**); Birkhead et al. (**2019**) | Shape is membrane mechanics + flight/incubation, not a curve family; the two modern camps then counter each other | Flight explains **~4%**; incubation **70–85%** (in alcids/penguins) |",
        "| Branching angles (Ch. XVI, after Hess) | Murray (**1926**) | Hess's rule (Thompson's source) is incomplete; minimum work gives r0³ = r1³ + r2³ | A computable law replaces a qualitative rule, 9 years later |",
        "| Growth curves / relative growth (Ch. III) | von Bertalanffy (**1957**); R. Soc. discussion (**1950**) | Curves replaced by a balance law (whose own exponent then failed universality); allometric constants are 'shorthand, not physiological entities' | No universal growth exponent |",
        "| Log-spiral shells as law (Ch. XI) | Raup (**1966**) | Demoted from law to coordinate system: 3+ parameters, drifting constants, empty morphospace regions unexplained | 1 constant → 3+ parameters + deviations as signal |",
        "| *Scarus*→*Pomacanthus* coaxial circles (p. 749) | **Nobody, in the literature** | No published quantitative test existed. The refutation (Möbius ~1%; outline worse than nothing, IoU 0.618 vs 0.662) is **REPO 2026**, not literature | Literature margin: none. Do not cite repo numbers as literature |",
        "",
        "## Per-claim literature verdict (all claims: Thompson, 1917)",
        "",
    ]
    for c in claims:
        rec = LIT[c["id"]]
        out["claims"][c["id"]] = {
            "citation_thompson_1917": f"Thompson (1917), Ch. {c['chapter']}, pp. {c['pages']}, Figs. {c['figures']}",
            "claim": c["claim"], "relation": rec["relation"], "who_and_when": rec["who_year"],
            "literature_citation": rec["citation"], "counterclaim": rec["counterclaim"],
            "quantitative_margin": rec["margin"],
            "repo_2026_status_separate": c["pass2"]["status"],
        }
        lines += [
            f"### {c['id']} — Thompson (1917), Ch. {c['chapter']}, pp. {c['pages']}, Figs. {c['figures']} — `{rec['relation']}`",
            "",
            f"**Claim (1917):** {c['claim']}",
            "",
            f"**Literature ({rec['who_year']}):** {rec['citation']}",
            "",
            f"**Counterclaim / verdict:** {rec['counterclaim']}",
            "",
        ]
        if rec["margin"]:
            lines += [f"**Quantitative margin:** {rec['margin']}", ""]
        lines += [f"**Repo 2026 (not literature):** status `{c['pass2']['status']}` -- see claim inventory.", ""]
    lines += [
        "## Rules for using this document",
        "",
        "1. Date both sides, always: 'Thompson (1917) claimed X; **Fejes Toth (1964)** showed Y (<0.35%)'. An undated 'Thompson was wrong' is not a citation.",
        "2. `SUPERSEDED` is not `REFUTED`: Murray (1926), von Bertalanffy (1957) and Bookstein (1989) replaced Thompson's *method* while keeping much of his question. Say which.",
        "3. `NO_DIRECT_TEST` (30 claims) means the literature has *not* spoken -- including several famous figures (primate skulls, horse skull, horns, forams). Do not let a neighbouring result (e.g. Bookstein on grids generally) masquerade as a test of a specific pair.",
        "4. Never cite REPO 2026 numbers as literature. The *Scarus*->*Pomacanthus* refutation is ours (2026) precisely because the literature left it untested for 109 years.",
        "5. Where the literature counters itself (trabeculae 1987 vs 2007; eggs 2017 vs 2019; TPS 1989 vs Bookstein 2023), record both rounds -- the second counterclaim is usually the more interesting one.",
        "",
    ]
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}: {dict(cnt)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
