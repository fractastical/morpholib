# Issue #1 — model vs. reference validation

Source: PlanformDB planformDB_2.5.0.edb (sha256-pinned decode)

| Check | Result | Detail |
|---|---|---|
| planarian.source | PASS | PlanformDB planformDB_2.5.0.edb (sha256-pinned decode) |
| planarian.wt_organs | PASS | WT eyes=2 brains=2 pharynx=1 VNC=2 (expect 2/2/1/2) |
| planarian.pharynx_midbody | PASS | pharynx at AP fraction 0.50 from anterior tip (expect ~0.50) |
| planarian.eyes_anterior | PASS | eyes at AP fraction 0.10 from anterior tip (expect <=0.25) |
| planarian.two_vnc_cords | PASS | WT has 2 line-organ VNCs drawn from their two end points (bilateral pair, not a Y-sorted zig-zag) |
| planarian.auricles | PASS | auricle template widens WT head 120 -> 140 model units (+17%) |
| planarian.wt_aspect_vs_specimen | PASS | model WT aspect 4.653 vs segmented specimen 3.726 (ratio 1.25, tolerance 0.65-1.35) |
| planarian.phenotypes_distinct | PASS | #1 Wild type: H1 T1 eyes2 phar1; #24 Double head, one pharynx: H2 T0 eyes4 phar1; #20 Headless: H0 T1 eyes0 phar1; #14 Two tails: H0 T2 eyes0 phar1 |
| raup.parameters_recovered | PASS | max |recovered - nominal| over 6 panels = 4.44e-15 (T = dz_c/dr_c, zc = T*rc) |
| raup.T_monotone | PASS | spire/width at T=0,0.5,1.5,3.0 = [0.0, 0.163, 0.489, 0.978] (strictly increasing) |
| raup.morphospace_ranking | PASS | spire/width ranking ['Turreted', 'High-spired', 'Conical', 'Globose', 'Planispiral', 'Evolute disc'] vs reference-taxon expectation ['Turreted', 'High-spired', 'Conical', 'Globose', 'Planispiral', 'Evolute disc'] |

Figures: `planform_model_vs_reference.png`, `raup_model_vs_reference.png` (docs/media/)

**All pass: True**
