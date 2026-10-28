# T1917-032 — Thompson (1917) Ch. XIII: the shapes of teeth or tusks, with a note on torsion

Loop `thompson1917-t1917-032` (campaign slot 226) validates one dated claim from
D'Arcy Thompson, *On Growth and Form* (1917), Ch. XIII, pp. 612-634, Figs. 318-324:

> Teeth and tusks grow by the same spiral/torsion geometry (with St Venant torsion
> curves describing the twisted prismatic forms).

## What Thompson actually says (pp. 612-634, IA scan `ongrowthform1917thom`, page = leaf - 20)

- p. 619: a horn (or tooth) elongating from a basal growth zone grows *straight* if the rate is
  identical all round, *curved* if greater on one side, into a *logarithmic spiral* if the ratio of
  rates is constant, and into a *spiral in space* (helical component) if the maximal and minimal
  rates are not diametrically opposite.
- p. 618: the cardboard-triangle stacking experiment; Ovis ammon edges 80 : 74 : 45 cm ~ 9 : 8 : 5.
- pp. 621-623: St Venant's torsion — a circular cylinder twists without warping, any other prism
  warps each section into an anticlastic surface; Fig. 321 (St Venant's contour diagram for the
  equilateral triangle) and Fig. 322 (Thompson's own clay-cast contours of an Ovis ammon ring).
- pp. 632-634: "Nail and claw, beak and tooth, all come under this category"; rodent incisors build
  an "unmistakeable logarithmic spiral" by summed growth-rate differences; the elephant tusk's tip
  "nearly coincides with the pole" of its spiral; Mastodon angustidens has a low spiral angle,
  mammoth a high one; very long tusks (mammoth, Babirussa) become "curves of double curvature,
  like a twisted horn".

Thompson gives *no* numerical tusk measurements and never names the narwhal in this chapter.

## Files

| file | role | licence |
|---|---|---|
| `scout_manifest.json` | Step 1 verification dossier (OpenAlex / Commons / IA hits, collision check) | — |
| `t1917_032_the_shapes_of_horns_and_of_tee_reference.json` | equations E1-E9, state variables, boundary conditions, parameter regimes, SHA-256 pins | — |
| `paper_citations.bib` | Thompson 1917, St Venant 1856, Kiladze & Chernova 2018, Kingsley & Ramsay 1988, Raup 1966, Commons credits | — |
| `mammuthus_primigenius_tusk_specimen.jpg` | *Mammuthus primigenius* tusk, Museum Wiesbaden (Commons, Ghedoghedo, 1280 px rendition) — log-spiral / accretion test | CC BY-SA 4.0 |
| `monodon_monoceros_tusk_spiral_specimen.jpg` | *Monodon monoceros* tusk close-up with spiral grooves (Commons/Flickr, 960 px rendition) — torsion analogue | CC BY 2.0 |
| `kiladze2018_narwhal_spiral_table1.csv` | Table 1 of Kiladze & Chernova (2018) Data in Brief 21:1200-1202, doi:10.1016/j.dib.2018.09.029 | CC BY 4.0 |
| `thompson1917_p623_fig321_322_st_venant_torsion_curves.jpg` | IA leaf scan of p. 623 (Figs. 321-322), grayscale 1100 px rendition | public domain |
| `thompson1917_p634_tusk_text.jpg` | IA leaf scan of p. 634 (elephant / mammoth / Babirussa tusk passage) | public domain |
| `result.json` | verdict record (claim_id, dataset digests, verdict, headline numbers) | — |

All digests are pinned in the reference JSON and re-verified at runtime by
`thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py`.

## Model summary

1. Equiangular spiral `r = r0 exp(theta cot psi)`, curvature `kappa = sin(psi) / r`.
2. Differential accretion `kappa = (v_max - v_min) / (w v_mean) = Delta / w`; with constant width a
   circle, with gnomonic widening `w = c_r r` (equivalently `w = c_s s` along the arc) a log spiral with
   `sin psi = Delta / c_r`, i.e. `tan psi = Delta / c_s`.
3. Screw accretion: bending `kappa` plus twist `tau` per unit arc gives a helix; with gnomonic
   scaling a conical log helico-spiral (Thompson's "curve of double curvature").
4. St Venant torsion: uniform twist `tau` makes every surface fibre a helix with `tan phi = rho tau`
   (Kiladze ratio `tan alpha = pi / (tau d)`), and warps a non-circular section by
   `u_z = (tau / 2a)(3x^2 y - y^3)` (triangle) or `u_z = tau (b^2 - a^2)/(a^2 + b^2) x y` (ellipse).

Lean 4 formalisation: `morphoeval/lean/loops/Thompson1917_T1917_032.lean`
(namespace `MathOfBiology.Loop_226`). Math-KG registration:
`morphoeval/math_kg_loops/thompson1917-t1917-032.yaml`.

## Running

```bash
python3 thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py --self-test
python3 thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py            # Pass 2 (full)
python3 thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py --pass1    # Pass 1 baseline
```

Outputs: `docs/media/t1917_032_the_shapes_of_horns_and_of_tee_comparison.png`,
`docs/media/t1917_032_the_shapes_of_horns_and_of_tee_growth.gif`, and `result.json` here.
