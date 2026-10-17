# Raup shell-coiling model: parameters

Parameter definitions, illustrative parameter sets and citations for David M. Raup's
model of shell coiling, read by `../../create_raup_animations.py` and `../../1966-raup.py`.

## Revision of 2026-10-17 — read this first

The 2025-01-05 contents of this folder were written by an AI assistant that did **not**
have the papers. The "definitions" it quoted, the equation/page/figure attributions, and
the six per-taxon parameter sets were not verifiable, and one of them was wrong in a way
that mattered: T was described as "vertical rise per radian", which produced a script in
which translation grew linearly in angle while the whorls grew exponentially, so T had no
visible effect and every shell rendered as a flat disc (MorphoEval audit 2026-10-13,
overall 37.3; [issue #1](https://github.com/fractastical/morpholib/issues/1)). The same
version also drew the generating curve in the horizontal plane (zero axial thickness) and
scaled the axis offset so that the rendered inner/outer-margin ratio was (1+D)/(3−D), not D.

This revision:

- restates the definitions from secondary sources that give Raup's model as equations
  (Noshita 2014; Okabe & Yoshimura 2017; the `ktch` package's `raup()`), says so, and
  drops the unverifiable page/equation attributions;
- replaces the six parameter sets with illustrative points chosen so that each descriptive
  name is geometrically true, and labels them `cf.` — the kind of shell the form resembles,
  not a measured specimen (the old values are kept under `empirical_shells_provenance`);
- corrects the JSTOR identifier of Raup 1966 (1301992, not 1301573) and adds DOIs.

`variable_extraction_log.md` is kept as a record of the 2025 extraction, with a notice.

## The model

A generating curve (the aperture, an ellipse) is swept around a fixed axis while every
dimension grows as a logarithmic spiral. With θ the coiling angle and φ the position on
the generating curve:

| | | |
|---|---|---|
| radial semi-axis | a(θ) = a₀ · W^(θ/2π) | **W**: every whorl is the previous one magnified by W |
| centre's distance from the axis | r_c = a · (1+D)/(1−D) | **D** = inner margin / outer margin = (r_c − a)/(r_c + a) |
| centre's position along the axis | z_c = T · r_c | **T** = dz_c/dr_c: axial movement per unit radial movement; T = 0 planispiral |
| axial semi-axis | b = a / S | **S**: shape of the generating curve (this ratio direction is the repository's convention) |
| surface | X = (r_c + a cos φ) cos θ, Y = (r_c + a cos φ) sin θ, Z = z_c + b sin φ | the generating curve lies in the plane through the axis |

Because a, r_c and z_c all scale with W^(θ/2π), the shell is self-similar — the property
Raup built the model on. In Okabe & Yoshimura's coordinates (unit circle centred at radial
y₀, axial x₀) the same definitions read T = x₀/y₀ and D = (y₀−1)/(y₀+1); Noshita's
formulation uses a translation parameter scaled by 2/(1+D) relative to this one.

`create_raup_animations.py` recovers W, D and T from the rendered mesh (`recover_parameters`)
and refuses to draw unless they match the inputs to 1e-9 and the spire grows monotonically
with T (`self_test`). Per-panel recovered values are written to `raup_shell_metrics.csv`.

## Illustrative parameter sets (`empirical_shells` in `variables.json`)

| panel | cf. | W | D | T | turns | why these values |
|---|---|---|---|---|---|---|
| Planispiral | *Nautilus* | 3.0 | 0.00 | 0.0 | 4 | involute planispiral: T = 0, inner margin on the axis |
| Turreted | *Turritella* | 1.3 | 0.05 | 3.5 | 8 | slow expansion, high translation: tall narrow spire of many whorls |
| Evolute disc | ammonite | 2.0 | 0.45 | 0.0 | 5 | planispiral with a wide umbilicus |
| Conical | *Conus* | 3.5 | 0.00 | 1.5 | 4 | fast expansion, moderate translation: cone with a short spire |
| High-spired | *Cerithium* | 1.6 | 0.10 | 2.5 | 7 | between turreted and conical |
| Globose | *Helix* | 3.0 | 0.10 | 1.0 | 4 | fast expansion, low translation: rounded, low spire |

These are points in the morphospace, not measurements. Raup's own placement of taxa is in
the block diagrams of the 1966 paper and in the ammonoid measurements of the 1967 paper;
neither has been digitised here. The `typical_range` values in `variables.json` (W 1.2–4,
D 0–0.5, T 0–4) are indicative bounds for the parameter-space sheet, not quoted from Raup.

## Citations

- Raup, D. M. (1962). Computer as aid in describing form in gastropod shells. *Science*
  138(3537), 150–152. doi:10.1126/science.138.3537.150
- Raup, D. M., & Michelson, A. (1965). Theoretical morphology of the coiled shell.
  *Science* 147(3663), 1294–1295. doi:10.1126/science.147.3663.1294
- Raup, D. M. (1966). Geometric analysis of shell coiling: general problems. *Journal of
  Paleontology* 40(5), 1178–1190. JSTOR 1301992, doi:10.2307/1301992
- Raup, D. M. (1967). Geometric analysis of shell coiling: coiling in ammonoids. *Journal
  of Paleontology* 41(1), 43–65.
- Noshita, K. (2014). Quantification and geometric analysis of coiling patterns in gastropod
  shells based on 3D and 2D image data. *Journal of Theoretical Biology* 363, 93–104.
  doi:10.1016/j.jtbi.2014.08.010
- Okabe, T., & Yoshimura, J. (2017). Optimal designs of mollusk shells from bivalves to
  snails. *Scientific Reports* 7, 42445. doi:10.1038/srep42445

`paper_citations.bib` carries the same entries in BibTeX.

## Files

- `variables.json` — definitions, ranges, the six parameter sets, formulation; read by both scripts
- `variable_extraction_log.md` — the 2025-01-05 extraction record (superseded; see notice at its top)
- `paper_citations.bib`, `paper_chronology.md`
- `comparisons/` — `generate_paper_comparisons.py` and its (git-ignored) outputs
