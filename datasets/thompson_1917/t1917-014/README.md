# T1917-014 — The Forms of Cells: are Lagena tests Delaunay surfaces of revolution?

Slot `thompson1917-t1917-014` (loop 211) of the Thompson (1917) claim-validation campaign.

## Claim under test (dated 1917)

> "The forms of single cells and cell-like organisms (Plateau's oil globules; Lagena and other
> forams/protists) are surfaces of revolution from the minimal-surface catalogue: sphere,
> cylinder, unduloid, nodoid, catenoid."
> — Thompson, D.W. (1917) *On Growth and Form*, Ch. V "The Forms of Cells", pp. 201-276,
> Figs. 61-67, 72-96.

Thompson's own wording for the test organism (p. 256, under Fig. 84 "Various species of
Lagena, after Brady"): the genus "consists of flask-shaped shells, whose surface is simply that
of an unduloid, or more frequently, like that of a flask itself, an unduloid combined with a
portion of a sphere". He adds (p. 257) that Lagena "suggests to our minds a hanging drop, drawn
out to a long and slender neck".

## Mathematics formalised

| id | equation | Thompson locus |
|----|----------|----------------|
| E1 | Laplace: `p = T (1/R1 + 1/R2) = 2 T H` | pp. 216-217 |
| E2 | CMC surface of revolution: `2H = 1/(r sqrt(1+r'^2)) - r''/(1+r'^2)^(3/2)` | p. 218 (verbal) |
| E3 | Delaunay first integral: `r/sqrt(1+r'^2) - H r^2 = C`; necks/bulges solve `H r^2 - r + C = 0`, so `r_min + r_max = 1/H`, `r_min r_max = C/H` | p. 218 (Delaunay 1841 footnote) |
| E4 | Kenmotsu parametrisation: `r(s) = sqrt(1 + B^2 + 2B sin 2Hs)/(2H)`, `dz/ds = (1 + B sin 2Hs)/sqrt(...)`; `B=0` cylinder, `0<B<1` unduloid, `B=1` spheres, `B>1` nodoid; catenoid is the `H=0` member `r = c cosh(z/c)` | Fig. 61 A-C, p. 218-219 |
| E5 | neck ratio `rho = r_min/r_max = (1-B)/(1+B) = (a-c)/(a+c)` (B = eccentricity of the rolling ellipse) | Fig. 61 B |
| E6 | unduloid period `Lambda = 4 a E(e)`; bead aspect ratio `Lambda/(2 r_max) = 2E(e)/(1+e)` lies in `[1, pi]` | — (consequence) |
| E7 | Delaunay flask: sphere `R` C1-joined (join radius `r_j^2 = R(1-B^2)/(4H(1-RH))`) to an unduloid, whose neck is C1-joined to a cylinder of length `L_neck` | p. 256 "unduloid combined with a portion of a sphere" |
| E8 | Delaunay bead: one unduloid bulge-to-neck window extended by its neck cylinder | p. 256 "simply that of an unduloid" |

Lean 4 proofs of the algebraic content (E3 root identities, E5 monotonicity, Kenmotsu radius
envelope, catalogue trichotomy) live in `morphoeval/lean/loops/Thompson1917_T1917_014.lean`.

## Data in this directory

| file | what | provenance | licence | SHA-256 |
|------|------|-----------|---------|---------|
| `thompson_1917_leaf0276_p256_fig84.jp2` | printed p. 256 of the 1917 Cambridge edition, Fig. 84 (13 *Lagena* after Brady 1884) | Internet Archive `ongrowthform1917thom`, JPEG-2000 leaf 0276 (page = leaf - 20) | public domain | `ed440583…9661f1` |
| `thompson_1917_fig84_lagena_after_brady.png` | greyscale crop of Fig. 84 (fractions x 0.18-0.87, y 0.265-0.68) that the script segments | derived from the leaf above | public domain | `c02d29af…a00dec9` |
| `kireenko_2022_bdj_e87457_fig8_station5656mc.jpg` | light-microscope plate of unilocular and multilocular benthic forams, scale bars 100 µm (61a/62a *Lagena* sp., 63 *Oolina*, 57-60 *Fissurina*; 64 *Globobulimina*, 65 *Pyrgo*, 66 *Quinqueloculina*) | Kireenko et al. 2022, Biodiversity Data Journal 10:e87457, via Wikimedia Commons (sha1 `3528580…6f41dc`) | CC BY 4.0 | `3b57acc0…a63ff6` |
| `dryad_28816_metadata.json`, `dryad_28816_files.json` | metadata + file manifest of the pre-scouted Dryad dataset doi:10.5061/dryad.28816 | Dryad API v2 | CC0 | `6c25c869…`, `f958f61f…` |
| `openalex_foraminifera_shell_morphology_outline.json` | pinned OpenAlex response for the slot's search query | OpenAlex API | CC0 | `524abe61…` |
| `scout_manifest.json` | Step 1 dossier (dataset verification, Commons search, collision check) | this loop | CC BY 4.0 | — |
| `t1917_014_the_forms_of_cells_reference.json` | equations, state variables, boundary conditions, regime table, biological claims, digests | this loop | CC BY 4.0 | — |
| `paper_citations.bib` | BibTeX for Thompson 1917, Delaunay 1841, Plateau 1873, Kenmotsu 1980/2003, Brady 1884, Kireenko 2022, Prazeres 2015 | this loop | — | — |
| `result.json` | verdict record for the claim (written by the script in Pass 2) | this loop | CC BY 4.0 | — |

Full digests are in `t1917_014_the_forms_of_cells_reference.json` → `input_dataset_sha256` and are
re-verified by the script at load time.

### Pre-scouted dataset: `DATASET_MISMATCH`

`thompson/thompson_1917_dataset_audit.json` names doi:10.5061/dryad.28816 (Prazeres, Uthicke &
Pandolfi 2015) as the top Dryad hit. Its metadata and file manifest were fetched and pinned: it
contains one spreadsheet of seawater pH/DIC carbonate-system parameters and one supporting
statistics document for an ocean-acidification experiment on the multilocular large benthic
foraminifera *Amphistegina lessonii* and *Marginopora vertebralis* (buoyant weight, ATPase,
micro-CT density). It holds no outlines of any unilocular test and cannot test the claim, so it
is **not** used. The same audit row sets `digitize_1917_figures: true`, and the source registry
names "digitise Thompson's own 1917 plates from the pinned IA scan" as the fallback for every
figure claim, so the verdict rests on (i) Thompson's own Fig. 84 specimens and (ii) modern CC-BY
photographs of the same genus.

## Method (script `thompson/claims/t1917-014_the-forms-of-cells.py`)

1. Verify SHA-256 of the inputs.
2. Segment the 13 Fig. 84 silhouettes (background-normalised threshold, closing, hole fill,
   opening, area > 3000 px) and the chosen panels of the Kireenko plate (Otsu on luminance
   inside pinned panel boxes, largest component, hole fill).
3. For each silhouette: principal-axis rotation, per-row left/right half-widths about a
   least-squares axis → meridional radius profile `r(z)`, bilateral asymmetry index
   `mean|r_L - r_R| / r_max` (tests the surface-of-revolution premise C1).
4. Fit by bounded least squares: Delaunay flask (E7), Delaunay bead (E8), ellipsoid null (not
   CMC), free-form degree-6 polynomial in `r^2` (upper bound of attainable fit).
5. Report normalised RMSE, Pearson r, KS statistic, 2-fold CV RMSE, fitted `B`, neck ratio,
   L/W, and a per-specimen Delaunay-compatibility decision; aggregate into a verdict.
6. Render `docs/media/t1917_014_the_forms_of_cells_comparison.png` and the hanging-drop →
   Lagena animation `docs/media/t1917_014_the_forms_of_cells_growth.gif`.

Run:

```bash
python3 thompson/claims/t1917-014_the-forms-of-cells.py --self-test
python3 thompson/claims/t1917-014_the-forms-of-cells.py
```
