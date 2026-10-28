# T1917-023 — Calcospherites grow by radial deposition about a centre (Thompson 1917, Ch. IX)

Claim-validation dataset for slot `thompson1917-t1917-023` of the 100-loop Thompson (1917)
campaign. The claim (`thompson/thompson_1917_claims.json`, id `T1917-023`):

> Concretions and calcospherites grow by radial deposition about a centre (Harting's artificial
> concretions in egg albumen mimic natural ones), giving concentric/spherical forms by physical
> deposition alone. — Thompson (1917) *On Growth and Form*, Ch. IX, pp. 411-487, Figs. 194-235

## What is pinned here

| file | what | provenance | licence |
|---|---|---|---|
| `thompson_1917_p421_figs195-196_harting_calcospherites.jpg` | printed p. 421: Fig. 195 (calcospherites in white of egg), Fig. 196 (single calcospherite, nucleus + striated border) | IA `ongrowthform1917thom`, BookReader leaf `n440` (leafNum 441 = page 421) | public domain (1917, NOT_IN_COPYRIGHT) |
| `thompson_1917_p422_fig197_aggregated_calcospherites.jpg` | printed p. 422: Fig. 197 (later stages: aggregated, polyhedral calcospherites), Fig. 198 (Mya shell / hinge tooth, after Carpenter) | IA leaf `n441` | public domain |
| `thompson_1917_p432_fig208-209_spherocrystal_otoliths.jpg` | printed p. 432: Fig. 208 (concentrically striated spherocrystal), Fig. 209 (plaice otoliths with four age-rings) | IA leaf `n451` | public domain |
| `carmel_ooids_thin_section_specimen.jpg` | **analogue**: Jurassic Carmel Fm. ooids in thin section, scale bar 0.5 mm, nucleus + concentric lamellae + radial-fibrous cortex | Wikimedia Commons `File:CarmelOoids.jpg`, Mark A. Wilson | public domain |
| `liesegang_rings_silver_chromate_gelatine_specimen.jpg` | **analogue**: silver chromate Liesegang rings in gelatine — the exact p. 427 experiment | Wikimedia Commons `File:Liesegang_rings.jpg`, Andras Buki | CC BY-SA 3.0 |
| `scout_manifest.json` | Step 1 dossier: IA digests, OCR key quotes, OpenAlex/Commons candidates, match verdict | — | — |
| `t1917_023_on_concretions_spicules_and_sp_reference.json` | Step 2/3 reference: equations E1-E6, state variables, BCs, parameter regimes, empirical reference values, SHA-256 of every vendored file | — | — |
| `paper_citations.bib` | BibTeX for Thompson, Harting, Liesegang spacing laws, Mullins-Sekerka, DLA/DBM, modern calcite spherulite work | — | — |
| `result.json` | the claim verdict written by the script (`--write-result`) | — | — |

Page images were fetched from `https://archive.org/download/ongrowthform1917thom/page/n<index>.jpg`
with `index = leafNum - 1` and `printed page = leafNum - 20`
(`ongrowthform1917thom_page_numbers.json`, SHA-1 `28ef66b3…` as recorded by IA). The OCR text used for
the quotes is `ongrowthform1917thom_djvu.txt` (IA SHA-1 `90eeaf8b64c7976c23592d164291f10718efda05`,
verified locally).

## Does the dataset match the claim?

Yes (`MATCH`, not `DATASET_MISMATCH`). The claim is about Harting's artificial CaCO3 concretions
in egg albumen; the pinned plates are Thompson's reproductions of those very specimens
(Figs. 195-197, 208) plus the natural counterpart he names (Fig. 209 otoliths, p. 425:
"the typical structure of a calcospherite is no other than that of a pearl, nor does it differ
essentially from that of the otolith"). The two Commons images are modern **analogues** and are
labelled as such everywhere (ooid = natural CaCO3 coated grain with the same radial + concentric
fabric; silver-chromate rings = Thompson's own p. 427 explanation of the concentric striation).

## Governing model that is tested (see reference JSON for the full table)

- **E1** interface-controlled radial deposition `dR/dt = v` (constant lamella thickness);
- **E2** diffusion-limited radial deposition `R dR/dt = κ` (lamellae thin outward);
- **E3** sphericity stability: uniform normal deposition keeps `max ρ − min ρ` fixed so the
  asphericity decays as `1/R`;
- **E4** equal-rate impingement: `|x−c₁|² − |x−c₂|²` is affine in `x`, so neighbours meet on
  perpendicular-bisector planes (Voronoi mosaic, mean 6 neighbours in 2-D) — Thompson's
  "polyhedral, or often regularly hexagonal, outlines";
- **E5** Liesegang supersaturation banding in spherical geometry (bands with `r_{n+1}/r_n → P`) —
  Thompson's "logarithmic ratio";
- **E6** dielectric-breakdown supply sweep `η ∈ {0, 0.5, 1, 1.5, 2}`: compact Eden disc at `η = 0`
  versus spinous DLA at `η ≥ 1` — Thompson's "scanty supply → spinous or digitiform projections".

E1-E5 are machine-checked in `morphoeval/lean/loops/Thompson1917_T1917_023.lean`
(namespace `MathOfBiology.Loop_218`, core Lean 4 only, 0 `sorry`).

## Reproduce

```bash
python3 thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py --self-test
python3 thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py          # figures + result.json
```

Outputs: `docs/media/t1917_023_on_concretions_spicules_and_sp_comparison.png`,
`docs/media/t1917_023_on_concretions_spicules_and_sp_growth.gif`,
`datasets/thompson_1917/t1917-023/result.json`.
