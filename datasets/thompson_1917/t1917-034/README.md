# `datasets/thompson_1917/t1917-034/` — Thompson (1917) Ch. XV: the shapes of eggs

Reference package for claim **T1917-034** (slot 228, `thompson1917-t1917-034`), validated by
`thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py`.

> *"The eggs of birds and all other hard-shelled eggs ... are simple solids of revolution; but they differ
> greatly in form, according to the configuration of the plane curve by the revolution of which the egg is,
> in a mathematical sense, generated."* — D'Arcy Thompson, *On Growth and Form* (1917), p. 652.

The claim under test: *birds' (and other hard-shelled) eggs are solids of revolution whose meridian curves form
a definite, species-varying family of ovoids, comparable geometrically across species.*

## Model

The meridian curve is written in the Preston (1953) four-parameter ovoid family that Biggins, Thompson &
Birkhead (2018) showed to be the most accurate published egg equation:

$$y(x) = \left(c_0 + c_1 x + c_2 x^2 + c_3 x^3\right)\sqrt{1 - x^2}, \qquad -1 \le x \le 1,$$

with the axis of revolution along \(x\) and the half-length normalised to 1. The sphere is \(c_0 = 1\),
\(c_1 = c_2 = c_3 = 0\); the prolate ellipsoid (Thompson's grebe / pelican) is \(c_1 = c_2 = c_3 = 0\),
\(0 < c_0 < 1\); the hen's ovoid and the guillemot's conical egg need \(c_1, c_3 \ne 0\). Each pole is locally
spherical with radius \(R_\pm = p(\pm 1)^2\), so Biggins' polar-asymmetry index has the closed form
\(\mathrm{PA} = \max\big(p(-1)^2/p(1)^2,\; p(1)^2/p(-1)^2\big)\) (verified against their numeric column on all
1389 eggs, \(r = 0.99997\)). Thompson's own envelope equation \(p_n + T(1/r + 1/r') = P\) (p. 658) is used as
a set of pole/equator curvature constraints that close the family (see `*_reference.json`).

## Contents

| file | what | license / provenance |
|---|---|---|
| `t1917_034_on_the_shapes_of_eggs_and_of_c_reference.json` | equations, state variables, parameter-regime tables, empirical claims C1–C8, SHA-256 pins | this loop |
| `scout_manifest.json` | Step 1 dossier (Dryad verification, OpenAlex, Commons, collision check) | this loop |
| `paper_citations.bib` | BibTeX for Thompson 1917/1908, Preston 1953, Biggins 2018, Deeming 2018, Stoddard 2017, Baker 2002, Troscianko 2014, Bashforth & Adams 1883 | this loop |
| `biggins2018_data_for_comparing_indices.csv` | 1389 eggs, 30 auk + penguin species: Preston \(c_0..c_3\), Elongation, Pointedness, PolarAsymmetry, egg mass, adult body mass | CC0, Dryad `doi:10.5061/dryad.8kv2b20` (`Analysis.zip` member) |
| `biggins2018_range_of_egg_shapes.csv` | 10 eggs spanning the avian shape range (kingfisher → jacana), incl. guillemot egg C126 | CC0, same |
| `biggins2018_comparison_of_fits.csv` | 132 eggs: Preston residual and error ratios of Baker / Troscianko / Carter alternatives | CC0, same |
| `biggins2018_C126_uria_aalge_silhouette.jpg` | back-lit silhouette of guillemot egg C126 (L 92.5 mm, B 49.8 mm); downscaled 5175×3450 → 1800×1200 | CC0, same |
| `deeming2018_jab_egg_shape_composition.xlsx` / `.csv` | 181 species, 30 orders: female & egg mass, shell / yolk / albumen mass, outline PC1 (elongation), PC2 (asymmetry) | CC0, Dryad `doi:10.5061/dryad.q8rp3` |
| `mwnh_0646_strix_aluco_egg.jpg` | Tawny Owl egg — Thompson's *spherical* class | CC BY-SA 3.0, Klaus Rassinger & Gerhard Cammerer, Museum Wiesbaden, via Wikimedia Commons |
| `mwnh_1921_podiceps_cristatus_eggs.jpg` | Great Crested Grebe eggs — *ellipsoidal, symmetric ends* | CC BY-SA 3.0, same |
| `mwnh_0308_vanellus_vanellus_eggs.jpg` | Northern Lapwing ("green plover") eggs — *conical plover egg* | CC BY-SA 3.0, same |
| `mwnh_0481_uria_aalge_eggs.jpg` | Common Guillemot eggs — *blunt one end, pointed the other* | CC BY-SA 3.0, same |
| `commons_bird_egg_collection_53631949639.jpg` | 1930–1950 British egg-collection drawer (context plate; not segmented) | CC BY 2.0, Tim Sheerman-Chase |
| `result.json` | claim verdict (written by Step 12) | this loop |

All SHA-256 digests of the vendored files, and of the upstream originals they were derived from, are pinned in
`t1917_034_on_the_shapes_of_eggs_and_of_c_reference.json` (`datasets.files`, `datasets.upstream_originals`) and
re-verified at runtime by the script (`--verify-only`).

## Dataset-match statement (Step 1)

`thompson/thompson_1917_dataset_audit.json` pre-scouted T1917-034 as `PUBLIC_SCRIPTABLE` with Dryad top hit
`doi:10.5061/dryad.q8rp3`. Both title-relevant Dryad hits were fetched (through Dryad's Anubis proof-of-work
interstitial), pinned, and found to **match** the claim's species and variables: cross-species bird egg
outline shape, with per-egg meridian-curve parameters (Biggins) and 181-species outline PCs with body-mass
covariates (Deeming). No analogue dataset was substituted.

## Note on Thompson's figures

The Ch. XV figures cited in the claim row (Figs. 328–330) belong to the sea-urchin corollary (pp. 661–669); the
egg discussion (pp. 652–660) is textual and carries the envelope equation. Thompson's named species were
therefore matched to museum photographs rather than digitised from figures.
