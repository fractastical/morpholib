# Thompson's fish transformations: real-species landmark data

Inputs for [`thompson/1917-thompson-scarus-pomacanthus.py`](../../thompson/1917-thompson-scarus-pomacanthus.py),
which fits D'Arcy Thompson's *Scarus* → *Pomacanthus* coordinate transformation
(*On Growth and Form*, 1917, p. 749, Figs. 375–376) to homologous landmarks on
real species.

## Files

| file | what | source | licence |
|---|---|---|---|
| `trimmed_tps_Claverie2014.csv` (266,054 bytes, SHA-256 `ab5315f3be273285c101299457093199e98ff0b1d0fcf08331cc303b2f07e6bd`) | 17 lateral-view landmarks (x, y) for 791 Indo-Pacific shore-fish species, one row per species, column `Genus_species` then `lm_1x … lm_17y` | Friedman ST, Martinez CM, Price SA, Wainwright PC (2019), Dryad [10.5061/dryad.fn457gr](https://doi.org/10.5061/dryad.fn457gr), file as named there; "a subset of previously unpublished data that were used in Claverie & Wainwright (2014)" | CC0 1.0 |
| (not vendored) `.thompson_cache/ongrowthform1917thom_0769.jp2` (SHA-256 `69cdc9e24bd757b476c763516fb9fcd45db7dc8c7e28f83a4c142417b762c6fa`) | page 749 of the 1917 Cambridge edition as the original JPEG-2000 leaf of the Internet Archive item [`ongrowthform1917thom`](https://archive.org/details/ongrowthform1917thom); the script crops Figs. 375–376 from it | Internet Archive / Biodiversity Heritage Library scan | public domain (Thompson d. 1948; 1917 publication) |

The CSV is vendored because CC0 allows it and because Dryad's file downloads sit
behind a browser challenge that defeats scripted fetching; the script checks the
hash on every run. The scan is fetched on first run (the jp2 inside the item's
zip is byte-stable, unlike the JPEGs the page viewer renders on the fly) and
hash-checked; `--no-scan` skips it.

## Landmarks

Claverie T, Wainwright PC (2014). A morphospace for reef fishes: elongation is
the dominant axis of body shape evolution. *PLoS ONE* 9(11):e112732,
https://doi.org/10.1371/journal.pone.0112732, Fig. 1 — read from the figure,
since the paper gives no landmark table: 1 anterior tip of the lower jaw,
2 anterior tip of the upper jaw, 3 dorsal-fin origin, 4 dorsal-fin end,
5 dorsal and 6 ventral insertion of the caudal fin, 7 anal-fin end, 8 anal-fin
origin, 9 dorsal and 10 ventral insertion of the pectoral fin, 11–17 head
points (opercle, orbit margins 15–16, jaw joint, nape 17). The paper states
that 1, 2 and 11–17 were chosen for feeding mechanics and 3–10 for locomotion.
Photographs: J. Randall's collection at the Bishop Museum. Coordinates are in
the photographs' pixel frames (y up, snout at low x); the script Procrustes-
superimposes them (unit centroid size, no reflection) and rotates the sample so
that the consensus snout → caudal-peduncle axis is horizontal.

## Which of Thompson's fish are in it

| Thompson 1917 | figures | in the dataset |
|---|---|---|
| *Scarus* sp. → *Pomacanthus* ("coaxial circles") | 375–376 | 22 *Scarus* spp., 9 *Pomacanthus* spp. — **tested** |
| "other fishes of the same family of Chaetodonts" ("small variations in the constants of the coaxial curves") | text, p. 749 | 54 *Chaetodon* spp. — tested as *Scarus* → each species |
| *Polyprion* → *Pseudopriacanthus altus* (radial coordinates) | 377–378 | no *Polyprion*; the target's family only (*Priacanthus* ×2, *Heteropriacanthus*, *Cookeolus*) |
| *Diodon* → *Orthagoriscus mola* (concentric circles / hyperbolae) | 381–382 | *Diodon hystrix*, *D. liturosus*; no *Mola* |
| *Argyropelecus olfersi* → *Sternoptyx diaphana* (shear, 70°) | 373–374 | neither (deep-sea) |

So the *Scarus* → *Pomacanthus* pair is the one of Thompson's examples that
these data can test; the others need other sources.

## What the script finds (checkpoint in the top-level README)

Genus means after Procrustes superimposition, residuals as RMS landmark
distance in Procrustes units (centroid size 1):

| transformation of the Cartesian grid | parameters | RMS residual | share of the Procrustes distance explained |
|---|---|---|---|
| similarity only (= the Procrustes distance) | 4 | 0.0312 | 0 % |
| affine | 6 | 0.0259 | 31 % |
| Möbius (isogonal coaxial circles, the literal reading of "coaxial circles") | 6 | 0.0311 | 1 % |
| Möbius ∘ affine (oblique coaxial circles — the two pencils need not be orthogonal, as in Thompson's Fig. 376 as drawn) | 8 | 0.0256 | 32 % |
| thin-plate spline (exact) | — | 0 | 100 %; bending energy 0.678 |

Noise floor: the *Pomacanthus* species lie at RMS 0.0132 from their own mean
(range 0.0090–0.0180), the *Scarus* species at 0.0099. A leave-one-landmark-out
fit of the 8-parameter family predicts held-out landmarks with RMS 0.0371,
worse than the Procrustes distance itself. The body is 56 % deeper relative to
length in *Pomacanthus* (depth/length by landmark extents 0.38 → 0.59; the affine
fit's singular-value ratio is 1.27), which no conformal map can
produce; the landmarks left worst by the affine fit are 7, 2, 4, 8 (anal-fin
end, snout, dorsal-fin end, anal-fin origin). Over all 3782 ordered pairs of
genus means (62 genera with ≥ 3 species) oblique coaxial circles explain a
median 43 %; *Scarus* → *Pomacanthus* at 32 % is at the 31st percentile, i.e.
not an unusually good case for Thompson's family. For *Scarus* → the 54
*Chaetodon* species the family explains 12–61 % (median 35 %).

Caveats the numbers inherit: the landmarks are fin insertions, jaw and head
points, not Thompson's body outline; genus means stand in for his unnamed
*Scarus* sp. and *Pomacanthus*; Thompson drew "approximately" coaxial circles
by eye and did not define the family, so the two parametrisations above are
readings of his figure, with the Möbius composition the more generous one.
