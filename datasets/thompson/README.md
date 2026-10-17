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
| (not vendored) `.thompson_cache/Perroquet_barbe_bleue_(Scarus_ghobban)_(Ifremer_00762-87435).jpg` (SHA-256 `732f1e68da7e3611696613f69215115b1b9056a4db310892d30de487a76a2a90`) | a collected *Scarus ghobban* photographed in lateral view on a white background; the script segments its silhouette | Jérôme Huet, IFREMER (image.ifremer.fr 00762-87435), via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Perroquet_barbe_bleue_(Scarus_ghobban)_(Ifremer_00762-87435).jpg) | CC BY 4.0 |
| (not vendored) `.thompson_cache/Semicircle_angelfish_(Pomacanthus_semicirculatus).jpg` (SHA-256 `5622e415149a8cea8512e7fd1f416a600ca24cb528f054826667fd74b20b687e`) | a living *Pomacanthus semicirculatus* in lateral view over sand; the script segments its silhouette inside a declared box | Elias Levy (Flickr 6851425758), via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Semicircle_angelfish_(Pomacanthus_semicirculatus).jpg) | CC BY 2.0 |

The CSV is vendored because CC0 allows it and because Dryad's file downloads sit
behind a browser challenge that defeats scripted fetching; the script checks the
hash on every run. The scan and the two photographs are fetched on first run
(the jp2 inside the item's zip is byte-stable, unlike the JPEGs the page viewer
renders on the fly; the Commons originals are served byte-stable) and
hash-checked; `--no-scan` and `--no-photos` skip them. Wikimedia requires a
descriptive `User-Agent`, which the script sends.

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

## Real outlines (the photographs)

Thompson deformed an outline, not landmarks, so the script also carries a real
*Scarus* silhouette onto a real *Pomacanthus* silhouette. The *Scarus ghobban*
specimen is segmented by its Lab colour distance from the white background; the
living *P. semicirculatus* by a marker watershed inside a declared box (markers:
background outside the box; fish on the 1.5 % of pixels furthest in Mahalanobis
distance from the frame-sampled sand colour plus the most textured pixels in the
convex hull of that core, which is what reaches the dark caudal peduncle; the
shadow below 0.745 of the height is cleared). Each silhouette is registered to
its own species' 17 landmarks from the table by a similarity fixed by two anchors
the silhouette gives — the anterior-most point → LM2, and the caudal peduncle
(narrowest column between the caudal fin and the body, tracked from the tail tip)
→ the midpoint of LM5–LM6 — so the landmarks are placed on the photographs, not
digitised on them. The five maps are fitted to the landmarks of *S. ghobban* →
*P. semicirculatus* (species, not genus, means) and applied to the *Scarus*
silhouette; agreement with the *Pomacanthus* silhouette is the IoU of the filled
outlines and the mean symmetric nearest-point distance between the boundaries.
Quarter-size images; silhouettes of 214,862 and 92,699 pixels.

| map fitted to the 17 landmarks | landmark RMS | silhouette IoU | boundary distance (% of body length) |
|---|---|---|---|
| similarity (no deformation) | 0.0388 | **0.662** | 5.6 |
| affine | 0.0322 | 0.652 | 6.5 |
| Möbius | 0.0381 | 0.645 | 5.9 |
| oblique coaxial circles (Fig. 376) | 0.0303 | 0.618 | 7.0 |
| thin-plate spline (exact on the landmarks) | 0 | 0.629 | 6.6 |

Every deformation read off the landmarks leaves the outline no better than no
deformation at all — including the thin-plate spline, which matches the 17
landmarks exactly. What the landmarks ask of the body the fins do not follow.
Caveats: one individual per species; a collected specimen against a living fish,
so fin posture differs (the angelfish's soft dorsal and anal lobes trail behind
the peduncle, which no map of the parrotfish can produce); the field
segmentation depends on the declared box and the shadow cut.
