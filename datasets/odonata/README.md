# Odonata wing data: fossil and extant

Inputs for [`2026-eshghi-odonata-wing-cells.py`](../../2026-eshghi-odonata-wing-cells.py).
Nothing here is hand-entered; every file was produced from the sources below on
2026-10-14 (see each file's provenance line).

## `eshghi2026_odonata_wing_cells.csv` (463 wings)

Per-wing morphometrics for 69 fossil Odonata wings (22 extinct species),
290 extant dragonfly wings (150 species) and 104 extant damselfly wings
(54 species): wing length (mm), number of segmented cells, and the mean /
median / SD / IQR / min / max cell length (mm) plus the fraction of cells in
the 0.5–1.5 mm band and below 0.5 mm.

Source: Supplementary Tables S1–S3 of

> Eshghi S, Buesch A, Gorb S (2026). Supplementary Material: "Critical Crack
> Length as an Evolutionary Constant in Odonata Wing Venation: Lessons from
> Fossils". Zenodo. https://doi.org/10.5281/zenodo.19899648 (CC BY 4.0)

Columns were renamed (`cell_len_median_mm` etc.) and species names
case-normalised; values are unchanged. Added columns:

- `family`, `family_source`: extant species matched with the GBIF
  species/match API (`https://api.gbif.org/v1/species/match?name=`), fossil
  species from the Paleobiology Database (below). One extant species
  (*Pruviogomphus poyobambus*) did not match and has an empty family.
- `max_ma`, `min_ma`, `age_ma`, `interval`, `formation` (fossils only): from
  `fossil_taxa_pbdb_ages.csv`; `age_ma` is the midpoint of the PBDB range.
- `image_name` (fossils only): file name of the venation drawing in the Zenodo
  archive (`Supplementary Images/Fossil Wing Images/<image_name>.png`). The
  images are *not* vendored here; the script downloads/caches the archive.

## `fossil_taxa_pbdb_ages.csv` (22 taxa)

For each fossil species in Table S1, the Paleobiology Database occurrence
query `data1.2/occs/list.json?base_name=<species>` (falling back to the genus
for *Cratopetalura petrulevicius*): number of occurrences, oldest `max_ma`,
youngest `min_ma`, family, interval and formation as recorded in PBDB.
Resulting ages: Solnhofen (Early Tithonian, 149.2–145.1 Ma), Yixian (Late
Barremian, 125.8–121.4 Ma), Crato (Late Aptian, 119.6–113.2 Ma; 18 species),
Ypresian (56.0–48.1 Ma; 3 species).

## `pbdb_odonatoptera_occurrences.csv` (1762 occurrences)

`https://paleobiodb.org/data1.2/occs/list.csv?base_name=Odonatoptera&show=class,coords&idreso=genus`,
reduced to the taxonomic and age columns. Used only for the genus-richness
backdrop of the deep-time animation (Carboniferous griffinflies such as
*Meganeuropsis permiana* to Holocene). PBDB data are CC BY 4.0; see
https://paleobiodb.org/#/faq for citation guidance.

## `fossil_wing_resegmentation.csv` (69 wings)

Produced by the script itself (`resegment_fossils()`), not by the authors: every
fossil venation drawing in the Zenodo archive re-segmented from the pixels. The
vein ink is skeletonised, the regions it encloses are the cells (all drawn
fragments are counted), and the scale is read from each drawing's own scale
bar (bar found in 68/69 drawings, its label read in 65; `scale_source` says
how each wing was calibrated — for #20 there is no bar and the tabulated wing
length is used, for #5, #60 and #61 the label could not be read reliably and
the bar is assigned the value consistent with the tabulated wing length).
Per wing: `n_cells`, median / mean / max cell length (maximum Feret diameter,
mm), median cell area (mm²), median circularity and the fraction of cells in
the 0.5–1.5 mm band, each beside the corresponding `table_*` value from Table S1.

Agreement with Table S1 (2026-10-14 run): median |ΔN|/N = 0.0 %, 90 % of wings
within 2 % in cell count; median cell length within 3.4 % (ours systematically
~3 % longer, as cells are bounded by vein mid-lines); largest cell within 0.9 %.
Discrepancies worth knowing about:

- **#7 *Euarchistigma peterknobli* (Right Wing)**: the drawing (10 mm bar) is a
  61 mm wing with 829 cells, median 1.15 mm; Table S1 lists 24.7 mm, 282 cells,
  median 0.76 mm. The drawing and the table row do not describe the same wing.
- **#23 / #24 *Paracordulagomphus aberrans* 1**: our values for #23 match the
  table's #24 and vice versa (530 vs 484 cells); the two rows appear swapped
  relative to the image names.
- *Cratostenophlebia schwickerti* (#8–#11, ~3000 cells each): we count 6–10 %
  more cells than the table; the skeleton of very dense venation splits a few
  cells the authors merged.
- Six *Euarchistigma* drawings are fragments or several specimens share one
  tabulated wing length (28.8 mm), so `drawing_feret_mm` differs from
  `table_wing_length_mm` there; the cell measurements are unaffected because
  the scale comes from the bar.

A draft message to the authors about #7 and #23/#24 is in
[`NOTE_TO_AUTHORS.md`](./NOTE_TO_AUTHORS.md) (not yet sent).

## Also used at runtime (downloaded to the git-ignored `.wing_cache/`, SHA-256 verified)

- Fossil wing drawings from the Zenodo archive above (CC BY 4.0);
  `ac1288ba…034703`, 89,961,457 bytes.
- Hoffmann J, Donoughe S, Li K, Salcedo MK, Rycroft CH (2018). A simple
  developmental model recapitulates complex insect wing venation patterns.
  *PNAS* 115(40):9905–9910. https://doi.org/10.1073/pnas.1721248115 — segmented
  North American dragonfly wings (`Dragonfly_NA_WingSegmentation.tar.gz` in
  https://github.com/hoffmannjordan/insect-wing-venation-patterns;
  `50997be3…ccb232`, 10,246,854 bytes), used as the extant venation image next
  to the Eocene Aeshnidae / Gomphidae fossils.
