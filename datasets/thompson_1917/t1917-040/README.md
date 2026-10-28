# T1917-040 — Dürer/Camper facial grids (Thompson 1917, Ch. XVII)

Loop `thompson1917-t1917-040` (slot 232) validates one dated claim:

> Human (and animal) heads/faces and proportions, in the Dürer/Camper tradition, are mutually
> transformable by coordinate deformation; differences of facial angle/proportion are whole-grid
> deformations. — Thompson (1917) *On Growth and Form*, Ch. XVII.

## Where the claim actually sits in the 1917 text

The claim inventory cites pp. ~724-735 / Figs. 353-358. In the Internet Archive scan
(`ongrowthform1917thom`, printed page = leaf − 19 here) those pages carry the four generic
transformation types; the human-head material is a few pages later:

| printed page | leaf | content |
|---|---|---|
| 729-731 | 748-750 | types (1) proportional stretch `x1, c·y1` (Fig. 353), (2) logarithmic `y → ε^y` (Figs. 355-356), (3) simple shear `x = ξ − η cot ω, y = η cosec ω` (Fig. 357), (4) radial (Fig. 358) |
| 740 | 759 | **Fig. 364 (after Dürer)** — three heads in rectangular proportion grids |
| 742 | 761 | **Fig. 366 (after Dürer)** — two heads in oblique (sheared) grids; footnote: Camper's facial angle "is the index to a general deformation" |

The vendored page images are in `thompson_1917_pages/` and SHA-256 pinned in the reference JSON.

## Dataset decision

* Pre-scouted Dryad pointer `doi:10.5061/dryad.0zpc86746` is a **cichlid fish** geometric-morphometric
  dataset → `DATASET_MISMATCH` for this human-profile claim; recorded in `scout_manifest.json`, not used.
* Primary claim source: Thompson's own Dürer figures (above).
* Biological specimens: nine CC0 / CC BY-SA / public-domain lateral human head photographs from
  Wikimedia Commons (`specimens/`), chosen for an unoccluded forehead-to-chin profile against a plain
  background. Two of them (`kubra_gumusay_2023_profile.jpg`, `beverly_sills_profile.jpg`) are kept as
  deliberate segmentation stress tests and are expected to be rejected by the QC gate. Licences,
  authors, source URLs and SHA-256 digests: `t1917_040_on_the_theory_of_transformatio_reference.json`.

## Files

| file | role |
|---|---|
| `scout_manifest.json` | Step 1 dossier: OpenAlex hits, Commons candidates (kept/rejected), Dryad mismatch, IA pages, collision check |
| `t1917_040_on_the_theory_of_transformatio_reference.json` | equations E1-E5, state variables, parameter regime, claims C1-C3, pinned SHA-256 digests, per-specimen processing annotations (ROI, deskew, facing) |
| `paper_citations.bib` | Thompson 1917, Dürer 1528, Camper 1794, Legan & Burstone 1980, Ricketts 1968, de Jong et al. 2014 |
| `specimens/*.jpg|png` | vendored Commons photographs (≤ 960 px wide derivatives of the originals) |
| `thompson_1917_pages/*.jpg` | IA page scans (1200 px wide) |
| `result.json` | verdict written by the script (claim_id, dataset, verdict, headline numbers) |
| `t1917_040_landmarks.csv`, `t1917_040_pairwise_fits.csv`, `t1917_040_metrics.json` | script outputs (Pass 2) |

## Reproduce

```bash
python3 thompson/claims/t1917-040_on-the-theory-of-transformations.py --self-test
python3 thompson/claims/t1917-040_on-the-theory-of-transformations.py
```

Outputs go to `docs/media/t1917_040_on_the_theory_of_transformatio_comparison.png`,
`docs/media/t1917_040_on_the_theory_of_transformatio_growth.gif` and this directory.

## Ethics note

Camper's facial-angle series was later misused for racial ranking. This loop tests only the
*geometric* content of Thompson's claim (is the difference between two individual profiles a single
whole-grid deformation?). No population-level inference is made or implied.
