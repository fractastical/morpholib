# T1917-051 — Thompson (1917) crocodylian skull transformations

Validates Thompson, D.W. (1917) *On Growth and Form*, Ch. XVII, pp. 752–753, Fig. 383:

> The skulls of crocodiles (*C. porosus*, *C. americanus*, *Notosuchus terrestris*) are mutually transformable by coordinate deformation.

## Primary dataset

- **Dryad** [10.5061/dryad.14fn1](https://doi.org/10.5061/dryad.14fn1) — Watanabe & Slice (2014), 78 3D landmarks, 204 crocodylian crania.
- Vendored: `dryad_14fn1_v3888.zip`, `WatanabeSlice_JEB_PreGPA.nts` (SHA-256 pinned in `t1917_051_on_the_theory_of_transformatio_reference.json`).

## Taxon overlap with Thompson's three names

| Thompson 1917 name | Dryad coverage |
|--------------------|----------------|
| *C. porosus* | Yes (`cropor`) |
| *C. americanus* | Ambiguous; test uses *C. acutus* (`croacu`) and *A. mississippiensis* (`allmis`) as extant analogues |
| *Notosuchus terrestris* | No (extinct; not in extant ontogeny set) |

## Specimen photograph

- Wikimedia Commons: *Alligator mississippiensis* skull (James St. John, CC BY 2.0), `alligator_mississippiensis_skull.jpg`.

## Script

`thompson/claims/t1917-051_on-the-theory-of-transformations.py` — warp metrics, figures, `result.json`.
