# T1917-024 — Thompson (1917) Ch. IX: sponge spicule form as Plateau-border deposition

Loop `thompson1917-t1917-024` (campaign slot 219) validates one dated claim from
D'Arcy Thompson, *On Growth and Form* (1917), Ch. IX "On Concretions, Spicules, and
Spicular Skeletons", pp. 411-487, Figs. 194-235:

> Sponge and alcyonarian spicule forms (needles, tri-/tetraradiates and their aggregates)
> are the forms physical deposition/crystallisation takes within the cell/field that
> secretes them.

Thompson's own operational version (pp. 443-449, Fig. 214, after Haeckel) is that a deposit
accumulating in the partitions of three contiguous cells takes a triradiate form with rays
at **120°** (Plateau junction); that **any inequality of specific tensions** between the
cells breaks the equiangular condition into a spicule with **two equal angles and one
unequal** (the sagittal form, grading to a T with rays "nearly in a straight line, and at
right angles to the third"); and that four cells in solid contact give a tetractine at the
**tetrahedral angle (~109°)**.

## Model formalised here

The Neumann tension triangle at a triple junction (`E1`-`E3` in the reference JSON):

- `gamma_1 / sin(theta_1) = gamma_2 / sin(theta_2) = gamma_3 / sin(theta_3)`, `sum theta = 360°`
- equal tensions -> 120°, 120°, 120° (Thompson P1)
- sagittal one-parameter family with `rho = gamma_cc / gamma_cm`:
  `cos(theta_pair) = -rho/2`, `theta_unpaired = 360° - 2 theta_pair = 2 arccos(rho/2)` (P2);
  limits `rho -> 0` gives (90°, 90°, 180°), `rho = 1` gives 120°
- tetrahedral `cos theta = -1/3` (109.47°) for four rays; orthogonal 90° for six (P3 and the hexactine limit)

Lean 4 proofs of the equiangular point, the "equiangular iff rho = 1" statement, the
monotone sagittal limit and the tetrahedral/orthogonal ray systems are in
`morphoeval/lean/loops/Thompson1917_T1917_024.lean`.

## Files

| file | what | licence | SHA-256 |
|---|---|---|---|
| `specimens/grantia_indica_zootaxa4426_fig53_sem.jpg` | **primary specimen**: SEM plate of *Grantia aff. indica* spicules (van Soest & de Voogd 2018, Zootaxa 4426, Fig. 53; triactines c, d; tetractines e, f) | CC BY 3.0 | `70194abe…bad59` |
| `specimens/grantia_spicules_readey_100x.jpg` | secondary population: brightfield spread of *Grantia* spicules, 100x (M. Readey 2020) | CC BY-SA 4.0 | `08641621…3f30` |
| `specimens/thompson1917_fig214_ia_leaf465_p445.jpg` | Thompson's own p. 445 with Fig. 214 (IA scan `ongrowthform1917thom`, leaf 465) | public domain | `019168c1…7988` |
| `thompson1917_ch9_pp443-449_ocr_excerpt.txt` | uncorrected IA OCR of the passage used | public domain | `52af6bf9…b741` |
| `dryad_8mv00jv_dataset_metadata.json`, `dryad_8mv00jv_files.json` | Dryad API metadata of the pre-assigned dataset (payload **not** fetched) | CC0 | see reference JSON |
| `scout_manifest.json` | Step 1 dossier: dataset verification, Commons/OpenAlex hits, collision check | — | — |
| `t1917_024_on_concretions_spicules_and_sp_reference.json` | equations, state variables, parameter regime, claims, pinned digests | — | — |
| `paper_citations.bib` | all sources (DOIs Crossref-verified 2026-10-28) | — | — |
| `result.json` | verdict record written by the loop | — | — |
| `measurements/` | per-spicule ray-angle tables written by the script | — | — |

Full digests are in `t1917_024_on_concretions_spicules_and_sp_reference.json`; the script
re-verifies the specimen digests at run time.

## Pre-assigned dataset: `doi:10.5061/dryad.8mv00jv` — DATASET_MISMATCH for the primary test

`thompson/thompson_1917_dataset_audit.json` pre-scouted this slot as PUBLIC_SCRIPTABLE on the
strength of the Dryad record *Three-dimensionally preserved soft-tissues and calcareous
hexactins in a Silurian sponge* (Nadhira et al. 2019). It was verified, not fitted:

- it is a **fossil reticulosan** (*Carduispongia pedicula*) with **hexactine** (six-rayed,
  orthogonal) spicules; Thompson's quantitative predictions concern **triradiate / tetraxon**
  spicules of living calcareous sponges and he explicitly brackets the hexactinellid case as
  "harder to understand";
- it holds **5.83 GB of tomogram slices and VAXML/STL meshes**, no ray-angle table;
- its metadata and per-file Dryad MD5 digests are pinned here; the payload was not fetched.

The override for Thompson claims allows Thompson's **own 1917 figures** from the pinned IA
scan as the dataset, so Fig. 214 (p. 445, leaf 465) is pinned and digitised, and the test is
run on a real CC BY SEM plate of the *same genus* (*Grantia*). The 90° hexactine network of the
Dryad specimen is cited only as the six-ray limiting case, with no verdict weight.

## Script

```bash
python3 thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py --self-test
python3 thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py
```

Outputs: `docs/media/t1917_024_on_concretions_spicules_and_sp_comparison.png`,
`docs/media/t1917_024_on_concretions_spicules_and_sp_growth.gif`,
`datasets/thompson_1917/t1917-024/measurements/*.json|csv`, and `result.json`.
