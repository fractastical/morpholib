# T1917-019 — "Tissue cells tessellate approximately as hexagons / as a froth"

Thompson, D'Arcy W. (1917) *On Growth and Form*, Ch. VII "The Forms of Tissues, or
Cell-Aggregates", pp. 293–345, Figs. 120–127 (pp. 317–325). Slot 214 of the
Thompson-1917 claim-validation campaign (`thompson1917-t1917-019`).

## The claim, as Thompson actually states it

| page | what Thompson says |
|---|---|
| 305 | "the general resemblance between a cellular tissue and a 'froth'" |
| 309 | least partition area is attained only when walls "meet together in groups of three ... at angles of 120°" (Lamarle) |
| 321 | "The hexagonal pattern is illustrated among organisms in countless cases, but those in which the pattern is perfectly regular ... are not so numerous. The hexagonal epithelium-cells of the pigment layer of the eye ... are a good example." |
| 323 | "only three partitions (in surface view) meet in a point ... at or nearly at the normal angle of 120°" |
| 325 | "the cells will be on the average hexagonal, but some will have fewer and some more sides than six" |
| 330 | "the closely packed hexagons gave the minimal extent of boundary in a plane" |

The claim therefore decomposes into three testable parts (see
`t1917_019_the_forms_of_tissues_or_cell_a_reference.json` → `claim_decomposition`):

* **C1 froth / mean six** — trivalent network, ⟨n⟩ = 6 − 6χ/F, a spread of 5- and 7-gons.
* **C2 least perimeter** — the honeycomb is the least-perimeter equal-area partition
  (Hales 2001); a tissue should sit close to that bound for its own cell-area distribution.
* **C3 "approximately hexagons"** — read strictly, most cells are 6-gons.

## Files

| file | role | SHA-256 |
|---|---|---|
| `human_corneal_endothelium_specimen.jpg` | **primary specimen** — human corneal endothelium in situ, alizarin-red cell outlines (Zhiguo He, Wikimedia Commons, CC BY-SA 4.0), 1290×1031, byte-identical to Commons original (SHA-1 `e0abd0d9…`) | `9daed5fb…a7a7b1` |
| `elodea_canadensis_leaf_specimen.jpg` | secondary specimen — *Elodea canadensis* leaf, 40×, the species of Thompson's Fig. 126 (Daemon Canchig, Commons, CC BY-SA 4.0); vendored as 1600-px JPEG | `ffa66c5d…` |
| `thompson_1917_p321_fig124_girardia.jpg` | IA leaf n340 = p. 321, Fig. 124 *Epidermis of Girardia* (after Goebel) | `63e3666f…` |
| `thompson_1917_p322_fig125_126_froth_elodea.jpg` | IA leaf n341 = p. 322, Fig. 125 soap-froth under pressure (after Rhumbler, A compressed / B relaxed) and Fig. 126 *Elodea canadensis* (after Berthold) | `1a31755f…` |
| `thompson_1917_p325_fig127_128_corals.jpg` | IA leaf n344 = p. 325, Figs. 127–128 *Lithostrotion* / *Cyathophyllum* | `059ccb34…` |
| `scout_manifest.json` | Step-1 verification dossier (IA scan, Commons metadata, OpenAlex hits, collision check, dataset-match assessment) | — |
| `t1917_019_the_forms_of_tissues_or_cell_a_reference.json` | equations E1–E6, state variables, boundary conditions, parameter regime, literature reference values, pinned digests | — |
| `paper_citations.bib` | Crossref-verified DOIs | — |
| `result.json` | the verdict (written by the loop script) | — |

Full digests are in `t1917_019_the_forms_of_tissues_or_cell_a_reference.json` →
`input_dataset_sha256`.

## Dataset-match statement (Step 1 override rule)

Thompson's Figs. 120–127 are line drawings: Fig. 124 is an idealised schematic and
Fig. 126 shows only eight cells, so the 1917 plates alone cannot yield a polygon-class
distribution. Fig. 125 (Rhumbler's soap froth, ~100 cells) is the physical froth the
claim invokes and is digitised by the script. Because Thompson names "the hexagonal
epithelium-cells ... of the eye" (p. 321) as his exemplar and makes the claim for
tissue cells in general, a human corneal endothelial monolayer is a **direct instance**
of the claim, not an analogue. Status: **MATCH** (no `DATASET_MISMATCH`).

## How the specimen is scored

```bash
python3 thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py --self-test
python3 thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py
```

The script (i) segments the alizarin-red outlines (dark ridges → watershed basins),
(ii) counts neighbours of every interior cell (= polygon class), (iii) computes
⟨n⟩, μ₂, p₆, the junction-valence fraction, Lewis and Aboav–Weaire fits, and the excess
perimeter relative to the isoperimetric honeycomb bound for the measured cell areas,
(iv) compares against a periodic Voronoi/Lloyd froth model swept over disorder δ, and
(v) writes `result.json` with the verdict.

Reference norms for a normal adult corneal endothelium (Sheng & Bullimore 2007,
Cornea 26:520): %hexagonal ≈ 62 % (range 55–70 %), CV of cell area ≈ 0.32.
Proliferating epithelia (Gibson et al. 2006): p₆ ≈ 0.46, p₅ ≈ 0.29, p₇ ≈ 0.21.
