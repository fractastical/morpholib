# T1917-058 — Thompson (1917) Ch. XVII, pp. 770–773, Figs. 404–408

**Claim (dated 1917):** *"The human skull, inscribed in Cartesian coordinates and deformed,
gives the skulls of chimpanzee, baboon and dog: ape/human skull difference is a
whole-diagram transformation (notably of the facial/cranial proportions)."*

Loop `thompson1917-t1917-058` (slot 247) validates this one dated claim. It is **not** a
new-model scout. Verdict file: `result.json` (written by the script).

## What is in this directory

| file | role | licence |
|---|---|---|
| `thompson_1917_leaf789_p770_figs404_405.jpg` | IA scan leaf 789 = printed p. 770: Fig. 404 (human skull in a 5×4 Cartesian net) and Fig. 405 (chimpanzee net) | public domain |
| `thompson_1917_leaf790_p771_figs406_407.jpg` | leaf 790 = p. 771: Fig. 406 (chimpanzee skull in its net), Fig. 407 (baboon) | public domain |
| `thompson_1917_leaf792_p773_fig408.jpg` | leaf 792 = p. 773: Fig. 408 (dog) | public domain |
| `fig404_…` … `fig408_…jpg` | full-resolution crops of the five figures | public domain |
| `homo_sapiens_lateral.jpg` | Wikimedia Commons *Human Skull Lateral View Unlabeled* (rendered anatomical model) | CC BY-SA 3.0, Andrewmeyerson |
| `pan_troglodytes_lateral.jpg` | Commons *Pan troglodytes 02 MWNH 230* (Museum Wiesbaden photo) | CC BY-SA 3.0, K. Rassinger & G. Cammerer |
| `papio_ursinus_lateral.jpg` | Commons *Papio ursinus 02 MWNH 715b* | CC BY-SA 3.0, K. Rassinger & G. Cammerer |
| `canis_familiaris_lateral.jpg` | Commons *Canis lupus familiaris 02 MWNH 788* (faces left; mirrored in analysis) | CC BY-SA 3.0, K. Rassinger & G. Cammerer |
| `dryad_7r85116_metadata.json` | pinned Dryad record of the dataset named in the pre-scout (New World monkeys → **DATASET_MISMATCH**, not used) | CC0 metadata |
| `scout_manifest.json` | Step 1 dossier: dataset verification, Commons/OpenAlex/Dryad provenance, collision check | — |
| `t1917_058_on_the_theory_of_transformatio_reference.json` | Step 2/3: equations E1–E7, parameter regime, the digitised Fig. 405 net (30 nodes), 11-landmark protocol and coordinates for 6 configurations, SHA-256 of every vendored file | — |
| `paper_citations.bib` | citations | — |
| `result.json` | verdict (claim_id, dataset, verdict, headline numbers, script, figure) | — |

## Dataset verification (T1917 override)

The pre-scout named `doi:10.5061/dryad.7r85116` (Aristide et al. 2018, Evolution). Its
record was fetched and pinned; it contains 3D landmarks of **49 platyrrhine species** and none
of the four taxa in the claim (human, chimpanzee, baboon, dog). It is therefore recorded as
`DATASET_MISMATCH` and is **not** fitted as an analogue. The claim is tested instead on the
two sources the override permits/implies:

* **A. Thompson's own 1917 figures** (exact species; public-domain IA scan, leaf = page + 20).
  Fig. 405's chimpanzee net was digitised node by node (skeleton-junction detection plus
  manual corners). The baboon (Fig. 407) and dog (Fig. 408) nets are drawn over heavy skull
  outlines and could not be extracted reliably; those taxa are tested on real specimens only.
* **B. Real lateral-view specimens of the exact taxa** from Wikimedia Commons (CC BY-SA 3.0),
  with 11 homologous landmarks digitised by the loop agent (uncertainty ≈ 12 px on the
  1280-px renditions, propagated by a jitter bootstrap in the script).

## How the claim is operationalised

Thompson's "deformed Cartesian net" is a smooth planar map `T`. We reconstruct `T` as the
thin-plate spline through the 30 digitised nodes of Fig. 405 (E1), decompose it into affine
("box stretch") and bending parts (E2–E3), measure the cell-area gradient Thompson calls
"approximately logarithmic" (E4), and — the actual test — apply `T` to the human landmarks of
Fig. 404 and ask whether the result is closer to a **real** chimpanzee skull than the human
configuration was (E5–E6). Baboon/dog/chimp ordering, "same order, greater intensity" for the
baboon, "different lines" for the dog, and the facial/cranial proportion statements are tested
on the real specimens (E7).

## Reproduce

```bash
python3 thompson/claims/t1917-058_on-the-theory-of-transformations.py --self-test
python3 thompson/claims/t1917-058_on-the-theory-of-transformations.py
```

Outputs: `docs/media/t1917_058_on_the_theory_of_transformatio_comparison.png`,
`docs/media/t1917_058_on_the_theory_of_transformatio_growth.gif`,
`datasets/thompson_1917/t1917-058/t1917_058_metrics.json`, `result.json`.
