# T1917-054 — Thompson (1917) Ch. XVII, Figs. 388–393: Archaeopteryx → Apatornis pelvis by grid interpolation

Slot 243 (`thompson1917-t1917-054`) of the Thompson-1917 claim-validation campaign.
Script: `thompson/claims/t1917-054_on-the-theory-of-transformations.py`.
Lean: `morphoeval/lean/loops/Thompson1917_T1917_054.lean`. KG law file: `morphoeval/math_kg_loops/thompson1917-t1917-054.yaml`.

## The claim (as registered in `thompson/thompson_1917_claims.json`)

> The pelves of Archaeopteryx and Apatornis (after Heilmann) and intermediate configurations can be
> found by interpolation between coordinate-transformed forms — i.e. interpolated grids generate
> biologically real intermediate forms.

What Thompson actually prints (p. 759) is weaker: three *equidistant* coordinate systems are
interpolated between the Archaeopteryx grid (Fig. 388, Cartesian) and the Apatornis grid (Fig. 389),
a pelvis outline is inscribed in each (Figs. 391–392), and the series is shown in Fig. 393 as
"three intermediate **hypothetical** types". The loop therefore splits the claim:

| part | statement | testable here |
|---|---|---|
| A (methodological) | interpolated grids yield well-defined, equidistant intermediate outlines | yes — from Fig. 393 itself |
| B (biological reality) | those intermediates match real intermediate Mesozoic bird pelves | no — no landmark data of intermediate taxa is pinned |

## Files

| file | what | source | licence | SHA-256 |
|---|---|---|---|---|
| `ongrowthform1917thom_p757_leaf0776_figs388-389.jpg` | p. 757: Fig. 388 Archaeopteryx pelvis in Cartesian grid; Fig. 389 Apatornis pelvis in transformed grid | IA `ongrowthform1917thom`, page image n776 (w1200) | public domain | `5214f007…ddc454` |
| `ongrowthform1917thom_p758_leaf0777_figs390-391.jpg` | p. 758: Fig. 390 the five superposed grid outlines; Fig. 391 first intermediate | same, n777 | public domain | `3b7d8054…7ec7a` |
| `ongrowthform1917thom_p759_leaf0778_fig392.jpg` | p. 759: Fig. 392 second and third intermediates | same, n778 | public domain | `17283e61…ebf7` |
| `ongrowthform1917thom_p760_leaf0779_fig393.jpg` | p. 760: Fig. 393 the five wash silhouettes (the quantitative surface) | same, n779 | public domain | `0e73dc9e…0023` |
| `archaeopteryx_lithographica_berlin_specimen_mfn.jpg` | real *Archaeopteryx* fossil, Berlin specimen (downscaled derivative) | [Commons](https://commons.wikimedia.org/wiki/File:Archaeopteryx_lithographica_(Berlin_specimen).jpg), H. Raab | CC BY-SA 3.0 | `96341aed…4b2c` |
| `apatornis_celer_holotype_synsacrum_marsh1880.jpg` | *Apatornis celer* holotype synsacrum, Marsh 1880 pl. | [Commons](https://commons.wikimedia.org/wiki/File:Apatornis.jpg) via BHL | public domain | `28213a31…a392` |
| `dryad_k1v34cf_deeming_mayr_2018_metadata.json` | Dryad API metadata of the pre-scouted dataset | `https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.k1v34cf` | CC0 | `4f2b721e…528c` |
| `scout_manifest.json` | Step 1 dossier (verification of the pre-scouted dataset, IA pins, Commons pins, OpenAlex hits) | this loop | CC BY 4.0 | — |
| `t1917_054_on_the_theory_of_transformatio_reference.json` | equations E1–E7, state variables, boundary conditions, parameter regime, empirical claims C1–C4, all digests | this loop | CC BY 4.0 | — |
| `paper_citations.bib` | references | this loop | CC BY 4.0 | — |
| `result.json` | claim verdict (written by the script) | this loop | CC BY 4.0 | — |

Full digests are in `t1917_054_on_the_theory_of_transformatio_reference.json` → `inputs_sha256` and are
re-verified by the script on every run. The IA page JPEGs are rendered on demand and are not
byte-stable across fetches, which is why they are vendored rather than fetched at run time.

## Pre-scouted dataset: `doi:10.5061/dryad.k1v34cf` — DATASET_MISMATCH (labelled analogue)

Deeming & Mayr (2018, *J. Evol. Biol.* 31:701–709) give the pelvic-canal width at the pubic symphysis,
estimated egg mass and body mass for 21 Mesozoic bird species. These are 1-D scalars; the claim is about
2-D pelvis outline shape and grid interpolation. The single supplementary PDF is also not retrievable by
script (API download needs an OAuth bearer token; the web route sits behind a browser proof-of-work
challenge, which this loop does not circumvent). Only the API metadata is pinned. The dataset is
therefore **not used for the verdict**; the verdict rests on Thompson's own figures, which the claim
override explicitly allows.

## Method (part A)

1. Segment the five silhouettes of Fig. 393 (Otsu → opening → connected components → five largest,
   ordered top-to-bottom).
2. Landmarks: acetabulum (largest interior hole) + the four process tips (anterior/posterior ilium,
   ischium, pubis) as radial-distance peaks of the boundary; arc-length semilandmarks between tips.
3. Fit the Archaeopteryx → Apatornis transformation φ (thin-plate spline; affine-only as the
   "simple transformation" control) and generate φ_t = (1−t)·id + t·φ at t = ¼, ½, ¾ (E2–E3).
4. Compare each predicted form with Heilmann's drawn intermediate (Procrustes distance, IoU, boundary
   RMSE, best-fit t*), against endpoint-copy and shuffled-t nulls, across a sweep of semilandmark
   density and TPS regularisation.

## Reproduce

```bash
python3 thompson/claims/t1917-054_on-the-theory-of-transformations.py --self-test
python3 thompson/claims/t1917-054_on-the-theory-of-transformations.py
```

Outputs: `docs/media/t1917_054_on_the_theory_of_transformatio_comparison.png`,
`docs/media/t1917_054_on_the_theory_of_transformatio_growth.gif`, and
`datasets/thompson_1917/t1917-054/result.json`.
