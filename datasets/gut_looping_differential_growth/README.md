# `datasets/gut_looping_differential_growth/` — Savin et al. (2011) & Nerurkar et al. (2017) Gut Looping Reference Package

Formal SHA-256-locked dataset and biological specimen image for `2011-savin-gut-looping-differential-growth.py` (Slot 062, `savin-gut-looping-differential-growth`).

## Contents

1. **`gut_looping_differential_growth_reference.json`**
   - **Savin, Kurpios, Shyer, Florescu, Liang, Mahadevan & Tabin (2011) *Nature* 476:57–62 (`doi:10.1038/nature10277`)**:
     - Differential growth strain $\varepsilon = (L_t - L_m)/L_m$ between the intestinal tube (rest length $L_t$, outer/inner radii $R_t, R_i$, moment of inertia $I_t = \frac{\pi}{4}(R_t^4 - R_i^4)$, Young's modulus $E_t$) and the dorsal mesentery sheet (rest length $L_m$, thickness $h_m$, Young's modulus $E_m$).
     - Characteristic elasto-geometric length scale $\ell_* = (E_t I_t / (E_m h_m))^{1/3}$ and universal scaling laws $\lambda = c_\lambda \ell_*$, $R = c_R \ell_*$ ($c_\lambda = 4.55$, $c_R = 1.68$, $\lambda/R \approx 2.71$).
     - Ontogenetic developmental series for Chick (*Gallus gallus*, E8–E16) and 14-point cross-species comparative table spanning Zebra finch (*Taeniopygia guttata*), Quail (*Coturnix japonica*), Chick (*Gallus gallus*), and Mouse (*Mus musculus*).
   - **Nerurkar, Mahadevan & Tabin (2017) *PNAS* 114(9):2277–2282 (`doi:10.1073/pnas.1618677114`)**:
     - BMP2 gain-of-function (`RCAS-BMP2`), BMP antagonist (`RCAS-Noggin`), and surgical mesentery ablation ($\varepsilon = 0$) controls at E12.

2. **`intestine_mesentery_specimen.jpg`**
   - Vertebrate small intestine loops attached to the fan-shaped dorsal mesentery extending from the duodenojejunal flexure to the ileocecal junction ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Mesentery_extending_from_the_duodenojejunal_flexure_to_the_ileocecal_junction..jpg), CC BY-SA 4.0).
   - SHA-256: `e12fee9ac1fb5f3df3959f3f4f21027c93db879b40952f06bd19eb49450a212b`.

3. **`scout_manifest.json`**
   - Step 1 literature & specimen scout dossier for Slot 062.
   - SHA-256: `0723b42b4d82508767d2add63d9a11cd739b8a06ba0d779bf7e2cafb232b539f`.

4. **`paper_citations.bib`**
   - BibTeX entries for Savin et al. (2011), Nerurkar et al. (2017), Kurpios et al. (2008), and Shyer et al. (2013).
   - SHA-256: `21703c48df51dd52720d4c5c1a4a62c34331dd34691085f970737de8ff32e206`.

## Pass 3 additions (v3, 2026-10-19)

5. **`fe_mesentery_mesh_reference.json`**
   - Mesh, material and Pitx2-gradient **assumptions** of the Pass 3 reduced model in which the gut rod shares its nodes with the free edge of a 2D constant-strain-triangle (St.Venant–Kirchhoff) membrane mesh of the fan-shaped dorsal mesentery: `n_s = 96`, `n_w = 5` (760 triangles, Euler characteristic 1), `W_m / L_m = 0.18`, root fraction 0.7, `ν_m = 0.45`, apicobasal Pitx2 field $E_m(\eta) = E_m[1 + a(2\eta - 1)]$ with default $a = 0.4$ (swept over 0–0.8).
   - `comparison_targets` re-express the pinned Savin et al. (2011) observations as contour length per loop $L_t / N_{obs}$ and loop radius $R_{obs}$ for the five chick stages; nothing in this file is fitted to simulation output.
   - **None of the `mesh_defaults` values are measurements** — Savin et al. (2011) do not tabulate the root-to-gut mesentery width, the fan geometry or a Pitx2 stiffness amplitude. They are flagged as modelling assumptions in the file and in the script header.
   - SHA-256: `8025ecf7fd54a0ef5fd939431b10976751aa9d358f90227ebcd20fa300411cba`.

6. **`paper_citations.bib`** now also contains Davis et al. (2008) *Dev. Cell* 15:134–145 (`doi:10.1016/j.devcel.2008.05.001`), the left–right asymmetric dorsal-mesentery architecture paper that motivates the residual “handedness is not selected by a bending-free membrane” item. Updated SHA-256: `201f4db588717f47dcc3be54621e01b2e1e42891e25c9a3a14c14fb20f7963a8`.
