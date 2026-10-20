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
