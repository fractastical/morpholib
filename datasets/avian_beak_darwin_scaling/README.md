# `datasets/avian_beak_darwin_scaling/` — Campàs et al. (2010) & Abzhanov et al. (2004, 2006) Darwin's Finch Affine Scaling-and-Shear Reference Package

Formal SHA-256-locked dataset and biological specimen image for `2021-alberti-avian-beak-darwin-scaling.py` (Slot 070, `alberti-avian-beak-darwin-scaling`).

## Contents

1. **`avian_beak_darwin_scaling_reference.json`**
   - **Campàs, Mallarino, Herrel, Abzhanov & Brenner (2010) *PNAS* 107(8):3356–3360 (`doi:10.1073/pnas.0911575107`)**:
     - 3-parameter affine scaling-and-shear group $\mathbf{M}(s_L, s_D, \gamma) = \begin{pmatrix} 1 & \gamma \\ 0 & 1 \end{pmatrix}\begin{pmatrix} s_L & 0 \\ 0 & s_D \end{pmatrix}$ collapsing all 6 *Geospiza* species onto a single Group A canonical curve under pure scaling $(s_L, s_D)$ and collapsing all 14 Darwin's finch species across Groups A, B (*Camarhynchus*, *Platyspiza*, *Cactospiza*), and C (*Certhidea*, *Pinaroloxias*) onto a single universal master beak curve after scaling + shear $(s_L, s_D, \gamma)$.
   - **Abzhanov, Protas, Grant, Grant & Tabin (2004) *Science* 305:1462–1465 (`doi:10.1126/science.1098095`) & Abzhanov, Kuo, Hartmann, Grant, Grant & Tabin (2006) *Nature* 442:563–567 (`doi:10.1038/nature04843`)**:
     - Orthogonal mesenchymal morphogen coordinates: Bmp4 controls beak depth $s_D$ and width $s_W$, whereas Calmodulin (CaM) controls beak length $s_L$, verified across `RCAS::Bmp4`, `RCAS::Noggin`, and `RCAS::CaM` retroviral perturbations.
   - SHA-256: `cf6430de0c52d725004a18fc5517900b0dba9b10af7935c03e4a534d0d40f10a`.

2. **`darwin_finches_specimen.jpg`**
   - John Gould's classic 1845 Darwin's finches lithograph (*Voyage of the Beagle* / Darwin 1845) depicting 1. *Geospiza magnirostris*, 2. *Geospiza fortis*, 3. *Camarhynchus parvulus* (*Geospiza parvula*), and 4. *Certhidea olivacea* ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Darwin%27s_finches_by_Gould.jpg), Public Domain).
   - SHA-256: `5ea7ada14a01ba77d532e9c254195c409ef669fabb5f9a3e5d3de1568621bdfa`.

3. **`scout_manifest.json`**
   - Step 1 literature & specimen scout dossier for Slot 070.
   - SHA-256: `4b321120fd7b273656d1f20e89aa9c5aeb47bfef844b04022679b66f24584ed0`.

4. **`paper_citations.bib`**
   - BibTeX entries for Campàs et al. (2010), Abzhanov et al. (2004), Abzhanov et al. (2006), and Fritz et al. (2014).
   - SHA-256: `f0b9fb929ffd2d09be0f276e6b197c32e913a051a97b3f84daf4fa199a2e7634`.
