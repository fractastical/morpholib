# `datasets/cortical_gyrification_folding/` — Richman et al. (1975) & Tallinen et al. (2014, 2016) Cortical Gyrification Reference Package

Formal SHA-256-locked dataset and biological specimen image for `1975-richman-cortical-gyrification-buckling.py` (Slot 061, `richman-cortical-gyrification-buckling`).

## Contents

1. **`cortical_gyrification_folding_reference.json`**
   - **Richman, Stewart, Hutchinson & Caviness (1975) *Science* 189(4196):18–21 (`doi:10.1126/science.1145181`)**:
     - Differential tangential cortical growth and mechanical buckling of the cerebral cortex of thickness $h$ and shear modulus $\mu_c$ bonded to a subcortical white-matter foundation of shear modulus $\mu_s$, predicting folding wavelength $\lambda = 2\pi h (\mu_c / (3\mu_s))^{1/3}$, and contrasting normogyric human cortex ($h = 2.60\text{ mm}$) against polymicrogyria ($h = 0.85\text{ mm}$) and lissencephaly ($h = 6.50\text{ mm}$, $g < g_c$).
   - **Tallinen, Chung, Biggins & Mahadevan (2014) *PNAS* 111(35):12667–12672 (`doi:10.1073/pnas.1406015111`) & Tallinen et al. (2016) *Nature Physics* 12(6):588–593 (`doi:10.1038/nphys3632`)**:
     - Multiplicative growth decomposition $\mathbf{F} = \mathbf{F}_e \mathbf{G}$ of a growing neo-Hookean cortical shell ($g \ge 1$) constrained by a soft white-matter core ($\mu_c/\mu_s \approx 1$), critical tangential expansion threshold $g_c = 1.18$, sulcal creasing instability, and human fetal ontogenetic series across gestational weeks GW22–GW38.
   - **Mota & Herculano-Houzel (2015) *Science* 349:74–77 (`doi:10.1126/science.aaa9101`) & Zilles et al. (1988) *Anatomy and Embryology* 179:173–179 (`doi:10.1007/BF00304699`)**:
     - 11-species mammalian comparative gyrification index ($\text{GI} = P_{\text{pial}} / P_{\text{hull}}$) and sulcal wavelength table spanning Rodentia, Carnivora, Artiodactyla, Cetacea, and Primates.
   - SHA-256: `94947373ac49d97dd01ff66ce60c00f5ce75b24f616bb8b141e01339ad65a48f`.

2. **`cerebral_cortex_gyri_specimen.jpg`**
   - Coronal cross-section of a real human cerebral hemisphere specimen showing the folded cortical gray-matter ribbon ($h \approx 2.6\text{ mm}$), deep sulci, outward gyri, and subcortical white-matter core ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Human_brain_frontal_(coronal)_section.JPG), CC BY 2.5).
   - SHA-256: `ff9a25e993fd5aadb99b44e0e7b8063e96a6a4c348228ce6ee69c6a8c01638da`.

3. **`scout_manifest.json`**
   - Step 1 literature & specimen scout dossier for Slot 061.
   - SHA-256: `7f5410ff8ad2c1ee8fbaff4cb6f4d78abd7afa1d3bec8fe315232a68ff2d8bf8`.

4. **`paper_citations.bib`**
   - BibTeX entries for Richman et al. (1975), Tallinen et al. (2014), Tallinen et al. (2016), Mota & Herculano-Houzel (2015), and Zilles et al. (1988).
   - SHA-256: `56bc1d0beedebe372b7ef164483aac30c86d8ae2bffead867902898a8dfb21d9`.

## Pass 3 additions (2026-10-19, `…-audit-v3`)

5. **`surface_sh_3d_reference.json`**
   - Parameters and provenance for the Pass 3 **3D hemispheric surface buckling surrogate**: a generalized Swift–Hohenberg film-on-curved-substrate equation (Stoop et al. 2015 *Nature Materials* 14:337–342, `doi:10.1038/nmat4202`; Cross & Hohenberg 1993 *Rev. Mod. Phys.* 65:851, `doi:10.1103/RevModPhys.65.851`) solved on the 2D surface of a triaxial ellipsoidal hemisphere (semi-axes $1.30R, 0.70R, 0.95R$; icosphere, 10 242 nodes, cotangent Laplace–Beltrami operator) with a locally preferred wavenumber $k_c(x) = 2\pi/\lambda(h(x))$ from the Richman cube law and an anterior–posterior cortical thickness gradient $h(x) = h_0(1 + \beta x/a)$ within the regional range of Fischl & Dale (2000) *PNAS* 97:11050 (`doi:10.1073/pnas.200033797`).
   - Also holds the honeycomb trivalent-network upper bound for Y-junction density ($4/(\sqrt{3}\lambda^2) = 2.3094\,\lambda^{-2}$) and the 3D crease-wall gyrification index definition $\text{GI}_{3D} = 1 + L_{zero}\bar d / A_{hull}$.
   - **Honesty note:** this is the closest reduced model to the full 3D tetrahedral neo-Hookean FEM on gestational MRI templates (Tallinen et al. 2016) requested by the Pass 2 audit. The sulcal *planform* (labyrinth, Y-junctions, dimples, regional wavelength) is emergent; the sulcal *depth* is imported from the 2D coronal solver. The volumetric FEM remains a Pass 4 item.
   - SHA-256: see `EXPECTED_SHA256` in `1975-richman-cortical-gyrification-buckling.py` (runtime-verified).

- `paper_citations.bib` now additionally contains `stoop2015curvature`, `fischl2000measuring`, and `cross1993pattern` (SHA-256 re-pinned in the script).
- New media: `docs/media/cortical_gyrification_folding_surface_3d.png`.
