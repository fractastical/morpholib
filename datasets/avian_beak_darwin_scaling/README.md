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

## Pass 3 addendum (2026-10-19) — reduced two-material finite-element fracture-risk reference

The Pass 2 audit asked for a "full micro-CT trabecular bone and keratin rhamphotheca finite-element fracture solver under asymmetric seed-cracking loads". No micro-CT volume is vendored, so Pass 3 implements the closest honest reduced version (script Section 4b): a 2D sagittal plane-stress Q4 finite-element model of the sheared/scaled upper beak with a Gibson–Ashby-homogenised trabecular core and a keratin rhamphotheca shell, loaded by tip-biting, base-biting and 30° oblique seed reactions (Soons et al. 2010, 2012, 2015). The residual (voxel-resolved 3D mesh, mediolateral asymmetry, crack propagation) is carried as a Pass 4 item.

5. **`beak_fem_fracture_reference.json`**
   - Material constants (bone $E_s = 7$ GPa, keratin $E_k = 1.5$ GPa, $\nu = 0.3$, order-of-magnitude strengths 150 / 60 MPa), Gibson–Ashby law $E_\mathrm{trab} = E_s\,(\mathrm{BV/TV})^2$ with default BV/TV = 0.55, rhamphotheca thickness 0.35 mm, Q4 discretisation (90 × 18 default; 60 × 12 / 90 × 18 / 135 × 27 convergence set), seed-contact patch 1.2 mm, the three load cases, closed-form Euler–Bernoulli / Timoshenko benchmarks with acceptance tolerances, and the qualitative Soons et al. predictions used as references.
   - **All constants and predictions were transcribed offline from memory (no network access in Pass 3) and must be re-verified against the papers in Pass 4.**
   - SHA-256: `bff86f96d0fcc2fd2ae2f01c5e34d1a418f701c85993a03e40b4a55b69ae5028`.

6. **`pass3_citations.bib`**
   - BibTeX entries for Soons et al. (2010) *Phil. Trans. R. Soc. B* 365:1093–1098 (`doi:10.1098/rstb.2009.0280`), Soons et al. (2012) *J. R. Soc. Interface* 9:1787–1796 (`doi:10.1098/rsif.2011.0910`), Soons et al. (2015) *PLoS ONE* 10:e0129479 (`doi:10.1371/journal.pone.0129479`), Herrel et al. (2005) *J. Evol. Biol.* 18:669–675 (`doi:10.1111/j.1420-9101.2004.00857.x`), Gibson & Ashby (1997) *Cellular Solids* (`doi:10.1017/CBO9781139878326`), Seki et al. (2005) *Acta Materialia* 53:5281–5296 (`doi:10.1016/j.actamat.2005.04.048`). DOIs unverified offline.
   - SHA-256: `37c2a856e17b123bf55fe3b24ddba36e9b464f1b2b1f1b7fcf0655e354d454b4`.

Outputs added by Pass 3: `docs/media/avian_beak_darwin_scaling_fem_fracture.png` (6-panel FEM figure) and `docs/media/avian_beak_darwin_scaling_fem_load_sweep.gif` (seed-contact position sweep on *G. fortis*).

## Pass 3 addendum (2026-10-19) — reduced two-material finite-element fracture-risk model

Pass 2 remediation item: *"Extend the 3D conical-parabolic beak surface to a full micro-CT trabecular bone and keratin rhamphotheca finite-element fracture solver under asymmetric seed-cracking loads."*  No micro-CT volume is vendored, so Pass 3 implements the closest honest reduced version — a 2D sagittal plane-stress Q4 finite-element solver (Section 4b of the script) with a Gibson–Ashby homogenised trabecular core, a keratin rhamphotheca shell, tip-/base-biting and oblique (rostrocaudal shear) seed loads, and a linear-elastic fracture-risk index.  Residual for Pass 4: voxel-resolved 3D micro-CT mesh, mediolateral load asymmetry, crack propagation.

5. **`beak_fem_fracture_reference.json`**
   - Material constants (cortical bone $E_s = 7$ GPa, trabecular BV/TV default 0.55 with $E_{trab} = E_s\,(\mathrm{BV/TV})^2$, keratin $E_k = 1.5$ GPa, $t_k = 0.35$ mm, order-of-magnitude strengths), Q4 discretisation (90×18 default, convergence meshes), Soons et al. (2015)-type load cases (tip $x/L = 0.08$, base $x/L = 0.70$, oblique 30°, 1.2 mm seed-contact traction), closed-form Euler–Bernoulli / Timoshenko benchmarks, and the qualitative Soons et al. (2010, 2012, 2015) predictions used as anchors.
   - **Provenance caveat:** every constant and empirical statement was transcribed *offline from memory* during the Pass 3 round (no network access); they are order-of-magnitude anchors, not digitised figure values, and all DOIs must be re-verified in Pass 4.
   - SHA-256: `bff86f96d0fcc2fd2ae2f01c5e34d1a418f701c85993a03e40b4a55b69ae5028`.

6. **`pass3_citations.bib`**
   - BibTeX for Soons et al. (2010) *Phil. Trans. R. Soc. B* 365:1093–1098 (`doi:10.1098/rstb.2009.0280`), Soons et al. (2012) *J. R. Soc. Interface* 9:1787–1796 (`doi:10.1098/rsif.2011.0910`), Soons et al. (2015) *PLoS ONE* 10:e0129479 (`doi:10.1371/journal.pone.0129479`), Herrel et al. (2005) *J. Evol. Biol.* 18:669–675 (`doi:10.1111/j.1420-9101.2004.00857.x`), Gibson & Ashby (1997) *Cellular Solids* (`doi:10.1017/CBO9781139878326`), Seki et al. (2005) *Acta Materialia* 53:5281–5296 (`doi:10.1016/j.actamat.2005.04.048`).  DOIs transcribed offline; unverified this pass.
   - SHA-256: `37c2a856e17b123bf55fe3b24ddba36e9b464f1b2b1f1b7fcf0655e354d454b4`.

New media: `docs/media/avian_beak_darwin_scaling_fem_fracture.png` (6-panel FEM figure) and `docs/media/avian_beak_darwin_scaling_fem_load_sweep.gif` (seed-contact position sweep, *G. fortis*).
