# `datasets/palate_ruguga_turing/` — Economou et al. (2012) *Nature Genetics* Mammalian Palate Rugae Turing Growth-Zone Reference Package

Formal SHA-256-locked dataset and biological specimen image for `2012-economou-palate-ruguga-turing.py` (Slot 008, `economou-palate-ruguga-turing`).

## Contents

1. **`palate_ruguga_turing_reference.json`**
   - **Economou et al. (2012) *Nature Genetics* 44(3):348–351 (`doi:10.1038/ng.1090`)**:
     - Two-component mesenchymal **FGF** activator ($A$, FGF10/FGF7, short-range diffusivity $D_A$) and epithelial **SHH** inhibitor ($H$, Sonic hedgehog, long-range signaling diffusivity $D_H > D_A$) reaction-diffusion PDE operating at a localized anterior-posterior growth zone immediately anterior to anchor ruga 8:
       $$\partial_t A = c_1 A - c_2 H + \rho_A - \mu_A A - \gamma A^3 + D_A \nabla^2 A, \quad \partial_t H = c_3 A - \mu_H H + D_H \nabla^2 H$$
     - Sequential palatal rugae intercalation order in *Mus musculus* (E12.0–E15.5): anchor ruga $8 \to 1 \to 2 \to 3 \to 4 \to 5 \to 6 \to 7 \to 7\text{b}$ as the palatal shelf elongates at velocity $v_g$, producing 9 transverse *Shh*-expressing palatal stripes with inter-ruga wavelength $\lambda_c = 2\pi / q_c$.
     - Explant microsurgical incisions (Economou et al. 2012 Fig. 2):
       - **Coronal (transverse) cut** across the ruga 8 growth zone isolates the newly expanded tissue from the inhibitory SHH halo of existing rugae, triggering premature stripe splitting/bifurcation and ectopic spot/doublet formation.
       - **Sagittal (longitudinal) cut** parallel to the AP palatal axis preserves the AP inhibitory gradient of adjacent rugae, maintaining parallel transverse stripe orientation on both cut halves.
     - Pharmacological and genetic perturbations (Economou et al. 2012 Figs. 3–4):
       - **Cyclopamine SHH blockade** ($\text{IC}_{50}$-graded reduction of $c_2$): dose-dependent broadening, fusion, and eventual widespread unpatterned activation when SHH inhibition is abolished.
       - **Exogenous FGF10 bead implantation**: creates a localized radial zone of SHH inhibition / ruga repulsion around the bead.
       - **Spry2$^{-/-}$; Spry4$^{+/-}$** (hyperactive FGF signaling, elevated $c_1$): compacted, high-density fragmented/ectopic rugae.
       - **Shh$^{\text{fl/fl}}$; K14-Cre** & **Fgf10$^{-/-}$** loss-of-function mutants.
   - **Pantalacci et al. (2008) *BMC Dev. Biol.* 8:116 (`doi:10.1186/1471-213X-8-116`)** & **Welsh & O'Brien (2009) *Dev. Biol.* 336:64–75 (`doi:10.1016/j.ydbio.2009.09.028`)**:
     - Developmental staging and sequential addition of mouse palatal rugae in the rugae growth zone (RGZ) anterior to ruga 8.

2. **`palate_rugae_specimen.jpg`**
   - High-resolution anatomical specimen photograph of the mammalian hard palate (*Palatum durum*, *Equus caballus*) showing bilateral transverse palatal rugae (*Rugae palatinae*, label 3) organized periodically along the anterior-posterior axis from the incisive papilla (label 1) to the posterior hard palate boundary (label 6), bisected by the median palatal raphe (*Raphe palati*, label 4) ([Wikimedia Commons `File:Palatum_horse.JPG`](https://commons.wikimedia.org/wiki/File:Palatum_horse.JPG), CC BY-SA 3.0).

3. **`scout_manifest.json`**
   - Step 1 literature & specimen scout dossier for Slot 008.

4. **`paper_citations.bib`**
   - BibTeX entries for Economou et al. (2012), Pantalacci et al. (2008), Welsh & O'Brien (2009), and Turing (1952).
