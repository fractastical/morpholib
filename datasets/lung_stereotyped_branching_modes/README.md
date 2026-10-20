# `datasets/lung_stereotyped_branching_modes/` — Metzger et al. (2008) & Menshykau et al. (2012, 2014) Lung Stereotyped Branching Modes Reference Package

Formal SHA-256-locked dataset and mammalian bronchial branching specimen for `2008-metzger-lung-stereotyped-branching.py` (Slot 053 of the Infinite Morphospace & MorphoEval Campaign).

## Contents

1. **`lung_stereotyped_branching_modes_reference.json`**
   - **Metzger, Klein, Martin & Krasnow (2008) *Nature* 453(7196):745–750 (`doi:10.1038/nature07005`)**:
     - Three modular genetically encoded branching subroutines governing embryonic mouse (*Mus musculus*, E11–E15) bronchial tree morphogenesis:
       1. **Domain Branching ($\mathbf{D}$):** Periodic lateral sprouting of daughter branches along a parent trunk at spacing $\lambda_{\text{DB}}$ and lateral angle $\theta_{\text{DB}} \approx 68^\circ$ in 4 circumferential rows separated by $90^\circ$ dihedral offsets.
       2. **Planar Bifurcation ($\mathbf{P}$):** Symmetric tip splitting in the same division plane ($\Delta\phi = 0^\circ$, $2\theta_{\text{bif}} \approx 75^\circ$), forming thin lobar edges.
       3. **Orthogonal Bifurcation ($\mathbf{O}$):** Symmetric tip splitting with a $90^\circ$ dihedral plane rotation between consecutive generations ($\Delta\phi = 90^\circ$), filling 3D lobar interior space.
     - Empirical E11–E15 mouse lung lineage benchmarks (`LL_L1_margin` through `RAc_V2_interior`) and developmental tip-count timecourse.
   - **Menshykau, Kraemer & Iber (2012) *Development* 139:4506–4514 (`doi:10.1242/dev.081679`) & Menshykau et al. (2014) *PLoS Comput. Biol.* 10(2):e1003421 (`doi:10.1371/journal.pcbi.1003421`)**:
     - Mesenchymal FGF10 ($F$) + epithelial SHH/Ptch1 receptor-ligand ($R$) Schnakenberg-type reaction-diffusion Turing system governing the biochemical switch between lateral domain sprouting ($\mathbf{D}$) and apical tip bifurcation ($\mathbf{P}/\mathbf{O}$).

2. **`bronchial_tree_specimen.jpg`**
   - Mammalian bronchial tree branching anatomy showing main bronchiole trunk with lateral domain branches ($\mathbf{D}$) transitioning into planar and orthogonal terminal bifurcations ($\mathbf{P}, \mathbf{O}$) and alveolar sacs.
   - Source: [Wikimedia Commons (`File:Bronchial_anatomy.jpg`)](https://commons.wikimedia.org/wiki/File:Bronchial_anatomy.jpg), CC BY 2.5 (Patrick J. Lynch, medical illustrator; C. Carl Jaffe, MD, cardiologist).
   - SHA-256: `3247bc8c35eddcad2681234c00203eb26a9bc95c49d5c441084d93c57033ca69`.

3. **`paper_citations.bib`** — BibTeX citations for Metzger et al. (2008), Menshykau et al. (2012, 2014), and Blanc et al. (2012).

4. **`scout_manifest.json`** — Step 1 (`01_scout`) OpenAlex + Wikimedia Commons scouting dossier.
