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

   - **Pass 3 additions (`metzger-lung-stereotyped-branching-modes-v2`):**
     - `empirical_mouse_lung_lineage_benchmarks[].lobe_cross_section_position_norm` — normalised lobe cross-section position of each lineage root (margin lineages 0.90, interior lineages 0.15). **Assigned from the Metzger (2008) lineage class, not measured**; it is the only lineage-level input to the pleura-confined model, which then predicts the dihedral mode.
     - `pleura_confinement_reference` — parameters and boundary condition of the reduced diffuse-domain phase-field pleura cavity (Li, Lowengrub, Rätz & Voigt 2009) used in Section 3b of the script, plus the closed-form slab profile used to verify the solver.
     - `postnatal_alveolarization_reference` — bulk (P4–P21) and continued (P14–P36) alveolarisation windows (Mund, Stampanoni & Schittny 2008, *Dev. Dyn.* 237:2108–2116) and the adult alveolar number ≈ 2.3 × 10⁶ (Knust, Ochs, Gundersen & Nyengaard 2009, *Anat. Rec.* 292:113–122) used as the single calibration target of the stochastic septation model in Section 4b.
     - The three Pass 3 references were cited offline from memory; their DOIs (`10.4310/CMS.2009.v7.n1.a4`, `10.1002/dvdy.21633`, `10.1002/ar.20747`) are flagged `verification: unverified` in the JSON and must be checked by the Pass 3 audit.

3. **`paper_citations.bib`** — BibTeX citations for Metzger et al. (2008), Menshykau et al. (2012, 2014), Blanc et al. (2012), and (Pass 3) Li et al. (2009), Mund et al. (2008), Knust et al. (2009).

4. **`scout_manifest.json`** — Step 1 (`01_scout`) OpenAlex + Wikimedia Commons scouting dossier.
