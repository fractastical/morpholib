# `datasets/gdnf_ret_ureteric_kidney/` — Menshykau & Iber (2013) & Short et al. (2014) Kidney Ureteric Bud GDNF–RET Turing Branching Reference Package

Formal SHA-256-locked dataset and biological specimen image for `2014-menshykau-gdnf-ret-ureteric-kidney.py` (Slot 055, `menshykau-gdnf-ret-ureteric-kidney`).

## Contents

1. **`gdnf_ret_ureteric_kidney_reference.json`**
   - **Menshykau & Iber (2013) *Physical Biology* 10(4):046003 (`doi:10.1088/1478-3975/10/4/046003`)**:
     - Schnakenberg-type cooperative ligand-receptor binding ($2\text{RET} + \text{GDNF} \to \text{RET}_2\text{GDNF}$) reaction-diffusion equations coupling slow epithelial RET receptor diffusion ($D_R = 0.02$) with rapid mesenchymal GDNF diffusion ($D_G = 1.0$, $d = D_G / D_R = 50$), positive Wnt11 feedback, and intracellular Sprouty1 (`Spry1`) antagonism.
     - Homogeneous steady state $(R^*, G^*) = (a + \rho_G, \rho_G / (a + \rho_G)^2)$, Turing dispersion relation, critical wavenumber $k_c^2 = \gamma (a + \rho_G)/\sqrt{D_R D_G}$, and circumferential tip-splitting mode selection ($m=1$ elongation, $m=2$ dichotomous bifurcation, $m=3$ planar/tetrahedral trifurcation).
   - **Short et al. (2014) *Developmental Cell* 29(2):188–202 (`doi:10.1016/j.devcel.2014.02.017`)**:
     - Global 3D optical projection tomography (OPT) morphometry of the developing mouse (*Mus musculus*) ureteric tree across E11.5–E15.5 (generations $g=1$ to $g=8$), tracking ureteric tip count $N_{\text{tips}}(g)$ from T-stage ($2$ tips) to $354$ tips, Horton–Strahler bifurcation ratio $R_b \approx 2.08$, segment length ratio $R_L \approx 1.38$, and bilateral vs trifurcating ampulla mode frequencies.
   - **Watanabe & Costantini (2004) *Dev. Biol.* 271:98–106 (`doi:10.1016/j.ydbio.2004.09.019`) & Basson et al. (2005) *Dev. Cell* 8:229–239 (`doi:10.1016/j.devcel.2004.12.004`) & Miyazaki et al. (2000) *J. Clin. Invest.* 105:863–873 (`doi:10.1172/JCI8256`)**:
     - 5-condition genetic and signaling perturbation suite: Wild-Type control, `Spry1-/-` ectopic supernumerary budding & trifurcation, exogenous GDNF-soaked bead attraction & ampulla swelling, BMP4 stalk-elongation inhibition, and `Gdnf-/-` / `Ret-/-` renal agenesis negative control.

2. **`kidney_ureteric_tree_specimen.jpg`**
   - E12.5 mouse (*Mus musculus*) metanephric kidney whole-mount cytokeratin-GFP ureteric bud branching tree and pErk1/2 immunofluorescence comparing wild-type control (Panel A, 12 ureteric tips) vs hypomorphic mutant (Panel B, 6 ureteric tips) ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Effect_of_targeted_genetic_inactivation_of_the_PRR_in_the_ureteric_bud_(UB)_on_UB_branching_and_Erk1-2_phosphorylation_in_E12.5_metanephroi.png), CC BY 4.0).
   - SHA-256: `dceaf5044c7014e44bf8a9ab0e728a670255e68952454f1e65c3d7a3412feff7`.

3. **`scout_manifest.json`**
   - Step 1 literature & specimen scout dossier for Slot 055.
   - SHA-256: `60b78a1876b45c42eb1aeffb592003c16e5df5dc486573211856568945f616fe`.

4. **`paper_citations.bib`**
   - BibTeX entries for Menshykau & Iber (2013), Short et al. (2014), Watanabe & Costantini (2004), Basson et al. (2005), and Miyazaki et al. (2000).
   - SHA-256: `3eb2da09bd233938ed2680a5d121650833ea7748b2b77464de0a1908e55c1484` (Pass 3: typo in the Miyazaki et al. 2000 title fixed, "ofoka" → "of"; previous digest `62f203a6…db331`).

5. **`renal_capsule_phase_field_reference.json`** (Pass 3)
   - Parameter table for the reduced 3D phase-field renal capsule / nephrogenic-zone model (Section 3b of the script) that closes the Pass 2 remediation item in reduced form: capsule surface-area expansion law anchored on the Short et al. (2014) observed tip number, diffuse-domain GDNF equation, and every constant with an explicit provenance status (`approximate anatomical anchor`, `modelling assumption`, `numerical`).
   - Qualitative references used by the three new `measure_morphoeval()` entries (tips-in-nephrogenic-zone fraction, nephrogenic-zone/core GDNF enrichment, capsule-crowding tip suppression) and the residual remediation statement (a true moving-mesh FEM is not implemented).
