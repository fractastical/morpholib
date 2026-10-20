# `datasets/diatom_frustule_phase_sep/` — Sumper (2002) & Lenoci–Camp (2008) Diatom Biosilica Phase-Separation Reference Package

Formal SHA-256-locked dataset and biological specimen scanning electron micrographs for `2002-sumper-diatom-frustule-phase-sep.py` (Slot 071).

## Contents

1. **`diatom_frustule_phase_sep_reference.json`**
   - **Sumper (2002) *Science* 295(5564):2430–2433 (`doi:10.1126/science.1070026`)**:
     - Hierarchical long-chain polyamine (LCPA) liquid-liquid phase separation within the Silica Deposition Vesicle (SDV) of *Coscinodiscus wailesii* generating three self-similar pore tiers: Generation 1 primary areolae/foramen ($d_1 = 1250\text{ nm}$, pitch $a_1 = 1850\text{ nm}$, $d_1/a_1 = 0.676$), Generation 2 cribrum pores ($d_2 = 220\text{ nm}$, pitch $a_2 = 380\text{ nm}$), and Generation 3 cribrella sieve pores ($d_3 = 42\text{ nm}$, pitch $a_3 = 75\text{ nm}$).
   - **Lenoci & Camp (2008) *Langmuir* 24(1):217–223 (`doi:10.1021/la7022784`)**:
     - Conserved fourth-order Cahn-Hilliard phase-separation PDE $\partial_t \phi = M \nabla^2 (\phi^3 - \phi - \epsilon^2 \nabla^2 \phi)$, linear spinodal dispersion relation $\sigma(k) = M k^2 (1 - 3\bar{\phi}^2 - \epsilon^2 k^2)$, critical wavenumber $k_c = \sqrt{(1 - 3\bar{\phi}^2)/(2\epsilon^2)}$, characteristic wavelength $\lambda_c = 2\pi / k_c$, and two-stage quenched hierarchical patterning.

2. **`coscinodiscus_diatom_specimen.jpg`**
   - Scanning electron micrographs of the internal and external valve plates of the centric marine diatom *Coscinodiscus wailesii* across three magnification scales ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Inner_and_external_plates_of_a_single_valve_of_the_centric_diatom_Coscinodiscus_wailesii.webp), CC BY-SA 4.0).
   - SHA-256: `47aca39b720b67b1b46754c89f679425db1012e92131277a3d09b1562274a1ad`.

3. **`scout_manifest.json`** — Step 1 (`01_scout`) OpenAlex and Wikimedia Commons scouting dossier for Slot 071.

4. **`paper_citations.bib`** — BibTeX entries for Sumper (2002), Lenoci & Camp (2008), Losic et al. (2006), and Cahn & Hilliard (1958).
