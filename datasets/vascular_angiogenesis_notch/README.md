# `datasets/vascular_angiogenesis_notch/` — Bentley, Gerhardt & Bates (2008), Hellström et al. (2007) & Jakobsson et al. (2010) VEGF–Dll4–Notch1 Angiogenic Tip/Stalk Selection Reference Package

Formal SHA-256-locked reference dataset and fluorescence biological specimen micrograph for `2021-ibrahim-vascular-angiogenesis-notch.py` (Slot 060 of the 100-Loop Project Delphi & MorphoEval Campaign).

## Contents

1. **`vascular_angiogenesis_notch_reference.json`**
   - **Bentley, Gerhardt & Bates (2008) *PLoS Comput. Biol.* 4(3):e1000052 (`doi:10.1371/journal.pcbi.1000052`)**:
     - Coupled endothelial VEGFR2 ($V_i$), Delta-like 4 ligand ($D_i$), and Notch1 receptor ($N_i$) lateral-inhibition ODE network along a sprouting vessel wall of $M=16$ cells, driving period-2 Brillouin-zone $k=\pi$ salt-and-pepper Tip–Stalk–Tip–Stalk alternation under physiological VEGF gradients and synchronous oscillations under excessive uniform VEGF overload.
   - **Hellström et al. (2007) *Nature* 445(7129):776–780 (`doi:10.1038/nature05571`)**:
     - Pharmacological $\gamma$-secretase Notch blockade (DAPT, $k_t \to 0$) and genetic *Dll4* haploinsufficiency (`Dll4+/-`), demonstrating transition from physiological salt-and-pepper sprouting ($50\%$ tip cells) to pathological hypersprouting ($75\%$–$100\%$ tip cells, non-perfused syncytial plexus).
   - **Jakobsson et al. (2010) *Nat. Cell Biol.* 12(10):943–953 (`doi:10.1038/ncb2103`) & Chappell et al. (2009) *Dev. Cell* 17(3):377–386 (`doi:10.1016/j.devcel.2009.06.004`)**:
     - Chimeric `WT : Vegfr2+/-` mosaic endothelial tip competition ($\approx 87.5\%$ WT tip occupancy) and soluble VEGFR1 (*sFlt1*) decoy-receptor VEGF sequestration rescuing salt-and-pepper patterning under high VEGF.

2. **`huvec_angiogenic_sprouting_plexus_specimen.jpg`**
   - Human umbilical vein endothelial cells (*Homo sapiens* HUVECs) self-assembling into a living angiogenic sprouting vascular network with calcein-AM green fluorescence staining, showing vascular cords (stalks), branch junctions ($N = 69$), vascular lacunae, and blind-ending filopodial sprouting tip cells ($N = 58$).
   - Source: [Wikimedia Commons (`File:Where vessels are born - HUVECs self-assemble into a living vascular network.png`)](https://commons.wikimedia.org/wiki/File:Where_vessels_are_born_-_HUVECs_self-assemble_into_a_living_vascular_network.png), CC BY 4.0.
   - SHA-256: `9bf4f3e2692bd2ac4ffc245ca2b8ac9c29762b90a89b3a9f4751e1b00e58132c`.

3. **`paper_citations.bib`** — BibTeX citations for Bentley et al. (2008), Hellström et al. (2007), Jakobsson et al. (2010), Chappell et al. (2009), Collier et al. (1996), and (Pass 3) Pries et al. (1995, 1998), Murray (1926), Chen et al. (2012), Franco et al. (2015), Lawson et al. (2001), le Noble et al. (2004), Wang et al. (1998), Lobov et al. (2007).

4. **`scout_manifest.json`** — Step 1 (`01_scout`) OpenAlex + Wikimedia Commons scouting dossier.

5. **`hemodynamic_pruning_av_specification_reference.json`** (Pass 3 remediation, SHA-256 `8764658d501c4560fe73821cd8b22015d87d3707f484947ab5fba1d788c52a97`)
   - Reference table for Section 3b of the script: the lumenized Hagen–Poiseuille / Kirchhoff ladder plexus built from the anastomosed sprouts, Pries–Secomb–Gaehtgens pressure-dependent wall-shear set point $\tau_e(P) = 100 - 86\,e^{-5000(\log_{10}\log_{10}P)^{5.4}}$ dyn cm$^{-2}$ with metabolic stimulus $k_m \log_{10}(1 + Q_\mathrm{ref}/Q)$, minimum red-cell-perfusable lumen radius (1.3 µm) used for topological pruning, and the Notch-gated ephrinB2/EphB4 arteriovenous readout.
   - Every parameter carries a `provenance` tag (`measured` / `derived` / `approximate` / `assumption`) so the audit can separate literature values from calibrations. `reference_values` gives the targets used by `measure_morphoeval()`: capillary wall shear band 1–2 Pa (Pries 1995), Murray exponent 3 (Murray 1926; Lean Theorem 4), Kirchhoff residual 0, WT arteriovenous segregation 1.0 vs 0.5 under Notch loss (Wang 1998; Lawson 2001), and the approximate 1.3–2× Notch-blockade hyperdensity band (Lobov 2007; Hellström 2007).
   - The script's `verify_hemodynamic_reference()` SHA-256-locks this file and cross-checks that `HEMO_PARAMS` equals the pinned values.
