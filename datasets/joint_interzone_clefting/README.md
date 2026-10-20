# `datasets/joint_interzone_clefting/` — Kavanagh et al. (2013) *PNAS* & Hiscock et al. (2017) *Dev. Cell* Synovial Joint Interzone Segmentation & Phalangeal Inhibitory Cascade Dataset

Formal SHA-256-locked reference dataset and vertebrate autopod radiograph specimen for `2016-onuff-joint-interzone-clefting.py` (Slot 018, `onuff-joint-interzone-clefting`).

## Contents

1. **`joint_interzone_clefting_reference.json`**
   - **Kavanagh, Shoval, Winslow, Alon, Leary, Kan & Tabin (2013) *PNAS* 110(45):18190–18195 (`doi:10.1073/pnas.1315213110`)**:
     - Developmental Inhibitory Cascade (IC) governing proximal-to-distal phalanx lengths $P_1, P_2, P_3, \dots$ along a growing tetrapod digit ray:
       $$P_n = P_1 \left[1 + (n-1)\frac{A - I}{I}\right] = P_1 + (n-1)\Delta$$
     - Exact midpoint affine cascade law for triphalangeal digits:
       $$P_2 = \frac{1}{2}(P_1 + P_3) \iff \frac{P_2}{P_1} = \frac{1}{2} + \frac{1}{2}\frac{P_3}{P_1} \iff \frac{P_2}{P_1 + P_2 + P_3} = \frac{1}{3}$$
       alongside the asymmetric molar-type inhibitory cascade line $\frac{P_2}{P_1} = \frac{1}{3} + \frac{2}{3}\frac{P_3}{P_1}$ across cursorial, generalist, avian (*Gallus gallus*), and arboreal/grasping tetrapod clades.
   - **Hiscock, Tschopp & Tabin (2017) *Developmental Cell* 41(5):459–465 (`doi:10.1016/j.devcel.2017.04.021`), Hartmann & Tabin (2001) *Cell* 104:341–351 (`doi:10.1016/S0092-8674(01)00222-7`), Brunet et al. (1998) *Science* 280:1455–1457 (`doi:10.1126/science.280.5368.1455`), and Storm et al. (1994) *Nature* 368:639–643 (`doi:10.1038/368639a0`)**:
     - Coupled BMP–GDF5–Noggin/Wnt9a reaction-diffusion Turing-wavefront segmentation of the continuous $Sox9^+$ chondrogenic digit rod into discrete phalanges separated by $Gdf5^+ / Wnt9a^+$ synovial joint interzones under a distal phalanx-forming region (PFR) FGF8 growth gradient, plus *Noggin* null (`Nog-/-`) complete joint fusion ($N_{\text{joints}} = 0$) and *Gdf5* brachypodism (`bp/bp`) proximal joint fusion perturbations.
   - SHA-256: `9acd7913813a1a7b5ee58b9b0ebf1f66bc963890566e7a035c08dfb6bcba6fa1`.

2. **`tetrapod_digit_interzone_specimen.jpg`**
   - Vertebrate autopod anteroposterior (AP) radiograph exhibiting segmented proximal ($P_1$), middle ($P_2$), and distal ($P_3$) phalanges separated by radiolucent MCP, PIP, and DIP synovial joint interzone clefts across Digits II, III, and IV.
   - Source: [Wikimedia Commons `File:Polydactyly_01_Rhand_AP.jpg`](https://commons.wikimedia.org/wiki/File:Polydactyly_01_Rhand_AP.jpg) (CC BY-SA 3.0).
   - SHA-256: `995e43a2a0cf781547dc8f573685a533013cea3502fb2ebbb9c1fc8a6cb14e90`.

3. **`scout_manifest.json`**
   - Step 1 (`01_scout`) literature & specimen discovery dossier for Slot 018.
   - SHA-256: `1ae24e293d2062b2bd117c1b8d3eb74094cd667079a1f5fcae706c4eef1cbcce`.

4. **`paper_citations.bib`**
   - BibTeX citations for Kavanagh et al. (2013), Hiscock et al. (2017), Hartmann & Tabin (2001), Brunet et al. (1998), and Storm et al. (1994).
   - SHA-256: `4d966fca533780af9267a998b5bfd43ac1a2d443cf05ed85b292f06e9c0a15ba`.
