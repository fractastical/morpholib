# `datasets/bmp_sox9_wnt_digits/` — Raspopovic et al. (2014) *Science* Bmp–Sox9–Wnt Digit Patterning Reference Package

Formal SHA-256-locked dataset and biological specimen image for `2014-raspopovic-bmp-sox9-wnt-digits.py` (Slot 016, `raspopovic-bmp-sox9-wnt-digits`).

## Contents

1. **`bmp_sox9_wnt_digits_reference.json`**
   - **Raspopovic, Marcon, Russo & Sharpe (2014) *Science* 345(6196):566–570 (`doi:10.1126/science.1252960`)**:
     - 3-node cell-autonomous transcription factor **Sox9** ($S$, $D_S = 0$) + diffusible **Bmp** ($B$, $D_B > 0$) + diffusible **Wnt** ($W$, $D_W > 0$) Turing reaction-diffusion network:
       $$\partial_t S = k_{ss} S + k_{sb} B - k_{sw} W - S^3, \quad \partial_t B = -k_{bs} S - \mu_B B + D_B \nabla^2 B, \quad \partial_t W = -k_{ws} S + k_{wb} B - \mu_W W + D_W \nabla^2 W$$
     - Reduced 2-diffuser effective Jacobian $J_{\text{eff}}$ and 3-node dispersion polynomial $\Delta(q^2) = \det(-J_3 + q^2 D)$, demonstrating diffusion-driven Turing instability without direct ligand self-activation ($J_{BB} = -\mu_B < 0, J_{WW} = -\mu_W < 0$) and strict out-of-phase eigenmode structure ($\text{corr}(S, B) \approx -1$, $\text{corr}(S, W) \approx -1$).
     - Distal AER-FGF + Hoxd gradient modulation preventing stripe bifurcation in the expanding fan-shaped mouse autopod (E11.5–E12.5) and pharmacological perturbation table (Wild-Type 5 digits, LDN-193189/Noggin BMP inhibition, CHIR99021 Wnt activation, IWP-2 Wnt inhibition, and FGF/Hox polydactyly).
   - **Marcon, Diego, Sharpe & Müller (2016) *eLife* 5:e14022 (`doi:10.7554/eLife.14022`)** & **Sheth et al. (2012) *Science* 338:1476–1480 (`doi:10.1126/science.1226804`)**:
     - Mathematical analysis of 3-node immobile-node receptor-ligand Turing networks and Hox/FGF wavelength control of digit count ($N_{\text{digits}} \in [3, 7]$).

2. **`sox9_limb_digits_specimen.jpg`**
   - Whole-mount in situ hybridization (WISH) of embryonic mouse (*Mus musculus*) limb buds at E12.5 (*Sox9*, Panels O–P) and E13.5 (*Col2a1*, Panels K–L), comparing wild-type pentadactyl (5 radial digit condensations) and polydactylous ($Tbx2^{-/-}$, 6 digit condensations) handplates (Farin et al. 2013 *PLoS Genetics* 9(4):e1003467, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Duplication_of_digit_4_is_preceded_by_reduced_apoptosis_and_expanded_chondrogenesis_in_the_posterior_limb_mesenchyme.png), CC BY 4.0).
   - SHA-256: `533f621904d6f2ce72365399ca1403c72b5db4dfa717c61e4834ca9beb366da5`.

3. **`scout_manifest.json`**
   - Step 1 literature & specimen scout dossier for Slot 016.
   - SHA-256: `f990b473b6a93cfa784b32a164ad808efbb730f34c6827cf7b733e236d97d5c9`.

4. **`paper_citations.bib`**
   - BibTeX entries for Raspopovic et al. (2014), Sheth et al. (2012), Marcon et al. (2016), and Farin et al. (2013).
