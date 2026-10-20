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

5. **`limb_bud_outgrowth_fem_reference.json`** (Pass 3)
   - Reference table for the moving-boundary P1 finite-element limb-bud outgrowth mesh added in Pass 3 (`simulate_growing_limb_bud_fem`): mesh / growth-law definition (polar fan, $r = r_{\text{prox}} + (R(t) - r_{\text{prox}})\,\xi$, $R(t) = \min(R_1, R_0 e^{g t})$), analytic anchors ($\sigma_{\max} = 0.2608$, $q_c = 1.57$, $\lambda_c = 4.002$, dilution threshold $\rho_c = \sigma_{\max}$, slope $d\sigma_{\text{eff}}/d\rho = -1$), the final-arc sweep expectation $N = \lfloor L_{\text{arc}} / \lambda_{\text{AER}} \rfloor$ and the qualitative Raspopovic et al. (2014) Fig. 4 outcomes (5 unbranched rays with the distal gradient, bifurcating rays without it).
   - **Caveat:** every value is either an analytic consequence of the baseline parameters (Lean Laws 4–6 in `morphoeval/lean/loops/Loop_016_bmp_sox9_wnt_digits.lean`) or a qualitative statement from the cited papers; no measured limb-bud growth curve is vendored (listed under `not_vendored`).

6. **`paper_citations_pass3.bib`** (Pass 3, SHA-256-pinned separately so item 4 keeps its original digest)
   - Crampin, Gaffney & Maini (1999) *Bull Math Biol* 61:1093–1120 (`doi:10.1006/bulm.1999.0131`, reaction-diffusion on growing domains), Boehm et al. (2010) *PLoS Biol* 8(7):e1000420 (`doi:10.1371/journal.pbio.1000420`, finite-element limb-bud growth), Marcon et al. (2011) *PLoS Comput Biol* 7(2):e1001071 (`doi:10.1371/journal.pcbi.1001071`, limb-bud growth map used by Raspopovic et al. 2014), and Shubin & Alberch (1986) *Evol Biol* 20:319–387 (`doi:10.1007/978-1-4615-6983-1_6`, digital-arch order of digit formation).

## Pass 3 solver summary (`2014-raspopovic-bmp-sox9-wnt-digits.py`, Section 3b)

- P1 linear-triangle FEM with lumped mass, implicit diffusion, explicit reaction; stiffness re-assembled on the deformed Lagrangian mesh; the conservative update $M_{t+dt}\,c_{t+dt} = M_t\,[c_t + dt\,f(c_t)]$ realises the Crampin–Gaffney–Maini dilution term $-(\nabla\!\cdot\!u)\,c$ exactly.
- FGF/Hoxd wavelength modulation modes: `scaled` ($D \propto (r/R(t))^2$, AER-anchored, default), `material` ($D \propto (r/R_1)^2$, Hox-like) and `none` (ablation). The $F \to D$ mapping is a declared reduction, not an equation of the paper.
- Verification: $K x^2 = -2M$ to $4\times10^{-16}$, $\sum M = $ area exactly, seeded-mode growth rate within 2.2 % of the analytic $\mathrm{Re}\,\sigma(q_c)$, ALE mass conservation exact.
