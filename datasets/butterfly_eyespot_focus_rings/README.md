# `datasets/butterfly_eyespot_focus_rings/` — Nijhout (1990) & Brakefield et al. (1996) Butterfly Wing Eyespot Focus & Concentric Ring Dataset

Formal SHA-256-locked dataset and biological specimen photograph for **Slot 098 (`2001-nijhout-butterfly-eyespot-focus-rings.py`)**.

## Contents

1. **`butterfly_eyespot_focus_rings_reference.json`**
   - **Nijhout (1990) *Proc. R. Soc. B* 239(1294):81–113 (`doi:10.1098/rspb.1990.0048`)**:
     - Stage-1 2D Meinhardt activator–inhibitor reaction–diffusion inside a wing-vein-bounded cell compartment $[0, L_x] \times [-W_y/2, W_y/2]$, where lateral inhibition from the longitudinal bounding veins ($y = \pm W_y/2$) and wing margin ($x = L_x$) confines activator self-enhancement to the compartment midline ($y = 0$) and isolates a discrete point-like eyespot organizer (**focus**) at $(x_f, 0)$.
   - **Brakefield et al. (1996) *Nature* 384:236–242 (`doi:10.1038/384236a0`) & Brunetti et al. (2001) *Curr. Biol.* 11(18):1451–1461 (`doi:10.1016/S0960-9822(01)00429-8`)**:
     - Stage-2 focal morphogen synthesis–diffusion–degradation PDE $\partial_t c = D \nabla^2 c - k c + S_0 \delta(\mathbf{x} - \mathbf{x}_{\text{focus}})$ with steady-state decay length $\lambda = \sqrt{D/k}$ and radial profile $c(r) = c_0 e^{-r/\lambda}$.
     - Concentric transcriptional thresholds $\theta_{\text{pupil}} > \theta_{\text{black}} > \theta_{\text{gold}}$ specifying the central UV-reflective white pupil, the middle *Spalt* (*Sal*) / *Distal-less* (*Dll*) black melanin disc ($R_{\text{black}} = \lambda \ln(c_0/\theta_{\text{black}})$), and the outer *Engrailed* (*En*) golden-yellow annulus ($R_{\text{gold}} = \lambda \ln(c_0/\theta_{\text{gold}})$).
     - Five experimental and genetic regimes: Wet-Season Wild-Type ($c_0 = 1.00$), Dry-Season Ecdysone Polyphenism ($c_0 = 0.42$), *Goldeneye* Mutant (*Spalt* threshold shift $\theta_{\text{black}} = 0.44$), Twin-Focus Fusion Hourglass ($d_{12} = 1.50\text{ mm} < 2 R_{\text{gold}}$), and Early Pupal Focal Microcautery Ablation ($c_0 = 0.05 \le \theta_{\text{gold}} \implies R_{\text{gold}} = 0$).

2. **`bicyclus_anynana_eyespot_specimen.jpg`**
   - *Bicyclus anynana* (Squinting Bush Brown, Nymphalidae: Satyrinae) ventral wing surface showing the primary forewing Cu1 eyespot (central white pupil at `(596, 231)`, black melanin core $R_{\text{black}} = 15.44\text{ px}$, golden-yellow outer ring $R_{\text{gold}} = 20.40\text{ px}$) and the submarginal hindwing M1–Cu2 eyespot series centered within each vein-bounded wing cell.
   - Source: [Wikimedia Commons (`File:Squinting bush brown (Bicyclus anynana anynana).jpg`)](https://commons.wikimedia.org/wiki/File:Squinting_bush_brown_(Bicyclus_anynana_anynana).jpg), CC BY-SA 4.0.
   - SHA-256: `f1826cee4460278241544fae35a28f0c6b2f69604cc670b950c8c1c98671e715`.

3. **`paper_citations.bib`** — BibTeX citations for Nijhout (1990), French & Brakefield (1995), Brakefield et al. (1996), Brunetti et al. (2001), and Beldade et al. (2002).

4. **`scout_manifest.json`** — Automated OpenAlex + Wikimedia Commons scouting dossier for Slot 098.
