# `datasets/french_flag_morphogen/` — Wolpert (1969) French Flag Morphogen & Chick Limb ZPA Reference Package

Formal SHA-256-locked dataset and biological specimen photograph for **Slot 011 (`1969-wolpert-french-flag-morphogen.py`)**.

## Contents

1. **`french_flag_morphogen_reference.json`**
   - **Wolpert (1969) *J. Theor. Biol.* 25(1):1–47 (`doi:10.1016/S0022-5193(69)80016-0`) & Crick (1970) *Nature* 225:420–422 (`doi:10.1038/225420a0`)**:
     - Linear source-sink positional gradient $c_{\text{lin}}(x) = c_0 (1 - x/L)$ with exact size-invariant French-Flag boundaries $x_1/L = 1 - T_1/c_0 = 1/3$, $x_2/L = 1 - T_2/c_0 = 2/3$.
     - Synthesis-Diffusion-Degradation (SDD) reaction-diffusion equation $\partial_t c = D \partial_{xx} c - k c$ with steady-state exponential profile $c_{\text{SDD}}(x) = c_0 e^{-x/\lambda}$, characteristic decay length $\lambda = \sqrt{D/k}$, boundary loci $x_i = \lambda \ln(c_0 / T_i)$, and positional noise error propagation $\sigma_x(x_i) = \sigma_c / |dc/dx|_{x_i} = (\lambda / T_i)\sigma_c$.
   - **Tickle, Summerbell & Wolpert (1975) *Nature* 254:199–202 (`doi:10.1038/254199a0`) & Riddle et al. (1993) *Cell* 75:1401–1416 (`doi:10.1016/0092-8674(93)90626-2`)**:
     - *Gallus gallus* wing bud Zone of Polarizing Activity (ZPA) / Sonic hedgehog (Shh) morphogen thresholds specifying posterior-to-anterior digits **4–3–2** in wild-type and **4–3–2–2–3–4** mirror-image digit duplication following an anterior ZPA graft.
   - **Ben-Zvi & Barkai (2010) *PNAS* 107(15):6924–6929 (`doi:10.1073/pnas.0912734107`)**:
     - Expansion-Repression (ER) dynamic feedback scaling $\lambda_{\text{eff}}(L) \propto L$ across Hamburger-Hamilton stages HH18–HH28 ($L = 180 \to 660\ \mu\text{m}$).

2. **`french_flag_specimen.jpg`**
   - Developing *Gallus gallus domesticus* embryo at 72 hours of incubation (HH18–20) in ovo, showing the central embryonic organizer and radially decaying *area vasculosa* morphogenetic territories.
   - Source: [Wikimedia Commons (`File:Embryo in chicken egg (Gallus domesticus) after 72 hours of incubation.jpg`)](https://commons.wikimedia.org/wiki/File:Embryo_in_chicken_egg_(Gallus_domesticus)_after_72_hours_of_incubation.jpg), CC BY 4.0.
   - SHA-256: `7163e90c56e1316e6398849083a6e48955cd20b124a6df7cf2178247cedd0c4f`.

3. **`paper_citations.bib`** — BibTeX citations for Wolpert (1969), Crick (1970), Tickle et al. (1975), Riddle et al. (1993), and Ben-Zvi & Barkai (2010).

4. **`scout_manifest.json`** — Automated OpenAlex + Wikimedia Commons scouting dossier for Slot 011.
