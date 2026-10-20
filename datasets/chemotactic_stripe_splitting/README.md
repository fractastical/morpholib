# `datasets/chemotactic_stripe_splitting/` — Painter, Maini & Othmer (1999) Chemotactic Stripe Splitting & Spot Selection Reference Package

Formal SHA-256-locked dataset and biological specimen photograph for `1999-painter-chemotactic-stripe-splitting.py` (Slot 005 of the 100-Loop Infinite Morphospace Campaign).

## Contents

1. **`chemotactic_stripe_splitting_reference.json`**
   - **Painter, Maini & Othmer (1999) *PNAS* 96(10):5549–5554 (`doi:10.1073/pnas.96.10.5549`) & Keller & Segel (1970) *J. Theor. Biol.* 26(3):399–415 (`doi:10.1016/0022-5193(70)90092-5`)**:
     - Governing Keller–Segel chemotactic-reaction-diffusion system with logistic cell proliferation and saturating chemoattractant production:
       $$\frac{\partial n}{\partial t} = D_n \nabla^2 n - \nabla \cdot \bigl(n \chi(c) \nabla c\bigr) + r n \left(1 - \frac{n}{n_0}\right), \qquad \frac{\partial c}{\partial t} = D_c \nabla^2 c + \frac{\alpha n}{1 + \beta n} - \gamma c$$
     - Canonical equal-diffusivity parameter set ($D_n = D_c = 1.0$, $r = 1.0$, $n_0 = 1.0$, $\alpha = 8.0$, $\beta = 1.0$, $\gamma = 4.0$, $\chi = 6.0$), homogeneous steady state $(n^*, c^*) = (1.0, 1.0)$, critical chemotactic sensitivity $\chi_c = 4.5$ ($\chi_{\text{eff},c} = (\sqrt{r D_c} + \sqrt{\gamma D_n})^2 = 9.0$), critical squared wavenumber $k_c^2 = \sqrt{r \gamma / (D_n D_c)} = 2.0$, and characteristic wavelength $\lambda_c = 2\pi / k_c = \pi\sqrt{2} \approx 4.4429$.
   - **Domain-Growth Stripe-Splitting Cascade & 2D Stripe-vs-Spot Selection (Painter et al. 1999 Figs. 1–4)**:
     - Five characteristic regimes across domain-growth stages $L / \lambda_c \in \{1, 2, 4, 8\}$ ($N_{\text{peaks}}: 1 \to 2 \to 4 \to 8$) plus a subcritical chemotaxis negative control ($\chi = 3.0 < \chi_c \implies 0$ stripes).

2. **`chemotactic_pattern_specimen.jpg`**
   - *Myxococcus xanthus* social bacterial morphogenesis exhibiting periodic chemotactic/rippling stripe ridges (right) and fruiting-body spot aggregates (left) with a $500\,\mu\text{m}$ scale bar.
   - Source: [Wikimedia Commons (`File:Myxococcus xanthus fruiting bodies and rippling.png`)](https://commons.wikimedia.org/wiki/File:Myxococcus_xanthus_fruiting_bodies_and_rippling.png), CC BY-SA 4.0.
   - SHA-256: `7b1780ffcb70f31342f1a37f3776dcb07f624fc5d2cf37f0f51aa11daefa345e`.

3. **`scout_manifest.json`**
   - Step 1 (`01_scout`) OpenAlex + Wikimedia Commons literature & specimen discovery dossier for Slot 005.
   - SHA-256: `4b4e20a991f65c81542fa6a014557e5379bbd6a64b0e6a97231e6e5b4f01ea2b`.

4. **`paper_citations.bib`** — BibTeX entries for Painter, Maini & Othmer (1999), Keller & Segel (1970), Myerscough & Murray (1992), and Painter & Hillen (2011).
