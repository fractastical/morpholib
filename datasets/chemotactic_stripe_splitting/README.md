# `datasets/chemotactic_stripe_splitting/` — Painter, Maini & Othmer (1999) *Pomacanthus* Turing–Chemotaxis Stripe Doubling & Keller–Segel Rippling Reference Package

Formal SHA-256-locked dataset and biological specimen photograph for `1999-painter-chemotactic-stripe-splitting.py` (Slot 005 of the 100-Loop Infinite Morphospace Campaign).

> **Pass 3 source correction (2026-10-19).** The seed DOI `10.1073/pnas.96.10.5549` resolves to
> Painter KJ, Maini PK, Othmer HG (1999) **"Stripe formation in juvenile *Pomacanthus* explained by a
> generalized Turing mechanism with chemotaxis"**, *PNAS* 96(10):5549–5554 (PMID 10318921, PMC21897).
> Passes 1–2 (and the SHA-256-pinned `scout_manifest.json`, which cannot be edited without breaking
> its pin) cite this DOI under the non-existent title "Stripe and spotilihood in chemotactic pattern
> formation". The Keller–Segel chemotaxis-logistic system in `chemotactic_stripe_splitting_reference.json`
> is a *generic* Keller–Segel model (Keller & Segel 1970; Painter & Hillen 2011) retained for the
> *Myxococcus* rippling comparison; it is **not** the PNAS 1999 model. The actual PNAS 1999 model is
> encoded in `pomacanthus_stripe_doubling_reference.json`.

## Contents

1. **`pomacanthus_stripe_doubling_reference.json`** *(Pass 3)*
   - Painter, Maini & Othmer (1999) *PNAS* 96(10):5549–5554 governing equations: Kondo–Asai (1995) piecewise-linear Turing morphogens $u, v$ plus chemotactic iridophores $n$ with $\chi(u) = \chi_0 / (k + u)^2$, $\chi_0 < 0$, on an exponentially growing domain $L(t) = L_0 e^{rt}$ with dilution.
   - Fig. 3 parameter set ($D_u = 0.007$, $D_v = 0.1$, $D_n = 5\times10^{-4}$, $\chi_0 = -10^{-4}$, $k = 1$, $L_0 = 11$, $r = 10^{-4}$), steady state $(u^*, v^*) = (3, 2.5)$, Jacobian, Turing quantities ($\operatorname{tr}J = -0.01$, $\det J = 0.005$, $h = 0.00458$, $k_c^2 = 3.2714$, $\lambda_c = 3.4739$, unstable band $[1.3848, 5.1581]$, Neumann half-wave modes $\{5, 6, 7\}$ in band, mode 6 selected $\Rightarrow$ 3 stripes), Fig. 4 2D domain ($1.9 \times 0.95$, growth $1.8\times10^{-4} / 2.2\times10^{-4}$), Fig. 5 break-up and Fig. 8 Lengyel–Epstein adult transition.
   - Kondo & Asai (1995) *Nature* 376:765 *P. semicirculatus* stripe-doubling table ($3 \to 6 \to 12$ vertical stripes at 2 / 6 / 12 months, relative body length $1 : 2 : 4$).
   - Pass 3 growing-curved-surface extension (surface of revolution, cylinder $K = 0$ vs spindle $K \neq 0$) expectations.

2. **`chemotactic_stripe_splitting_reference.json`**
   - Generic Keller–Segel chemotaxis-logistic system (Keller & Segel 1970; Painter & Hillen 2011):
     $$\frac{\partial n}{\partial t} = D_n \nabla^2 n - \nabla \cdot \bigl(n \chi(c) \nabla c\bigr) + r n \left(1 - \frac{n}{n_0}\right), \qquad \frac{\partial c}{\partial t} = D_c \nabla^2 c + \frac{\alpha n}{1 + \beta n} - \gamma c$$
   - Canonical equal-diffusivity parameter set ($D_n = D_c = 1.0$, $r = 1.0$, $n_0 = 1.0$, $\alpha = 8.0$, $\beta = 1.0$, $\gamma = 4.0$, $\chi = 6.0$), homogeneous steady state $(n^*, c^*) = (1.0, 1.0)$, critical chemotactic sensitivity $\chi_c = 4.5$ ($\chi_{\text{eff},c} = (\sqrt{r D_c} + \sqrt{\gamma D_n})^2 = 9.0$), critical squared wavenumber $k_c^2 = \sqrt{r \gamma / (D_n D_c)} = 2.0$, and characteristic wavelength $\lambda_c = 2\pi / k_c = \pi\sqrt{2} \approx 4.4429$.
   - Growing-domain mode-doubling cascade (Crampin, Gaffney & Maini 1999): five regimes $L / \lambda_c \in \{1, 2, 4, 8\}$ ($N_{\text{peaks}}: 1 \to 2 \to 4 \to 8$) plus a subcritical negative control ($\chi = 3.0 < \chi_c \implies 0$ stripes); 2D stripe-vs-spot selection follows Ermentrout (1991).

3. **`chemotactic_pattern_specimen.jpg`**
   - *Myxococcus xanthus* social bacterial morphogenesis exhibiting periodic rippling stripe ridges (right) and fruiting-body spot aggregates (left) with a $500\,\mu\text{m}$ scale bar.
   - Source: [Wikimedia Commons (`File:Myxococcus xanthus fruiting bodies and rippling.png`)](https://commons.wikimedia.org/wiki/File:Myxococcus_xanthus_fruiting_bodies_and_rippling.png), CC BY-SA 4.0.
   - SHA-256: `7b1780ffcb70f31342f1a37f3776dcb07f624fc5d2cf37f0f51aa11daefa345e`.

4. **`scout_manifest.json`**
   - Step 1 (`01_scout`) OpenAlex + Wikimedia Commons literature & specimen discovery dossier for Slot 005. Carries the superseded paper title (see correction note above); pinned and therefore unchanged.
   - SHA-256: `4b4e20a991f65c81542fa6a014557e5379bbd6a64b0e6a97231e6e5b4f01ea2b`.

5. **`paper_citations.bib`** — BibTeX entries for Painter, Maini & Othmer (1999, corrected title), Keller & Segel (1970), Myerscough & Murray (1992), Painter & Hillen (2011), Kondo & Asai (1995), Crampin, Gaffney & Maini (1999), Ermentrout (1991) and Balagam & Igoshin (2015).
