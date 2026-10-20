# `datasets/pomacanthus_stripe_insertion/` — Kondo & Asai (1995) *Pomacanthus* Reaction-Diffusion Stripe Insertion & Y-Branch Unzipping Reference Package

Formal SHA-256-locked dataset and biological specimen photograph for `1995-kondo-pomacanthus-stripe-insertion.py` (Slot 004 of the 100-Loop Infinite Morphospace Campaign).

## Contents

1. **`pomacanthus_stripe_insertion_reference.json`**
   - **Kondo & Asai (1995) *Nature* 376(6543):765–768 (`doi:10.1038/376765a0`) & Gierer & Meinhardt (1972) *Kybernetik* 12(1):30–39 (`doi:10.1007/BF00289234`)**:
     - Piecewise-linear saturating activator–inhibitor reaction–diffusion PDE on an exponentially expanding domain $L(t) = L_0 \exp(\gamma t)$:
       $$\frac{\partial A}{\partial t} = F(A, I) - g_A A + D_A \nabla^2 A, \qquad \frac{\partial I}{\partial t} = G(A, I) - g_I I + D_I \nabla^2 I$$
       with piecewise-linear synthesis rates $F(A, I) = \min\!\bigl(F_{\max}, \max(0, c_1 A + c_2 I + c_3)\bigr)$ and $G(A, I) = \min\!\bigl(G_{\max}, \max(0, c_4 A + c_5)\bigr)$.
     - Canonical Figure 1 parameters ($c_1 = 0.08$, $c_2 = -0.08$, $c_3 = 0.05$, $c_4 = 0.10$, $c_5 = -0.15$, $g_A = 0.03$, $g_I = 0.06$, $D_A = 0.007$, $D_I = 0.10$, $F_{\max} = 0.20$, $G_{\max} = 0.50$), interior steady state $(A^*, I^*) = (3.0, 2.5)$, critical diffusivity ratio $d_c = 5.2 + 1.6\sqrt{10} \approx 10.2596$, critical squared wavenumber $k_c^2 = 229/70 \approx 3.2714$, and intrinsic wavelength $\lambda_c = 2\pi / k_c \approx 3.4739$.
   - ***Pomacanthus semicirculatus* 1D Mid-Trough Stripe Insertion Cascade ($3 \to 6 \to 12 \to 24$ stripes) vs *Pomacanthus imperator* 2D Y-Branch Horizontal Unzipping (Kondo & Asai 1995 Figs. 1–4)**:
     - Five characteristic regimes across body-length doubling stages $s \in \{1, 2, 4, 8\}$ ($L_{\text{body}} \in \{2, 4, 8, 16\}\text{ cm}$, $N_{\text{stripes}} \in \{3, 6, 12, 24\}$) plus a subcritical inhibitor-diffusion negative control ($D_I = 0.050 \implies d = 7.14 < d_c \implies 0$ stripes).

2. **`pomacanthus_semicirculatus_specimen.jpg`**
   - Juvenile Semicircle angelfish (*Pomacanthus semicirculatus*), Kasai Rinkai Park Aquarium, Tokyo, exhibiting alternating wide primary white stripes and thinner newly inserted secondary blue/white mid-trough stripes.
   - Source: [Wikimedia Commons (`File:Pomacanthus semicirculatus juvenile Kasai aquarium.jpg`)](https://commons.wikimedia.org/wiki/File:Pomacanthus_semicirculatus_juvenile_Kasai_aquarium.jpg), CC BY-SA 4.0.
   - SHA-256: `5564f850c1905fb676889fc409d2cf82779d183184616f101547d8f9b85c6871`.

3. **`scout_manifest.json`**
   - Step 1 (`01_scout`) OpenAlex + Wikimedia Commons literature & specimen discovery dossier for Slot 004.
   - SHA-256: `bf877731da64b97b22c53a6f3bb3c5908b6b32b768fc5a849deaeaf53aab2172`.

4. **`paper_citations.bib`** — BibTeX entries for Kondo & Asai (1995), Gierer & Meinhardt (1972), Crampin, Gaffney & Maini (1999), and Kondo, Iwashita & Yamaguchi (2009).
