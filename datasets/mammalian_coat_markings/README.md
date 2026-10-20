# `datasets/mammalian_coat_markings/` — Murray (1981) Animal Coat Markings Reference Package

Formal SHA-256-locked dataset and biological specimen photograph for `1981-murray-mammalian-coat-markings.py` (Slot 003 of the 100-Loop Infinite Morphospace Campaign).

## Contents

1. **`mammalian_coat_markings_reference.json`**
   - **Murray (1981) *J. Theor. Biol.* 88(1):161–199 (`doi:10.1016/0022-5193(81)90334-9`) & Thomas (1975) *Biochimie* 57(4):499–507 (`doi:10.1016/S0300-9084(75)80312-9`)**:
     - Nondimensionalized immobilized-enzyme substrate-inhibition reaction-diffusion system:
       $$\frac{\partial u}{\partial t} = \gamma\left(a - u - \frac{\rho u v}{1 + u + K u^2}\right) + \nabla^2 u, \qquad \frac{\partial v}{\partial t} = \gamma\left(\alpha(b - v) - \frac{\rho u v}{1 + u + K u^2}\right) + d\nabla^2 v$$
     - Canonical parameter set ($a=92$, $b=64$, $\alpha=1.5$, $\rho=18.5$, $K=0.1$, $d=10.0$), homogeneous steady state $(u_0, v_0) = (9.933833, 9.289222)$, Jacobian $J_0$, critical diffusivity ratio $d_c = 9.629185$, and unstable wavenumber band $\gamma k^2 \in [0.523585, 0.964837]$.
   - **Conical Tail Eigenmode Transition (Murray 1981 Section 4, Figs. 9–12)**:
     - Tapered cylinder local wavenumber $k_{n,m}^2(r(x)) = (n\pi/L)^2 + (m/r(x))^2$, critical radius $r_{\text{crit}} = m\sqrt{\gamma / (\gamma k_+^2)} = 1.0985$ below which circumferential $m \ge 1$ spot modes exceed the upper Turing cutoff and transition to $m=0$ axial rings.
   - **Scale Factor $\gamma$ Morphospace Table**:
     - Five characteristic scale regimes from $\gamma = 0.4$ (uniform mouse coat) through $\gamma = 3.5$ (1D tail rings), $\gamma = 8.0$ (tapered spot-to-ring tail), $\gamma = 25.0$ (adult *Panthera pardus* 2D flank spots), to $\gamma = 120.0$ (large-mammal high-mode crowding).

2. **`panthera_pardus_specimen.jpg`**
   - Adult African leopard (*Panthera pardus pardus*), Kruger National Park, South Africa.
   - Photograph by Bernard DUPONT ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Leopard_(Panthera_pardus_pardus)_Kruger.jpg), CC BY-SA 4.0).
   - SHA-256: `86533a5dcf9cfee6755343024fce8a8d9cf29f124866e610fd8e322b1662278e`.

3. **`scout_manifest.json`**
   - Step 1 (`01_scout`) OpenAlex + Wikimedia Commons literature & specimen discovery dossier for Slot 003.
   - SHA-256: `28eabc9cf247b03e2720b1cc457536d2da82b199e8686cd930cd9f373afabdab`.

4. **`paper_citations.bib`** — BibTeX entries for Murray (1981), Thomas (1975), Murray (1988), and Liu, Liaw & Maini (2006).
