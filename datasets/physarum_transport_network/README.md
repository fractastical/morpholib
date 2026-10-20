# `datasets/physarum_transport_network/` — Tero et al. (2010) *Science* & Tero, Kobayashi & Nakagaki (2007) *J. Theor. Biol.* Adaptive *Physarum* Transport Network Dataset

Formal SHA-256-locked reference dataset and *Physarum polycephalum* acellular plasmodial vein network photograph for `2010-tero-physarum-transport-network.py` (Slot 057 of the Infinite Morphospace & MorphoEval Campaign).

## Governing Mathematical Mechanism

1. **Hagen–Poiseuille Kirchhoff Flow on a Plasmodial Tube Network** (`doi:10.1126/science.1177894`, `doi:10.1016/j.jtbi.2006.07.015`):
   - On a graph $G = (V, E)$ with edge lengths $L_{ij}$ and hydraulic conductivities $D_{ij} = \frac{\pi r_{ij}^4}{8\eta}$, protoplasmic shuttle streaming driven by hydrostatic pressure differences $p_i - p_j$ obeys:
     $$Q_{ij} = \frac{D_{ij}}{L_{ij}}(p_i - p_j), \qquad \sum_{j \in \mathcal{N}(i)} Q_{ij} = S_i$$
     where $S_{s_1} = +I_0$, $S_{s_2} = -I_0$ at the active food source/sink pair $(s_1, s_2)$ and $S_i = 0$ at all internal nodes.

2. **Nonlinear Shear-Stress Tube Conductivity Adaptation & $\gamma_c = 1$ Phase Transition**:
   - Tube conductivity adapts to flux magnitude $|Q_{ij}|$ with nonlinearity exponent $\gamma > 0$ and decay rate $\mu > 0$:
     $$\frac{dD_{ij}}{dt} = f(|Q_{ij}|) - \mu D_{ij}, \qquad f(|Q|) = \frac{|Q|^\gamma}{1 + |Q|^\gamma} \;\;\text{or}\;\; |Q|^\gamma$$
   - For two parallel tubes of lengths $L_1 < L_2$ sharing source-sink flux $Q_1 + Q_2 = I_0$, the ratio $\frac{Q_1}{Q_2} = \frac{D_1 / L_1}{D_2 / L_2}$ evolves such that:
     - $\gamma < 1$: unique stable coexistence equilibrium with $\frac{D_1^*}{D_2^*} = \left(\frac{L_2}{L_1}\right)^{\frac{\gamma}{1-\gamma}}$ (redundant parallel loops survive).
     - $\gamma = 1$: critical shortest-path threshold where $\frac{d}{dt}\ln(D_1/D_2) \propto L_1^{-1} - L_2^{-1} > 0$, driving $D_2 \to 0$ and selecting the shortest path.
     - $\gamma > 1$: winner-takes-all instability where shorter paths strictly outcompete longer paths to form Steiner minimum trees / Minimum Spanning Trees.
   - Under stochastic multi-source switching across $N_f = 36$ Kanto food sources, $\gamma \in [1.4, 1.8]$ Pareto-optimizes normalized Total Length ($\text{TL}_{\text{MST}} \approx 1.75$), Transport Efficiency ($\text{MD}_{\text{MST}} \approx 0.85$), and Fault Tolerance ($\text{FT} \approx 0.84$), matching both biological *Physarum* networks and the Tokyo rail system.

3. **Murray's Cube Law at $\gamma = 4/3$** (`doi:10.1073/pnas.12.3.207`):
   - Since $D_{ij} \propto r_{ij}^4$ and steady-state conductivity satisfies $D_{ij}^* \propto |Q_{ij}^*|^\gamma$, setting $\gamma = 4/3$ yields $|Q_{ij}^*| \propto (D_{ij}^*)^{3/4} \propto r_{ij}^3$. By Kirchhoff current conservation $Q_0 = Q_1 + Q_2$ at a symmetric Y-bifurcation, parent and daughter radii satisfy Murray's cube law:
     $$r_0^3 = r_1^3 + r_2^3$$

## Files

- `physarum_transport_network_reference.json` — Governing parameters, two-path competition benchmarks, Tero et al. (2010) Fig. 3 $\gamma \in [0.8, 2.4]$ phase transition curve ($\text{TL}_{\text{MST}}, \text{MD}_{\text{MST}}, \text{FT}$), Tokyo rail vs *Physarum* empirical benchmarks, and specimen metadata.
- `physarum_polycephalum_specimen.jpg` — *Physarum polycephalum* yellow plasmodial transport network on decaying wood (`1024x768` px, CC BY 2.5, Frank Fox / Wikimedia Commons, `sha256: 86a37f7c73c8527b41b24519e8f141a8e9df8e4ed5600fc60d5263160dfc67e5`).
- `paper_citations.bib` — BibTeX references with DOIs.
- `scout_manifest.json` — Step 1 (`01_scout`) literature and specimen manifest.
