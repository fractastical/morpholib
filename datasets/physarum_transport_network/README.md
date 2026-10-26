# `datasets/physarum_transport_network/` — Tero et al. (2010) *Science* & Tero, Kobayashi & Nakagaki (2007) *J. Theor. Biol.* Adaptive *Physarum* Transport Network Dataset

Formal SHA-256-locked reference dataset and *Physarum polycephalum* acellular plasmodial vein network photograph for `2010-tero-physarum-transport-network.py` (Slot 057 of the Project Delphi & MorphoEval Campaign).

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

## Pass 3 (2026-10-19, audit v3) — Peristaltic contraction wave & shuttle streaming

4. **Traveling actomyosin peristaltic wave coupled to the Kirchhoff network** (`doi:10.1073/pnas.1305049110`, Kamiya 1981):
   - Tube radius $r_e(t) = r_e\,[1 - \varepsilon \cos(\omega t - 2\pi\, \mathbf{x}_e\!\cdot\!\mathbf{n}/\lambda)]$ with $\omega = 2\pi/T_c$, $T_c \approx 100$ s and $\lambda \approx L_{\text{organism}}$; instantaneous conductivity $D_e \rho_e^4$; wall motion injects the mean-corrected nodal volume source $S_i^{\text{peri}}(t) = -\tfrac12 \sum_{e \ni i} \dot V_e$ (closed plasmodium, $\sum_i S_i^{\text{peri}} = 0$).
   - Fast timescale: the Kirchhoff system is re-solved over one cycle; signed fluxes $Q_e(t)$ reverse twice per cycle (shuttle streaming); the cycle average $\langle Q_e \rangle$ is a second-order net transport that vanishes for frozen conductance.
   - Slow timescale ($1/\mu \gg T_c$): the Tero reinforcement acts on $\langle |Q_e| \rangle_{T_c}$, so the wave feeds back on long-timescale pruning (larger $\varepsilon$ → more loops, higher FT, higher TL$_{\text{MST}}$).
   - Honesty change: TL$_{\text{MST}}$, MD$_{\text{MST}}$, FT and Betti-1 are now computed from the pruned graph (fixed threshold $D \ge 0.1\,D_{\max}$); the Pass 2 closed-form curves fitted to the reference table were removed.

### Additional files (Pass 3)

- `peristaltic_shuttle_streaming_reference.json` — contraction period (100 s, range 60–120 s), 2 flow reversals per cycle, optimal wavelength $\lambda/L \approx 1$, qualitative shuttle-flow dominance, model-scale parameters and the $\varepsilon$ calibration disclosure. SHA-256 pinned in the script (`EXPECTED_PERI_REF_SHA256`). **DOIs and recalled values are flagged `doi_verified_offline: false` / `value_provenance: recalled` — no network access was available in Pass 3.**
- `paper_citations.bib` gained `kamiya1981physical` (DOI unverified offline).
- New media: `docs/media/physarum_transport_network_shuttle_streaming.gif` (network coloured by signed flux over one contraction cycle).
