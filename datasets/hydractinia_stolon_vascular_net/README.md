# `datasets/hydractinia_stolon_vascular_net/` — Buss, Anderson & Perry (2015) *Biol. Bull.*, Dudgeon et al. (1999) *Am. Nat.* & Blackstone (1999) *Syst. Biol.* Colonial Hydroid Gastrovascular Stolon Network Dataset

Formal SHA-256-locked reference dataset and Yale Peabody Museum *Hydractinia symbiolongicarpus* colony specimen micrograph (`YPM IZ 080458`) for `2020-badin-hydractinia-stolon-vascular-net.py` (Slot 100 of the Project Delphi & MorphoEval Campaign).

## Governing Mathematical Mechanism

1. **Pulsatile Polyp Volume-Displacement Oscillators & Hagen–Poiseuille Kirchhoff Gastrovascular Flow** (`doi:10.1086/BBLv228n2p146`, `doi:10.1086/303167`):
   - Colonial hydroids (*Hydractinia symbiolongicarpus*, *Podocoryna carnea*) consist of muscular feeding polyps ($p \in V_{\text{polyp}}$) connected by a planar chitin-sheathed stolonal network $G = (V, E)$ with time-varying lumen radii $r_{ij}(t)$ and segment lengths $L_{ij}$.
   - Polyps contract rhythmically as active volume-displacement oscillators:
     $$V_p(t) = V_0 + \Delta V_p \sin(\omega_p t + \phi_p), \qquad S_p(t) = -\frac{dV_p}{dt} = -\eta \Delta V_0 \omega_p \cos(\omega_p t + \phi_p)$$
   - Volumetric lumen flux $Q_{ij}(t)$ along stolon $(i,j)$ obeys Hagen–Poiseuille pipe conductance coupled to Kirchhoff nodal volume conservation:
     $$Q_{ij}(t) = G_{ij}(p_i(t) - p_j(t)), \qquad G_{ij} = \frac{\pi r_{ij}^4}{8\mu L_{ij}}, \qquad \sum_{j \in \mathcal{N}(i)} Q_{ij}(t) = S_i(t)$$
   - Endodermal wall shear stress $\tau_{w,ij}(t)$ scales with flux and inverse cubic radius:
     $$\tau_{w,ij}(t) = \frac{4\mu |Q_{ij}(t)|}{\pi r_{ij}^3} = \frac{r_{ij} |p_i(t) - p_j(t)|}{2 L_{ij}}$$

2. **Shear-Stress-Gated Stolon Tip Outgrowth, Anastomosis Loop Closure & Regression Across the Runner-to-Sheet Continuum** (`doi:10.1080/106351599260454`, `doi:10.1046/j.1525-142x.2001.01039.x`):
   - Stolon lumen radius adapts to RMS pulsatile wall shear stress $\bar{\tau}_{w,ij}$:
     $$\frac{dr_{ij}}{dt} = k_r (r_{\max} - r_{ij}) \frac{\bar{\tau}_{w,ij}^2}{\tau_{\text{reg}}^2 + \bar{\tau}_{w,ij}^2} - k_d r_{ij}$$
   - **Tip Outgrowth & Branching:** Peripheral stolon tips elongate and sprout lateral branches when peak pulsatile flow exceeds $Q_{\text{branch}}$.
   - **Anastomosis Loop Closure:** When an elongating stolon tip contacts an existing stolon within capture distance $\delta_{\text{fuse}}$, tip-to-stolon fusion inserts a new edge into the planar graph, incrementing the Euler cyclomatic loop number $b_1 = E - V + 1$ by $+1$ and increasing planar meshedness:
     $$\alpha = \frac{E - V + 1}{2V - 5} \in [0, 1]$$
   - **Low-Shear Regression:** Stolon segments whose RMS wall shear stress falls below $\tau_{\text{reg}}$ (or under polyp excision / $\text{MgCl}_2$ contraction arrest where $\Delta V_p = 0 \implies Q_{ij} = 0 \implies \tau_w = 0 < \tau_{\text{reg}}$) collapse their lumen and regress into empty chitinous perisarc tubes.
   - **Heterochronic Runner-to-Sheet Continuum (Blackstone 1999):** Low polyp pumping / high peripheral elongation produces sparse *Podocoryna*-like runner trees ($\alpha \approx 0.085$), whereas 2,4-DNP redox uncoupling ($\alpha \approx 0.365$) and high *Hydractinia* polyp pumping ($\alpha \approx 0.790$) shift the colony into a dense anastomosed stolonal mat ("sheet" morph).

## Files

- `hydractinia_stolon_vascular_net_reference.json` — Governing hydraulic parameters, runner-to-sheet morphospace benchmarks (*Podocoryna* runner, 2,4-DNP uncoupled shift, *Hydractinia* web, *Hydractinia* dense sheet, and $\text{MgCl}_2$ / polyp-excision regression control), 8-point $\eta$ pump-gain sweep, and specimen metadata.
- `hydractinia_symbiolongicarpus_specimen.jpg` — Yale Peabody Museum *Hydractinia symbiolongicarpus* colony macro micrograph (`YPM IZ 080458`, `1920x1330` px, CC0 1.0 Public Domain, `sha256: 059b83cd797d0516af91662cb738ccf2b8b79cb91888ee8e559b0d73ce47d9cb`).
- `paper_citations.bib` — BibTeX references with verified DOIs.
- `scout_manifest.json` — Step 1 (`01_scout`) literature and specimen manifest.
