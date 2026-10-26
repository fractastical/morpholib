# `datasets/vascular_cube_law_branching/` — Murray (1926a, 1926b), Zamir (1976) & Sherman (1981) Vascular Branching Reference Package

Formal SHA-256-locked dataset and biological specimen fundus photographs for `1959-murray-vascular-cube-law-branching.py` (Slot 052 of the Project Delphi & MorphoEval Campaign).

## Contents

1. **`vascular_cube_law_branching_reference.json`**
   - **Murray (1926a) *Proc. Natl. Acad. Sci. USA* 12(3):207–214 (`doi:10.1073/pnas.12.3.207`)**:
     - Poiseuille viscous power + metabolic blood-volume maintenance cost functional $P(r) = \frac{8 \eta L Q^2}{\pi r^4} + b \pi r^2 L$, stationarity $\partial P/\partial r = 0 \implies Q = \frac{\pi}{4}\sqrt{b/\eta}\, r^3$, scale-invariant wall shear stress $\tau_w = \sqrt{\eta b}$, and cubic radius conservation $r_0^3 = r_1^3 + r_2^3$ at arterial bifurcations.
   - **Murray (1926b) *J. Gen. Physiol.* 9(6):835–841 (`doi:10.1085/jgp.9.6.835`)**:
     - Exact optimal arterial branching angles $\cos(\theta_1) = \frac{r_0^4 + r_1^4 - r_2^4}{2 r_0^2 r_1^2}$, $\cos(\theta_2) = \frac{r_0^4 + r_2^4 - r_1^4}{2 r_0^2 r_2^2}$, and $\cos(\theta_1 + \theta_2) = \frac{r_0^4 - r_1^4 - r_2^4}{2 r_1^2 r_2^2}$ ($\theta_1 = \theta_2 = 37.47^\circ$, $\theta_1 + \theta_2 = 74.93^\circ$ for symmetric daughters $r_1 = r_2 = 2^{-1/3} r_0$).
   - **Zamir (1976) *J. Theor. Biol.* 62(1):227–251 (`doi:10.1016/0022-5193(76)90058-8`) & Sherman (1981) *J. Gen. Physiol.* 78(4):431–453 (`doi:10.1085/jgp.78.4.431`)**:
     - Asymmetry ratio table $\alpha = r_2/r_1 \in [0.2, 1.0]$, 7-generation retinal arteriole hierarchy, and empirical human retinal arterial bifurcation benchmarks ($J_{01}\text{–}J_{08}$).

2. **`retinal_vasculature_specimen.jpg`**
   - Human retinal fundus vasculature (*Homo sapiens*), showing normal optic-disc radial arterial/venous bifurcations (panel a) and peripheral microvascular Y-junctions with vessel tortuosity (panel b).
   - Source: [Wikimedia Commons (`File:Retinal_vessel_tortuosity.jpg`)](https://commons.wikimedia.org/wiki/File:Retinal_vessel_tortuosity.jpg), CC BY 4.0.
   - SHA-256: `bc548d4150e070a509caa34cf5cc04d46650fa4fbe15f460f952ca54781dde76`.

3. **`paper_citations.bib`** — BibTeX citations for Murray (1926a, 1926b), Zamir (1976), Sherman (1981) and, since Pass 3, Pries et al. (1992), Fåhraeus & Lindqvist (1931) and Hubbard et al. (1999).

4. **`scout_manifest.json`** — Step 1 (`01_scout`) OpenAlex + Wikimedia Commons scouting dossier.

5. **`fahraeus_lindqvist_retinal_cup_reference.json`** (Pass 3) —
   - Pries, Neuhaus & Gaehtgens (1992) in-vitro Fåhraeus–Lindqvist fit $\eta_{rel}(D) = 220e^{-1.3D} + 3.2 - 2.44e^{-0.06D^{0.645}}$ ($H_D = 0.45$), the normalisation used by the script, and a parity table at $D = 10\ldots1000\,\mu$m.
   - Generalised Murray stationarity with radius-dependent viscosity: $2P_{met} = (4-\epsilon)P_{visc}$, $\epsilon = d\ln\eta/d\ln r$, $Q^*(r) = \tfrac{\pi}{2} r^3\sqrt{b/(\eta(r)(4-\epsilon))}$ (Lean 4 `fahraeus_lindqvist_stationarity_shift`).
   - Hemispherical retinal-cup geometry (globe radius 12 mm, optic disc at 4°, growth limit 95°, four arterial arcades at 45°/135°/225°/315° interdigitated with four venous arcades at 0°/90°/180°/270°, AVR = 0.67), geodesic step and tangent-plane rotation formulas, and the repulsive arteriovenous coupling rule (Lean 4 `arteriovenous_coupled_cube_balance`).
   - DOIs for the three Pass 3 sources were recalled from the standard citations and **not verified online** (no network access during the remediation round).

## Caveats carried for audit

- `empirical_retinal_junctions_zamir_1979` (J01–J08) in `vascular_cube_law_branching_reference.json` carries a `murray_predicted_total_deg` column and observed angles that agree with Murray's prediction to ≤0.3°; the table has no page/figure provenance and Zamir, Medeiros & Cunningham (1979) is not in `paper_citations.bib`. Treat it as an illustrative benchmark, not an independently digitised dataset, until a page-pinned source is vendored.
- `sherman_1981_generational_hierarchy` is generated from $r_g = 82\,\mu\text{m}\cdot 2^{-g/3}$ and $\tau_w = \sqrt{\eta b}$ rather than digitised from Sherman's Table I.
