# Clock-and-Wavefront Somite Segmentation Dataset (`datasets/clock_and_wavefront/`)

Reference parameters, formal equations, and biological specimen photography for **Slot 021 (`slot-021-clock-and-wavefront`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing **Cooke & Zeeman (1976)** *"A clock and wavefront model for control of the number of repeated structures during animal morphogenesis"* (*J. Theor. Biol.* 58(2):455–476, DOI: `10.1016/S0022-5193(76)80131-2`).

## Governing Equations

1. **Somite Wavelength Law:**
   \[
   S = v \, T_c
   \]
   where \(v\) is the speed of the anteriorly-to-posteriorly regressing maturation wavefront and \(T_c\) is the autonomous intracellular segmentation-clock period.
2. **Somite Count & Size-Invariance Scaling Law:**
   \[
   N = \frac{L}{S} = \frac{L}{v \, T_c}, \qquad v = \frac{L}{T_{\text{total}}} \implies N = \frac{T_{\text{total}}}{T_c}
   \]
   explaining Cooke's (1975) experimental discovery in *Xenopus laevis* that surgically reduced miniature embryos (\(L' = \alpha L\) with scaled wavefront speed \(v' = \alpha v\)) form the exact same number of somites \(N\) as normal embryos, with each somite scaled to length \(S' = \alpha S\).
3. **Thom–Zeeman Cusp Catastrophe Wavefront Dynamics:**
   \[
   \tau_\chi \frac{\partial \chi(x,t)}{\partial t} = -\bigl(\chi^3 - a(x,t)\,\chi - b_{\text{osc}}(x,t)\bigr)
   \]
   where \(a(x,t) = \tanh\!\bigl((v t - x)/\sigma_w\bigr)\) is the regressing positional wavefront bifurcation parameter and \(b_{\text{osc}}(x,t) = b_0 + A_c \cos(2\pi t / T_c - k_\phi x)\) is the intracellular segmentation clock gating catastrophic epithelial somite cleft formation.

4. **Goldbeter–Gonze–Pourquié (2007) FGF8/RA Bistable Determination Front (Pass 3):**
   \[
   \tau_{FR}\frac{dF}{dt} = \alpha_F\,M_F(x,t)\frac{K^n}{K^n+R^n} - F, \qquad
   \tau_{FR}\frac{dR}{dt} = \alpha_R\frac{K^n}{K^n+F^n} - R, \qquad
   M_F(x,t) = e^{-(x_{tb}(t)-x)/\lambda_F},\; \lambda_F = v_{tb}/k_{dM}
   \]
   with the tailbud at \(x_{tb}(t) = x_{tb0} + v_{tb} t\) (Dubrulle & Pourquié 2004 fgf8 mRNA decay gradient). The toggle is bistable for \(M_F \in [M_{lo}, M_{hi}]\); tailbud-derived FGF-high cells switch irreversibly to the RA-high state when \(M_F < M_{lo}\), so the determination front \(x_w(t) = x_{tb}(t) - \lambda_F\ln(1/M_{lo})\) regresses at exactly the elongation speed, \(dx_w/dt = v_{tb}\). The cusp parameters of Law 3 are read off the toggle, \(a = \tanh((R-F)/\kappa_a)\), \(m = F^4/(F^4+K_{mob}^4)\), so **the wavefront velocity \(v\) of Law 1 is no longer prescribed**. An ectopic FGF8 bead (Dubrulle et al. 2001) is modelled as a decaying Gaussian fgf8 source.
5. **Anterior/Posterior Half-Somite Cleft Rule & Farhadifar (2007) Vertex Sheet (Pass 3):**
   \[
   E = \sum_c\Bigl[\tfrac{K_A}{2}(A_c-A_0)^2 + \tfrac{\Gamma}{2}P_c^2\Bigr] + \sum_e \Lambda_e \ell_e, \qquad
   \Lambda_e = \Lambda_0 + \Delta\Lambda\,[\chi_{ant} < 0 < \chi_{post}]
   \]
   The alternating \(\chi = \pm 1\) compartments are anterior/posterior half-somites (Saga & Takeda 2001); only posterior→anterior interfaces are loaded with extra line tension, so \(2N\) compartments give \(N\) somite blocks and \(N-1\) internal clefts (Lean theorem `meinhardt_1986_half_somite_cleft_count`). On the 2D sheet the lattice zig-zag cleft rotates toward the mediolateral axis (28.2° → 10.3°, ΔΛ = 0 control 27.3°; SEM specimen clefts 7.9°) and straightens (tortuosity 1.30 → 1.03). Not modelled: T1 transitions, apical constriction / rosette formation, 3D epithelialisation, Xenopus 90° cell-block rotation.

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`0beabd3277c705e62b894794a2448ec9572377059c72a185a6148e2f17f0b327`)
- `clock_and_wavefront_reference.json` — Formal parameterization and Cooke (1975, 1976) empirical *Xenopus* somite size-invariance table
- `somite_embryo_specimen.jpg` — Dorsal scanning electron micrograph of a vertebrate embryo showing the central neural tube and bilateral segmented somite rows (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons)
- `fgf_ra_vertex_reference.json` — Pass 3 reference table: reduced Goldbeter (2007) FGF8/RA toggle parameters and derived front quantities, \(v_{tb}\) sweep, Dubrulle (2001) bead perturbation, Farhadifar (2007) vertex-sheet parameters and reference values (Bi et al. 2015 shape index 3.81; specimen cleft angle 7.9°); SHA-256 recorded at runtime by `verify_dataset_integrity()`
- `paper_citations.bib` — BibTeX citations for Cooke & Zeeman (1976), Cooke (1975), Palmeirim et al. (1997), Goldbeter et al. (2007), Dubrulle & Pourquié (2004), Dubrulle et al. (2001), Farhadifar et al. (2007), Bi et al. (2015) and Saga & Takeda (2001)

## Reproduction

```bash
./venv/bin/python3 1976-cooke-clock-and-wavefront.py --self-test          # ~5 s, 6 assertion groups, 13 measurements
./venv/bin/python3 1976-cooke-clock-and-wavefront.py                      # PNG + vertex-sheet PNG + GIF (~10 s)
./venv/bin/python3 1976-cooke-clock-and-wavefront.py --summary --wavefront-mode fgf_ra --v-tailbud 3.3333
```
