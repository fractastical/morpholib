# Romanesco & Cauliflower Fractal Inflorescence Reference Dataset (`slot-040-roman-esco-fractal-inflorescence`)

## Overview

This dataset vendors the quantitative gene-regulatory network (GRN) parameters, plastochron acceleration scaling benchmarks, Fibonacci parastichy contact numbers, and empirical specimen photograph for **Azpeitia et al. (2021) *Science* 373(6551):192-197** (*"Cauliflower fractal forms arise from perturbations of floral gene networks"*) and **Prusinkiewicz et al. (2007) *Science* 316(5830):1452-1456** (*"Evolution and development of inflorescence architectures"*).

## Governing Mechanism

1. **Wild-Type vs `ap1 cal` Floral GRN Bistability Collapse**:
   - In wild-type *Arabidopsis thaliana* and *Brassica oleracea*, an auxin maximum at the shoot apical meristem (SAM) flank transiently induces **LEAFY (`LFY`)**, which activates **APETALA1 (`AP1`)** and **CAULIFLOWER (`CAL`)**. `AP1/CAL` engage in positive auto-feedback and strongly repress the shoot-identity gene **TERMINAL FLOWER 1 (`TFL1`)**, locking the primordium into a determinate flower.
   - In `ap1 cal` loss-of-function mutants ($g_{\text{AP1}} = 0$), the transient `LFY` spike fails to lock in floral fate (`AP1/CAL = 0`), but suffices to activate the MADS-box genes **SUPPRESSOR OF OVEREXPRESSION OF CONSTANS 1 (`SOC1`)**, **AGAMOUS-LIKE 24 (`AGL24`)**, and **FRUITFULL (`FUL`)**, which maintain a persistent **"floral memory"** state (`SOC1+ / TFL1+`) that suppresses leaf/bract outgrowth while reverting the lateral primordium into a new inflorescence shoot meristem that reiterates spiral phyllotaxis ($\phi = 2\pi(2 - \varphi) \approx 137.508^\circ$) across 4-6 hierarchical orders.
2. **Constant vs Accelerating Plastochron (`Cauliflower` vs `Romanesco`)**:
   - **Cauliflower** ($\beta = 0$): Plastochron $\Delta t_k = \Delta t_0$ is constant ($r_p(t) = r_0$), so primordium count grows linearly $N_{\text{Caul}}(T) = r_0 T$ and meristems pack into a flat hemispherical dome ($H/R \approx 0.56$, Fibonacci parastichies $(8, 13)$, 3D box-counting fractal dimension $D_f \approx 2.38$).
   - **Romanesco** ($\beta > 0$): Plastochron decreases linearly with meristem age ($\Delta t(t) = \frac{\Delta t_0}{1 + \beta t}$, rate $r_p(t) = r_0 + \beta t$), so cumulative primordia grow quadratically $N_{\text{Rom}}(T) = r_0 T + \frac{1}{2}\beta T^2$. Rapidly accumulating primordia combined with vertical internode elevation sculpt acute conical logarithmic spirals ($H/R \approx 1.48$, higher-order Fibonacci parastichies $(13, 21)$, 3D box-counting fractal dimension $D_f \approx 2.71$).

3. **PIN1 Polar Auxin Transport on the Epidermal (L1) Mesh (added in Pass 3)**:
   - A reduced cell-based model (Jönsson et al. 2006 *PNAS*; Smith et al. 2006 *PNAS*) is laid over the curved dome / cone surface: PIN1 is allocated up the auxin gradient, $P_{ij} = P_{\text{tot}} A_j / (\kappa + \sum_k A_k)$, auxin maxima emerge, a peripheral-zone cell above threshold initiates a primordium that then acts as an auxin sink and is advected outward by surface growth. The emergent initiation rate scales with the free perimeter of the initiation ring, $r_p(R) = kR$ (Douady & Couder 1992 Hofmeister rule), so a linearly growing meristem $R(t) = R_0 + v_r t$ gives Law 2's $\beta = k v_r$. The `transport_T = 0` ablation (Reinhardt et al. 2003 *pin1* / NPA naked pin) yields no primordia.
   - This is **not** a continuum finite-element solve on the manifold; parameters, geometry and limitations are in `pin1_auxin_transport_reference.json`.

## Provenance caveats (Pass 3 audit)

- `plastochron_beta_sweep_benchmark_series` and `genotype_morphometric_benchmarks` are Pass 1 **interpolations/syntheses** anchored on the two Azpeitia et al. (2021) end-points (Cauliflower vs Romanesco); they are not tables published in the paper. Agreement of the prescribed H/R envelope with this series is a self-consistency check.
- `radial_expansion_rate_v_r = 0.16` and the internode `dz` values are Pass 1 vendored values not traced to a specific figure.
- The Kim (2004) *Phys. Rev. E* 70:046109 entry (source of $D_f \approx 2.71 / 2.38$) could not be independently re-verified in Pass 3 and is flagged provisional in `paper_citations.bib`.
- The Pass 1 `kempin1995` BibTeX entry mixed two different Kempin et al. (1995) papers; fixed in Pass 3.

## Files and SHA-256 Provenance

- `roman_esco_fractal_inflorescence_reference.json` — Reference parameters, plastochron acceleration sweep series, and genotype morphometrics anchored on Azpeitia et al. (2021) *Science* and Prusinkiewicz et al. (2007) *Science* (see caveats above).
- `pin1_auxin_transport_reference.json` — (Pass 3) PIN1 epidermal-mesh model parameters, cone geometries, literature anchors (Jönsson 2006, Smith 2006, Reinhardt 2003, Douady & Couder 1992, Smyth 1990) and the coupling statement $\beta = k v_r$.
- `paper_citations.bib` — Primary BibTeX citations (`10.1126/science.abg5935`, `10.1126/science.1140429`, `10.1105/tpc.2.8.755`, `10.1126/science.7824951`, `10.1103/PhysRevE.70.046109`, `10.1073/pnas.0509839103`, `10.1073/pnas.0510457103`, `10.1038/nature02081`, `10.1103/PhysRevLett.68.2098`).
- `scout_manifest.json` — OpenAlex & Wikimedia Commons scouting dossier for Slot 040.
- `romanesco_brassica_oleracea_specimen.jpg` — High-resolution CC BY-SA 4.0 photograph of *Brassica oleracea* var. *botrytis* (Romanesco) by Ivar Leidus (Wikimedia Commons, `1280x1024`, SHA-256: `5fba6c320b478d6c7243b4c80924582f82cbc05f9c72d11860a21dc31bfeeb7b`).
