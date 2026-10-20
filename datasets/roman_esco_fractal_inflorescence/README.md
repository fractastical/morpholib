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

## Files and SHA-256 Provenance

- `roman_esco_fractal_inflorescence_reference.json` — Reference parameters, plastochron acceleration sweep series, and genotype morphometrics from Azpeitia et al. (2021) *Science* and Prusinkiewicz et al. (2007) *Science*.
- `paper_citations.bib` — Primary BibTeX citations (`10.1126/science.abg5935`, `10.1126/science.1140429`, `10.1105/tpc.2.8.755`, `10.1126/science.7824951`, `10.1103/PhysRevE.70.046109`).
- `scout_manifest.json` — OpenAlex & Wikimedia Commons scouting dossier for Slot 040.
- `romanesco_brassica_oleracea_specimen.jpg` — High-resolution CC BY-SA 4.0 photograph of *Brassica oleracea* var. *botrytis* (Romanesco) by Ivar Leidus (Wikimedia Commons, `1280x1024`, SHA-256: `5fba6c320b478d6c7243b4c80924582f82cbc05f9c72d11860a21dc31bfeeb7b`).
