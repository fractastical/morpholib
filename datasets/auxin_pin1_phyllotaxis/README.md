# Auxin–PIN1 Polarized Transport & Phyllotaxis Dataset (`datasets/auxin_pin1_phyllotaxis`)

## Overview
Reference dataset and empirical spiral rosette specimen for **Slot 032 (`jonsson-auxin-pin1-meristem` / `2006-jonsson-auxin-pin1-meristem.py`)**, formalizing the landmark up-the-gradient PIN1 polar auxin transport models of shoot apical meristem (SAM) phyllotaxis:

1. **Jönsson, Heisler, Shapiro, Meyerowitz & Mjolsness (2006)** — *An auxin-driven polarized transport model for phyllotaxis*, *PNAS* 103(5):1633–1638 (`10.1073/pnas.0509839103`).
2. **Smith, Guyomarc'h, Mandel, Reinhardt, Kuhlemeier & Prusinkiewicz (2006)** — *A plausible model of phyllotaxis*, *PNAS* 103(5):1301–1306 (`10.1073/pnas.0510457103`).
3. **Reinhardt et al. (2003)** — *Regulation of phyllotaxis by polar auxin transport*, *Nature* 426:255–260 (`10.1038/nature02081`).

## Governing Equations

- **Up-the-Gradient Boltzmann PIN1 Membrane Allocation:**
  \[
  P_{ij} = \frac{P_i \exp(c_1 a_j)}{\sum_{k \in N(i)} \exp(c_1 a_k)}, \qquad \sum_{j \in N(i)} P_{ij} = P_i, \qquad \frac{P_{ij}}{P_{ik}} = \exp(c_1 (a_j - a_k))
  \]
- **Intercellular Active Polar Auxin Transport vs. Passive Diffusion:**
  \[
  \frac{da_i}{dt} = \rho_0 - \mu a_i + T \sum_{j \in N(i)} \left(P_{ji} a_j - P_{ij} a_i\right) + D \sum_{j \in N(i)} (a_j - a_i)
  \]
- **Turing-Like Up-the-Gradient Instability Threshold:**
  Around the homogeneous steady state \(a^* = \rho_0 / \mu\) on a \(z\)-coordinated L1 lattice (\(P_{ij}^* = P_i / z\)), linear perturbation \(\delta a_j \propto e^{i \mathbf{k}\cdot\mathbf{x}_j}\) yields effective active anti-diffusion when:
  \[
  T \frac{P_i}{z} (c_1 a^* - 1) > D \quad \Longleftrightarrow \quad c_1 a^* > 1 + \frac{z D}{T P_i}
  \]

## Pass 3 additions (PIN1 trafficking kinetics & subepidermal canalization)

4. **Geldner, Friml, Stierhof, Jürgens & Palme (2001)** — *Auxin transport inhibitors block PIN1 cycling and vesicle trafficking*, *Nature* 413:425–428 (`10.1038/35096571`) — BFA exocytosis block / PIN1 internalisation.
5. **Paciorek et al. (2005)** — *Auxin inhibits endocytosis and promotes its own efflux from cells*, *Nature* 435:1251–1256 (`10.1038/nature03633`) — auxin-inhibited endocytosis variant.
6. **Bayer et al. (2009)** — *Integration of transport-based models for phyllotaxis and midvein formation*, *Genes Dev* 23:373–384 (`10.1101/gad.497009`) — epidermal up-the-gradient + subepidermal with-the-flux coupling.
7. **Mitchison (1980, 1981)** — canalization hypothesis (`10.1098/rspb.1980.0015`, `10.1098/rstb.1981.0154`).
8. **Feugier, Mochizuki & Iwasa (2005)** — flux-based carrier dynamics with Hill feedback (`10.1016/j.jtbi.2005.03.017`).
9. **Scarpella, Marcos, Friml & Berleth (2006)** — PIN1-marked preprocambial strands 1–2 cells wide (`10.1101/gad.1402406`).

- **Explicit PIN1 exocytosis/endocytosis kinetics (Lean 4 Law 4):**
  \[
  \frac{dP_{ij}}{dt} = k_{exo} f(a_j) P^{cyt}_i - k_{endo} P_{ij}, \qquad P^{cyt}_i + \sum_j P_{ij} = P_{tot}, \qquad
  P^*_{ij} = \frac{k_{exo} f(a_j) P_{tot}}{k_{endo} + k_{exo}\sum_k f(a_k)}
  \]
  (`pin1_trafficking_kinetics` block: \(k_{exo} = 3.0\), \(k_{endo} = 1.2\ \mathrm{h^{-1}}\); BFA sets \(k_{exo}=0\)).
- **With-the-flux canalization (Lean 4 Law 5):** \(f_{ij} = f_0 + J_{ij}^n/(K^n + J_{ij}^n)\) on a 15 × 12 subepidermal slice beneath one primordium (`subepidermal_canalization` block). This is a **quasi-3D reduction** (one vertical slice, no growth), not a full 3D meristem.

## Contents
- `scout_manifest.json` — OpenAlex & Wikimedia Commons literature/specimen dossier.
- `auxin_pin1_phyllotaxis_reference.json` — Quantitative parameters, 4 phyllotactic mode regimes (Spiral \(137.5^\circ\), Distichous \(180^\circ\), Decussate \(90^\circ\), Tricussate \(120^\circ\)), NPA/*pin1* block + exogenous IAA microdroplet rescue benchmarks, and (schema 1.1.0) the `pin1_trafficking_kinetics` and `subepidermal_canalization` parameter blocks. Note: the mode divergence angles and the WT ring profile are benchmark *targets* that the script currently prescribes rather than derives (see audit v3 remediation).
- `paper_citations.bib` — BibTeX citations with verified DOIs.
- `rosette_phyllotaxis_specimen.jpg` — *Aeonium tabuliforme* flat Fibonacci spiral rosette (`File:Aeonium tabuliforme 0.jpg`, Wikimedia Commons, CC BY-SA 3.0).
