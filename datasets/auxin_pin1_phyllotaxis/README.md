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

## Contents
- `scout_manifest.json` — OpenAlex & Wikimedia Commons literature/specimen dossier.
- `auxin_pin1_phyllotaxis_reference.json` — Quantitative parameters, 4 phyllotactic mode regimes (Spiral \(137.5^\circ\), Distichous \(180^\circ\), Decussate \(90^\circ\), Tricussate \(120^\circ\)), and NPA/*pin1* block + exogenous IAA microdroplet rescue benchmarks.
- `paper_citations.bib` — BibTeX citations with verified DOIs.
- `rosette_phyllotaxis_specimen.jpg` — *Aeonium tabuliforme* flat Fibonacci spiral rosette (`File:Aeonium tabuliforme 0.jpg`, Wikimedia Commons, CC BY-SA 3.0).
