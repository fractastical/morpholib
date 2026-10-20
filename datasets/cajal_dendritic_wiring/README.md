# Ramón y Cajal Minimum-Wiring-Length & Conduction-Delay Dendritic Trees Dataset (`datasets/cajal_dendritic_wiring/`)

Reference dataset and empirical Ramón y Cajal Purkinje neuron dendritic arbor specimen for **MorphoEval Slot 058 (`caleo-cajal-dendritic-wiring`)**, implemented in `2007-caleo-cajal-dendritic-wiring.py` and formally verified in `morphoeval/lean/loops/Loop_058_cajal_dendritic_wiring.lean`.

## Primary Literature & DOIs

1. **Cuntz, H., Forstner, F., Borst, A., & Häusser, M. (2010).** One Rule to Grow Them All: A General Theory of Neuronal Dendritic Trees. *PLoS Computational Biology*, 6(8), e1000877. DOI: [`10.1371/journal.pcbi.1000877`](https://doi.org/10.1371/journal.pcbi.1000877)
2. **Cuntz, H., Borst, A., & Segev, I. (2007).** Optimization principles of dendritic structure. *Proceedings of the National Academy of Sciences*, 104(33), 13319–13324. DOI: [`10.1073/pnas.0704731104`](https://doi.org/10.1073/pnas.0704731104)
3. **Cuntz, H., Forstner, F., Haag, J., & Borst, A. (2008).** A Quantitative Description of Sensory Neuron Organization in *Drosophila*: From Individual Dendritic Trees to the Population. *PLoS Computational Biology*, 4(12), e1000251. DOI: [`10.1371/journal.pcbi.1000251`](https://doi.org/10.1371/journal.pcbi.1000251)
4. **Rall, W. (1959).** Branching dendritic trees and motoneuron membrane resistivity. *Experimental Neurology*, 1(5), 491–527. DOI: [`10.1016/0014-4886(59)90046-9`](https://doi.org/10.1016/0014-4886(59)90046-9)
5. **Wen, Q., & Chklovskii, D. B. (2008).** A Cost-Benefit Analysis of Neuronal Morphology. *Journal of Neurophysiology*, 99(5), 2320–2328. DOI: [`10.1152/jn.00280.2007`](https://doi.org/10.1152/jn.00280.2007)
6. **Ramón y Cajal, S. (1899).** *Textura del Sistema Nervioso del Hombre y de los Vertebrados*. Madrid: Nicolás Moya. DOI: [`10.5962/bhl.title.48637`](https://doi.org/10.5962/bhl.title.48637)

## Files & Integrity

- `scout_manifest.json`: Literature and Wikimedia Commons scout dossier for Slot 058.
- `cajal_dendritic_wiring_reference.json`: Reference parameters and quantitative benchmarks for Cuntz et al. (2010) greedy MST balancing factor $bf \in [0, 1]$ Pareto trade-off between total wiring length $W$ and total soma-to-synapse path length $P$, 2D ($W \propto N^{1/2} A^{1/2}$) and 3D ($W \propto N^{2/3} V^{1/3}$) scaling laws, Rall's $3/2$ diameter power law ($d_{\text{parent}}^{3/2} = \sum_k d_{\text{daughter},k}^{3/2}$), and empirical specimen morphometrics.
- `paper_citations.bib`: BibTeX citations with verified DOIs.
- `cajal_purkinje_dendritic_arbor_specimen.jpg`: Santiago Ramón y Cajal's (1899) Golgi-stained drawing of a human cerebellar Purkinje cell (`File:Cajal - a purkinje neuron from the human cerebellum.jpg`, Wikimedia Commons, Public Domain, 720×1024 px, SHA-256: `6255a396655b56c44c7be2d07b5b2d4fc3da21b9ec8b169602dad21a25542a37`).
- `electrotonic_ca_spike_reference.json` (**Pass 3**, SHA-256: `adf04c13baa46a5e1a456871556d87d1ef028c4ce79dd2637e93c3e2dbf6b5f5`): passive-cable constants, Morris–Lecar hotspot parameters, synaptic protocol, analytic checks and registered model predictions for the coupled passive-cable / active Ca²⁺ hotspot coincidence-detection analysis (see below).

## Pass 3 Addendum — Rall Passive Cable Electrotonics Coupled to an Active Dendritic Ca²⁺ Spike Hotspot

The Pass 2 audit (`2026-10-19-cajal-dendritic-wiring-audit-v2`) carried one remediation item: *couple passive Rall 3/2 electrotonic cable segments to active dendritic Ca²⁺ spike conductance equations to evaluate synaptic coincidence detection across varying Cuntz balancing factors*. Pass 3 closes it with the deliberately reduced model recorded in `electrotonic_ca_spike_reference.json` and implemented in Section 2b of `2007-caleo-cajal-dendritic-wiring.py`:

1. **Passive electrotonics (Rall 1959; Koch 1999 Eq. 2.37).** Every Rall-3/2 segment $e$ of the Cuntz tree receives an electrotonic length $X_e = l_e / \lambda_f(d_e)$ with $\lambda_{DC}(d) = \sqrt{R_m d / 4 R_i}$ and the sinusoidal steady-state reduction $\lambda_f = \lambda_{DC}\sqrt{2/(1+\sqrt{1+(\omega\tau_m)^2})}$ at the alpha-synapse frequency $f_{syn} = 1/(2\pi\tau_{syn}) = 53.05$ Hz. Passive constants ($R_m = 122\,\text{k}\Omega\,\text{cm}^2$, $R_i = 115\,\Omega\,\text{cm}$, $C_m = 0.77\,\mu\text{F/cm}^2$) follow Roth & Häusser (2001) Purkinje fits, **transcribed offline and not re-verified against the paper in this pass** (flagged in the JSON `note`). Lean 4 Theorem 5 (`koch_ac_length_constant_bound`) proves $\lambda_f \le \lambda_{DC}$.
2. **Active hotspot (Morris & Lecar 1981).** Two synaptic inputs on a nearest-neighbour terminal-tip pair are attenuated by $\exp(-X)$ to their lowest-common-ancestor branch node, which hosts a single isopotential Morris–Lecar Ca²⁺/K⁺ compartment (canonical Hopf parameter set) standing in for the all-or-none Purkinje dendritic Ca²⁺ spike of Llinás & Sugimori (1980). This is a **single-compartment stand-in, not a Purkinje P-type channel fit nor a multi-compartment active cable**.
3. **Coincidence detection (Polsky, Mel & Schiller 2004 protocol).** Supralinearity gain (paired vs single peak), coincidence window $\Delta t_c$ (largest inter-input delay still firing the hotspot), and same-branch fraction are swept across the eight Cuntz balancing factors already grown for the Pareto sweep. The passive ablation ($g_{Ca} = 0$, K⁺ gate frozen) reproduces exact linear superposition (gain = 2.000000; Lean 4 Theorem 4 `rall_passive_membrane_linear_superposition`); the active hotspot yields gain ≈ 4.3 at $bf = 0.20$ and the window collapses from ≈ 5.2 ms ($bf = 0.2$) to ≈ 0.7 ms in the $bf = 1$ radial-star limit, where no tip pair shares a sub-branch hotspot.

Reference values in the JSON `coincidence_detection_predictions` block are **model predictions / analytic nulls** (linear gain 2, same-branch fraction 0 at $bf = 1$, negative sign of window vs $bf$), not empirical literature values; the synaptic peak current $I_{peak}$ is a protocol choice tuned so that single inputs stay sub-threshold. Residual for Pass 4: fit the hotspot to Llinás–Sugimori P-type Ca²⁺ spike kinetics and resolve it as a multi-compartment active cable.

### Additional Pass 3 Literature (DOIs not re-verified online in this offline pass)

7. **Roth, A., & Häusser, M. (2001).** Compartmental models of rat cerebellar Purkinje cells based on simultaneous somatic and dendritic patch-clamp recordings. *J. Physiol.*, 535(2), 445–472. DOI: `10.1111/j.1469-7793.2001.00445.x`
8. **Morris, C., & Lecar, H. (1981).** Voltage oscillations in the barnacle giant muscle fiber. *Biophys. J.*, 35(1), 193–213. DOI: `10.1016/S0006-3495(81)84782-0`
9. **Llinás, R., & Sugimori, M. (1980).** Electrophysiological properties of in vitro Purkinje cell dendrites in mammalian cerebellar slices. *J. Physiol.*, 305, 197–213. DOI: `10.1113/jphysiol.1980.sp013358`
10. **Polsky, A., Mel, B. W., & Schiller, J. (2004).** Computational subunits in thin dendrites of pyramidal cells. *Nat. Neurosci.*, 7(6), 621–627. DOI: `10.1038/nn1253`
11. **Koch, C. (1999).** *Biophysics of Computation: Information Processing in Single Neurons.* Oxford University Press (Ch. 2, Eq. 2.37; no DOI).
