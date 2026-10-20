# Kriegman, Blackiston, Levin & Bongard (2020, 2021) Xenobot Evolutionary Biomechanics & Kinematic Self-Replication Reference Dataset

Primary reference dataset and biological specimen metadata for `2020-kriegman-xenobot-evolutionary-biobots.py` (Slot 083 of the Infinite Morphospace & MorphoEval Campaign).

## Governing Equations — Kriegman et al. (2020) *PNAS* 117:1853–1859 & Kriegman et al. (2021) *PNAS* 118:e2112672118

1. **Voxel-Based Passive Ectoderm & Rhythmic Contractile Cardiac Spring-Lattice Mechanics:**

$$L_{ij}(t) = L_{ij,0}\bigl(1 + \varepsilon_{\text{act}}\, a_{ij} \sin(2\pi f_{\text{act}} t + \phi_{ij})\bigr), \qquad a_{ij} \in \{0\text{ (ectoderm)},\, 1\text{ (cardiac)}\}$$

$$m_i \frac{d^2 \mathbf{x}_i}{dt^2} + c_{\text{damp}} \frac{d\mathbf{x}_i}{dt} = \sum_{j \in \mathcal{N}(i)} k_{ij}\bigl(\|\mathbf{x}_j - \mathbf{x}_i\| - L_{ij}(t)\bigr)\frac{\mathbf{x}_j - \mathbf{x}_i}{\|\mathbf{x}_j - \mathbf{x}_i\|} + \mathbf{F}_{\text{ground}}(\mathbf{x}_i, \dot{\mathbf{x}}_i)$$

2. **Age-Fitness Pareto Optimization (AFPO) & Phase-Noise Transferability Filter:**

$$F(\mathcal{G}) = \frac{\|\mathbf{x}_{\text{COM}}(T) - \mathbf{x}_{\text{COM}}(0)\|}{L_{\text{body}}}, \qquad F_{\text{robust}}(\mathcal{G}) = \mathbb{E}_{\phi \sim \mathcal{U}(-\pi,\pi)}\bigl[F(\mathcal{G}, \boldsymbol{\phi})\bigr]$$

3. **C-Shaped ("Pac-Man") Kinematic Self-Replication Compaction Law (Kriegman et al. 2021):**

$$\frac{dN_{\text{pile}}}{dt} = \eta_{\text{sweep}}(\theta_{\text{mouth}})\, v_{\text{orbit}}\, \rho_{\text{free}}\, W_{\text{mouth}}, \qquad N_{\text{daughters}} = \left\lfloor \frac{N_{\text{compacted}}}{N_{\text{thresh}}} \right\rfloor$$

## Files & SHA-256 Provenance

| File | Description | SHA-256 |
|---|---|---|
| `xenopus_biobot_specimen.jpg` | Fluorescence + brightfield micrograph of an evolved *Xenopus laevis* Xenobot (`File:Xenobot.jpg`, Wikimedia Commons, CC BY 4.0, 1600×1200 px) showing passive animal-cap ectoderm (tan/brown), ventral concavity indentation, and red-fluorescent contractile cardiac progenitor tissue | `13e063704f07254fd35cc867e7368242cfe27201d452c753342213eaa194d71f` |
| `scout_manifest.json` | Step 1 OpenAlex & Wikimedia Commons scout dossier (`datasets/xenobot_evolutionary_biobots/scout_manifest.json`) | `abbd461fb51acbc9ac8dd8bc9d21a485be1593f35e24019fbbea76b7b1ac7f67` |
| `paper_citations.bib` | BibTeX entries for Kriegman et al. (2020), Kriegman et al. (2021), Blackiston et al. (2021), and Cheney et al. (2013) | `4f92785fb031ce11702e10ade9fe33c2212f58441456984003e3ff7ef37b3833` |
| `xenobot_evolutionary_biobots_reference.json` | Canonical voxel spring-lattice parameters, AFPO fitness definitions, 5 experimental Xenobot regimes, and SHA-256 locks | verified at runtime |
