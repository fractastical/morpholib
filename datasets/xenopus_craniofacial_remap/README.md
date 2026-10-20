# Vandenberg, Adams & Levin (2012) & Pinet et al. (2019) Picasso Tadpole Craniofacial Attractor Remodeling Reference Dataset

Primary reference dataset and biological specimen metadata for `2019-bischof-xenopus-craniofacial-remap.py` (Slot 085 of the Infinite Morphospace & MorphoEval Campaign).

## Governing Equations — Vandenberg, Adams & Levin (2012) *Dev. Dyn.* 241:863-878 & Pinet et al. (2019) *Development* 146:dev175893

1. **Closed-Loop Procrustes Landmark Attractor Dynamics Across Nieuwkoop–Faber Stages $45 \to 65$:**

$$\frac{d\mathbf{x}_i}{dt} = \eta_{\text{MMP}} \left[ -K_i (\mathbf{x}_i - \mathbf{x}_i^*) - \sum_{j=1}^{N} C_{ij}\big((\mathbf{x}_i - \mathbf{x}_j) - (\mathbf{x}_i^* - \mathbf{x}_j^*)\big) \right], \qquad i \in \{\text{L-eye}, \text{R-eye}, \text{L-naris}, \text{R-naris}, \text{jaw}\}$$

where $\mathbf{X}(t) = (\mathbf{x}_1(t), \dots, \mathbf{x}_5(t)) \in \mathbb{R}^{5 \times 2}$ are the Procrustes-normalized 2D craniofacial landmark coordinates of the *Xenopus laevis* tadpole, $\mathbf{X}^* = (\mathbf{x}_1^*, \dots, \mathbf{x}_5^*)$ is the target post-metamorphic froglet craniofacial anatomy at NF stage 65, $K_i > 0$ are landmark-specific target-homing gains, $C_{ij} = C_{ji} \ge 0$ are inter-organ relational spring couplings preserving relative facial proportions, and $\eta_{\text{MMP}} \in [0, 1]$ is matrix metalloproteinase (MMP)-dependent extracellular matrix remodeling mobility.

2. **Procrustes Lyapunov Error Functional & Strict Monotonic Dissipation:**

$$V(\mathbf{X}) = \frac{1}{2}\sum_{i=1}^{N} K_i \|\mathbf{x}_i - \mathbf{x}_i^*\|^2 + \frac{1}{4}\sum_{i,j=1}^{N} C_{ij} \|(\mathbf{x}_i - \mathbf{x}_j) - (\mathbf{x}_i^* - \mathbf{x}_j^*)\|^2 \ge 0$$

$$\frac{dV}{dt} = -\frac{1}{\eta_{\text{MMP}}}\sum_{i=1}^{N} \left\|\frac{d\mathbf{x}_i}{dt}\right\|^2 \le 0, \qquad V(\mathbf{X}) = 0 \iff \mathbf{X} = \mathbf{X}^*$$

3. **Open-Loop Fixed-Displacement Null Hypothesis Failure vs. Closed-Loop Attractor Equifinality:**
   - **Open-Loop Fixed-Displacement Null Model (Vandenberg et al. 2012 Fig. 1A):** If craniofacial metamorphosis were a hardwired vector translation $\dot{\mathbf{x}}_i^{\text{open}} = (\mathbf{x}_i^* - \mathbf{x}_i^{\text{WT}}(0))/T$, any surgical/teratogenic perturbation $\Delta\mathbf{x}_i(0) = \mathbf{x}_i(0) - \mathbf{x}_i^{\text{WT}}(0) \ne \mathbf{0}$ is preserved permanently at stage 65 ($\mathbf{x}_i^{\text{open}}(T) - \mathbf{x}_i^* = \Delta\mathbf{x}_i(0)$).
   - **Closed-Loop Picasso Tadpole Remodeling (Vandenberg et al. 2012 Figs. 2–7):** Under closed-loop target-morphology error minimization, scrambled eyes, nares, and jaws move along novel trajectories with speed proportional to defect magnitude and stop at $\mathbf{X}^*$, restoring bilateral symmetry.
   - **MMP Inhibition Ablation (Pinet et al. 2019):** Blocking matrix metalloproteinase ECM remodeling ($\eta_{\text{MMP}} \to 0.04$, GM6001) arrests craniofacial correction.

## Files & SHA-256 Provenance

| File | Description | SHA-256 |
|---|---|---|
| `xenopus_craniofacial_specimen.jpg` | Dorsal whole-mount photograph of *Xenopus laevis* showing the triangular head contour, snout apex, and bilateral dorsolateral eyes (Wikimedia Commons `File:Xenopus laevis (10.3897-zookeys.1022.51422) Figure 4.jpg`, CC BY 4.0, 1512×1441 px) | `8832f70a57f3d1c4f998ef0b567cddafaf6a885d2c90522327ecbda95f5a68b3` |
| `scout_manifest.json` | Step 1 OpenAlex & Wikimedia Commons scout dossier for Slot 085 | `96534373d259eb281c52914c7f7e9c723d458bc7d96b18ac70376cd93343c73a` |
| `paper_citations.bib` | BibTeX entries for Vandenberg, Adams & Levin (2012), Pinet et al. (2019), Pezzulo & Levin (2016), and Pai et al. (2012) | `f5f4d62c485aca759b843060b76d6af04855e9b073d28a071ecffbdb05e637f7` |
| `xenopus_craniofacial_remap_reference.json` | Canonical Procrustes attractor equations, 5 craniofacial landmarks, 6 experimental regimes, and SHA-256 locks | verified at runtime |
