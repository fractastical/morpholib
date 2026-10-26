# Pezzulo & Levin (2015) / Friston, Levin et al. (2015) Bioelectric Target Morphology Memory Reference Dataset

Primary reference dataset and biological specimen metadata for `2015-pezzulo-bioelectric-target-morphology-memory.py` (Slot 084 of the Project Delphi & MorphoEval Campaign).

## Governing Equations — Pezzulo & Levin (2015) *J. R. Soc. Interface* 12:20150772 & Friston, Levin, Sengupta & Pezzulo (2015) *J. R. Soc. Interface* 12:20141383

1. **Variational Free Energy of Anatomical Homeostasis:**

$$F(\mu, \mathbf{x}) = \frac{1}{2}\Pi_s \|\mathbf{s}(\mathbf{x}) - \mathbf{g}(\mu)\|^2 + \frac{1}{2}\Pi_p \|\mu - \mu_{\text{target}}\|^2 \ge 0$$

where $\mathbf{x}(t)$ is the current anatomical state (e.g. proximodistal salamander limb segment lengths $\mathbf{x} = (L_{\text{stylopod}}, L_{\text{zeugopod}}, L_{\text{autopod}})$ with total length $L(t) = \sum_i x_i(t)$, or 2D craniofacial/ectopic organ coordinates), $\mu(t)$ is the internal bioelectric representation ($V_{\text{mem}}$ positional belief), $\mu_{\text{target}}$ is the stored bioelectric target morphology setpoint, $\Pi_s \ge 0$ is sensory precision, and $\Pi_p > 0$ is prior setpoint precision.

2. **Coupled Active-Inference Perception & Morphogenetic Action (Least-Action Gradient Flow):**

$$\frac{d\mu}{dt} = -\kappa_\mu \frac{\partial F}{\partial \mu} = \kappa_\mu \Pi_s J_g(\mu)^\top (\mathbf{s}(\mathbf{x}) - \mathbf{g}(\mu)) - \kappa_\mu \Pi_p (\mu - \mu_{\text{target}})$$

$$\frac{d\mathbf{x}}{dt} = -\kappa_a \frac{\partial F}{\partial \mathbf{x}} = -\kappa_a \Pi_s J_s(\mathbf{x})^\top (\mathbf{s}(\mathbf{x}) - \mathbf{g}(\mu))$$

Along trajectories of this coupled gradient flow, Variational Free Energy is a strict Lyapunov function:

$$\frac{dF}{dt} = \nabla_\mu F \cdot \dot{\mu} + \nabla_{\mathbf{x}} F \cdot \dot{\mathbf{x}} = -\frac{1}{\kappa_\mu}\|\dot{\mu}\|^2 - \frac{1}{\kappa_a}\|\dot{\mathbf{x}}\|^2 \le 0$$

3. **Exact Stop Condition & Bioelectric Setpoint Rewriting:**
   - **Equifinal Regeneration & Exact Stop ($L(t) \to L^*$):** Both proximal shoulder amputation ($L_0 = 0.20 L^*$) and distal wrist amputation ($L_0 = 0.70 L^*$) converge to the exact target length $L^* = 1.00$ where $F(\mu^*, \mathbf{x}^*) = 0$ and $\dot{L} = 0$.
   - **Tail-to-Limb / Ectopic Organ Remodeling:** A tail blastema grafted to the flank/shoulder limb field ($\psi_0 = -1.0$) minimizes global body-plan free energy by remodeling into a limb ($\psi(t) \to +1.0, L(t) \to L^*$).
   - **Rewritten Bioelectric Target Memory ($\mu_{\text{target}}'$):** Shifting the bioelectric prior setpoint to $\mu_{\text{target}}' = 1.45$ redirects subsequent regeneration to the new target morphology $L^{*\prime} = 1.45$ without altering genomic sequence.
   - **Denervation / Sensory-Precision Blockade Control ($\Pi_s = 0$):** Setting $\Pi_s = 0$ (nerve transection / sensory deafferentation) yields $\dot{\mathbf{x}} \equiv 0$, arresting regeneration at the amputation stump $L(t) \equiv L_0$.

## Files & SHA-256 Provenance

| File | Description | SHA-256 |
|---|---|---|
| `salamander_limb_regeneration_specimen.jpg` | Whole-mount underwater photograph of *Ambystoma mexicanum* (axolotl) displaying the head, feathery external gills, and articulated right forelimb with stylopod, zeugopod, elbow joint, and autopod digits (Wikimedia Commons `File:Ambystoma mexicanum 1.jpg`, CC BY-SA 3.0, 1200×1079 px) | `fe57ed2c5909e7af46ae0f3d35fb7e2eac68ab1d07e1351c4cd6966bebb6500f` |
| `scout_manifest.json` | Step 1 OpenAlex & Wikimedia Commons scout dossier for Slot 084 | `7d75a04159b0641d96cf0abb85cc5b14e30221edf1f4477204d399b65e53d7e0` |
| `paper_citations.bib` | BibTeX entries for Pezzulo & Levin (2015), Friston, Levin et al. (2015), Pezzulo & Levin (2016), and Vandenberg, Adams & Levin (2012) | `4235fafcb607658185b08942ac4388f24d20972023fa1f3506e22fd9640d34d7` |
| `bioelectric_target_morphology_memory_reference.json` | Canonical Variational Free Energy equations, parameters, 5 experimental regimes, and SHA-256 locks | verified at runtime |
