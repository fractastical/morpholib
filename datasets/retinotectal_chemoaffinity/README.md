# Prestige & Willshaw (1975) & Brown et al. (2000) Retinotectal Chemoaffinity & Synaptic Competition Dataset

Primary reference dataset and biological specimen metadata for `1976-willshaw-retinotectal-chemoaffinity.py` (Slot 086 of the Project Delphi & MorphoEval Campaign).

## Governing Equations — Prestige & Willshaw (1975) *Proc. R. Soc. B* 190:77-98 & Brown et al. (2000) *Cell* 102:77-88

1. **Counter-Gradient Retinal EphA Receptor & Tectal Ephrin-A Ligand Profiles:**

$$R_A(x_i) = R_0 e^{\alpha x_i} + \Delta R_{\text{EphA3}}\,\mathbb{I}(i \in \text{Isl2}^+), \qquad L_A(y_j) = L_0 e^{\beta y_j}$$

where $x_i \in [0, 1]$ is the nasal-to-temporal retinal ganglion cell (RGC) coordinate, $y_j \in [0, 1]$ is the anterior-to-posterior (rostral-to-caudal) optic tectum / superior colliculus coordinate, and $\Delta R_{\text{EphA3}}$ is the exogenous EphA3 receptor knock-in level expressed on $50\%$ ($\text{Isl2}^+$) of RGCs in Brown et al. (2000).

2. **Mass-Action Reciprocal Chemoaffinity Energy & Topographic Inversion Law:**

$$E_{\text{chem}}(x_i, y_j) = \frac{\kappa_{\text{chem}}}{2}\left(R_A(x_i) L_A(y_j) - \gamma_0\right)^2, \qquad \gamma_0 = R_0 L_0 e^{\alpha}$$

Minimizing $E_{\text{chem}}(x, y)$ yields the exact wild-type inverted retinotectal map:

$$R_A(x) L_A(y^*(x)) = \gamma_0 \implies e^{\alpha(x + y_{\text{WT}}^*(x))} = e^\alpha \implies y_{\text{WT}}^*(x) = 1 - x$$

In *Isl2-EphA3* knock-in mice ($\Delta R_{\text{EphA3}} > 0$), elevated EphA signaling shifts $\text{Isl2}^+$ axons rostrally ($y_{\text{KI}}^*(x) < y_{\text{WT}}^*(x)$), while Group II competition displaces wild-type $\text{Isl2}^-$ axons caudally, duplicating the topographic map in `ki/ki` homozygotes.

3. **Prestige–Willshaw Group II Synaptic Competition & Surgical Compression/Expansion:**

$$\frac{dW_{ij}}{dt} = W_{ij}\left[-E_{\text{chem}}(x_i, y_j) + \kappa_{\text{coop}}(K_{\text{ret}} W K_{\text{tec}})_{ij} - \mu_{\text{ret}}\Big(\sum_k W_{ik} - 1\Big) - \mu_{\text{tec}}\Big(\sum_m W_{mj} - \frac{N_{\text{ret}}}{N_{\text{tec}}^{\text{active}}}\Big)\right]$$

Under caudal half-tectum ablation ($y \in [0, 0.5]$), Group II competition compresses the full retina onto the rostral half-tectum ($y_{\text{comp}}^*(x) = \frac{1}{2}(1-x)$, slope $-0.50$). Under temporal half-retina ablation ($x \in [0, 0.5]$), the surviving nasal half-retina expands across the full tectum ($y_{\text{exp}}^*(x) = 1 - 2x$, slope $-2.00$).

## Files & SHA-256 Provenance

| File | Description | SHA-256 |
|---|---|---|
| `zebrafish_retinotectal_specimen.jpg` | Laser scanning confocal micrograph of 72 hpf *Danio rerio* larval retina showing concentric inner & outer neural rings and radial projections (Wikimedia Commons `File:Concentricity.jpg` by Jaydeep Sidhaye, CC BY-SA 4.0, 1024×1024 px) | `e16ecb3325a48823c83838109780a83b54fb3cb20b89b6d0b2bb378c6495b755` |
| `scout_manifest.json` | Step 1 OpenAlex & Wikimedia Commons scout dossier for Slot 086 | `5ccb8a42580e507024b1ce82344260c035b8ba795974b5c8e1ae3f9913e81935` |
| `paper_citations.bib` | BibTeX entries for Prestige & Willshaw (1975), Willshaw & von der Malsburg (1976), Brown et al. (2000), and Koulakov & Tsigankov (2004) | `aad29a956cd55dc1658f3221ad64ffe187feefc73da44bb813f48bb340e0909e` |
| `retinotectal_chemoaffinity_reference.json` | Canonical EphA/ephrin-A gradient parameters, 6 experimental regimes (WT, `ki/+`, `ki/ki`, half-tectum compression, half-retina expansion, `Ephrin-A2/A5` DKO scrambled control), and SHA-256 locks | verified at runtime |
