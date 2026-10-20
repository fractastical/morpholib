# Venus Flytrap Snap-Buckling Dataset (`datasets/venus_flytrap_snap_buckling/`)

Reference dataset and biological specimen assets for **Slot 066 (`forterre-venus-flytrap-snap-buckling`)**:

- **Primary Landmark Paper:** Forterre, Y., Skotheim, J. M., Dumais, J., & Mahadevan, L. (2005). *How the Venus flytrap snaps.* **Nature**, 433(7024), 421–425. DOI: [`10.1038/nature03185`](https://doi.org/10.1038/nature03185)
- **Foundational Poroelastic Speed Limit Paper:** Skotheim, J. M., & Mahadevan, L. (2005). *Physical limits and design principles for plant and fungal movements.* **Science**, 308(5726), 1308–1310. DOI: [`10.1126/science.1107976`](https://doi.org/10.1126/science.1107976)
- **Review of Plant Biomechanics & Elastic Instabilities:** Forterre, Y. (2013). *Slow, fast and furious: understanding the physics of plant movements.* **Journal of Experimental Botany**, 64(15), 4745–4760. DOI: [`10.1093/jxb/ert230`](https://doi.org/10.1093/jxb/ert230)
- **Follow-up Snapping Mechanics Study:** Sachse, R., Westermeier, A., Mylo, M., Nadasdi, J., Bischoff, M., Speck, T., & Poppinga, S. (2020). *Snapping mechanics of the Venus flytrap (Dionaea muscipula).* **PNAS**, 117(27), 16035–16042. DOI: [`10.1073/pnas.2002707117`](https://doi.org/10.1073/pnas.2002707117)

## Biological Specimen (`dionaea_muscipula_trap_lobe_specimen.jpg`)

- **Organism:** *Dionaea muscipula* (Venus flytrap open bilobed leaf trap showing doubly-curved anthocyanin-red inner lobes, central yellowish-green midrib hinge, three sensory trigger hairs per lobe, and marginal interlocking cilia teeth)
- **Source:** Wikimedia Commons (`File:Venus Flytrap showing trigger hairs.jpg`)
- **License:** CC BY-SA 2.5

## Pass 3 additions (`trigger_hair_ap_reference.json`)

Pass 3 closes the Pass 2 remediation item *"couple the doubly-curved shallow shell snap-buckling dynamics to a spatially resolved finite-element model of trigger hair mechanosensitive ion-channel action potential propagation across the lobe"* with an honest **reduced** electro-mechanical model (Section 3b of `2005-forterre-venus-flytrap-snap-buckling.py`):

1. **Trigger-hair mechanotransduction** — Boltzmann open probability of the stretch-activated FLYC1 (MscS-like) conductance vs hair bending force in micronewtons (Scherzer et al. 2019; Procko et al. 2021) → receptor potential clamped on the hair footprint.
2. **P1 finite-element action potential** — monodomain cable equation with FitzHugh–Nagumo kinetics on a deterministic linear-triangle mesh of the lobe planform (≈2 000 nodes), IMEX time stepping; validated against the exact Nagumo front speed `c = sqrt(2D/τ_v)(1/2 − a)` (McKean 1970), h-refinement, and a `v ∝ D^0.5` ablation sweep; lobe conduction velocity lands in the 6–20 cm/s range of Hodick & Sievers (1988) / Hedrich & Neher (2018).
3. **Two-touch Ca²⁺ counting memory** — per-node Ca²⁺ integrator (Böhm et al. 2016; Suda et al. 2020): one AP is subthreshold, two APs within `Δ_max = τ_Ca ln(c/(1−c))` reach the motor threshold.
4. **Coupling to the shell** — the lumped-mass fraction `A(t)` of Ca²⁺-activated lobe area gates the active rest curvature `κ_x0(t)` of the Pass 2 shell solver; closure latency after the second touch is emergent.

`trigger_hair_ap_reference.json` records the model constants (provenance `model`) separately from recalled literature values (provenance `literature-approx` / `theory-exact`). **All eight new DOIs were recalled offline and are not verified online** (flagged `doi_verified_offline: false` and `note = {DOI unverified offline (Pass 3)}` in `paper_citations.bib`). The SHA-256-pinned specimen image and `venus_flytrap_snap_buckling_reference.json` are unchanged.

Known limitations of the reduced model: flat planform mesh (electrical problem not solved on the curved mid-surface); FitzHugh–Nagumo instead of multi-channel Hodgkin–Huxley kinetics; 0-D hair mechanics; `D`, `τ_v`, `c_Ca`, `τ_Ca`, hair positions and footprint radius are modelling choices.

## Files

- `README.md`
- `paper_citations.bib` (4 Pass 1/2 + 8 Pass 3 entries)
- `scout_manifest.json`
- `venus_flytrap_snap_buckling_reference.json` (Pass 1/2, SHA-256 pinned, unchanged)
- `trigger_hair_ap_reference.json` (Pass 3, SHA-256 pinned)
- `dionaea_muscipula_trap_lobe_specimen.jpg` (SHA-256 pinned, unchanged)
