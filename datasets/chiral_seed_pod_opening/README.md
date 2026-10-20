# Chiral Seed Pod Opening Dataset (`datasets/chiral_seed_pod_opening/`)

Reference dataset and biological specimen assets for **Slot 065 (`liang-chiral-seed-pod-opening`)**:

- **Primary Landmark Paper:** Armon, S., Efrati, E., Kupferman, R., & Sharon, E. (2011). *Geometry and Mechanics in the Opening of Chiral Seed Pods.* **Science**, 333(6050), 1726–1730. DOI: [`10.1126/science.1203874`](https://doi.org/10.1126/science.1203874)
- **Foundational Incompatible Shell Theory:** Efrati, E., Sharon, E., & Kupferman, R. (2009). *Elastic Theory of Unconstrained Non-Euclidean Plates.* **Journal of the Mechanics and Physics of Solids**, 57(4), 762–775. DOI: [`10.1016/j.jmps.2008.12.004`](https://doi.org/10.1016/j.jmps.2008.12.004)
- **Follow-up Shape Selection Study:** Armon, S., Arbel, H., Moshe, M., & Sharon, E. (2014). *Shape Selection in Chiral Self-Shaping Latex Strips.* **Soft Matter**, 10(16), 2733–2740. DOI: [`10.1039/c3sm52313f`](https://doi.org/10.1039/c3sm52313f)

## Biological Specimen (`bauhinia_chiral_seed_pod_specimen.jpg`)

- **Organism:** *Bauhinia variegata* (Orchid tree dehisced chiral helical seed pod valves, intact flat pre-dehiscence seed pod, and seeds)
- **Source:** Muséum de Toulouse (`MHNT.BOT.2011.3.22`), Wikimedia Commons (`File:Bauhinia variegata MHNT.BOT.2011.3.22.jpg`), photograph by Roger Culos
- **License:** CC BY-SA 3.0

## Files

| File | Role | Pinned in script |
|---|---|---|
| `chiral_seed_pod_opening_reference.json` | Pass 2 pitchfork / cutting-angle / 4-system benchmark tables | yes (SHA-256) |
| `bauhinia_chiral_seed_pod_specimen.jpg` | biological specimen (never modified) | yes (SHA-256) |
| `scout_manifest.json` | source scouting manifest | yes (SHA-256) |
| `paper_citations.bib` | BibTeX for all cited sources (Pass 3 added 4 entries) | yes (SHA-256) |
| `suture_fracture_ballistic_reference.json` | **Pass 3** suture-fracture / ballistic-ejection parameter table with per-constant `provenance` flags | yes (SHA-256) |

## Pass 3 addition — suture fracture & ballistic seed ejection (`suture_fracture_ballistic_reference.json`)

Closes the Pass 2 remediation item *"couple the quasi-static non-Euclidean shell energy minimizer to transient elastodynamic fracture propagation along the pod suture line to model ballistic seed ejection velocities"* with an honest **reduced 1-D model** (not a 3-D elastodynamic FEM):

1. **Energy coupling.** The Pass 2 L-BFGS-B minimizer gives the relaxed valve energy `e_final(κ₀, w̃)`; the closed flat pod stores `e_init = κ₀²`. With the Efrati et al. (2009) bending prefactor `B_eff = Y h³ / (12(1+ν))` the released energy density is `Δe = B_eff (κ₀² − e_final)` and the suture energy release rate is `G = Δe W / h` (≈ 278 J/m² for the fully dried default pod).
2. **Desiccation drive.** `κ₀(t) = κ₀,f (1 − e^{−t/τ_dry})` (τ_dry = 72 h, `provenance: model`).
3. **Transient fracture.** Freund (1990) mode-I equation of motion `da/dt = c_R max(0, 1 − Γ_c(a)/G(t))` over a seeded heterogeneous suture toughness field `Γ_c(a)` (mean 120 J/m², CV 0.15, `provenance: model`), giving Griffith initiation, stick-slip jumps and arrest.
4. **Seed launch.** Elastic-catapult partition `v_seed = √(2η ΔU_seg / (m_seg/3 + m_seed))` (Vogel 2005), valve coil time from the first cantilever flexural mode, drag-ballistic range (RK4).
5. **Benchmarks** (all `claim_boundary: transfer_boundary`, DOIs *unverified offline*): *Cardamine hirsuta* ≈ 10 m/s (Hofhuis et al. 2016 Cell), *Hura crepitans* ≈ 70 m/s (Swaine & Beer 1977 New Phytol.). No published *Bauhinia variegata* launch-speed or dispersal-distance measurement was available offline, so the predicted ≈ 2.8 m/s / ≈ 1.7 m range is an unvalidated model prediction.

Every constant in the JSON carries `provenance: source | derived | model`; the `model` ones (ρ, Γ_c, CV_Γ, a₀, τ_dry, n_seeds, m_seed, r_seed, C_d, η, launch angle/height) are free parameters that the audit should treat as such.
