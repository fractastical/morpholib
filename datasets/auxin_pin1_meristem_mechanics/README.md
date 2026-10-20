# Arabidopsis Shoot Apical Meristem Mechanics & Auxin-PIN1-Microtubule Coupling Dataset

Reference parameters, empirical benchmarks, BibTeX citations, and biological specimen micrograph for **Slot 033 (`slot-033-auxin-pin1-meristem-mechanics` / `mirabet-auxin-pin1-meristem-mechanics`)**:

- **Primary Script:** `2008-hamant-meristem-microtubule-stress.py`
- **Lean 4 Module:** `morphoeval/lean/loops/Loop_033_auxin_pin1_meristem_mechanics.lean`
- **Math KG Spec:** `morphoeval/math_kg_loops/mirabet-auxin-pin1-meristem-mechanics.yaml`

## Primary Literature

1. **Hamant, O., Heisler, M. G., Jönsson, H., Krupinski, P., Uyttewaal, M., Bokov, P., Corson, F., Sahlin, P., Boudaoud, A., Meyerowitz, E. M., Couder, Y., & Traas, J. (2008).** Developmental patterning by mechanical signals in *Arabidopsis*. *Science*, 322(5908), 1650–1655. `doi:10.1126/science.1165594`
2. **Heisler, M. G., Hamant, O., Krupinski, P., Uyttewaal, M., Ohno, C., Jönsson, H., Traas, J., & Meyerowitz, E. M. (2010).** Alignment between PIN1 polarity and microtubule orientation in the shoot apical meristem reveals a tight coupling between morphogenesis and auxin transport. *PLoS Biology*, 8(10), e1000516. `doi:10.1371/journal.pbio.1000516`
3. **Jönsson, H., Heisler, M. G., Shapiro, B. E., Meyerowitz, E. M., & Mjolsness, E. (2006).** An auxin-driven polarized transport model for phyllotaxis. *Proceedings of the National Academy of Sciences*, 103(5), 1633–1638. `doi:10.1073/pnas.0509839103`
4. **Mirabet, V., Besnard, F., Vernoux, T., & Boudaoud, A. (2012).** Noise and robustness in phyllotaxis. *PLoS Computational Biology*, 8(2), e1002389. `doi:10.1371/journal.pcbi.1002389`

## Secondary Literature (added in Pass 3 for the 3D multi-layered mesh and division planes)

5. **Louveaux, M., Julien, J.-D., Mirabet, V., Boudaoud, A., & Hamant, O. (2016).** Cell division plane orientation based on tensile stress in *Arabidopsis thaliana*. *PNAS*, 113(30), E4294–E4303. `doi:10.1073/pnas.1600677113` — mechanical division rule (new wall along maximal tension); shoot-apex statistics (65–77 % shortest plane; 81 % symmetric divisions; epidermis ≈ 5 µm thick).
6. **Besson, S., & Dumais, J. (2011).** Universal rule for the symmetric division of plant cells. *PNAS*, 108(15), 6294–6299. `doi:10.1073/pnas.1011866108` — Gibbs-measure plane selection with universal β = 20.6.
7. **Reddy, G. V., Heisler, M. G., Ehrhardt, D. W., & Meyerowitz, E. M. (2004).** Real-time lineage analysis reveals oriented cell divisions associated with morphogenesis at the shoot apex of *Arabidopsis thaliana*. *Development*, 131(17), 4225–4237. `doi:10.1242/dev.01261` — L1/L2 anticlinal clonal layers; corpus divides periclinally and anticlinally.
8. **Kutschera, U., & Niklas, K. J. (2007).** The epidermal-growth-control theory of stem elongation: an old and a new perspective. *Journal of Plant Physiology*, 164(11), 1395–1409. `doi:10.1016/j.jplph.2007.08.002` — thick outer epidermal walls as the load-bearing tensile skin (motivates the L1/inner stiffness ratio of the mesh).

## Vendored Files

- `auxin_pin1_meristem_mechanics_reference.json` — Pressurized shell principal stresses ($\sigma_{\theta\theta}, \sigma_{mm}$) across 4 SAM curvature zones, CMT nematic order parameters, PIN1–CMT co-alignment statistics, laser ablation / lateral compression / oryzalin perturbation benchmarks, and (Pass 3) the `secondary_papers_pass3` and `multilayer_mesh_parameters_pass3` blocks (sheet geometry, stiffness ratio, Lockhart parameters, division rules and the Louveaux 2016 / Reddy 2004 division-plane benchmarks). The SHA-256 was re-pinned in the script after the Pass 3 additions.
- `arabidopsis_sam_meristem_specimen.jpg` — Toluidine-blue stained longitudinal section of *Arabidopsis thaliana* shoot apical meristem dome and emerging lateral primordium with saddle boundary crease (`File:Arabidopsis round SAM.jpg`, Wikimedia Commons, CC BY-SA 4.0). **Unchanged since Pass 1** (SHA-256 `3a8b54b7…`).
- `paper_citations.bib` — BibTeX entries with verified DOIs (5 Pass 1–2 entries + 4 Pass 3 secondary entries).
- `scout_manifest.json` — Step 1 (`01_scout`) OpenAlex + Wikimedia Commons discovery dossier.

## Pass 3 model scope (honest statement)

The Pass 3 "3D multi-layered L1/L2/L3 polyhedral mesh" in `2008-hamant-meristem-microtubule-stress.py` is a **cell-centre Voronoi model** (polyhedral cells = bounded 3D Voronoi regions of the cell centres; wall springs on shared faces; turgor on the exterior L1 faces; Jacobi-relaxed quasi-statics; Lockhart creep; Louveaux-2016 / Besson–Dumais-2011 division planes). It is **not** a finite-element vertex model: there are no vertex degrees of freedom, no wall bending, no cell-volume constraint, and new contacts are inserted stress-free. The mesh reproduces the circumferential maximal-tension direction on the flank and the Lean Law 4 load share, but **not** the radial growth of the thin-shell anisotropy magnitude, and the tension-based division rule **under-produces** the shortest plane (0.38 vs 0.65–0.77). Both failures are reported in the audit rather than tuned away.
