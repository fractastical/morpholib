# `datasets/joint_interzone_clefting/` — Synovial joint interzone clefting & phalangeal Inhibitory Cascade

Reference data vendored for `2016-onuff-joint-interzone-clefting.py` (MorphoEval slot 018,
sim_id `onuff-joint-interzone-clefting`). Every file is SHA-256 pinned in the script's
`EXPECTED_SHA256` registry and verified fail-closed on each run (`verify_dataset_integrity()`).

| File | Added | Content |
|---|---|---|
| `joint_interzone_clefting_reference.json` | Pass 1/2 | Kavanagh et al. (2013) 12-taxon cross-clade phalanx morphospace, BMP-GDF5-Noggin RD baseline parameters, 6 genetic/signalling perturbation conditions, specimen digit-ray annotations |
| `tetrapod_digit_interzone_specimen.jpg` | Pass 1 | Hand radiograph (Wikimedia Commons `Polydactyly_01_Rhand_AP.jpg`, CC BY-SA 3.0), 552 x 640 px; Digits II-IV segmented into P1/P2/P3 with MCP/PIP/DIP radiolucent clefts. **The subject is juvenile: the physes are open, so a second shallower radiolucent minimum (the growth plate) lies ~10-12 px distal to each joint space.** The SHA-256-pinned specimen must never be removed or replaced. |
| `scout_manifest.json` | Pass 1 | Scout dossier for the slot |
| `paper_citations.bib` | Pass 1, extended Pass 3 | BibTeX for all cited sources (Pass 3 appended Drachman & Sokoloff 1966, Kahn 2009, Roddy 2011, Carter & Wong 1988, Edwards 1994, Dowthwaite 1998, Mow 1980, Maroudas 1976) |
| `synovial_cavitation_reference.json` | Pass 3 | Reduced Biot poroelastic cavitation / articular morphogenesis model: equations, dimensionless baseline parameters, dimensional scaling note, three literature cavitation conditions (motile WT, paralysed / muscle-less, Nog-/-), and the specimen joint-line arch protocol |

## Pass 3 additions (2026-10-19, audit v3)

The Pass 2 audit asked for the 2D GDF5-Noggin reaction-diffusion digit ray to be coupled to a
fluid-poroelastic synovial cavitation and articular morphogenesis solver. The reduced model
implemented in `simulate_poroelastic_cavitation()` is:

- hyaluronan synthesis by Gdf5+ interzone cells at a mechanosensitive rate
  `dH/dt = k_H phi(m_load) G - delta_H H + D_H lap(H)` (Dowthwaite 1998);
- Biot storage equation for excess pore pressure with a linearised Donnan osmotic source and free
  perichondrial drainage `S dp/dt = kappa_p lap(p) + gamma_osm (beta_osm H - p)`, `p = 0` outside
  the rod (Mow 1980; Maroudas 1976);
- interzone cohesion `sigma_coh = sigma_0 (1 - kappa_G G)` (Edwards 1994) and an irreversible cavity
  phase field `dc/dt = r_c max(p - sigma_coh, 0)(1 - c)`;
- a Carter & Wong (1988) articular arch `a(y) = A_arc phi(m_load)(1 - (y/W)^2)` of the joint line
  whose sagitta vanishes under immobilisation (Roddy 2011);
- Kahn (2009) chondrogenic reversion of unloaded interzone cells.

Both the cavitation gate (`MathOfBiology.Loop_018.poroelastic_cavitation_requires_load_and_interzone`)
and the Darcy lens law (`MathOfBiology.Loop_018.darcy_lens_cavity_midline_maximum`) are machine-verified
in `morphoeval/lean/loops/Loop_018_joint_interzone_clefting.lean`.

### Claim boundary

The solver is 2D, one-way coupled (RD -> poroelastic), with a rigid solid skeleton (storage equation
only; no `u-p` Biot displacement field, effective stress, or capsule). Dimensional embryonic interzone
poroelastic constants are not available in the cited literature; only adult cartilage scales (Mow 1980,
Maroudas 1976) are recorded. The specimen joint-line arch is measured on a juvenile radiograph with
+/-9 px lateral offsets; Digit II's distal half has insufficient bone contrast and is excluded by the
contrast gate.
