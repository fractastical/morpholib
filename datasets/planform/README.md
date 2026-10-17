# PlanformDB 2.5.0 & Bioelectric Patterning Fixtures

This directory holds reference metadata, SHA-256 digests, and notice-compliant derived fixtures linking **PlanformDB 2.5.0** (Lobo, Malone & Levin 2013) to planarian bioelectric anterior–posterior (AP) patterning models (`2013-lobo-planform-morphology.py`, `2021-Cervera–Levin–Mafe.py`).

## Canonical Database Pin

- **File**: `planformDB_2.5.0.edb` (SQLite 3 database distributed by the Lobo Lab at <https://lobolab.umbc.edu/planform/>)
- **SHA-256 (`planformDB.sha256`)**: `9effd13ddb87664b9ef7a9b6c9c1959b502fcdd6c1b06ee016501b2d0be83b89`
- **Notice**: `"All contents of Planform, Copyright 2012-2016 Daniel Lobo. Use for research and educational purposes is free of charge, provided proper citation is given to the database and the functional work that gave rise to the datasets used."`

## Derived Bioelectric Fixtures (`MesmerPrism/rusty-matter`)

Vendored from [`MesmerPrism/rusty-matter`](https://github.com/MesmerPrism/rusty-matter) at commit `ad111fa7ecfd8f43d292ff3b1d27d3af2741d828` (dual-licensed `MIT OR Apache-2.0`, preserving the Lobo Lab PlanformDB 2.5.0 notice):

| File | SHA-256 | Upstream Path | Description |
| :--- | :--- | :--- | :--- |
| `rusty_matter_planformdb_derived_v0.json` | `c50e51dc7489b54d370ee0c6a1835f00ab5789dc54af432d369d9bead6162a11` | `fixtures/fields/planformdb-derived-v0.json` | 14 curated `PlanformDB 2.5.0` result sets from Oviedo et al. (2010, `doi:10.1016/j.ydbio.2009.12.012`): octanol gap-junction blockade by AP crop position (`Exps 415–419`), triple innexin RNAi (`Dj-Inx-12` + `Dj-Inx-5/13`) by AP crop position (`Exps 446–450`), and VNC disruption + octanol timing window (`Exps 441–444`). |
| `rusty_matter_planarian_traces.json` | `842e11b933d4d0c15d7642f35fddfd7e53eb5efcbd93a5fc667b0c198f5125e9` | `fixtures/fields/planarian-traces.json` | Deterministic reference traces for the 5-band planarian AP bioelectric circuit (`Baseline`, `TransverseCutWound`, `GapBlock`, `TransientDepolarizationMemory`, `TransientDepolarizationNoMemoryControl`). |
| `rusty_matter_source_dynamics_targets.json` | `de1dba3ebe4a560894e5441538e1019f2f549994e66cef5096e20636c179a8da` | `fixtures/fields/source-dynamics-targets.json` | Literature checkpoint targets for Durant et al. (2019, `doi:10.1016/j.bpj.2019.01.029`), Oviedo et al. (2010), Beane et al. (2011, 2013), Emmons-Bell et al. (2015), and Lobo, Malone & Levin (2013). |

## Claim Boundaries

- **Empirical records (`rusty_matter_planformdb_derived_v0.json`)**: Report exact experiment IDs, specimen counts ($n$), observation times, and morphology frequencies (`#1 Wild Type` vs. `#2 Double head, two pharynges`) from `PlanformDB 2.5.0`.
- **Mechanistic models (`2021-Cervera–Levin–Mafe.py` and `rusty_matter_planarian_traces.json`)**: Compare two qualitative/semi-quantitative mathematical abstractions of gap-junction-mediated AP polarity:
  1. **Cervera, Levin & Mafe (2021)** 1D reaction–diffusion morphogen wavefront shift $x^*(f_2)$ under gap-junction diffusivity reduction $D_X = f_2 D_{X,0}$, and
  2. **`MesmerPrism/rusty-matter` 5-band bioelectric circuit** with voltage-difference-gated gap-junction conductance, hysteresis pattern memory $M_i$, and Head/Tail readout layers (plus a memory-disabled negative control).
- Neither mechanistic abstraction claims exact biophysical voltage calibration in millivolts or direct gene-regulatory parameter fitting beyond the cited papers.
