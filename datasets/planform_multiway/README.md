# Multiway Hypergraph Rewriting & McKay IR Canonicalization for PlanformDB 2.5.0 and LimbformDB 2.1.0

This directory holds the SHA-256-locked reference fixture (`planform_multiway_reference.json`) and citations linking [`WolframInstitute/HypergraphRewritingEngine`](https://github.com/WolframInstitute/HypergraphRewritingEngine) (`v1.0.0-rc2`, Assar 2026) to the empirical planarian and limb regeneration databases **PlanformDB 2.5.0** (Lobo, Malone & Levin 2013) and **LimbformDB 2.1.0** (Lobo et al. 2014).

## SHA-256 Provenance Pins

- **`planform_multiway_reference.json`**: `a5f598a0e09e047d5e9b1232a2c2f9caddd19e1a0680071911079f102089d06b` (`167,672` bytes)
- **`WolframInstitute__HypergraphRewriteEngine-1.0.0.paclet` (`v1.0.0-rc2`)**: `0c8406c412c8999837ded884b1d6d9340405933eda47e5ceca29852a22e16db9`
- **`MacOSX-ARM64/hg_evolve` (`commit 8a326ddf9b8d05aba8b1d290e832d3257abd0cf7`)**: `74c2014eb08515c341695a2a31e0219c255e338c4e7cf42c1a9f062857b42b88`
- **`planformDB_2.5.0.edb`**: `9effd13ddb87664b9ef7a9b6c9c1959b502fcdd6c1b06ee016501b2d0be83b89` (`744,448` bytes)
- **`limbformDB_2.1.0.ldb`**: `9c26bf3930bcc225f900f414cbd3a2ad3ac2f4d99e694d389b4fc36de15099f0` (`897,024` bytes)
- **`datasets/planform/planform_bioelectric_series_v1.json`**: `4e18934438eb666e701418c0710b5eeedd49fa15f654718fd0e9635b2776284d`

## Lobo Lab Notice

All contents of Planform and Limbform, Copyright 2012-2016 Daniel Lobo. Use for research and educational purposes is free of charge, provided proper citation is given to the databases (`doi:10.1093/bioinformatics/btt088`, `doi:10.1242/bio.20123400`, `doi:10.1093/bioinformatics/btu582`, `doi:10.1002/reg2.13`) and the functional work that gave rise to the datasets used. This derived fixture contains only McKay IR canonical hypergraph isomorphism classes, hashes, and aggregate topological counts; raw SQLite database rows, region radii, and morphology images remain in the upstream `.edb`/`.ldb` files.
