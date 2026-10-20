# Echinoderm Bilateral-to-Pentaradial Hydrocoel Ring Symmetry Breaking & Lovén's Law Reference Dataset (`datasets/echinoderm_pentaradial_symmetry/`)

## Overview

This dataset vendors the empirical reference measurements, governing mathematical equations, literature citations, and biological specimen imagery for **Slot 075 (`slot-075-echinoderm-pentaradial-symmetry`)** of the 100-Loop Infinite Morphospace & MorphoEval Campaign, modeled by `2005-meinhardt-echinoderm-pentaradial-symmetry.py`.

## Primary Literature Sources

1. **Morris, V. B. (2012).** *Homology and the evolution of echinoderm pentaradiality.* Evolution & Development **14**(4): 313–326. DOI: [`10.1111/j.1525-142X.2012.00552.x`](https://doi.org/10.1111/j.1525-142X.2012.00552.x)
   - Formulates the developmental model in which metamorphosis from a bilateral echinoderm larva (pluteus) to a pentaradial adult rudiment proceeds through closure of the left hydrocoel crescent into a circular hydrocoel ring, producing five primary podia (ambulacral rays) in the sequential order governed by Lovén's Law.
2. **Mooi, R. & David, B. (1998).** *Evolution Within a Bizarre Phylum: Homologies of the First Echinoderms.* American Zoologist **38**(6): 965–974. DOI: [`10.1093/icb/38.6.965`](https://doi.org/10.1093/icb/38.6.965)
   - Establishes the **Extraxial-Axial Theory (EAT)** of echinoderm body-plan evolution, demonstrating that the five pentaradial ambulacra arise strictly from the axial growth zone templated by the hydrocoel ring via ocular plate addition (the Ocular Plate Rule, OPR), whereas interambulacral and aboral plates belong to perforate/imperforate extraxial body wall.
3. **Lovén, S. (1874).** *Études sur les Échinoïdées.* Kongliga Svenska Vetenskaps-Akademiens Handlingar **11**(7): 1–91. DOI: [`10.5962/bhl.title.46385`](https://doi.org/10.5962/bhl.title.46385)
   - Discovers **Lovén's Law**: in all crown-group echinoids, when the five ambulacral rays are numbered counter-clockwise in oral view as $\mathrm{I, II, III, IV, V}$ (with interambulacrum $5$ posterior and ambulacrum $\mathrm{III}$ anterior) and each biserial ambulacrum is divided into half-ambulacra $(a, b)$ following the counter-clockwise numbering, the enlarged primary peristomial plate follows the invariant asymmetric pattern **$\mathrm{I}b, \mathrm{II}b, \mathrm{III}a, \mathrm{IV}a, \mathrm{V}a$** (`ba-ab-ba-ab-ab` around the ring from $\mathrm{I}$ to $\mathrm{V}$), reflecting bilateral initiation along the closing hydrocoel crescent in order $(\mathrm{II, IV, I, III, V}) \equiv (\mathrm{C, A, D, B, E})$.
4. **Duboc, V., Röttinger, E., Lapraz, F., Besnardeau, L. & Lepage, T. (2005).** *Left-right asymmetry in the sea urchin embryo is regulated by Nodal signaling on the right side.* Developmental Cell **9**(1): 147–158. DOI: [`10.1016/j.devcel.2005.05.008`](https://doi.org/10.1016/j.devcel.2005.05.008)
   - Proves that right-sided *Nodal* expression in *Paracentrotus lividus* and *Strongylocentrotus purpuratus* pluteus larvae restricts hydrocoel rudiment formation to the left side ($1 \times 5 = 5$ primary podia); morpholino knockdown of *Nodal* derepresses the right coelomic pouch, producing **twin bilateral pentaradial rudiments** ($2 \times 5 = 10$ primary podia), while bilateral *Nodal* overexpression eliminates rudiment formation ($0$ podia).
5. **Meinhardt, H. (2008).** *Models of biological pattern formation: from elementary steps to the organization of embryonic axes.* Current Topics in Developmental Biology **81**: 1–63. DOI: [`10.1016/S0070-2153(07)81001-5`](https://doi.org/10.1016/S0070-2153(07)81001-5)
   - Formalizes Gierer–Meinhardt activator–inhibitor reaction–diffusion on closed circular rings of radius $R_{\text{ring}}$, where periodic boundary conditions quantize angular wavenumbers to $k_m = m / R_{\text{ring}}$ (Laplacian eigenvalues $\mu_m = m^2 / R_{\text{ring}}^2$), selecting mode $m^* = 5$ ($\Delta\theta = 72^\circ$) when $2\pi R_{\text{ring}} / \lambda_c \in (4.5, 5.5)$.

## Vendored Files & SHA-256 Provenance

| File | Description | License | SHA-256 |
| :--- | :--- | :--- | :--- |
| `echinoderm_pentaradial_test_specimen.jpg` | *Tripneustes gratilla* regular echinoid test (aboral view, 720x505 JPEG) showing 5 lavender-purple ambulacral rays at $72.0^\circ \pm 1.4^\circ$ spacing | CC BY-SA 4.0 | `953b2f7e380296278a4a7b9152db50ed9e55d93ff61ddfd3b6b169390df0a399` |
| `echinoderm_pentaradial_symmetry_reference.json` | Governing equations, hydrocoel ring mode staircase ($m=3..7$), Lovén's Law sequence, Duboc Nodal L-R perturbations, and specimen calibration | CC BY 4.0 | Verified at runtime |
| `scout_manifest.json` | Step 1 OpenAlex & Wikimedia Commons scout dossier for Slot 075 | CC BY 4.0 | `0b333ebe2a2ed5246cc828f7d6844de2a10b8a0d11c95720a243cbd9285605ee` |
| `paper_citations.bib` | BibTeX citations for Morris (2012), Mooi & David (1998), Lovén (1874), Duboc et al. (2005), and Meinhardt (2008) | CC BY 4.0 | `fd436211809ea80d0d039ec1e52e6279bbc1b10ef1a11639f5e73e47a56c2e0d` |
