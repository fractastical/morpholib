# Phyllotaxis Reference Dataset (`datasets/phyllotaxis/`)

Primary mathematical specifications, empirical *Helianthus annuus* parastichy survey tables, and Creative Commons specimen photographs used by `1979-vogel-phyllotaxis.py`.

## Primary Literature & Data Sources

1. **Vogel, H. (1979).** *A better way to construct the sunflower head.* Mathematical Biosciences, 44(3–4), 179–189. DOI: [10.1016/0025-5564(79)90080-4](https://doi.org/10.1016/0025-5564(79)90080-4)
   - Planar generative spiral model:
     $$r_n = c\sqrt{n}, \qquad \theta_n = n\,\alpha_0, \qquad \alpha_0 = \frac{360^\circ}{\tau^2} \approx 137.507764^\circ$$
2. **Douady, S., & Couder, Y. (1992, 1996).** *Phyllotaxis as a physical self-organized growth process.* Physical Review Letters, 68(13), 2098–2101. DOI: [10.1103/PhysRevLett.68.2098](https://doi.org/10.1103/PhysRevLett.68.2098); Journal of Theoretical Biology, 178(3), 255–273. DOI: [10.1006/jtbi.1996.0024](https://doi.org/10.1006/jtbi.1996.0024)
   - Self-organized Hofmeister inhibitory dynamics on a radially advecting meristem ring controlled by dimensionless plastochron ratio $G = V T / R_0$.
3. **Swinton, J., Ochu, E., & The MSI Turing's Sunflower Consortium (2016).** *Novel Fibonacci and non-Fibonacci structure in the sunflower: results of a citizen science experiment.* Royal Society Open Science, 3(5), 160091. DOI: [10.1098/rsos.160091](https://doi.org/10.1098/rsos.160091) (Dryad dataset DOI: [10.5061/dryad.f9k77](https://doi.org/10.5061/dryad.f9k77))
   - Empirical distribution of 768 photoreviewed parastichy counts and 368 parastichy pairs across 657 *Helianthus annuus* capitula, plus historical Schoute (1938, $N=319$) and Weisse (1897, $N=140$) surveys.

## Files & Cryptographic Provenance

| File | Description | License | SHA-256 |
|---|---|---|---|
| `phyllotaxis_reference.json` | Machine-readable equations, Fibonacci-type sequences, Swinton et al. (2016) Tables 1–4, and named specimens | CC BY 4.0 | Verified by `1979-vogel-phyllotaxis.py --self-test` |
| `paper_citations.bib` | BibTeX entries for Vogel (1979), Douady & Couder (1992, 1996), and Swinton et al. (2016) | CC0 | — |
| `helianthus_annuus_capitulum_specimen.jpg` | Real *Helianthus annuus* capitulum spiral seedhead (`File:Sunflower Spiral.jpg`, 1000×750 px) | CC BY-SA 2.0 | `c76483eea3e564fee9cf41913e03fc0874ea95b23e82aa901f5d083d074546ff` |
| `helianthus_parastichy_annotated.jpg` | Annotated *Helianthus annuus* parastichy spiral photograph (`File:Goldener Schnitt Bluetenstand Sonnenblume.jpg`, Dr. Helmut Haß, Koblenz, 286×231 px) | CC BY-SA 3.0 | `7cd34354e23584cbc558f459c8eb3b1418092067b1f624b1b381a7eb8915f00e` |
