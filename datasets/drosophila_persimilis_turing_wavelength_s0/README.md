# Drosophila persimilis — Turing wavelength at 2.1 µm (slot 3198)

Holocene (extant), **0 Ma**. Detail scale **2.1 µm** (2.098 × 10⁻⁶ m, log10 = −5.678), tier `3_subcellular_um`.

This slot is the subcellular detail variant of Turing pigment-stripe / ocellus wavelength selection for *Drosophila persimilis* (Pacific Coast sister obscura fly; chromosomal-inversion reproductive isolation). It is not the 117 µm primordium slot (`noor-drosophila-persimilis-turing-wavelength`, slot 2891).

## What is modelled

Schnakenberg (1979) activator–substrate kinetics on a periodic corneal patch, with the Turing critical wavenumber

k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v), λ_c = 2π / k_c

tuned so the focal species sits at **λ = 2.098 µm**. Diffusion length, not reaction scheme, changes along the Drosophila chronoseries at this scale:

| Chronospecies | Role | Body length | λ at 2.1 µm tier |
|---|---|---:|---:|
| *D. paulistorum* | predecessor | 2.36 mm | 1.947 µm |
| *D. persimilis* | focal, 0 Ma | 2.82 mm | **2.098 µm** |
| *D. planitibia* | successor | 5.30 mm | 2.735 µm |

Body lengths are the catalog adult sizes. Wavelength scales as L^0.42 (diffusion length scales more slowly than body length). The exponent is a transfer for this detail tier, not a regression on measured pigment periods of these three species.

## Sources

Crossref-verified primary papers are in `paper_citations.bib`. Noor et al. (2001, 10.1073/pnas.221274498) is the inversion-isolation paper for *D. persimilis*. Dobzhansky & Spassky (1959) is the predecessor semispecies. Kambysellis et al. (1995) is the Hawaiian ecological-shift phylogeny that includes the picture-wing successor. Turing (1952), Schnakenberg (1979), and Gierer & Meinhardt (1972) are the reaction–diffusion sources. Ready, Hanson & Benzer (1976) describe the retinal neurocrystalline lattice whose rhabdomere scale motivates a ~2 µm period. The 2.098 µm figure itself is the assigned tier-3 detail wavelength, not a numeral printed in Noor (2001).

## Specimen plate

Wikimedia Commons returned no CC-licensed bitmap of *D. persimilis* (only unrelated public-domain PDFs). The file `drosophila_persimilis_specimen.jpg` is the repository species-gallery plate `datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_persimilis.jpg`. It is a full-frame golden-brown spiral rendering with a species caption, not a transmission electron micrograph of 2.1 µm pigment granules. SHA-256 is pinned in the reference JSON. The image-fitness terminal GIF frame is locked to this plate; quantitative wavelength claims come from the Schnakenberg field.
