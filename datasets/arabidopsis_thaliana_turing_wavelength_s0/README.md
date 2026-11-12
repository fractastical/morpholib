# Arabidopsis thaliana — Turing wavelength at 112 µm (slot 3117)

Holocene extant thale cress (*Arabidopsis thaliana*), chronoseries point 1/1 at **0 Ma**. The modelled observable is the Turing spot wavelength

**λ_c = 111.6 µm = 0.0001116 m** (log10 = −3.952),

tier `5_primordium_100um`, detail variant. This is the characteristic spacing of a trichome-initiation spot on a leaf-primordium epidermis, not the length of the mature branched hair and not a geological chronospecies boundary. The Arabidopsis series in this campaign has no fossil predecessor; what changes across the scale ladder, at the same 0 Ma age, is the number of critical wavelengths that fit in the tissue:

| Scale | Length | Modes along one side | Pattern |
| --- | --- | --- | --- |
| Predecessor (pavement cell) | 18.6 µm = λ_c/6 | 0 | Homogeneous; the cell is subcritical |
| **This slot** | **111.6 µm** on a 446.4 µm primordium patch | **4** | Spot selected by k_c |
| Successor (young leaf) | 1.7856 mm = 16 λ_c | 16 | Same λ_c, more trichomes inserted as the domain grows |

## Sources

- Turing (1952) `10.1098/rstb.1952.0012` — critical wavenumber k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v), λ_c = 2π/k_c.
- Bouyer et al. (2008) `10.1371/journal.pbio.0060141` — TTG1 is trapped by GL3 in incipient trichomes and depleted in the neighbourhood (activator–depletion).
- Balkunde et al. (2020) `10.1016/j.celrep.2020.108497` — the core network combines that depletion with long-range inhibition (CPC).
- Di Patti et al. (2023) `10.1073/pnas.2309616120` — wild-type trichome patterns have fat-tailed power spectra, a signature they read as noise-driven stochastic Turing patterns.

The script integrates a Schnakenberg reduction (activator u = GL1/GL3/TTG1 complex, mobile substrate v = TTG1) with diffusivity ratio D_v/D_u = 8.4 chosen so the law’s λ_c is 111.6 µm. It is not a re-fit of the full Balkunde gene network and it does not grow the three branches of the hair.

## Specimen

`arabidopsis_thaliana_trichome_specimen.jpg` is a 960 px preview of Heiti Paves’ scanning electron micrograph *Müürlooga (Arabidopsis thaliana) lehekarv (trihhoom) 311 0804* (Wikimedia Commons, CC BY-SA 3.0). A second cryo-SEM of the leaf epidermis (Emmanuel Boutet, CC BY-SA 3.0) is vendored as `arabidopsis_thaliana_epidermis_stomata.jpg`. The species-gallery file `Arabidopsis_thaliana.jpg` shows an *Aeonium tabuliforme* rosette and is not used.

SHA-256 digests are recorded in `arabidopsis_thaliana_turing_wavelength_s0_reference.json`.
