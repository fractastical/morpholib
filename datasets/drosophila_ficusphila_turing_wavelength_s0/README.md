# *Drosophila ficusphila* — Turing wavelength at 2.1 µm (slot 3185)

Detail-scale reaction–diffusion model for the longitudinal sex-comb tooth array of *Drosophila ficusphila* (Fig-Breeding Oriental Sophophora Fly).

- **Scale:** 2.1 µm (2.072×10⁻⁶ m, log10 = −5.684), tier `3_subcellular_um` (tier 3 of 8). This is the critical Turing wavelength λ_c of the tooth-ridge / pigment-stripe field, not the ~100 µm primordium-scale sibling slot.
- **Age:** Holocene (Extant), **0 Ma**. Project chronoseries 14/41. Lineage label: Sophophora suzukii–biarmipes Wing-Spot Clade.
- **Predecessor / successor (comparative series of extant species, not a stratigraphic ancestor chain):** *Drosophila eugracilis* (reduced tarsal sex-comb teeth) and *Drosophila grimshawi* (Hawaiian picture-wing lace pigmentation).

## What this scale changes

At 2.072 µm the packed longitudinal comb selects a shorter critical wavelength than either neighbour in the series. The scenario (reconstructed in the script, not a published SEM table) holds the Schnakenberg Jacobian fixed and changes only activator diffusivity, equivalent to a linear tooth/pigment density:

| species | role at 0 Ma | density relative to *D. ficusphila* | λ_c |
|---|---|---|---|
| *D. eugracilis* | predecessor | 2/3 (reduced teeth) | 3.108 µm |
| *D. ficusphila* | focal | 1 | 2.072 µm |
| *D. grimshawi* | successor | 1/2 (coarser picture-wing pigment cells) | 4.144 µm |

Those diffusivities are not the α/β power-law parameters of the 100 µm ficusphila, eugracilis, or grimshawi sibling scripts.

## Sources

Sex-comb development: Tanaka, Barmina & Kopp (2009) PNAS, doi:10.1073/pnas.0807875106; Tanaka et al. (2011) PLoS Biology, doi:10.1371/journal.pbio.1001131; Kopp (2011) Evol. Dev., doi:10.1111/j.1525-142X.2011.00507.x; Kopp & True (2002) Evol. Dev., doi:10.1046/j.1525-142x.2002.02017.x.

Wavelength law: Turing (1952) Phil. Trans. R. Soc. B, doi:10.1098/rstb.1952.0012, instantiated with Schnakenberg (1979) kinetics, doi:10.1016/0022-5193(79)90042-0.

## Specimen image

Wikimedia Commons returned no bitmap for this species during scouting. The pinned plate is the repository gallery file `datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_ficusphila.jpg` (CC-licensed gallery asset), copied here as `drosophila_ficusphila_specimen.jpg`. It is a stylized sepia spiral pigment field with a 100 µm scale bar, not an SEM of a sex-comb tooth. SHA-256 is recorded in the reference JSON. The model does not claim the plate is a micrograph at 2.1 µm.

## Run

```bash
python3 models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py
```
