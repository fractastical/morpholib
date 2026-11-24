# Anomalocaris canadensis — Turing wavelength at 24 µm (508 Ma)

Slot 3113 of the 10,000-Animal Basic Look. One animal, one scale, one horizon.

- **Animal:** *Anomalocaris canadensis* Whiteaves, a radiodont apex predator. Aristotelian path II. Anaima (Invertebrata) → Entoma (Arthropoda) → Anomalocaris.
- **Scale:** 24 µm detail tier (`4_cellular_10um`). Exact slot length **2.397×10⁻⁵ m** (23.97 µm, log10 = −4.620). This is the critical Turing wavelength λ_c = 2π/k_c of a sub-facet ocellus / screening-pigment field, not the ommatidial facet diameter.
- **Age:** Middle Cambrian, Burgess Shale, Wuliuan, **508 Ma**. Chronoseries index 1/1 for this species. Lineage bracket: Trilobite Segmentation & Eye Fossil Series.

## What this scale is

Paterson et al. (2011, doi:10.1038/nature10689) describe paired eyes 2–3 cm across from the Emu Bay Shale (~515 Ma), with at least 16,000 hexagonally packed lenses in one eye. A hemisphere of diameter 2.5 cm tiled by 16,000 hexagons has a center spacing of about 266 µm. The 23.97 µm mode selected here is roughly one-eleventh of that reconstructed facet spacing. It sits inside a facet, at cellular scale, where an activator–inhibitor system can pick a pigment-spot wavelength.

Burgess Shale compressions of *A. canadensis* (Daley & Edgecombe 2014, doi:10.1666/13-067) preserve stalked eyes under a small oval carapace. They do not resolve a 24 µm pigment lattice. The museum photograph vendored here is a **cast** (TMP 2023.003.0003, previously ROMIP 51211; Commons description: length about 20 cm), not a thin section of corneal cuticle.

## What changes relative to the lineage brackets

There is no second *A. canadensis* chronospecies in this slot.

- **Predecessor (Emu Bay Shale, ~515 Ma).** The published optical observable is the ommatidial lattice (eye diameter 2–3 cm, ≥16,000 lenses). The reconstructed facet spacing is hundreds of micrometres. This slot’s k_c is a different, shorter mode.
- **This node (Burgess Shale, 508 Ma).** λ_c = 2.397×10⁻⁵ m on a no-flux corneal patch. Stalked eyes are present; the 24 µm pigment wavelength is a model scale, not a counted lens row.
- **Successor (Fezouata Biota, Early Ordovician, 488–472 Ma).** Van Roy & Briggs (2011, doi:10.1038/nature09920) report anomalocaridids larger than any Cambrian body. A tissue-local Turing wavelength does not grow with body length: k_c stays put if the diffusivities and the Jacobian stay put. No 24 µm eye pigment is described for those giants.

Lubensky et al. (2011, doi:10.1073/pnas.1015302108) show that the *Drosophila* R8 lattice is a switch-and-template mechanism, not a Turing instability. This script does not transfer that Drosophila circuit onto *Anomalocaris*. It uses Turing’s dispersion relation and Schnakenberg kinetics as the quantitative law named for this slot (Turing wavenumber k_c; form class stripes, spots, and ocelli).

## Reproduce

```bash
python3 models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py
```

Outputs:

- `docs/media/anomalocaris_canadensis_turing_wavelength_s0_comparison.png`
- `docs/media/anomalocaris_canadensis_turing_wavelength_s0_growth.gif`

Specimen image: `anomalocaris_canadensis_TMP2023_003_0003_tyrrell.jpg`, CC BY-SA 4.0, Chris Woodrich, 2025-07-13, Royal Tyrrell Museum. SHA-256 is pinned in `anomalocaris_canadensis_turing_wavelength_s0_reference.json`.
