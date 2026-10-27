# Drosophila sechellia — 1.5 µm Turing wavelength (slot 3202, detail)

Holocene (extant) **0 Ma**. Phenomenon scale **1.5 µm** (1.548×10⁻⁶ m, log10 = −5.810), tier `3_subcellular_um`, detail variant `s0`.

This slot is the subcellular detail of pigment-granule / ocellar-pigment wavelength selection in *Drosophila sechellia*. It is not the 88 µm cellular tergite model in `datasets/drosophila_sechellia_turing_wavelength/` (slot 2711).

## What is modelled

Turing (1952) two-species reaction–diffusion on a periodic patch of ocellar pigment epithelium. The classical critical wavenumber

```
k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v),    λ_c = 2π / k_c
```

is placed at **λ_c = 1.500 µm** by the *D. sechellia* Jacobian (activator gain f_u = 1.600, inhibitor induction g_u = 4.500, near onset h_min = −0.035). Inhibitor diffusivity is eight times the activator diffusivity. A cubic saturation −u³ bounds the activator.

## What changes relative to the chronoseries neighbours

All three species are Holocene (0 Ma). The comparison is at **this** 1.5 µm detail scale, with shared diffusivities and a shared near-onset depth. Only activator gain and the co-adapted inhibitor induction change.

| Chronospecies | Role | f_u | g_u | λ_c |
| --- | --- | --- | --- | --- |
| *D. santomea* | predecessor | 1.327 | 3.568 | 1.921 µm |
| *D. sechellia* | this slot (31/41) | 1.600 | 4.500 | 1.500 µm |
| *D. serrata* | successor | 1.458 | 4.002 | 1.680 µm |

*D. santomea* (Jeong et al. 2008, tan cis-regulatory loss) has the lower activator gain and the longer detail-scale pitch. *D. sechellia* (Sucena & Stern 2000; McGregor et al. 2007; Frankel et al. 2011) sits at the short-wavelength end: loss of the shavenbaby larval-trichome template leaves the ocellar ommochrome lattice at 1.500 µm. *D. serrata* (Chenoweth & Blows 2005) is intermediate. These Jacobians are not the power-law stubs used by the 88–102 µm sibling scripts.

Shoup (1966) measured pigment-granule development in *D. melanogaster*, not a 1.5 µm spacing in *D. sechellia*. The 1.5 µm figure is this slot’s assigned detail scale and the wavelength the dispersion relation is built to select. The ovary micrograph is an anatomical anchor, not a claim that ovarian nuclei are that lattice.

## Specimen

`drosophila_sechellia_ovary_specimen.jpg` — Wikimedia Commons, *Drosophila sechellia* ovary stained with a nucleic-acid dye, CC BY 2.0.

- File page: https://commons.wikimedia.org/wiki/File:Drosophila_sechellia_ovary_stained_with_nucleic_acid_specific_dye_(16378644002).jpg
- SHA-256: `5a765a91794b001a5efe1674294da4bb94bae5791d2d21db6d732bfb9d8faaff`

## Run

```bash
python3 models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py
```

Outputs:

- `docs/media/drosophila_sechellia_turing_wavelength_s0_comparison.png`
- `docs/media/drosophila_sechellia_turing_wavelength_s0_growth.gif`

Lean: `morphoeval/lean/loops/Loop_3202_drosophila_sechellia_turing_wavelength_s0.lean` (`MathOfBiology.Loop_3202`).
