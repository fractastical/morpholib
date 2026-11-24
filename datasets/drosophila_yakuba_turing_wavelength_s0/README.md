# Drosophila yakuba — subcellular Turing wavelength (slot 3212)

Holocene (extant), **0 Ma**. Phenomenon scale **2.2 µm** (`2.157e-06 m`, log10 = −5.666, tier `3_subcellular_um`, detail variant).

This slot is the terminus (41/41) of the Drosophila chronoseries in the Sophophora melanogaster-subgroup radiation (0–12 Ma). The predecessor chronospecies is *Drosophila willistoni*. There is no successor.

## What changes at this scale and age

*D. willistoni* (willistoni group, Neotropical) is carried in this repository at the **cellular** pigment-band tier, λ = 86 µm (`8.63e-05 m`, slot 2931). *D. yakuba* (melanogaster subgroup, Afrotropical savanna; yellow/tan abdominal melanic gradation) is scored here one tier down: the same Schnakenberg activator–substrate law selects a **subcellular** melanin-granule / ocellus wavelength

λ\* = 2.157 µm.

The wavenumber ratio against the predecessor’s cellular band is

k_yakuba / k_willistoni = 8.63e-05 / 2.157e-06 = 40.01.

That is a scale descent, not a copy of the 97 µm cellular-tier *yakuba* sibling (slot 2731). Parameters here are Schnakenberg `(a, b, D_u, D_v) = (0.10, 0.90, 1, 20)`, not that sibling’s power-law `(α, β) = (1.48, 0.252)`.

## Biology and the law

Carbone et al. (2005, Genetics, doi:10.1534/genetics.105.044412) map QTL for the pigmentation difference between *D. yakuba* and its sister *D. santomea*. Llopart et al. (2002) showed the difference is genetic. Wittkopp, True & Carroll (2002) showed Yellow and Ebony act reciprocally on cuticle melanin versus NBAD sclerotin. This script does **not** claim those papers wrote a PDE. It reconstructs Turing’s (1952) diffusion-driven instability with Schnakenberg (1979) kinetics, reads Yellow as the short-range activator and the Ebony/NBAD branch as the long-range substrate, and pins the selected wavelength to 2.157 µm.

## Run

```bash
python3 models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py
```

Outputs:

- `docs/media/drosophila_yakuba_turing_wavelength_s0_comparison.png`
- `docs/media/drosophila_yakuba_turing_wavelength_s0_growth.gif`
- `datasets/drosophila_yakuba_turing_wavelength_s0/drosophila_yakuba_turing_wavelength_s0_metrics.json`

Specimen `drosophila_yakuba_specimen.jpg` is the project species-gallery canonical image (SHA-256 pinned in the reference JSON). Wikimedia Commons did not return a clean CC whole-animal photograph of *D. yakuba*; the one hit was a multi-species spermatocyte immunostain (CC BY-SA 4.0) and was not used as the wavelength target.
