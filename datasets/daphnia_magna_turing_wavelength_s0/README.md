# *Daphnia magna* — ocellus pigment Turing wavelength (slot 3165, detail scale)

Holocene (extant), **0 Ma**. Phenomenon scale **2.7 µm** (**2.736 × 10⁻⁶ m**, log10 = −5.563), tier `3_subcellular_um`.

This slot is the detail-scale (s0) reaction–diffusion wavelength for the water flea *Daphnia magna* (II. Anaima → Entoma → *Daphnia*). The chronoseries has a single entry (1/1). There is no predecessor chronospecies and no successor chronospecies, so the age delta along the series is zero. What changes is the **scale**, not the geological age:

- below this tier, the relevant length is a screening-pigment melanosome (~0.5 µm);
- at this tier, Schnakenberg kinetics select a critical wavelength λ_c = 2π / k_c = 2.736 µm;
- above this tier, the compound-eye ommatidium is a cellular pitch (tens of micrometres; Smith & Macagno 1990 resolve single ommatidia).

The governing law is the Turing wavenumber. Kinetics are Schnakenberg (1979), not the activator–inhibitor form of Gierer & Meinhardt (1972) and not the placeholder coefficients used by any sibling *Daphnia* slot. Parameters are a = 0.062, b = 1.180, D_u = 0.016, D_v = 0.640.

## Specimen

`daphnia_magna_head_sem.jpg` is a lateral scanning-electron micrograph of the head of *Daphnia magna*, CC BY-SA 4.0, Paulo Corgosinho, 2016, Wikimedia Commons `File:Daphnia magna head.jpg`. The printed scale bar is 100 µm (156 px on this file, 0.641 µm/px) at ×130. That micrograph resolves the cuticle reticulum. It does **not** resolve a 2.736 µm pigment wavelength as a countable stripe (about 4.3 px on this sensor). The simulation domain is a 119 µm patch of pigment epithelium, one scale-bar long, filled by the critical wavelength.

## Run

```bash
python3 models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py
```

Outputs:

- `docs/media/daphnia_magna_turing_wavelength_s0_comparison.png`
- `docs/media/daphnia_magna_turing_wavelength_s0_growth.gif`

## Claim boundary

Reported in the sources: Turing’s dispersion relation; Schnakenberg’s two-species kinetics; pattern formation in the *D. magna* visual system (Macagno 1981); ultraviolet photoreceptors inside single ommatidia (Smith & Macagno 1990); inducible cuticular melanin (Hebert & Emery 1990).

Reconstructed here: the critical wavelength is fixed at the assigned 2.736 µm tier, and a spectral Schnakenberg solver is tested against its own linear dispersion relation.

Not claimed: that a published measurement found a 2.736 µm stripe period in the ocellus, or that the ×130 cuticle polygons are that Turing field.
