# Dugesia japonica — Turing wavelength, 6.9 µm detail scale (slot 3213)

Holocene (extant), **0 Ma**. Chronoseries **1/1**: there is no fossil predecessor and no successor chronospecies. The length in this slot is **6.912 µm** (6.912×10⁻⁶ m, log10 = **−5.160**), tier `3_subcellular_um`, detail variant. That length is the critical Turing wavelength λ_c = 2π / k_c of a saturated Schnakenberg pigment field, not a published caliper measurement of a *Dugesia japonica* cell.

## What changes at this scale

On a detail patch L = 7 × 6.912 µm = 48.384 µm the unstable band holds seven linear wavelengths. The same animal read at body-plan scale (millimetres) would pack that wavelength into a fine dorsal texture; it would not be the mode that sets anterior–posterior polarity. The scale-predecessor state is the spatially uniform steady state. Setting D_u = D_v makes the Turing gap negative and every Fourier mode decays. The ocellus cup is the scale-successor envelope: an anatomical mask taken from the specimen plate, larger than one wavelength, not a second reaction–diffusion mode.

Sibling slot `dugesia_japonica_turing_wavelength` (slot 166) uses a different parameter template. This slot does not reuse it. Operating point: a = 0.05, b = 1.40, saturation κ = 0.15, D_v / D_u = 48, seed 3213.

## Specimens

| File | Licence | What it is |
| --- | --- | --- |
| `dugesia_japonica_ocellus_pigment_nogi2009.png` | CC BY 2.5 | Nogi, Zhang, Chan & Marchant (2009), *PLoS Negl. Trop. Dis.* 3:e464. Regenerated *D. japonica* with dorsal pigment and ocelli. Not a 6.9 µm micrograph. |
| `dugesia_japonica_field_kalininsky_ccby4.jpg` | CC BY 4.0 | Timur Kalininsky, iNaturalist 226612281. Wild animals on substrate. |

SHA-256 digests are in `dugesia_japonica_turing_wavelength_s0_reference.json`.

## Law

k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v), with f_u, g_v from the saturated Schnakenberg Jacobian at the homogeneous steady state. D_u is solved so λ_c = 6.912 µm. Citations: `paper_citations.bib`.

## Run

```bash
python3 models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py
```
