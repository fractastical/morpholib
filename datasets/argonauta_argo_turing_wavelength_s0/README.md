# Argonauta argo — Turing ocellus wavelength (slot 3121, detail scale)

Holocene (extant) female paper nautilus, *Argonauta argo* Linnaeus, 1758.
This slot is the **95 µm** detail variant (phenomenon size **9.48e-05 m**, log10 = **−4.023**, tier `4_cellular_10um`) at geological age **0 Ma**. The Argonauta chronoseries in this campaign has a single node (1/1): there is no sampled predecessor or successor chronospecies, so no inter-node wavelength delta is defined.

The pattern under test is one pigment ocellus (bright centre, inhibitory dark ring) at that cellular scale, not the centimetre-scale brood shell and not the millimetre-scale shell ribs.

## Model

Schnakenberg kinetics (Gierer–Meinhardt / Turing class) reduced to the critical radial mode:

- steady state `(u, v) = (a + b, b / (a + b)²)` with `a = 0.10`, `b = 0.90`
- diffusivities `D_u = 1`, `D_v = 18` (inhibitor diffuses farther)
- critical wavenumber `k_c` from `k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)`
- the length unit is fixed so `2π / k_c` equals the ocellus period measured on the 100 µm-scale plate
- the tier label **95 µm** is compared with the measured inhibitory-halo radius (they are not forced equal)

An equal-diffusion ablation (`D_u = D_v`) keeps every wavenumber stable. A γ sweep checks that the selected wavelength moves as `γ^(−1/2)`.

## Specimens

| File | What it is |
| --- | --- |
| `argonauta_argo_turing_wavelength_s0_specimen.jpg` | Repository species-gallery plate (800×600) with an embedded 100 µm scale bar. SHA-256 pinned. This is the pattern target at the detail scale. |
| `argonauta_argo_shell_cc0.jpg` | CC0 whole-shell photograph (Di Turner, iNaturalist observation 15508745, via Wikimedia Commons). Anatomical voucher and negative scale control: shell ribbing is not the 95 µm ocellus. |

Finn & Norman (2010) document the brood shell as a gas-mediated buoyancy organ. They do not report a 95 µm Turing wavelength. Turing (1952), Gierer & Meinhardt (1972), and Murray (1981) supply the wavelength-selection law. Kondo & Asai (1995) is cited only as a pigment reaction–diffusion wave in another animal.

## Run

```bash
python3 models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py
```

Outputs:

- `docs/media/argonauta_argo_turing_wavelength_s0_comparison.png`
- `docs/media/argonauta_argo_turing_wavelength_s0_growth.gif`
