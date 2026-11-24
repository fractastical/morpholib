# *Danio albolineatus* — 37 µm Turing wavelength (loop 3161)

Holocene (extant) pearl danio at geological age **0 Ma**. The modelled phenomenon is the cellular-scale Turing wavelength of the melanophore–xanthophore interaction, **37 µm = 3.686e-05 m** (log10 = −4.433, tier 4/8 `4_cellular_10um`, detail variant). This is not the adult body-stripe repeat (the sibling 1.3 mm slot).

## Chronoseries

Danionin Pigment Evolution Series, Danio chronoseries **2/5**:

| Position | Species | Pattern at this cellular tier | What changes |
| --- | --- | --- | --- |
| Predecessor | *Danio aesculapii* | Vertical bars | Inhibitor diffusion is anisotropic (farther along the AP axis), so only the AP-varying mode is unstable and its wavelength is longer. |
| **This slot** | ***Danio albolineatus*** | Uniform interstripe | Isotropic coupling sits closest to the Turing threshold. The least-unstable wavenumber is the 37 µm mode, and its amplitude stays low. |
| Successor | *Danio margaritatus* | Spots / ocelli | A shorter activator range and a higher inhibitor ratio push the kinetics into the spot regime at a shorter wavelength. |

## Model

Schnakenberg reaction–diffusion (Turing 1952 dispersion; Nakamasu et al. 2009 short-range mutual exclusion and long-range melanophore dependence on xanthophores):

```
∂u/∂t = ∇·(D_u ∇u) + a − u + u²v
∂v/∂t = ∇·(D_v ∇v) + b − u²v
```

Critical wavenumber of the onset formula:

```
k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)
λ_c  = 2π / k_c
```

`D_u` for the pearl danio is solved so the fastest-growing numerical mode sits at λ = 3.686e-05 m. Periodic boundaries on an 8 × 5 wavelength flank patch. Equal diffusion (D_u = D_v) is the negative control and is linearly stable.

## Specimen

Public-domain photograph, Wikimedia Commons `File:Danio albolineatus.jpg` (Steven Coburn / Sc99cs), SHA-256 pinned in `danio_albolineatus_turing_wavelength_s0_reference.json`. At whole-body magnification one pixel is about 150 µm, so a 37 µm mode is sub-pixel on the animal. The animation keeps that photograph as the specimen context and shows the resolved cellular field in an inset.

## Run

```bash
python3 models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py
```

Outputs:

- `docs/media/danio_albolineatus_turing_wavelength_s0_comparison.png`
- `docs/media/danio_albolineatus_turing_wavelength_s0_growth.gif`

## Sources

Patterson, Bain & Parichy 2014, *Nat. Commun.* (10.1038/ncomms6299); Nakamasu et al. 2009, *PNAS* (10.1073/pnas.0808622106); Inaba, Yamanaka & Kondo 2012, *Science* (10.1126/science.1212821); Volkening & Sandstede 2018, *Nat. Commun.* (10.1038/s41467-018-05629-z); Podobnik et al. 2020, *Nat. Commun.* (10.1038/s41467-020-20021-6); Spiewak et al. 2018, *PLoS Genet.* (10.1371/journal.pgen.1007538); Turing 1952 (10.1098/rstb.1952.0012).
