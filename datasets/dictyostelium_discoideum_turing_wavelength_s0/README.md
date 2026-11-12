# Dictyostelium discoideum — Turing wavelength at 2.304 µm (slot 3168)

Holocene (extant) social amoeba, geological age **0 Ma**. Chronoseries index **1/1**: this slot has no earlier fossil chronospecies. The comparison formalized here is between spatial regimes of the same extant organism, not between named predecessor and successor morphospecies.

## Scale

| Quantity | Value |
| --- | --- |
| Phenomenon | Turing critical wavelength of a cortical activator–inhibitor field |
| λ_c | **2.3 µm** (exact slot coordinate **2.304 µm** = **2.304e-06 m**) |
| log10(λ_c / m) | **−5.638** |
| Tier | **3/8 `3_subcellular_um`**, detail variant |
| Cortical patch | 11.52 µm = five critical wavelengths (one vegetative cell) |
| Form class | Stripes, spots, and ocelli |
| Law | Turing wavenumber k_c |

The 2.304 µm figure is the Delphi catalog coordinate for this slot. It sits in the few-micrometre band of *Dictyostelium* cortical PIP3 / F-actin waves (Taniguchi et al. 2013; Bretschneider et al. 2009; Wasnik & Mukhopadhyay 2013). It is not a transcribed wet-lab mean that some paper printed as 2.304.

## What changes at this scale

- **Molecular predecessor regime (≈ 0.20 µm).** A receptor-cluster domain shorter than λ_c admits no integer Turing mode (`11520 nm / 2304 nm = 5` on the cell; `200 / 2304 = 0` on the cluster). The instability band is closed.
- **This slot (2.304 µm, 0 Ma).** Schnakenberg kinetics with diffusion ratio δ = D_v / D_u = 24 select k_c = 2π / 2.304 µm⁻¹ inside one cell. Five wavelengths fit across 11.52 µm.
- **Aggregation successor regime (order 1 mm).** Extracellular cAMP relay (Martiel & Goldbeter 1987; Höfer, Sherratt & Maini 1995; Sawai, Thomason & Cox 2005) selects a much longer wavelength because the diffusion length leaves the cortex. The cortical wavenumber is not the field wavenumber. At 0 Ma both regimes are expressed in the same extant species; they are different scales, not different chronospecies.

## Model

Schnakenberg (1979) kinetics on a periodic square, parameters fixed so the discrete-Laplacian growth rate peaks at λ = 2.304 µm:

```
∂u/∂t = D_u ∇²u + a − u + u²v
∂v/∂t = D_v ∇²v + b − u²v
k_c²  = (D_v f_u + D_u g_v) / (2 D_u D_v)
λ_c   = 2π / k_c
```

Steady state u₀ = a + b = 1.10, v₀ = b / u₀². a = 0.16, b = 0.94, D_u = 0.03208776235376616 µm² s⁻¹, D_v = 0.770106296490388 µm² s⁻¹. Equal diffusion (D_u = D_v) is the negative control: the unstable band disappears.

## Specimen

`dictyostelium_discoideum_turing_wavelength_s0_specimen.jpg` is the public-domain Wikimedia photograph *Dictyostelium Fruiting Bodies* (Bruno in Columbus, 2008). SHA-256 `e2d2b739211bda975bc175167debf554af4991ce21bc8b1106d1053e6bc95a0b`. It is a sorocarp micrograph, not a 2.3 µm cortical frame. That scale gap is a transfer boundary: the photograph anchors species identity and computer-vision texture; the wavelength law is checked against the linear dispersion relation.

## Run

```bash
python3 models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py
```

Outputs:

- `docs/media/dictyostelium_discoideum_turing_wavelength_s0_comparison.png`
- `docs/media/dictyostelium_discoideum_turing_wavelength_s0_growth.gif`
- `datasets/dictyostelium_discoideum_turing_wavelength_s0/dictyostelium_discoideum_turing_wavelength_s0_metrics.json`

## Claim boundary

The script reconstructs Turing wavenumber selection. It does not claim that sorocarps are pigment ocelli, that Taniguchi phase singularities were integrated, or that a paper measured λ = 2.304 µm.
