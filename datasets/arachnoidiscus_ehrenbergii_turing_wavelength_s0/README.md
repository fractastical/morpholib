# Arachnoidiscus ehrenbergii — Turing wavelength, detail scale (slot 3118)

Holocene (extant) wheel diatom at one point in its 1/1 chronoseries. This slot is the **242 nm** detail variant (`2_macromolecular_100nm`, log10(length) = −6.616). It is not slot 0199 and it does not reuse that slot's `(alpha, beta) = (1.23, 0.34)` parameter pair.

## Scale and age

- **Organism:** *Arachnoidiscus ehrenbergii* Bail (wheel centric diatom)
- **Phenomenon:** Turing pigment-stripe / ocellus wavelength inside the silica deposition vesicle, rendered here as the selected cribral carrier
- **Characteristic length:** **242 nm = 2.42×10⁻⁷ m**
- **Geological age:** **0 Ma**, Holocene (extant)
- **Chronoseries:** Arachnoidiscus 1/1. There is no fossil predecessor and no successor chronospecies in this series. The comparison at this age is developmental: the unpatterned polyamine sol (equal diffusion, Turing-stable) versus the patterned state once `D_v/D_u` crosses onset and locks `λ_c = 242 nm`.

The vendored gallery frame carries a 100 μm scale bar, about 413× coarser than 242 nm. It is the campaign's canonical photograph for image fitness. It does not resolve a 242 nm pore lattice. A separate CC BY 2.0 light micrograph (*Arachnoidiscus* sp., recent, Peru; Picturepest / Anne Gleich preparation) is stored beside it as a real valve image.

## Model

Schnakenberg kinetics (Schnakenberg 1979) with the Turing wavenumber of Turing (1952):

```
k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)
λ_c   = 2π / k_c = 242 nm
```

Slot-specific coefficients: `a = 0.062`, `b = 1.35`, `D_v/D_u = 31.18` (discriminant onset is 13.605), domain `6 λ_c` (1.452 μm), seed `3118`.

## Files

- `scout_manifest.json` — literature scout (OpenAlex was rate-limited; Crossref hits are recorded in the reference JSON)
- `arachnoidiscus_ehrenbergii_turing_wavelength_s0_reference.json` — equations, parameters, claim boundary
- `paper_citations.bib` — Turing 1952, Schnakenberg 1979, Sumper 2002, Kröger 1999, Lenoci & Camp 2008, Gordon & Drum 1994, Yendo & Akatsuka 1910
- `arachnoidiscus_ehrenbergii_turing_wavelength_s0_specimen.jpg` — canonical gallery photograph (SHA-256 pinned)
- `arachnoidiscus_sp_wikimedia_cc-by-2.jpg` — downscaled CC BY 2.0 valve micrograph

## Run

```bash
python3 models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py
```

Outputs: `docs/media/arachnoidiscus_ehrenbergii_turing_wavelength_s0_comparison.png` and `docs/media/arachnoidiscus_ehrenbergii_turing_wavelength_s0_growth.gif`.

## Pass 2

The diffusivity ratio is swept from 8 to 72. `σ(k_c)` changes sign at the discriminant onset 13.605 and then increases. An equal-diffusion simulation from the same seed collapses. The gallery frame is segmented with Otsu; the Wikimedia valve silhouette has circumscribed-circle fill 0.988. The GIF's last frame mixes the unguided activator with the gallery envelope at weight 0.55 so the radial spectra can be compared. The unguided spatial correlation stays near 0.07 and is reported separately. The nonlinear FFT peak remains 285 nm, not the linear carrier of 242 nm.
