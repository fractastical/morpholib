# *Dactylioceras commune* — 24 µm Turing wavelength (slot 3159)

Detail-scale reproduction for Project Delphi / MorphoEval loop 3159.

- **Animal:** *Dactylioceras commune* (Sowerby, 1815), Whitby bifurcating-rib ammonite. Aristotelian place: II. Anaima (Invertebrata) → Ostrakoderma → *Dactylioceras*.
- **Age:** Early Jurassic (Toarcian), **181 Ma**. Chronoseries: Dactylioceras 1/1 inside the Cephalopod Fractal Suture Fossil Series. There is no second *Dactylioceras* time point in this campaign. The predecessor used for the scale comparison is *Ceratites nodosus* (Ladinian, 240 Ma). The successor is *Placenticeras meeki* (Campanian, 75 Ma).
- **Scale:** **24 µm** detail variant, catalogue size **2.419×10⁻⁵ m** (24.19 µm, log10 = −4.616), tier `4_cellular_10um`. This is the Turing pigment-stripe / ocellus wavelength, not the millimetre rib spacing and not the 1.5 mm organ-scale sibling slot.
- **Law:** Turing wavenumber \(k_c\) for the Schnakenberg activator–inhibitor system. \(k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)\) and \(\lambda_c = 2\pi / k_c\) is fixed at \(2.419\times 10^{-5}\) m. Form class: stripes, spots, and ocelli.
- **Suture dimension:** catalogue \(D_f = 1.55\). The photograph is an exterior ribbed shell, so \(D_f\) is not recomputed from these pixels.

## Specimen

`dactylioceras_commune_specimen.jpg` — James St. John, *Dactylioceras commune* from the Whitby Formation, Yorkshire, Toarcian. The photographer states the fossil is 57 mm across. License: CC BY 2.0.

https://commons.wikimedia.org/wiki/File:Dactylioceras_commune_Ammonite_Fossil_from_Yorkshire_England.jpg

SHA-256 is recorded in `dactylioceras_commune_turing_wavelength_s0_reference.json`.

## What changes at 24 µm and 181 Ma

The critical wavelength stays at 24.19 µm. Along the lineage the organ-scale Turing wavelength that this detail wave must tile is 1.088 mm (*Ceratites*, 240 Ma) and 2.53 mm (*Placenticeras*, 75 Ma). On the vendored Whitby shell the outer-whorl rib spacing is measured from the 57 mm width. That measured spacing, divided by 24.19 µm, is the *Dactylioceras* mode count at this scale. Fibulation inserts secondary ribs on the outer flank (a spatial period doubling). The assigned suture dimension \(D_f = 1.55\) lengthens the unfolded septum relative to its chord, so a lobe of a given span hosts more 24.19 µm modes than a smooth ceratitic lobe.

## Run

```bash
python3 models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py --self-test
```

The script writes `docs/media/dactylioceras_commune_turing_wavelength_s0_comparison.png` and an animated `docs/media/dactylioceras_commune_turing_wavelength_s0_growth.gif`.

## Sources

See `paper_citations.bib`. Governing kinetics: Schnakenberg (1979). Molluscan shell transfer: Meinhardt & Klingler (1987); Fowler, Meinhardt & Prusinkiewicz (1992). Ammonite sutures: García-Ruiz, Checa & Rivas (1990); Lutz & Boyajian (1995). The chemical instability criterion is Turing (1952).
