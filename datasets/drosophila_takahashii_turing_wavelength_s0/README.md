# Drosophila takahashii — 2.4 µm Turing wavelength (slot 3208)

Holocene extant fly, geological age **0 Ma**. Phenomenon scale **2.4 µm** (precise size **2.362e-06 m**, log10 = **−5.627**), tier `3_subcellular_um` (3/8). The structure is the Turing pigment stripe / ocellus wavelength at this detail scale, in the Sophophora suzukii–biarmipes wing-spot clade (chronoseries 37/41).

The predecessor chronospecies is *Drosophila suzukii* (cellular wing-spot envelope 98.45 µm). The successor is *Drosophila teissieri* (cellular envelope 107 µm). Both are also extant. What changes here is the wavenumber, not the geological age: forty-one 2.362 µm lattices fit inside one suzukii envelope (98450 nm // 2362 nm = 41), and the teissieri envelope is 8550 nm longer without rescaling this lattice.

The script `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` integrates Schnakenberg kinetics (Schnakenberg 1979) and selects the Turing wavenumber \(k_c\) (Turing 1952; Murray 1981) so the analytic wavelength is 2.362 µm. Sex-comb context is Tanaka, Barmina & Kopp (2009) and Kopp (2011). The suzukii comparative anchor is Atallah et al. (2014).

The specimen file `Drosophila_takahashii_specimen.jpg` is the project gallery plate (SHA-256 `be067c44d300a455ec7c6c637159bbcd36b34ac85d816a2571d97f5da5166487`). A Commons search returned only *Drosophila pseudotakahashii*, which was not used.

```bash
python3 models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py --self-test --render-mode field
python3 models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py --render-mode composite
```

`field` is the Pass 1 colormap. `composite` (default) is the Pass 2 render inside the Otsu body mask.
