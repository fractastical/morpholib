# Dickinsonia costata — cellular Turing wavelength (slot 3167)

**Animal.** *Dickinsonia costata*, an Ediacaran glide-symmetric segmented proarticulate. Cholesterol biomarkers (Bobrovskiy et al. 2018) place it among animals. Aristotelian path used by this campaign: II. Anaima (Invertebrata) → Zoophyta & Vermes → *Dickinsonia*.

**This scale.** 56 µm (5.564×10⁻⁵ m, log10 = −4.255), tier 4/8 `4_cellular_10um`. The quantity is the Turing pigment-stripe / ocellus wavelength on a module septum, not the body-scale module pitch.

**This age.** Late Ediacaran, White Sea, 558 Ma. The Dickinsonia chronoseries in this campaign is 1/1: there is no named predecessor or successor chronospecies. What changes at 56 µm relative to the organ-tier sibling (2.2 mm module pitch, same 558 Ma horizon) is the unstable wavenumber. One 2200 µm septum holds 39 pitches of 56 µm. At fixed reaction time the diffusivities scale as the square of that ratio (39² = 1521).

**Law.** Turing reaction–diffusion wavelength selection. Form class: stripes. Quantitative law: critical wavenumber k_c from the dispersion determinant, with the nonlinearly selected wavelength anchored at 56 µm.

**Specimen.** `dickinsonia_costata_turing_wavelength_s0_specimen.jpg` is `File:DickinsoniaCostata.jpg` from Wikimedia Commons, photographed by Verisimilus, license CC BY 2.5. Source: https://commons.wikimedia.org/wiki/File:DickinsoniaCostata.jpg

**Run.**

```bash
python3 models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py --pass-id 1
python3 models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py --self-test
```

Pass 1 writes a field-only baseline animation. The default and `--self-test` paths write the specimen-matched growth animation, the diffusion sweep, and the equal-diffusion ablation.
