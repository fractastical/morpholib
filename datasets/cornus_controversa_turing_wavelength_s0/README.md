# *Cornus controversa* — Turing wavelength at 1.939 mm (slot 3155)

Holocene extant pagoda dogwood (*Cornus controversa*), age **0 Ma**, chronoseries **1/1**. This slot is the organ-detail scale **1.9 mm** (exactly **0.001939 m**, log10 = **−2.712**, tier `6_organ_mm`).

The quantitative law is Turing’s critical wavenumber \(k_c\) (Turing 1952), realized here as a Schnakenberg activator–inhibitor system (Schnakenberg 1979) whose analytic wavelength is pinned to

\[
\lambda_c = 2\pi / k_c = 1.939\,\mathrm{mm}.
\]

The domain is a periodic square lamina patch of side \(8\lambda_c = 15.512\,\mathrm{mm}\). It is not a model of the metre-scale pagoda crown. The vendored photograph is a whole-crown public-domain image and contains no scale bar, so 1.939 mm is not a length that can be read off those pixels. Crown-scale computer-vision measurements are reported separately from the lamina-patch wavelength.

## What changes at this scale and age

The Cornus chronoseries in this slot has no predecessor and no successor (index 1 of 1, 0 Ma). There is no temporal parameter shift to copy from a fossil chronospecies. The change that this scale isolates is the state variable: at the crown scale the observable is branching geometry; at 1.939 mm the observable is the selected chemical wavenumber \(k_c\). A finer cellular scale would replace the continuum wavelength with a layer-identity (chimera) state. Those are scale transitions, not speciation events inside this series.

The cream margin of the cultivar ‘Variegata’ is a periclinal chimera. This dataset does not claim that chimera is a Turing pattern.

## Files

- `cornus_controversa_specimen.jpg` — byte copy of `datasets/species_gallery/III_Phyta_Embryophyta/Cornus_controversa.jpg` (upstream: Wikimedia Commons `File:Cornus controversa 'variegata' - NBGB - IMG 4246.JPG`, Botanic Garden Meise, public domain). SHA-256 `be3da2bb8b817bec105efa4dec6064d234f1c47ab458bdf8059d885529dccff4`.
- `cornus_controversa_turing_wavelength_s0_reference.json` — equations, parameters, claim boundary.
- `paper_citations.bib` — Crossref-checked citations.
- `scout_manifest.json` — literature scout. Slot 3155 is outside `campaign_100_slots.yaml`, so the scout used direct OpenAlex and Crossref queries rather than `--slot 3155`.

## Run

```bash
python3 models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py
```
