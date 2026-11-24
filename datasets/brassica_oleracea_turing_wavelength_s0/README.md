# Brassica oleracea — Turing wavelength at 172 µm (slot 3137)

Holocene extant **0 Ma**. Chronoseries **1/1** (no fossil Brassica predecessor in this loop). Scale tier `5_primordium_100um`, detail variant: characteristic length **172 µm** (0.000172 m, log10 = −3.764).

The model is Schnakenberg kinetics (Schnakenberg 1979) used as a concrete instance of Turing’s wavenumber selection (Turing 1952). Inhibitor diffusivity is solved so the fastest-growing mode has wavelength **172 µm** on a meristem patch of **24 × 172 µm = 4128 µm**.

## What changes at this scale

There is no second chronospecies. The comparison is across scale on the same extant curd:

- Nominal primordium tier **100 µm** on the 4128 µm patch packs 41 peaks. This detail variant packs 24. Growth rate at 100 µm is positive but below the 172 µm peak.
- Successor cellular scale **15 µm** (one epidermal cell) has σ < 0 at the diffusivities that select 172 µm. The cell is not the patterned mode: 172 / 15 = 11 cells per selected wave.
- Equal diffusion (D_u = D_v) is the negative control: σ_max < 0 and the activator standard deviation collapses.

## What this is not

The vendored photograph is a studio macrograph of a Romanesco curd (Ivar Leidus, CC BY-SA 4.0), copied from `datasets/species_gallery/III_Phyta_Embryophyta/Brassica_oleracea.jpg`. It does not resolve 172 µm. Image fitness compares texture statistics of the animated spot field with that photograph. Azpeitia et al. 2021 (DOI `10.1126/science.abg5999`) explain Romanesco cones by a floral gene network and an accelerating plastochron, which is a different mechanism from this Turing wavelength. OpenAlex does not resolve the DOI `10.1126/science.abg5935` used by the sibling fractal-inflorescence dataset; this slot cites `abg5999`.

## Run

```bash
python3 models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py
```

Outputs: `docs/media/brassica_oleracea_turing_wavelength_s0_comparison.png` and `docs/media/brassica_oleracea_turing_wavelength_s0_growth.gif`.

## Files

- `brassica_oleracea_turing_wavelength_s0_specimen.jpg` — gallery copy, SHA-256 `7306e7e95f1cbeccb6d0c1fbc7e9e210033186b5eecce65421c0ce867b4f03cd`
- `brassica_oleracea_turing_wavelength_s0_reference.json` — parameters, Jacobian, claim boundary
- `paper_citations.bib` — Turing 1952, Schnakenberg 1979, Gierer & Meinhardt 1972, Kondo & Miura 2010, Azpeitia et al. 2021, Prusinkiewicz et al. 2007, Duclos & Björkman 2008
- `scout_manifest.json` — OpenAlex / Wikimedia scout (slot 3137 is outside `campaign_100_slots.yaml`)
- `solver_cache.json` — last deterministic run (seed 3137)
