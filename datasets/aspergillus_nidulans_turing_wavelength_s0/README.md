# Aspergillus nidulans — Turing wavelength, detail scale (slot 3122)

Holocene extant colony of *Aspergillus nidulans* at **0 Ma**. Chronoseries **1/1**: this slot has no predecessor and no successor. The characteristic length is the Turing onset wavelength

**λ_c = 1.284×10⁻⁵ m = 12.84 µm (nominal 13 µm, log10 = −4.891)**,

tier `4_cellular_10um`, detail variant. Against the 10 µm tier floor the wavelength is 2.84 µm longer, so the critical wavenumber k_c = 2π/λ_c is 4.893×10⁵ m⁻¹ rather than 6.283×10⁵ m⁻¹. Nothing in this folder reuses slot 182’s template constants (1.31, 0.25, 12, 4, 48).

## Specimen

`aspergillus_nidulans_turing_wavelength_s0_specimen.jpg` is a public-domain macrograph of an *A. nidulans* mutant colony (Wikimedia Commons, File:Aspergillus_nidulans_mutant.jpg, uploader C1041012, 1 March 2007). SHA-256 `282a45a621880a6946f449f6b931beaa6e435d12c7ff1d15cec22592c855e4ed`. The gallery fallback `datasets/species_gallery/IV_Protista_Monera_Mycota/Aspergillus_nidulans.jpg` was not copied; a primary Commons photograph was available.

The photograph is a colony-scale spiral. It does not resolve 13 µm. The cellular model is a Schnakenberg reaction–diffusion system whose onset wavelength is set to 12.84 µm. The growth animation uses that activator as a weak carrier on the measured colony optical density (Trinci radial expansion for the front).

## Model

Schnakenberg (1979) kinetics on a periodic grid, Turing (1952) conditions, k_c from the standard onset formula. Pigment biology is the background in Adams, Wieser & Yu (1998) and Clutterbuck (1969), not a published PDE. See `paper_citations.bib` and `aspergillus_nidulans_turing_wavelength_s0_reference.json`.

```bash
python3 models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py --self-test
```
