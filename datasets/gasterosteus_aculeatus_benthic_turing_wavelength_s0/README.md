# Gasterosteus aculeatus benthic — cellular Turing wavelength (43 µm, 0 Ma)

Detail-scale slot 3234 of the 10,000-animal Basic Look. This folder is the
source pack for one animal, one scale, and one point in the pelvic-reduction
chronoseries. It does not replace the organ-scale sibling
`datasets/gasterosteus_aculeatus_benthic_turing_wavelength/` (2.2 mm flank bar).

## Assignment

- Animal: *Gasterosteus aculeatus* benthic, post-glacial Pitx1-deleted freshwater
  low-plate stickleback. I. Enaima (Vertebrata) → Pisces (Ichthyes) → Gasterosteus.
- Scale: 43 µm (4.315×10⁻⁵ m, log10 = −4.365), tier 4/8 `4_cellular_10um`.
  The quantity is the critical Turing wavelength λ_c = 2π/k_c of a dermal
  chromatophore spot (ocellus), not the millimetre flank bar.
- Age: Holocene post-glacial lake, extant, 0 Ma. Chronoseries position 6/6.
  Predecessor: marine *G. aculeatus*, Late Pleistocene, 0.015 Ma, organ-scale
  wavelength 2.4 mm.

## What changes at this scale and age

Marine ancestors carry a complete row of about 32 bony lateral plates and a
flank bar whose organ-scale repeat is 2.4 mm. The benthic chronospecies, at
0 Ma, has fixed the recurrent Pitx1 pelvic-enhancer deletion (Chan et al. 2010)
and the low-plate Eda allele (Colosimo et al. 2005; typically 0–7 plates).
Loss of the plate row removes millimetre-spaced bony interruptions from the
dermis. On the continuous dermal sheet the unstable wavenumber selected by
Schnakenberg kinetics is the cellular mode k_c ≈ 1.46×10⁵ m⁻¹ (λ_c = 43.15 µm),
about 51× shorter than the benthic organ-scale bar (2.2 mm) and about 56×
shorter than the marine organ-scale bar (2.4 mm). Juvenile pigment divergence
is genetic (Greenwood, Jones, Chan et al. 2011) and Kit-ligand dosage is a
known cis-regulatory pigment input in sticklebacks (Miller et al. 2007). The
activator–inhibitor signs follow the teleost chromatophore Turing literature
(Kondo & Asai 1995; Nakamasu et al. 2009). Those rate constants were measured
in angelfish and zebrafish, not in benthic stickleback skin.

## Files

- `gasterosteus_aculeatus_benthic_turing_wavelength_s0_reference.json` — equations,
  parameters, claim boundary, and pinned hashes.
- `paper_citations.bib` — Crossref-checked sources.
- `scout_manifest.json` — literature and Commons scout for slot 3234.
- `gasterosteus_aculeatus_benthic_turing_wavelength_s0_specimen.jpg` — repository
  canonical detail plate for this taxon (100 µm scale bar, single chromatophore
  focus). Used as the computer-vision comparison target. It is a gallery plate,
  not a published micrograph of a counted 43 µm melanophore lattice.
- `gasterosteus_aculeatus_wikimedia_lateral.jpg` — whole-animal lateral photograph
  of *Gasterosteus aculeatus* by Ron Offermans, Wikimedia Commons,
  CC BY-SA 3.0 (https://commons.wikimedia.org/wiki/File:Gasterosteus_aculeatus.jpg).
  The file is not labeled as the benthic plate morph.

## Reproduce

```bash
python3 models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py
```

Outputs:

- `docs/media/gasterosteus_aculeatus_benthic_turing_wavelength_s0_comparison.png`
- `docs/media/gasterosteus_aculeatus_benthic_turing_wavelength_s0_growth.gif`

Seed is 3234. Dependencies are numpy, scipy, pillow, matplotlib, and pyyaml.
