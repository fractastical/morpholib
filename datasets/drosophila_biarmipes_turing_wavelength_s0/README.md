# Drosophila biarmipes — 1.5 µm Turing wavelength (slot 3177, detail s0)

Holocene (extant), geological age **0 Ma**. Lineage: Sophophora suzukii–biarmipes wing-spot clade. Delphi Drosophila chronoseries position 6/41. Predecessor chronospecies *Drosophila ananassae*. Successor chronospecies *Drosophila bipectinata*. This order is the catalog chronoseries, not a claim that *D. biarmipes* is the ancestor of *D. bipectinata*.

## Scale

The assigned phenomenon is the **1.5 µm** detail-scale Turing pigment stripe / ocellus wavelength (catalog SI length **1.486×10⁻⁶ m**, log10 = **−5.828**, tier `3_subcellular_um`). This is the melanin-granule / pigment-precursor wavelength *inside* the male wing spot, not the ~100 µm organ-scale spot itself (that coarser slot is `gompel-drosophila-biarmipes-turing-wavelength`, 96 µm).

## What changes at this scale

Schnakenberg activator–inhibitor kinetics on a periodic cuticle patch, with diffusivities fixed so the linear critical wavelength of *D. biarmipes* is 1.486 µm.

- *D. ananassae* (predecessor): spot-enhancer gain is absent, so the activator Jacobian entry \(f_u\) stays negative. The same diffusivities are subcritical (\(\sigma_{\max}<0\)): no 1.5 µm pigment lattice.
- *D. biarmipes* (this slot, 0 Ma): the derived *yellow* spot cis-regulatory element (Gompel et al. 2005) is represented as a raised autocatalytic balance \(b\). \(\sigma_{\max}>0\) and \(\lambda_c=1.486\,\mu\mathrm{m}\).
- *D. bipectinata* (successor): a stronger production balance at the **same** diffusivities shortens \(\lambda_c\) (more granules per patch). Spot *number* at organ scale is not this slot’s state variable.

Gompel et al. 2005 and Prud’homme et al. 2006 establish the cis-regulatory origin of the wing spot. They do **not** report a measured 1.5 µm reaction–diffusion wavelength. The wavelength is the catalog detail scale; the script reconstructs Turing wavenumber selection at that scale.

## Specimen

`drosophila_biarmipes_specimen.jpg` is the repository species-gallery plate (copied from `datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_biarmipes.jpg`). Wikimedia Commons returned no CC-licensed bitmap of *D. biarmipes* on 2026-10-27 (only related videos/PDFs). The plate carries a 100 µm scale bar and is not a transmission electron micrograph of a 1.5 µm lattice.

## Reproduce

```bash
python3 models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py
```

Outputs: `docs/media/drosophila_biarmipes_turing_wavelength_s0_comparison.png` and `docs/media/drosophila_biarmipes_turing_wavelength_s0_growth.gif`.
