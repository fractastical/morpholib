# Drosophila eugracilis — Turing wavelength, detail scale (slot 3184)

**Animal.** *Drosophila eugracilis* (Southeast Asian Gracile Fruit Fly), reduced tarsal sex-comb teeth. Aristotelian place: II. Anaima (Invertebrata) → Entoma (Arthropoda) → *Drosophila*.

**Scale.** 1.5 µm exactly as assigned: \(1.485\times 10^{-6}\) m, \(\log_{10} = -5.828\), tier 3/8 `3_subcellular_um`, detail variant. This is the Turing pigment-stripe / ocellus wavelength, not the ~101 µm primordium slot (`tanaka-drosophila-eugracilis-turing-wavelength`).

**Age.** Holocene (Extant), **0 Ma**. Lineage label: Sophophora melanogaster Subgroup Radiation (0–12 Ma). Drosophila chronoseries position **13/41**. Catalog predecessor *Drosophila erecta*; catalog successor *Drosophila ficusphila*. Geological age does not change between these three extant tips. What changes at this scale is Murray’s reaction-rate parameter \(\gamma\) and therefore the selected wavenumber \(k_c\).

**Law.** Turing wavenumber \(k_c\) for Schnakenberg activator–substrate kinetics (cohort 1, Chemical Turing / RD; form: stripes, spots, and ocelli).

\[
k_c^{2} = \frac{D_v f_u + D_u g_v}{2 D_u D_v}, \qquad \lambda_c = \frac{2\pi}{k_c}.
\]

With the Jacobian scaled by \(\gamma\), \(\lambda_c(\gamma) = \lambda_c(1)/\sqrt{\gamma}\). The length unit is fixed so that the focal species sits at 1.485 µm when \(\gamma = 1\).

| Chronospecies | Role | \(\gamma\) | Predicted \(\lambda_c\) |
|---|---|---:|---:|
| *D. erecta* | predecessor | 0.50 | 2.100 µm |
| *D. eugracilis* | focal, 0 Ma | 1.00 | 1.485 µm |
| *D. ficusphila* | successor | \((1.485/1.180)^2\) | 1.180 µm |

*D. erecta* (longer inhibitor range; female dark/light colour morphs) selects fewer, broader pigment domains. *D. eugracilis* (reduced sex comb) sits on the assigned 1.5 µm ocellar/granule scale. *D. ficusphila* (longitudinal sex-comb tooth row) selects a shorter wavelength and more foci across the same 17.820 µm detail window (12 focal wavelengths).

**Specimen plate.** `drosophila_eugracilis_turing_wavelength_s0_specimen.jpg` is the project gallery plate (SHA-256 pinned in the reference JSON). Wikimedia Commons returned no bitmap for this species on 2026-10-27, so the gallery fallback was vendored. The plate is a canonical brown spiral field with a 100 µm scale bar, not a scanning-electron micrograph of *D. eugracilis* ocelli. The simulation is scored against that plate; it does not claim the plate’s bar is 1.5 µm.

**Run.**

```bash
python3 models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py --pass 2
```

Primary sources are Crossref-checked in `paper_citations.bib`. The sibling primordium script’s DOI `10.1016/j.ydbio.2009.04.012` is a different paper and is not used here. The sex-comb citation is Tanaka, Barmina & Kopp (2009), PNAS, `10.1073/pnas.0807875106`.
