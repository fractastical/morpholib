# Euperipatoides rowelli — Turing wavelength at 1.1 µm (Loop 3223)

Holocene (extant) chronospecies, geological age **0 Ma**. Aristotelian place: II. Anaima (Invertebrata) → Entoma (Arthropoda) → *Euperipatoides*. Lineage: Panarthropod & Invertebrate Extremophiles. This is chronoseries step **1/1** (no successor).

## Scale

The assigned phenomenon scale is **1.1 µm** (stored exactly as **1.12 × 10⁻⁶ m**, log10 ≈ −5.951, tier `3_subcellular_um`, detail variant). That length is the Turing wavelength λ selected by the fastest-growing wavenumber k_σ of a Schnakenberg activator–inhibitor system on a patch of the ocellus rhabdom / screening-pigment field.

It is **not** the published photoreceptor centre-to-centre pitch. Ljungholm & Nilsson (2021) place that pitch at 2–3 µm (model mesh ≈ 2.5 µm) inside an eye about 100 µm across (doi:10.1371/journal.pcbi.1008808). Kirwan et al. (2018) figure the microvillus transition with a 1 µm scale bar (doi:10.1242/jeb.175802). Mayer (2006) identifies the screening-pigment cells (doi:10.1016/j.asd.2006.06.003). The 1.12 µm detail wavelength is the slot’s reconstructed critical spacing in that microvillus / pigment-granule regime: 89 such wavelengths fit across a 100 µm eye, and 1120 nm is strictly finer than the 2500 nm receptor pitch.

## What changes at this scale and age

- **Predecessor** (outside this one-step chronoseries): a Cambrian stem lobopodian (~508 Ma). At a 10 µm preservation floor the 1.12 µm spacing is not countable, so N = 0.
- **This chronospecies** (*E. rowelli*, 0 Ma): k_σ is a physiological eigenvalue of the living retina. N = floor(100 µm / 1.12 µm) = 89. Diffusion ratio D_v/D_u = 22 is a property of that hydrated field, not of a fossil impression.
- **Successor:** none. The Holocene population is the terminus of the Euperipatoides chronoseries.

## Law

Schnakenberg (1979) kinetics with Turing’s (1952) dispersion relation. Parameters are slot-specific (a = 0.12, b = 0.88, D_u = 1, D_v = 22, seed 3223) and are not copied from a sibling animal. γ is fixed so that argmax σ(k) equals 2π / 1.12 µm on a domain 14 wavelengths wide (15.68 µm).

## Specimen

`euperipatoides_rowelli_specimen.jpg` — Alan Couch, CC BY 2.0, Wikimedia Commons `File:Euperipatoides_rowelli.jpg` (Nikon Coolpix P90, 2012-03-16). SHA-256 is recorded in the reference JSON. The whole-animal photograph is the CV target. The 15.68 µm computed patch is magnified onto the segmented body only as a visualization; the run does not claim that macroscopic annuli have period 1.12 µm.

## Run

```bash
python3 models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py --self-test
```
