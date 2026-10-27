# Ambystoma mexicanum — Turing wavelength at 184 µm (Loop 3107)

Holocene (extant) axolotl, *Ambystoma mexicanum*. This slot is one point in evolutionary history and one length: **age 0 Ma**, **scale 184 µm** (exact lattice length **0.0001843 m**, 184.3 µm, log10 = −3.734, tier `5_primordium_100um`, detail variant).

The Ambystoma chronoseries for this loop is **1/1**. There is no predecessor or successor chronospecies to retune. What changes is the length, relative to the same living species:

- Shorter than 184 µm, larval chromatophores are still a migrating neural-crest population (cell diameters of tens of micrometres). Erickson (1993) describes that migratory segregation. No chemical wavelength has been selected yet.
- At 184.3 µm the Schnakenberg system sits just above the Turing threshold. The onset wavenumber \(k_c = 2\pi / 184.3\,\mu\mathrm{m}\) is the first unstable pigment spacing on a flank primordium patch.
- Longer than 184 µm, Murray’s (1981) scale argument applies: a growing flank stretches a fixed primordium wavelength into adult ocelli of millimetre spacing. That adult length is a later scale of the same Holocene species, not a second chronospecies, and it is not this slot.

## Law

Cohort: Turing reaction–diffusion wavelength selection (chemical Turing / RD). Form: stripes, spots, and ocelli. Quantitative law: Turing wavenumber \(k_c\).

Activator \(u\) and inhibitor \(v\) obey Schnakenberg kinetics

\[
\partial_t u = D_u\nabla^2 u + a - u + u^2 v, \qquad
\partial_t v = D_v\nabla^2 v + b - u^2 v
\]

with \((a,b)=(0.10,0.90)\), steady state \((u_0,v_0)=(1,0.9)\), and diffusion ratio \(D_v/D_u = 9\) (critical ratio ≈ 8.57). \(D_u\) is fixed so the onset wavenumber equals \(2\pi/184.3\,\mu\mathrm{m}\). Boundary conditions on the computed patch are periodic. The circular frame in the animation is only the display mask of the vendored micrograph, not a change of boundary condition.

Parameters are for this scale. They are not copied from any Ambystoma maculatum sibling slot.

## Sources

Crossref-checked:

- Turing (1952) doi:10.1098/rstb.1952.0012 — dispersion law.
- Murray (1981) doi:10.1016/0022-5193(81)90334-9 — coat pre-pattern and domain-growth scale argument.
- Epperlein & Löfberg (1989) doi:10.1007/978-3-642-75056-4 — larval pigment pattern of *Ambystoma mexicanum*.
- Frost-Mason & Mason (1996) doi:10.1387/ijdb.8877441 — what the axolotl pigmentary system has actually shown.
- Epperlein, Ziegler & Perris (1988) doi:10.1007/bf00219740 — identification of axolotl pigment cells.

None of those papers reports a measured wavelength of 184.3 µm. That length is the Project Delphi scale coordinate for this slot. The script’s claim boundary says so.

## Specimen

`ambystoma_mexicanum_chromatophores_specimen.jpg` is the Wikimedia Commons original of “Cromatóforos de larva de axolote pardo (Ambystoma mexicanum)”, by Brandon Antonio Segura Torres & Priscilla Vieto Bonilla, CC BY-SA 4.0.

https://commons.wikimedia.org/wiki/File:Cromatóforos_de_larva_de_axolote_pardo_(Ambystoma_mexicanum).jpg

SHA-256 `cc106cdbb5c28a40a47946801d35e6f13ada02665b8c773180123279a6cf22c8`.

The species-gallery file `datasets/species_gallery/I_Enaima_Vertebrata/Ambystoma_mexicanum.jpg` is a limb-skeleton plate. It is not the pigment specimen for this loop.

The micrograph has no scale bar, so melanophore spacing is reported in pixels and as an area fraction. It is not converted into micrometres.

## Run

```bash
python3 models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py
```

Outputs:

- `docs/media/ambystoma_mexicanum_turing_wavelength_s0_comparison.png`
- `docs/media/ambystoma_mexicanum_turing_wavelength_s0_growth.gif`

Formal rules: `morphoeval/lean/loops/Loop_3107_ambystoma_mexicanum_turing_wavelength_s0.lean`.
