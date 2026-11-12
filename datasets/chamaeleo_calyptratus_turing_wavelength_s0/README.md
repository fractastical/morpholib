# *Chamaeleo calyptratus* — Turing wavelength at 260.4 µm (slot 3144)

Holocene (extant), **0 Ma**. Characteristic scale **260.4 µm** (0.0002604 m, log10 = −3.584), tier `5_primordium_100um`, detail variant. Form: vertical pigment bars and ocelli. Law: Turing wavenumber \(k_c\).

The chronoseries in this slot has one point. There is no fossil *C. calyptratus* predecessor to difference against. The comparison is across scale at the same age:

| neighbour | length | what the carrier does |
| --- | --- | --- |
| predecessor cutoff | 80 µm | shorter than \(\lambda_c\); the unstable mode does not fit |
| **this slot** | **260.4 µm** | one selected wavelength; dorsoventral bars |
| successor envelope | 2.604 mm (10 \(\lambda_c\)) | an adult lateral bar is many carrier wavelengths, not a new \(k_c\) |

The 80 µm and 2.604 mm figures are scale-separation cutoffs for this loop, not caliper readings off the photograph. The Wikimedia photograph has no scale bar, so it is not converted into micrometres.

## Sources

- Turing 1952, *Phil. Trans. R. Soc. B* 237:37–72. DOI `10.1098/rstb.1952.0012`. Dispersion relation.
- Schnakenberg 1979, *J. Theor. Biol.* 81:389–400. DOI `10.1016/0022-5193(79)90042-0`. Kinetics used in the script.
- Murray 1981, *J. Theor. Biol.* 88:161–199. DOI `10.1016/0022-5193(81)90334-9`. Coat-pattern wavenumber.
- Ligon & McGraw 2013, *Biol. Lett.* 9:20130892. DOI `10.1098/rsbl.2013.0892`. Rapid lateral-stripe signalling in *C. calyptratus*. Does not report a micrometre period.
- Teyssier, Saenko, van der Marel & Milinkovitch 2015, *Nat. Commun.* 6:6368. DOI `10.1038/ncomms7368`. Iridophore photonics, measured on *Furcifer pardalis*, lattice spacing of order 100 nm. Cited for the dermal stack only. This slot does not treat that paper as a measurement of 260.4 µm.

Parameters are solved in `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` (`A = 0.062`, `B = 1.150`, supercritical ratio 1.45, `Dv_y/Dv_x = 0.42`). They are not the slot-123 template (`alpha_scale = 1.63`).

## Specimen

`Chamaeleo_calyptratus_HZell_CC-BY-SA-3.0.jpg`

Veiled chameleon, Staatliches Museum für Naturkunde Karlsruhe. Photograph by H. Zell, 2007-12-23, Wikimedia Commons `File:Chamaeleo calyptratus 01.jpg`, licence CC BY-SA 3.0. SHA-256 is recorded in the reference JSON.

The gallery file `datasets/species_gallery/I_Enaima_Vertebrata/Chamaeleo_calyptratus.jpg` is a synthetic purple spiral with a caption, not a photograph of the animal, so it is not the specimen used here.

## Reproduce

```bash
python3 models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py
```
