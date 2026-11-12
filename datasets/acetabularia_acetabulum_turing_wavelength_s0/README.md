# Acetabularia acetabulum — Turing wavelength, 25.28 µm (slot 3103)

Holocene extant mermaid's wineglass, age **0 Ma**. Chronoseries length 1: there is no predecessor and no successor chronospecies. The scale is the tier-4 detail length **25.28 µm = 2.528×10⁻⁵ m** (log10 = −4.597, `4_cellular_10um`).

The developmental predecessor of this scale is the unpatterned apex of the same cell. At this scale a periodic patch of cap wall, 202.24 µm on a side, holds eight pigment ridges. Eleven copies of that patch tile a young cap whose margin radius is 354 µm (88 ridges). Later growth of the same Holocene cap adds ridges at the same wavelength. It does not introduce another species, and it is not the organism-scale whorl model in the sibling slot.

## Sources

- Turing, A. M. (1952). The chemical basis of morphogenesis. *Phil. Trans. R. Soc. B* 237:37–72. https://doi.org/10.1098/rstb.1952.0012
- Dumais, J. & Harrison, L. G. (2000). Whorl morphogenesis in the dasycladalean algae: the pattern formation viewpoint. *Phil. Trans. R. Soc. B* 355:281–305. https://doi.org/10.1098/rstb.2000.0565
- Harrison, L. G. & Kolář, M. (1988). Coupling between reaction-diffusion prepattern and expressed morphogenesis, applied to desmids and dasyclads. *J. Theor. Biol.* https://doi.org/10.1016/s0022-5193(88)80213-3
- Harrison, L. G., Graham, K. T. & Lakowski, B. C. (1988). Calcium localization during *Acetabularia* whorl formation. *Development* 104:255–262. https://doi.org/10.1242/dev.104.2.255
- Holloway, D. M. & Harrison, L. G. (2007). Pattern selection in plants. *Ann. Bot.* https://doi.org/10.1093/aob/mcm295

Dumais & Harrison conclude that reaction–diffusion along the membrane is the whorl mechanism the experiments favour. Holloway & Harrison integrate a Brusselator on a growing surface and obtain whorled branches. The 25.28 µm target is this slot's tier length, not a spacing printed in those papers. Under the nominal diffusivities the onset wavelength is 22.67 µm and the fastest linear mode is 25.43 µm.

## Specimens

- `acetabularia_acetabulum_wikimedia_cc0.jpg` — CC0 cap close-up by Manuel Cardeñosa, iNaturalist photo 209045521, via Wikimedia Commons. This is the image-fitness target. SHA-256 `0773237a26517128a91d0c34e0b16adecdd490932a5b923cfd6415715b54aebc`.
- `acetabularia_acetabulum_wikimedia_ccby.jpg` — CC BY 4.0 colony by Yves Bas, iNaturalist photo 112232286. SHA-256 `512808d84df1d5b386e28bcdad1ab651ad77529e189b57cf8ab00619d7e2367d`.
- `acetabularia_acetabulum_specimen.jpg` — copy of the project gallery plate (100 µm scale bar), kept as the documented fallback. SHA-256 `17641545f872481e451f25234ae988af0c16a529dd25aff08a53bd400e2c23a1`.

## Run

```bash
python3 models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py
```

Outputs: `docs/media/acetabularia_acetabulum_turing_wavelength_s0_comparison.png` and `docs/media/acetabularia_acetabulum_turing_wavelength_s0_growth.gif`.
