# *Aulacopleura koninckii* meraspis — 3.1 µm Turing wavelength (slot 3126)

Silurian (Wenlock) trilobite, meraspid degree 12, at the **detail subcellular scale**.

- Phenomenon scale: **3.1 µm** (`3.125e-06` m, log10 = −5.505, tier `3_subcellular_um`)
- Geological age: **428.5 Ma** (Wenlock Epoch)
- Cohort: Turing reaction–diffusion wavelength selection; quantitative law `k_c`
- Chronoseries position: **1/3**. Successor morph in this series is *Aulacopleura koninckii* holaspis18 (18 thoracic segments). There is no earlier member of the three-step series.

The 3.125 µm length is the classical critical wavelength `λ_c = 2π / k_c` of a near-onset Schnakenberg pigment system in the meraspid integument. It is the spacing selected for cuticular pigment lirae. It is not the thoracic-segment spacing (hundreds of micrometres) and not the 183 µm primordium stripe used by the sibling meraspis Turing slot.

At this scale the meraspid → holaspid step does **not** retune `k_c`. Hemianamorphosis adds thoracic segments (12 → 18, surplus 6). The same epithelial wavelength tiles a longer trunk, so the thorax hosts 18/12 = 1.5 times as many segment-bounded copies of the lira pattern. Parameters are specific to the 3.125 µm detail tier (diffusion ratio 2% above onset). They are not copied from the primordium-scale sibling.

## Reproduce

```bash
python3 models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py
```

Outputs:

- `docs/media/aulacopleura_koninckii_meraspis_turing_wavelength_s0_comparison.png`
- `docs/media/aulacopleura_koninckii_meraspis_turing_wavelength_s0_growth.gif`
- `datasets/aulacopleura_koninckii_meraspis_turing_wavelength_s0/metrics.json`

## Specimen

`aulacopleura_koninckii_meraspis_turing_wavelength_s0_specimen.jpg`

Wikimedia Commons file *Aulacopleura konincki.jpg*, own work by Dwergenpaartje, licence **CC BY-SA 3.0** (https://creativecommons.org/licenses/by-sa/3.0). SHA-256 `50ac5dbca82619780eb38716b9107395ccdf809491a0b8b20c71b9c94dfd4d05`.

The photograph is a dorsal exoskeleton. It does not resolve a 3.1 µm lira field. The model’s specimen metrics use the silhouette, furrow count, and structure-tensor of this image; the wavelength test is the dispersion relation of the PDE against linear theory.

## Sources

DOIs were resolved through `doi.org` content negotiation on 2026-10-27. OpenAlex returned HTTP 429 during scouting; Crossref did as well. See `paper_citations.bib` and `aulacopleura_koninckii_meraspis_turing_wavelength_s0_reference.json`.
