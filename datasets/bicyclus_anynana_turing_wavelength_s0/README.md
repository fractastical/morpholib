# *Bicyclus anynana* — Turing ocellus wavelength at 1.507 mm (Holocene, 0 Ma)

Slot 3133 of the 10,000-Animal Basic Look. One animal, one scale, one point in evolutionary history.

- **Animal:** *Bicyclus anynana* (Squinting Bush Brown), II. Anaima (Invertebrata) → Entoma (Arthropoda) → *Bicyclus*
- **Scale:** 1.507 mm (0.001507 m, log10 = −2.822, tier `6_organ_mm`, detail variant). This is the critical Turing wavelength λ_c = 2π/k_c of the ventral wing-cell activator, not a whole-wing length and not a sub-millimetre focal decay length.
- **Age:** Holocene (Extant), 0 Ma. Bicyclus chronoseries 1/1. The assigned lineage string is empty: there is no fossil predecessor or successor in this slot.

## What this scale selects

Sekimura, Venkataraman & Madzvamuse (2015, *PLOS ONE* 10:e0141434, Eq. 3.1–3.2 and Table 1) place a Gierer–Meinhardt activator–inhibitor system inside each wing cell. Veins are Dirichlet sources of activator at twice the homogeneous steady state; the proximal boundary is a Dirichlet source of strength k_p times that steady state; the wing margin and the entire inhibitor field are zero-flux. Low proximal activator (k_p = 0) lets a focus form. High proximal activator (k_p = 2) suppresses it.

On the infinite line the same kinetics have a most unstable wavenumber k_c. This slot anchors that wavelength to the adult organ scale specified for the Holocene terminus:

λ_c = 2π/k_c = 1.507 mm = 0.001507 m.

The modelled wing cell is one critical wavelength wide (anterior–posterior) and 2.2 dimensionless units long (proximal–distal). That single-mode cell produces one focus. The same kinetics on a cell two wavelengths wide insert a second focus. Because the chronoseries contains only the living species, that width contrast is the formal predecessor/successor comparison: a narrower pupal cell, or a longer-margined descendant at fixed λ_c, changes the focus count N = round(L/λ_c) while the Holocene adult holds λ_c at 1.507 mm.

## Files

- `bicyclus_anynana_turing_wavelength_s0_reference.json` — equations, Table 1 parameters, boundary conditions, specimen SHA-256
- `paper_citations.bib` — Sekimura et al. 2015, Gierer & Meinhardt 1972, Turing 1952, Connahs et al. 2019, Brakefield et al. 1996
- `scout_manifest.json` — OpenAlex / Wikimedia scout (slot 3133 is outside `campaign_100_slots.yaml`)
- `bicyclus_anynana_specimen.jpg` — CC BY-SA 4.0 ventral photograph, SHA-256 `cc4f97c92c75001c3bcaaaa1237f30dbaf5e8b7a58d1a8c8b237c96efc59855f`

Reproduction: `python3 models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py --self-test`
