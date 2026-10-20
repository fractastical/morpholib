# Slot 010 Reference Dataset — Somite-Prepatterned Avian Plumage Stripes (`datasets/avian_plumage_somite_stripes/`)

Reference parameters, comparative galliform stripe width benchmarks, heterospecific somite-to-dermis grafting shifts, an *agouti* dose axis, Haupaix et al. (2018) feather-row benchmarks, the Lin et al. (2013) feather-follicle ring-niche model, and biological specimen photography for **`2018-haupaix-avian-plumage-somite-stripes.py`**.

## Primary Literature

1. **Haupaix, N., Curantz, C., Bailleul, R., Beck, S., Robic, A., & Manceau, M. (2018).** *The periodic coloration in birds forms through a prepattern of somite origin.* **Science**, 361(6408), eaar4777. (`doi:10.1126/science.aar4777`, PMID 30237324)
2. **Inaba, M., Jiang, T.-X., Liang, Y.-C., Tsai, S., Lai, Y.-C., Widelitz, R. B., & Chuong, C.-M. (2019).** *Instructive role of melanocytes during pigment pattern formation of the avian skin.* **PNAS**, 116(14), 6884–6890. (`doi:10.1073/pnas.1816107116`, PMID 30886106)
3. **Bailleul, R., Curantz, C., Desmarquet-Trin Dinh, C., Hidalgo, M., Touboul, J., & Manceau, M. (2019).** *Symmetry breaking in the embryonic skin triggers directional and sequential plumage patterning.* **PLOS Biology**, 17(10), e3000448. (`doi:10.1371/journal.pbio.3000448`)
4. **Lin, S.-J., Foley, J., Jiang, T.-X., et al. (2013).** *Topology of feather melanocyte progenitor niche allows complex pigment patterns to emerge.* **Science**, 340(6139), 1442–1445. (`doi:10.1126/science.1230374`)

### Pass 3 citation corrections (2026-10-19)

Passes 1–2 cited Haupaix et al. (2018) under a non-existent title ("The periodicelly striped plumage of galliform chicks …") and Inaba et al. (2019) as a non-existent *Science* 363:eaaw1969 article. Both were corrected against the publishers' records (science.org / pnas.org) in `paper_citations.bib`, `avian_plumage_somite_stripes_reference.json`, the script docstring and this README; the two files were re-pinned (table below).

## What the paper actually reports (Pass 3 additions to the reference JSON)

`haupaix2018_feather_row_benchmarks` transcribes the quantitative statements of Haupaix et al. (2018): the dorsal stripe sequence cb / ly / lb (+ cy); *y* feathers = pheomelanin band with black base and tip; *agouti* bands A1′ (feather row 2), A1 (rows 3–5), A2 (central rows); wild-type *Coturnix japonica* ly stripe = feather rows 2–5 (4 rows), recessive-black RB−/− = rows 3–4 (2 rows, earlier switch back to eumelanin), Fawn Fa−/− = rows 1/2–6 (5–6 rows, no eumelanin before hatching); yellow band longer at the stripe centre than at the edge (fig. S1); stripe boundaries 2–4 mm (cb) and 3–9 mm (ly) from the dorsal midline at wing level.

`feather_follicle_niche_model` specifies the Pass 3 reduced model: a hexagonal follicle lattice (9 feather rows per somitic stripe period) whose ring niches (Lin et al. 2013) read the body-wide Agouti field; the Haupaix yellow-pulse-duration law `A(tau) = A_i - a + a cos(2 pi (tau - 1/2))` with feather type `[A_i >= theta_A] + [A_i - 2a >= theta_A]`; and the opened-cylinder chevron law `s(ell) = s_0 + k |ell|`. Both laws are machine-checked in `morphoeval/lean/loops/Loop_010_avian_plumage_somite_stripes.lean` (Theorems 4–5).

`pass3_provenance_flags` records what is **not** from the papers: the mm stripe widths of the 5-species panel and the quail Λ = 0.95 mm / w_y = 0.42 mm canonical widths, the "Agouti bead" dose regimes (Haupaix's dose series is the RB−/− / Fa−/− strains; ASIP beads are Lin et al. 2013 Fig. 3D and whiten feathers), the pheasant→quail and guineafowl→quail chimeras (Haupaix report quail somites → pheasant hosts only), and the rational PDE kinetics written in the JSON versus the cubic-saturating linearisation the script integrates. These are retained for continuity and flagged in the v3 audit.

## Vendored Files & SHA-256 Provenance Locks

| File | Description | SHA-256 |
| :--- | :--- | :--- |
| `avian_plumage_somite_stripes_reference.json` | Governing somite-Agouti + Cx40/melanocyte reaction-diffusion PDE parameters, 5-species galliform stripe table, somite grafting chimeras, agouti dose axis, Haupaix (2018) feather-row benchmarks, Lin (2013) follicle-niche model, provenance flags (re-pinned in Pass 3) | `4cbeeca03b0939bfc06eecb2de4e95a747dbba7eeb9b0924761797243a716113` |
| `galliform_chick_stripes_specimen.jpg` | Japanese quail (*Coturnix japonica*) dorsal striped plumage specimen (Wikimedia Commons, CC BY-SA 2.0, Wayne Cheng) — unchanged since Pass 1 | `78337aba30f626146f0f306c46fee403610f012f1aa456646a397df0ea001169` |
| `scout_manifest.json` | Automated literature & specimen scout dossier for Slot 010 — unchanged since Pass 1 | `acb7ac44b807e24bb7a38d4ca85195937830454b65e5c330b52976bab87975a2` |
| `paper_citations.bib` | BibTeX citations for Haupaix et al. (2018), Inaba et al. (2019), Bailleul et al. (2019), and Lin et al. (2013) (titles/venues corrected and re-pinned in Pass 3) | `368cf6829da0805619497a86a8f2c1043493ad9baadd0e69c21721bd5d883f08` |
