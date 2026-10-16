# MorphoEval report — audit `2026-10-14-odonata-audit`

Rubric **0.1.0** · checkpoint `fbbcaf6` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are 0–100 (emitted as `pk:hasProbability` = score/100 on the LLM assessment of the dimension hypothesis); Overall is the rubric-weighted mean and the claim's `pk:weightOfEvidence`. Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Faithfulness | Reproducibility | **Overall** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Odonata wing cells through deep time (`2026-eshghi-odonata-wing-cells.py`) | 70 | 85 | 50 | 85 | 90 | 70 | 70 | 75.7 | 76.7 | **76.0** | mostly faithful |

## Findings and remediation

### Odonata wing cells through deep time  —  overall 76.0 (mostly faithful)

A data comparison rather than a simulation: the one trait the Hoffmann venation model predicts (cell size set by an inhibition length) is compared on 69 fossil and 394 extant wings from Eshghi et al. 2026, with PBDB ages and Hoffmann et al.'s own segmentations as the extant counterpart. The fossil measurements are reproduced independently from the published drawings (cell counts to a median 0.0 %, median cell length to 3.4 %), which also exposes one table row that does not match its drawing and one swapped pair. Finding: median cell length stays at 0.78–0.99 mm across 150 Myr and a 6x range of wing length; the inhibition length is approximately, not strictly, conserved.

- **Model fidelity to source: 70** — The prediction under test is stated and operationalised exactly as it follows from the source model (constant cell size -> N ∝ L², median cell ∝ L⁰) and the measured quantity is the one Hoffmann et al. and Eshghi et al. both use (vein-bounded cell, length = maximum Feret diameter, verified by per-wing agreement with Table S1). The model itself is not run here (that is 2018-hoffmann-wing-venation.py), and log-log OLS is the script's own choice of test, not the source's.  
  evidence: `2026-eshghi-odonata-wing-cells.py` L36–43; `2026-eshghi-odonata-wing-cells.py` L137–142; `2026-eshghi-odonata-wing-cells.py` L404–413
- **Morphological realism / specimen match: 85** — Every image shown is a real specimen: the fossil venation drawings are rendered at true millimetre scale from their own scale bars and colour-coded by the re-measured cell length, the extant counterpart is Hoffmann et al.'s segmented Arigomphus / Aeshna wings, and the final animation frame places an Eocene and a living aeshnid at the same scale. The script's segmentation of the fossil drawings reproduces the authors' counts to a median 0.0 % (90 % of wings within 2 %). Not full marks because the extant wings are shown only as Hoffmann's domain boundaries, not photographs.  
  evidence: `odonata_fossil_resegmentation.png` L0–0; `2026-eshghi-odonata-wing-cells.py` L352–380; `2026-eshghi-odonata-wing-cells.py` L582–596
- **Parameter expressivity: 50** — No generative parameters: the script analyses measured wings. The groups (fossil / dragonfly / damselfly) and the fitted exponents are outputs, not controls. Neutral score, not applicable.  
  evidence: `2026-eshghi-odonata-wing-cells.py` L144–173
- **Morphometric validation: 85** — This is the script's strength. Cell count, median / mean / maximum cell length, cell area and circularity are re-measured on all 69 fossil drawings and compared wing by wing with the published Table S1 (median |ΔN|/N 0.0 %, median cell length 3.4 %, largest cell 0.9 %); extant counts are cross-checked against Hoffmann et al.'s independent segmentation of 42 shared species (median 3.1 %, r 0.992); disagreements are printed, not hidden (#7 drawing vs table row; #23/#24 swapped). The scale of each drawing is read from its own bar. Short of full marks because 4 of 69 drawings fall back to the tabulated wing length for calibration and the +3 % length bias is explained but not corrected.  
  evidence: `2026-eshghi-odonata-wing-cells.py` L416–446; `2026-eshghi-odonata-wing-cells.py` L448–483; `2026-eshghi-odonata-wing-cells.py` L265–349; `2026-eshghi-odonata-wing-cells.py` L935–946; `datasets/odonata/fossil_wing_resegmentation.csv` L1–70
- **Source and data traceability: 90** — Both papers carry DOIs, the data archive is a CC BY 4.0 Zenodo DOI, PBDB and GBIF query URLs and the retrieval date are recorded, the derived tables are vendored with a provenance README, and the two downloaded inputs are pinned by SHA-256 and verified on every run.  
  evidence: `2026-eshghi-odonata-wing-cells.py` L12–25; `2026-eshghi-odonata-wing-cells.py` L94–101; `2026-eshghi-odonata-wing-cells.py` L176–191; `datasets/odonata/README.md` L1–20
- **Code-to-model traceability: 70** — Each analysis step names the source it follows and the decoding of the drawings is documented where it happens; the cell-length convention (maximum Feret diameter, cells bounded by vein mid-lines) was established by matching the authors' table rather than quoted from their methods text, which the script did not have access to.  
  evidence: `2026-eshghi-odonata-wing-cells.py` L27–35; `2026-eshghi-odonata-wing-cells.py` L353–364; `2026-eshghi-odonata-wing-cells.py` L383–392
- **Reproducibility package: 70** — Deterministic, invocation documented with a no-download mode, inputs cached, resumed and hash-verified, every derived number written to CSV. Dependencies are lower-bound pins only and scikit-image, which the re-segmentation needs, is not in requirements.txt; the full run needs ~100 MB of downloads.  
  evidence: `2026-eshghi-odonata-wing-cells.py` L45–54; `2026-eshghi-odonata-wing-cells.py` L193–227; `2026-eshghi-odonata-wing-cells.py` L80–85; `requirements.txt` L1–7

Remediation:
- Add scikit-image to requirements.txt and pin versions for the Python 3.14 environment.
- Read the "mm" label with a proper template match so
- Report the discrepancies (#7,
- Correct the +3 % mid-line bias by eroding cells by half the measured vein width, or state the authors' convention once their methods text is available.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Odonata wing cells through deep time
- `odonata_fossil_resegmentation.png` — sha256 `ab6f3b9f89fbcff0…`, 286,190 B, 1663×1323 px

| measurement | value | reference | method |
|---|---:|---|---|
| cell count ∝ wing length^b, fossil | 1.7811 exponent | 2.0 | OLS slope of log10 cell count on log10 wing length, n=69 wings (Eshghi et al. 2026 tables) *(r = 0.739)* |
| median cell length ∝ wing length^b, fossil | -0.1673 exponent | 0.0 | OLS slope of log10 median cell length on log10 wing length |
| median cell length over wings, fossil | 0.7837 mm | — | median over wings of the per-wing median cell length (Eshghi et al. 2026 tables) *(IQR 0.67–0.94 mm)* |
| cell count ∝ wing length^b, extant_dragonfly | 1.111 exponent | 2.0 | OLS slope of log10 cell count on log10 wing length, n=290 wings (Eshghi et al. 2026 tables) *(r = 0.816)* |
| median cell length ∝ wing length^b, extant_dragonfly | 0.2171 exponent | 0.0 | OLS slope of log10 median cell length on log10 wing length |
| median cell length over wings, extant_dragonfly | 0.9885 mm | — | median over wings of the per-wing median cell length (Eshghi et al. 2026 tables) *(IQR 0.87–1.09 mm)* |
| cell count ∝ wing length^b, extant_damselfly | 1.0078 exponent | 2.0 | OLS slope of log10 cell count on log10 wing length, n=104 wings (Eshghi et al. 2026 tables) *(r = 0.872)* |
| median cell length ∝ wing length^b, extant_damselfly | 0.2765 exponent | 0.0 | OLS slope of log10 median cell length on log10 wing length |
| median cell length over wings, extant_damselfly | 0.8991 mm | — | median over wings of the per-wing median cell length (Eshghi et al. 2026 tables) *(IQR 0.83–0.99 mm)* |
| fossil drawings re-segmented | 69 count | 69 | fossil venation drawings in the Zenodo archive segmented by the script |
| scale bar label read from the drawing | 65 count | 69 | drawings whose scale-bar label was read from the glyphs and found consistent with the tabulated wing length *(the rest use the bar with a value inferred from the tabulated wing length, or that length alone)* |
| cell count, median |ΔN|/N vs Table S1 | 0.0 ratio | 0.0 | per-wing relative difference between the script's cell count and the published count, median over 69 wings *(90% of wings within 2%; max 194% (#7, drawing and table row disagree))* |
| median cell length, median |Δ| vs Table S1 | 0.0339 ratio | 0.0 | per-wing relative difference of the median cell length (maximum Feret diameter), median over wings *(signed mean +3.5% (cells bounded by vein mid-lines))* |
| largest cell length, median |Δ| vs Table S1 | 0.0094 ratio | 0.0 | as above for the largest cell of each wing |
| fossil median cell length, re-segmented | 0.8275 mm | 0.7837 | median over wings of the script's own per-wing median cell length |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.5.4
- scipy: 1.16.3
- pandas: 2.3.3
- scikit-image: 0.26.0
- matplotlib: 3.10.7
- Pillow: 12.0.0

Inputs: datasets/odonata/*.csv (vendored, provenance in its README) plus two downloads verified by SHA-256 in the script: the Eshghi et al. 2026 Zenodo archive (ac1288ba…034703, 89,961,457 bytes) and Hoffmann et al.'s Dragonfly_NA_WingSegmentation.tar.gz (50997be3…ccb232).
