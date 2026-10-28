# MorphoEval report — audit `2026-10-28-t1917-014-the-forms-of-cells-audit-v2`

Rubric **0.1.0** · checkpoint `a1dca587c` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-28-t1917-014-the-forms-of-cells-audit -> 2026-10-28-t1917-014-the-forms-of-cells-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-014 — Lagena and other unilocular tests vs. the Plateau/Delaunay catalogue (Pass 2) (`thompson/claims/t1917-014_the-forms-of-cells.py`) | 80 | 62 | 72 | 70 | 88 | 80 | 78 | 77.3 | 65.0 | 81.9 | 73.8 | **73.8** | mostly faithful |

## Findings and remediation

### Thompson (1917) T1917-014 — Lagena and other unilocular tests vs. the Plateau/Delaunay catalogue (Pass 2)  —  overall 73.8 (mostly faithful)

v2 keeps the exact Delaunay catalogue of v1 and adds everything Pass 1 lacked: the Kireenko (2022) CC-BY photographs are segmented (Otsu, largest component, plate-wide 100 um scale-bar calibration), each silhouette is fitted by E7 flask, E8 bead, an ellipsoid null, a sphere+cylinder ablation and a free-form degree-6 reference, with nRMSE, Pearson r, KS statistic and 2-fold CV; a B/t regime sweep tabulates the catalogue; polythalamous controls are run through the same pipeline; result.json records the verdict. Result: MIXED — 11/18 unilocular tests compatible (flask-like 11/13 at median nRMSE 0.050 vs free-form 0.049, null 0.116, ablation 0.148; spindles 0/5 at 0.209), Delaunay beats the ablation on 17/18 and the null on 13/18, 1/2 controls pass (specificity caveat). Overall 73.8/100.

- **Model fidelity to source: 80** — Catalogue implemented from the Kenmotsu first-integral form with all five members and the correct limits, self-tested against the root identities, the elliptic period and the closed-form flask join radius; 18 Lean theorems (0 sorry) check the integer algebra including the bead aspect bound and join feasibility. The sphere+cylinder ablation shows the unduloid shoulder is load-bearing (17/18 tests fit worse without it). Deductions: join is C1 not C2, neck has no apertural lip, the nodoid branch is characterised in the sweep but not offered as a fit model.  
  evidence: `thompson/claims/t1917-014_the-forms-of-cells.py` L141–206; `thompson/claims/t1917-014_the-forms-of-cells.py` L209–262; `morphoeval/lean/loops/Thompson1917_T1917_014.lean` L32–221; `thompson/claims/t1917-014_the-forms-of-cells.py` L503–526
- **Morphological realism / specimen match: 62** — Real specimens are now compared side by side (panels E-F): 4/5 photographed unilocular tests (Fissurina 57a/58a, Lagena 62a, Oolina 63) are traced by the Delaunay composite to within 0.018-0.051 nRMSE, Lagena 61a fails (0.083 vs free-form 0.018: its neck tapers like a cone rather than an unduloid), and the 13 drawings split flask/spindle as in Pass 1. Deductions: only one plate and five unilocular photographs; the egg-shaped Quinqueloculina control also matches, so the visual match is not specific to single-chambered tests; the drawn spindles are visibly not reproduced (the honest negative).  
  evidence: `docs/media/t1917_014_the_forms_of_cells_comparison.png` L0–0; `thompson/claims/t1917-014_the-forms-of-cells.py` L309–352; `datasets/thompson_1917/t1917-014/t1917_014_metrics.json` L0–0
- **Parameter expressivity: 72** — parameter_sweep tabulates catalogue member, neck ratio (strictly decreasing, asserted), period/(2a) and bead aspect ratio (within [1, pi], asserted) over 14 values of B including the nodoid branch (axial back-tracking detected), and the flask join radius over t for three B values; the GIF animates B and L_neck. Fitted B spans 0.30-0.85 across specimens. Deductions: no CLI to override parameters; the t sweep is plotted for one B only.  
  evidence: `thompson/claims/t1917-014_the-forms-of-cells.py` L503–526; `thompson/claims/t1917-014_the-forms-of-cells.py` L790–822; `thompson/claims/t1917-014_the-forms-of-cells.py` L934–940
- **Morphometric validation: 70** — Every silhouette gets nRMSE, Pearson r, KS statistic and 2-fold CV for five models; the Delaunay residual is judged against the per-specimen free-form floor (capped at 0.10) rather than a fixed number; the ellipsoid null (median 0.116) and the sphere+cylinder ablation (0.148) are both clearly worse than Delaunay (0.057); photograph sizes are calibrated from the scale bars (Lagena 186-276 um, Oolina 224 um, Fissurina 204-227 um) and cross-checked against the typical unilocular size range. Deductions: n = 2 negative controls, one of which passes; KS p-values are dominated by the row count and only the statistic is informative; no published Lagena outline morphometrics exist to compare the fitted B distribution against.  
  evidence: `thompson/claims/t1917-014_the-forms-of-cells.py` L437–500; `datasets/thompson_1917/t1917-014/result.json` L0–0; `thompson/claims/t1917-014_the-forms-of-cells.py` L99–107
- **Source and data traceability: 88** — Claim text and figure pinned to IA leaf 0276 with JP2 and crop SHA-256; photograph pinned to the Commons file (sha1 and SHA-256) with DOI, licence and caption; both inputs re-verified at load time; Dryad mismatch documented with pinned metadata; OpenAlex query pinned. Deduction: Kenmotsu/Kireenko DOIs are not resolved programmatically in this round.  
  evidence: `thompson/claims/t1917-014_the-forms-of-cells.py` L130–137; `datasets/thompson_1917/t1917-014/scout_manifest.json` L0–0; `datasets/thompson_1917/t1917-014/paper_citations.bib` L1–105
- **Code-to-model traceability: 80** — Module docstring maps E3-E8 and the comparators to pages 216-263; every model function names its equation; the decision rule is spelled out with its three constants; the math KG YAML and Lean module cite v2 line ranges. The constants 0.06 / 1.5 / 0.10 / 0.12 remain judgement calls, now documented as such.  
  evidence: `thompson/claims/t1917-014_the-forms-of-cells.py` L1–62; `morphoeval/math_kg_loops/thompson1917-t1917-014.yaml` L0–0; `thompson/claims/t1917-014_the-forms-of-cells.py` L112–115
- **Reproducibility package: 78** — Deterministic (no RNG), --self-test 4 s covering identities, joins, sweep invariants, synthetic recovery, comparator ordering and both segmentations; full run 11 s writes figure, GIF, metrics and result.json; environment versions recorded here. Deductions: no per-script pinned requirements file; repo .gitignore excludes PNGs so the figure and the Fig. 84 crop must be force-added.  
  evidence: `thompson/claims/t1917-014_the-forms-of-cells.py` L885–931; `thompson/claims/t1917-014_the-forms-of-cells.py` L934–957

Remediation:
- {'id': 'R1', 'item': 'Independent second assessor (different provider) — both passes are builder-scored'}
- {'id': 'R8', 'item': 'More photographed unilocular tests (Kireenko 2022 Fig. 19: Lagena 112/113, Procerolagena 114, Oolina 115/116) and more polythalamous controls so the specificity caveat rests on n > 2'}
- {'id': 'R9', 'item': 'Offer the nodoid (B > 1) and the catenoid as fit models so every catalogue member is tested, not only characterised'}
- {'id': 'R10', 'item': 'Derive the nRMSE gate from a segmentation-noise model (e.g. bootstrap over threshold 0.035-0.055) instead of fixed constants'}
- {'id': 'R11', 'item': 'Test the surface-of-revolution premise in 3-D (apertural views 61b/62b are nearly circular; quantify their circularity)'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-014 — Lagena and other unilocular tests vs. the Plateau/Delaunay catalogue (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `96b8d320bb6057308c24557dd6f3e3dce73bc0bc00dfdf30f45d3f4ca513f1dc`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-014/kireenko_2022_bdj_e87457_fig8_station5656mc.jpg` (`3b57acc0dfc98374…`), `datasets/thompson_1917/t1917-014/paper_citations.bib` (`a21c514d2fded5d2…`), `datasets/thompson_1917/t1917-014/scout_manifest.json` (`53e6b01a4f0e6102…`), `datasets/thompson_1917/t1917-014/t1917_014_the_forms_of_cells_reference.json` (`0a5ac92486643991…`), `datasets/thompson_1917/t1917-014/thompson_1917_fig84_lagena_after_brady.png` (`c02d29af384791df…`), `datasets/thompson_1917/t1917-014/thompson_1917_leaf0276_p256_fig84.jp2` (`ed440583a8b144d5…`)
- `datasets/thompson_1917/t1917-014/result.json` — artifact_sha256 `637a5df7a7141bf4…`, 3,343 B, 
- `datasets/thompson_1917/t1917-014/scout_manifest.json` — artifact_sha256 `53e6b01a4f0e6102…`, 9,226 B, 
- `datasets/thompson_1917/t1917-014/t1917_014_metrics.json` — artifact_sha256 `b0f3e32ae986527f…`, 70,771 B, 
- `docs/media/t1917_014_the_forms_of_cells_comparison.png` — artifact_sha256 `c1a3b83ef9c29710…`, 1,300,317 B, 2200×2000 px
- `morphoeval/math_kg_loops/thompson1917-t1917-014.yaml` — artifact_sha256 `49bced3e9b340f52…`, 10,581 B, 

| measurement | value | reference | method |
|---|---:|---|---|
| unilocular test silhouettes analysed (13 drawings + 5 photographs) | 18 count | 18 | Fig. 84 background-normalised segmentation; Kireenko plate Otsu segmentation |
| Delaunay-compatible unilocular tests | 11 count | 18 | best of E7/E8 with nRMSE <= max(0.06, 1.5 x free-form) and asymmetry <= 0.12 *(verdict mixed; fraction 0.61)* |
| median best Delaunay RMSE / r_max (tests) | 0.05701286222506893 dimensionless | 0.049349736256689175 | RMSE over all silhouette rows *(reference = free-form degree-6 noise floor)* |
| median ellipsoid null RMSE / r_max (tests) | 0.11577851450872066 dimensionless | 0.05701286222506893 | null model vs Delaunay baseline |
| median sphere+cylinder ablation RMSE / r_max (tests) | 0.14803657580333934 dimensionless | 0.05701286222506893 | unduloid shoulder ablated (C0 join) vs Delaunay baseline |
| tests where Delaunay beats the ellipsoid null | 13 count | 18 | nRMSE comparison |
| median 2-fold CV RMSE / r_max (best Delaunay) | 0.057012463389534036 dimensionless | — | fit on even rows, evaluate on odd rows and vice versa |
| median KS statistic observed vs fitted radii (best Delaunay) | 0.21800646403520837 dimensionless | — | scipy.stats.ks_2samp |
| median Pearson r observed vs fitted radius | 0.9860352328913662 dimensionless | — | np.corrcoef |
| median bilateral asymmetry index mean|r_L - r_R| / r_max | 0.046377228937456674 dimensionless | 0.0 | half-widths about the least-squares axis *(0 for a perfect surface of revolution)* |
| flask-like tests (L/W <= pi) Delaunay-compatible | 11 count | 13 | subset within the unduloid bead aspect bound |
| spindle tests (L/W > pi) Delaunay-compatible | 0 count | 5 | subset outside the bead aspect bound (E6, Lean bead_aspect_bound) |
| polythalamous negative controls Delaunay-compatible | 1 count | 0 | Globobulimina 64a, Quinqueloculina 66a fitted with the same pipeline *(control median nRMSE 0.042)* |
| Lagena sp. test length from 100 um scale bar | 230.77825929938606 um | 300.0 | scale-bar calibrated silhouette length (61a, 62a) *(order-of-magnitude check: Lagena tests are typically 0.15-0.6 mm (Jones 1984 key to unilocular hyaline forams))* |
| fitted Kenmotsu B range | 0.30-0.85 dimensionless | — | best catalogue member per specimen |

## Environment at checkpoint

- python: 3.12
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0
- scikit-image: 0.26.0
- lean: 4.35.0

Branch cursor/t1917-014-the-forms-of-cells-8998; Pass 2 script committed as a1dca587c on top of origin/main fd43e5aba. Lean module unchanged since Pass 1 and re-checked (exit 0, 18 theorems, 0 sorry). --self-test 4 s, full run 11 s.
