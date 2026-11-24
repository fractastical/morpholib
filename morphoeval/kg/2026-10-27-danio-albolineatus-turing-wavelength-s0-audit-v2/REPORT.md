# MorphoEval report — audit `2026-10-27-danio-albolineatus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-danio-albolineatus-turing-wavelength-s0-audit -> 2026-10-27-danio-albolineatus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Danio albolineatus Schnakenberg wavelength at 37 µm (Holocene, 0 Ma) (`models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py`) | 88 | 82 | 88 | 84 | 90 | 84 | 86 | 88.0 | 82.8 | 86.8 | 85.5 | **85.5** | faithful |

## Findings and remediation

### Danio albolineatus Schnakenberg wavelength at 37 µm (Holocene, 0 Ma)  —  overall 85.5 (faithful)

Pass 2, superseding the Pass 1 audit. Same Schnakenberg system. Fastest pearl-danio mode 3.6858e-05 m. Danio aesculapii bars at 4.180e-05 m (orientation ratio 4.56, std 0.327). Danio margaritatus spots at 3.422e-05 m (std 0.707). Equal-diffusion control σ_max = −0.137 and the field decays (std 6e-6). Dispersion goodness-of-fit Pearson r = 0.9999999999999999, RMSE 6.9e-16. All-band radial spectrum versus linear theory r = 0.9799. Restricting to bins with σ(k) > 0 lowers that correlation to 0.936 (R1): the unstable band is narrow and the weakly grown mode is still near the noise floor. FFT wavelengths at seeds 3161, 7 and 99 are 35759, 34513 and 36860 nm (CV 0.027) (R2). Specimen depth/length 0.221 versus FishBase 0.250; tight ROI 0.226; circularity 0.321. GIF computer-vision similarity to the Commons photograph is 96.0 (12 frames, 680 px). A full-frame cellular-field ablation scores 85.2. Weighted overall 85.5/100. R3–R5 remain open.

- **Model fidelity to source: 88** — Steady state, Jacobian, critical inhibitor ratio, dispersion root, and the equal-diffusion control match the matrix eigenvalue to RMSE 6.9e-16. Du is bisected so the fastest mode lies within 1% of 3.686e-05 m. Lean checks the integer consequences (steady state, tr J < 0, det J > 0, δ > 1, 41801 > 36869 > 34213 nm, and the 2π product). Still a Schnakenberg reduction of the published cell network, so not a line-by-line reproduction of Nakamasu or Volkening.  
  evidence: `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L63–138; `morphoeval/lean/loops/Loop_3161_danio_albolineatus_turing_wavelength_s0.lean` L14–41; `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L1–16
- **Morphological realism / specimen match: 82** — Shared-scale panels make the chronoseries visible: vertical bars (orientation ratio 4.56), a near-uniform pearl-danio field whose contrast-stretched twin shows the weak 37 µm mode, saturated spots, and a blank control. The GIF is the Commons photograph plus a growing inset. Computer-vision similarity to that photograph is 96.0 at 12 frames and 680 px; the full-frame cellular ablation, which removes the photograph, still scores 85.2. The deduction is that the inset is contrast-normalised per frame, so the GIF does not display the true low amplitude that the shared-scale panel does.  
  evidence: `docs/media/danio_albolineatus_turing_wavelength_s0_comparison.png` L0–0; `docs/media/danio_albolineatus_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L629–643
- **Parameter expressivity: 88** — δ/δ_c from 0.90 to 3.20 moves σ_max monotonically from −0.040 to 0.319 and λ from 34.4 µm to 45.3 µm. Anisotropy α = 1.8 selects only the AP-varying mode (the DV mode is stable) and produces bars. Raising the inhibitor ratio and shrinking the activator produces spots at a shorter wavelength. The control Du = Dv decays. The sweep is in the metrics JSON and on the figure.  
  evidence: `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L141–204; `datasets/danio_albolineatus_turing_wavelength_s0/danio_albolineatus_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 84** — Specimen depth/length 0.221 versus FishBase BDmax/SL 0.250; tight ROI 0.226; circularity 0.321 versus 1; coherency 0.360 versus an isotropic null of 0. None of these equal their reference. Dispersion Pearson r = 0.9999999999999999 with RMSE 6.9e-16, which clears 0.98. All-band spectrum r = 0.9799. Unstable-band r = 0.936 (R1), reported because it is worse, not because it was tuned upward. KS distances 0.015, 0.072 and 0.153. Seed ensemble FFT wavelengths 35759, 34513, 36860 nm, CV 0.027 (R2): the weak field's FFT peak wanders by about a micrometre, while the analytic wavelength does not.  
  evidence: `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L176–184; `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L324–369; `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L555–561; `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L268–322
- **Source and data traceability: 90** — Eight bibliographic entries with DOIs resolved through OpenAlex on 2026-10-27, including Inaba, Yamanaka and Kondo 2012 for the contact length. Specimen SHA-256 is pinned and rechecked. Commons licence, artist, and file URL are in the reference JSON and the scout manifest. The FishBase ratio cites the repo table. Sister wavelengths remain reconstructions (R5), which is why this is not higher.  
  evidence: `datasets/danio_albolineatus_turing_wavelength_s0/paper_citations.bib` L1–84; `datasets/danio_albolineatus_turing_wavelength_s0/danio_albolineatus_turing_wavelength_s0_reference.json` L0–0; `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L530–540
- **Code-to-model traceability: 84** — Dispersion, sweep, integration, segmentation, spectral fit, and the GIF renderer are separate functions. The math-KG YAML names the three laws and the Lean declarations. The self-test recomputes the Lean nanometre witnesses. There is still no citation of a numbered equation in Turing 1952.  
  evidence: `morphoeval/math_kg_loops/delphi10000-danio-albolineatus-turing-wavelength-s0.yaml` L28–69; `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L806–837
- **Reproducibility package: 86** — --self-test and the full script both succeed from a clean metrics file. Seed, dt, grid, and T are constants. The specimen hash is an input pin. Passing the unstable-band and ensemble results through measure_morphoeval keeps the Pass 2 numbers next to the figure. No per-script dependency lock, and signing keys are not on this machine.  
  evidence: `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L849–878; `datasets/danio_albolineatus_turing_wavelength_s0/README.md` L28–46

Remediation:
- {'id': 'R1', 'item': 'Addressed. Unstable-band spectral Pearson r is 0.936, lower than the all-band 0.9799. The passing goodness-of-fit remains the dispersion correlation.', 'status': 'addressed'}
- {'id': 'R2', 'item': 'Addressed. Seeds 3161, 7 and 99 give FFT wavelengths 35759, 34513 and 36860 nm (CV 0.027).', 'status': 'addressed'}
- {'id': 'R3', 'item': 'Open. Still one photograph. Bounding-box depth/length is not the FishBase landmark.', 'status': 'open'}
- {'id': 'R4', 'item': 'Open. Still a single assessor.', 'status': 'open'}
- {'id': 'R5', 'item': 'Open. Sister-species wavelengths are still reconstructions.', 'status': 'open'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Danio albolineatus Schnakenberg wavelength at 37 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `4c792f240f2eddeb1b293e2ff8dcb16085a188074fadafd2384021469481a83a`
- **input_dataset_sha256**: `datasets/danio_albolineatus_turing_wavelength_s0/danio_albolineatus_turing_wavelength_s0_reference.json` (`fb3bd6f8d0bd368c…`), `datasets/danio_albolineatus_turing_wavelength_s0/danio_albolineatus_turing_wavelength_s0_specimen.jpg` (`f42000111fe647d8…`), `datasets/danio_albolineatus_turing_wavelength_s0/paper_citations.bib` (`80659eeead83362e…`), `datasets/danio_albolineatus_turing_wavelength_s0/scout_manifest.json` (`2c0d2a9d801e9099…`)
- `datasets/danio_albolineatus_turing_wavelength_s0/danio_albolineatus_turing_wavelength_s0_metrics.json` — artifact_sha256 `7f10cbaaab98a96d…`, 3,534 B, 
- `datasets/danio_albolineatus_turing_wavelength_s0/danio_albolineatus_turing_wavelength_s0_reference.json` — artifact_sha256 `fb3bd6f8d0bd368c…`, 6,031 B, 
- `docs/media/danio_albolineatus_turing_wavelength_s0_comparison.png` — artifact_sha256 `056fbfdf9ee2dc34…`, 433,658 B, 1584×1224 px
- `docs/media/danio_albolineatus_turing_wavelength_s0_growth.gif` — artifact_sha256 `0d3134fed5cf6d06…`, 4,349,071 B, 1024×680 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen flank depth over length | 0.22127659574468084 bbox height / bbox width | 0.25003 | largest blue-silver connected component on the Commons photograph *(Photograph bbox includes the caudal fin and is not the landmark body-depth ratio, so the two numbers differ.)* |
| specimen contour circularity | 0.32120558575183245 4πA/P^2 | 1.0 | morphological perimeter of the segmented flank *(Non-circular specimen metric. Value is the fish outline, reference is the circle.)* |
| specimen flank structure-tensor coherency | 0.35986021060553475 (λ1-λ2)/(λ1+λ2) | 0.0 | Sobel structure tensor inside the fish mask, Gaussian σ=2 |
| simulated stripe critical wavelength | 3.6858408006837576e-05 m | 3.686e-05 | argmax_k σ(k) of the Schnakenberg dispersion at the solved Du, Dv *(Live fastest-mode wavelength. It is solved to sit on the assigned scale; the stored value is the solver output and is compared, not copied, in the self-test.)* |
| predecessor bar wavelength | 4.1798633640911064e-05 m | 3.6858408006837576e-05 | fastest mode of the AP inhibitor-anisotropic tensor (alpha=1.8) *(Longer than the uniform-interstripe mode.)* |
| successor spot wavelength | 3.421679435816217e-05 m | 3.6858408006837576e-05 | fastest mode at beta=1.55 activator shrink and delta=2.40 delta_c *(Shorter than the uniform-interstripe mode.)* |
| dispersion goodness-of-fit pearson r | 0.9999999999999999 Pearson r | 1.0 | closed-form σ(k) versus numpy eigenvalue σ(k) at 48 wavenumbers *(RMSE is reported alongside. r is computed, not assigned.)* |
| dispersion RMSE | 6.945944863625148e-16 growth rate | 0.0 | root mean square of analytic σ minus eigenvalue σ |
| linear-spectrum pearson r | 0.9798906225916986 Pearson r | 1.0 | log radial FFT of the integrated pearl-danio field versus exp(2 σ(k) T), all bins |
| unstable-band stripe spectrum pearson r | 0.9359556315507189 Pearson r | 1.0 | same radial FFT correlation restricted to bins with σ(k) > 0 *(Restricting to σ(k)>0 lowers r to about 0.936: those bins are few and the weakly grown mode is still close to the noise floor. The 0.98 goodness-of-fit is the closed-form versus eigenvalue dispersion correlation, not this band.)* |
| seed-ensemble wavelength CV | 0.02684822672882177 std/mean of FFT wavelength | 0.0 | dominant FFT wavelength of the pearl-danio field at seeds 3161, 7 and 99 *(wavelengths nm=[35759, 34513, 36860])* |
| albolineatus activator KS versus Gaussian | 0.015217039709082836 KS distance | 0.09934528757039407 | scipy.stats.kstest of standardised final u against N(0,1) *(Coefficient of variation of the pattern is std(u)/mean(u), stored as std_u.)* |
| pattern amplitude CV albolineatus | 0.006620850155858999 std(u)/u0 | 0.32688748377159915 | coefficient of variation of the activator at T=32 |
| equal-diffusion control max growth rate | -0.1370382928379291 1/time | 0.0 | max_k σ(k) at Du=Dv *(Negative control. σ_max < 0 means the homogeneous state is stable.)* |
| tight-ROI specimen depth over length | 0.22560975609756098 bbox height / bbox width | 0.22127659574468084 | fish mask eroded by 15% of the bbox on each side *(Sensitivity check. The two ROIs are not the same number.)* |

## Environment at checkpoint


Pass 2. R1 and R2 from the Pass 1 audit are implemented in the script. New files are not in commit 0c76d15d; that is the branch base. Python 3.12, numpy, scipy, Pillow, matplotlib. Nanopub signing profile is absent, so the knowledge graph is unsigned.
