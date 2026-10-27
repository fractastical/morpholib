# MorphoEval report — audit `2026-10-27-antirrhinum-majus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Antirrhinum majus Turing ocellus wavelength at 259 µm (Holocene, 0 Ma) (`models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py`) | 86 | 80 | 91 | 84 | 90 | 86 | 88 | 87.7 | 81.5 | 88.1 | 85.3 | **85.3** | faithful |

## Findings and remediation

### Antirrhinum majus Turing ocellus wavelength at 259 µm (Holocene, 0 Ma)  —  overall 85.3 (faithful)

Two-species activator-inhibitor system on a periodic petal-epidermis patch. Rates f_u, f_v, g_u, g_v = 1.25, -1.50, 2.00, -1.75 per hour; D_u, D_v = 640, 4341 µm²/h. Trace -0.50, determinant 0.8125, Turing margin positive, equal-diffusion control σ_max = -0.25 h⁻¹. k_c = argmax σ(k) gives λ_c = 259.13 µm at 0 Ma (assigned 259.2 µm). A joint diffusivity sweep (×0.35 to ×4) recovers k with Pearson r = 0.9992 and RMSE 0.00114 µm⁻¹. The gallery field's 100 µm bar (111 px) measures a ring pitch of 77.8 µm; KS against the simulated ring gaps is 1.0 because the two pitches do not overlap. GIF is 800×600, 12 frames. Computer-vision similarity of the last frame to the gallery field is 83.6. Overall 85.3. The radial PSD still reads k about 7% low at unit scale (remediation R1).

- **Model fidelity to source: 86** — The script integrates the stated Jacobian, the dispersion σ² − tr(k)σ + Δ(k) = 0, cubic saturation, and the equal-diffusion ablation. k_c is argmax σ(k), which is the selected wavelength off onset; the shortcut k² = h/(2 D_u D_v) is documented as a different quantity. Ten Lean theorems with zero sorry encode the integer trace, determinant, Turing margin, and the ablation. Deduction: the rates are reconstructed for this 259 µm slot, not fitted in a snapdragon paper, and the radial PSD that reads k back is biased (R1).  
  evidence: `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L124–187; `morphoeval/lean/loops/Loop_3114_antirrhinum_majus_turing_wavelength_s0.lean` L24–42; `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L197–243
- **Morphological realism / specimen match: 80** — The last GIF frame is a green radial ocellus in the same 800×600 framing as the gallery field. Direct CV similarity is 83.6 (orientation, radial FFT, coherency, occupancy). The gallery pitch measured from its 100 µm bar is 77.8 µm, while the simulation is one 259.13 µm wave per ring, so the resemblance is of pattern class, not of physical pitch. The CC BY-SA Sète flower is an organ-scale voucher and is not the image the GIF is scored against.  
  evidence: `docs/media/antirrhinum_majus_turing_wavelength_s0_comparison.png` L0–0; `docs/media/antirrhinum_majus_turing_wavelength_s0_growth.gif` L0–0; `datasets/antirrhinum_majus_turing_wavelength_s0/antirrhinum_majus_turing_wavelength_s0_metrics.json` L0–0
- **Parameter expressivity: 91** — Scaling D_u and D_v together from ×0.35 to ×4 moves λ_c from 153 µm to 518 µm. Measured Fourier peaks track analytical k_c with Pearson r = 0.9992 and RMSE 0.00114 µm⁻¹. The equal-diffusion ablation is outside that sweep and stays stable, so the wavelength control is the diffusivity scale, not a sign error.  
  evidence: `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L255–279; `docs/media/antirrhinum_majus_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 84** — The gallery bar is detected as a 111 px white run and read as 100 µm. Four radial peaks give pitch 77.8 µm (CV of the gaps), foreground fraction, circularity, and structure-tensor coherency. Those values are not set equal to the model wavelength, so the comparison is not self-referential. KS of ring-gap samples is 1.0. Pearson r and RMSE are on the diffusivity sweep. Deduction: at unit scale the PSD peak reports λ ≈ 279 µm against λ_c = 259 µm (about 7% low in k). That bias is R1, not a hidden residual.  
  evidence: `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L423–488; `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L513–535; `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L716–743
- **Source and data traceability: 90** — Turing 1952, Shang 2010, Schwinn 2006, Noda 1994, and Coen 2004 were checked against Crossref on 2026-10-27 (OpenAlex returned 429). Both images are SHA-256 pinned. The flower file is the CC BY-SA 4.0 Sète photograph. Deduction: the gallery field is a campaign asset with a drawn scale bar, not a DOI-backed micrograph.  
  evidence: `datasets/antirrhinum_majus_turing_wavelength_s0/paper_citations.bib` L1–70; `datasets/antirrhinum_majus_turing_wavelength_s0/antirrhinum_majus_turing_wavelength_s0_reference.json` L48–62
- **Code-to-model traceability: 86** — The module docstring states the PDE, the trace, the determinant, and the distinction between argmax σ(k) and the Δ-minimizer. trace_det, sigma_of_k2, k_c_and_sigma, and grow_ocellus follow those lines. The Lean file names the same integers.  
  evidence: `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L25–54; `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L310–360
- **Reproducibility package: 88** — --self-test checks trace, determinant, Turing margin, λ_c within 1% of 259.2 µm, Pearson r > 0.98, the ablation, the 111 px bar, and the sub-wavelength cell. Seed 3114. Dependencies are numpy, scipy, pillow, matplotlib. Deduction: no per-script lockfile; the checkpoint commit does not yet contain these files.  
  evidence: `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L767–805; `datasets/antirrhinum_majus_turing_wavelength_s0/README.md` L28–32

Remediation:
- {'id': 'R1', 'item': 'Radial PSD uses integer pixel bins and reads k about 7% low at unit diffusivity (measured λ ≈ 279 µm vs λ_c 259.13 µm). Re-bin in physical wavenumber and re-measure Pearson r and RMSE.'}
- {'id': 'R2', 'item': 'Single assessor. A second provider has not scored this script.'}
- {'id': 'R3', 'item': 'Annotate the kink in σ(k) at the edge of the unstable band so it is not mistaken for a numerical glitch.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Antirrhinum majus Turing ocellus wavelength at 259 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `9aa96d9174bbc264f2f19508483526998fd8a90a8bb5b00ee7c2a039134c8ca4`
- **input_dataset_sha256**: `datasets/antirrhinum_majus_turing_wavelength_s0/antirrhinum_majus_sete_cc-by-sa.jpg` (`4d2cd99ca49de264…`), `datasets/antirrhinum_majus_turing_wavelength_s0/antirrhinum_majus_turing_wavelength_s0_reference.json` (`0918abca4751ad6b…`), `datasets/antirrhinum_majus_turing_wavelength_s0/antirrhinum_majus_turing_wavelength_s0_specimen.jpg` (`f3e9bf78a4453bc1…`)
- `datasets/antirrhinum_majus_turing_wavelength_s0/antirrhinum_majus_turing_wavelength_s0_metrics.json` — artifact_sha256 `f2156e4fe25bc43a…`, 4,677 B, 
- `docs/media/antirrhinum_majus_turing_wavelength_s0_comparison.png` — artifact_sha256 `836c3bdba61945c9…`, 533,696 B, 1664×1092 px
- `docs/media/antirrhinum_majus_turing_wavelength_s0_growth.gif` — artifact_sha256 `7ec4b0240db6d0c7…`, 3,276,935 B, 800×600 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| analytical critical wavelength lambda_c | 259.1268 µm | 259.2 | 2π / argmax_k σ(k) with D_u=640, D_v=4341 µm²/h *(D_v=4341 places argmax σ(k) 0.02 µm under the assigned 259.2 µm)* |
| diffusivity-scale sweep Pearson r of measured vs analytical k_c | 0.99921 correlation | 0.98 | 8-point joint D_u, D_v scale sweep, Fourier-space linear evolution, radial PSD peak *(RMSE(k) = 0.001144 µm⁻¹)* |
| diffusivity-scale sweep RMSE of wavenumber | 0.001144 µm⁻¹ | 0.005 | sqrt mean square error between analytical k_c and Fourier peak |
| equal-diffusion ablation sigma_max | -0.25 h⁻¹ | 0.0 | same linear solver with D_u = D_v = 640 *(spatial std ratio after growth = 0.0041)* |
| specimen ocellus ring spacing | 77.7778 µm | 259.1268 | radial gray peaks × (100 µm / detected scale-bar pixels) on the gallery field *(CV = 0.047, bar = 111 px, coherency = 0.236)* |
| specimen image foreground fraction | 0.6837 fraction | 0.5 | contrast-to-border mask on the gallery ocellus |
| specimen image structure-tensor coherency | 0.2359 dimensionless | 0.25 | mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor inside the mask |
| domain-growth mode-count Pearson r | 1.0 correlation | 0.98 | N = k_measured L / 2π versus L/λ_c from one wavelength to a 10 mm lobe *(N(10 mm) = 37.09)* |
| successor petal mode count at 10 mm | 38.591 wavelengths | 1.0 | 10000 µm / λ_c |

## Environment at checkpoint


Pass 1 working tree on top of 0c76d15d. Loop files are not in that commit. Signing was not available (~/.nanopub/profile.yml absent).
