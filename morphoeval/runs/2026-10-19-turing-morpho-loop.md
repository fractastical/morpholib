# 12-Step Closed-Loop Walkthrough: Turing (1952), Pearson (1993) & Kondo–Asai (1995) Reaction–Diffusion Morphogenesis (`1952-turing-morpho.py`)

**Run ID:** `2026-10-19-turing-morpho-loop`  
**Target Simulation:** `turing-gray-scott` (`1952-turing-morpho.py`)  
**Primary Literature:**
- Turing AM (1952). *The chemical basis of morphogenesis.* Phil. Trans. R. Soc. B 237(641):37–72. [doi:10.1098/rstb.1952.0012](https://doi.org/10.1098/rstb.1952.0012)
- Pearson JE (1993). *Complex patterns in a simple system.* Science 261(5118):189–192. [doi:10.1126/science.261.5118.189](https://doi.org/10.1126/science.261.5118.189)
- Lee KJ, McCormick WD, Pearson JE, Swinney HL (1994). *Experimental observation of self-replicating spots in a reaction-diffusion system.* Nature 369(6477):215–218. [doi:10.1038/369215a0](https://doi.org/10.1038/369215a0)
- Kondo S, Asai R (1995). *A reaction-diffusion wave on the skin of the marine angelfish Pomacanthus.* Nature 376(6543):765–768. [doi:10.1038/376765a0](https://doi.org/10.1038/376765a0)

![Turing Closed-Loop Telemetry Animation](../../docs/media/turing_morpho_closed_loop.gif)

---

## 1. Executive Summary

| Metric | Pass 1 (v1 Baseline) | Pass 2 (v2 Remediated) | Delta |
|---|---:|---:|---:|
| **Primary Assessor (`gemini-4-argon`)** | `50.9 / 100` | `91.9 / 100` | **`+41.0`** |
| **2nd Assessor (`claude-sonnet-5-5-high`)** | `25.1 / 100` | `79.1 / 100` | **`+54.0`** |
| **Dual-Agent Consensus Score** | **`38.0 / 100`** | **`85.5 / 100`** | **`+47.5`** |
| **Model Fidelity (`w=20%`)** | `54` | `92` | `+38` |
| **Specimen Match (`w=25%`)** | `14` | `82` | `+68` |
| **Parameter Expressivity (`w=10%`)** | `38` | `81` | `+43` |
| **Morphometric Validation (`w=15%`)** | `30` | `75` | `+45` |
| **Source Traceability (`w=10%`)** | `45` | `94` | `+49` |
| **Code Traceability (`w=8%`)** | `45` | `89` | `+44` |
| **Reproducibility Package (`w=12%`)** | `61` | `89` | `+28` |
| **Lean 4 Theorems (`MathOfBiology.lean` Sec. 5)** | `1` | `4` (0 `sorry`) | `+3` |
| **Signed Nanopublications** | `10` (`2026-10-19-turing-audit`) | `10` (`2026-10-19-turing-audit-v2`, via `npx:supersedes`) | `20` total |

---

## 2. Pass 1 Diagnosis (Steps 1–11)

During **Step 7 (Dual-agent audit v1)**, the independent second assessor (`claude-sonnet-5-5-high`, subagent `644535e1-add1-4d86-bd2b-a2b4d6caea92`) audited the baseline 595-line `1952-turing-morpho.py` and identified five substantive defects:

1. **Missing Turing (1952) Linear Dispersion & Ring of 20 Cells:** Although the script's filename and docstring cited Turing (1952), it contained zero equations from Turing's paper—no 2-morphogen linearized reaction-diffusion Jacobian $J_0$, no Fourier dispersion relation $\sigma(k^2)$, no critical diffusivity ratio $d_c$, no analytical wavelength $\lambda_c = 2\pi / k_c$, and no simulation of Turing's Section 9 ring of $N=20$ cells.
2. **Missing Pearson (1993) Bifurcation Curves & Flawed $D_u = D_v$ Control:** The Saddle-Node parabola $4(F+k)^2 = F$ and Hopf locus $\operatorname{tr}(J_0) = 0$ were omitted; 3 of the 4 hardcoded regimes lay outside the Saddle-Node curve where no nontrivial homogeneous steady state exists; and the equal-diffusion control ($D_u = D_v = 0.10$) halved $D_u$ and tested finite-amplitude front extinction toward the always-stable trivial state $(1,0)$ rather than linear Turing stability around a nontrivial homogeneous steady state $(U_*, V_*)$.
3. **Limited Parameter Expressivity:** Only 4 patterned points were hardcoded without Pearson's 12 pattern classes ($\alpha$–$\mu$), without a diffusivity-ratio sweep across $d = D_u / D_v$, and without CLI overrides (`--F`, `--k`, `--Du`, `--Dv`).
4. **Zero Biological Specimen Validation (`specimen-match: 14`):** No biological specimen photograph or computer-vision segmentation of *Pomacanthus* marine angelfish skin stripes / Y-junctions (Kondo & Asai 1995 *Nature* 376:765–768) or spotted boxfish hexagonal arrays was present.
5. **Missing Formal Reference Package & Unsynchronized Outputs:** `datasets/turing/` did not exist, `turing_patterns_black_neon.gif` was out of sync with `docs/media/turing_gray_scott.gif`, and `morphoeval/lean/MathOfBiology.lean` Section 5 had only 1 trivial equilibrium check.

---

## 3. Pass 2 Remediation (Step 12 $\to$ Steps 3–11)

1. **Step 3 (Formal Source — `datasets/turing/`):**
   - Vendored `datasets/turing/turing_reference.json` (`SHA-256: c7af2f0182e306757cedc56fe5e575bf27993fc813f065f0ec2d281a3f447eac`) containing Turing's (1952) Section 8–10 Jacobian parameters and $N=20$ ring mode spectrum, Pearson's (1993) 12 pattern classes ($\alpha$–$\mu$), and Kondo & Asai's (1995) ontogenetic *Pomacanthus* stripe-insertion growth series.
   - Vendored CC BY-SA 4.0 biological specimen photographs of adult *Pomacanthus imperator* (`datasets/turing/pomacanthus_imperator_stripes_specimen.jpg`) and *Ostracion meleagris* (`datasets/turing/ostracion_meleagris_spots_specimen.jpg`) with runtime SHA-256 verification.
2. **Step 4 (Pin the Rules in Lean 4 — `morphoeval/lean/MathOfBiology.lean` Section 5):**
   - Formalized and kernel-verified 4 theorems (`0 errors, 0 sorry`):
     - `gray_scott_homogeneous_equilibrium` (`turing-gray-scott-1`)
     - `turing_equal_diffusion_no_instability` (`turing-gray-scott-2`)
     - `turing_differential_diffusion_instability` (`turing-gray-scott-3`, matching the exact $4\times$-scaled Jacobian $(a=4, b=-6, c=5, d=-6)$ of `1952-turing-morpho.py`)
     - `pearson_saddle_node_discriminant` (`turing-gray-scott-4`, proving both $\Delta_{\text{SN}} = F - 4s^2 = 0$ on $F = 4(F+k)^2$ and the steady-state Hopf trace identity $\operatorname{tr}(J_0) = k - V_+^2 = 0$ on $V_+^2 = k$).
3. **Step 5 & Step 6 (Upgraded `1952-turing-morpho.py` & 15-Panel Synthesis Figure):**
   - Implemented `compute_turing_1952_dispersion_and_ring()` ($d_c = 3.927051$, $k_c = 1.1180\text{ rad/cell}$, $\lambda_c = 5.6199\text{ cells}$, dominant ring mode $s^* = 4$ with 4 stationary morphogen peaks).
   - Implemented `compute_pearson_bifurcation_curves()` with the exact Hopf locus $F_{\text{Hopf}}(k) = \frac{1}{2}(\sqrt{k} - 2k - \sqrt{k(1 - 4\sqrt{k})})$ ($\max |\operatorname{tr}(J_0)| = 2.36\times 10^{-16} < 10^{-12}$), plotted all 12 Pearson classes ($\alpha$–$\mu$), paired a genuine steady-state Turing labyrinth ($F=0.054, k=0.0616, D_u=0.20, D_v=0.10, \sigma(k_c) = +0.00164, \lambda^* = 13.92\text{ px}$ vs. $\lambda_c = 11.92\text{ px}$) with a steady-state equal-diffusion control ($D_u = D_v = 0.20, \sigma = -0.00224, \sigma_V = 1.91\times 10^{-7}$), and added a 5-point diffusivity-ratio sweep across $d \in \{1.0, 1.25, 1.5, 2.0, 2.5\}$.
   - Implemented `segment_biological_turing_specimens()` and `simulate_kondo_asai_growing_domain_stripes()` to segment *Pomacanthus imperator* stripes (spacing $9.55\text{ px}$, $\text{CV} = 0.149$, 22 Y-branch dislocations) and *Ostracion meleagris* spots (121 DoG spots) and test Kondo & Asai's (1995) stripe-insertion scaling law via a white-noise-initialized 1D Turing PDE across growing domains ($r = 0.9981$).
4. **Steps 7–11 (Re-Audit, Re-Measure & Cryptographic Superseding):**
   - Re-audited with `claude-sonnet-5-5-high` (`826c6028-10df-4445-bb20-774a141618ed`), raising the 2nd-assessor score from `25.1` to `79.1` and the dual-agent consensus score from **`38.0 / 100` to `85.5 / 100` (`+47.5`)**.
   - Emitted and signed 10 v2 nanopubs in `morphoeval/kg/2026-10-19-turing-audit-v2/signed/`, each linked via `npx:supersedes` to the Pass 1 trusty URIs.
