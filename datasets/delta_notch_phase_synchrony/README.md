# Zebrafish Delta–Notch Delayed Phase Synchrony Dataset (`datasets/delta_notch_phase_synchrony/`)

Reference parameters, delayed Kuramoto phase-oscillator equations, collective frequency shift and linear stability criteria, DAPT initiation/failure/rescue perturbation regimes, and biological specimen photography for **Slot 023 (`slot-023-delta-notch-phase-synchrony`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing:

- **Riedel-Kruse, I. H., Müller, C. & Oates, A. C. (2007).** *"Synchrony dynamics during initiation, failure, and rescue of the segmentation clock."* *Science* 317(5846):1911–1915. DOI: `10.1126/science.1142538`
- **Morelli, L. G., Ares, S., Herrgen, L., Schröter, C., Jülicher, F. & Oates, A. C. (2009).** *"Delayed coupling theory of vertebrate segmentation."* *HFSP Journal* 3(1):55–66. DOI: `10.2976/1.3027088`
- **Herrgen, L., Ares, S., Morelli, L. G., Schröter, C., Jülicher, F. & Oates, A. C. (2010).** *"Intercellular coupling regulates the period of the segmentation clock."* *Current Biology* 20(14):1244–1253. DOI: `10.1016/j.cub.2010.06.034`
- **Horikawa, K., Ishimatsu, K., Yoshimoto, E., Kondo, S. & Takeda, H. (2006).** *"Noise-resistant and synchronized oscillation of the segmentation clock."* *Nature* 441(7094):719–723. DOI: `10.1038/nature04861`

## Governing Mathematical Laws

1. **Delayed Kuramoto Phase-Oscillator Lattice (Riedel-Kruse et al. 2007 *Science* Eq. 1; Morelli et al. 2009 *HFSP J.* Eq. 1):**
   \[
   \frac{d\theta_i(t)}{dt} = \omega_0 + \frac{\kappa}{z}\sum_{j \in N(i)} \sin\big(\theta_j(t - \tau_c) - \theta_i(t)\big) + \xi_i(t)
   \]
   where \(\theta_i(t)\) is the segmentation clock phase of presomitic mesoderm (PSM) cell \(i\), \(\omega_0 = 2\pi / T_0\) is the intrinsic single-cell frequency, \(\kappa\) is the intercellular Delta–Notch coupling strength, \(z = |N(i)|\) is the lattice coordination number, \(\tau_c\) is the intercellular DeltaC ligand synthesis/trafficking/cleavage time delay, and \(\xi_i(t)\) is Gaussian white phase noise with variance \(2 D_\phi\).
2. **Collective Synchronization Frequency Shift & Linear Stability Criterion (Morelli et al. 2009 *HFSP J.* Eqs. 2–5; Herrgen et al. 2010 *Current Biology*):**
   \[
   \Omega = \omega_0 - \kappa \sin(\Omega \tau_c), \qquad
   \kappa_{\text{eff}} = \kappa \cos(\Omega \tau_c) > 0
   \]
   The in-phase synchronized state \(\theta_i(t) = \Omega t\) shifts the collective clock frequency from \(\omega_0\) to \(\Omega\) and is linearly stable if and only if the effective coupling stiffness \(\kappa_{\text{eff}} = \kappa \cos(\Omega \tau_c) > 0\). When \(\cos(\Omega \tau_c) < 0\) (\(\Omega \tau_c \in (\pi/2, 3\pi/2)\)), in-phase synchrony destabilizes into anti-phase checkerboard disorder.
3. **Kuramoto Order Parameter \(R(t)\), Anterior Limit of Defects (ALD), and Posterior Rescue Point (PRP) (Riedel-Kruse et al. 2007 *Science* Figs. 1–4):**
   \[
   R(t) = \left|\frac{1}{N}\sum_{j=1}^{N} e^{i\theta_j(t)}\right|, \qquad
   \text{ALD}(s_{\text{in}}) = s_{\text{in}} + \Delta n_{\text{ALD}}, \qquad
   \text{PRP}(s_{\text{out}}) = s_{\text{out}} + \Delta n_{\text{PRP}}
   \]
   Blocking Notch cleavage with the \(\gamma\)-secretase inhibitor DAPT at somite stage \(s_{\text{in}}\) (\(\kappa \to 0\)) causes exponential decay of phase coherence \(R(t) = R(t_{\text{in}}) e^{-D_\phi (t - t_{\text{in}})}\), yielding \(\Delta n_{\text{ALD}} \approx 5.5\) normal somites before \(R(t)\) drops below \(R_{\text{crit}} \approx 0.50\). Washing out DAPT at stage \(s_{\text{out}}\) restores \(\kappa_{\text{eff}} > 0\) and re-synchronizes the PSM above \(R_{\text{crit}}\) at the Posterior Rescue Point \(\text{PRP}(s_{\text{out}}) = s_{\text{out}} + \Delta n_{\text{PRP}}\).

## Pass 3 Additions (2026-10-19): Kinematic Waves, Mib1 Gate, Emergent ALD/PRP

Additional sources (also in `paper_citations.bib` and the reference JSON `metadata.sources`):

- **Itoh, M. et al. (2003).** *"Mind bomb is a ubiquitin ligase that is essential for efficient activation of Notch signaling by Delta."* *Developmental Cell* 4(1):67–82. DOI: `10.1016/S1534-5807(02)00409-4`
- **Dubrulle, J., McGrew, M. J. & Pourquié, O. (2001).** *"FGF signaling controls somite boundary position and regulates segmentation clock control of spatiotemporal Hox gene activation."* *Cell* 106(2):219–232. DOI: `10.1016/S0092-8674(01)00437-8`
- **Soroldoni, D. et al. (2014).** *"A Doppler effect in embryonic pattern formation."* *Science* 345(6193):222–225. DOI: `10.1126/science.1253089`

4. **Kinematic Phase-Wave Law for an advected FGF8/Wnt frequency profile (Morelli et al. 2009; Lean `morelli_2009_kinematic_wave_phase_lag_law`):**
   \[
   \omega(\xi) = \omega_0 e^{-\xi/\lambda_F}, \qquad
   k(x) = \frac{\Omega}{v}\Big(1 - \frac{\omega(x)}{\omega_0}\Big), \qquad
   N_w = \frac{1}{S}\int_0^L \Big(1 - \frac{\omega(x)}{\omega_0}\Big)\,dx
   \]
   with \(\xi\) the distance from the tailbud in somite lengths \(S\), \(\lambda_F = L/\ln(1/f_A)\) (\(f_A = \omega(\text{front})/\omega_0 = 0.5\), PSM length \(L = 6S\)), and \(v = S/T_{\text{seg}}\) the anterior cell flow. The coupling delay is scaled with the local period, \(\tau_c(x)\,\omega(x) = \tau_c\,\omega_0\) (declared assumption). Default analytic \(N_w = 1.61\) waves per PSM, anterior wavelength \(S/(1-f_A) = 2S\).
5. **Mib1 DeltaC ubiquitination gate (Itoh et al. 2003; Lean `itoh_2003_mib1_ubiquitination_gate_monotone`):**
   \[
   \frac{du}{dt} = k_{ub} M (1-u) - k_{endo} u, \qquad u_{ss} = \frac{k_{ub} M}{k_{ub} M + k_{endo}}, \qquad
   \kappa(t) = \kappa_{\max}\, u(t)\, g(t), \qquad g = \frac{1}{1 + (C_{\text{DAPT}}/IC_{50})^n}
   \]
   with \(k_{ub} = 0.08\), \(k_{endo} = 0.02\ \text{min}^{-1}\) (\(u_{ss}^{WT} = 0.8\), \(\kappa_{\max} = 0.0875\ \text{min}^{-1}\)), DAPT dose \(C/IC_{50} = 20\), Hill \(n = 2\), washout clearance \(\tau_{\text{wash}} = 8\) min. These rate constants are the script's declared reduced choices, not measured values.
6. **Emergent defect-span law (Lean `riedel_2007_emergent_defect_span_independent_of_determination_lag`):** \(\text{ALD} = s_{\text{cross}} + \Delta n_{\text{det}}\), \(\text{PRP} = s_{\text{recross}} + \Delta n_{\text{det}}\) with \(\Delta n_{\text{det}} = 2.5\) somites, where \(s_{\text{cross}}/s_{\text{recross}}\) are the first stages at which the half-period-smoothed lateral Kuramoto \(R\) of the anterior somite-length block falls below / recovers above \(R_{\text{crit}} = 0.5\). \(D_\phi = 0.003\ \text{rad}^2/\text{min}\) and \(\sigma_\omega = 0.014\ \text{rad}/\text{min}\) are calibrated to the observed ALD data.

Corrections: the canonical `collective_frequency_Omega_rad_per_min` / `collective_period_Tsync_min` / `effective_stiffness_kappa_eff_per_min` entries of the reference JSON were changed from 0.2896 / 21.70 / 0.0566 (not roots of the collective-frequency equation) to 0.283333 / 22.176 / 0.062255; the script's `--self-test` asserts parity.

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`e3250a43b4bc0c22e8061af719f2506b5ca6f3208aa0f2cd6239a6a9b5e01718`)
- `delta_notch_phase_synchrony_reference.json` — Formal delayed Kuramoto equations, collective frequency/stability parameters, and DAPT initiation/failure/rescue perturbation table
- `zebrafish_psm_synchrony_specimen.jpg` — Dorsal vertebrate embryo scanning electron micrograph showing central neural tube, presomitic mesoderm, and bilateral segmented somite rows (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons). **Species note (Pass 3):** the Wikimedia source is titled *"Aves Neural tube and somites in chick embryo"*, i.e. this is a **chick (*Gallus gallus*) embryo**, not *Danio rerio*; it serves only as a vertebrate somite-metamere stand-in for the segmentation morphometrics. The file name is retained because it is SHA-256-pinned.
- `paper_citations.bib` — BibTeX citations for Riedel-Kruse et al. (2007), Morelli et al. (2009), Herrgen et al. (2010), Horikawa et al. (2006), Itoh et al. (2003), Dubrulle et al. (2001), and Soroldoni et al. (2014)
