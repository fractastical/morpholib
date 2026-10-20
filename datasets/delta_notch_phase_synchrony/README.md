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

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`e3250a43b4bc0c22e8061af719f2506b5ca6f3208aa0f2cd6239a6a9b5e01718`)
- `delta_notch_phase_synchrony_reference.json` — Formal delayed Kuramoto equations, collective frequency/stability parameters, and DAPT initiation/failure/rescue perturbation table
- `zebrafish_psm_synchrony_specimen.jpg` — Dorsal vertebrate embryo scanning electron micrograph showing central neural tube, presomitic mesoderm, and bilateral segmented somite rows (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Riedel-Kruse et al. (2007), Morelli et al. (2009), Herrgen et al. (2010), and Horikawa et al. (2006)
