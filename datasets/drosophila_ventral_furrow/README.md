# Pulsed Apical Constriction & Viscous-Shell Biomechanics of *Drosophila* Ventral Furrow Invagination (`datasets/drosophila_ventral_furrow/`)

Reference parameters, formal 2D incompressible viscous-vertex cross-sectional equations, Conte et al. (2012) biomechanical synergy regimes, Martin et al. (2009) ratcheted actomyosin pulse parameters, and biological specimen photography for **Slot 046 (`slot-046-drosophila-ventral-furrow`, `he-drosophila-ventral-furrow`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing:

- **Conte, V., Muñoz, J. J., Baum, B. & Miodownik, M. (2012).** *"Biomechanical analysis of ventral furrow formation in Drosophila melanogaster."* *PLoS ONE* 7(11):e49973. DOI: `10.1371/journal.pone.0049973`
- **Martin, A. C., Kaschube, M. & Wieschaus, E. F. (2009).** *"Pulsed contractions of an actin–myosin network drive apical constriction."* *Nature* 457(7228):495–499. DOI: `10.1038/nature07522`
- **He, B., Doubrovinski, K., Polyakov, O. & Wieschaus, E. (2014).** *"Apical constriction drives tissue-scale hydrodynamic flow to mediate cell elongation."* *Nature* 508(7496):392–396. DOI: `10.1038/nature13070`
- **Polyakov, O., He, B., Swan, M., Shaevitz, J. W., Kaschube, M. & Wieschaus, E. (2014).** *"Passive mechanical forces control cell-shape change during Drosophila ventral furrow formation."* *Biophysical Journal* 107(4):998–1010. DOI: `10.1016/j.bpj.2014.07.013`

## Governing Mathematical Laws

1. **Incompressible Trapezoidal Wedge Geometry (`incompressible_wedge_cell_lengthening_and_basal_expansion`):**
   \[
   A_i = \frac{1}{2}\left(w_{\text{ap},i} + w_{\text{ba},i}\right) h_i = A_0 = 262.4\,\mu\text{m}^2.
   \]
   Under apical constriction (\(w_{\text{ap},i} < w_0\)), cytoplasmic incompressibility forces apicobasal cell lengthening (\(h_i > h_0\) when basal width is held fixed by a stiff basal cortex) or basal expansion (\(w_{\text{ba},i} > w_0\) when basal relaxation and apicobasal shortening occur).
2. **Martin et al. (2009) Twist-Stabilized Ratcheted Myosin-II Pulse Dynamics (`twist_ratcheted_pulse_stepwise_constriction_vs_recoil`):**
   \[
   \frac{dL_{\text{ap}}}{dt} = -\alpha_{\text{pulse}} M(t) + \beta_{\text{relax}}\left(1 - r_{\text{twist}}\right)\left(L_0 - L_{\text{ap}}\right).
   \]
   With full Twist ratchet stabilization (\(r_{\text{twist}} = 1\)), each actomyosin pulse produces a strictly negative net step-wise apical width change (\(\Delta L_{\text{net}} < 0\)), whereas unratcheted recoil (\(r_{\text{twist}} = 0\)) relaxes back toward \(L_0\) between pulses.
3. **Conte et al. (2012) Synergistic Biomechanical Invagination Law (`conte_2012_synergistic_ventral_furrow_invagination`):**
   Apical constriction alone generates only a shallow groove (\(D_{\text{furrow}} = 16.5\,\mu\text{m}\)) with over-elongated mesoderm cells (\(h = 48.1\,\mu\text{m}\)); adding basal relaxation increases invagination to \(29.5\,\mu\text{m}\); full synergy of apical constriction + basal relaxation + lateral shortening + ectodermal pushing achieves complete ventral furrow invagination (\(D_{\text{furrow}} = 41.0\,\mu\text{m}\)).

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`0c6354ab1d934ba37fd34e35649c381535c9a595e26224e52959895c156e8c1f`)
- `drosophila_ventral_furrow_reference.json` — Formal viscous-vertex parameters, ratcheted pulse ODE, and 6-regime Conte/Martin/He dataset
- `drosophila_ventral_furrow_specimen.jpg` — Fluorescent confocal micrographs of *Drosophila melanogaster* ventral furrow invagination stained for DAPI (nuclei, blue), Neurotactin (membranes, green), and Myosin II heavy chain (apical cortex, red) across 4 transverse invagination stages (`f426366f7ede436da44782b28ebac9bd013cace1f8fe44b6ed382b77ad86c419`, CC BY 4.0, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Conte et al. (2012), Martin et al. (2009), He et al. (2014), and Polyakov et al. (2014)
