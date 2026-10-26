# Viscoelastic Apical Purse-String Constriction & Epithelial Invagination Dataset (`datasets/apical_constriction_invagination/`)

Reference parameters, formal viscoelastic mechanical equations, experimental genetic perturbation tables, and biological specimen photography for **Slot 092 (`slot-092-apical-constriction-invagination`)** of the Project Delphi & MorphoEval Campaign, reproducing:

- **Odell, G. M., Oster, G., Alberch, P. & Burnside, B. (1981).** *"The mechanical basis of morphogenesis. I. Epithelial folding and morphogenesis."* *Developmental Biology* 85(2):446–462. DOI: `10.1016/0012-1606(81)90434-6`
- **Martin, A. C., Kaschube, M. & Wieschaus, E. F. (2009).** *"Pulsed contractions of an actin–myosin network drive apical constriction."* *Nature* 457(7228):495–499. DOI: `10.1038/nature07522`
- **Conte, V., Muñoz, J. J. & Miodownik, M. (2008).** *"A 3D finite element model of ventral furrow invagination in the Drosophila melanogaster embryo."* *Journal of the Mechanical Behavior of Biomedical Materials* 1(2):188–198. DOI: `10.1016/j.jmbbm.2007.10.002`
- **Polyakov, O., He, B., Swan, M., Shaevitz, J. W., Kaschube, M. & Wieschaus, E. (2014).** *"Passive mechanical forces control cell-shape change during Drosophila ventral furrow formation."* *Biophysical Journal* 107(4):998–1010. DOI: `10.1016/j.bpj.2014.07.013`

## Governing Mathematical Laws

1. **Odell et al. (1981) Stretch-Activated Bistable Apical Purse-String Rest-Length Law & Martin et al. (2009) Pulsed Ratchet:**
   \[
   \tau_0 \frac{dL_{0,i}}{dt} = -\left(L_{0,i} - L_{\text{target}}(L_{\text{ap},i})\right), \qquad
   L_{\text{target}}(L_{\text{ap},i}) =
   \begin{cases}
   L_1 & \text{if } L_{\text{ap},i} \le L_{\text{thresh}}, \\
   L_2 & \text{if } L_{\text{ap},i} > L_{\text{thresh}} \quad (0 < L_2 < L_1),
   \end{cases}
   \]
   with active viscoelastic apical tension \(T_{\text{ap},i} = k_{\text{ap}}(L_{\text{ap},i} - L_{0,i}) + \mu_{\text{ap}}\dot{L}_{\text{ap},i}\).
2. **Incompressible Cytoplasmic Volume Conservation & Wedge Spontaneous Curvature Law:**
   \[
   A_i = \frac{1}{2}\left(L_{\text{ap},i} + L_{\text{bas},i}\right) h_i = A_0, \qquad
   \kappa_i = \frac{2\left(L_{\text{bas},i} - L_{\text{ap},i}\right)}{h_i\left(L_{\text{bas},i} + L_{\text{ap},i}\right)} = \frac{L_{\text{bas},i} - L_{\text{ap},i}}{A_0} > 0 \quad (L_{\text{ap},i} < L_{\text{bas},i}).
   \]
3. **Martin et al. (2009) & Odell et al. (1981) Five-Regime Perturbation Hierarchy:**
   Wild-type Twist+Snail ratcheted stretch propagation (\(D_{\text{inv}} = 38.0\,\mu\text{m}\), \(L_{\text{ap}} = 2.1\,\mu\text{m}\)) > Non-propagating local constriction (\(D_{\text{inv}} = 21.0\,\mu\text{m}\)) > *concertina* uncoordinated constriction (\(D_{\text{inv}} = 15.5\,\mu\text{m}\)) > *snail* unratcheted pulses (\(D_{\text{inv}} = 4.5\,\mu\text{m}\)) > *twist; snail* double mutant / Y-27632 ROCK-inhibited negative control (\(D_{\text{inv}} = 0.0\,\mu\text{m}\)).

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier
- `apical_constriction_invagination_reference.json` — Formal viscoelastic purse-string parameters and 5-regime perturbation dataset
- `drosophila_ventral_furrow_specimen.jpg` — Fluorescent confocal micrographs of *Drosophila melanogaster* ventral furrow invagination stained for DAPI (nuclei, blue), Neurotactin (membranes, green), and Myosin II heavy chain (apical cortex, red) across 4 transverse invagination stages (`f426366f7ede436da44782b28ebac9bd013cace1f8fe44b6ed382b77ad86c419`, CC BY 4.0, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Odell et al. (1981), Martin et al. (2009), Conte et al. (2008), and Polyakov et al. (2014)
