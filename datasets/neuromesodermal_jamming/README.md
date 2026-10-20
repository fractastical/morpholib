# Active Fluid-to-Solid Jamming Transition in Zebrafish Body Axis Elongation (`datasets/neuromesodermal_jamming/`)

Reference parameters, formal active-foam rigidity-percolation and Bingham viscoplastic equations, experimental perturbation benchmarks, and biological specimen lightsheet microscopy for **Slot 030 (`slot-030-neuromesodermal-jamming`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing:

- **Mongera, A., Rowghanian, P., Gustafson, H. J., Shelton, E., Kealhofer, D. A., Carn, E. K., Serwane, F., Lucio, A. A., Giammona, J. & Campàs, O. (2018).** *"A fluid-to-solid jamming transition underlies vertebrate body axis elongation."* *Nature* 561(7723):401–405. DOI: `10.1038/s41586-018-0479-2`
- **Kim, S., Pochitaloff, M., Stooke-Vaughan, G. A. & Campàs, O. (2021).** *"Embryonic tissues as active foams."* *Nature Physics* 17(7):859–866. DOI: `10.1038/s41567-021-01215-1`
- **Lawton, A. K., Nandi, A., Stulberg, M. J., Dray, N., Sneddon, M. W., Emonet, T., Blawzdziewicz, J. & Holley, S. A. (2013).** *"Regulated tissue fluidity steers zebrafish body elongation."* *Development* 140(3):573–582. DOI: `10.1242/dev.090381`

## Governing Mathematical Laws

1. **Extracellular Space Compaction & Rigidity-Percolation Yield Stress Law:**
   \[
   \phi_{\text{pack}}(x) = 1 - \phi_{\text{ECS}}(x), \qquad \sigma_y(x) = \sigma_0 \max\!\left(0,\, \phi_{\text{pack}}(x) - \phi_c\right)^\beta = \sigma_0 \max\!\left(0,\, \phi_{\text{ECS,crit}} - \phi_{\text{ECS}}(x)\right)^\beta
   \]
   Along the posterior-to-anterior axis \(x \in [0, L]\) of the *Danio rerio* tailbud, extracellular space fraction \(\phi_{\text{ECS}}(x)\) decreases from \(\sim 14.5\%\) in the posterior Mesodermal Progenitor Zone (MPZ) to \(\sim 4.9\%\) in the Anterior Presomitic Mesoderm (A-PSM), crossing the critical rigidity-percolation threshold \(\phi_c \approx 0.860\) (\(\phi_{\text{ECS,crit}} \approx 0.140\), Maxwell isostatic coordination \(z_c = 4\) in 2D) and driving a \(\sim 10\)-fold surge in supracellular yield stress \(\sigma_y(x)\) from \(\sim 11\ \text{Pa}\) in the MPZ to \(\sim 118\ \text{Pa}\) in the A-PSM.

2. **Active Foam T1 Cage-Breaking Rate & Fluid-to-Solid Jamming Criterion:**
   \[
   k_{T1}(x) = k_0 \exp\!\left(-c_B \left(\frac{\sigma_y(x)}{\sigma_A(x)}\right)^{\gamma_k}\right)
   \]
   In the posterior MPZ, active Myosin-II cell-scale tension fluctuations exceed the local yield stress (\(\sigma_A(x) \approx 50\ \text{Pa} > \sigma_y(x) \approx 11\ \text{Pa}\)), melting the cellular foam via frequent T1 neighbor exchanges (\(k_{T1} \approx 0.08\ \text{min}^{-1}\)). Anterior to the jamming transition point (\(x^* \approx 90\ \mu\text{m}\) where \(\sigma_y(x^*) = \sigma_A(x^*)\)), \(\sigma_y(x) \gg \sigma_A(x)\) locks cells into a solid-like jammed state.

3. **Bingham Viscoplastic Backstop & *cdh2* Knockdown Bilateral Asymmetry Law:**
   \[
   \dot{\gamma}(x) = \frac{\max(0,\, \sigma_{\text{eff}}(x) - \sigma_y(x))}{\eta_p}, \qquad v_{\text{posterior}} = J_{\text{MPZ}} \frac{\sigma_{y,\text{APSM}}}{\sigma_{y,\text{APSM}} + \sigma_{y,\text{MPZ}} + \sigma_{\text{ref}}}
   \]
   The high yield stress of the solid A-PSM acts as an internal mechanical backstop that rectifies mass addition in the fluid MPZ into unidirectional posterior elongation (\(v_{\text{elong}} \approx 1.52\ \mu\text{m/min}\)) while preserving left-right bilateral symmetry. Knockdown of N-cadherin (*cdh2*) expands \(\phi_{\text{ECS}}\) above \(\phi_{\text{ECS,crit}}\), unjamming the PSM, collapsing the mechanical backstop, slowing elongation (\(0.78\ \mu\text{m/min}\)), and inducing left-right body-axis asymmetry.

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`5770ebf01ff6d270833f332b1aaa6b74d65624310c14e308356239ab3963d813`)
- `neuromesodermal_jamming_reference.json` — Formal active-foam jamming equations, Mongera et al. (2018) AP spatial profiles, and 5-regime perturbation dataset
- `zebrafish_tailbud_psm_specimen.jpg` — Lightsheet fluorescence micrograph of the *Danio rerio* posterior tailbud neuromesodermal progenitor zone (MPZ), bilateral presomitic mesoderm (PSM), and central notochord from Open Microscopy Image Data Resource study `idr0051` / Dray & Lawton et al. (2013) (`e987d8b6218b96461c1213f9a11d20fb5d6585c6074608bf060c16edf725d500`, CC BY 4.0)
- `paper_citations.bib` — BibTeX citations for Mongera et al. (2018), Kim et al. (2021), Lawton et al. (2013), Serwane et al. (2017), and Dray et al. (2013) (`d68afd1715fd7fcf85a531f8b6a792c73b5067343b14e40635265341de5b7ce1`)
