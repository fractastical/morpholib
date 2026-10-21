# Nacre Spiral Dislocation Terraces Reference Dataset (`slot-073`)

Reference dataset and empirical specimen assets for **Cartwright & Checa (2007)** *J. R. Soc. Interface* (`10.1098/rsif.2006.0188`), **Cartwright et al. (2009)** *PNAS* (`10.1073/pnas.0900867106`), and **Burton, Cabrera & Frank (1951)** *Phil. Trans. R. Soc. A* (`10.1098/rsta.1951.0006`) screw-dislocation spiral and target terrace morphogenesis in molluscan mother-of-pearl (nacre).

## Files & Provenance

- `nacre_spiral_dislocation_terraces_reference.json` — Governing BCF equations ($v(r) = v_\infty(1 - r_c/r)$, $\lambda_0 \approx 18.9848\,r_c$, $\omega \approx 0.3310\,v_\infty/r_c$, inter-lamellar chitin-protein mortar spacing $h_0 \approx 0.50\,\mu\text{m}$), 12-species cross-taxa bivalve sheet vs gastropod/cephalopod tower nacre scaling table, and 4 chiral dislocation morphospace regimes.
- `nacre_aragonite_tablets_specimen.jpg` — Scanning electron micrograph (SEM) of fractured molluscan mother-of-pearl showing stacked polygonal aragonite tablets ($h_0 \approx 0.50\,\mu\text{m}$) and screw-dislocation spiral terraces (`File:Bruchfläche eines Perlmuttstücks.JPG`, Wikimedia Commons, Public domain, SHA-256: `e4c6590464523f2170a8fa277e51c2f2b6b0916f29d35f1dfc65f3d4ccc317b9`).
- `scout_manifest.json` — OpenAlex and Wikimedia Commons literature/specimen scout dossier (`6084a2ce133990fce4f01ef79f7791e832ed538acc6c7523a64a372101fe78f2`).
- `paper_citations.bib` — BibTeX citations for Cartwright & Checa (2007), Cartwright et al. (2009), Burton, Cabrera & Frank (1951), and Cartwright et al. (2012) (`3df311760cfbb29985fb6093131f63db46cb338741dd963864c25ed3a4794446`).

## Pass 3 (v3) additions

- `membrane_perforation_twinning_reference.json` — (1) orthorhombic aragonite lattice parameters (De Villiers 1971; $a = 4.9611$, $b = 7.9672$, $c = 5.7407$ Å), the $\{110\}$ cyclic twin angle $\tau = 2\arctan(a/b) = 63.82^\circ$, misfit $3\tau - 180^\circ = 11.46^\circ$, the default $[100]/[010]$ step-velocity ratio $a\sqrt{3}/b = 1.0785$ and the analytic twin suppression factors $f_2 = 0.074$, $f_6 = 0.947$; (2) parameters and stacking-mode references of the stacked-sheet driven Allen–Cahn chitin-membrane perforation model (columnar gastropod: central bridges, offset $\to 0$; sheet bivalve: brick-wall, nominal offset $0.5\,D_\text{tab}$). Values flagged `model_parameter` are free parameters of the reduced model. Entered offline; DOIs not re-verified (see `pass3_citations.bib`). SHA-256 pinned in the script (`EXPECTED_SHA256`).
- `pass3_citations.bib` — De Villiers (1971) Am. Mineral.; Schäffer et al. (1997) Chem. Mater. `10.1021/cm960429i`; Checa, Cartwright & Willinger (2011) J. Struct. Biol. `10.1016/j.jsb.2011.09.011`.

### Known caveats (carried for audit)

- The SEM specimen (`Bruchfläche eines Perlmuttstücks.JPG`) carries no scale bar: `calibration_um_per_px = 0.019867` was chosen so that the segmented inter-lamellar spacing equals $h_0 = 0.50\,\mu$m, and the `reference_*` SEM values in `nacre_spiral_dislocation_terraces_reference.json` were copied from the script's own segmentation. The $h_0$ measurement is therefore a consistency check, not an independent test; only the dimensionless CV and tablet aspect ratio are calibration-independent.
- The 12-species `cross_taxa_nacre_terrace_scaling` table has `lambda_obs_um` and `omega_obs_rad_hr` entries that lie within ~1 % of the BCF law for every species (Pearson $r = 0.9999$); these are not traceable to a measured table in the cited papers and should be treated as illustrative until replaced by digitised literature values.
