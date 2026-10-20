# Sick et al. (2006) *Science* & Andl et al. (2002) *Dev. Cell* — WNT/DKK Hair Follicle Spacing Reference Dataset

Primary reference dataset, reaction-diffusion parameter invariants, and biological specimen metadata for `2002-sick-wnt-dkk-hair-follicle.py` (Slot 006 of the Infinite Morphospace & MorphoEval Campaign).

## Governing Equations — Sick et al. (2006) *Science* 314(5804):1447–1450

$$\frac{\partial w}{\partial t} = \rho_w + \frac{c_w w^2}{h (1 + \kappa w^2)} - \mu_w w + D_w \nabla^2 w$$

$$\frac{\partial h}{\partial t} = \rho_h + c_h w^2 - \mu_h h + D_h \nabla^2 h$$

where:
- $w(\mathbf{x}, t)$: short-range autocatalytic epidermal WNT activator concentration ($D_w = 0.01$)
- $h(\mathbf{x}, t)$: long-range rapidly diffusing extracellular DKK1/DKK2 inhibitor concentration ($D_h = 0.40$, diffusivity ratio $d = D_h / D_w = 40.0 > d_c = (\sqrt{2}+1)^2 \approx 5.8284$)
- $\rho_w$: basal epidermal WNT expression rate (elevated under *K14-Wnt* transgenic overexpression)
- $\rho_h$: basal DKK inhibitor expression rate (elevated under *Foxn1-Dkk1* / *K14-Dkk1* transgenic expression)
- $\kappa$: saturating nonlinear autocatalysis parameter governing placode enlargement and annular ring splitting
- $\mu_w, \mu_h$: WNT and DKK turnover rates ($\mu_h > \mu_w$ required for homogeneous stability)

## Experimental Phenotype Benchmarks (Sick et al. 2006 Figs. 2–4)

1. **Primary Wave (E14.5) $\to$ Inter-Follicular Secondary Wave Insertion (E16.5):**
   During embryonic skin expansion ($L_0 \to 2 L_0$), primary hair follicle placodes establish long-range DKK inhibitory zones. As inter-follicular distances double, DKK concentration at the inter-placode midpoints drops below threshold, triggering autonomous insertion of secondary wave hair follicle placodes midway between primary follicles.
2. **Transgenic *K14-Wnt* Overexpression (Elevated $\rho_w$):**
   Ectopic basal WNT expression shifts the reaction-diffusion pattern toward high-density clustered placode pairs and enlarged coalescing follicular patches (Sick et al. 2006 Fig. 2B–D).
3. **Moderately Elevated DKK Activity (Enlarged/Ringing Follicles):**
   Partial saturation ($\kappa > 0$) combined with altered inhibitor balance produces enlarged, hollow annular ("ringing") placode morphologies (Sick et al. 2006 Fig. 3–4).
4. **Severe *Dkk1* Overexpression (Andl et al. 2002 *Dev. Cell* 2:643–653):**
   High ectopic $\rho_h$ completely suppresses WNT autocatalysis ($N_{\text{placodes}} = 0$, alopecia/hairless phenotype).

## Files & SHA-256 Provenance

| File | Description | SHA-256 |
|---|---|---|
| `hair_follicle_placode_specimen.jpg` | Whole-mount *in situ* hybridization of embryonic mouse (*Mus musculus*, E12.5–E13.5) vibrissa/hair follicle placode rows (`File:Gad1 transcripts in the developing vibrissae.jpg`, Maddox & Condie 2004 *BMC Dev. Biol.* 4:4, CC BY 2.5, 1200×2018 px) | `f48a7d76ae5984e50c79fe2c2f0fd446f78ac39860d82a5a87b59a676c61af19` |
| `scout_manifest.json` | Step 1 OpenAlex & Wikimedia Commons scout dossier (`morphoeval/scout_literature.py --slot 6`) | `39fe775496e89c5150f884eaded96283f321c29eb9ba02284be6e9d54921de31` |
| `paper_citations.bib` | BibTeX entries for Sick et al. (2006), Andl et al. (2002), Gierer & Meinhardt (1972), and Maddox & Condie (2004) | `473c1747b6bcb451d238feb8deb245fcf112302220c8b91a9b91073b810ad7f2` |
| `wnt_dkk_hair_follicle_reference.json` | Canonical WNT/DKK PDE parameters, dispersion invariants, primary/secondary wave and transgenic phenotype specifications, and SHA-256 locks | verified at runtime |
