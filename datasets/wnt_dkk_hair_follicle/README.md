# Sick et al. (2006) *Science* & Andl et al. (2002) *Dev. Cell* — WNT/DKK Hair Follicle Spacing Reference Dataset

Primary reference dataset, reaction-diffusion parameter invariants, and biological specimen metadata for `2002-sick-wnt-dkk-hair-follicle.py` (Slot 006 of the Infinite Morphospace & MorphoEval Campaign).

## Governing Equations — Sick et al. (2006) *Science* 314(5804):1447–1450

$$\frac{\partial w}{\partial t} = \rho_w + \frac{c_w w^2}{h (1 + \kappa w^2)} - \mu_w w + D_w \nabla^2 w$$

$$\frac{\partial h}{\partial t} = \rho_h + c_h w^2 - \mu_h h + D_h \nabla^2 h$$

where:
- $w(\mathbf{x}, t)$: short-range autocatalytic epidermal WNT activator concentration ($D_w = 0.015$; the Pass 2 README value $0.01$ disagreed with `canonical_parameters` and the script — corrected in Pass 3)
- $h(\mathbf{x}, t)$: long-range rapidly diffusing extracellular DKK1/DKK2 inhibitor concentration ($D_h = 0.45$, diffusivity ratio $d = D_h / D_w = 30.0 > d_c = (\sqrt{2}+1)^2 \approx 5.8284$)
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

## Pass 3 Addition — 3D Dermal Condensate FGF20/BMP4 Chemotaxis & Hair Peg Downgrowth

The Pass 2 audit asked for the 2D epidermal sheet to be coupled to a 3D dermal condensate model. The `dermal_condensate_model` block of `wnt_dkk_hair_follicle_reference.json` pins the reduced-order coupling implemented in Section 3b of the script:

$$\partial_t f = D_f \nabla^2 f - \mu_f f + s_f\, S(x,y)\, \delta_{z=0}, \qquad \partial_t b = D_b \nabla^2 b - \mu_b b + s_b \rho,$$

$$\partial_t \rho = D_\rho \nabla^2 \rho - \chi_f \nabla\!\cdot\!(m \nabla f) + \chi_b \nabla\!\cdot\!(m \nabla b), \quad m = \rho\,(1 - \rho/\rho_{\max}), \qquad \dot z_i = v_{\text{peg}}\, \mathrm{clip}\!\left(\tfrac{e_i - e_{\text{thr}}}{\rho_{\max} - e_{\text{thr}}}, 0, 1\right)$$

on a $42 \times 42 \times 12$ dermal slab (lateral periodic, zero-flux in depth) beneath the fixed WT placode field $S(x,y)$. Mesenchymal cells (Keller & Segel 1970; volume filling after Painter & Hillen 2002) migrate up the placode-derived FGF20 gradient (Huh et al. 2013; Glover et al. 2017) and down the dermally produced BMP4 gradient (Botchkarev et al. 1999; Mou et al. 2006); the per-placode hair peg depth $z_i$ is slaved to the dermal condensate beneath placode $i$ and staged ordinally after Paus et al. (1999) as a *model convention*.

Benchmarks (all deterministic, seed 2006): WT forms one dermal condensate beneath each of the 15 placodes (mean lateral offset 0.41 cells); *Fgf20*-null (`fgf20_scale = 0`) and chemotaxis-null (`chi_f = 0`) form 0 condensates and no pegs while the placodes persist (Huh et al. 2013); 5× BMP4 excess suppresses condensation; the epidermis-free dermal subsystem is linearly stable for every wavenumber (condensation is epidermally instructed) and the face-flux scheme conserves mesenchymal mass to round-off.

**Pass 3 counting correction:** placodes are now periodic local WNT maxima (5×5 wrap-around maximum filter). The Pass 2 counts (WT 16, *K14-Wnt* 32/22, enlarged 12) used `ndi.label` without the periodic seam and double-counted seam-straddling placodes; the corrected values are WT 15 (14 seam-aware components), *K14-Wnt* 30, enlarged 10.

**Honest reduction:** the epidermis stays planar (it is a source term, not a deforming epithelium) and downgrowth is a scalar depth per placode, not a moving epithelial interface; follicle-type (guard/awl/zigzag) dependence on condensate size is not modelled.

## Files & SHA-256 Provenance

| File | Description | SHA-256 |
|---|---|---|
| `hair_follicle_placode_specimen.jpg` | Whole-mount *in situ* hybridization of embryonic mouse (*Mus musculus*, E12.5–E13.5) vibrissa/hair follicle placode rows (`File:Gad1 transcripts in the developing vibrissae.jpg`, Maddox & Condie 2004 *BMC Dev. Biol.* 4:4, CC BY 2.5, 1200×2018 px) | `f48a7d76ae5984e50c79fe2c2f0fd446f78ac39860d82a5a87b59a676c61af19` |
| `scout_manifest.json` | Step 1 OpenAlex & Wikimedia Commons scout dossier (`morphoeval/scout_literature.py --slot 6`) | `39fe775496e89c5150f884eaded96283f321c29eb9ba02284be6e9d54921de31` |
| `paper_citations.bib` | BibTeX entries for Sick et al. (2006), Andl et al. (2002), Gierer & Meinhardt (1972), Maddox & Condie (2004) and, added in Pass 3, Keller & Segel (1970), Huh et al. (2013), Glover et al. (2017), Botchkarev et al. (1999), Mou et al. (2006), Paus et al. (1999), Painter & Hillen (2002) | `7843d2b728a0e5379c8727a3802c028723c145d08402e4d262afa0710298ab35` (Pass 2 digest: `473c1747…`) |
| `wnt_dkk_hair_follicle_reference.json` | Canonical WNT/DKK PDE parameters, dispersion invariants, primary/secondary wave and transgenic phenotype specifications, Pass 3 `dermal_condensate_model`, and SHA-256 locks | verified at runtime |
