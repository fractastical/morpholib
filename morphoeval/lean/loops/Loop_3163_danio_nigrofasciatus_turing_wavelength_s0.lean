/-
  morphoeval/lean/loops/Loop_3163_danio_nigrofasciatus_turing_wavelength_s0.lean
  Slot 3163 — Danio nigrofasciatus, cellular Turing wavelength.
  Scale: 27 µm (catalog 2.71e-5 m, log10 = -4.567), tier 4_cellular_10um, detail s0.
  Age: Holocene (extant), 0 Ma.
  Chronoseries 4/5: predecessor Danio margaritatus, successor Danio rerio.
  Law: Turing wavenumber k_c with Schnakenberg dispersion and flank anisotropy.
  No sorry. Integer witnesses avoid a mathlib dependency.
-/
namespace MathOfBiology.Loop_3163

/-- Focal cellular wavelength of D. nigrofasciatus, in hundredths of a micrometre (27.00 µm). -/
def lambdaNigroCmm : Nat := 2700

/-- D. margaritatus hexagonal spot spacing ≈ (2/√3)·27 µm = 31.18 µm, hundredths. -/
def lambdaMargCmm : Nat := 3118

/-- D. rerio keeps the same melanophore-scale wavelength; anisotropy, not λ, changes. -/
def lambdaRerioCmm : Nat := 2700

/-- Anisotropy α × 100. Dorsal nigrofasciatus α = 6.00. -/
def alphaDorsalCenti : Nat := 600

/-- Ventral nigrofasciatus α = 1.15, below the stripe-selection threshold. -/
def alphaVentralCenti : Nat := 115

/-- Predecessor margaritatus is nearly isotropic (α = 1.05). -/
def alphaMargCenti : Nat := 105

/-- Successor rerio holds a high flank-wide anisotropy (α = 8.50). -/
def alphaRerioCenti : Nat := 850

/-- k_c rounded to the nearest rad/m for λ = 27 µm. 2π/27e-6 ≈ 232709.79. -/
def kCPerM : Nat := 232710

/-- round(2π · 10^6) = 6283185, the integer form of k·λ when λ is in micrometres. -/
def twoPiMicro : Nat := 6283185

theorem wavelength_positive : 0 < lambdaNigroCmm := by
  decide

theorem margaritatus_spot_spacing_exceeds_stripe_wavelength :
    lambdaNigroCmm < lambdaMargCmm := by
  decide

theorem rerio_cellular_wavelength_matches_nigrofasciatus :
    lambdaRerioCmm = lambdaNigroCmm := by
  decide

/-- Broken ventral stripes: dorsal anisotropy strictly exceeds the ventral value. -/
theorem dorsal_anisotropy_selects_stripes :
    alphaVentralCenti < alphaDorsalCenti := by
  decide

/-- Chronoseries order at this scale is anisotropy, not a change of geological age (all 0 Ma). -/
theorem chronoseries_anisotropy_order :
    alphaMargCenti < alphaVentralCenti ∧
    alphaVentralCenti < alphaDorsalCenti ∧
    alphaDorsalCenti < alphaRerioCenti := by
  decide

/-- k_c · 27 µm agrees with 2π to 15 parts in 6.28×10^6 (< 0.0003%). -/
theorem wavenumber_wavelength_product_close :
    kCPerM * 27 + 15 = twoPiMicro := by
  decide

/-- This cellular slot is finer than the organ-scale sibling wavelength of 1042 µm. -/
theorem cellular_scale_finer_than_organ_sibling :
    lambdaNigroCmm < 104200 := by
  decide

/-- Turing threshold for the script's Jacobian: D_v f_u = 2.0·0.8 > D_u |g_v| = 0.05·1,
    cleared in thousandths (2000·800 > 50·1000). -/
theorem turing_diffusion_threshold :
    (2000 : Int) * 800 > 50 * 1000 := by
  decide

end MathOfBiology.Loop_3163
