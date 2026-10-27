/-
  Loop 3131 — Baculites compressus at the 94.16 µm detail scale.
  Tier 4_cellular_10um. Late Cretaceous Pierre Shale, 73.5 Ma.
  Chemistry is shared across the cephalopod suture series; the aperture
  factor χ is what changes relative to Placenticeras meeki (predecessor)
  and Nipponites mirabilis (morphological successor).
  Wavelengths are stored in hundredths of a micrometre so that
  9416 stands for 94.16 µm = 9.416 × 10⁻⁵ m.
-/
namespace MathOfBiology.Loop_3131

/-- Critical Turing wavelength of Baculites compressus, χ = 1, in 0.01 µm. -/
def lambdaBaculites : Nat := 9416

/-- Placenticeras meeki readout, χ = 3/2, in 0.01 µm. -/
def lambdaPlacenticeras : Nat := 14124

/-- Nipponites mirabilis readout, χ = 6215/9416, in 0.01 µm. -/
def lambdaNipponites : Nat := 6215

/-- Organ-scale sibling wavelength (slot 1239) in whole micrometres: 4.3 mm. -/
def lambdaOrganUm : Nat := 4300

/-- Integer diffusion ratio used above threshold (r = 9; the script uses 9.2). -/
def diffusionRatio : Nat := 9

/-- Integer floor of the Schnakenberg critical ratio r_c ≈ 8.568. -/
def criticalRatioFloor : Nat := 8

theorem detail_scale_far_below_organ_rib :
    lambdaOrganUm * 100 > lambdaBaculites * 4 := by
  decide

theorem chronoseries_wavelength_order :
    lambdaNipponites < lambdaBaculites ∧ lambdaBaculites < lambdaPlacenticeras := by
  decide

theorem placenticeras_aperture_is_three_halves :
    2 * lambdaPlacenticeras = 3 * lambdaBaculites := by
  decide

theorem inhibitor_range_above_turing_threshold :
    diffusionRatio > criticalRatioFloor := by
  decide

/-- Eight critical wavelengths span the 753.28 µm detail window (8 × 94.16). -/
theorem eight_wavelength_detail_window :
    8 * lambdaBaculites = 75328 := by
  decide

/-- Marginal wavenumber k_c² has a positive numerator–denominator product
    whenever both factors are positive (D_v f_u + D_u g_v > 0 and D_u, D_v > 0). -/
def kcSqNum (dv fu du gv : Int) : Int := dv * fu + du * gv

def kcSqDen (du dv : Int) : Int := 2 * du * dv

theorem turing_wavenumber_positive
    (dv fu du gv : Int)
    (hnum : 0 < dv * fu + du * gv)
    (hden : 0 < 2 * du * dv) :
    0 < kcSqNum dv fu du gv * kcSqDen du dv := by
  dsimp [kcSqNum, kcSqDen]
  exact Int.mul_pos hnum hden

end MathOfBiology.Loop_3131
