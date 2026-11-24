/-
  Loop 3180 — Drosophila cyrtoloma, Turing wavenumber at the 4.3 µm detail tier.
  Scale: 4.315e-6 m (4.315 µm, log10 = -5.365, tier 3_subcellular_um).
  Age: Holocene, 0 Ma. Chronoseries 9/41.
  Predecessor wavelength hypothesis: D. buzzatii, 7840 nm.
  Focal carrier: D. cyrtoloma, 4315 nm.
  Successor wavelength hypothesis: D. deflecta, 6050 nm.
  Integer Jacobian used by the dispersion theorems (reaction scale γ = 1, ratio Dv/Du = 18):
    f_u = 40, f_v = -50, g_u = 60, g_v = -50.
  These signs match a Turing-capable activator-inhibitor Jacobian
  (trace < 0, det > 0, differential-diffusion threshold only when Dv > Du).
  They are a scaled witness for the inequality, not the floating-point Schnakenberg numbers.
-/
namespace MathOfBiology.Loop_3180

/-- Detail-tier carrier wavelengths in nanometres: 0 cyrtoloma, 1 buzzatii, 2 deflecta. -/
def lambdaNm (species : Nat) : Nat :=
  match species with
  | 0 => 4315
  | 1 => 7840
  | 2 => 6050
  | _ => 0

/-- The 4.315 µm cyrtoloma carrier is shorter than the buzzatii predecessor hypothesis. -/
theorem cyrtoloma_shorter_than_buzzatii :
    lambdaNm 0 < lambdaNm 1 := by
  decide

/-- Deflecta sits strictly between the focal carrier and the coarser buzzatii carrier. -/
theorem deflecta_between_cyrtoloma_and_buzzatii :
    lambdaNm 0 < lambdaNm 2 ∧ lambdaNm 2 < lambdaNm 1 := by
  decide

/-- Homogeneous Jacobian trace f_u + g_v is negative, so diffusion is required for instability. -/
theorem homogeneous_trace_negative :
    (40 + (-50) : Int) < 0 := by
  decide

/-- det J = f_u g_v - f_v g_u = 1000 > 0. -/
theorem jacobian_determinant_positive :
    ((40 * (-50) - (-50) * 60 : Int)) = 1000 := by
  decide

/-- Equal diffusion (Dv = Du = 1) makes h = Dv f_u + Du g_v negative, so the Turing threshold fails. -/
theorem equal_diffusion_not_turing :
    ((1 * 40 + 1 * (-50) : Int)) < 0 := by
  decide

/-- At Dv/Du = 18 the dispersion threshold h² > 4 Du Dv det holds. -/
theorem cyrtoloma_dispersion_threshold :
    let h : Int := 18 * 40 + 1 * (-50)
    let detJ : Int := 40 * (-50) - (-50) * 60
    h * h > 4 * 1 * 18 * detJ := by
  decide

/-- The 207.6 µm primordium spot (slot 3061) holds 48 cycles of this 4315 nm carrier. -/
theorem primordium_spot_holds_forty_eight_carriers :
    207600 / 4315 = 48 := by
  decide

end MathOfBiology.Loop_3180
