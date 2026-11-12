/-
  Loop 3103 — Acetabularia acetabulum, Turing wavelength at the 25.28 µm detail scale.
  Holocene extant (0 Ma). Chronoseries length 1: no predecessor or successor chronospecies.

  Integer encoding (so the proofs are `decide`, with no `sorry` and no mathlib):
    Jacobian entries × 100 at the Brusselator steady state (A, B) = (2, 2.9):
      f_u = 190, g_v = -400, det = 400.
    Hoop diffusivities × 100 in µm²/s: D_u = 1020, D_v = 12240.
  These are the nominal parameters of
  models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py.
  They are not the dimensionless (α, β) template of sibling slot 0192.
-/

namespace MathOfBiology.Loop_3103

/-- Trace × 100. f_u + g_v = 1.9 − 4 = −2.1, so the homogeneous state is stable. -/
def trace_c : Int := 190 + (-400)

theorem homogeneous_state_stable : trace_c < 0 := by
  decide

/-- D_v f_u + D_u g_v in the ×100 encoding. Positive: inhibitor diffusion can destabilise. -/
def cross_term_nominal : Int := 12240 * 190 + 1020 * (-400)

theorem activator_inhibitor_cross_term_positive : cross_term_nominal > 0 := by
  decide

/-- (D_v f_u + D_u g_v)² versus 4 D_u D_v det, same encoding. The nominal point is inside the Turing band. -/
def disc_lhs : Int := cross_term_nominal * cross_term_nominal

def disc_rhs : Int := 4 * 1020 * 12240 * 400

theorem turing_discriminant_nominal : disc_lhs > disc_rhs := by
  decide

/-- Equal diffusivities (the ablation). The cross term changes sign, so there is no Turing band. -/
def cross_term_equal_diffusion : Int := 1020 * 190 + 1020 * (-400)

theorem ablation_equal_diffusion_no_turing_band : cross_term_equal_diffusion < 0 := by
  decide

/-- Slot length: 25.28 µm = 2528 hundredths of a micrometre = 2.528×10⁻⁵ m. -/
def lambda_c_hundredths_um : Int := 2528

theorem detail_scale_below_100_um :
    lambda_c_hundredths_um = 2528 ∧ lambda_c_hundredths_um < 10000 := by
  decide

/-- How many wavelengths fit on a patch. The 202.24 µm wall patch holds eight. -/
def mode_count (domain_hund lambda_hund : Int) : Int := domain_hund / lambda_hund

theorem eight_ridges_on_detail_patch : mode_count (8 * 2528) 2528 = 8 := by
  decide

end MathOfBiology.Loop_3103
