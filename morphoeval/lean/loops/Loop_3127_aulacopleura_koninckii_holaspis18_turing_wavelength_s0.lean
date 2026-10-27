/-
  Loop 3127 — Aulacopleura koninckii holaspis18
  Scale: 15 µm detail variant (1.549e-5 m, tier 4_cellular_10um)
  Age: Silurian Wenlock, 428.3 Ma
  Law: Turing wavenumber k_c for a two-species reaction–diffusion system.
  Integer encodings match models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py
  (rates in 10^-3 s^-1; pitches in µm; λ = 15 µm for the floor-division mode count).
-/
namespace MathOfBiology.Loop_3127

/-- Trace of the holaspis18 cuticle Jacobian, milli-s^-1: f_u + g_v = 720 + (−1280). -/
def traceMilli : Int := 720 + (-1280)

theorem trace_negative : traceMilli < 0 := by
  decide

/-- det(J) = f_u g_v − f_v g_u with f_v = −860 and g_u = 1100 (milli-s^-1). -/
def detMilli : Int := 720 * (-1280) - (-860) * 1100

theorem homogeneous_state_stable : detMilli > 0 ∧ traceMilli < 0 := by
  decide

/-- Inhibitor must out-diffuse the activator: R · f_u + g_v > 0 at R = 40. -/
def turingNumeratorMilli : Int := 40 * 720 + (-1280)

theorem inhibitor_diffusion_opens_a_band : turingNumeratorMilli > 0 := by
  decide

/-- Floor mode count N = pitch_µm / 15 µm. Pitches: meraspis 79, holaspis18 175, holaspis22 224. -/
def modeCount (pitchUm : Nat) : Nat := pitchUm / 15

theorem holaspis18_mode_between_chronospecies :
    modeCount 79 < modeCount 175 ∧ modeCount 175 < modeCount 224 := by
  decide

/-- Same λ, longer pitch, more repeats: the 428.3 Ma holaspis18 count is not the meraspid count. -/
theorem holaspis18_not_meraspid_mode : modeCount 175 ≠ modeCount 79 := by
  decide

end MathOfBiology.Loop_3127
