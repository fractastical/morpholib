/-
  Loop 3135 — Bos taurus, Holocene extant (0 Ma), detail-scale Turing wavelength.

  Scale: 3.137 µm (3.137e-06 m, log10 = -5.503), tier 3/8 `3_subcellular_um`.
  Law: Gierer–Meinhardt activator–inhibitor wavenumber selection.
  Chronoseries 1/1 has no predecessor age; the theorems below record the
  trace sign, the inhibitor-range cross term, and the scale-neighbour
  comparison (500 nm melanosome < λ_c < 20 µm melanocyte soma).
-/
namespace MathOfBiology.Loop_3135

/-- Activator and inhibitor removal rates in 10^-2 per dimensionless time. -/
def muU : Int := 85
def muV : Int := 140

/-- tr(J) = μ_u − μ_v at the homogeneous steady state with basal production off. -/
def traceJ (a b : Int) : Int := a - b

theorem holocene_detail_trace_negative :
    traceJ muU muV < 0 := by
  decide

/-- Diffusion ratio D_v:D_u = 145:10. Numerator of D_v μ_u − D_u μ_v. -/
def crossNumerator : Int := 145 * muU - 10 * muV

theorem inhibitor_range_cross_positive :
    crossNumerator > 0 := by
  decide

/-- Assigned detail wavelength in nanometres (3.137 µm). -/
def lambdaNm : Int := 3137

theorem wavelength_inside_tier3 :
    1000 < lambdaNm ∧ lambdaNm < 10000 := by
  decide

/-- Tier-2 predecessor: a 500 nm stage-IV melanosome is shorter than λ_c. -/
theorem predecessor_melanosome_below_wavelength :
    500 < lambdaNm := by
  decide

/-- Tier-4 successor: a 20 µm melanocyte soma holds at least six wavelengths. -/
theorem successor_cell_admits_multiple_wavelengths :
    20000 / lambdaNm >= 6 := by
  decide

/-- Bos chronoseries 1/1 is Holocene extant, age 0 Ma. -/
def ageMa : Int := 0

theorem holocene_extant_chronoseries_age :
    ageMa = 0 := by
  decide

end MathOfBiology.Loop_3135
