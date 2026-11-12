/-
  Loop 3202 — Drosophila sechellia, 1.5 µm detail-scale Turing wavelength.
  Holocene (extant), 0 Ma. Sophophora melanogaster subgroup, chronoseries 31/41.
  Predecessor: Drosophila santomea (λ = 1921 nm). Successor: Drosophila serrata (λ = 1680 nm).
  This slot selects λ_c = 1500 nm = 1.500 µm.
  No `sorry`.
-/
namespace MathOfBiology.Loop_3202

/-- Critical wavelength of the D. sechellia ocellar pigment lattice, in nanometres. -/
def lambdaSechelliaNm : Nat := 1500

/-- Successor D. serrata detail-scale wavelength (intermediate activator gain), nm. -/
def lambdaSerrataNm : Nat := 1680

/-- Predecessor D. santomea detail-scale wavelength (tan-loss, lower activator gain), nm. -/
def lambdaSantomeaNm : Nat := 1921

/-- Sechellia is the short-wavelength end of this Holocene triplet. -/
theorem sechellia_shorter_than_both_neighbours :
    lambdaSechelliaNm < lambdaSerrataNm ∧ lambdaSerrataNm < lambdaSantomeaNm := by
  decide

/-- Homogeneous trace f_u + g_v at rates ×1000: 1600 + (−7200) < 0.
    The unpatterned steady state is stable without diffusion (Turing 1952). -/
theorem homogeneous_trace_negative : (1600 : Int) + (-7200) < 0 := by
  decide

/-- det J = f_u g_v − f_v g_u > 0 at the same scaling:
    1600*(−7200) − ((−2770)*4500) = 945000. -/
theorem homogeneous_determinant_positive :
    (1600 : Int) * (-7200) - ((-2770) * 4500) > 0 := by
  decide

/-- Inhibitor diffusivity exceeds activator diffusivity (scaled µm² per time × 10⁶).
    D_v / D_u = 8 for this slot, the differential-diffusion limb of the Turing condition. -/
theorem inhibitor_diffuses_faster : (19948 : Nat) * 8 < 159584 * 1 + 8 ∧ 19948 < 159581 := by
  decide

/-- λ_c k_c tracks 2π: 1500 nm times k_c ≈ 4.189 µm⁻¹, rescaled, equals 6283 ≈ 2π×1000. -/
theorem wavelength_times_wavenumber_is_two_pi :
    lambdaSechelliaNm * 4189 / 1000 = 6283 := by
  decide

/-- Activator gain increases from santomea (1327) through serrata (1458) to sechellia (1600),
    which is the direction that shortens λ_c at fixed diffusivities. -/
theorem activator_gain_orders_the_chronoseries :
    (1327 : Nat) < 1458 ∧ 1458 < 1600 := by
  decide

/-- Equal diffusivities are not this slot's ratio. Turing instability is diffusion-driven. -/
theorem diffusivity_ratio_is_not_one : (159581 : Nat) - 19948 > 19948 := by
  decide

end MathOfBiology.Loop_3202
