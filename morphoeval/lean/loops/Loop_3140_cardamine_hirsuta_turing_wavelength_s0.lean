/-
  Loop 3140 — Cardamine hirsuta, Holocene extant (0 Ma).
  Detail-scale Turing pigment-ocellus wavelength λ_c = 154.1 µm.
  Schnakenberg point used only by this slot: a = 0.10, b = 0.90,
  D_u = 0.04, D_v = 0.42 (ratio 21/2).
  Jacobian at the homogeneous steady state: f_u = 4/5, g_v = -1, det = 1, trace = -1/5.
  Classical wavenumber identity: k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) = 185/21.
-/
namespace MathOfBiology.Loop_3140

/-- Numerator 0.296 and denominator 0.0336 of k_c^2, cleared by 10^4:
    2960 / 336 = 185 / 21. -/
theorem kc_sq_cross_multiply : (2960 : Int) * 21 = 336 * 185 := by
  decide

/-- Squared Turing inequality at this point.
    L = D_v f_u + D_u g_v = 0.296, and 4 D_u D_v det = 0.0672.
    L^2 > 4 D_u D_v det is 87616 > 67200 in units of 10^-6. -/
theorem turing_inequality_squared : (87616 : Int) > 67200 := by
  decide

/-- Diffusion ratio D_v / D_u = 0.42 / 0.04 = 21/2. -/
theorem diffusion_ratio_21_over_2 : (42 : Int) * 2 = 4 * 21 := by
  decide

/-- The homogeneous steady state is stable without diffusion: trace = 0.8 + (-1) < 0. -/
theorem trace_negative_without_diffusion : (8 : Int) + (-10) < 0 := by
  decide

/-- Eight classical wavelengths of 16 px fill the 128-px detail patch.
    A 16 µm cell is shorter than the 154 µm wavelength, so the mode does not fit there. -/
theorem detail_domain_holds_eight_wavelengths : (16 : Int) * 8 = 128 := by
  decide

theorem cell_shorter_than_wavelength : (16 : Int) < 154 := by
  decide

/-- Chronoseries 1/1 terminates at the Holocene: successor gap = length - index = 0,
    and the age in Ma is that same fixed point. -/
theorem holocene_terminal_chronoseries : (1 : Int) - 1 = 0 := by
  decide

/-- A positive margin between the squared Turing sides stays positive after subtraction. -/
theorem dispersion_margin_positive (lhs rhs : Int) (h : lhs > rhs) : lhs - rhs > 0 := by
  omega

end MathOfBiology.Loop_3140
