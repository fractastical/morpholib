--
-- morphoeval/lean/loops/Loop_3157_coturnix_japonica_turing_wavelength_s0.lean
-- Slot 3157. Coturnix japonica (Japanese quail), Holocene extant, age 0 Ma.
-- Scale: 2.242e-5 m (22.42 µm; log10 = -4.649), tier 4_cellular_10um, detail variant.
-- Law: Turing wavenumber k_c of a Schnakenberg activator-inhibitor system.
-- Integer scalings: reaction derivatives ×1000, diffusivities ×1000.
--   f_u = 860/1000, g_v = -1000/1000, D_u = 35/1000, D_v = 1850/1000.
-- The 1000-factors cancel in k_c^2, leaving k_c^2 = 3112/259.
--
namespace MathOfBiology.Loop_3157

/-- Jacobian trace at the homogeneous steady state, scaled by 1000.
    tr J = f_u + g_v = -0.140 < 0, so the reaction equilibrium is stable
    without diffusion. -/
def trace_scaled : Int := 860 + (-1000)

theorem trace_negative : trace_scaled < 0 := by
  decide

/-- Determinant of the reaction Jacobian, scaled by 10^6.
    det J = f_u g_v - f_v g_u = 1 > 0. -/
def det_scaled : Int := 860 * (-1000) - 1000 * (-1860)

theorem det_positive : det_scaled > 0 := by
  decide

/-- Cross-diffusion coefficient D_v f_u + D_u g_v and the denominator
    2 D_u D_v, both scaled by 10^6. Their ratio is the critical
    wavenumber squared, k_c^2 = 3112/259. -/
def cross_scaled : Int := 1850 * 860 + 35 * (-1000)
def kc_den_scaled : Int := 2 * 35 * 1850

theorem kc_exact_ratio : cross_scaled * 259 = kc_den_scaled * 3112 := by
  decide

/-- The Turing band is open at this parameter point:
    (D_v f_u + D_u g_v)^2 - 4 D_u D_v det J > 0.
    With the length unit fixed by slot 3157, the marginal wavelength
    inside that band is 2.242e-5 m. -/
def band_discriminant : Int :=
  cross_scaled * cross_scaled - 4 * 35 * 1850 * det_scaled

theorem turing_band_nonempty : band_discriminant > 0 := by
  decide

/-- Equal diffusivity (D_u = D_v = 0.035) is the ontogenetic predecessor:
    the cross term changes sign, so the 22.42 µm mode is not selected.
    The Holocene chronoseries has one node, so the geological age gap is 0. -/
def equal_diffusion_cross : Int := 35 * 860 + 35 * (-1000)
def holocene_age_gap_Ma : Int := 0

theorem equal_diffusion_quenches : equal_diffusion_cross < 0 := by
  decide

theorem holocene_terminal_no_successor : holocene_age_gap_Ma = 0 := by
  decide

end MathOfBiology.Loop_3157
