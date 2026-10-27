/-
  Loop 3177 — Drosophila biarmipes, detail scale s0.
  Phenomenon: Turing pigment stripe / ocellus wavelength, 1.5 µm
  (catalog 1.486e-6 m, log10 = -5.828, tier 3_subcellular_um).
  Age: Holocene (Extant), 0 Ma.
  Chronoseries 6/41, Sophophora suzukii–biarmipes wing-spot clade.
  Predecessor: Drosophila ananassae (subcritical activator).
  Successor: Drosophila bipectinata (stronger autocatalysis, shorter λ).

  Schnakenberg Jacobian at the biarmipes steady state (a, b) = (1/10, 9/10),
  reported in thousandths: f_u = 800, f_v = 1000, g_u = -1800, g_v = -1000.
  Diffusivities frozen in millionths of µm²: D_u = 10351, D_v = 372636
  so that the linear critical wavelength is the 1.486 µm detail scale.
  Patch side 17.832 µm = 17832 milli-µm holds exactly 17832/1486 = 12 wavelengths.
-/

namespace MathOfBiology.Loop_3177

/-- Jacobian trace f_u + g_v in thousandths. Negative trace: the well-mixed
    steady state is not a source. -/
def jacobianTraceMilli : Int := 800 + (-1000)

theorem jacobian_trace_negative : jacobianTraceMilli < 0 := by
  decide

/-- det J = f_u g_v - f_v g_u in millionths. Positive determinant together
    with the negative trace makes both eigenvalues stable without diffusion. -/
def jacobianDetMicro : Int := 800 * (-1000) - 1000 * (-1800)

theorem jacobian_det_positive : jacobianDetMicro > 0 := by
  decide

/-- Turing inequality (D_v f_u + D_u g_v)² > 4 D_u D_v det J.
    D_u and D_v are in millionths of µm²; f_u and g_v are in thousandths;
    the right-hand side is realigned by 10⁶ so both sides share one scale.
    The inequality is the diffusion-driven instability at 1.5 µm. -/
theorem turing_diffusion_inequality :
    let num : Int := 372636 * 800 + 10351 * (-1000)
    num * num > 4 * 10351 * 372636 * 1000000 := by
  decide

/-- A 17.832 µm cuticle patch at the 1.486 µm detail wavelength holds
    twelve pigment wavelengths. Integer division is exact: 1486 * 12 = 17832. -/
theorem detail_patch_holds_twelve_wavelengths :
    (17832 : Int) / 1486 = 12 := by
  decide

/-- Successor chronospecies D. bipectinata raises the autocatalytic ratio.
    f_u(bipectinata) = 33/37 and f_u(biarmipes) = 4/5; cross-multiplication
    33*5 > 4*37 is the kinetic change at this scale (same diffusivities,
    stronger activator, shorter selected wavelength). -/
theorem successor_autocatalysis_stronger :
    (33 : Int) * 5 > 4 * 37 := by
  decide

/-- Predecessor D. ananassae has f_u = -1 + 2*(0.25)/(0.80) = -0.375,
    i.e. -1000 + (2*250*1000)/800 = -375 thousandths. Autocatalysis is off,
    so the 1.5 µm band stays subcritical. -/
theorem predecessor_activator_subcritical :
    (-1000 : Int) + (2 * 250 * 1000) / 800 < 0 := by
  decide

end MathOfBiology.Loop_3177
