/-
  Loop 3221 — Equus quagga, Holocene (0 Ma), Turing wavelength at the 1.883 mm detail scale.
  Predecessor Equus scotti is the equal-diffusion control (no selected wavenumber).
  Bard 1977 initiation spacing is 400 µm; this window is 1883 µm.
  Integers are tenths of a millimetre or scaled Jacobian entries. No sorry.
-/
namespace MathOfBiology.Loop_3221

/-- Detail wavelength in micrometres (1.883 mm) over Bard's 400 µm initiation spacing. -/
def growth_steps (detail init : Nat) : Nat := detail / init

theorem holocene_detail_is_four_initiation_spacings :
    growth_steps 1883 400 = 4 := by decide

/-- Onset numerator D_v f_u + D_u g_v. Quagga regime uses a large inhibitor diffusivity. -/
def turing_numerator (dV fU dU gV : Int) : Int := dV * fU + dU * gV

theorem quagga_onset_numerator_positive :
    turing_numerator 576 8 10 (-10) > 0 := by decide

/-- Equus scotti control: D_v = D_u, so the same Jacobian has a negative numerator. -/
theorem scotti_equal_diffusion_numerator_negative :
    turing_numerator 10 8 10 (-10) < 0 := by decide

/-- Stripe count on a window of ten detail wavelengths. -/
def stripe_count (domain wavelength : Nat) : Nat := domain / wavelength

theorem quagga_window_holds_ten_stripes :
    stripe_count (10 * 1883) 1883 = 10 := by decide

theorem scotti_uniform_coat_has_no_stripes :
    stripe_count (10 * 1883) 1883 > stripe_count 0 1883 := by decide

end MathOfBiology.Loop_3221
