-- Loop 3227. Fractofusus misrai, detail-scale Turing wavelength.
-- Phenomenon: 63.3 µm = 6.33e-5 m (tier 4_cellular_10um), Mistaken Point, 565 Ma.
-- Chronoseries 1/1. The organ-scale neighbour is the 3.275 mm module (slot 1251).
-- Fifty-one detail-scale wavelengths fit in that module; fifty-two do not.
namespace MathOfBiology.Loop_3227

/-- Wavelength in tenths of a micrometre: 633 = 63.3 µm = 6.33e-5 m. -/
def lambdaTenths : Nat := 633

/-- Organ-scale primary module in tenths of a micrometre: 32750 = 3.275 mm. -/
def organTenths : Nat := 32750

/-- Scale separation against the 3.275 mm module, not a second chronospecies. -/
theorem scale_separation :
    51 * lambdaTenths < organTenths ∧ organTenths < 52 * lambdaTenths := by
  decide

/-- Nominal inhibitor/activator diffusivity ratio D_v / D_u at this detail scale. -/
def diffusionRatio : Nat := 20

theorem inhibitor_diffuses_faster : diffusionRatio > 1 := by
  decide

/-- Diffusion-weighted trace in milli-units.
    f_u ≈ 0.809 and |g_v| ≈ 0.884 at a = 0.09, b = 0.85.
    h = (D_v/D_u) f_u - |g_v| > 0 is the condition that opens a Turing band. -/
theorem turing_h_positive : 20 * 809 - 884 > 0 := by
  decide

/-- 565 Ma lies inside the mid-Ediacaran window ca. 575–560 Ma
    reported by Gehling & Narbonne 2007. -/
theorem age_inside_mid_ediacaran : 560 ≤ 565 ∧ 565 ≤ 575 := by
  decide

end MathOfBiology.Loop_3227
