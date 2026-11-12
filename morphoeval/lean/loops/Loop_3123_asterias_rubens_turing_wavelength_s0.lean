/-
  morphoeval/lean/loops/Loop_3123_asterias_rubens_turing_wavelength_s0.lean
  Asterias rubens, slot 3123.
  Scale: 115.2 µm = 0.0001152 m (tier 5/8 primordium, detail).
  Age: Holocene (extant), 0 Ma. Chronoseries length 1/1.
  Law: Turing wavenumber. Five critical wavelengths fill a 576.0 µm patch,
  one per pentaradial sector. The 10 µm cellular tier is finer; the 1 mm
  arm-tip organ tier is coarser. Schnakenberg ratio r = 7.62 sits above onset,
  so the diffusion-drive term is positive while the homogeneous trace is negative.
-/
namespace MathOfBiology.Loop_3123

/-- Catalog wavelength and domain, in tenths of a micrometre. -/
def lambdaTenths : Int := 1152
def nWaves : Int := 5
def domainTenths : Int := 5760

theorem five_wavelengths_fill_domain :
    nWaves * lambdaTenths = domainTenths := by
  decide

/-- Homogeneous Schnakenberg state is stable with diffusion off.
    Numerator of trace = f_u + g_v = 47/55 − 9801/10000. -/
theorem homogeneous_trace_negative :
    (47 * 10000 + (-9801) * 55) < 0 := by
  decide

/-- At D_v/D_u = 7.62 the drive D_v f_u + D_u g_v is positive.
    Numerator of (762/100)·(47/55) − 9801/10000 on denominator 110000. -/
theorem diffusion_drive_positive :
    (762 * 47 * 20 - 9801 * 11) > 0 := by
  decide

/-- This scale sits strictly between a 10 µm cellular predecessor tier
    and a 1000 µm organ successor tier (tenths of a micrometre). -/
theorem scale_between_cellular_and_organ :
    100 < lambdaTenths ∧ lambdaTenths < 10000 := by
  decide

end MathOfBiology.Loop_3123
