/-
  Loop 3107 — Ambystoma mexicanum, Turing onset wavelength at the 184 µm primordium scale.
  Exact lattice length 184.3 µm = 0.0001843 m (log10 = −3.734), Holocene, age 0 Ma.
  Chronoseries Ambystoma 1/1: one extant node. The integers below are the Schnakenberg
  Jacobian at (a, b) = (0.10, 0.90), diffusion ratio 9, and an 8-wavelength patch.
  No `sorry`.
-/
namespace MathOfBiology.Loop_3107

/-- Onset wavelength in tenths of a micrometre: 184.3 µm. -/
def lambdaTenthUm : Nat := 1843

/-- Eight onset wavelengths span the periodic primordium patch. -/
def nWavelengths : Nat := 8

def domainTenthUm : Nat := nWavelengths * lambdaTenthUm

theorem eight_wavelengths_span_the_flank_patch :
    domainTenthUm = 14744 := by
  decide

/-- Jacobian entries ×10 at the steady state (u₀, v₀) = (1, 0.9):
    f_u = 0.8, g_v = −1, f_v = 1, g_u = −1.8. -/
def fuTenth : Int := 8
def gvTenth : Int := -10
def fvTenth : Int := 10
def guTenth : Int := -18

def traceTenth : Int := fuTenth + gvTenth

/-- det(J) × 100 = f_u g_v − f_v g_u. -/
def detHundred : Int := fuTenth * gvTenth - fvTenth * guTenth

theorem diffusion_free_equilibrium_is_stable :
    traceTenth < 0 ∧ detHundred > 0 := by
  decide

/-- Inhibitor diffusivity over activator diffusivity. -/
def diffusionRatio : Nat := 9

theorem inhibitor_outruns_activator :
    diffusionRatio > 1 := by
  decide

/-- 1000 × D_v/D_u, compared with the ceiling of 1000 × r_c.
    r_c ≈ 8.5676 for this Jacobian, so 8568 > r_c and 9.000 > 8.568. -/
def ratioMilli : Nat := 9000
def criticalRatioMilliCeiling : Nat := 8568

theorem mildly_supercritical :
    ratioMilli > criticalRatioMilliCeiling := by
  decide

/-- Numerator of D_v f_u + D_u g_v, scaled by 10, at ratio 9: 9×0.8 − 1 = 6.2 > 0,
    so the onset wavenumber is real. -/
def onsetNumeratorTenth : Int := (9 : Int) * fuTenth + gvTenth

theorem onset_wavenumber_is_real :
    onsetNumeratorTenth > 0 := by
  decide

end MathOfBiology.Loop_3107
