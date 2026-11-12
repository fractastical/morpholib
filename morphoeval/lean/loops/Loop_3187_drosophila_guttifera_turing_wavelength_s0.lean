/-
  Loop 3187 — Drosophila guttifera, detail-tier Turing wavelength.
  Scale: 2.12 µm = 2120 nm (2.12e-6 m, log10 ≈ −5.674), tier 3_subcellular_um, variant s0.
  Age: Holocene (extant), 0 Ma.
  Predecessor chronospecies: Drosophila grimshawi (vein-locked anisotropy).
  Successor chronospecies: Drosophila heteroneura (elevated basal activator, no sensillum spot).

  Schnakenberg (1979) linearisation at a = 1/10, b = 9/10, Du = 16/1000, Dv = 1250/1000:
    f_u = 4/5, g_v = −1, det J = 1
    D_v f_u + D_u g_v = 1 − 2/125 = 123/125
    4 D_u D_v det J = 10/125
  Clearing the positive denominator 125^2,
    (123/125)^2 > 10/125  iff  123^2 > 10 * 125.
  Turing's marginal-wavenumber condition (k_c^2 > 0 and a positive discriminant)
  is the content of `marginal_turing_discriminant_is_positive`.
-/

namespace MathOfBiology.Loop_3187

/-- Focal linear wavelength in nanometres. 2120 nm = 2.12 µm. -/
def lambdaNm : Nat := 2120

/-- Wavelengths across the modelled cuticle patch. -/
def patchWaves : Nat := 8

/-- Activator diffusivity in thousandths (Du = 0.016). -/
def duMilli : Nat := 16

/-- Inhibitor diffusivity in thousandths (Dv = 1.250). -/
def dvMilli : Nat := 1250

/-- Grimshawi vein-parallel / vein-transverse diffusivity, numerator (3.15 = 315/100). -/
def anisoGrimshawiNum : Nat := 315

def anisoDen : Nat := 100

/-- Guttifera basal activator in thousandths (a = 0.100). -/
def activatorMilliGuttifera : Nat := 100

/-- Heteroneura basal activator in thousandths (a = 0.240). -/
def activatorMilliHeteroneura : Nat := 240

/-- The 8-wavelength patch is 16.960 µm long. -/
theorem patch_spans_eight_wavelengths :
    patchWaves * lambdaNm = 16960 := by
  decide

/-- Inhibitor diffusion exceeds activator diffusion by more than twenty-fold,
    the separation Turing (1952) requires for a finite unstable band. -/
theorem inhibitor_diffuses_faster_than_activator :
    dvMilli > 20 * duMilli := by
  decide

/-- Marginal discriminant of the focal Jacobian is strictly positive:
    123² > 10·125, hence (D_v f_u + D_u g_v)² > 4 D_u D_v det J. -/
theorem marginal_turing_discriminant_is_positive :
    (123 : Nat) * 123 > 10 * 125 := by
  decide

/-- At 0 Ma the three chronospecies are not the same prepattern.
    Grimshawi is anisotropic (315 > 100). Heteroneura's basal activator
    (240) is strictly above guttifera's (100), and guttifera's value is the
    focal a = 0.10 used to calibrate λ = 2120 nm. -/
theorem chronospecies_prepatterns_differ :
    anisoGrimshawiNum > anisoDen ∧
      activatorMilliHeteroneura > activatorMilliGuttifera ∧
      activatorMilliGuttifera = 100 := by
  decide

end MathOfBiology.Loop_3187
