/-
  Loop 3194 — Drosophila mojavensis at the 2.4 µm detail scale (2.38e-06 m, log10 = -5.623).
  Age: Holocene (Extant), 0 Ma. Chronoseries 23/41.
  Predecessor Drosophila miranda (3237 nm). Focal 2380 nm. Successor Drosophila novamexicana (1948 nm).
  Law: Turing wavenumber. Wavelength tracks sqrt(diffusivity) at fixed Schnakenberg Jacobian and R = D_v/D_u = 80.
  No sorry.
-/

namespace MathOfBiology.Loop_3194

/-- Focal epicuticle wavelength in nanometres. Catalog label 2.4 µm; SI value 2.38e-06 m. -/
def lambdaMojavensisNm : Nat := 2380

/-- Predecessor D. miranda. Diffusivity scale s = (3237/2380)^2 > 1, so the wavelength is longer. -/
def lambdaMirandaNm : Nat := 3237

/-- Successor D. novamexicana. Diffusivity scale s = (1948/2380)^2 < 1, so the wavelength is shorter. -/
def lambdaNovamexicanaNm : Nat := 1948

theorem chronoseries_wavelength_decreases :
    lambdaMirandaNm > lambdaMojavensisNm ∧
      lambdaMojavensisNm > lambdaNovamexicanaNm := by
  decide

/-- Square law: ordering of λ² is the ordering of the diffusivity scale s. -/
def wavelengthSq (lam : Nat) : Nat := lam * lam

theorem diffusivity_square_law_order :
    wavelengthSq lambdaMirandaNm > wavelengthSq lambdaMojavensisNm ∧
      wavelengthSq lambdaMojavensisNm > wavelengthSq lambdaNovamexicanaNm := by
  decide

/-- Integer Turing margin at diffusion ratio 80.
    f_u and g_v are the Schnakenberg Jacobian entries in thousandths
    (f_u ≈ 0.315, g_v ≈ -1.166). A positive numerator is required for a real k_c. -/
def kcNumerator (ratio fu gv : Int) : Int := ratio * fu + gv

theorem mojavensis_kc_numerator_positive :
    kcNumerator 80 315 (-1166) > 0 := by
  decide

/-- Equal-diffusion ablation (ratio 1) has a negative trace-diffusion numerator for these
    kinetics, and sits strictly below the ratio-80 margin. -/
theorem differential_diffusion_exceeds_ablation :
    kcNumerator 80 315 (-1166) > kcNumerator 1 315 (-1166) := by
  decide

end MathOfBiology.Loop_3194
