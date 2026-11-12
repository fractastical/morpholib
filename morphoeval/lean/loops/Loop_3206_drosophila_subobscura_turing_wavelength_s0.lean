/-
  Loop 3206 — Drosophila subobscura, Turing wavenumber at the 3.0 µm detail scale.
  λ = 2970 nm = 2.97 µm (2.97e-6 m, log10 = -5.527), Holocene, 0 Ma.
  Predecessor: Drosophila simulans, 82560 nm (82.56 µm).
  Successor: Drosophila suzukii, 98450 nm (98.45 µm).
  Diffusion product uses Du = 41 and Dv = 530 in units of 10^-3.
-/
namespace MathOfBiology.Loop_3206

/-- Detail-scale Turing wavelength of Drosophila subobscura, in nanometres. -/
def lambdaDetailNm : Nat := 2970

/-- Predecessor Drosophila simulans cellular-stripe wavelength, in nanometres. -/
def lambdaSimulansNm : Nat := 82560

/-- Successor Drosophila suzukii wing-spot wavelength, in nanometres. -/
def lambdaSuzukiiNm : Nat := 98450

/-- Activator–inhibitor diffusion product. Both diffusivities are positive. -/
def diffusionProduct (Du Dv : Nat) : Nat := Du * Dv

theorem diffusion_product_positive :
    diffusionProduct 41 530 > 0 := by
  decide

/-- k_subobscura / k_simulans = λ_simulans / λ_detail > 27. -/
theorem detail_wavenumber_exceeds_predecessor :
    lambdaDetailNm * 27 < lambdaSimulansNm := by
  decide

/-- k_subobscura / k_suzukii = λ_suzukii / λ_detail > 33. -/
theorem detail_wavenumber_exceeds_successor :
    lambdaDetailNm * 33 < lambdaSuzukiiNm := by
  decide

/-- Integer mode counts on each chronospecies' own wavelength: 27 and 33. -/
theorem chronospecies_mode_counts :
    lambdaSimulansNm / lambdaDetailNm = 27 ∧ lambdaSuzukiiNm / lambdaDetailNm = 33 := by
  decide

end MathOfBiology.Loop_3206
