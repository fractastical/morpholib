/-
  Loop 3176 — Drosophila ananassae, Turing wavenumber at the 2.1 µm detail scale.
  Geological age 0 Ma (Holocene, extant). Lineage: Sophophora ananassae–montium.
  Chronoseries 5/41. Predecessor: Drosophila americana (139 µm primordium wavelength).
  Successor: Drosophila biarmipes (96 µm wing-spot wavelength).
  Wavelengths are stored in nanometres so the comparisons are exact integers.
  Diffusion coefficients are milli-units of the nominal Schnakenberg pair
  (Du = 0.604, Dv = 11.893, fu = 0.825, |gv| = 1.588).
-/
namespace MathOfBiology.Loop_3176

def lambdaS0Nm : Nat := 2106
def lambdaAmericanaNm : Nat := 138700
def lambdaBiarmipesNm : Nat := 95980

/-- Six critical wavelengths span the 12.636 µm intracellular window. -/
def cellPatchNm : Nat := 12636

/-- h = Dv·fu − Du·|gv| in milli-units. Positive h is the trace-diffusion
    condition Dv fu + Du gv > 0 when gv is negative. -/
def turingH (Du Dv fu gvAbs : Nat) : Int :=
  (Dv : Int) * (fu : Int) - (Du : Int) * (gvAbs : Int)

theorem six_wavelengths_span_cell_patch :
    6 * lambdaS0Nm = cellPatchNm := by decide

/-- The americana primordium wavelength is more than 60× the ananassae
    detail wavelength, and still coarser than the biarmipes wing-spot mode. -/
theorem americana_coarser_than_ananassae_detail :
    lambdaAmericanaNm > 60 * lambdaS0Nm ∧ lambdaBiarmipesNm < lambdaAmericanaNm := by decide

/-- The biarmipes successor re-expands above the 2.106 µm detail scale
    (more than 40×) without returning to the americana primordium scale. -/
theorem biarmipes_reexpands_above_detail :
    lambdaBiarmipesNm > 40 * lambdaS0Nm ∧ lambdaBiarmipesNm < 50 * lambdaS0Nm := by decide

/-- Nominal intracellular ratio Dv/Du = 19.7 satisfies the Turing h-condition.
    Equal diffusion (the ablation) does not. -/
theorem nominal_ratio_is_turing_equal_diffusion_is_not :
    turingH 604 11893 825 1588 > 0 ∧ turingH 604 604 825 1588 < 0 := by decide

end MathOfBiology.Loop_3176
