/-
  Loop 3205 — Drosophila simulans, detail-scale Turing wavelength.
  Scale: 1.92 µm (1920 nm), tier 3_subcellular_um, Holocene (0 Ma).
  Chronoseries neighbours at this scale: predecessor Drosophila silvestris
  (2640 nm), successor Drosophila subobscura (1600 nm).
  Standalone core Lean 4. No imports. No sorry.
-/
namespace MathOfBiology.Loop_3205

/-- Detail-scale critical wavelengths in nanometres. -/
def lambdaSimulansNm : Nat := 1920
def lambdaSilvestrisNm : Nat := 2640
def lambdaSubobscuraNm : Nat := 1600

/-- Clasper-arc window used for the mode count: 50 simulans wavelengths. -/
def arcWindowNm : Nat := 96000

/-- Silvestris/simulans = 11/8 and simulans/subobscura = 6/5. -/
theorem chronospecies_wavelength_ratios :
    lambdaSilvestrisNm * 8 = lambdaSimulansNm * 11
      ∧ lambdaSimulansNm * 5 = lambdaSubobscuraNm * 6 := by
  decide

/-- At 0 Ma the simulans detail wavelength sits strictly between its
    catalog neighbours: subobscura is finer, silvestris is coarser. -/
theorem detail_wavelength_strictly_between_neighbours :
    lambdaSubobscuraNm < lambdaSimulansNm ∧ lambdaSimulansNm < lambdaSilvestrisNm := by
  decide

def modeCount (window lam : Nat) : Nat := window / lam

/-- Packing the same 96 µm arc with each neighbour's wavelength reverses
    the wavelength order: silvestris 36 < simulans 50 < subobscura 60. -/
theorem mode_count_orders_the_chronoseries :
    modeCount arcWindowNm lambdaSilvestrisNm < modeCount arcWindowNm lambdaSimulansNm
      ∧ modeCount arcWindowNm lambdaSimulansNm < modeCount arcWindowNm lambdaSubobscuraNm := by
  decide

/-- 0 = silvestris, 1 = simulans, 2 = subobscura.
    The epandrial posterior lobe is a melanogaster-subgroup novelty. -/
def posteriorLobePresent (species : Nat) : Nat :=
  if species = 1 then 1 else 0

theorem posterior_lobe_is_simulans_novelty :
    posteriorLobePresent 0 = 0
      ∧ posteriorLobePresent 1 = 1
      ∧ posteriorLobePresent 2 = 0 := by
  decide

/-- Diffusionless Schnakenberg Jacobian, scaled by 5 from (a, b) = (0.1, 0.9):
    fu = 4, gv = -5. Trace is negative, so the homogeneous state is stable
    before diffusion is added. -/
theorem diffusionless_trace_is_stable :
    ((4 : Int) + (-5)) < 0 := by
  decide

/-- det J = fu·gv − fv·gu with fv = 5 and gu = −9. -/
theorem jacobian_determinant_positive :
    ((4 : Int) * (-5) - (5 * (-9))) > 0 := by
  decide

/-- Diffusion drive h = Dv·fu + Du·gv at Du = 1, Dv = 20. -/
theorem diffusion_drive_positive :
    ((20 : Int) * 4 + 1 * (-5)) > 0 := by
  decide

/-- Turing discriminant h² − 4·Du·Dv·det J is positive, so a finite band of
    wavenumbers is unstable and a critical wavelength is selected. -/
theorem turing_discriminant_positive :
    ((20 : Int) * 4 + 1 * (-5)) * ((20 : Int) * 4 + 1 * (-5))
      > 4 * 1 * 20 * ((4 : Int) * (-5) - 5 * (-9)) := by
  decide

/-- Equal diffusivities collapse h onto Du·trace. Trace is negative, so
    this Jacobian cannot select a wavelength when Du = Dv. -/
theorem equal_diffusivity_blocks_wavelength_selection :
    ((20 : Int) * ((4 : Int) + (-5))) < 0 := by
  decide

end MathOfBiology.Loop_3205
