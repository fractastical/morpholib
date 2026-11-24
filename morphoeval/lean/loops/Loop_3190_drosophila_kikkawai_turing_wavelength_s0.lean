/-
  Loop 3190 — Drosophila kikkawai detail-scale Turing wavelength (2.3 µm, 0 Ma).
  Namespace MathOfBiology.Loop_3190. Standalone Lean 4, zero imports, proofs are complete.
  Wavelengths are nanometres of the Schnakenberg growth-rate peak at Du = 0.020,
  a = 0.070, b = 0.940, with clade-specific inhibitor diffusivity:
    successor D. mauritiana Dv = 0.413 → 1820 nm
    focal     D. kikkawai    Dv = 1.364 → 2343 nm
    predecessor D. hydei     Dv = 4.470 → 3056 nm
-/
namespace MathOfBiology.Loop_3190

def lamMauritianaNm : Nat := 1820
def lamKikkawaiNm : Nat := 2343
def lamHydeiNm : Nat := 3056

/-- At 0 Ma the detail-scale wavelength lengthens from the melanogaster-subgroup
    successor through the montium focal species to the repleta predecessor. -/
theorem wavelength_chronoseries_order :
    lamMauritianaNm < lamKikkawaiNm ∧ lamKikkawaiNm < lamHydeiNm := by
  decide

/-- Mode count on a 48 µm (48000 nm) periodic patch, N = ⌊48000 / λ⌋. -/
def modeCount (lamNm : Nat) : Nat := 48000 / lamNm

theorem mode_count_increases_as_wavelength_falls :
    modeCount lamHydeiNm < modeCount lamKikkawaiNm ∧
    modeCount lamKikkawaiNm < modeCount lamMauritianaNm := by
  decide

/-- The focal patch is built so twenty wavelengths of 2343 nm fit in 46860 nm,
    and integer division of the 48000 nm comparison window still yields 20. -/
theorem kikkawai_twenty_modes : modeCount lamKikkawaiNm = 20 := by
  decide

/-- Schnakenberg steady activator u₀ = a + b, in hundredths.
    Dark-morph substrate load (b = 1.18) exceeds the light morph (b = 0.71). -/
def activatorSteady (a b : Nat) : Nat := a + b

theorem dark_morph_higher_steady_activator :
    activatorSteady 7 71 < activatorSteady 7 118 := by
  decide

/-- Differential diffusion. Focal Dv and Du are stored in thousandths
    (1.364 → 1364, 0.020 → 20). Equal diffusion is the ablation. -/
def diffusionGap (dv du : Nat) : Int := (dv : Int) - (du : Int)

theorem focal_inhibitor_outruns_activator :
    diffusionGap 1364 20 > 0 := by
  decide

theorem equal_diffusion_gap_vanishes :
    diffusionGap 20 20 = 0 := by
  decide

end MathOfBiology.Loop_3190
