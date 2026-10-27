/-
  morphoeval/lean/loops/Loop_3167_dickinsonia_costata_turing_wavelength_s0.lean
  Dickinsonia costata — Turing wavenumber at the cellular tier.
  Scale: 56 µm (5.564e-5 m), tier 4/8 `4_cellular_10um`.
  Age: Late Ediacaran, White Sea, 558 Ma. Chronoseries 1/1 (no second chronospecies).
  The organ-tier sibling pitch is 2200 µm. Integer arithmetic records the
  scale separation: one module septum holds 39 cellular wavelengths, and
  diffusivity at fixed reaction time rescales by 39² = 1521.
-/
namespace MathOfBiology.Loop_3167

/-- Law 1. Dispersion product of the two diffusivities (both must be positive). -/
def law1_invariant (du dv : Int) : Int := du * dv

theorem law1_positive_scaling (du dv : Int) (hdu : 0 < du) (hdv : 0 < dv) :
    0 < law1_invariant du dv := by
  dsimp [law1_invariant]
  exact Int.mul_pos hdu hdv

/-- Law 2. Critical wavelength is unchanged by an auxiliary wavenumber offset. -/
def law2_balance (lam kExtra : Int) : Int := lam + kExtra - kExtra

theorem law2_exact_cancellation (lam kExtra : Int) :
    law2_balance lam kExtra = lam := by
  dsimp [law2_balance]
  omega

/-- Cellular pigment pitch versus the organ-module pitch at the same 558 Ma horizon. -/
def cellularPitchUm : Int := 56
def organModuleUm : Int := 2200

def stripesPerSeptum : Int := organModuleUm / cellularPitchUm

theorem cellular_pitch_finer_than_organ_module :
    cellularPitchUm < organModuleUm := by
  decide

theorem septum_holds_39_cellular_wavelengths :
    stripesPerSeptum = 39 := by
  decide

/-- Diffusivity rescales with the square of the wavelength ratio at fixed reaction time. -/
def diffusionScaleFactor : Int := stripesPerSeptum * stripesPerSeptum

theorem diffusion_rescales_by_1521 :
    diffusionScaleFactor = 1521 := by
  decide

/-- White Sea horizon of the single Dickinsonia costata chronospecies in this series. -/
def whiteSeaAgeMa : Int := 558

theorem white_sea_age_positive :
    0 < whiteSeaAgeMa := by
  decide

end MathOfBiology.Loop_3167
