/-
  morphoeval/lean/loops/Loop_3235_gekko_gecko_turing_wavelength_s0.lean
  Gekko gecko — Turing wavenumber selection at the 242 µm detail scale.
  Tier 5/8 primordium, detail variant. Holocene (extant), 0 Ma.
  Chronoseries 1/1: no predecessor and no successor chronospecies.
  Versus the tier-center wavelength 100 µm the mode count on a fixed field
  falls by (100/242)², which is strictly less than 1/5.
-/

namespace MathOfBiology.Loop_3235

/-- Selected ocellus wavelength in micrometres (slot 3235 detail scale). -/
def lambdaUm : Nat := 242

/-- Tier-5 center wavelength in micrometres (`5_primordium_100um`). -/
def tierCenterUm : Nat := 100

/-- Holocene age of the single chronoseries node, in Ma. -/
def ageMa : Nat := 0

/-- Operating diffusion ratio as parts per thousand of the kinetic threshold.
    The script sets D_v/D_u = 1.350 × r_crit. -/
def ratioMarginPpt : Nat := 1350

/-- Positive product of the two diffusivities (integer stand-in for D_u D_v > 0). -/
def diffusivityProduct (du dv : Int) : Int := du * dv

theorem diffusivityProduct_positive (du dv : Int) (hu : 0 < du) (hv : 0 < dv) :
    0 < diffusivityProduct du dv := by
  dsimp [diffusivityProduct]
  exact Int.mul_pos hu hv

theorem detail_wavelength_exceeds_tier_center :
    lambdaUm > tierCenterUm := by
  decide

/-- 242/100 > 2.4, i.e. this detail scale is more than 2.4× the tier center. -/
theorem wavelength_ratio_exceeds_two_point_four :
    lambdaUm * 10 > tierCenterUm * 24 := by
  decide

/-- (100/242)² < 1/5, so a fixed field carries fewer than one fifth as many
    242 µm modes as 100 µm modes. -/
theorem mode_count_falls_versus_tier_center :
    tierCenterUm * tierCenterUm * 5 < lambdaUm * lambdaUm := by
  decide

theorem holocene_terminal_age :
    ageMa = 0 := by
  decide

theorem supercritical_margin :
    ratioMarginPpt > 1000 := by
  decide

end MathOfBiology.Loop_3235
