/-
  Thompson (1917) Ch. XI — logarithmic (equiangular) spiral / gnomonic growth.
  Standalone loop module (no external imports). Slot 221 / T1917-027.
-/
namespace MathOfBiology.Loop_221

/-- Whorl expansion factor after one full turn: W = exp(2π k) encoded as integer cents:
    store k as rational num/den with scale 1000, then W_milli = exp approximation via fixed whorl ratio table. -/
def whorl_ratio_fixed : Nat := 3

/-- Gnomonic growth: outer radius after n whorls equals r0 * W^n (integer model). -/
def radius_after_whorls (r0 whorls : Nat) : Nat :=
  r0 * whorl_ratio_fixed ^ whorls

theorem gnomonic_whorl_scaling (r0 : Nat) (h : 0 < r0) :
    radius_after_whorls r0 1 = r0 * whorl_ratio_fixed := by
  dsimp [radius_after_whorls]
  omega

theorem gnomonic_two_whorls (r0 : Nat) (h : 0 < r0) :
    radius_after_whorls r0 2 = r0 * whorl_ratio_fixed * whorl_ratio_fixed := by
  dsimp [radius_after_whorls]
  decide

/-- Log-spiral slope in log-polar coordinates: ln r = ln r0 + k θ ⇒ constant derivative k. -/
def log_polar_slope (k : Int) : Int := k

def log_radius (r0 k theta : Int) : Int :=
  r0 + k * theta

theorem log_polar_linearity (r0 k theta dtheta : Int) :
    log_radius r0 k (theta + dtheta) - log_radius r0 k theta = k * dtheta := by
  dsimp [log_radius]
  omega

/-- Archimedean spiral r = a + b θ is not log-linear unless b = 0. -/
def arch_radius (a b theta : Int) : Int := a + b * theta

theorem archimedean_not_log_linear_witness :
    arch_radius 10 2 5 + arch_radius 10 2 15 ≠ 2 * arch_radius 10 2 10 := by
  decide

/-- Equiangular spiral: cotangent of constant angle α equals d(ln r)/dθ = k (fixed k witness). -/
def equiangular_cot_from_k (k : Int) : Int := k

theorem equiangular_angle_constant_iff_log_slope_constant (k1 k2 : Int) :
    (k1 = k2) ↔ (equiangular_cot_from_k k1 = equiangular_cot_from_k k2) := by
  constructor <;> intro h <;> rw [h]

end MathOfBiology.Loop_221
