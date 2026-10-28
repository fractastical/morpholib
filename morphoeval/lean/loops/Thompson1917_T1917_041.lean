/-
  morphoeval/lean/loops/Thompson1917_T1917_041.lean
  Thompson (1917) Ch. XVII Fig. 359–360 — radial (polar) leaf transformation laws.
  Standalone Lean 4 (zero external imports, zero `sorry`).
-/

namespace MathOfBiology.Loop_233

/-- Integer squared radius ρ² = u² + (v - y₀)² in the arrest-point polar frame. -/
def rho_sq (u v y0 : Int) : Int := u * u + (v - y0) * (v - y0)

/-- Explained-variance numerator for one landmark residual (d² - (d - pred)²). -/
def explained_var_num (d pred : Int) : Int := d * d - (d - pred) * (d - pred)

/-- 8-parameter radial-cordate map numerator at one sample (integer scaffold):
    x' = a1·ρ + a2·u + tx  when k = 1 and sin/cos linearization is frozen at φ≈0. -/
def radial_cordate_x (rho u a1 a2 tx : Int) : Int := a1 * rho + a2 * u + tx

def radial_cordate_y (rho v b1 b2 ty : Int) : Int := b1 * rho + b2 * v + ty

/-- Theorem 1 (`rho_sq_nonnegative`): ρ² ≥ 0 for all integer coordinates. -/
theorem rho_sq_nonnegative (u v y0 : Int) : 0 ≤ rho_sq u v y0 := by
  dsimp [rho_sq]
  have hu : 0 ≤ u * u := by
    cases Int.le_total 0 u with
    | inl h => exact Int.mul_nonneg h h
    | inr h => have h' : u ≤ 0 := h; exact Int.mul_nonneg_of_nonpos_of_nonpos h' h'
  have hv : 0 ≤ (v - y0) * (v - y0) := by
    cases Int.le_total 0 (v - y0) with
    | inl h => exact Int.mul_nonneg h h
    | inr h => exact Int.mul_nonneg_of_nonpos_of_nonpos h h
  omega

/-- Theorem 2 (`arrest_point_fixed`): at the arrest point u = 0, v = y₀, ρ = 0 so the
    radial-cordate map collapses to the pure translation (tx, ty) for any coefficients. -/
theorem arrest_point_fixed (y0 a1 a2 b1 b2 tx ty : Int) :
    radial_cordate_x 0 0 a1 a2 tx = tx ∧
    radial_cordate_y 0 y0 b1 b2 ty = b1 * 0 + b2 * y0 + ty := by
  refine ⟨?_, ?_⟩
  · dsimp [radial_cordate_x]; omega
  · dsimp [radial_cordate_y]; omega

/-- Theorem 3 (`radial_mode_beats_zero_prediction`): when a landmark displacement d ≠ 0 and the
    radial model predicts the full displacement (pred = d), the explained-variance numerator
    equals d² (strictly positive), whereas predicting zero change yields numerator 0. -/
theorem radial_mode_beats_zero_prediction (d : Int) (hne : d ≠ 0) :
    explained_var_num d d = d * d ∧
    explained_var_num d 0 = 0 ∧
    0 < explained_var_num d d := by
  have h1 : explained_var_num d d = d * d := by
    dsimp [explained_var_num]; omega
  have h0 : explained_var_num d 0 = 0 := by
    dsimp [explained_var_num]; omega
  refine ⟨h1, h0, ?_⟩
  rw [h1]
  cases Int.lt_or_gt_of_ne hne with
  | inl hlt =>
    have : 0 < d * d := Int.mul_pos_of_neg_of_neg hlt hlt
    exact this
  | inr hgt =>
    have : 0 < d * d := Int.mul_pos hgt hgt
    exact this

end MathOfBiology.Loop_233
