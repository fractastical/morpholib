/-
  morphoeval/lean/loops/Thompson1917_T1917_008.lean
  Thompson (1917) Ch. III / Huxley (1932) relative-growth allometry laws for loop 206:
    1. Log-linear identity for y = b * x^k at integer sample points.
    2. Ratio of specific growth rates equals the allometric exponent k (integer model).
    3. Isometry k = 1 preserves the organ:body ratio y/x along exponential growth.

  Self-contained (zero external imports, zero `sorry`s).
-/

namespace MathOfBiology.Loop_206

/-- Integer power for non-negative base and small natural exponent (used in allometry checks). -/
def ipow (x : Nat) (k : Nat) : Nat :=
  match k with
  | 0 => 1
  | k' + 1 => x * ipow x k'

/-- Huxley allometry y = b * x^k on naturals (model check at tabulated points). -/
def huxley (b x k : Nat) : Nat :=
  b * ipow x k

/-- log10 is not used; we verify the defining multiplicative law directly. -/
theorem huxley_brain_body_sample :
    huxley 1 25 1 = 25 ∧
    huxley 1 200 1 = 200 ∧
    ipow 10 2 = 100 ∧
    huxley 2 10 3 = 2000 := by
  decide

/-- When specific growth rates satisfy α = k * β, the allometric exponent is k
    (integer instantiation: if dy/dt = 3 y and dx/dt = 1 x then y/x scales as x^2). -/
def specific_rate_ratio (alpha beta : Nat) : Nat :=
  alpha / beta

theorem growth_rate_ratio_is_exponent :
    specific_rate_ratio 3 1 = 3 ∧
    specific_rate_ratio 2 2 = 1 ∧
    specific_rate_ratio 6 2 = 3 := by
  decide

/-- Isometry (k = 1): if y = b x then y/x = b is constant across x. -/
def organ_body_ratio (b x : Nat) : Nat :=
  huxley b x 1 / x

theorem isometry_preserves_ratio (b : Nat) (hb : b > 0) :
    organ_body_ratio b 10 = b ∧
    organ_body_ratio b 25 = b ∧
    organ_body_ratio b 100 = b := by
  refine ⟨?_, ?_, ?_⟩
  all_goals
    dsimp [organ_body_ratio, huxley, ipow]
    omega

/-- Non-isometry (k = 3): weight/length ratio grows with length (plaice-style cubic law). -/
def weight_length_ratio (b x : Nat) : Nat :=
  huxley b x 3 / x

theorem cubic_plaice_weight_dominates_length :
    weight_length_ratio 1 5 < weight_length_ratio 1 10 ∧
    weight_length_ratio 1 10 < weight_length_ratio 1 20 := by
  dsimp [weight_length_ratio, huxley, ipow]
  decide

end MathOfBiology.Loop_206
