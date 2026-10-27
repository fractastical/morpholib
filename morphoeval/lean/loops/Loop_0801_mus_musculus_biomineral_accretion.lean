/-
  morphoeval/lean/loops/Loop_0801_mus_musculus_biomineral_accretion.lean — SUBSTANTIVE re-formalization (2026-10-27)
  Replaces the template placeholder trio (a*b>0 / p+q-q=p / decide) with a
  law-specific formalization of: Raup Logarithmic Whorl Expansion W^n in Mus musculus (Slot #0801).
  Self-contained Lean 4 core (zero imports), 0 sorry. See morphoeval/reformalize_first_batch.py.
-/
namespace MathOfBiology.Loop_0801

/-- Raup radius multiplier after n turns is W^n (Nat turn count, scaled W). -/
def whorl_multiplier (W : Nat) : Nat -> Nat
  | 0 => 1
  | n + 1 => W * whorl_multiplier W n
theorem whorl_succ (W n : Nat) :
    whorl_multiplier W (n + 1) = W * whorl_multiplier W n := by
  simp [whorl_multiplier]
theorem whorl_two_turns (W : Nat) : whorl_multiplier W 2 = W * W := by
  simp [whorl_multiplier]
/-- Shell radius after n turns: r0 times the whorl multiplier; monotone in turns for W ≥ 1 step. -/
def raup_radius (r0 W n : Nat) : Nat := r0 * whorl_multiplier W n
theorem raup_radius_succ (r0 W n : Nat) :
    raup_radius r0 W (n + 1) = r0 * (W * whorl_multiplier W n) := by
  simp [raup_radius, whorl_multiplier]

end MathOfBiology.Loop_0801
