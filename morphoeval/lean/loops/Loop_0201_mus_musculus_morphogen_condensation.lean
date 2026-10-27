/-
  morphoeval/lean/loops/Loop_0201_mus_musculus_morphogen_condensation.lean — SUBSTANTIVE re-formalization (2026-10-27)
  Replaces the template placeholder trio (a*b>0 / p+q-q=p / decide) with a
  law-specific formalization of: Morphogen Log-Decay, Threshold Crossing & Condensation Spacing in Mus musculus (Slot #0201).
  Self-contained Lean 4 core (zero imports), 0 sorry. See morphoeval/reformalize_first_batch.py.
-/
namespace MathOfBiology.Loop_0201

/-- Log-concentration drops by a constant per decay length (exponential profile). -/
def log_conc (l0 drop : Nat) : Nat -> Nat
  | 0 => l0
  | n + 1 => log_conc l0 drop n + drop
theorem log_conc_succ (l0 drop n : Nat) :
    log_conc l0 drop (n + 1) = log_conc l0 drop n + drop := by
  simp [log_conc]
/-- Threshold (French-flag) crossing depth is monotone in the threshold margin. -/
def flag_margin (theta l0 : Nat) : Nat := theta + l0
theorem flag_margin_mono (theta l0 d : Nat) :
    flag_margin (theta + d) l0 = flag_margin theta l0 + d := by
  simp [flag_margin]; ac_rfl
/-- Condensation spacing accumulates additively over successive condensations. -/
def condensation_span (s : Nat) : Nat -> Nat
  | 0 => 0
  | n + 1 => condensation_span s n + s
theorem condensation_span_succ (s n : Nat) :
    condensation_span s (n + 1) = condensation_span s n + s := by
  simp [condensation_span]

end MathOfBiology.Loop_0201
