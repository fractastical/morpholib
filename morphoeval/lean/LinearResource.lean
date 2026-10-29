/-
  morphoeval/lean/LinearResource.lean
  =============================================================================
  Prototype of the linear-resource DSL proposed in docs/LINEAR_TYPE_EXTENSION_SPEC.md
  (Project Delphi ontology extension: proofKind "conservation-by-typing").
  Core Lean 4 only (no Mathlib). Verify with:  lean morphoeval/lean/LinearResource.lean
  (0 sorry, no additional axioms). STATUS: PROTOTYPE — not yet emitted into the KG.

  Idea: a conserved quantity lives in a ledger (one Int per compartment). The ONLY
  program constructors are `skip`, `transfer`, and `seq`. There is no constructor
  that creates or destroys resource, so conservation is a property of the typing
  discipline (every well-scoped program preserves the ledger total), proved once,
  and inherited by every biological instantiation.
-/
namespace MathOfBiology.Linear

abbrev Ledger := List Int

/-- Add `q` to compartment `i` (no-op when `i` is out of range). -/
def bump (L : Ledger) (i : Nat) (q : Int) : Ledger :=
  L.set i (L.getD i 0 + q)

theorem length_bump (L : Ledger) (i : Nat) (q : Int) : (bump L i q).length = L.length := by
  simp [bump]

theorem sum_bump (L : Ledger) (i : Nat) (q : Int) (h : i < L.length) :
    (bump L i q).sum = L.sum + q := by
  induction L generalizing i with
  | nil => simp at h
  | cons a rest ih =>
    cases i with
    | zero => simp [bump, Int.add_assoc, Int.add_comm]
    | succ k =>
      have hk : k < rest.length := by simpa using h
      have := ih k hk
      simp only [bump] at this ⊢
      simp only [List.set_cons_succ, List.getD_cons_succ, List.sum_cons]
      rw [this]
      omega

/-- The linear DSL. `transfer i j q` moves `q` units from compartment `i` to `j`. -/
inductive Prog where
  | skip : Prog
  | transfer (i j : Nat) (q : Int) : Prog
  | seq : Prog → Prog → Prog

/-- Typing judgment: every compartment index a program mentions is in range for `n` cells. -/
inductive WellScoped (n : Nat) : Prog → Prop where
  | skip : WellScoped n .skip
  | transfer {i j q} : i < n → j < n → WellScoped n (.transfer i j q)
  | seq {p₁ p₂} : WellScoped n p₁ → WellScoped n p₂ → WellScoped n (.seq p₁ p₂)

def exec : Prog → Ledger → Ledger
  | .skip, L => L
  | .transfer i j q, L => bump (bump L i (-q)) j q
  | .seq p₁ p₂, L => exec p₂ (exec p₁ L)

theorem length_exec (p : Prog) (L : Ledger) : (exec p L).length = L.length := by
  induction p generalizing L with
  | skip => rfl
  | transfer i j q => simp [exec, length_bump]
  | seq p₁ p₂ ih₁ ih₂ => simp [exec, ih₁, ih₂]

/-- Soundness of the linear discipline: well-scoped programs conserve the ledger total. -/
theorem conserves (n : Nat) (p : Prog) (hp : WellScoped n p) (L : Ledger) (hL : L.length = n) :
    (exec p L).sum = L.sum := by
  induction hp generalizing L with
  | skip => rfl
  | transfer hi hj =>
    rename_i i j q
    have hi' : i < L.length := by omega
    have hj' : j < (bump L i (-q)).length := by rw [length_bump]; omega
    simp only [exec]
    rw [sum_bump _ _ _ hj', sum_bump _ _ _ hi']
    omega
  | seq _ _ ih₁ ih₂ =>
    simp only [exec]
    rw [ih₂ _ (by rw [length_exec]; exact hL), ih₁ _ hL]

-- ---------------------------------------------------------------------------
-- Instantiation 1: Jönsson et al. (2006) PIN1-mediated auxin flux (Loop_032).
-- The per-edge flux is the existing MathOfBiology definition; the *step* is a transfer,
-- so total auxin conservation is `conserves`, not a separate Int identity.
-- ---------------------------------------------------------------------------
def intercellular_auxin_flux (T D P_ij P_ji a_i a_j : Int) : Int :=
  T * (P_ji * a_j - P_ij * a_i) + D * (a_j - a_i)

def auxin_edge_step (T D P_ij P_ji : Int) (i j : Nat) (L : Ledger) : Prog :=
  .transfer j i (intercellular_auxin_flux T D P_ij P_ji (L.getD i 0) (L.getD j 0))

theorem auxin_edge_step_conserves (T D P_ij P_ji : Int) (i j : Nat) (L : Ledger)
    (hi : i < L.length) (hj : j < L.length) :
    (exec (auxin_edge_step T D P_ij P_ji i j L) L).sum = L.sum :=
  conserves L.length _ (.transfer hj hi) L rfl

/-- Jönsson's antisymmetry (`jonsson_2006_intercellular_flux_conservation`, Loop_032 L64),
    unchanged: the per-edge flux seen from `j` is the negative of the flux seen from `i`. -/
theorem auxin_flux_antisymmetric (T D P_ij P_ji a_i a_j : Int) :
    intercellular_auxin_flux T D P_ji P_ij a_j a_i = -(intercellular_auxin_flux T D P_ij P_ji a_i a_j) := by
  simp only [intercellular_auxin_flux, Int.mul_add, Int.mul_neg, Int.sub_eq_add_neg, Int.neg_add,
    Int.neg_neg]
  omega

/-- What the DSL adds: the reverse edge is *syntactically* the reverse transfer, so a
    simulator that builds its step from `auxin_edge_step` cannot double-count an edge. -/
theorem auxin_edge_step_reverse (T D P_ij P_ji : Int) (i j : Nat) (L : Ledger) :
    auxin_edge_step T D P_ji P_ij j i L =
      .transfer i j (-(intercellular_auxin_flux T D P_ij P_ji (L.getD i 0) (L.getD j 0))) := by
  unfold auxin_edge_step
  rw [auxin_flux_antisymmetric]

-- ---------------------------------------------------------------------------
-- Instantiation 2: Murray (1926) bifurcation as a split of flow (MathOfBiology L782-816).
-- ---------------------------------------------------------------------------
def murray_bifurcation (parent left right : Nat) (q_left q_right : Int) : Prog :=
  .seq (.transfer parent left q_left) (.transfer parent right q_right)

theorem murray_bifurcation_conserves (n parent left right : Nat) (q_left q_right : Int)
    (hp : parent < n) (hl : left < n) (hr : right < n) (L : Ledger) (hL : L.length = n) :
    (exec (murray_bifurcation parent left right q_left q_right) L).sum = L.sum :=
  conserves n _ (.seq (.transfer hp hl) (.transfer hp hr)) L hL

end MathOfBiology.Linear
