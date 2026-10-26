/-
  morphoeval/lean/Ruliobiology.lean
  =============================================================================
  Wolfram (2024, 2025, 2026) Ruliobiology — the symmetric k=5, r=1 cellular
  automaton "rulioorganism" as ONE Lean 4 object that is both executed and
  reasoned about.  The functions below (`step`, `develop`, `lifetime`,
  `excise`, `evolve`) are the simulation; the theorems are about those same
  functions, not about a separate scalar re-statement of them.

  Kernel-check (0 sorry, 0 custom axioms, no Mathlib):
    lean morphoeval/lean/Ruliobiology.lean

  Run as the development engine (JSON on stdout):
    lean --run morphoeval/lean/Ruliobiology.lean develop <genome75> <maxSteps>
    lean --run morphoeval/lean/Ruliobiology.lean excise  <genome75> <maxSteps> <t0> <lo> <hi> <k>
    lean --run morphoeval/lean/Ruliobiology.lean evolve  <seed> <nMutations> <maxSteps>

  <genome75> is a 75-character string over {0,1,2,3,4}: one colour per
  left-right reflection orbit of (l, c, r) ∈ {0..4}^3, indexed exactly as in
  models/2026-wolfram-ruliobiology.py (`build_symmetric_k5_orbits`).
-/

namespace Ruliobiology

/-! ## 1. Colours, rows, rules -/

abbrev Color := Fin 5
abbrev Row   := List Color

/-- A local rule `f(l, c, r)`. -/
abbrev Rule := Color → Color → Color → Color

/-- Reflection symmetry `f(l, c, r) = f(r, c, l)`. -/
def Rule.Symmetric (f : Rule) : Prop := ∀ l c r, f l c r = f r c l

/-- Quiescence `f(0, 0, 0) = 0`: the void stays void. -/
def Rule.Quiescent (f : Rule) : Prop := f 0 0 0 = 0

/-! ## 2. Genomes: 75 reflection orbits of the 125 neighbourhood triples -/

/-- Canonical representative `(min l r, c, max l r)` of the orbit of `(l, c, r)`. -/
def canon (l c r : Color) : Color × Color × Color :=
  if l ≤ r then (l, c, r) else (r, c, l)

theorem canon_comm (l c r : Color) : canon l c r = canon r c l := by
  unfold canon
  by_cases h₁ : l ≤ r <;> by_cases h₂ : r ≤ l <;> simp [h₁, h₂]
  · exact ⟨Fin.le_antisymm h₁ h₂, Fin.le_antisymm h₂ h₁⟩
  · simp [Fin.le_def] at h₁ h₂; omega

/-- Orbit index of a canonical triple `(a, c, b)` with `a ≤ b`, in the
    first-encounter order of the Python reference implementation
    (lexicographic in `(a, c, b)`): `Σ_{a' < a} 5(5 - a') + c(5 - a) + (b - a)`. -/
def orbitIndex (a c b : Color) : Nat :=
  let a := a.val; let c := c.val; let b := b.val
  (List.range a).foldl (fun acc a' => acc + 5 * (5 - a')) 0 + c * (5 - a) + (b - a)

def orbitOf (l c r : Color) : Nat :=
  let (a, c, b) := canon l c r
  orbitIndex a c b

/-- A genome assigns a colour to each of the 75 orbits. -/
abbrev Genome := Fin 75 → Color

/-- The rule a genome encodes. Quiescence of orbit 0 is enforced by construction,
    exactly as `orbit_genome_to_lookup` does with `lut[0] = 0`. -/
def ruleOf (g : Genome) : Rule := fun l c r =>
  if h : orbitOf l c r < 75 then (if orbitOf l c r = 0 then 0 else g ⟨orbitOf l c r, h⟩) else 0

def allTriples : List (Color × Color × Color) :=
  (List.finRange 5).flatMap fun l =>
    (List.finRange 5).flatMap fun c =>
      (List.finRange 5).map fun r => (l, c, r)

/-- **Law 1 (combinatorics), proved by enumeration rather than by plugging constants
    into `k²(k+1)/2`.** There are 125 triples and exactly 75 reflection orbits, and
    `orbitIndex` is a bijection from canonical triples onto `{0, …, 74}`. -/
theorem triples_count : allTriples.length = 125 := by decide +kernel

theorem orbit_count :
    ((allTriples.map fun (l, c, r) => canon l c r).eraseDups).length = 75 := by decide +kernel

theorem orbitIndex_lt : ∀ t ∈ allTriples, orbitOf t.1 t.2.1 t.2.2 < 75 := by decide +kernel

theorem orbitIndex_surjective :
    ((allTriples.map fun (l, c, r) => orbitOf l c r).eraseDups).length = 75 := by decide +kernel

theorem orbitOf_zero : orbitOf 0 0 0 = 0 := by decide

/-- `ruleOf g` is reflection-symmetric for *every* genome. -/
theorem ruleOf_symmetric (g : Genome) : (ruleOf g).Symmetric := by
  intro l c r
  unfold ruleOf orbitOf
  rw [canon_comm l c r]

/-- `ruleOf g` is quiescent for *every* genome. -/
theorem ruleOf_quiescent (g : Genome) : (ruleOf g).Quiescent := by
  simp [Rule.Quiescent, ruleOf, orbitOf_zero]

/-! ## 3. One developmental step

`win f a b xs y z` applies `f` to every radius-1 window of the row `xs` read
inside the context `a b … y z`.  A step pads the organism with void on both
sides, so `step f row = win f 0 0 row 0 0`.  The output always has exactly
`row.length + 2` cells; trimming the void afterwards gives the organism's
true extent. -/

def win {β : Type} (f : Color → Color → Color → β) :
    Color → Color → Row → Color → Color → List β
  | a, b, [],      y, z => [f a b y, f b y z]
  | a, b, x :: xs, y, z => f a b x :: win f b x xs y z

def step (f : Rule) (row : Row) : Row := win f 0 0 row 0 0

def trimL (row : Row) : Row := row.dropWhile (· == 0)
def trim  (row : Row) : Row := (trimL (trimL row).reverse).reverse

theorem win_length (f : Rule) (xs : Row) :
    ∀ a b y z, (win f a b xs y z).length = xs.length + 2 := by
  induction xs with
  | nil => intros; simp [win]
  | cons x xs ih => intros; simp [win, ih]

/-- Width before trimming grows by exactly two per step. -/
theorem step_length (f : Rule) (row : Row) : (step f row).length = row.length + 2 :=
  win_length f row 0 0 0 0

theorem dropWhile_length_le (p : Color → Bool) (l : Row) : (l.dropWhile p).length ≤ l.length := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    simp only [List.dropWhile]
    split
    · exact Nat.le_succ_of_le ih
    · simp

theorem trim_length_le (row : Row) : (trim row).length ≤ row.length := by
  unfold trim trimL
  rw [List.length_reverse]
  exact Nat.le_trans (dropWhile_length_le _ _)
    (by rw [List.length_reverse]; exact dropWhile_length_le _ _)

/-- **Law 3 (light cone), step form.** Trimmed width grows by at most two cells per
    step (one per side), for *any* rule — this is the causal ceiling `Δw ≤ +1` per
    side that the Python code checks numerically (`lightcone_violations = 0`). -/
theorem lightcone_step (f : Rule) (row : Row) :
    (trim (step f row)).length ≤ row.length + 2 :=
  Nat.le_trans (trim_length_le _) (Nat.le_of_eq (step_length f row))

/-- **Law 3 (quiescence is absorbing).** For a quiescent rule, a void row stays void:
    once a rulioorganism has died it cannot be resurrected. -/
theorem win_void (f : Rule) (hq : f.Quiescent) :
    ∀ n, win f 0 0 (List.replicate n (0 : Color)) 0 0 = List.replicate (n + 2) 0 := by
  have hq' : f 0 0 0 = 0 := hq
  intro n
  induction n with
  | zero => simp [win, hq', List.replicate]
  | succ n ih => simp [List.replicate_succ, win, hq', ih]

theorem trimL_replicate (n : Nat) : trimL (List.replicate n (0 : Color)) = [] := by
  induction n with
  | zero => rfl
  | succ n ih => simp [trimL, List.replicate_succ]

theorem step_void (f : Rule) (hq : f.Quiescent) (n : Nat) :
    trim (step f (List.replicate n 0)) = [] := by
  simp [step, win_void f hq, trim, trimL]

/-! ## 4. Reflection symmetry of development

The reference Python centres every row on its own midpoint when it builds the
space–time grid.  That is only correct if every row is a palindrome.  Here that
is a theorem, not an assumption. -/

theorem win_append (f : Rule) (xs : Row) :
    ∀ a b x y z, win f a b (xs ++ [x]) y z = win f a b xs x y ++ [f x y z] := by
  induction xs with
  | nil => intros; simp [win]
  | cons w ws ih => intros; simp [win, ih]

theorem win_reverse (f : Rule) (hs : f.Symmetric) (xs : Row) :
    ∀ a b y z, (win f a b xs y z).reverse = win f z y xs.reverse b a := by
  induction xs with
  | nil => intro a b y z; simp [win, hs a b y, hs b y z]
  | cons x rest ih =>
    intro a b y z
    simp only [win, List.reverse_cons, List.reverse_cons, ih, win_append]
    rw [hs a b x]

/-- **Law 2 (reflection symmetry).** For a symmetric rule, stepping commutes with
    mirroring the organism. -/
theorem step_reverse (f : Rule) (hs : f.Symmetric) (row : Row) :
    (step f row).reverse = step f row.reverse :=
  win_reverse f hs row 0 0 0 0

/-- Hence a mirror-symmetric organism stays mirror-symmetric. -/
theorem step_palindrome (f : Rule) (hs : f.Symmetric) (row : Row)
    (h : row.reverse = row) : (step f row).reverse = step f row := by
  rw [step_reverse f hs, h]

/-! ## 5. Parts and wholes: exactly two cells per side feel the context

Wolfram's "bulk orchestration" observation — an excised organ develops
differently in isolation than in situ — has a precise local form for `r = 1`:
in one step only the two outermost output cells on each side depend on the
surrounding tissue `a b … y z`; the interior is context-free. -/

theorem win_interior (f : Rule) (x₁ x₂ : Color) (mid : Row) (w₁ w₂ : Color) :
    ∀ a b y z, win f a b (x₁ :: x₂ :: (mid ++ [w₁, w₂])) y z =
      f a b x₁ :: f b x₁ x₂ :: (win f x₁ x₂ mid w₁ w₂ ++ [f w₁ w₂ y, f w₂ y z]) := by
  intro a b y z
  rw [show mid ++ [w₁, w₂] = (mid ++ [w₁]) ++ [w₂] by simp]
  simp only [win, win_append]
  simp

/-! ## 6. Development as iteration of `step` -/

/-- A living organism: its live cells and the absolute position of its leftmost one.
    Keeping the offset makes the space–time diagram exact rather than re-centred. -/
structure Organism where
  offset : Int
  cells  : Row
  deriving Repr

def leadingVoid (row : Row) : Nat := (row.takeWhile (· == 0)).length

def Organism.step (f : Rule) (o : Organism) : Organism :=
  let grown := Ruliobiology.step f o.cells
  ⟨o.offset - 1 + leadingVoid grown, trim grown⟩

def seed (c : Color) : Organism := ⟨0, [c]⟩

/-- The live stages of development from `o`, at most `cap` steps, stopping at death. -/
def develop (f : Rule) : Nat → Organism → List Organism
  | _,     ⟨_, []⟩ => []
  | 0,     o       => [o]
  | n + 1, o       => o :: develop f n (o.step f)

/-- Wolfram's lifetime: the number of live stages, or `none` if still alive at `cap`
    (unbounded / periodic "tumour", fitness 0 in the Python reference). -/
def lifetime (f : Rule) (cap : Nat) (c : Color) : Option Nat :=
  let d := develop f cap (seed c)
  if d.length ≤ cap then some d.length else none

def fitness (f : Rule) (cap : Nat) (c : Color) : Nat := (lifetime f cap c).getD 0

theorem develop_length_le (f : Rule) : ∀ n o, (develop f n o).length ≤ n + 1 := by
  intro n
  induction n with
  | zero => intro ⟨off, cells⟩; cases cells <;> simp [develop]
  | succ n ih =>
    intro ⟨off, cells⟩
    cases cells with
    | nil => simp [develop]
    | cons x xs => simp [develop]; exact ih _

theorem organism_step_width (f : Rule) (o : Organism) :
    (o.step f).cells.length ≤ o.cells.length + 2 :=
  lightcone_step f o.cells

/-- **Law 3 (light cone), trajectory form.** Every stage reachable in `n` steps from `o`
    is at most `2n` cells wider than `o`; from a single seed, `W(t) ≤ 2t + 1`. -/
theorem lightcone (f : Rule) : ∀ n o, ∀ p ∈ develop f n o,
    p.cells.length ≤ o.cells.length + 2 * n := by
  intro n
  induction n with
  | zero =>
    intro ⟨off, cells⟩ p hp
    cases cells <;> simp [develop] at hp
    subst hp; simp
  | succ n ih =>
    intro ⟨off, cells⟩ p hp
    cases cells with
    | nil => simp [develop] at hp
    | cons x xs =>
      simp only [develop, List.mem_cons] at hp
      rcases hp with rfl | hp
      · simp
      · have h₁ := ih _ p hp
        have h₂ := organism_step_width f ⟨off, x :: xs⟩
        simp at h₂ ⊢; omega

theorem lightcone_seed (f : Rule) (n : Nat) (c : Color) :
    ∀ p ∈ develop f n (seed c), p.cells.length ≤ 2 * n + 1 := by
  intro p hp
  have := lightcone f n (seed c) p hp
  simp [seed] at this; omega

/-! ## 7. Coding and non-coding genome: development only reads the windows it visits

`windowsOf row` lists the neighbourhood triples a step actually consults.  Two
rules that agree on those triples take the same step; two genomes that agree
on the orbits *used* during development produce identical organisms.  This is
the theorem the Python code relies on when it skips re-simulation for a
mutation outside `used_orbits`. -/

def windowsOf (row : Row) : List (Color × Color × Color) :=
  win (fun l c r => (l, c, r)) 0 0 row 0 0

theorem win_map (f : Rule) (xs : Row) : ∀ a b y z,
    win f a b xs y z = (win (fun l c r => (l, c, r)) a b xs y z).map fun (l, c, r) => f l c r := by
  induction xs with
  | nil => intros; simp [win]
  | cons x xs ih => intros; simp [win, ih]

theorem step_congr (f g : Rule) (row : Row)
    (h : ∀ t ∈ windowsOf row, f t.1 t.2.1 t.2.2 = g t.1 t.2.1 t.2.2) :
    step f row = step g row := by
  simp only [step, win_map f, win_map g]
  exact List.map_congr_left fun ⟨l, c, r⟩ ht => h (l, c, r) ht

/-- Every neighbourhood consulted during `n` steps of development from `o`. -/
def used (f : Rule) (n : Nat) (o : Organism) : List (Color × Color × Color) :=
  (develop f n o).flatMap fun p => windowsOf p.cells

/-- **Law 4 (non-coding neutrality), rule form.** -/
theorem develop_congr (f g : Rule) : ∀ n o,
    (∀ t ∈ used f n o, f t.1 t.2.1 t.2.2 = g t.1 t.2.1 t.2.2) →
    develop f n o = develop g n o := by
  intro n
  induction n with
  | zero => intro ⟨off, cells⟩ _; cases cells <;> simp [develop]
  | succ n ih =>
    intro ⟨off, cells⟩ h
    cases cells with
    | nil => simp [develop]
    | cons x xs =>
      have hstep : Organism.step f ⟨off, x :: xs⟩ = Organism.step g ⟨off, x :: xs⟩ := by
        simp only [Organism.step]
        rw [step_congr f g (x :: xs) fun t ht => h t (by
          simp only [used, develop, List.flatMap_cons, List.mem_append]; exact Or.inl ht)]
      simp only [develop, hstep, List.cons.injEq, true_and]
      apply ih
      intro t ht
      apply h
      simp only [used, develop, List.flatMap_cons, List.mem_append]
      rw [hstep]
      exact Or.inr ht

/-- **Law 4 (non-coding neutrality), genome form.** A point mutation on an orbit that
    development never reads leaves every stage of development unchanged. -/
theorem develop_genome_congr (g g' : Genome) (n : Nat) (o : Organism)
    (h : ∀ t ∈ used (ruleOf g) n o, ∀ hlt : orbitOf t.1 t.2.1 t.2.2 < 75,
          g ⟨_, hlt⟩ = g' ⟨_, hlt⟩) :
    develop (ruleOf g) n o = develop (ruleOf g') n o := by
  apply develop_congr
  intro t ht
  unfold ruleOf
  by_cases hlt : orbitOf t.1 t.2.1 t.2.2 < 75
  · simp only [hlt, dite_true, h t ht hlt]
  · simp only [hlt, dite_false]

/-! ## 8. Adaptive evolution (Wolfram 2024): accept a mutation iff it does not shorten life -/

def accept (cur m : Nat) : Bool := m > 0 && m ≥ cur

def ratchet (cur m : Nat) : Nat := if accept cur m then m else cur

/-- **Law 5 (monotone fitness ratchet).** Lifetime fitness never decreases along an
    adaptive lineage, so the fitness trace is a sum of non-negative breakthroughs. -/
theorem ratchet_ge (cur m : Nat) : ratchet cur m ≥ cur := by
  unfold ratchet accept
  by_cases h : m ≥ cur
  · by_cases h0 : m > 0 <;> simp [h, h0]
  · simp [h]

theorem ratchet_telescopes (f₀ m₁ m₂ : Nat) :
    f₀ + (ratchet f₀ m₁ - f₀) + (ratchet (ratchet f₀ m₁) m₂ - ratchet f₀ m₁)
      = ratchet (ratchet f₀ m₁) m₂ := by
  have h₁ := ratchet_ge f₀ m₁
  have h₂ := ratchet_ge (ratchet f₀ m₁) m₂
  omega

/-! ## 9. Executable surface: genomes as 75-digit strings, JSON out -/

def Color.ofNat (n : Nat) : Color := ⟨n % 5, Nat.mod_lt _ (by decide)⟩

def genomeOfArray (a : Array Color) : Genome := fun i => a.getD i.val 0

def parseGenome (s : String) : Option (Array Color) :=
  if s.length = 75 && s.all (fun ch => '0' ≤ ch && ch ≤ '4') then
    some (s.toList.map fun ch => Color.ofNat (ch.toNat - '0'.toNat)).toArray
  else none

def Row.digits (row : Row) : String :=
  String.ofList (row.map fun c => Char.ofNat (c.val + '0'.toNat))

def genomeDigits (a : Array Color) : String := Row.digits a.toList

def excise (o : Organism) (lo hi : Nat) : Organism :=
  ⟨o.offset + lo, (o.cells.drop lo).take (hi - lo)⟩

def usedOrbits (f : Rule) (n : Nat) (o : Organism) : List Nat :=
  ((used f n o).map fun (l, c, r) => orbitOf l c r).eraseDups

def stagesJson (stages : List Organism) : String :=
  "[" ++ ",".intercalate (stages.map fun o =>
    s!"\{\"offset\":{o.offset},\"cells\":\"{Row.digits o.cells}\"}") ++ "]"

def developJson (a : Array Color) (cap : Nat) : String :=
  let f := ruleOf (genomeOfArray a)
  let stages := develop f cap (seed 1)
  let terminated : Bool := decide (stages.length ≤ cap)
  let orbits := usedOrbits f cap (seed 1)
  s!"\{\"genome\":\"{genomeDigits a}\",\"cap\":{cap},\"terminated\":{terminated}," ++
  s!"\"lifetime\":{if terminated then toString stages.length else "null"}," ++
  s!"\"fitness\":{if terminated then stages.length else 0}," ++
  s!"\"max_width\":{stages.foldl (fun m o => max m o.cells.length) 0}," ++
  s!"\"used_orbits\":{orbits}," ++
  s!"\"stages\":{stagesJson stages}}"

/-- In-situ window vs. isolated excision of the same cells, `k` steps later. -/
def exciseJson (a : Array Color) (cap t₀ lo hi k : Nat) : String :=
  let f := ruleOf (genomeOfArray a)
  let stages := develop f cap (seed 1)
  match stages[t₀]? with
  | none => "{\"error\":\"t0 beyond lifetime\"}"
  | some o =>
    let organ := excise o lo hi
    let isolated := develop f k organ
    let inSitu := (develop f k o).map fun p =>
      -- the same absolute positions, read from the whole organism
      let a := (organ.offset - p.offset).toNat
      excise p a (a + (hi - lo))
    s!"\{\"genome\":\"{genomeDigits a}\",\"t0\":{t₀},\"lo\":{lo},\"hi\":{hi},\"k\":{k}," ++
    s!"\"isolated\":{stagesJson isolated},\"in_situ\":{stagesJson inSitu}}"

/-- SplitMix64, so lineages are reproducible without any library. -/
def splitmix (s : UInt64) : UInt64 × UInt64 :=
  let s := s + 0x9E3779B97F4A7C15
  let z := s
  let z := (z ^^^ (z >>> 30)) * 0xBF58476D1CE4E5B9
  let z := (z ^^^ (z >>> 27)) * 0x94D049BB133111EB
  (z ^^^ (z >>> 31), s)

structure Lineage where
  genome   : Array Color
  fit      : Nat
  trace    : Array Nat
  breaks   : Array (Nat × Nat × String)   -- (step, fitness, genome)

def evolve (rngSeed : UInt64) (nMut cap : Nat) : Lineage := Id.run do
  let mut rng := rngSeed
  let mut g : Array Color := Array.replicate 75 0
  let mut fit := fitness (ruleOf (genomeOfArray g)) cap 1          -- = 1 for the null rule
  let mut coding := usedOrbits (ruleOf (genomeOfArray g)) cap (seed 1)
  let mut trace : Array Nat := #[]
  let mut breaks : Array (Nat × Nat × String) := #[(0, fit, genomeDigits g)]
  for i in [0:nMut] do
    let (r₁, s₁) := splitmix rng
    let (r₂, s₂) := splitmix s₁
    rng := s₂
    let orbit := 1 + (r₁ % 74).toNat
    let delta := 1 + (r₂ % 4).toNat
    let old := g.getD orbit 0
    let g' := g.set! orbit (Color.ofNat (old.val + delta))
    if coding.contains orbit then
      let f' := ruleOf (genomeOfArray g')
      let mfit := fitness f' cap 1
      if accept fit mfit then
        if mfit > fit then breaks := breaks.push (i + 1, mfit, genomeDigits g')
        g := g'; fit := mfit
        coding := usedOrbits f' cap (seed 1)
    else
      g := g'                                   -- neutral drift: Law 4 says nothing changes
    trace := trace.push fit
  return ⟨g, fit, trace, breaks⟩

def lineageJson (L : Lineage) : String :=
  s!"\{\"genome\":\"{genomeDigits L.genome}\",\"fitness\":{L.fit}," ++
  s!"\"trace\":{L.trace.toList}," ++
  "\"breakthroughs\":[" ++ ",".intercalate (L.breaks.toList.map fun (i, f, g) =>
    s!"\{\"step\":{i},\"fitness\":{f},\"genome\":\"{g}\"}") ++ "]}"

end Ruliobiology

open Ruliobiology in
def main (args : List String) : IO UInt32 := do
  let usage := "usage: develop <genome75> <cap> | excise <genome75> <cap> <t0> <lo> <hi> <k> | evolve <seed> <nMut> <cap>"
  match args with
  | ["develop", gs, cap] =>
    match parseGenome gs, cap.toNat? with
    | some g, some cap => IO.println (developJson g cap); return 0
    | _, _ => IO.eprintln usage; return 2
  | ["excise", gs, cap, t₀, lo, hi, k] =>
    match parseGenome gs, cap.toNat?, t₀.toNat?, lo.toNat?, hi.toNat?, k.toNat? with
    | some g, some cap, some t₀, some lo, some hi, some k =>
      IO.println (exciseJson g cap t₀ lo hi k); return 0
    | _, _, _, _, _, _ => IO.eprintln usage; return 2
  | ["evolve", s, n, cap] =>
    match s.toNat?, n.toNat?, cap.toNat? with
    | some s, some n, some cap => IO.println (lineageJson (evolve s.toUInt64 n cap)); return 0
    | _, _, _ => IO.eprintln usage; return 2
  | _ => IO.eprintln usage; return 2
