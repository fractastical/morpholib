/-
  morphoeval/lean/loops/Loop_053_lung_stereotyped_branching_modes.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 053:
  Metzger et al. (2008) Nature & Menshykau et al. (2012, 2014) Stereotyped
  Mouse Lung Branching Subroutines (Domain D, Planar P, Orthogonal O) and
  FGF10-SHH-Ptch1 Receptor-Ligand Turing Mode Selection
  (`metzger-lung-stereotyped-branching`).

  Covers 5 governing mathematical laws in pure core Lean 4 (zero external imports,
  zero `sorry`s; Laws 4-5 added in Pass 3):
    1. Metzger (2008) Domain & Bifurcation Tip-Multiplicity Composition Law:
       Domain branching `D_m` sprouts `m` lateral daughter branches from a continuing
       parent trunk (`m + 1` active tips), and each subsequent Planar (`P`) or
       Orthogonal (`O`) bifurcation doubles the distal tip count (`n -> 2 n`), so two
       rounds of bifurcation yield `4 n` tips and a `D_m -> (P/O)^2` lobar scaffold
       yields `4 m + 1` distal tips.
    2. Metzger (2008) Planar Invariance vs Orthogonal Dihedral Rotation Periodicity:
       Planar bifurcation (`Δφ = 0°`) leaves the division plane orientation `φ`
       strictly invariant across generations, whereas Orthogonal bifurcation
       (`Δφ = 90°`) alternates between perpendicular planes and advances by exactly
       `180°` every two generations (`φ_{g+2} - φ_g = 180`), restoring the unoriented
       cleavage plane with period 2.
    3. Menshykau et al. (2012, 2014) FGF10-SHH-Ptch1 Receptor-Ligand Flux Cancellation
       & Homogeneous Steady-State Law:
       For the Schnakenberg-type receptor-ligand reaction rates
       `f = a_R - mu_R_R + r2_F` and `g = a_F - r2_F` (where `r2_F = R^2 F` and
       `mu_R_R = μ_R R`), the cubic signaling term `R^2 F` cancels in the sum
       `f + g = a_R + a_F - mu_R_R`, and at the steady state `mu_R_R = a_R + a_F`,
       `r2_F = a_F`, both reaction residuals vanish (`f = 0 ∧ g = 0`).
    4. (Pass 3) Pleura-Confined FGF10 Supply Anisotropy Mode Selection:
       the selector `if fP > fO + c then 0 else 90` only returns the Metzger dihedrals
       `{0°, 90°}`, yields Orthogonal for an isotropic supply (`fP = fO`, `c ≥ 0`) and
       Planar when the orthogonal candidates lie beyond the pleura (`fO = 0`, `fP > c`).
    5. (Pass 3) Mean-Field Septation Closed Form:
       `N_{k+1} = mu * N_k`, `N_0 = n0` has `N_k = n0 * mu ^ k` (induction), and a
       `D_m -> (P/O)^2` scaffold followed by `k` binary septation rounds carries
       `(4 m + 1) * 2 ^ k` alveoli.

  Verify with the Lean 4 kernel:
    lean morphoeval/lean/loops/Loop_053_lung_stereotyped_branching_modes.lean
-/

namespace MathOfBiology.Loop_053

-- ============================================================================
-- Law 1 (`metzger-lung-stereotyped-branching-1`):
-- Domain Branching (D) & Bifurcation (P/O) Tip-Multiplicity Composition Law
-- ============================================================================

/-- A single Planar (`P`) or Orthogonal (`O`) bifurcation doubles active tips `n -> 2 * n`. -/
def bifurcate_tips (n : Nat) : Nat :=
  2 * n

/-- A Domain Branching (`D_m`) trunk with `m` lateral domain buds each undergoing
    two generations of `P`/`O` bifurcation plus 1 continuing main trunk tip. -/
def domain_composite_tips (m : Nat) : Nat :=
  bifurcate_tips (bifurcate_tips m) + 1

/-- Theorem 1 (`metzger_subroutine_tip_multiplicity`):
    Two consecutive `P` or `O` bifurcations quadruple any tip count `n` (`4 * n`),
    and a domain-branching scaffold with `m` lateral branches followed by two
    bifurcation rounds produces exactly `4 * m + 1` distal tips. -/
theorem metzger_subroutine_tip_multiplicity (n m : Nat) :
    bifurcate_tips (bifurcate_tips n) = 4 * n ∧
    domain_composite_tips m = 4 * m + 1 := by
  simp [bifurcate_tips, domain_composite_tips]
  omega


-- ============================================================================
-- Law 2 (`metzger-lung-stereotyped-branching-2`):
-- Planar (Δφ = 0°) Invariance vs Orthogonal (Δφ = 90°) Dihedral Rotation Law
-- ============================================================================

/-- Planar bifurcation (`P`) preserves division-plane dihedral angle `φ` (`Δφ = 0°`). -/
def planar_step_deg (phi : Int) : Int :=
  phi + 0

/-- Orthogonal bifurcation (`O`) rotates the division plane by `Δφ = 90°`. -/
def orthogonal_step_deg (phi : Int) : Int :=
  phi + 90

/-- Theorem 2 (`metzger_planar_orthogonal_dihedral_periodicity`):
    (1) Any two consecutive Planar bifurcations preserve the initial plane `φ` (`Δφ = 0°`), and
    (2) Two consecutive Orthogonal bifurcations advance the dihedral angle by exactly `180°`
        (`orthogonal_step_deg (orthogonal_step_deg phi) - phi = 180`), completing a full half-turn
        that restores the unoriented division plane while spanning both perpendicular axes. -/
theorem metzger_planar_orthogonal_dihedral_periodicity (phi : Int) :
    planar_step_deg (planar_step_deg phi) = phi ∧
    orthogonal_step_deg (orthogonal_step_deg phi) - phi = 180 := by
  simp [planar_step_deg, orthogonal_step_deg]
  omega


-- ============================================================================
-- Law 3 (`metzger-lung-stereotyped-branching-3`):
-- Menshykau et al. (2012, 2014) FGF10-SHH-Ptch1 Receptor-Ligand Steady State
-- ============================================================================

/-- Receptor reaction rate `f = a_R - mu_R_R + r2_F` where `mu_R_R = μ_R R` and `r2_F = R^2 F`. -/
def receptor_reaction (a_R mu_R_R r2_F : Int) : Int :=
  a_R - mu_R_R + r2_F

/-- Ligand (FGF10) reaction rate `g = a_F - r2_F` where `r2_F = R^2 F`. -/
def ligand_reaction (a_F r2_F : Int) : Int :=
  a_F - r2_F

/-- Theorem 3 (`menshykau_fgf10_shh_receptor_ligand_steady_state`):
    (1) Summing receptor and ligand reaction rates cancels the cubic signaling flux `r2_F = R^2 F`
        identically: `f + g = a_R + a_F - mu_R_R`.
    (2) At the homogeneous steady state `mu_R_R = a_R + a_F` and `r2_F = a_F`, both reaction
        rates vanish identically (`f = 0 ∧ g = 0`). -/
theorem menshykau_fgf10_shh_receptor_ligand_steady_state (a_R a_F mu_R_R r2_F : Int) :
    receptor_reaction a_R mu_R_R r2_F + ligand_reaction a_F r2_F = a_R + a_F - mu_R_R ∧
    receptor_reaction a_R (a_R + a_F) a_F = 0 ∧
    ligand_reaction a_F a_F = 0 := by
  simp [receptor_reaction, ligand_reaction]
  omega


-- ============================================================================
-- Law 4 (`metzger-lung-stereotyped-branching-4`, Pass 3):
-- Pleura-Confined FGF10 Supply Anisotropy Mode Selection
-- ============================================================================

/-- Pass 3 pleura-confined mode selector (Section 3b of the script): given the FGF10
    supply `fP` at the planar candidate daughter tips, the supply `fO` at the orthogonal
    candidates and the anisotropy threshold `c` (all scaled to integers), Planar
    (`Δφ = 0°`) is selected iff `fP > fO + c`, otherwise Orthogonal (`Δφ = 90°`). -/
def pleura_mode_select (fP fO c : Int) : Int :=
  if fP > fO + c then 0 else 90

/-- Theorem 4 (`pleura_anisotropy_mode_selection`):
    (1) The selector only ever returns the two Metzger dihedral values `0°` or `90°`;
    (2) an isotropic FGF10 supply (deep lobe interior, `fP = fO`) with a non-negative
        threshold always yields Orthogonal bifurcation (`90°`);
    (3) if the orthogonal candidates lie beyond the pleura (`fO = 0`) and the planar
        supply exceeds the threshold (`fP > c`), the selector flattens the lineage to
        Planar (`0°`) — pleura-constrained lobe flattening. -/
theorem pleura_anisotropy_mode_selection (fP fO c : Int) :
    (pleura_mode_select fP fO c = 0 ∨ pleura_mode_select fP fO c = 90) ∧
    (0 ≤ c → pleura_mode_select fP fP c = 90) ∧
    (fP > c → pleura_mode_select fP 0 c = 0) := by
  refine ⟨?_, ?_, ?_⟩
  · unfold pleura_mode_select
    split <;> simp
  · intro hc
    unfold pleura_mode_select
    split
    · omega
    · rfl
  · intro h
    unfold pleura_mode_select
    split
    · rfl
    · omega


-- ============================================================================
-- Law 5 (`metzger-lung-stereotyped-branching-5`, Pass 3):
-- Mean-Field Saccular-to-Alveolar Septation Closed Form
-- ============================================================================

/-- Mean-field septation recursion (Section 4b): starting from `n0` terminal units at the
    end of branching, each septation round multiplies the expected alveolar count by the
    mean offspring number `mu` (`mu = 2` for a binary secondary septum). -/
def septation_mean (n0 mu : Nat) : Nat → Nat
  | 0 => n0
  | k + 1 => mu * septation_mean n0 mu k

/-- Theorem 5 (`septation_mean_closed_form`):
    (1) `N_k = n0 * mu ^ k` for every number of septation rounds `k` (proved by induction);
    (2) composed with Law 1, a `D_m -> (P/O)^2` lobar scaffold followed by `k` binary
        septation rounds carries exactly `(4 m + 1) * 2 ^ k` alveoli. -/
theorem septation_mean_closed_form (n0 mu k m : Nat) :
    septation_mean n0 mu k = n0 * mu ^ k ∧
    septation_mean (domain_composite_tips m) 2 k = (4 * m + 1) * 2 ^ k := by
  have h : ∀ (a b j : Nat), septation_mean a b j = a * b ^ j := by
    intro a b j
    induction j with
    | zero => simp [septation_mean]
    | succ j ih =>
      simp [septation_mean, ih, Nat.pow_succ]
      rw [Nat.mul_comm b, Nat.mul_assoc]
  refine ⟨h n0 mu k, ?_⟩
  rw [h, (metzger_subroutine_tip_multiplicity 0 m).2]

end MathOfBiology.Loop_053
