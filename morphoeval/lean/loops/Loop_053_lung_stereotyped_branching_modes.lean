/-
  morphoeval/lean/loops/Loop_053_lung_stereotyped_branching_modes.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 053:
  Metzger et al. (2008) Nature & Menshykau et al. (2012, 2014) Stereotyped
  Mouse Lung Branching Subroutines (Domain D, Planar P, Orthogonal O) and
  FGF10-SHH-Ptch1 Receptor-Ligand Turing Mode Selection
  (`metzger-lung-stereotyped-branching`).

  Covers 3 governing mathematical laws in pure core Lean 4 (zero external imports,
  zero `sorry`s):
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

end MathOfBiology.Loop_053
