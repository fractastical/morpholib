/-
  morphoeval/lean/loops/Loop_054_kidney_ureteric_bud_gdnf_ret.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 054:
  Menshykau & Iber (2013) Phys. Biol., Zubkov et al. (2015) Bull. Math. Biol. &
  Watanabe & Costantini (2004) Dev. Biol. Kidney Ureteric Bud GDNF-RET-Wnt11
  Ligand-Receptor Turing Branching Mechanism (`menshykau-fgf10-shh-lung-turing`).

  Covers 3 governing mathematical laws in pure core Lean 4 (zero external imports,
  zero `sorry`s):
    1. Menshykau & Iber (2013) GDNF-RET Receptor-Ligand Flux Cancellation &
       Homogeneous Steady-State Law:
       For the Schnakenberg-type receptor-ligand reaction rates
       `f(a, R, r2G) = a - R + r2G` and `g(b, r2G) = b - r2G` (where `r2G = R^2 G`),
       the cubic cooperative signaling term `R^2 G` cancels identically in the sum
       `f + g = a + b - R`, and at the steady state `R* = a + b`, `r2G* = b`, both
       reaction rates vanish identically (`f = 0 ∧ g = 0`).
    2. Menshykau & Iber (2013) Steady-State Jacobian Positivity & Autocatalytic
       RET Activation Law:
       At `(R*, G*) = (a + b, b / (a + b)^2)`, the scaled Jacobian entries
       `f_R_s = b - a`, `f_G_s = (a + b)^3`, `g_R_s = -2 * b`, `g_G_s = -(a + b)^3`
       satisfy `f_R_s * g_G_s - f_G_s * g_R_s = (a + b)^4` (`det(J_0) = (a + b)^2 > 0`),
       and at wild-type scaled parameters `(a, b) = (1, 9)` (`a = 0.1, b = 0.9`),
       receptor self-activation `f_R_s = 8 > 0`, trace `f_R_s + g_G_s = -992 < 0`,
       and `det_s = 10000 > 0`.
    3. Zubkov et al. (2015) & Watanabe & Costantini (2004) Tip-Splitting Mode
       Selection & Dichotomous vs Hypoplastic Generation Scaling Law:
       The dimensionless tip arc-to-wavelength ratio `rho_10 = 10 * L_tip / lambda_c`
       selects elongation (`m = 1` for `rho_10 < 15`), dichotomous bifurcation
       (       `m = 2` for `15 ≤ rho_10 < 25`), or trichotomous trifurcation (`m = 3` for
       `rho_10 ≥ 25`), and 4 generations of dichotomous branching yield `2^4 = 16`
       ureteric tips vs `4` tips under 2-generation hypoplastic arrest (`4 < 16`).
    4. (Pass 3) Stromal BMP4/7 - Gremlin1 - FGF7 Feedback Gain Monotonicity Law:
       integer-scaled gain num/den with `num = (1 + F)(1 + G)`, `den = (1 + G) + B`
       is non-increasing in BMP, non-decreasing in FGF7, strictly < 1 under Gremlin1
       loss with BMP present, and exactly 1 under full stromal ablation.
    5. (Pass 3) Orthogonal Dihedral Rotation Cascade Law: with short-axis mode
       selection and daughters elongated along the parent T-bar, the daughter
       splitting axis is `(θ + 90) % 180`, the dihedral angle is exactly 90 deg,
       the cascade has period 2, and the orthogonal-daughter ablation gives 0 deg.

  Verify with the Lean 4 kernel:
    lean morphoeval/lean/loops/Loop_054_kidney_ureteric_bud_gdnf_ret.lean
-/

namespace MathOfBiology.Loop_054

-- ============================================================================
-- Law 1 (`menshykau-fgf10-shh-lung-turing-1`):
-- GDNF-RET Receptor-Ligand Flux Cancellation & Homogeneous Steady-State Law
-- ============================================================================

/-- Epithelial RET receptor reaction rate `f = a - R + r2G` where `r2G = R^2 G`. -/
def ret_reaction (a R r2G : Int) : Int :=
  a - R + r2G

/-- Mesenchymal GDNF ligand reaction rate `g = b - r2G` where `r2G = R^2 G`. -/
def gdnf_reaction (b r2G : Int) : Int :=
  b - r2G

/-- Theorem 1 (`menshykau_gdnf_ret_flux_cancellation_and_steady_state`):
    (1) Summing RET receptor and GDNF ligand reaction rates cancels the cubic
        cooperative endocytosis flux `r2G = R^2 G` identically:
        `ret_reaction a R r2G + gdnf_reaction b r2G = a + b - R`.
    (2) At the homogeneous steady state `R* = a + b` and `r2G* = b`, both
        reaction residuals vanish identically (`f = 0 ∧ g = 0`). -/
theorem menshykau_gdnf_ret_flux_cancellation_and_steady_state (a b R r2G : Int) :
    ret_reaction a R r2G + gdnf_reaction b r2G = a + b - R ∧
    ret_reaction a (a + b) b = 0 ∧
    gdnf_reaction b b = 0 := by
  simp [ret_reaction, gdnf_reaction]
  omega


-- ============================================================================
-- Law 2 (`menshykau-fgf10-shh-lung-turing-2`):
-- Steady-State Reaction Jacobian Positivity & Autocatalytic RET Activation Law
-- ============================================================================

/-- Scaled Jacobian entry `(a + b) * f_R = b - a` at `(R*, G*)`. -/
def jac_f_R_scaled (a b : Int) : Int :=
  b - a

/-- Scaled Jacobian entry `(a + b) * f_G = (a + b)^3` at `(R*, G*)`. -/
def jac_f_G_scaled (a b : Int) : Int :=
  (a + b) * (a + b) * (a + b)

/-- Scaled Jacobian entry `(a + b) * g_R = -2 * b` at `(R*, G*)`. -/
def jac_g_R_scaled (b : Int) : Int :=
  -2 * b

/-- Scaled Jacobian entry `(a + b) * g_G = -(a + b)^3` at `(R*, G*)`. -/
def jac_g_G_scaled (a b : Int) : Int :=
  -((a + b) * (a + b) * (a + b))

/-- Scaled Jacobian determinant `(a + b)^2 * det(J_0) = f_R_s * g_G_s - f_G_s * g_R_s`. -/
def jac_det_scaled (a b : Int) : Int :=
  jac_f_R_scaled a b * jac_g_G_scaled a b - jac_f_G_scaled a b * jac_g_R_scaled b

/-- Theorem 2 (`menshykau_gdnf_ret_jacobian_stability_and_turing_prerequisite`):
    At the Menshykau & Iber (2013) wild-type scaled parameters `(a, b) = (1, 9)`
    (representing `a = 0.1, b = 0.9` scaled by 10):
    (1) Receptor self-activation is strictly positive (`jac_f_R_scaled 1 9 = 8 > 0`),
    (2) Homogeneous mode trace is strictly negative (`jac_f_R_scaled 1 9 + jac_g_G_scaled 1 9 = -992 < 0`),
    (3) Scaled Jacobian determinant equals `(1 + 9)^4 = 10000 > 0`. -/
theorem menshykau_gdnf_ret_jacobian_stability_and_turing_prerequisite :
    jac_f_R_scaled 1 9 = 8 ∧
    jac_f_R_scaled 1 9 + jac_g_G_scaled 1 9 = -992 ∧
    jac_det_scaled 1 9 = 10000 := by
  decide


-- ============================================================================
-- Law 3 (`menshykau-fgf10-shh-lung-turing-3`):
-- Tip-Splitting Mode Selection & Dichotomous vs Hypoplastic Generation Scaling
-- ============================================================================

/-- Number of receptor-ligand `R^2 G` maxima on a ureteric bud ampulla as a
    function of `rho_10 = 10 * (L_tip / lambda_c)`:
    `< 15` -> 1 peak (elongation), `15..24` -> 2 peaks (dichotomous bifurcation),
    `≥ 25` -> 3 peaks (trichotomous trifurcation). -/
def tip_splitting_peaks (rho_10 : Nat) : Nat :=
  if rho_10 < 15 then 1
  else if rho_10 < 25 then 2
  else 3

/-- One generation of dichotomous ureteric bud bifurcation doubles active tips `n -> 2 * n`. -/
def dichotomous_step (n : Nat) : Nat :=
  2 * n

/-- One generation of trichotomous ampulla splitting triples active tips `n -> 3 * n`. -/
def trichotomous_step (n : Nat) : Nat :=
  3 * n

/-- Four generations of dichotomous ureteric branching from `n` initial tips. -/
def dichotomous_gen4 (n : Nat) : Nat :=
  dichotomous_step (dichotomous_step (dichotomous_step (dichotomous_step n)))

/-- Theorem 3 (`ureteric_tip_splitting_modes_and_generation_scaling`):
    (1) `tip_splitting_peaks` partitions elongation (`rho_10 = 12 -> 1`), wild-type
        dichotomous bifurcation (`rho_10 = 20 -> 2`), and Spry1_KO trichotomous
        trifurcation (`rho_10 = 30 -> 3`),
    (2) Four generations of dichotomous branching quadruple-doubles any tip count
        (`dichotomous_gen4 n = 16 * n`), yielding `16` tips from a single ureteric bud,
    (3) Hypoplastic 2-generation arrest yields `4` tips (`4 < 16`), whereas one
        trichotomous generation followed by three dichotomous generations yields `24` tips. -/
theorem ureteric_tip_splitting_modes_and_generation_scaling (n : Nat) :
    tip_splitting_peaks 12 = 1 ∧
    tip_splitting_peaks 20 = 2 ∧
    tip_splitting_peaks 30 = 3 ∧
    dichotomous_gen4 n = 16 * n ∧
    dichotomous_step (dichotomous_step 1) < dichotomous_gen4 1 ∧
    dichotomous_step (dichotomous_step (dichotomous_step (trichotomous_step 1))) = 24 := by
  refine ⟨by decide, by decide, by decide, ?_, by decide, by decide⟩
  simp [dichotomous_gen4, dichotomous_step]
  omega


-- ============================================================================
-- Law 4 (`menshykau-fgf10-shh-lung-turing-4`, Pass 3):
-- Stromal BMP4/7 - Gremlin1 - FGF7 Feedback Gain Monotonicity Law
-- gain = (1 + k_F7 FGF7) / (1 + k_B BMP / (1 + k_G GREM1)).
-- In integer-scaled form the gain is the ratio num/den with
--   num = (1 + F) * (1 + G)     (FGF7 potentiation, Gremlin1 sequestration)
--   den = (1 + G) + B           (free BMP inhibition after Gremlin1 binding)
-- so gain ≥ 1 iff num ≥ den. (Python: `stromal_signaling_gain`.)
-- ============================================================================

/-- Numerator of the integer-scaled stromal gain `(1 + F) * (1 + G)`. -/
def stromal_gain_num (F G : Nat) : Nat :=
  (1 + F) * (1 + G)

/-- Denominator of the integer-scaled stromal gain `(1 + G) + B`. -/
def stromal_gain_den (B G : Nat) : Nat :=
  (1 + G) + B

/-- Theorem 4 (`stromal_bmp_gremlin_fgf7_gain_monotonicity`):
    (1) BMP never raises the gain: the denominator is non-decreasing in `B`.
    (2) FGF7 never lowers the gain: the numerator is non-decreasing in `F`.
    (3) Gremlin1 loss (`G = 0`) with any BMP `B > 0` and no FGF7 (`F = 0`) gives
        `num < den` — the gain drops strictly below 1 (Grem1 null outgrowth failure).
    (4) Full stromal ablation (`B = G = F = 0`) gives `num = den` (unit gain). -/
theorem stromal_bmp_gremlin_fgf7_gain_monotonicity (B G F : Nat) :
    stromal_gain_den B G ≤ stromal_gain_den (B + 1) G ∧
    stromal_gain_num F G ≤ stromal_gain_num (F + 1) G ∧
    stromal_gain_num 0 0 < stromal_gain_den (B + 1) 0 ∧
    stromal_gain_num 0 0 = stromal_gain_den 0 0 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [stromal_gain_den]
  · simp [stromal_gain_num]
    exact Nat.mul_le_mul_right _ (by omega)
  · simp [stromal_gain_num, stromal_gain_den]
  · simp [stromal_gain_num, stromal_gain_den]


-- ============================================================================
-- Law 5 (`menshykau-fgf10-shh-lung-turing-5`, Pass 3):
-- Orthogonal Dihedral Rotation Cascade Law (reduced 3D ampulla geometry)
-- On an anisotropic dome in the dichotomous band the two-lobe mode along the
-- SHORT cross-section axis grows fastest; if daughters are elongated along the
-- parent T-bar (long axis ∥ parent splitting axis), each generation's splitting
-- axis is the parent's axis rotated by 90 deg (mod 180), so the dihedral angle
-- between successive splitting planes is exactly 90 deg.
-- (Python: `select_splitting_axis`, `dihedral_rotation_cascade`.)
-- ============================================================================

/-- Splitting-axis angle (deg, mod 180) of the daughter when the short axis wins
    and the daughter long axis is parallel to the parent splitting axis `θ`. -/
def daughter_axis_short_wins (θ : Nat) : Nat :=
  (θ + 90) % 180

/-- Dihedral angle (deg, in [0, 90]) between two axes given mod 180:
    the smaller of the two cyclic differences. -/
def dihedral_deg (α β : Nat) : Nat :=
  min ((α + 180 - β % 180) % 180) ((β + 180 - α % 180) % 180)

/-- Theorem 5 (`orthogonal_dihedral_rotation_cascade`):
    For any parent splitting axis `θ < 180`,
    (1) the daughter axis is orthogonal to the parent (dihedral = 90),
    (2) the granddaughter axis returns to the parent axis (period-2 cascade),
    (3) the orthogonal-daughter ablation (long axis ⊥ parent, short axis = parent
        axis) yields zero dihedral rotation. -/
theorem orthogonal_dihedral_rotation_cascade (θ : Nat) (h : θ < 180) :
    dihedral_deg θ (daughter_axis_short_wins θ) = 90 ∧
    daughter_axis_short_wins (daughter_axis_short_wins θ) = θ ∧
    dihedral_deg θ θ = 0 := by
  refine ⟨?_, ?_, ?_⟩
  · simp only [dihedral_deg, daughter_axis_short_wins]
    omega
  · simp only [daughter_axis_short_wins]
    omega
  · simp only [dihedral_deg]
    omega

end MathOfBiology.Loop_054
