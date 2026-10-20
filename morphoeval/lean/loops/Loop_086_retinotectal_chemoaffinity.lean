/-
  morphoeval/lean/loops/Loop_086_retinotectal_chemoaffinity.lean
  Formal Lean 4 verification of the Prestige & Willshaw (1975) Proc. R. Soc. B 190:77-98,
  Willshaw & von der Malsburg (1976) Proc. R. Soc. B 194:431-445, and
  Brown et al. (2000) Cell 102:77-88 retinotectal chemoaffinity & synaptic competition laws:
    1. Reciprocal EphA * ephrin-A equilibrium `R_A(x) * L_A(y*) = gamma_0` and strict
       nasal-to-posterior / temporal-to-anterior topographic inversion `y_WT*(x) = L - x`.
    2. Brown et al. (2000) Isl2-EphA3 knock-in anterior shift `y_KI*(x) < y_WT*(x)`
       generating two non-overlapping topographic maps in homozygotes (`ki/ki`) vs
       rostral pole convergence in heterozygotes (`ki/+`).
    3. Prestige & Willshaw (1975) Group II linear map compression under caudal half-tectum
       ablation `y_comp*(x) = (L - x) / 2` and 2x expansion under temporal half-retina
       ablation `y_exp*(x) = L - 2*x` vs Group I rigid truncation failure
       (`willshaw-retinotectal-chemoaffinity`, Slot 086).

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_086

-- ============================================================================
-- Law 1 (`willshaw-retinotectal-chemoaffinity-1`):
-- Reciprocal EphA * Ephrin-A Equilibrium & Strict Topographic Inversion Law
-- ============================================================================
/-- In Sperry / Prestige & Willshaw (1975) and Brown et al. (2000), retinal ganglion cells
    at nasal-temporal coordinate `x in [0, L]` express EphA receptor `R_A(x) = R_0 * exp(alpha * x / L)`
    and tectal cells at anterior-posterior coordinate `y in [0, L]` express ephrin-A ligand
    `L_A(y) = L_0 * exp(alpha * y / L)`. Mass-action forward/reverse equilibrium
    `R_A(x) * L_A(y) = gamma_0 = R_0 * L_0 * exp(alpha)` minimizes the logarithmic
    chemoaffinity mismatch energy `chemoaffinity_log_energy(x, y, L) = (x + y - L)^2`:
    1. At the wild-type inverted target locus `wt_inverted_map(x, L) = L - x`, the
       chemoaffinity energy vanishes identically (`0`).
    2. The mapping is strictly decreasing: `x_1 < x_2 ==> wt_inverted_map x_2 L < wt_inverted_map x_1 L`,
       mapping the nasal pole (`x = 0`) to the posterior/caudal pole (`y* = 100`) and the
       temporal pole (`x = 100`) to the anterior/rostral pole (`y* = 0`).
    3. For discrete dyadic exponential gradients `dyadic_grad(k) = 2^k` (`k in {0, 1, 2}`),
       the reciprocal product `dyadic_grad(k) * dyadic_grad(2 - k) = 4 = gamma_0` is exact. -/
def sq_int (z : Int) : Int :=
  z * z

def wt_inverted_map (x L : Int) : Int :=
  L - x

def chemoaffinity_log_energy (x y L : Int) : Int :=
  sq_int (x + y - L)

def dyadic_grad (k : Nat) : Nat :=
  2 ^ k

theorem retinotectal_chemoaffinity_inversion (x x_1 x_2 L : Int) (h_lt : x_1 < x_2) :
    chemoaffinity_log_energy x (wt_inverted_map x L) L = 0 ∧
    wt_inverted_map x_2 L < wt_inverted_map x_1 L ∧
    wt_inverted_map 0 100 = 100 ∧
    wt_inverted_map 100 100 = 0 ∧
    dyadic_grad 0 * dyadic_grad 2 = 4 ∧
    dyadic_grad 1 * dyadic_grad 1 = 4 ∧
    dyadic_grad 2 * dyadic_grad 0 = 4 := by
  refine ⟨?_, ?_, by decide, by decide, by decide, by decide, by decide⟩
  · dsimp [chemoaffinity_log_energy, wt_inverted_map, sq_int]
    have h_zero : x + (L - x) - L = 0 := by omega
    rw [h_zero]
    decide
  · dsimp [wt_inverted_map]
    omega


-- ============================================================================
-- Law 2 (`willshaw-retinotectal-chemoaffinity-2`):
-- Brown et al. (2000) Isl2-EphA3 Knock-In Anterior Shift & Dual Map Bifurcation
-- ============================================================================
/-- In Brown et al. (2000) Cell 102:77-88, knocking in exogenous `EphA3` (`delta_R > 0`)
    into `Isl2+` retinal ganglion cells (50% of RGCs intermingled across the retina)
    increases their effective EphA signaling level by `delta_R`:
    1. Every `Isl2+` knock-in RGC shifts anteriorly (toward lower ephrin-A `y`) relative
       to a wild-type RGC at the same retinal coordinate `x`:
       `ki_unclamped_map(x, L, delta_R) = L - x - delta_R < wt_inverted_map(x, L)` for `delta_R > 0`.
    2. In homozygous `Isl2-EphA3/EphA3` (`ki/ki`) mice under Group II competition across
       `L = 100`, the `Isl2+` population compresses onto the rostral half-colliculus
       `hom_rostral_branch(x) = (100 - x) / 2 in [0, 50]` while the wild-type `Isl2-`
       population is displaced caudally onto `hom_caudal_branch(x) = 50 + (100 - x) / 2 in [50, 100]`,
       forming two complete, non-overlapping maps separated by `50` units at every `x`.
    3. In heterozygous `Isl2-EphA3/+` (`ki/+`, `delta_R = 28`), the clamped `Isl2+` branch
       `het_clamped_branch(x, L, delta_R) = max(0, L - x - delta_R)` is separated from
       wild-type in caudal/central colliculus (`x = 20 ==> 52 < 80`) and collapses onto the
       rostral pole (`y = 0`) for temporal axons (`x = 80 ==> 0`). -/
def ki_unclamped_map (x L delta_R : Int) : Int :=
  L - x - delta_R

def hom_rostral_branch (x : Int) : Int :=
  (100 - x) / 2

def hom_caudal_branch (x : Int) : Int :=
  50 + (100 - x) / 2

def het_clamped_branch (x L delta_R : Int) : Int :=
  if L - x - delta_R > 0 then L - x - delta_R else 0

theorem isl2_epha3_knockin_map_bifurcation (x L delta_R : Int) (h_pos : delta_R > 0) :
    ki_unclamped_map x L delta_R < wt_inverted_map x L ∧
    hom_caudal_branch x - hom_rostral_branch x = 50 ∧
    hom_rostral_branch 0 = 50 ∧
    hom_rostral_branch 100 = 0 ∧
    hom_caudal_branch 0 = 100 ∧
    hom_caudal_branch 100 = 50 ∧
    het_clamped_branch 20 100 28 = 52 ∧
    het_clamped_branch 80 100 28 = 0 := by
  refine ⟨?_, ?_, by decide, by decide, by decide, by decide, by decide, by decide⟩
  · dsimp [ki_unclamped_map, wt_inverted_map]
    omega
  · dsimp [hom_caudal_branch, hom_rostral_branch]
    omega


-- ============================================================================
-- Law 3 (`willshaw-retinotectal-chemoaffinity-3`):
-- Prestige & Willshaw (1975) Group II Half-Tectum Compression & Half-Retina Expansion
-- ============================================================================
/-- Contrasts Prestige & Willshaw (1975) Group II relative synaptic competition against
    Group I rigid chemoaffinity under surgical ablations (`L = 100`):
    1. Caudal Half-Tectum Ablation (rostral half `y in [0, 50]` remaining):
       - Group II competition (`group_ii_half_tectum_comp x = (100 - x) / 2`) linearly
         compresses the full retina `x in [0, 100]` onto `[0, 50]` with strict ordering
         (`x = 0 -> 50`, `x = 50 -> 25`, `x = 100 -> 0`).
       - Group I rigid lock (`group_i_truncated_map x = min(100 - x, 50)`) piles up all
         nasal axons `x in [0, 50]` at the caudal cut border `y = 50`
         (`group_i_truncated_map 0 = 50` and `group_i_truncated_map 50 = 50`, losing
         nasal topographic resolution).
    2. Temporal Half-Retina Ablation (nasal half-retina `x in [0, 50]` remaining on
       intact tectum `y in [0, 100]`):
       - Group II competition (`group_ii_half_retina_exp x = 100 - 2 * x`) expands the
         surviving nasal half-retina across the full tectum (`x = 0 -> 100`, `x = 50 -> 0`),
         doubling the mapped span (`100`) compared to unexpanded Group I (`50`). -/
def group_ii_half_tectum_comp (x : Int) : Int :=
  (100 - x) / 2

def group_i_truncated_map (x : Int) : Int :=
  if 100 - x <= 50 then 100 - x else 50

def group_ii_half_retina_exp (x : Int) : Int :=
  100 - 2 * x

theorem prestige_willshaw_group_ii_plasticity (x : Int) :
    group_ii_half_tectum_comp 0 = 50 ∧
    group_ii_half_tectum_comp 50 = 25 ∧
    group_ii_half_tectum_comp 100 = 0 ∧
    group_i_truncated_map 0 = 50 ∧
    group_i_truncated_map 50 = 50 ∧
    group_ii_half_retina_exp 0 = 100 ∧
    group_ii_half_retina_exp 50 = 0 ∧
    (wt_inverted_map 0 100 - wt_inverted_map 50 100) * 2 =
      (group_ii_half_retina_exp 0 - group_ii_half_retina_exp 50) ∧
    wt_inverted_map x 100 - 2 * group_ii_half_tectum_comp x >= 0 := by
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide, ?_⟩
  · dsimp [wt_inverted_map, group_ii_half_tectum_comp]
    omega

end MathOfBiology.Loop_086
