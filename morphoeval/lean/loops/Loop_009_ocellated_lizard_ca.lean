/-
  morphoeval/lean/loops/Loop_009_ocellated_lizard_ca.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 009:
  Manukyan et al. (2017) Nature 544(7649):173-179 — "A living mesoscopic
  cellular automaton made of skin scales" (`manukyan-ocellated-lizard-ca`).

  Pure core Lean 4 (zero external dependencies, zero `sorry`s).
  Verify with:
    lean morphoeval/lean/loops/Loop_009_ocellated_lizard_ca.lean
-/

namespace MathOfBiology.Loop_009

-- ============================================================================
-- Law 1 (`manukyan-ocellated-lizard-ca-1`):
-- Hexagonal Scale First-Nearest-Neighbor Partition & Labyrinthine Symmetry
-- ============================================================================

/-- On a 2D 6-regular hexagonal skin-scale lattice, if an interior scale has
    `n_g ≤ 6` green first-nearest neighbors, its number of black neighbors is
    `n_b = 6 - n_g`. -/
def hex_black_neighbors (n_g : Nat) : Nat :=
  6 - n_g

/-- Theorem 1 (`hex_scale_neighbor_partition_and_symmetry`):
    For any interior hexagonal scale with `n_g ≤ 6` green neighbors:
    (1) Total first-nearest neighbor count is conserved: `n_g + hex_black_neighbors n_g = 6`.
    (2) The labyrinthine chain configuration `n_g = 3` is the unique symmetric balance
        point where green and black neighbor counts are equal (`n_g = hex_black_neighbors n_g`). -/
theorem hex_scale_neighbor_partition_and_symmetry (n_g : Nat) (h : n_g ≤ 6) :
    n_g + hex_black_neighbors n_g = 6 ∧
    (n_g = hex_black_neighbors n_g ↔ n_g = 3) := by
  unfold hex_black_neighbors
  omega


-- ============================================================================
-- Law 2 (`manukyan-ocellated-lizard-ca-2`):
-- Inter-Scale Border Thickness Reduction Quenches Diffusive Flux (`P < 1`)
-- ============================================================================

/-- Scaled inter-scale border diffusive flux `J_border = P_pct * flux_unit`
    across a thinned scale border with percentage thickness factor `P_pct ∈ [0, 100]`
    (where `P_pct = 17` corresponds to `P = 0.17` in *Timon lepidus* and `P_pct = 100`
    corresponds to unattenuated continuum skin `P = 1.0`). -/
def interscale_border_flux (P_pct flux_unit : Int) : Int :=
  P_pct * flux_unit

/-- Theorem 2 (`border_diffusivity_reduction_quenches_interscale_flux`):
    For any positive interior diffusive flux scale `flux_unit ≥ 1` and physiological
    border reduction `1 ≤ P_pct ≤ 99`:
    (1) Complete border insulation (`P_pct = 0`) yields zero inter-scale flux (`J = 0`),
        decoupling adjacent scales.
    (2) Physiological *Timon lepidus* border thinning (`P_pct = 17`) strictly attenuates
        inter-scale diffusive flux (`17 * flux_unit < 100 * flux_unit`) while keeping
        inter-scale coupling positive (`17 * flux_unit > 0`), and any thinned border
        (`P_pct ≤ 99`) satisfies `P_pct < 100`. -/
theorem border_diffusivity_reduction_quenches_interscale_flux
    (P_pct flux_unit : Int)
    (hP_lo : P_pct ≥ 1) (hP_hi : P_pct ≤ 99)
    (hF : flux_unit ≥ 1) :
    interscale_border_flux 0 flux_unit = 0 ∧
    interscale_border_flux 17 flux_unit > 0 ∧
    interscale_border_flux 17 flux_unit < interscale_border_flux 100 flux_unit ∧
    P_pct < 100 := by
  unfold interscale_border_flux
  omega


-- ============================================================================
-- Law 3 (`manukyan-ocellated-lizard-ca-3`):
-- Outer-Totalistic Hexagonal Cellular Automaton Labyrinthine Fixed Point (`n_g = 3`)
-- ============================================================================

/-- Deterministic core of Manukyan et al.'s (2017) hexagonal outer-totalistic
    cellular automaton state-update rule on binary scale color `s ∈ {0 (black), 1 (green)}`
    as a function of green neighbor count `n_g ∈ {0..6}`:
    - A green scale (`s = 1`) remains green (`1`) iff `2 ≤ n_g ≤ 3`, and flips to
      black (`0`) when overcrowded by green neighbors (`n_g ≥ 4`, lateral inhibition)
      or isolated (`n_g ≤ 1`).
    - A black scale (`s = 0`) flips to green (`1`) when surrounded mostly by black
      neighbors (`n_g ≤ 1`, i.e. `n_b ≥ 5`), and remains black (`0`) for `n_g ≥ 2`. -/
def manukyan_ca_step (s n_g : Nat) : Nat :=
  if s = 1 then
    if 2 ≤ n_g ∧ n_g ≤ 3 then 1 else 0
  else
    if n_g ≤ 1 then 1 else 0

/-- Theorem 3 (`manukyan_outer_totalistic_ca_labyrinth_fixed_point`):
    Under the Manukyan hexagonal outer-totalistic CA rule:
    (1) A labyrinthine chain neighborhood (`n_g = 3`, `n_b = 3`) is a simultaneous
        stationary fixed point for both green scales (`manukyan_ca_step 1 3 = 1`)
        and black scales (`manukyan_ca_step 0 3 = 0`).
    (2) Overcrowded green interiors (`s = 1, n_g = 5`) undergo lateral inhibition
        and flip to black (`manukyan_ca_step 1 5 = 0`).
    (3) All-black interiors (`s = 0, n_g = 1, n_b = 5`) undergo activator nucleation
        and flip to green (`manukyan_ca_step 0 1 = 1`). -/
theorem manukyan_outer_totalistic_ca_labyrinth_fixed_point :
    manukyan_ca_step 1 3 = 1 ∧
    manukyan_ca_step 0 3 = 0 ∧
    manukyan_ca_step 1 5 = 0 ∧
    manukyan_ca_step 0 1 = 1 := by
  decide

end MathOfBiology.Loop_009
