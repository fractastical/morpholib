/-
  morphoeval/lean/loops/Loop_076_sponge_spicule_hexactin.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 076:
  Uriz, Turon, Becerro & Agell (2003) Microsc. Res. Tech. 62(4):279-299,
  Aizenberg et al. (2005) Science 309(5732):275-278 &
  Weaver et al. (2007) J. Struct. Biol. 158(1):93-106:
  Silicatein Axial Filament Crystallographic Symmetry (Cubic O_h 6-Ray Hexactins
  at θ = 90° vs Tetrahedral T_d 4-Ray Tetractins/Calthrops at cos θ = -1/3),
  Concentric Lamellar Biosilica Thinning (h_n ∝ r_n⁻¹) & Crack-Deflection
  Toughness, and Euplectella Checkerboard Diagonal Bracing Shear Stiffness
  (`bavestrello-sponge-spicule-hexactin`).

  Zero external dependencies, zero `sorry`s.
  Verify with:
    lean morphoeval/lean/loops/Loop_076_sponge_spicule_hexactin.lean
-/

namespace MathOfBiology.Loop_076

-- ============================================================================
-- Law 1: Cubic O_h 6-Ray Hexactin Orthogonal Vector Equilibrium & Dot Products
-- ============================================================================

/-- A 3D integer lattice vector `(x, y, z) ∈ ℤ³` representing a crystallographic
    silicatein axial filament ray direction. -/
structure Vec3Int where
  x : Int
  y : Int
  z : Int
  deriving DecidableEq, Repr

/-- Vector addition in `ℤ³`. -/
def vec3_add (u v : Vec3Int) : Vec3Int :=
  { x := u.x + v.x, y := u.y + v.y, z := u.z + v.z }

/-- Euclidean dot product `u · v = u_x v_x + u_y v_y + u_z v_z` in `ℤ³`. -/
def vec3_dot (u v : Vec3Int) : Int :=
  u.x * v.x + u.y * v.y + u.z * v.z

/-- Squared Euclidean norm `‖u‖² = u · u` in `ℤ³`. -/
def vec3_norm_sq (u : Vec3Int) : Int :=
  vec3_dot u u

/-- Zero vector `0 = (0, 0, 0) ∈ ℤ³`. -/
def vec3_zero : Vec3Int :=
  { x := 0, y := 0, z := 0 }

/-- The 6 canonical cubic `O_h` hexactin ray vectors `±e_x, ±e_y, ±e_z`
    of Hexactinellida (*Euplectella aspergillum*) indexed by `k ∈ {1..6}`. -/
def hexactin_ray : Nat → Vec3Int
  | 1 => { x :=  1, y :=  0, z :=  0 }
  | 2 => { x := -1, y :=  0, z :=  0 }
  | 3 => { x :=  0, y :=  1, z :=  0 }
  | 4 => { x :=  0, y := -1, z :=  0 }
  | 5 => { x :=  0, y :=  0, z :=  1 }
  | 6 => { x :=  0, y :=  0, z := -1 }
  | _ => vec3_zero

/-- Vector sum of all 6 cubic `O_h` hexactin rays `∑_{k=1}^6 u_k`. -/
def hexactin_vector_sum : Vec3Int :=
  vec3_add (hexactin_ray 1)
    (vec3_add (hexactin_ray 2)
      (vec3_add (hexactin_ray 3)
        (vec3_add (hexactin_ray 4)
          (vec3_add (hexactin_ray 5) (hexactin_ray 6)))))

/-- Theorem 1 (`hexactin_Oh_orthogonal_six_ray_equilibrium_and_dot_products`):
    Proves that for the 6 cubic `O_h` silicatein axial filament rays `±e_x, ±e_y, ±e_z`
    of Hexactinellida hexactins:
    (a) Every ray is a unit vector (`‖u_k‖² = 1` for `k ∈ {1..6}`);
    (b) The 6-ray vector sum vanishes identically (`∑_{k=1}^6 u_k = (0, 0, 0)`),
        establishing mechanical force balance at the central spicule node;
    (c) The 3 antipodal colinear ray pairs `(1,2), (3,4), (5,6)` have dot product `-1`
        (`cos 180° = -1`); and
    (d) Orthogonal adjacent ray pairs across all 3 Cartesian axes have dot product `0`
        (`cos 90° = 0`, `θ = 90°`). -/
theorem hexactin_Oh_orthogonal_six_ray_equilibrium_and_dot_products :
    vec3_norm_sq (hexactin_ray 1) = 1 ∧
    vec3_norm_sq (hexactin_ray 2) = 1 ∧
    vec3_norm_sq (hexactin_ray 3) = 1 ∧
    vec3_norm_sq (hexactin_ray 4) = 1 ∧
    vec3_norm_sq (hexactin_ray 5) = 1 ∧
    vec3_norm_sq (hexactin_ray 6) = 1 ∧
    hexactin_vector_sum = vec3_zero ∧
    vec3_dot (hexactin_ray 1) (hexactin_ray 2) = -1 ∧
    vec3_dot (hexactin_ray 3) (hexactin_ray 4) = -1 ∧
    vec3_dot (hexactin_ray 5) (hexactin_ray 6) = -1 ∧
    vec3_dot (hexactin_ray 1) (hexactin_ray 3) = 0 ∧
    vec3_dot (hexactin_ray 1) (hexactin_ray 4) = 0 ∧
    vec3_dot (hexactin_ray 1) (hexactin_ray 5) = 0 ∧
    vec3_dot (hexactin_ray 1) (hexactin_ray 6) = 0 ∧
    vec3_dot (hexactin_ray 3) (hexactin_ray 5) = 0 ∧
    vec3_dot (hexactin_ray 4) (hexactin_ray 6) = 0 := by
  decide

-- ============================================================================
-- Law 2: Tetrahedral T_d 4-Ray Tetractin / Calthrop Cosine Law (cos θ = -1/3)
-- ============================================================================

/-- The 4 alternating cube-vertex tetrahedral `T_d` ray vectors of Demospongiae /
    Calcarea tetractins (calthrops) in `ℤ³`, each of squared length `‖v_k‖² = 3`. -/
def tetractin_ray : Nat → Vec3Int
  | 1 => { x :=  1, y :=  1, z :=  1 }
  | 2 => { x :=  1, y := -1, z := -1 }
  | 3 => { x := -1, y :=  1, z := -1 }
  | 4 => { x := -1, y := -1, z :=  1 }
  | _ => vec3_zero

/-- Vector sum of all 4 tetrahedral `T_d` tetractin/calthrop rays `∑_{k=1}^4 v_k`. -/
def tetractin_vector_sum : Vec3Int :=
  vec3_add (tetractin_ray 1)
    (vec3_add (tetractin_ray 2)
      (vec3_add (tetractin_ray 3) (tetractin_ray 4)))

/-- Theorem 2 (`tetractin_Td_calthrop_tetrahedral_cosine_and_equilibrium`):
    Proves that for the 4 tetrahedral `T_d` silicatein axial filament rays `v_1..v_4`
    of Demospongiae calthrops/tetractins:
    (a) All 4 rays have identical squared Euclidean norm `‖v_k‖² = 3`;
    (b) The 4-ray vector sum vanishes identically (`∑_{k=1}^4 v_k = (0, 0, 0)`);
    (c) Every one of the `binom(4, 2) = 6` distinct ray pairs `(i < j)` has exact
        dot product `v_i · v_j = -1`; and
    (d) Consequently `3 * (v_i · v_j) + ‖v_i‖² = 0`, establishing the exact
        tetrahedral cosine identity `cos θ = (v_i · v_j) / ‖v_i‖² = -1/3`
        (`θ = arccos(-1/3) ≈ 109.4712°`). -/
theorem tetractin_Td_calthrop_tetrahedral_cosine_and_equilibrium :
    vec3_norm_sq (tetractin_ray 1) = 3 ∧
    vec3_norm_sq (tetractin_ray 2) = 3 ∧
    vec3_norm_sq (tetractin_ray 3) = 3 ∧
    vec3_norm_sq (tetractin_ray 4) = 3 ∧
    tetractin_vector_sum = vec3_zero ∧
    vec3_dot (tetractin_ray 1) (tetractin_ray 2) = -1 ∧
    vec3_dot (tetractin_ray 1) (tetractin_ray 3) = -1 ∧
    vec3_dot (tetractin_ray 1) (tetractin_ray 4) = -1 ∧
    vec3_dot (tetractin_ray 2) (tetractin_ray 3) = -1 ∧
    vec3_dot (tetractin_ray 2) (tetractin_ray 4) = -1 ∧
    vec3_dot (tetractin_ray 3) (tetractin_ray 4) = -1 ∧
    3 * vec3_dot (tetractin_ray 1) (tetractin_ray 2) + vec3_norm_sq (tetractin_ray 1) = 0 := by
  decide

-- ============================================================================
-- Law 3: Concentric Lamellar Biosilica Radial Thinning (h_n ∝ r_n⁻¹) &
--        Crack-Deflection Fracture Toughness Amplification
-- ============================================================================

/-- Radial thickness `h_n = (k_cat * A_ring) / r_n` of the `n`-th concentric
    biosilica cylinder deposited at radius `r_n > 0` under constant per-layer
    cross-sectional silica area `A_ring` and silicatein catalytic activity `k_cat`
    (Weaver et al. 2007 J. Struct. Biol. 158:93-106). -/
def lamellar_layer_thickness (k_cat A_ring r_n : Nat) : Nat :=
  (k_cat * A_ring) / r_n

/-- Effective fracture energy score `G_lam(G_mono, dG_interlayer, N_layers) =
    G_mono + N_layers * dG_interlayer` of a laminated spicule composed of `N_layers`
    concentric silica cylinders separated by organic silicatein interlayers
    (Aizenberg et al. 2005 Science 309:275-278). -/
def laminated_spicule_fracture_energy (G_mono dG_interlayer N_layers : Nat) : Nat :=
  G_mono + N_layers * dG_interlayer

/-- Theorem 3 (`aizenberg_concentric_lamellar_thinning_and_fracture_toughness`):
    Proves that:
    (a) Under silicatein-inhibited ablation (`k_cat = 0`), radial biosilica accretion
        vanishes (`h_n(0, A_ring, r_n) = 0`);
    (b) For active silicatein catalysis (`k_cat = 1, A_ring = 120`), concentric silica
        lamellae thin strictly with increasing radial distance `r_n ∈ {2, 3, 4, 6, 10}`
        (`60 > 40 > 30 > 20 > 12`) while preserving exact ring area `r_n * h_n = 120`; and
    (c) Increasing the number of concentric organic-silica interlayers across
        `N ∈ {0 (monolithic), 6, 12, 25 (Euplectella)}` strictly amplifies spicule
        fracture energy (`100 < 292 < 484 < 900`). -/
theorem aizenberg_concentric_lamellar_thinning_and_fracture_toughness :
    lamellar_layer_thickness 0 120 2 = 0 ∧
    lamellar_layer_thickness 1 120 2 = 60 ∧
    lamellar_layer_thickness 1 120 3 = 40 ∧
    lamellar_layer_thickness 1 120 4 = 30 ∧
    lamellar_layer_thickness 1 120 6 = 20 ∧
    lamellar_layer_thickness 1 120 10 = 12 ∧
    lamellar_layer_thickness 1 120 2 > lamellar_layer_thickness 1 120 3 ∧
    lamellar_layer_thickness 1 120 3 > lamellar_layer_thickness 1 120 4 ∧
    lamellar_layer_thickness 1 120 4 > lamellar_layer_thickness 1 120 6 ∧
    lamellar_layer_thickness 1 120 6 > lamellar_layer_thickness 1 120 10 ∧
    2 * lamellar_layer_thickness 1 120 2 = 120 ∧
    6 * lamellar_layer_thickness 1 120 6 = 120 ∧
    laminated_spicule_fracture_energy 100 32 0 = 100 ∧
    laminated_spicule_fracture_energy 100 32 6 = 292 ∧
    laminated_spicule_fracture_energy 100 32 12 = 484 ∧
    laminated_spicule_fracture_energy 100 32 25 = 900 ∧
    laminated_spicule_fracture_energy 100 32 25 > laminated_spicule_fracture_energy 100 32 12 ∧
    laminated_spicule_fracture_energy 100 32 12 > laminated_spicule_fracture_energy 100 32 0 := by
  decide

-- ============================================================================
-- Law 4: Euplectella Alternating Open/Closed Checkerboard Diagonal Lattice
--        Shear Stiffness Law (Aizenberg et al. 2005 Science 309:275-278)
-- ============================================================================

/-- Effective in-plane shear stiffness `K_shear(K_bend, K_diag, f_num) =
    2 * K_bend + f_num * K_diag` (scaled by 2) for a square hexactin lattice
    with diagonal bracing fraction `f_diag = f_num / 2` where `f_num ∈ {0, 1, 2}`
    corresponds to unbraced (`f = 0`), alternating checkerboard (`f = 1/2`),
    and fully diagonally braced (`f = 1`) square cells. -/
def lattice_shear_stiffness_scaled (K_bend K_diag f_num : Nat) : Nat :=
  2 * K_bend + f_num * K_diag

/-- Theorem 4 (`euplectella_checkerboard_diagonal_bracing_shear_stiffness`):
    Proves that for canonical frame-bending stiffness `K_bend = 10` and axial
    diagonal strut contribution `K_diag = 130` (`cos² 45° * E A / L`):
    (a) The unbraced square hexactin grid (`f_num = 0`) has scaled shear stiffness `20`;
    (b) The *Euplectella* alternating open/closed checkerboard lattice (`f_num = 1`,
        `f_diag = 0.5`) raises scaled shear stiffness to `150` (a `7.5×` increase
        using only half the diagonal struts); and
    (c) Full diagonal bracing (`f_num = 2`, `f_diag = 1.0`) reaches `280`, establishing
        strict monotonicity `K_shear(0) < K_shear(1) < K_shear(2)`. -/
theorem euplectella_checkerboard_diagonal_bracing_shear_stiffness :
    lattice_shear_stiffness_scaled 10 130 0 = 20 ∧
    lattice_shear_stiffness_scaled 10 130 1 = 150 ∧
    lattice_shear_stiffness_scaled 10 130 2 = 280 ∧
    lattice_shear_stiffness_scaled 10 130 1 > 7 * lattice_shear_stiffness_scaled 10 130 0 ∧
    lattice_shear_stiffness_scaled 10 130 0 < lattice_shear_stiffness_scaled 10 130 1 ∧
    lattice_shear_stiffness_scaled 10 130 1 < lattice_shear_stiffness_scaled 10 130 2 := by
  decide

-- ============================================================================
-- PASS 3 · Law 5: Cohesive-Zone Laminate Progressive Fracture under Three-Point
--   Bending (Weaver et al. 2007; He & Hutchinson 1989; Hillerborg et al. 1976)
-- ============================================================================

/-- He & Hutchinson (1989) crack-deflection test for zero elastic mismatch: the crack
    deflects along the organic interlayer iff `G_i / G_s < 1/4`, i.e. `4 G_i < G_s`
    (fracture energies in tenths of J/m²). -/
def interlayer_deflects (G_i G_s : Nat) : Bool :=
  4 * G_i < G_s

/-- Hillerborg cohesive process-zone length `l_cz = E G_i / T_max²`
    (E in MPa, G_i in tenths of J/m², T_max in MPa; result in tenths of µm). -/
def hillerborg_length (E G_i T_max : Nat) : Nat :=
  E * G_i / (T_max * T_max)

/-- External work absorbed by the laminate up to final failure in the progressive-failure
    model: the elastic energy `W_el` stored at the first through-layer crack (identical to
    the monolithic rod of the same outer radius, which fails catastrophically there) plus
    the interface dissipation `G_i · 2πr · a` of each of the `N` deflected interlayer
    cracks (`dW_interface` per layer). -/
def laminate_fracture_work (W_el dW_interface N : Nat) : Nat :=
  W_el + N * dW_interface

/-- Theorem 5 (`cohesive_laminate_progressive_fracture_work_exceeds_monolith`):
    (a) With the Pass 3 default constants the organic interlayer (`G_i = 1.2 J/m²`,
        `G_s = 8 J/m²`) satisfies the He-Hutchinson deflection criterion, while a fused
        interlayer (`G_i = G_s`) does not — so the fused ablation fails like a monolith;
    (b) the Hillerborg length for `E = 36 GPa`, `G_i = 1.2 J/m²`, `T_max = 40 MPa` is
        exactly `27 µm` (`270` tenths of µm);
    (c) for every positive stored energy, positive per-layer interface dissipation and at
        least one deflected interlayer, the laminate's work of fracture strictly exceeds
        the monolith's (`W_el`), and it is monotone non-decreasing in the layer count. -/
theorem cohesive_laminate_progressive_fracture_work_exceeds_monolith :
    interlayer_deflects 12 80 = true ∧
    interlayer_deflects 80 80 = false ∧
    hillerborg_length 36000 12 40 = 270 ∧
    (∀ W_el dW N : Nat, 1 ≤ dW → 1 ≤ N →
        W_el < laminate_fracture_work W_el dW N) ∧
    (∀ W_el dW N : Nat,
        laminate_fracture_work W_el dW N ≤ laminate_fracture_work W_el dW (N + 1)) := by
  refine ⟨by decide, by decide, by decide, ?_, ?_⟩
  · intro W_el dW N hdW hN
    unfold laminate_fracture_work
    exact Nat.lt_add_of_pos_right (Nat.mul_pos hN hdW)
  · intro W_el dW N
    unfold laminate_fracture_work
    have h : N * dW ≤ (N + 1) * dW := Nat.mul_le_mul_right dW (Nat.le_succ N)
    exact Nat.add_le_add_left h W_el

-- ============================================================================
-- PASS 3 · Law 6: External Helical Surface Ridges — sin²(2ψ) Torsional Projection
--   and Parallel-Axis Ovalization Gain (Aizenberg et al. 2005)
-- ============================================================================

/-- Netting-analysis torsional projection of a helical ridge, written in terms of
    `t = 100 · sin² ψ ∈ [0, 100]`: `sin² ψ cos² ψ ∝ t (100 − t)`. -/
def ridge_shear_projection (t : Nat) : Nat :=
  t * (100 - t)

/-- Twice the squared sine of the ridge azimuths for `N` equally spaced ridges
    (`2 sin²(2πk/N)`, exact rationals ×2) for `N = 4` and `N = 8`, and four times for
    `N = 12` (30° steps). -/
def two_sin_sq_4 : List Nat := [0, 2, 0, 2]
def two_sin_sq_8 : List Nat := [0, 1, 2, 1, 0, 1, 2, 1]
def four_sin_sq_12 : List Nat := [0, 1, 3, 4, 3, 1, 0, 1, 3, 4, 3, 1]

/-- Parallel-axis second-moment gain `I_eff / I_wall = 1 + N_r A_r / 2` in percent for
    `N_r` ridges of area ratio `A_r` (percent). -/
def ridge_ovalization_gain_percent (N_r A_r : Nat) : Nat :=
  100 + N_r * A_r / 2

/-- Cage torsional stiffness at ψ = 45° (×2 scaling): `2 K_wall + N_r · 2 k_r`. -/
def cage_torsion_45_scaled (K_wall2 k_r2 N_r : Nat) : Nat :=
  K_wall2 + N_r * k_r2

/-- Theorem 6 (`euplectella_helical_ridge_shear_projection_and_parallel_axis_gain`):
    (a) The helical-ridge torsional projection `t (100 − t)` is maximised at `t = 50`
        (`sin² ψ = 1/2`, i.e. `ψ = 45°`) over the whole admissible range `t ≤ 100`;
    (b) for `N_r ∈ {4, 8, 12}` equally spaced ridges `Σ_k sin²(2πk/N_r) = N_r / 2`
        (sums `4 = 2·(4/2)`, `8 = 2·(8/2)`, `24 = 4·(12/2)`), which is the identity behind
        the closed-form ovalization gain;
    (c) with `A_r = 25 %`: `N_r = 8` gives a `200 %` (2.00×) second-moment gain and
        `N_r = 12` gives `250 %`, strictly monotone in the ridge count;
    (d) the Pass 3 default cage (`2 K_wall = 15`, `2 k_r = 13`, `N_r = 8`) has scaled
        torsional stiffness `119` (`59.5` vs the `7.5` checkerboard wall, gain `7.93×`),
        strictly above the Law-4 wall alone. -/
theorem euplectella_helical_ridge_shear_projection_and_parallel_axis_gain :
    ridge_shear_projection 50 = 2500 ∧
    ((List.range 101).all (fun t => decide (ridge_shear_projection t ≤ 2500))) = true ∧
    two_sin_sq_4.sum = 2 * (4 / 2) ∧
    two_sin_sq_8.sum = 2 * (8 / 2) ∧
    four_sin_sq_12.sum = 4 * (12 / 2) ∧
    ridge_ovalization_gain_percent 8 25 = 200 ∧
    ridge_ovalization_gain_percent 12 25 = 250 ∧
    ridge_ovalization_gain_percent 8 25 < ridge_ovalization_gain_percent 12 25 ∧
    cage_torsion_45_scaled 15 13 8 = 119 ∧
    cage_torsion_45_scaled 15 13 0 < cage_torsion_45_scaled 15 13 8 := by
  decide

end MathOfBiology.Loop_076
