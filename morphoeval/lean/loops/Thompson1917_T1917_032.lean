/-
  morphoeval/lean/loops/Thompson1917_T1917_032.lean
  Loop thompson1917-t1917-032 (slot 226) — claim T1917-032.

  Formal Lean 4 verification of the governing rules behind
    Thompson, D.W. (1917) On Growth and Form, Ch. XIII
    "The Shapes of Horns, and of Teeth or Tusks: with a Note on Torsion",
    pp. 612-634, Figs. 318-324,
  claim: "Teeth and tusks grow by the same spiral/torsion geometry (with St Venant
  torsion curves describing the twisted prismatic forms)."

  Companion script : thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py
  Companion KG     : morphoeval/math_kg_loops/thompson1917-t1917-032.yaml

  All definitions and theorems are self-contained in Lean 4 core (no Mathlib, no imports,
  zero `sorry`s).  Real-valued laws are formalised on their integer / natural-number
  skeletons (the algebraic identities and sign/monotonicity structure), which is exactly
  what the Python script checks numerically.
-/

namespace MathOfBiology.Loop_226

-- ============================================================================
-- Law 1 (`thompson1917-t1917-032-1`): Gnomonic self-similarity of the tusk spiral
-- ============================================================================
/-- Thompson (p. 634): the tusk is the arc of a logarithmic spiral whose tip lies near
    the pole, and "the elongated tooth has the selfsame curvature that it had when it
    was short" (p. 633).  Sampling the spiral `r(θ) = r₀ exp(kθ)` once per whorl gives
    the geometric sequence `r_n = r₀ W^n` with whorl-expansion `W = exp(2πk)` (Raup 1966).
    Gnomonic growth is the statement that each whorl is a magnified copy of the last. -/
def gnomon_radius (r0 W n : Nat) : Nat := r0 * W ^ n

/-- Whorl-to-whorl magnification by the constant factor `W`. -/
theorem gnomon_radius_succ (r0 W n : Nat) :
    gnomon_radius r0 W (n + 1) = W * gnomon_radius r0 W n := by
  unfold gnomon_radius
  rw [Nat.pow_succ]
  ac_rfl

/-- Cauchy functional equation of the equiangular spiral: `r(θ₁ + θ₂) · r(0) = r(θ₁) · r(θ₂)`.
    This is the defining self-similarity that distinguishes the logarithmic spiral from the
    Archimedean (`r_n = r₀ + n·a`) alternative. -/
theorem gnomon_radius_self_similar (r0 W m n : Nat) :
    gnomon_radius r0 W (m + n) * r0 = gnomon_radius r0 W m * gnomon_radius r0 W n := by
  unfold gnomon_radius
  rw [Nat.pow_add]
  ac_rfl

/-- The Archimedean alternative `r_n = r₀ + n a` fails self-similarity as soon as the pitch
    `a` is positive (concrete witness `r₀ = 1, a = 1, m = n = 1`): `3·1 ≠ 2·2`. -/
def archimedean_radius (r0 a n : Nat) : Nat := r0 + n * a

theorem archimedean_not_self_similar :
    archimedean_radius 1 1 (1 + 1) * 1 ≠ archimedean_radius 1 1 1 * archimedean_radius 1 1 1 := by
  decide

-- ============================================================================
-- Law 2 (`thompson1917-t1917-032-2`): Differential-accretion bending law
-- ============================================================================
/-- Thompson (p. 619): "If the rate of growth be identical on all sides of this zone, the
    horn will grow straight; if it be greater on one side than on the other, the horn will
    become curved."  The bending moment of one accreted slab is the growth-rate difference
    `v_max - v_min` across the section (the curvature is `κ = (v_max - v_min)/(w·v̄)`). -/
def bend_moment (vmax vmin : Int) : Int := vmax - vmin

/-- Equal rates on both sides give a straight tooth (zero bending). -/
theorem bend_moment_equal_rates_straight (v : Int) : bend_moment v v = 0 := by
  unfold bend_moment
  omega

/-- Swapping the fast and slow sides mirrors the curve: Thompson's "conjugate spirals",
    the homonymous (sheep) vs heteronymous (antelope, markhor) handedness of p. 619. -/
theorem bend_moment_swap_mirror (a b : Int) : bend_moment b a = - bend_moment a b := by
  unfold bend_moment
  omega

/-- The tooth curves towards the slower-growing side: the sign of the moment is the sign of
    `v_max - v_min`, and a larger rate difference bends more. -/
theorem bend_moment_sign_and_monotone (a a' b : Int) :
    (0 < bend_moment a b ↔ b < a) ∧ (a < a' → bend_moment a b < bend_moment a' b) := by
  unfold bend_moment
  constructor
  · omega
  · intro h
    omega

/-- Accretion law `κ · w = Δ` with Δ fixed by the growth-rate ratio: if the section widens
    gnomonically (`w' = W·w`, `W > 1`) between two stations, the curvature must fall
    (`κ' < κ`) — the tusk is tightly hooked at the tip and gently curved at the root.
    With constant width (`W = 1`, rodent incisor) the curvature is constant: a circle. -/
theorem accretion_curvature_falls_with_width
    (kappa kappa' w W : Int) (hw : 0 < w) (hk : 0 < kappa) (hW : 1 < W)
    (hlaw : kappa' * (W * w) = kappa * w) : kappa' < kappa := by
  apply Int.not_le.mp
  intro hge
  have hWw_pos : 0 < W * w := Int.mul_pos (by omega) hw
  have h1 : kappa * (W * w) ≤ kappa' * (W * w) :=
    Int.mul_le_mul_of_nonneg_right hge (Int.le_of_lt hWw_pos)
  have hw_lt : w < W * w := by
    have : 1 * w < W * w := Int.mul_lt_mul_of_pos_right hW hw
    omega
  have h2 : kappa * w < kappa * (W * w) := Int.mul_lt_mul_of_pos_left hw_lt hk
  omega

theorem accretion_constant_width_is_circle (kappa kappa' w : Int) (hw : 0 < w)
    (hlaw : kappa' * (1 * w) = kappa * w) : kappa' = kappa := by
  have h : kappa' * w = kappa * w := by
    have := hlaw
    rw [Int.one_mul] at this
    exact this
  have hne : w ≠ 0 := by omega
  exact Int.eq_of_mul_eq_mul_right hne h

-- ============================================================================
-- Law 3 (`thompson1917-t1917-032-3`): St Venant torsion curves of the twisted prism
-- ============================================================================
/-- St Venant (1856) / Thompson Fig. 321: the axial warping of a twisted equilateral prism
    (centroid at the origin, vertex on the +x axis) is proportional to the harmonic cubic
    `3x²y − y³`.  Its zero set is the three medians (`y = 0`, `y² = 3x²`), so the section
    splits into six alternating sectors of elevation / depression and every face is
    "convex upwards on one half, and concave upwards on the other half" (p. 622). -/
def tri_warp (x y : Int) : Int := 3 * x * x * y - y * y * y

/-- The median through the vertex `X` (the x-axis) is a nodal line of the warping. -/
theorem tri_warp_median_nodal (x : Int) : tri_warp x 0 = 0 := by
  simp [tri_warp]

/-- Anticlastic antisymmetry across the `OX` median: elevation on one side, equal
    depression on the other (the solid vs dotted contours of Fig. 321). -/
theorem tri_warp_anticlastic (x y : Int) : tri_warp x (-y) = - tri_warp x y := by
  unfold tri_warp
  simp only [Int.mul_neg, Int.neg_mul, Int.neg_neg, Int.sub_neg, Int.neg_sub]
  omega

/-- Mirror symmetry across the `OY` axis (the figure is symmetric in `x`). -/
theorem tri_warp_mirror (x y : Int) : tri_warp (-x) y = tri_warp x y := by
  unfold tri_warp
  simp only [Int.mul_neg, Int.neg_mul, Int.neg_neg]

/-- Six alternating sign sectors, witnessed at six sample points 60° apart (integer
    approximations of `(cos θ, sin θ)` at `θ = 30°, 90°, 150°, 210°, 270°, 330°`
    scaled by 2: the sign alternates `+ − + − + −` around the centroid. -/
theorem tri_warp_six_sectors :
    0 < tri_warp 7 4 ∧ tri_warp 0 8 < 0 ∧ 0 < tri_warp (-7) 4 ∧
    tri_warp (-7) (-4) < 0 ∧ 0 < tri_warp 0 (-8) ∧ tri_warp 7 (-4) < 0 := by
  decide

/-- St Venant's elliptic prism warps as `u_z ∝ (b² − a²)·x·y`; for the circular section
    `a = b` the warping vanishes identically — "a twist leaves each circular section
    unchanged" (Thompson p. 622; the cylindrical horns of oxen show no St Venant curves). -/
def ellipse_warp (a b x y : Int) : Int := (b * b - a * a) * x * y

theorem ellipse_warp_circle_vanishes (a x y : Int) : ellipse_warp a a x y = 0 := by
  unfold ellipse_warp
  simp

/-- Exchanging the semi-axes inverts the saddle (the anticlastic surface flips sign). -/
theorem ellipse_warp_axis_swap (a b x y : Int) : ellipse_warp b a x y = - ellipse_warp a b x y := by
  unfold ellipse_warp
  have h : (a * a - b * b) = -(b * b - a * a) := by omega
  rw [h, Int.neg_mul, Int.neg_mul]

-- ============================================================================
-- Law 4 (`thompson1917-t1917-032-4`): Uniform-torsion helix of the surface fibres
-- ============================================================================
/-- A prism (tusk) twisted at the uniform rate `τ` carries every surface fibre at radius `ρ`
    onto a helix with `tan φ = ρ·τ` (φ = inclination from the axis).  The sign of `τ` is the
    handedness (sinistral in every narwhal tusk, Kingsley & Ramsay 1988), and the pitch
    `2π/τ` is independent of `ρ`, so on a tapering tusk the fibres become more axial
    towards the tip. -/
def helix_slope (rho tau : Int) : Int := rho * tau

/-- Outer fibres are more inclined than inner ones for a right-handed (`τ > 0`) twist:
    strict monotonicity in `ρ`. -/
theorem helix_slope_monotone_in_radius (rho1 rho2 tau : Int) (h : rho1 < rho2) (ht : 0 < tau) :
    helix_slope rho1 tau < helix_slope rho2 tau := by
  unfold helix_slope
  exact Int.mul_lt_mul_of_pos_right h ht

/-- Reversing the twist reverses every fibre's inclination (mirror-image tusk). -/
theorem helix_slope_handedness_flip (rho tau : Int) : helix_slope rho (-tau) = - helix_slope rho tau := by
  unfold helix_slope
  exact Int.mul_neg rho tau

/-- The handedness of every fibre is decided by the sign of `τ` alone, whatever its radius. -/
theorem helix_slope_sign_from_twist (rho tau : Int) (hr : 0 < rho) :
    (0 < helix_slope rho tau ↔ 0 < tau) ∧ (helix_slope rho tau < 0 ↔ tau < 0) := by
  unfold helix_slope
  constructor
  · constructor
    · intro hpos
      apply Int.not_le.mp
      intro hle
      have : rho * tau ≤ 0 := Int.mul_nonpos_of_nonneg_of_nonpos (Int.le_of_lt hr) hle
      omega
    · intro hpos
      exact Int.mul_pos hr hpos
  · constructor
    · intro hneg
      apply Int.not_le.mp
      intro hge
      have : 0 ≤ rho * tau := Int.mul_nonneg (Int.le_of_lt hr) hge
      omega
    · intro hneg
      exact Int.mul_neg_of_pos_of_neg hr hneg

/-- Thompson's stacked-slab experiment (p. 618): Ovis ammon edge lengths 80 : 74 : 45 cm.
    The lateral asymmetry `80 − 74 = 6` is small against the main gradient
    `80 + 74 − 2·45 = 64`, so the horn is "nearly" a plane spiral with a weak helical
    component; exchanging the outer and inner edges flips the lateral sign (heteronymous
    antelope / markhor twist), exactly as `bend_moment_swap_mirror` predicts. -/
theorem ovis_ammon_slab_gradient :
    bend_moment 80 74 = 6 ∧ bend_moment (80 + 74) (2 * 45) = 64 ∧
    bend_moment 74 80 = -6 ∧ 10 * 6 < 64 := by
  decide

end MathOfBiology.Loop_226
