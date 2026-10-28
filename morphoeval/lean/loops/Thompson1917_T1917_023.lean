/-
  morphoeval/lean/loops/Thompson1917_T1917_023.lean
  Formal Lean 4 verification of the geometric content of Thompson (1917) On Growth and Form,
  Ch. IX "On Concretions, Spicules, and Spicular Skeletons", pp. 411-487 (claim T1917-023):
  calcospherites and concretions grow by radial deposition about a centre (Harting 1872,
  artificial CaCO3 concretions in egg albumen), giving concentric / spherical forms by physical
  deposition alone (`thompson1917-t1917-023`, Slot 218).

  Laws formalised (integer-scaled, so that every statement is decidable in core Lean 4):
    Law 1  interface-controlled radial deposition  R_n = R_0 + v n  (constant lamella thickness,
           time-translation / semigroup property)                           — Thompson pp. 421-422, 437
    Law 2  diffusion-limited radial deposition  R_n^2 = R_0^2 + 2 kappa n  (lamellae thin outward)
                                                                              — Zener / Frank / Mullins-Sekerka
    Law 3  sphericity stability: uniform normal deposition keeps  max rho - min rho  fixed, so the
           cross-multiplied asphericity is non-increasing in time              — Thompson pp. 425-426
    Law 4  equal-rate impingement: the difference of squared distances to two centres is affine in
           the point, hence two spherites meet on a plane (Voronoi mosaic)     — Thompson pp. 422, 426
    Law 5  Liesegang "logarithmic ratio": geometric band radii  r_n = r_0 P^n  have gaps that grow
           by the factor P per band                                            — Thompson p. 427

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_218

-- ============================================================================
-- Law 1 (`thompson1917-t1917-023-1`):
-- Interface-controlled radial deposition about a centre (Thompson pp. 421-422, 437)
-- ============================================================================
/-- Radius after `n` deposition intervals when the front advances by a fixed normal
    increment `v` per interval (`dR/dt = v`, integer-scaled). -/
def radius_lin (R0 v n : Int) : Int :=
  R0 + v * n

/-- Lamella deposited in the `n`-th interval. -/
def lamella_lin (R0 v n : Int) : Int :=
  radius_lin R0 v (n + 1) - radius_lin R0 v n

/-- Every lamella has the same thickness `v` (the concentric shells are equally spaced), and the
    growth law is a semigroup: depositing `n` then `m` intervals equals depositing `n + m`. -/
theorem lamella_thickness_constant (R0 v n : Int) :
    lamella_lin R0 v n = v := by
  dsimp [lamella_lin, radius_lin]
  simp [Int.mul_add]
  omega

theorem radial_deposition_semigroup (R0 v n m : Int) :
    radius_lin (radius_lin R0 v n) v m = radius_lin R0 v (n + m) := by
  dsimp [radius_lin]
  simp [Int.mul_add]
  omega

-- ============================================================================
-- Law 2 (`thompson1917-t1917-023-2`):
-- Diffusion-limited radial deposition: lamellae thin outward (Zener 1949 / Mullins-Sekerka 1963)
-- ============================================================================
/-- Squared radius after `n` intervals under quasi-steady diffusion-limited supply
    (`R dR/dt = kappa`  =>  `R^2 = R_0^2 + 2 kappa t`, integer-scaled). -/
def radius_sq_parabolic (R0 kappa n : Int) : Int :=
  R0 * R0 + 2 * kappa * n

/-- The increment of the squared radius per interval is the constant `2 kappa`. -/
theorem parabolic_shell_identity (R0 kappa n : Int) :
    radius_sq_parabolic R0 kappa (n + 1) - radius_sq_parabolic R0 kappa n = 2 * kappa := by
  dsimp [radius_sq_parabolic]
  simp [Int.mul_add]
  omega

/-- Difference of squares factorises: `(Ra - Rb) * (Ra + Rb) = Ra^2 - Rb^2`. -/
theorem sq_diff_factor (Ra Rb : Int) :
    (Ra - Rb) * (Ra + Rb) = Ra * Ra - Rb * Rb := by
  rw [Int.sub_mul, Int.mul_add, Int.mul_add]
  have hc : Ra * Rb = Rb * Ra := Int.mul_comm Ra Rb
  omega

/-- Three successive radii `R1 < R2 < R3` whose squares grow by the same constant `c > 0`
    (`R2^2 - R1^2 = R3^2 - R2^2 = c`) have strictly thinning lamellae: `R3 - R2 < R2 - R1`.
    This is Thompson's/Harting's outer lamellae becoming finer when supply is diffusion-limited. -/
theorem lamellae_thin_outward (R1 R2 R3 c : Int)
    (h0 : 0 < R1) (h12 : R1 < R2) (h23 : R2 < R3)
    (hA : R2 * R2 - R1 * R1 = c) (hB : R3 * R3 - R2 * R2 = c) :
    R3 - R2 < R2 - R1 := by
  have eA : (R2 - R1) * (R2 + R1) = c := by rw [sq_diff_factor]; exact hA
  have eB : (R3 - R2) * (R3 + R2) = c := by rw [sq_diff_factor]; exact hB
  have hd1 : 0 < R2 - R1 := Int.sub_pos_of_lt h12
  have hd2 : 0 < R3 - R2 := Int.sub_pos_of_lt h23
  have hS : R2 + R1 < R3 + R2 := by omega
  have hS1 : 0 ≤ R2 + R1 := by omega
  -- suppose, for contradiction, that the outer lamella is not thinner
  apply Int.not_le.mp
  intro hge
  have h1 : (R2 - R1) * (R2 + R1) ≤ (R3 - R2) * (R2 + R1) :=
    Int.mul_le_mul_of_nonneg_right hge hS1
  have h2 : (R3 - R2) * (R2 + R1) < (R3 - R2) * (R3 + R2) :=
    Int.mul_lt_mul_of_pos_left hS hd2
  have h3 : (R2 - R1) * (R2 + R1) < (R3 - R2) * (R3 + R2) := Int.lt_of_le_of_lt h1 h2
  rw [eA, eB] at h3
  exact Int.lt_irrefl c h3

-- ============================================================================
-- Law 3 (`thompson1917-t1917-023-3`):
-- Sphericity stability under uniform normal deposition (Thompson pp. 425-426)
-- ============================================================================
/-- Largest and smallest radial extent of a (possibly aspherical) seed after `n` intervals of
    uniform normal deposition `v`: every point of the front advances by the same amount. -/
def rho_max (Rmax0 v n : Int) : Int := Rmax0 + v * n
def rho_min (Rmin0 v n : Int) : Int := Rmin0 + v * n

/-- The absolute asphericity `max rho - min rho` is a conserved quantity of uniform deposition. -/
theorem absolute_asphericity_conserved (Rmax0 Rmin0 v n : Int) :
    rho_max Rmax0 v n - rho_min Rmin0 v n = Rmax0 - Rmin0 := by
  dsimp [rho_max, rho_min]
  omega

/-- Cross-multiplied statement that the *relative* asphericity
    `(max rho - min rho) / min rho` is non-increasing in time: for `n <= m` and a seed with
    `0 < Rmin0 <= Rmax0`, `(Rmax0 - Rmin0) * rho_min(n) <= (Rmax0 - Rmin0) * rho_min(m)`,
    i.e. `A(m) <= A(n)`. The concretion therefore rounds off as it grows, whatever the seed. -/
theorem asphericity_nonincreasing (Rmax0 Rmin0 v n m : Int)
    (hseed : Rmin0 ≤ Rmax0) (hv : 0 ≤ v) (hnm : n ≤ m) :
    (Rmax0 - Rmin0) * rho_min Rmin0 v n ≤ (Rmax0 - Rmin0) * rho_min Rmin0 v m := by
  dsimp [rho_min]
  have hA : 0 ≤ Rmax0 - Rmin0 := Int.sub_nonneg_of_le hseed
  have hmono : Rmin0 + v * n ≤ Rmin0 + v * m := by
    have := Int.mul_le_mul_of_nonneg_left hnm hv
    omega
  exact Int.mul_le_mul_of_nonneg_left hmono hA

-- ============================================================================
-- Law 4 (`thompson1917-t1917-023-4`):
-- Equal-rate impingement plane / Voronoi mosaic (Thompson pp. 422, 426)
-- ============================================================================
/-- Squared Euclidean distance from the point `(x, y, z)` to the centre `(a, b, c)`. -/
def sqdist (x y z a b c : Int) : Int :=
  (x - a) * (x - a) + (y - b) * (y - b) + (z - c) * (z - c)

/-- Two spherites nucleated at `c1 = (a1,b1,c1)` and `c2 = (a2,b2,c2)` growing at the same
    rate have equal radii when they meet, so their common boundary is the locus
    `sqdist p c1 = sqdist p c2`. The difference of squared distances is *affine* in `p`:
      `sqdist p c1 - sqdist p c2 = 2 (a2 - a1) x + 2 (b2 - b1) y + 2 (c2 - c1) z
                                   + (a1^2 + b1^2 + c1^2) - (a2^2 + b2^2 + c2^2)`,
    hence the boundary is a plane (the perpendicular bisector) and `N` centres partition space
    into convex Voronoi polyhedra — Thompson's "polyhedral, or often regularly hexagonal" crowded
    calcospherites and "contiguous soap-bubble" pairs. -/
theorem impingement_locus_is_affine (x y z a1 b1 c1 a2 b2 c2 : Int) :
    sqdist x y z a1 b1 c1 - sqdist x y z a2 b2 c2 =
      2 * (a2 - a1) * x + 2 * (b2 - b1) * y + 2 * (c2 - c1) * z
        + (a1 * a1 + b1 * b1 + c1 * c1) - (a2 * a2 + b2 * b2 + c2 * c2) := by
  dsimp [sqdist]
  grind

/-- The midpoint of two centres lies on their impingement plane (so a pair of equal
    spherites is always split symmetrically, as Harting's "dumb-bell" pairs are). -/
theorem midpoint_on_impingement_plane (a1 b1 c1 a2 b2 c2 : Int) :
    sqdist (a1 + a2) (b1 + b2) (c1 + c2) (2 * a1) (2 * b1) (2 * c1) =
      sqdist (a1 + a2) (b1 + b2) (c1 + c2) (2 * a2) (2 * b2) (2 * c2) := by
  dsimp [sqdist]
  have hx : a1 + a2 - 2 * a1 = -(a1 + a2 - 2 * a2) := by omega
  have hy : b1 + b2 - 2 * b1 = -(b1 + b2 - 2 * b2) := by omega
  have hz : c1 + c2 - 2 * c1 = -(c1 + c2 - 2 * c2) := by omega
  rw [hx, hy, hz]
  simp [Int.neg_mul_neg]

-- ============================================================================
-- Law 5 (`thompson1917-t1917-023-5`):
-- Liesegang "logarithmic ratio" band spacing (Thompson p. 427; Jablczynski 1923)
-- ============================================================================
/-- Radius of the `n`-th concentric band when successive bands stand in the constant ratio `P`
    (Jablczynski spacing law, integer-scaled: `r_n = r_0 * P^n`). -/
def band_radius (r0 P : Nat) (n : Nat) : Nat :=
  r0 * P ^ n

/-- Gap between band `n` and band `n + 1`. -/
def band_gap (r0 P : Nat) (n : Nat) : Nat :=
  band_radius r0 P (n + 1) - band_radius r0 P n

/-- Successive radii stand in the constant ratio `P`: `r_{n+1} = P * r_n`. -/
theorem band_ratio_constant (r0 P n : Nat) :
    band_radius r0 P (n + 1) = P * band_radius r0 P n := by
  dsimp [band_radius]
  rw [Nat.pow_succ]
  simp [Nat.mul_comm, Nat.mul_left_comm]

/-- Each gap is `P` times the previous gap, so for `P >= 1` the zones "stand farther and farther
    apart ... as they recede from the centre" (Thompson p. 427). -/
theorem geometric_band_gap_grows (r0 P n : Nat) :
    band_gap r0 P (n + 1) = P * band_gap r0 P n := by
  dsimp [band_gap]
  rw [band_ratio_constant r0 P (n + 1), band_ratio_constant r0 P n]
  exact (Nat.mul_sub _ _ _).symm

theorem band_gap_monotone (r0 P n : Nat) (hP : 1 ≤ P) :
    band_gap r0 P n ≤ band_gap r0 P (n + 1) := by
  rw [geometric_band_gap_grows]
  exact Nat.le_mul_of_pos_left (band_gap r0 P n) hP

end MathOfBiology.Loop_218
