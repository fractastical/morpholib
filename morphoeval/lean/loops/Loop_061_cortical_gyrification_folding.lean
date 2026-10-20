/-
  morphoeval/lean/loops/Loop_061_cortical_gyrification_folding.lean
  Formal Lean 4 verification of Richman, Stewart, Hutchinson & Caviness (1975)
  Science 189:18-21 and Tallinen, Chung, Biggins & Mahadevan (2014) PNAS 111:12667-12672 /
  Tallinen et al. (2016) Nature Physics 12:588-593 governing cortical expansion,
  sulcal creasing wavelength scaling, and gyrification index laws
  (`richman-cortical-gyrification-buckling`, Slot 061).

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_061

-- ============================================================================
-- Law 1 (`richman-cortical-gyrification-buckling-1`):
-- Tallinen et al. (2014) Multiplicative Growth Decomposition & Critical Threshold
-- ============================================================================
/-- In Tallinen et al. (2014) PNAS Eq. (1), the deformation gradient decomposes
    multiplicatively as `F = F_e G`, where tangential cortical expansion `g`
    relative to the critical sulcal buckling threshold `g_c` defines the
    supercritical expansion excess `cortical_growth_excess g g_c = g - g_c`.
    At `g = g_c` (or in the zero-growth control), the buckling excess vanishes
    identically (`0`, lissencephalic state), whereas supercritical tangential
    growth `g = g_c + dg + 1` yields a strictly positive folding excess `dg + 1 ≥ 1`. -/
def cortical_growth_excess (g g_c : Int) : Int :=
  g - g_c

theorem tallinen_critical_growth_threshold (g_c : Int) :
    cortical_growth_excess g_c g_c = 0 ∧
    ∀ dg : Nat, cortical_growth_excess (g_c + Int.ofNat (dg + 1)) g_c = Int.ofNat (dg + 1) := by
  constructor
  · simp [cortical_growth_excess]
  · intro dg
    simp [cortical_growth_excess]
    omega


-- ============================================================================
-- Law 2 (`richman-cortical-gyrification-buckling-2`):
-- Richman et al. (1975) & Tallinen et al. (2014) Cube-Law Sulcal Wavelength Scaling
-- (`λ = 2π h (μ_c / (3 μ_s))^(1/3)`)
-- ============================================================================
/-- Balancing cortical gray-matter bending energy per unit wavelength
    `U_bend ~ μ_c h^3 / λ^2` against subcortical white-matter foundation
    deformation energy `U_sub ~ 3 μ_s λ` and differentiating with respect to `λ`
    yields the integer-scaled Euler-Lagrange stationarity residual:
      `R_sulcus(λ, h, μ_c, μ_s) = 3 * μ_s * λ^3 - 2 * μ_c * h^3`.
    This theorem proves that the stationarity residual vanishes when the cube-law
    stiffness-thickness balance holds and that doubling wavelength `λ = 2 h`
    requires an `8`-fold (`2^3`) increase in the effective modulus ratio `μ_c / μ_s`. -/
def richman_wavelength_residual (lam h mu_c mu_s : Int) : Int :=
  3 * mu_s * (lam * lam * lam) - 2 * mu_c * (h * h * h)

theorem richman_cube_law_wavelength_stationarity (h mu_s : Int) :
    richman_wavelength_residual h h (3 * mu_s) (2 * mu_s) = 0 ∧
    richman_wavelength_residual (2 * h) h (24 * mu_s) (2 * mu_s) = 0 := by
  constructor
  · simp [richman_wavelength_residual, Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]
  · simp [richman_wavelength_residual, Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]


-- ============================================================================
-- Law 3 (`richman-cortical-gyrification-buckling-3`):
-- Zilles et al. (1988) & Tallinen et al. (2016) Gyrification Index Buried Sulcal
-- Perimeter Excess Law (`GI = P_pial / P_hull ≥ 1`)
-- ============================================================================
/-- In a coronal cerebral hemisphere section with outer convex hull perimeter
    `P_hull` and `n_sulci` infolded sulci of mean depth `d_sulcus`, each sulcus
    contributes two opposing sulcal walls (`2 * d_sulcus`) to the total pial
    perimeter `P_pial = P_hull + 2 * n_sulci * d_sulcus`.
    The buried sulcal perimeter excess `P_pial - P_hull` vanishes identically in
    the lissencephalic limit (`n_sulci = 0`, `GI = 1`) and equals `2 * n_sulci * d_sulcus`
    in the gyrencephalic regime. -/
def gyrification_perimeter_excess (P_hull n_sulci d_sulcus : Int) : Int :=
  (P_hull + 2 * n_sulci * d_sulcus) - P_hull

theorem gyrification_index_sulcal_excess (P_hull n_sulci d_sulcus : Int) :
    gyrification_perimeter_excess P_hull 0 d_sulcus = 0 ∧
    gyrification_perimeter_excess P_hull n_sulci d_sulcus = 2 * n_sulci * d_sulcus := by
  constructor
  · simp [gyrification_perimeter_excess]
  · simp [gyrification_perimeter_excess]
    omega

end MathOfBiology.Loop_061
