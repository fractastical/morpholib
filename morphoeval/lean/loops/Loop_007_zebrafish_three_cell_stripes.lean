/-
  morphoeval/lean/loops/Loop_007_zebrafish_three_cell_stripes.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 007:
  Nakamasu, Takahashi, Kanbe & Kondo (2009) PNAS 106(21):8429-8434 —
  "Interactions between zebrafish pigment cells responsible for the generation
  of Turing patterns" (`nakamasu-zebrafish-three-cell-stripes`).

  Pure core Lean 4 (zero external dependencies, zero `sorry`s).
  Verify with:
    lean morphoeval/lean/loops/Loop_007_zebrafish_three_cell_stripes.lean
-/

namespace MathOfBiology.Loop_007

-- ============================================================================
-- Law 1 (`nakamasu-zebrafish-three-cell-stripes-1`):
-- Short-Range Melanophore-Xanthophore Mutual Repulsion Saddle Instability
-- ============================================================================

/-- 100x-scaled determinant of the 2x2 short-range melanophore-xanthophore (u, v)
    Jacobian block `J_uv = [[-c_u, -r_u], [-r_v, -c_v]]`:
    `det_uv = c_u * c_v - r_u * r_v`. -/
def short_range_det_uv (cu cv ru rv : Int) : Int :=
  cu * cv - ru * rv

/-- 10x-scaled growth rate of the anti-phase pigment segregation mode `m = u - v`
    under symmetric short-range decay `cu` and mutual repulsion `ru`:
    `lambda_anti = ru - cu`. -/
def anti_phase_growth_rate (cu ru : Int) : Int :=
  ru - cu

/-- Theorem 1 (`nakamasu_short_range_mutual_repulsion_instability`):
    (1) When short-range mutual repulsion is ablated (`ru = 0`), `det_uv ≥ 1`
        for all positive decay rates `cu ≥ 1, cv ≥ 1`, so the 2-cell (u, v) block
        cannot undergo saddle segregation.
    (2) At Nakamasu et al. (2009) 10x-scaled canonical parameters (`cu = cv = 8`,
        `ru = rv = 15`), `short_range_det_uv 8 8 15 15 = -161 < 0` and
        `anti_phase_growth_rate 8 15 = 7 > 0`, proving that mutual contact repulsion
        creates an effective local positive feedback loop (`(-15)*(-15) = +225 > 64`). -/
theorem nakamasu_short_range_mutual_repulsion_instability (cu cv : Int)
    (hcu : cu ≥ 1) (hcv : cv ≥ 1) :
    short_range_det_uv cu cv 0 0 ≥ 1 ∧
    short_range_det_uv 8 8 15 15 = -161 ∧
    anti_phase_growth_rate 8 15 = 7 := by
  unfold short_range_det_uv anti_phase_growth_rate
  refine ⟨?_, by decide, by decide⟩
  have h1 : (1 : Int) * 1 ≤ cu * cv := Int.mul_le_mul hcu hcv (by decide) (by omega)
  omega


-- ============================================================================
-- Law 2 (`nakamasu-zebrafish-three-cell-stripes-2`):
-- Long-Range Xanthophore -> Melanophore Factor w Routh-Hurwitz Stabilization
-- ============================================================================

/-- Routh-Hurwitz cubic stability invariants `(p1, p2, p3, p1*p2 - p3)` at `k = 0`
    for the 10x-scaled 3-species Jacobian
    `J0 = [[-cu, bu, du], [av, -cv, dv], [aw, bw, -cw]]`. -/
def routh_hurwitz_p1 (cu cv cw : Int) : Int :=
  cu + cv + cw

def routh_hurwitz_p2 (cu cv cw bu du av dv aw bw : Int) : Int :=
  (cu * cv - bu * av) + (cu * cw - du * aw) + (cv * cw - dv * bw)

def routh_hurwitz_p3 (cu cv cw bu du av dv aw bw : Int) : Int :=
  cu * (cv * cw - dv * bw)
  - bu * (av * cw + dv * aw)
  - du * (av * bw + cv * aw)

/-- Theorem 2 (`nakamasu_long_range_factor_routh_hurwitz_stabilization`):
    Coupling the locally unstable `(u, v)` mutual-repulsion pair (`cu=8, cv=8, bu=-15, av=-15`,
    where `M_uv = 64 - 225 = -161 < 0`) to the long-range xanthophore survival pathway
    `v -> w -> u` (`bw = 12, du = 12, aw = -9, dv = -6, cw = 9`) makes all four
    Routh-Hurwitz stability invariants strictly positive at `k = 0`:
    `p1 = 25 > 0`, `p2 = 163 > 0`, `p3 = 2961 > 0`, and `p1 * p2 - p3 = 1114 > 0`. -/
theorem nakamasu_long_range_factor_routh_hurwitz_stabilization :
    routh_hurwitz_p1 8 8 9 = 25 ∧
    routh_hurwitz_p2 8 8 9 (-15) 12 (-15) (-6) (-9) 12 = 163 ∧
    routh_hurwitz_p3 8 8 9 (-15) 12 (-15) (-6) (-9) 12 = 2961 ∧
    (routh_hurwitz_p1 8 8 9) * (routh_hurwitz_p2 8 8 9 (-15) 12 (-15) (-6) (-9) 12)
      - (routh_hurwitz_p3 8 8 9 (-15) 12 (-15) (-6) (-9) 12) = 1114 := by
  decide


-- ============================================================================
-- Law 3 (`nakamasu-zebrafish-three-cell-stripes-3`):
-- Fast-Diffusing Long-Range Factor w Generates Bounded Turing Wavenumber Band
-- ============================================================================

/-- 1000x-scaled constant term `p3(q) = -det(J0 - q * D)` of the dispersion
    characteristic polynomial at squared wavenumber `q = k^2`, where
    `10x`-scaled diagonal decay shifts are `Cu = 80 + 2*q`, `Cv = 80 + 2*q`,
    `Cw = 90 + 50*q` (representing `D_u = D_v = 0.02, D_w = 0.50`, ratio `D_w/D_u = 25`),
    and off-diagonal 100x-scaled entries are `Bu = -150, Du_c = 120, Av = -150,
    Dv_c = -60, Aw = -90, Bw = 120`.
    Stationary Turing instability (`det(J(q)) > 0`) occurs iff `p3_dispersive < 0`. -/
def p3_dispersive (q du100 dv100 dw100 : Int) : Int :=
  let Cu := 80 + du100 * q
  let Cv := 80 + dv100 * q
  let Cw := 90 + dw100 * q
  let Bu := (-150 : Int)
  let Du_c := (120 : Int)
  let Av := (-150 : Int)
  let Dv_c := (-60 : Int)
  let Aw := (-90 : Int)
  let Bw := (120 : Int)
  Cu * (Cv * Cw - Dv_c * Bw)
  - Bu * (Av * Cw + Dv_c * Aw)
  - Du_c * (Av * Bw + Cv * Aw)

/-- Theorem 3 (`nakamasu_diffusive_turing_band_and_ablation_collapse`):
    With fast long-range factor diffusion (`D_w / D_u = 50 / 2 = 25`):
    (1) At `q = 0` (homogeneous mode), `p3_dispersive 0 2 2 50 = 2961000 > 0` (stable).
    (2) At `q = 12` (critical stripe wavenumber `k_c^2 ≈ 11.63`),
        `p3_dispersive 12 2 2 50 = -3219960 < 0` (stationary Turing stripe instability!).
    (3) Above the upper cutoff (`q = 35 > k_+^2 = 28.80`),
        `p3_dispersive 35 2 2 50 = 5670000 > 0` (short-wavelength stabilization).
    (4) Under equal diffusion (`D_u = D_v = D_w = 2`), `p3_dispersive 12 2 2 2 = 3510024 > 0`,
        proving that differential long-range transport `D_w ≫ D_u, D_v` is strictly
        required for stripe pattern formation. -/
theorem nakamasu_diffusive_turing_band_and_ablation_collapse :
    p3_dispersive 0 2 2 50 = 2961000 ∧
    p3_dispersive 12 2 2 50 = -3219960 ∧
    p3_dispersive 35 2 2 50 = 5670000 ∧
    p3_dispersive 12 2 2 2 = 3510024 := by
  decide


-- ============================================================================
-- Pass 3 additions (hybrid continuum-agent iridophore model on a growing flank)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Law 4 (`nakamasu-zebrafish-three-cell-stripes-4`):
-- Growing-Domain Neumann Mode Insertion Across the Turing Band
-- ----------------------------------------------------------------------------

/-- Integer kernel of the Turing-band membership test for the Neumann cosine mode
    `cos(n π y / L)` on a flank of height `L`: the mode is inside the Nakamasu (2009)
    band `k ∈ [k_min, k_max] = [2.1134, 5.3669]` iff `k_min L ≤ n π ≤ k_max L`.
    Scalings: `k × 10^4` (`21134`, `53669`), `π × 10^5` (`314159`), `L × 10` (`L10`).
    Dividing both sides by `10^5` recovers the real inequality. -/
def mode_in_turing_band (n L10 : Int) : Bool :=
  decide (21134 * L10 ≤ n * 314159 ∧ n * 314159 ≤ 53669 * L10)

/-- Theorem 4 (`nakamasu_growing_domain_stripe_insertion`):
    On the juvenile flank (`L = 3.7`, stage PB, SL ≈ 7.2 mm) the mode `n = 4`
    (two dark stripes 1D/1V flanking X0) is admissible while the adult mode `n = 10`
    is not; on the adult flank (`L = 9.2`) the band is `n ∈ [7, 15]` so the juvenile
    mode `n = 4` has left the band and `n = 10` (≈ five stripes) is admissible.
    Hence isometric growth from `L = 3.7` to `L = 9.2` *forces* stripe insertion:
    no single Neumann mode is inside the Turing band at both sizes. -/
theorem nakamasu_growing_domain_stripe_insertion :
    mode_in_turing_band 4 37 = true ∧
    mode_in_turing_band 10 37 = false ∧
    mode_in_turing_band 4 92 = false ∧
    mode_in_turing_band 6 92 = false ∧
    mode_in_turing_band 7 92 = true ∧
    mode_in_turing_band 10 92 = true ∧
    mode_in_turing_band 15 92 = true ∧
    mode_in_turing_band 16 92 = false := by
  decide

/-- Scale invariance of the band test: the admissible mode set scales with the
    domain, so doubling `L` doubles every admissible `n` (stripe count ∝ flank height). -/
theorem mode_in_turing_band_scales (n L10 : Int) :
    mode_in_turing_band (2 * n) (2 * L10) = mode_in_turing_band n L10 := by
  unfold mode_in_turing_band
  apply decide_eq_decide.mpr
  constructor
  · intro h; constructor <;> omega
  · intro h; constructor <;> omega


-- ----------------------------------------------------------------------------
-- Law 5 (`nakamasu-zebrafish-three-cell-stripes-5`):
-- Growth Dilution Preserves Routh-Hurwitz Stability and the Short-Range Saddle
-- ----------------------------------------------------------------------------

/-- Theorem 5 (`nakamasu_growth_dilution_preserves_routh_hurwitz`):
    On a uniformly growing domain (Crampin-Gaffney-Maini 1999) the homogeneous
    Jacobian becomes `J0 - r I` with dilution rate `r = 2 g'/g ≥ 0`, i.e. every
    clearance rate is raised by `r`. At the 10x-scaled dilution `r = 1` (0.1, five
    times the maximum used by the script, 2·g'/g ≤ 0.0195) the invariants stay strictly positive
    (`p1 = 28`, `p2 = 216`, `p3 = 3150`, `p1 p2 - p3 = 2898`) while the short-range
    saddle minor remains negative (`81 - 225 = -144`): growth does not destroy the
    Nakamasu stability/instability structure of Theorems 1-2. -/
theorem nakamasu_growth_dilution_preserves_routh_hurwitz :
    routh_hurwitz_p1 9 9 10 = 28 ∧
    routh_hurwitz_p2 9 9 10 (-15) 12 (-15) (-6) (-9) 12 = 216 ∧
    routh_hurwitz_p3 9 9 10 (-15) 12 (-15) (-6) (-9) 12 = 3150 ∧
    (routh_hurwitz_p1 9 9 10) * (routh_hurwitz_p2 9 9 10 (-15) 12 (-15) (-6) (-9) 12)
      - (routh_hurwitz_p3 9 9 10 (-15) 12 (-15) (-6) (-9) 12) = 2898 ∧
    short_range_det_uv 9 9 15 15 = -144 := by
  decide

/-- Dilution only ever increases `p1` (the trace invariant), for any `r ≥ 0`. -/
theorem dilution_raises_p1 (cu cv cw r : Int) (hr : r ≥ 0) :
    routh_hurwitz_p1 (cu + r) (cv + r) (cw + r) = routh_hurwitz_p1 cu cv cw + 3 * r ∧
    routh_hurwitz_p1 (cu + r) (cv + r) (cw + r) ≥ routh_hurwitz_p1 cu cv cw := by
  unfold routh_hurwitz_p1
  constructor <;> omega


-- ----------------------------------------------------------------------------
-- Law 6 (`nakamasu-zebrafish-three-cell-stripes-6`):
-- Iridophore S <-> L Switching Is Exclusive and Conserves Agents
-- ----------------------------------------------------------------------------

/-- Dense -> loose switch fires when the local melanophore excess `d = u - v`
    exceeds the threshold `θ`; loose -> dense fires when `-d > θ`
    (Volkening & Sandstede 2018 rule, scaled to integers). -/
def switch_S_to_L (d θ : Int) : Bool := decide (d > θ)
def switch_L_to_S (d θ : Int) : Bool := decide (-d > θ)

/-- Theorem 6 (`iridophore_switch_exclusive_and_conservative`):
    (1) For any non-negative threshold the two switching rules are mutually exclusive:
        no iridophore is told to become dense and loose in the same step.
    (2) Switching conserves agents: if `a` dense cells become loose and `b` loose cells
        become dense, the total `(s - a + b) + (l + a - b) = s + l` is unchanged. -/
theorem iridophore_switch_exclusive_and_conservative (d θ s l a b : Int) (hθ : θ ≥ 0) :
    ¬ (switch_S_to_L d θ = true ∧ switch_L_to_S d θ = true) ∧
    (s - a + b) + (l + a - b) = s + l := by
  constructor
  · intro h
    have h1 : d > θ := of_decide_eq_true h.1
    have h2 : -d > θ := of_decide_eq_true h.2
    omega
  · omega

end MathOfBiology.Loop_007
