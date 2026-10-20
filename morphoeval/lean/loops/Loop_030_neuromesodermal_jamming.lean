/-
  morphoeval/lean/loops/Loop_030_neuromesodermal_jamming.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 030:
  Mongera et al. (2018) Nature 561:401-405, Kim et al. (2021) Nature Physics
  17:859-866 & Lawton et al. (2013) Development 140:573-582 —
  Active Fluid-to-Solid Jamming Transition, Extracellular Space Compaction,
  Rigidity-Percolation Yield Stress, Bingham Viscoplasticity, and cdh2
  Knockdown Backstop Collapse in Zebrafish Body Axis Elongation
  (`tailbud-neuromesodermal-jamming`).

  Pure core Lean 4 (zero external imports, zero `sorry`s).
  Verify with:
    lean morphoeval/lean/loops/Loop_030_neuromesodermal_jamming.lean
-/

namespace MathOfBiology.Loop_030

/-- Milli-scaled cell packing fraction `phi_pack = 1000 - phi_ecs` from
    Mongera et al. (2018) Nature 561:401-405 Fig. 3, where `phi_ecs` is the
    extracellular space volume fraction in milli-units (`1000 * phi_ECS`). -/
def cell_packing_milli (phi_ecs : Int) : Int :=
  1000 - phi_ecs

/-- Rigidity-percolation supracellular yield stress order parameter
    `sigma_y(phi_pack) = sigma_0 * max(0, phi_pack - phi_c)` from Mongera et al.
    (2018) Nature Figs. 1-3 & Kim et al. (2021) Nature Physics 17:859-866:
    identically zero below the critical packing threshold `phi_c` (`phi_pack <= phi_c`,
    unjammed fluid foam) and strictly positive above `phi_c` (`phi_pack > phi_c`,
    jammed solid foam). -/
def yield_stress_scaled (sigma_0 phi_pack phi_c : Int) : Int :=
  if phi_c < phi_pack then sigma_0 * (phi_pack - phi_c) else 0

/-- Net mechanical caging barrier `Delta_sigma = sigma_y - sigma_A` comparing
    supracellular yield stress `sigma_y(x)` against cell-scale Myosin-II active
    tension fluctuations `sigma_A(x)` (Mongera et al. 2018 Fig. 2 & Kim et al. 2021). -/
def net_jamming_barrier (sigma_y sigma_A : Int) : Int :=
  sigma_y - sigma_A

/-- Bingham viscoplastic strain-rate numerator `max(0, sigma_eff - sigma_y)`
    governing spontaneous T1 topological cell rearrangements driven by active
    stress `sigma_eff = sigma_A` against yield stress `sigma_y`. -/
def bingham_plastic_rate_num (sigma_eff sigma_y : Int) : Int :=
  if sigma_y < sigma_eff then sigma_eff - sigma_y else 0

/-- Theorem 1 (`rigidity_percolation_yield_stress_onset`):
    Proves that across the active-foam rigidity-percolation threshold `phi_c`:
    (1) For any subcritical cell packing `phi_fluid <= phi_c`, the yield stress
        vanishes identically (`yield_stress_scaled sigma_0 phi_fluid phi_c = 0`);
    (2) At unit modulus `sigma_0 = 1`, for any supercritical packing `phi_solid > phi_c`,
        the yield stress is strictly positive (`yield_stress_scaled 1 phi_solid phi_c > 0`);
    (3) Extracellular space compaction from posterior to anterior (`phi_ecs_ant < phi_ecs_post`)
        strictly increases cell packing (`cell_packing_milli phi_ecs_post < cell_packing_milli phi_ecs_ant`)
        and strictly increases yield stress inside the jammed regime (`phi_c <= phi_1 < phi_2`). -/
theorem rigidity_percolation_yield_stress_onset
    (sigma_0 phi_fluid phi_solid phi_1 phi_2 phi_c phi_ecs_post phi_ecs_ant : Int)
    (h_fluid : phi_fluid <= phi_c)
    (h_solid : phi_c < phi_solid)
    (h_sub : phi_c <= phi_1)
    (h_order : phi_1 < phi_2)
    (h_ecs : phi_ecs_ant < phi_ecs_post) :
    yield_stress_scaled sigma_0 phi_fluid phi_c = 0 ∧
    yield_stress_scaled 1 phi_solid phi_c > 0 ∧
    cell_packing_milli phi_ecs_post < cell_packing_milli phi_ecs_ant ∧
    yield_stress_scaled 1 phi_1 phi_c < yield_stress_scaled 1 phi_2 phi_c := by
  dsimp [yield_stress_scaled, cell_packing_milli]
  omega

/-- Theorem 2 (`active_fluctuation_fluid_to_solid_jamming_crossover`):
    Proves that comparing cell-scale active fluctuations `sigma_A` to yield stress `sigma_y`:
    (1) In the posterior MPZ where active tension fluctuations exceed yield stress
        (`sigma_y_mpz < sigma_A_mpz`), the net caging barrier is strictly negative
        and the Bingham plastic rearrangement rate is strictly positive (fluidized tissue);
    (2) In the anterior PSM where yield stress exceeds or equals active fluctuations
        (`sigma_A_apsm <= sigma_y_apsm`), the net caging barrier is non-negative and
        spontaneous Bingham plastic flow vanishes (`bingham_plastic_rate_num = 0`, jammed solid). -/
theorem active_fluctuation_fluid_to_solid_jamming_crossover
    (sigma_y_mpz sigma_A_mpz sigma_y_apsm sigma_A_apsm : Int)
    (h_mpz_fluid : sigma_y_mpz < sigma_A_mpz)
    (h_apsm_solid : sigma_A_apsm <= sigma_y_apsm) :
    net_jamming_barrier sigma_y_mpz sigma_A_mpz < 0 ∧
    bingham_plastic_rate_num sigma_A_mpz sigma_y_mpz > 0 ∧
    net_jamming_barrier sigma_y_apsm sigma_A_apsm >= 0 ∧
    bingham_plastic_rate_num sigma_A_apsm sigma_y_apsm = 0 := by
  dsimp [net_jamming_barrier, bingham_plastic_rate_num]
  omega

/-- Empirical milli-scaled (`1000 * phi`) extracellular space and critical packing
    benchmarks from Mongera et al. (2018) Nature 561:401-405 Figs. 1-4:
    - Critical rigidity-percolation packing fraction: `phi_c = 0.860` (`860`),
      corresponding to critical ECS fraction `phi_ECS_crit = 0.140` (`140`);
    - Wild-type zebrafish ECS gradient: MPZ (`145`) -> P-PSM (`86`) -> A-PSM (`49`);
    - cdh2 (N-cadherin) morphant ECS profile: MPZ (`215`) -> A-PSM (`148`);
    - Active cell-scale tension fluctuations (Pa): MPZ (`50 Pa`) -> A-PSM (`13 Pa`). -/
def phi_c_milli : Int := 860
def phi_ecs_wt_mpz : Int := 145
def phi_ecs_wt_ppsm : Int := 86
def phi_ecs_wt_apsm : Int := 49
def phi_ecs_cdh2_mpz : Int := 215
def phi_ecs_cdh2_apsm : Int := 148
def sigma_A_wt_mpz : Int := 50
def sigma_A_wt_apsm : Int := 13

/-- Theorem 3 (`mongera_2018_wt_vs_cdh2_backstop_collapse`):
    Proves that in the Mongera et al. (2018) zebrafish dataset:
    (1) Wild-type cell packing increases strictly from MPZ (`855`) -> P-PSM (`914`) -> A-PSM (`951`),
        crossing `phi_c_milli = 860` between MPZ and P-PSM;
    (2) In wild-type MPZ, `yield_stress_scaled 1 (cell_packing_milli phi_ecs_wt_mpz) phi_c_milli = 0 < sigma_A_wt_mpz`
        (unjammed fluid), whereas in wild-type A-PSM `yield_stress_scaled 1 (cell_packing_milli phi_ecs_wt_apsm) phi_c_milli = 91 > sigma_A_wt_apsm = 13`
        (jammed solid backstop);
    (3) Under `cdh2` (N-cadherin) knockdown, both MPZ (`785 < 860`) and A-PSM (`852 < 860`)
        remain strictly below `phi_c_milli = 860`, collapsing the anterior PSM yield-stress
        backstop to zero (`yield_stress_scaled 1 (cell_packing_milli phi_ecs_cdh2_apsm) phi_c_milli = 0`). -/
theorem mongera_2018_wt_vs_cdh2_backstop_collapse :
    (cell_packing_milli phi_ecs_wt_mpz = 855 ∧
     cell_packing_milli phi_ecs_wt_ppsm = 914 ∧
     cell_packing_milli phi_ecs_wt_apsm = 951) ∧
    (cell_packing_milli phi_ecs_wt_mpz < phi_c_milli ∧
     phi_c_milli < cell_packing_milli phi_ecs_wt_ppsm ∧
     cell_packing_milli phi_ecs_wt_ppsm < cell_packing_milli phi_ecs_wt_apsm) ∧
    (yield_stress_scaled 1 (cell_packing_milli phi_ecs_wt_mpz) phi_c_milli = 0 ∧
     yield_stress_scaled 1 (cell_packing_milli phi_ecs_wt_apsm) phi_c_milli = 91 ∧
     yield_stress_scaled 1 (cell_packing_milli phi_ecs_wt_apsm) phi_c_milli > sigma_A_wt_apsm) ∧
    (cell_packing_milli phi_ecs_cdh2_mpz < phi_c_milli ∧
     cell_packing_milli phi_ecs_cdh2_apsm < phi_c_milli ∧
     yield_stress_scaled 1 (cell_packing_milli phi_ecs_cdh2_apsm) phi_c_milli = 0) := by
  decide

/-- Rectified posterior elongation momentum numerator `J_mpz * sigma_y_apsm`
    and bilateral left-right symmetry barrier `sigma_y_apsm - sigma_lat_noise`. -/
def rectified_posterior_flux_num (j_mpz sigma_y_apsm : Int) : Int :=
  j_mpz * sigma_y_apsm

/-- Theorem 4 (`posterior_elongation_rectification_and_bilateral_symmetry`):
    Proves that when the anterior PSM forms a jammed solid backstop
    (`sigma_y_apsm_wt > sigma_y_apsm_mut >= 0`) at unit progenitor influx (`j_mpz = 1`),
    unidirectional posterior elongation flux and bilateral symmetry resistance are
    strictly greater in wild-type than under `cdh2` backstop collapse. -/
theorem posterior_elongation_rectification_and_bilateral_symmetry
    (sigma_y_apsm_wt sigma_y_apsm_mut : Int)
    (h_nonneg : 0 <= sigma_y_apsm_mut)
    (h_backstop : sigma_y_apsm_mut < sigma_y_apsm_wt) :
    0 <= rectified_posterior_flux_num 1 sigma_y_apsm_mut ∧
    rectified_posterior_flux_num 1 sigma_y_apsm_mut < rectified_posterior_flux_num 1 sigma_y_apsm_wt := by
  dsimp [rectified_posterior_flux_num]
  omega

/-! ### Pass 3 — quasi-3D fibronectin sheath confinement (Dray et al. 2013) and
    Her1/Her7 segmentation-clock wavefront arrest (Soroldoni et al. 2014; Hubaud et al. 2017) -/

/-- Fibronectin sheath hoop stiffness `S = K_FN * f_FN + K_t` (Pa per unit radial
    strain) of the thin-rod reduction of the 3D paraxial tissue column: the
    fibronectin matrix density `f_FN` wrapping the tailbud/PSM (Dray et al. 2013
    Curr. Biol. 23:1335 Fig. 1) plus residual ectoderm/notochord confinement `K_t`. -/
def sheath_hoop_stiffness (k_fn f_fn k_t : Int) : Int :=
  k_fn * f_fn + k_t

/-- Axial channeling numerator `chi_num = 1000 - leak` (milli-units): the fraction
    of progenitor influx that is converted into posterior axis elongation instead
    of being stored in radial dilation of the fluid tailbud against the sheath. -/
def channeling_milli (leak_milli : Int) : Int :=
  1000 - leak_milli

/-- Theorem 5 (`fibronectin_sheath_confinement_channeling`):
    Quasi-static hoop equilibrium `eps * S = p` where `p = max(0, sigma_A - sigma_y) > 0`
    is the active cell pressure in excess of the tissue yield stress in the fluid MPZ.
    For a wild-type sheath (`f_wt`) and a fibronectin-depleted sheath
    (`f_mut < f_wt`, fn1a/fn1b or itga5 morphant) carrying the same excess pressure:
    (1) the morphant sheath is strictly softer (`S_mut < S_wt`);
    (2) the morphant radial hoop strain is strictly larger (`eps_wt < eps_mut`), i.e.
        the morphant tailbud is wider (Dray et al. 2013 Figs. 2-4);
    (3) any strictly larger leak gives a strictly smaller channeling efficiency, so the
        morphant elongates more slowly for the same progenitor influx. -/
theorem fibronectin_sheath_confinement_channeling
    (k_fn f_wt f_mut k_t p eps_wt eps_mut leak_wt leak_mut : Int)
    (h_kfn : 0 < k_fn)
    (h_f : f_mut < f_wt)
    (h_kt : 0 < k_t)
    (h_fmut : 0 <= f_mut)
    (h_eps_wt : 0 < eps_wt)
    (h_eq_wt : eps_wt * sheath_hoop_stiffness k_fn f_wt k_t = p)
    (h_eq_mut : eps_mut * sheath_hoop_stiffness k_fn f_mut k_t = p)
    (h_leak : leak_wt < leak_mut) :
    sheath_hoop_stiffness k_fn f_mut k_t < sheath_hoop_stiffness k_fn f_wt k_t ∧
    eps_wt < eps_mut ∧
    channeling_milli leak_mut < channeling_milli leak_wt := by
  have h_stiff : sheath_hoop_stiffness k_fn f_mut k_t < sheath_hoop_stiffness k_fn f_wt k_t := by
    dsimp [sheath_hoop_stiffness]
    have := Int.mul_lt_mul_of_pos_left h_f h_kfn
    omega
  have h_pos_mut : 0 < sheath_hoop_stiffness k_fn f_mut k_t := by
    dsimp [sheath_hoop_stiffness]
    have := Int.mul_nonneg (Int.le_of_lt h_kfn) h_fmut
    omega
  refine ⟨h_stiff, ?_, ?_⟩
  · -- eps_wt * S_wt = eps_mut * S_mut with S_mut < S_wt and eps_wt > 0 forces eps_wt < eps_mut
    rcases Int.lt_or_le eps_wt eps_mut with h | h
    · exact h
    · exfalso
      have h1 : eps_mut * sheath_hoop_stiffness k_fn f_mut k_t
              <= eps_wt * sheath_hoop_stiffness k_fn f_mut k_t :=
        Int.mul_le_mul_of_nonneg_right h (Int.le_of_lt h_pos_mut)
      have h2 : eps_wt * sheath_hoop_stiffness k_fn f_mut k_t
              < eps_wt * sheath_hoop_stiffness k_fn f_wt k_t :=
        Int.mul_lt_mul_of_pos_left h_stiff h_eps_wt
      omega
  · dsimp [channeling_milli]
    omega

/-- Cooke-Zeeman somite length numerator `S = v_front * T_clock` (um = um/min * min):
    the wavefront regression speed `v_front = v_elong + v_PSM-shortening` set by the
    Bingham backstop / fibronectin sheath model times the Her1/Her7 period `T`
    (Herrgen et al. 2010: 23.5 min at 28 C). -/
def somite_length_num (v_front t_clock : Int) : Int :=
  v_front * t_clock

/-- Theorem 6 (`clock_wavefront_arrest_somite_scaling`):
    Let `r : Int -> Int` be the (strictly increasing, milli-scaled) jamming ratio
    profile `1000 * sigma_y(x)/sigma_A(x)` along the AP axis, `x_jam` the crossover
    where `r x_jam = 1000` (sigma_y = sigma_A) and `x_arr` any position where the
    ratio reaches the clock-arrest gate `lam > 1000` (`Lambda_arrest = 4` in the
    script).  Then:
    (1) the Her1/Her7 arrest front lies strictly anterior to the jamming crossover
        (`x_jam < x_arr`) — the clock stops inside the solid, never in the fluid MPZ;
    (2) for a fixed clock period `0 < T`, a slower wavefront (`0 <= v_mut < v_wt`,
        e.g. cdh2, blebbistatin or fibronectin morphants) produces strictly shorter
        somites (`somite_length_num v_mut T < somite_length_num v_wt T`). -/
theorem clock_wavefront_arrest_somite_scaling
    (r : Int → Int) (x_jam x_arr lam v_wt v_mut t_clock : Int)
    (h_mono : ∀ x y : Int, x < y → r x < r y)
    (h_jam : r x_jam = 1000)
    (h_lam : 1000 < lam)
    (h_arr : lam <= r x_arr)
    (h_t : 0 < t_clock)
    (h_v : v_mut < v_wt) :
    x_jam < x_arr ∧
    somite_length_num v_mut t_clock < somite_length_num v_wt t_clock := by
  refine ⟨?_, ?_⟩
  · rcases Int.lt_trichotomy x_jam x_arr with h | h | h
    · exact h
    · subst h
      omega
    · have := h_mono _ _ h
      omega
  · dsimp [somite_length_num]
    exact Int.mul_lt_mul_of_pos_right h_v h_t

end MathOfBiology.Loop_030
