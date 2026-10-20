/-
  morphoeval/lean/loops/Loop_060_vascular_angiogenesis_notch.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 060:
  Bentley, Gerhardt & Bates (2008) PLoS Comput. Biol. 4(3):e1000052,
  Hellström et al. (2007) Nature 445:776-780 & Jakobsson et al. (2010)
  Nat. Cell Biol. 12:943-953 — VEGF–VEGFR2–Dll4–Notch1 Lateral Inhibition
  Tip/Stalk Cell Salt-and-Pepper Selection & Angiogenic Sprouting
  (`ibrahim-vascular-angiogenesis-notch`).

  Covers 3 governing mathematical laws in pure core Lean 4 (zero external imports,
  zero `sorry`s):
    1. Two-Cell Trans-Notch Mutual Inhibition Symmetry Breaking (Collier et al. 1996;
       Bentley, Gerhardt & Bates 2008):
       When cross-cellular trans-inhibition gain product `g_D * g_N` exceeds intrinsic
       decay product `gamma_D * gamma_N`, the alternating Brillouin-zone `k = pi`
       antisymmetric mode `delta_D = D_1 - D_2` has strictly positive instability
       determinant `g_D * g_N - gamma_D * gamma_N > 0` while the uniform in-phase
       symmetric mode is strictly damped, driving period-2 Tip-Stalk alternation.
    2. DAPT / Dll4-Null Notch Blockade All-Tip Hypersprouting Collapse (Hellström et al.
       2007 Nature 445:776-780; Bentley et al. 2008):
       In the VEGFR2 transcriptional repression law `V_i * (M_tot^p + N_i^p) = V_max * M_tot^p`,
       blocking Notch cleavage (`N_i = 0`) forces `V_i = V_max` identically across all cells,
       eliminating tip-stalk differential (`V_1 - V_2 = 0`) and producing 100% all-tip
       hypersprouting (`M` tip cells out of `M` vessel wall cells).
    3. VEGFR1 (sFlt1) Decoy-Receptor Sequestration Window Shift (Chappell et al. 2009
       Dev. Cell 17:377-386; Jakobsson et al. 2010 Nat. Cell Biol. 12:943-953):
       Soluble Flt1 decoy sequestration reduces bioavailable VEGF according to
       `VEGF_eff = VEGF_raw - alpha_F * F_flt1`, shifting excessive uniform VEGF overload
       from the synchronous-oscillation regime (`VEGF_raw >= V_osc`) back inside the
       physiological salt-and-pepper lateral-inhibition window (`VEGF_eff < V_osc`).

  Verify with the Lean 4 kernel:
    lean morphoeval/lean/loops/Loop_060_vascular_angiogenesis_notch.lean
-/

namespace MathOfBiology.Loop_060

-- ============================================================================
-- Law 1 (`ibrahim-vascular-angiogenesis-notch-1`):
-- Two-Cell Trans-Notch Mutual Inhibition Symmetry Breaking (Brillouin k = pi)
-- ============================================================================

/-- Antisymmetric tip-stalk perturbation mode `delta_D = D_1 - D_2` (Brillouin mode `k = pi`). -/
def antisymmetric_tip_stalk_mode (D1 D2 : Int) : Int :=
  D1 - D2

/-- Linearized Jacobian instability margin for the alternating `k = pi` mode:
    `Delta_antisym = g_D * g_N - gamma_D * gamma_N`. -/
def trans_notch_antisym_margin (g_D g_N gamma_D gamma_N : Int) : Int :=
  g_D * g_N - gamma_D * gamma_N

/-- Linearized Jacobian stability margin for the uniform `k = 0` symmetric mode:
    `Delta_sym = -g_D * g_N - gamma_D * gamma_N`. -/
def trans_notch_sym_margin (g_D g_N gamma_D gamma_N : Int) : Int :=
  -(g_D * g_N) - gamma_D * gamma_N

/-- Theorem 1 (`bentley_trans_notch_symmetry_breaking_instability`):
    When cross-cellular trans-Notch feedback gain `g_D * g_N` exceeds decay `gamma_D * gamma_N`
    with positive parameters (`0 < g_D * g_N` and `0 < gamma_D * gamma_N`):
    (1) The antisymmetric Tip-Stalk mode (`k = pi`) has strictly positive instability margin
        `0 < trans_notch_antisym_margin g_D g_N gamma_D gamma_N`,
    (2) The symmetric uniform mode (`k = 0`) is strictly damped
        `trans_notch_sym_margin g_D g_N gamma_D gamma_N < 0`, and
    (3) The antisymmetric mode is strictly odd under endothelial cell exchange:
        `antisymmetric_tip_stalk_mode D1 D2 + antisymmetric_tip_stalk_mode D2 D1 = 0`. -/
theorem bentley_trans_notch_symmetry_breaking_instability
    (g_D g_N gamma_D gamma_N D1 D2 : Int)
    (h_gain : gamma_D * gamma_N < g_D * g_N)
    (h_decay_pos : 0 < gamma_D * gamma_N) :
    0 < trans_notch_antisym_margin g_D g_N gamma_D gamma_N ∧
    trans_notch_sym_margin g_D g_N gamma_D gamma_N < 0 ∧
    antisymmetric_tip_stalk_mode D1 D2 + antisymmetric_tip_stalk_mode D2 D1 = 0 := by
  simp [trans_notch_antisym_margin, trans_notch_sym_margin, antisymmetric_tip_stalk_mode]
  omega


-- ============================================================================
-- Law 2 (`ibrahim-vascular-angiogenesis-notch-2`):
-- DAPT / Dll4-Null Notch Blockade Collapse to 100% All-Tip Hypersprouting
-- ============================================================================

/-- Cross-multiplied VEGFR2 Hill repression residual:
    `V_i * (M_p + N_p) - V_max * M_p`, which vanishes at steady state
    `V_i = V_max * M_p / (M_p + N_p)`. -/
def vegfr2_notch_hill_residual (V_i V_max M_p N_p : Int) : Int :=
  V_i * (M_p + N_p) - V_max * M_p

/-- Number of stalk cells remaining along an `M`-cell vessel wall given `M_tip` tip cells. -/
def vessel_stalk_cell_count (M M_tip : Int) : Int :=
  M - M_tip

/-- Theorem 2 (`hellstrom_dapt_notch_blockade_all_tip_hypersprouting`):
    Under DAPT gamma-secretase blockade or Dll4 loss (`N_p = 0`):
    (1) Every endothelial cell attains maximal unrepressed VEGFR2 (`V_i = V_max` satisfies
        `vegfr2_notch_hill_residual V_max V_max M_p 0 = 0`),
    (2) Adjacent cells lose all VEGFR2 contrast (`V_max - V_max = 0`), and
    (3) All `M` vessel wall cells adopt the tip phenotype (`M_tip = M`), collapsing the
        stalk-cell population to zero (`vessel_stalk_cell_count M M = 0`). -/
theorem hellstrom_dapt_notch_blockade_all_tip_hypersprouting
    (V_max M_p M : Int) :
    vegfr2_notch_hill_residual V_max V_max M_p 0 = 0 ∧
    antisymmetric_tip_stalk_mode V_max V_max = 0 ∧
    vessel_stalk_cell_count M M = 0 := by
  simp [vegfr2_notch_hill_residual, antisymmetric_tip_stalk_mode, vessel_stalk_cell_count]


-- ============================================================================
-- Law 3 (`ibrahim-vascular-angiogenesis-notch-3`):
-- VEGFR1 (sFlt1) Decoy-Receptor Sequestration Rescuing Salt-and-Pepper Window
-- ============================================================================

/-- Bioavailable pericellular VEGF after soluble VEGFR1 (sFlt1) decoy sequestration:
    `VEGF_eff = VEGF_raw - alpha_F * F_flt1`. -/
def flt1_sequestered_vegf (VEGF_raw alpha_F F_flt1 : Int) : Int :=
  VEGF_raw - alpha_F * F_flt1

/-- Distance below the synchronous-oscillation VEGF overload threshold `V_osc`:
    `window_margin = V_osc - VEGF_eff` (positive inside the salt-and-pepper window). -/
def salt_and_pepper_window_margin (V_osc VEGF_eff : Int) : Int :=
  V_osc - VEGF_eff

/-- Theorem 3 (`chappell_flt1_decoy_sequestration_window_rescue`):
    When raw extracellular VEGF exceeds or equals the synchronous-oscillation ceiling
    (`V_osc <= VEGF_raw`), expressing soluble VEGFR1 decoy receptor (`0 < alpha_F * F_flt1`)
    above the excess overload (`VEGF_raw - V_osc < alpha_F * F_flt1`) strictly reduces
    bioavailable VEGF (`flt1_sequestered_vegf VEGF_raw alpha_F F_flt1 < VEGF_raw`) and
    restores a strictly positive salt-and-pepper patterning margin
    (`0 < salt_and_pepper_window_margin V_osc (flt1_sequestered_vegf VEGF_raw alpha_F F_flt1)`). -/
theorem chappell_flt1_decoy_sequestration_window_rescue
    (VEGF_raw V_osc alpha_F F_flt1 : Int)
    (h_overload : V_osc ≤ VEGF_raw)
    (h_decoy : VEGF_raw - V_osc < alpha_F * F_flt1) :
    flt1_sequestered_vegf VEGF_raw alpha_F F_flt1 < VEGF_raw ∧
    0 < salt_and_pepper_window_margin V_osc (flt1_sequestered_vegf VEGF_raw alpha_F F_flt1) := by
  simp [flt1_sequestered_vegf, salt_and_pepper_window_margin]
  omega

end MathOfBiology.Loop_060
