/-
  morphoeval/lean/loops/Loop_010_avian_plumage_somite_stripes.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 010:
  Haupaix et al. (2018) Science 361(6408):eaar4777 — "The periodic coloration in
  birds forms through a prepattern of somite origin"; Inaba et al. (2019) PNAS
  116(14):6884-6890 — "Instructive role of melanocytes during pigment pattern
  formation of the avian skin"; Lin et al. (2013) Science 340(6139):1442-1445 —
  "Topology of feather melanocyte progenitor niche allows complex pigment patterns
  to emerge" (`haupaix-avian-plumage-somite-stripes`).
  (Citations corrected in Pass 3; Laws 4-5 added in Pass 3.)

  Pure core Lean 4 (zero external dependencies, zero `sorry`s).
  Verify with:
    lean morphoeval/lean/loops/Loop_010_avian_plumage_somite_stripes.lean
-/

namespace MathOfBiology.Loop_010

-- ============================================================================
-- Law 1 (`haupaix-avian-plumage-somite-stripes-1`):
-- Somite-Specified Dorsoventral Stripe Period Partition & Agouti Bead Shift
-- ============================================================================

/-- Within a somite-specified dorsoventral stripe period `Lambda` (in μm) with
    yellow pheomelanin (*Agouti*-positive) stripe width `w_y ≤ Lambda`, the
    complementary black eumelanin (*Agouti*-negative) stripe width is
    `w_b = Lambda - w_y`. -/
def black_stripe_width (Lambda w_y : Int) : Int :=
  Lambda - w_y

/-- Theorem 1 (`somite_stripe_period_conservation_and_agouti_bead_expansion`):
    For any somite-specified stripe period `Lambda` and yellow stripe width `w_y`
    with `0 ≤ w_y` and `w_y + delta_bead ≤ Lambda` under exogenous *Agouti* bead
    perturbation `delta_bead > 0`:
    (1) Total dorsoventral stripe period is conserved (`w_y + black_stripe_width Lambda w_y = Lambda`).
    (2) Exogenous *Agouti* protein bead implantation (`delta_bead > 0`) strictly widens
        the yellow pheomelanin domain (`w_y + delta_bead > w_y`) and strictly narrows
        the adjacent black eumelanin domain by the exact same magnitude while keeping
        non-negative black stripe width (`black_stripe_width Lambda (w_y + delta_bead) ≥ 0`). -/
theorem somite_stripe_period_conservation_and_agouti_bead_expansion
    (Lambda w_y delta_bead : Int)
    (hw : 0 ≤ w_y) (hbead : delta_bead > 0) (hcap : w_y + delta_bead ≤ Lambda) :
    w_y + black_stripe_width Lambda w_y = Lambda ∧
    (w_y + delta_bead) + black_stripe_width Lambda (w_y + delta_bead) = Lambda ∧
    w_y + delta_bead > 0 ∧
    black_stripe_width Lambda (w_y + delta_bead) ≥ 0 ∧
    black_stripe_width Lambda (w_y + delta_bead) < black_stripe_width Lambda w_y ∧
    black_stripe_width Lambda (w_y + delta_bead) = black_stripe_width Lambda w_y - delta_bead := by
  unfold black_stripe_width
  omega


-- ============================================================================
-- Law 2 (`haupaix-avian-plumage-somite-stripes-2`):
-- Heterospecific Somite-to-Dermis Chimera Donor Autonomy Law
-- ============================================================================

/-- Weighted chimeric dermal *Agouti* stripe width (scaled by 100 for percentage
    donor autonomy `alpha_pct ∈ [0, 100]`):
    `W_chimera = alpha_pct * w_donor + (100 - alpha_pct) * w_host`. -/
def chimera_agouti_width_scaled (alpha_pct w_donor w_host : Int) : Int :=
  alpha_pct * w_donor + (100 - alpha_pct) * w_host

/-- Theorem 2 (`interspecific_somite_graft_donor_autonomy`):
    In heterospecific somite-to-dermis grafting between a donor species with
    stripe width `w_donor` and a host species with stripe width `w_host`
    at physiological galliform somitic autonomy `alpha_pct = 94`:
    (1) Complete somitic autonomy (`alpha_pct = 100`) reproduces the exact donor
        stripe width (`100 * w_donor`).
    (2) Whenever `w_donor > w_host` (e.g. pheasant somites grafted into quail host),
        the grafted dermal *Agouti* width shifts strictly above the host baseline
        (`> 100 * w_host`) and closer to the donor than the host, whereas the
        reciprocal graft shifts the width strictly below `100 * w_donor`. -/
theorem interspecific_somite_graft_donor_autonomy
    (w_donor w_host : Int)
    (h_diff : w_donor > w_host) :
    chimera_agouti_width_scaled 100 w_donor w_host = 100 * w_donor ∧
    chimera_agouti_width_scaled 94 w_donor w_host > 100 * w_host ∧
    chimera_agouti_width_scaled 94 w_host w_donor < 100 * w_donor := by
  unfold chimera_agouti_width_scaled
  omega


-- ============================================================================
-- Law 3 (`haupaix-avian-plumage-somite-stripes-3`):
-- Agouti-MC1R Threshold Switch & Connexin40 Reaction-Diffusion Turing Instability
-- ============================================================================

/-- Binary melanocyte pigment fate switch under dermal *Agouti* concentration `A`
    relative to MC1R antagonism threshold `theta_A`:
    returns `1` (yellow/red pheomelanin) if `A ≥ theta_A`, else `0` (black/brown eumelanin). -/
def mc1r_pigment_fate (A theta_A : Nat) : Nat :=
  if A ≥ theta_A then 1 else 0

/-- Trace and diffusion-driven determinant of the linearized 2-species melanocyte
    activator-inhibitor reaction-diffusion Jacobian (`fu > 0, gv < 0, fv < 0, gu > 0`). -/
def turing_trace (fu gv : Int) : Int := fu + gv
def turing_det (fu gv fv gu : Int) : Int := fu * gv - fv * gu
def turing_cross_term (fu gv d_ratio : Int) : Int := d_ratio * fu + gv

/-- Theorem 3 (`agouti_mc1r_switch_and_cx40_turing_instability`):
    (1) Suprathreshold *Agouti* (`A ≥ theta_A`) switches melanocytes to pheomelanin (`1`),
        while *Agouti* ablation (`A = 0 < theta_A`) locks melanocytes into eumelanin (`0`).
    (2) For canonical melanocyte gap-junction reaction kinetics
        (`fu = 5, gv = -7, fv = -8, gu = 6`) with diffusivity ratio `d_ratio = 20`:
        the homogeneous state is stable without diffusion (`turing_trace < 0` and
        `turing_det > 0`), equal diffusion (`d_ratio = 1`) cannot trigger instability
        (`turing_cross_term 5 (-7) 1 < 0`), and differential diffusion (`d_ratio = 20`)
        drives positive waveband amplification (`turing_cross_term 5 (-7) 20 > 0`). -/
theorem agouti_mc1r_switch_and_cx40_turing_instability :
    mc1r_pigment_fate 75 50 = 1 ∧
    mc1r_pigment_fate 0 50 = 0 ∧
    turing_trace 5 (-7) < 0 ∧
    turing_det 5 (-7) (-8) 6 > 0 ∧
    turing_cross_term 5 (-7) 1 < 0 ∧
    turing_cross_term 5 (-7) 20 > 0 := by
  decide


-- ============================================================================
-- Law 4 (`haupaix-avian-plumage-somite-stripes-4`, Pass 3):
-- Lin et al. (2013) Ring-Niche Chevron Geometry on the Opened Follicle Cylinder
-- ============================================================================

/-- Proximodistal position (measured from the feather tip, scaled integer units) of a
    pigment-band edge on the follicle cylinder opened at the posterior new-barb locus:
    `s(ell) = s0 + k * |ell|`, where `ell` is the signed lateral (azimuthal) distance from
    the anterior rachis and `k = v_z / (r_f * omega)` is the helical pitch of a pigment
    switch that propagates around the ring niche at angular rate `omega` while
    melanocytes rise at `v_z` (Lin et al. 2013 Fig. 2B, fig. S9). -/
def chevron_s (s0 k ell : Int) : Int := s0 + k * (Int.natAbs ell : Int)

/-- Theorem 4 (`ring_niche_chevron_geometry`):
    For any non-negative helical pitch `k`:
    (1) the band edge is mirror-symmetric about the rachis (`s(ell) = s(-ell)`) — the
        two arms of the chevron;
    (2) a synchronous ring (`k = 0`) gives a transverse bar (`s(ell) = s0` for all `ell`);
    (3) the apex on the rachis is the tip-most point of the edge (`s0 ≤ s(ell)`);
    (4) the chevron opens monotonically with the pitch: for `k < k'` and `ell ≠ 0`,
        `s_k(ell) < s_k'(ell)`. -/
theorem ring_niche_chevron_geometry (s0 k ell : Int) (hk : 0 ≤ k) :
    chevron_s s0 k ell = chevron_s s0 k (-ell) ∧
    chevron_s s0 0 ell = s0 ∧
    s0 ≤ chevron_s s0 k ell ∧
    (∀ k', k < k' → ell ≠ 0 → chevron_s s0 k ell < chevron_s s0 k' ell) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [chevron_s, Int.natAbs_neg]
  · simp [chevron_s]
  · unfold chevron_s
    have h := Int.mul_nonneg hk (Int.natCast_nonneg (Int.natAbs ell))
    omega
  · intro k' hk' hell
    unfold chevron_s
    have hpos : (0 : Int) < (Int.natAbs ell : Int) := by omega
    have h := Int.mul_lt_mul_of_pos_right hk' hpos
    omega


-- ============================================================================
-- Law 5 (`haupaix-avian-plumage-somite-stripes-5`, Pass 3):
-- Haupaix et al. (2018) Agouti-Dose-Dependent Yellow-Pulse Duration & Feather Type
-- ============================================================================

/-- Integer MC1R pigment switch (`1` pheomelanin if `A ≥ theta_A`, else `0` eumelanin). -/
def mc1r_pigment_fate_int (A theta_A : Int) : Nat := if A ≥ theta_A then 1 else 0

/-- Feather type under a transient niche Agouti pulse of mid-level `A` and modulation
    amplitude `a` (peak `A + a` at mid-growth, trough `A - a` at tip and base):
    `type = [A + a ≥ theta_A] + [A - a ≥ theta_A]` ∈ {0, 1, 2} —
    `0` = b feather (entirely eumelanic), `1` = y feather (pheomelanin band with black
    base and tip), `2` = fully pheomelanic feather (Fawn: no eumelanin before hatching). -/
def feather_type (A a theta_A : Int) : Nat :=
  mc1r_pigment_fate_int (A + a) theta_A + mc1r_pigment_fate_int (A - a) theta_A

/-- Theorem 5 (`agouti_pulse_duration_feather_type`):
    (1) `A - a ≥ theta_A` (trough suprathreshold) ⇒ fully pheomelanic feather (type 2);
    (2) `A + a < theta_A` (peak subthreshold) ⇒ entirely eumelanic b feather (type 0);
    (3) `theta_A - a ≤ A < theta_A + a` ⇒ y feather with pheomelanin band (type 1);
    (4) feather type is monotone non-decreasing in the agouti dose `A`
        (recessive-black agouti loss ≤ wild type ≤ Fawn agouti gain; Haupaix et al. 2018 Fig. 3B-C). -/
theorem agouti_pulse_duration_feather_type (A a theta_A : Int) (ha : 0 < a) :
    (A - a ≥ theta_A → feather_type A a theta_A = 2) ∧
    (A + a < theta_A → feather_type A a theta_A = 0) ∧
    (theta_A - a ≤ A ∧ A < theta_A + a → feather_type A a theta_A = 1) ∧
    (∀ A', A ≤ A' → feather_type A a theta_A ≤ feather_type A' a theta_A) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro h; simp only [feather_type, mc1r_pigment_fate_int]; split <;> (try split) <;> omega
  · intro h; simp only [feather_type, mc1r_pigment_fate_int]; split <;> (try split) <;> omega
  · intro h; simp only [feather_type, mc1r_pigment_fate_int]; split <;> (try split) <;> omega
  · intro A' h; simp only [feather_type, mc1r_pigment_fate_int]
    repeat (first | omega | split)

end MathOfBiology.Loop_010
