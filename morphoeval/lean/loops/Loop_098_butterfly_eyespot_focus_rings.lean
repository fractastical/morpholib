/-
  morphoeval/lean/loops/Loop_098_butterfly_eyespot_focus_rings.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 098:
  Nijhout (1990) & Brakefield et al. (1996) Butterfly Wing-Vein Focus
  Determination & Concentric French-Flag Eyespot Ring Morphogenesis
  (`nijhout-butterfly-eyespot-focus-rings`).

  Pure core Lean 4 (zero external dependencies, zero `sorry`s).
  Verify with the Lean 4 kernel:
    lean morphoeval/lean/loops/Loop_098_butterfly_eyespot_focus_rings.lean
-/

namespace MathOfBiology.Loop_098

-- ============================================================================
-- 1. Concentric Focal Morphogen Decay & Nested Threshold Ordering
-- ============================================================================

/-- Concentric eyespot ring radius `R_i(lam, L0, Li) = lam * (L0 - Li)` derived
    from the steady-state radial focal morphogen decay `c(r) = c0 * exp(-r / lam)`
    where `L0 = ln(c0)` is the log focal source strength and `Li = ln(theta_i)`
    is the log transcriptional activation threshold (with natural truncation at 0
    when `L0 ≤ Li`). -/
def eyespot_ring_radius (lam L0 Li : Nat) : Nat :=
  lam * (L0 - Li)

/-- Four-zone concentric scale-fate classifier (`3 = White UV Pupil`,
    `2 = Black Spalt/Dll Melanin Disc`, `1 = Gold Engrailed Outer Ring`,
    `0 = Basal Wing Ground`) for local morphogen concentration `c` against
    nested thresholds `theta_gold < theta_black < theta_pupil`. -/
def eyespot_ring_zone (c theta_pupil theta_black theta_gold : Nat) : Nat :=
  if theta_pupil ≤ c then 3
  else if theta_black ≤ c then 2
  else if theta_gold ≤ c then 1
  else 0

/-- Theorem 1 (`nijhout_concentric_threshold_ordering`):
    For any positive morphogen decay length `lam > 0` and strictly ordered
    transcriptional log-thresholds `L_gold < L_black < L_pupil < L0`
    (representing `theta_gold < theta_black < theta_pupil < c0`), the concentric
    eyespot ring radii satisfy `0 < R_pupil < R_black < R_gold`. -/
theorem nijhout_concentric_threshold_ordering
    (lam L0 L_pupil L_black L_gold : Nat)
    (h_lam : 0 < lam)
    (h_gb : L_gold < L_black)
    (h_bp : L_black < L_pupil)
    (h_p0 : L_pupil < L0) :
    0 < eyespot_ring_radius lam L0 L_pupil ∧
    eyespot_ring_radius lam L0 L_pupil < eyespot_ring_radius lam L0 L_black ∧
    eyespot_ring_radius lam L0 L_black < eyespot_ring_radius lam L0 L_gold := by
  unfold eyespot_ring_radius
  have h1 : 0 < L0 - L_pupil := by omega
  have h2 : L0 - L_pupil < L0 - L_black := by omega
  have h3 : L0 - L_black < L0 - L_gold := by omega
  refine ⟨Nat.mul_pos h_lam h1, ?_, ?_⟩
  · exact Nat.mul_lt_mul_of_pos_left h2 h_lam
  · exact Nat.mul_lt_mul_of_pos_left h3 h_lam


-- ============================================================================
-- 2. Seasonal Polyphenism Annulus Width Invariance & Goldeneye Threshold Shift
-- ============================================================================

/-- Outer golden-yellow (*Engrailed*) annulus width `ΔR = R_gold - R_black`. -/
def gold_annulus_width (lam L0 L_black L_gold : Nat) : Nat :=
  eyespot_ring_radius lam L0 L_gold - eyespot_ring_radius lam L0 L_black

/-- Theorem 2 (`brakefield_polyphenism_annulus_invariance`):
    Whenever focal log-source strength exceeds the black ring threshold
    (`L_gold ≤ L_black ≤ L0`), the outer gold-to-black annulus width evaluates to
    `ΔR = lam * (L_black - L_gold)`, which is strictly invariant across seasonal
    ecdysone polyphenism (`L0_dry` vs `L0_wet`), whereas elevating the *Spalt*
    threshold (`L_black < L_black_goldeneye ≤ L0`) in the *Goldeneye* mutant
    strictly expands the gold annulus width. -/
theorem brakefield_polyphenism_annulus_invariance
    (lam L0_dry L0_wet L_black L_black_ge L_gold : Nat)
    (h_lam : 0 < lam)
    (h_gb : L_gold ≤ L_black)
    (h_ge : L_black < L_black_ge)
    (h_dry : L_black_ge ≤ L0_dry)
    (h_wet : L0_dry ≤ L0_wet) :
    gold_annulus_width lam L0_wet L_black L_gold = lam * (L_black - L_gold) ∧
    gold_annulus_width lam L0_dry L_black L_gold =
      gold_annulus_width lam L0_wet L_black L_gold ∧
    gold_annulus_width lam L0_wet L_black L_gold <
      gold_annulus_width lam L0_wet L_black_ge L_gold := by
  unfold gold_annulus_width eyespot_ring_radius
  have h_sub_wet : (L0_wet - L_gold) - (L0_wet - L_black) = L_black - L_gold := by omega
  have h_sub_dry : (L0_dry - L_gold) - (L0_dry - L_black) = L_black - L_gold := by omega
  have h_sub_ge  : (L0_wet - L_gold) - (L0_wet - L_black_ge) = L_black_ge - L_gold := by omega
  have h_w1 : lam * (L0_wet - L_gold) - lam * (L0_wet - L_black) = lam * (L_black - L_gold) := by
    rw [← Nat.mul_sub_left_distrib, h_sub_wet]
  have h_w2 : lam * (L0_dry - L_gold) - lam * (L0_dry - L_black) = lam * (L_black - L_gold) := by
    rw [← Nat.mul_sub_left_distrib, h_sub_dry]
  have h_w3 : lam * (L0_wet - L_gold) - lam * (L0_wet - L_black_ge) = lam * (L_black_ge - L_gold) := by
    rw [← Nat.mul_sub_left_distrib, h_sub_ge]
  refine ⟨h_w1, ?_, ?_⟩
  · rw [h_w2, h_w1]
  · rw [h_w1, h_w3]
    have h_lt : L_black - L_gold < L_black_ge - L_gold := by omega
    exact Nat.mul_lt_mul_of_pos_left h_lt h_lam


-- ============================================================================
-- 3. Early Pupal Focal Microcautery Ablation & Complete Eyespot Loss
-- ============================================================================

/-- Theorem 3 (`nijhout_focal_cautery_ablation_collapse`):
    When early pupal focal microcautery reduces the focal log-source below the
    outermost gold threshold (`L0 ≤ L_gold ≤ L_black ≤ L_pupil`) and local
    concentration `c < theta_gold`, all three concentric ring radii vanish
    identically (`R_pupil = 0`, `R_black = 0`, `R_gold = 0`) and every scale
    differentiates into basal wing ground (`eyespot_ring_zone = 0`). -/
theorem nijhout_focal_cautery_ablation_collapse
    (lam L0 L_pupil L_black L_gold c theta_pupil theta_black theta_gold : Nat)
    (h_cautery : L0 ≤ L_gold)
    (h_gb : L_gold ≤ L_black)
    (h_bp : L_black ≤ L_pupil)
    (h_c : c < theta_gold)
    (h_tgb : theta_gold ≤ theta_black)
    (h_tbp : theta_black ≤ theta_pupil) :
    eyespot_ring_radius lam L0 L_pupil = 0 ∧
    eyespot_ring_radius lam L0 L_black = 0 ∧
    eyespot_ring_radius lam L0 L_gold = 0 ∧
    eyespot_ring_zone c theta_pupil theta_black theta_gold = 0 := by
  unfold eyespot_ring_radius eyespot_ring_zone
  have hp0 : L0 - L_pupil = 0 := by omega
  have hb0 : L0 - L_black = 0 := by omega
  have hg0 : L0 - L_gold = 0 := by omega
  have hnc1 : ¬ (theta_pupil ≤ c) := by omega
  have hnc2 : ¬ (theta_black ≤ c) := by omega
  have hnc3 : ¬ (theta_gold ≤ c) := by omega
  simp [hp0, hb0, hg0, hnc1, hnc2, hnc3]


-- ============================================================================
-- 4. Twin-Focus Superposition Symmetry & Hourglass Bridge Overlap
-- ============================================================================

/-- Superposed morphogen concentration along the inter-focal axis `[0, d12]`
    generated by two equal-strength eyespot foci at `x = 0` and `x = d12`. -/
def twin_focus_superposition (phi : Nat → Nat) (d12 x : Nat) : Nat :=
  phi x + phi (d12 - x)

/-- Overlap bridge width `2 * R_gold - d12` between two adjacent eyespots of
    outer gold radius `R_gold` separated by inter-focal distance `d12`. -/
def twin_focus_bridge_overlap (R_gold d12 : Nat) : Nat :=
  2 * R_gold - d12

/-- Theorem 4 (`beldade_twin_focus_hourglass_fusion`):
    Whenever two eyespot foci of gold radius `R_gold` are separated by
    `d12 < 2 * R_gold`, the inter-focal gold bridge overlap width is strictly
    positive (`0 < twin_focus_bridge_overlap R_gold d12`), and for every point
    `x ≤ d12` along the inter-focal axis the superposed morphogen profile is
    strictly invariant under reflection `x ↦ d12 - x`. -/
theorem beldade_twin_focus_hourglass_fusion
    (phi : Nat → Nat) (R_gold d12 x : Nat)
    (h_fuse : d12 < 2 * R_gold)
    (hx : x ≤ d12) :
    0 < twin_focus_bridge_overlap R_gold d12 ∧
    twin_focus_superposition phi d12 (d12 - x) = twin_focus_superposition phi d12 x := by
  constructor
  · unfold twin_focus_bridge_overlap
    omega
  · unfold twin_focus_superposition
    have h_sub : d12 - (d12 - x) = x := by omega
    rw [h_sub, Nat.add_comm]

end MathOfBiology.Loop_098
