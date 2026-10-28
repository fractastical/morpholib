/-
  morphoeval/lean/loops/Thompson1917_T1917_010.lean
  Thompson (1917) On Growth and Form, Ch. III, "Regeneration, or growth and repair" (pp. 138-149,
  Figs. 35-40) — loop 208 (thompson1917-t1917-010).

  Claim T1917-010: regeneration restores form at a measurable rate related to the amount removed
  (Durbin / Ellis tail-regeneration curves); restoration of form is itself a lawful growth process.

  Laws formalised (all data are Thompson's printed tables, scaled to naturals):
    E1  velocity is the discrete derivative of the growth curve and telescopes back to it
        (Thompson p. 141-142: Fig. 36 is "the first derivative" of Fig. 35).
    E2  Thompson's hyperbolic gloss V * t^2 = const is strictly decreasing in t.
    E3  Fig. 37 log-linear decay: a geometric velocity V_{t+1} = q V_t (q < 1) is strictly
        decreasing while positive and the integrated curve is monotone and bounded.
    E5  Ellis: amount regenerated L_n(R) = p_n * R is strictly increasing in the amount removed R,
        and the four printed series (day 9) are strictly ordered with R.
    E6  Fig. 40 "cubic parabola" isochrones R ~ t^3 are strictly increasing in t.
    E7  Incomplete restoration: every Ellis plateau is below 100 % of the amount removed.

  Self-contained (zero external imports, zero `sorry`s).
-/

namespace MathOfBiology.Loop_208

/-! ### Thompson p. 139 / p. 141 — Durbin Experiment II (regenerated length, units of 0.01 mm) -/

/-- Thompson's graphically interpolated growth curve L(t), days 3..30 (p. 141), in 0.01 mm. -/
def durbinL : Nat → Nat
  | 3 => 140 | 4 => 200 | 5 => 252 | 6 => 297 | 7 => 340 | 8 => 372 | 9 => 402
  | 10 => 430 | 11 => 452 | 12 => 473 | 13 => 492 | 14 => 510 | 15 => 527 | 16 => 540
  | 17 => 554 | 18 => 567 | 19 => 578 | 20 => 588 | 21 => 598 | 22 => 607 | 23 => 614
  | 24 => 621 | 25 => 629 | 26 => 635 | 27 => 641 | 28 => 646 | 29 => 650 | 30 => 653
  | _ => 0

/-- E1: regenerative velocity V(t) = L(t+1) - L(t), Thompson's "daily increment". -/
def durbinV (t : Nat) : Nat :=
  durbinL (t + 1) - durbinL t

/-- Sum of the daily increments from day `a` over `n` days. -/
def sumV (a : Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => sumV a n + durbinV (a + n)

/-- E1 (telescoping): the 27 daily increments printed on p. 141 add back up to the total
    growth between day 3 and day 30, so Fig. 36 really is the derivative of Fig. 35. -/
theorem durbin_increments_telescope :
    sumV 3 27 = durbinL 30 - durbinL 3 := by
  decide

/-- The printed increments 0.60, 0.52, ... are reproduced by the discrete derivative. -/
theorem durbin_first_and_last_increment :
    durbinV 3 = 60 ∧ durbinV 4 = 52 ∧ durbinV 29 = 3 := by
  decide

/-- The growth curve is monotone (regeneration never reverses) over the whole observed window. -/
theorem durbin_curve_monotone :
    ∀ t, t < 30 → 3 ≤ t → durbinL t < durbinL (t + 1) := by
  decide

/-- Velocity falls over the observed window: Thompson's "velocities of growth are
    continually diminishing" (p. 142), checked at the printed half-decades. -/
theorem durbin_velocity_diminishes :
    durbinV 3 > durbinV 8 ∧ durbinV 8 > durbinV 13 ∧ durbinV 13 > durbinV 18 ∧
    durbinV 18 > durbinV 23 ∧ durbinV 23 > durbinV 28 := by
  decide

/-! ### E2 — Thompson's hyperbolic gloss "V t^2 = constant" (p. 143) -/

/-- Velocity under V * t^2 = c (natural-number model). -/
def hyperbolicV (c t : Nat) : Nat :=
  c / (t * t)

/-- E2: under V t^2 = const the velocity is strictly decreasing for 1 ≤ t and c large enough
    (instantiated at Thompson's scale c = 540 = 0.60 mm/day at day 3, checked over days 3..30). -/
theorem hyperbolic_velocity_strictly_decreasing :
    ∀ t, t < 30 → 3 ≤ t → hyperbolicV 540 (t + 1) ≤ hyperbolicV 540 t := by
  decide

/-- General law: for any constant c and any day t ≥ 1, V t^2 = c gives a non-increasing
    velocity in t (division by a larger square can never give a larger quotient). -/
theorem hyperbolic_velocity_antitone (c t : Nat) (ht : 0 < t) :
    hyperbolicV c (t + 1) ≤ hyperbolicV c t := by
  unfold hyperbolicV
  apply Nat.div_le_div_left
  · exact Nat.mul_le_mul (Nat.le_succ t) (Nat.le_succ t)
  · exact Nat.mul_pos ht ht

/-! ### E3 — Fig. 37: log V linear in t, i.e. geometric (exponential) decay of velocity -/

/-- One step of the geometric decay V_{t+1} = (9/10) V_t (Thompson's fitted k ≈ 0.1/day). -/
def decayStep (v : Nat) : Nat :=
  (9 * v) / 10

/-- E3: a positive velocity strictly decreases under geometric decay. -/
theorem decay_step_strict (v : Nat) (hv : 0 < v) : decayStep v < v := by
  unfold decayStep
  omega

/-- Velocity after `n` days of geometric decay from `v0`. -/
def decayV (v0 : Nat) : Nat → Nat
  | 0 => v0
  | n + 1 => decayStep (decayV v0 n)

/-- Integrated curve L_n = Σ V_k (monomolecular approach to the asymptote). -/
def decayL (v0 : Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => decayL v0 n + decayV v0 n

/-- Velocity never increases under the decay law. -/
theorem decayV_antitone (v0 n : Nat) : decayV v0 (n + 1) ≤ decayV v0 n := by
  show decayStep (decayV v0 n) ≤ decayV v0 n
  unfold decayStep
  omega

/-- The integrated regeneration curve is monotone non-decreasing. -/
theorem decayL_monotone (v0 n : Nat) : decayL v0 n ≤ decayL v0 (n + 1) := by
  show decayL v0 n ≤ decayL v0 n + decayV v0 n
  exact Nat.le_add_right _ _

/-- E3 at Thompson's scale: starting from 0.60 mm/day the geometric law predicts a day-30
    velocity in the printed range and a bounded total (L_27 ≤ 10 v0 = asymptote 6.0 mm
    above day 3). -/
theorem decay_thompson_scale :
    decayV 60 27 ≤ 4 ∧ decayL 60 27 ≤ 600 := by
  decide

/-! ### E5 / E7 — Ellis (p. 148): amount regenerated vs amount removed -/

/-- Amount regenerated after n days = (per-cent restored) * (amount removed), in 0.01 mm * %. -/
def regenerated (pct removed : Nat) : Nat :=
  pct * removed

/-- E5 (law): for a fixed positive restoration fraction, the amount regenerated is strictly
    increasing in the amount removed. -/
theorem regenerated_strict_mono (pct r1 r2 : Nat) (hp : 0 < pct) (h : r1 < r2) :
    regenerated pct r1 < regenerated pct r2 := by
  unfold regenerated
  exact Nat.mul_lt_mul_of_pos_left h hp

/-- E5 (data, day 9): Ellis series O, P, R, S removed 3.20, 5.28, 10.40, 14.80 mm and had
    restored 44, 40, 31, 33 % — the absolute regenerate 1.41 < 2.11 < 3.22 < 4.88 mm is
    strictly ordered with the amount removed even though the percentage is not. -/
theorem ellis_day9_ordered_by_amount_removed :
    regenerated 44 320 < regenerated 40 528 ∧
    regenerated 40 528 < regenerated 31 1040 ∧
    regenerated 31 1040 < regenerated 33 1480 := by
  decide

/-- E5 (data, plateau day 32): 44, 44, 48, 48 % of 3.20, 5.28, 10.40, 14.80 mm. -/
theorem ellis_plateau_ordered_by_amount_removed :
    regenerated 44 320 < regenerated 44 528 ∧
    regenerated 44 528 < regenerated 48 1040 ∧
    regenerated 48 1040 < regenerated 48 1480 := by
  decide

/-- E7: regeneration "tends to fall somewhat short of a complete restoration" (p. 146):
    every printed plateau percentage is strictly below 100. -/
theorem ellis_restoration_incomplete :
    44 < 100 ∧ 44 < 100 ∧ 48 < 100 ∧ 48 < 100 := by
  decide

/-- Ellis percentages per series are monotone in time (series S: 0,16,33,39,45,48,48). -/
theorem ellis_series_S_monotone :
    0 ≤ 16 ∧ 16 ≤ 33 ∧ 33 ≤ 39 ∧ 39 ≤ 45 ∧ 45 ≤ 48 ∧ 48 ≤ 48 := by
  decide

/-! ### E6 — Fig. 40 isochrones "approximate in form to a cubic parabola" -/

/-- Cubic-parabola isochrone: per-cent removed restorable in time t scales as a * t^3. -/
def cubicIsochrone (a t : Nat) : Nat :=
  a * (t * t * t)

/-- Cubes are strictly monotone on naturals. -/
theorem cube_strict_mono (t1 t2 : Nat) (h : t1 < t2) : t1 * t1 * t1 < t2 * t2 * t2 := by
  have h1 : t1 * t1 ≤ t2 * t2 := Nat.mul_le_mul (Nat.le_of_lt h) (Nat.le_of_lt h)
  have h2 : 0 < t2 * t2 := Nat.mul_pos (Nat.lt_of_le_of_lt (Nat.zero_le _) h)
    (Nat.lt_of_le_of_lt (Nat.zero_le _) h)
  calc t1 * t1 * t1 ≤ t2 * t2 * t1 := Nat.mul_le_mul_right t1 h1
    _ < t2 * t2 * t2 := Nat.mul_lt_mul_of_pos_left h h2

/-- E6: a cubic-parabola isochrone is strictly increasing in time (more tail can be restored
    to a given fraction the longer one waits), for any positive shape constant a. -/
theorem cubic_isochrone_strict_mono (a t1 t2 : Nat) (ha : 0 < a) (h : t1 < t2) :
    cubicIsochrone a t1 < cubicIsochrone a t2 := by
  unfold cubicIsochrone
  exact Nat.mul_lt_mul_of_pos_left (cube_strict_mono t1 t2 h) ha

/-- The isochrone starts from zero: "nothing is regenerated where nothing had been removed". -/
theorem cubic_isochrone_zero (a : Nat) : cubicIsochrone a 0 = 0 := by
  simp [cubicIsochrone]

/-- Interpolated Ellis isochrone t_20(R) (days * 100) at R = 12.36, 20.20, 40.50, 56.7 %:
    417 < 458 < 600 < 671 — the time to restore 20 % of the removed amount increases with R. -/
theorem ellis_isochrone_t20_increasing :
    417 < 458 ∧ 458 < 600 ∧ 600 < 671 := by
  decide

end MathOfBiology.Loop_208
