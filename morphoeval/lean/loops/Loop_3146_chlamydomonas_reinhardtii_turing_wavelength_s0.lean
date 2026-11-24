/-
  morphoeval/lean/loops/Loop_3146_chlamydomonas_reinhardtii_turing_wavelength_s0.lean

  Slot 3146 — Chlamydomonas reinhardtii, Holocene (0 Ma), detail scale.
  Phenomenon length 10.32 nm = 10320 pm is the selected Turing pigment stripe
  inside one eyespot carotenoid globule.

  Scale neighbors (chronoseries is 1/1, so these are length scales, not fossils):
    monomer long axis     <  this stripe           <  quarter-wave globule spacing
    3200 pm (3.2 nm)      <  10320 pm (10.32 nm)   <  90000 pm (90 nm)

  Six such stripes (61920 pm) still fit inside a 100 nm globule (100000 pm).
  Diffusion ratio D_v/D_u = 36 exceeds a ratio-8 onset margin.
  Parameters are not those of sibling slot 0195 (scaled basal rate 163).
-/

namespace MathOfBiology.Loop_3146

def monomerPm : Nat := 3200
def stripePm : Nat := 10320
def quarterWavePm : Nat := 90000
def globulePm : Nat := 100000
def stripeCount : Nat := 6
def dvOverDuTimes100 : Nat := 3600
def siblingSlot0195ScaledBasal : Nat := 163
def thisSlotScaledBasal : Nat := 7

/-- The 10.32 nm stripe sits strictly above a single carotenoid monomer. -/
theorem stripe_above_carotenoid_monomer : monomerPm < stripePm := by
  decide

/-- The same stripe sits strictly below the quarter-wave globule spacing. -/
theorem stripe_below_quarter_wave : stripePm < quarterWavePm := by
  decide

/-- A six-stripe periodic patch fits inside one ~100 nm carotenoid globule. -/
theorem six_stripes_fit_in_globule : stripeCount * stripePm < globulePm := by
  decide

/-- Reference inhibitor/activator diffusion ratio 36 exceeds the ratio-8 onset margin. -/
theorem diffusion_ratio_above_onset : dvOverDuTimes100 > 8 * 100 := by
  decide

/-- Schnakenberg basal parameter of this slot is not slot 0195's scaled alpha. -/
theorem distinct_from_slot_0195 : thisSlotScaledBasal ≠ siblingSlot0195ScaledBasal := by
  decide

end MathOfBiology.Loop_3146
