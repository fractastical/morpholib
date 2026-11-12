/-
  Loop 3128 — Aulacopleura koninckii holaspis22, Turing wavelength at 15 µm.
  Geological age 428.1 Ma (Wenlock). Predecessor morphotype t18 releases
  thoracic segments through stage s18; this morphotype t22 releases them
  through stage s22 (Hughes, Hong, Hou & Fusco 2017).
  Meraspid Dyar ratio 1.115 is encoded as 1115/1000.
  Holaspid Dyar ratio 1.103 is encoded as 1103/1000.
  Critical wavelength on this slot: λ_c = 1.514e-5 m.
-/
namespace MathOfBiology.Loop_3128

/-- Four extra meraspid molts separate t22 holaspid entry from t18. -/
theorem t22_entry_four_molts_after_t18 :
    (22 : Nat) - 18 = 4 := by decide

/-- 1.115^4 > 22/18. Four Dyar steps enlarge length by more than the segment-count ratio. -/
theorem dyar_enlargement_exceeds_segment_ratio :
    (1115 : Nat) ^ 4 * 9 > 11 * (1000 : Nat) ^ 4 := by decide

/-- Holaspid per-molt increment 1.103 is strictly slower than the meraspid increment 1.115. -/
theorem holaspid_increment_slower_than_meraspid :
    (1103 : Nat) < 1115 := by decide

/-- Wavelengths per tergite increase: (L22/22) / (L18/18) > 1 when L22/L18 = 1.115^4. -/
theorem per_tergite_wavelength_count_increases :
    (1115 : Nat) ^ 4 * 18 > 22 * (1000 : Nat) ^ 4 := by decide

end MathOfBiology.Loop_3128
