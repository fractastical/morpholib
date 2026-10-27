/-
  Loop 3210 — Drosophila virilis, detail scale.
  Phenomenon: Turing pigment stripe / ocellus wavelength, 3.101 µm = 3101 nm.
  Age: Holocene (extant), 0 Ma. Lineage: subgenus Drosophila virilis–repleta–guttifera radiation.
  Chronoseries 39/41. Predecessor Drosophila teissieri. Successor Drosophila willistoni.
  Shared diffusivities; species differ by basal activator production a.
  Wavelengths below are the nanometre roundings of the Schnakenberg onset law k_c
  at shared Dv/Du = 10, a_teissieri = 0.058, a_virilis = 0.100, and quenched willistoni a = 0.360.
  The 136 µm sibling primordium slot is 136000 nm and is used only as a scale separator.
-/
namespace MathOfBiology.Loop_3210

/-- Selected wavelength of D. teissieri at a = 0.058, in nanometres. -/
def lambdaTeissieriNm : Nat := 2924

/-- Selected wavelength of D. virilis at a = 0.100, the assigned 3.1 µm detail scale. -/
def lambdaVirilisNm : Nat := 3101

/-- D. willistoni at a = 0.360 is past the Turing boundary, so no wavenumber is selected. -/
def lambdaWillistoniNm : Nat := 0

/-- Sibling primordium-scale slot (136 µm), in nanometres. Not a parameter of this model. -/
def lambdaPrimordiumSiblingNm : Nat := 136000

/-- Basal activator production in thousandths: teissieri 58, virilis 100, willistoni 360. -/
def activatorMilli (species : Nat) : Nat :=
  match species with
  | 0 => 58
  | 1 => 100
  | 2 => 360
  | _ => 0

theorem predecessor_selects_shorter_wavelength :
    lambdaTeissieriNm < lambdaVirilisNm := by
  decide

theorem successor_quenches_selected_wavelength :
    lambdaWillistoniNm = 0 ∧ 0 < lambdaVirilisNm := by
  decide

theorem activator_rises_along_the_chronoseries :
    activatorMilli 0 < activatorMilli 1 ∧ activatorMilli 1 < activatorMilli 2 := by
  decide

/-- One virilis tergite (adult length / 7) holds more  detail-scale modes than one teissieri tergite. -/
theorem virilis_tergite_holds_more_modes :
    (3400000 / 7) / lambdaVirilisNm > (2500000 / 7) / lambdaTeissieriNm := by
  decide

/-- The 3.1 µm detail wavelength fits between 43 and 44 times inside the 136 µm primordium sibling. -/
theorem detail_scale_separates_from_primordium_sibling :
    lambdaVirilisNm * 43 < lambdaPrimordiumSiblingNm ∧
      lambdaPrimordiumSiblingNm < lambdaVirilisNm * 44 := by
  decide

end MathOfBiology.Loop_3210
