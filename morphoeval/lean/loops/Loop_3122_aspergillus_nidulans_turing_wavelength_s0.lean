/-
  morphoeval/lean/loops/Loop_3122_aspergillus_nidulans_turing_wavelength_s0.lean
  Slot 3122 — Aspergillus nidulans, Holocene extant (0 Ma), chronoseries 1/1.
  Detail scale: Turing pigment-stripe onset wavelength
  λ = 1.284e-5 m = 12840 nm (nominal 13 µm), tier 4_cellular_10um.
  There is no predecessor chronospecies and no successor in this series.
  What changes at this scale is spatial, against the 10 µm tier floor:
  the detail variant is 2840 nm longer, so k_c = 2π/λ is smaller.
  These integers are not the slot-182 template (48, 16).
-/
namespace MathOfBiology.Loop_3122

/-- Onset wavelength of this Holocene detail variant, in nanometres. -/
def lambda_nm : Int := 12840

/-- Floor of tier 4 (`4_cellular_10um`), 10 µm. -/
def tier_floor_nm : Int := 10000

def detail_margin_nm : Int := lambda_nm - tier_floor_nm

/-- Margin of this detail variant above the 10 µm tier floor. -/
theorem detail_variant_above_tier_floor :
    detail_margin_nm = 2840 := by
  decide

/-- Eight onset wavelengths fill a 102.72 µm conidial window. -/
theorem eight_modes_fill_conidial_window :
    8 * lambda_nm = 102720 := by
  decide

/-- A 100 µm scale bar holds seven full 12840 nm waves and not eight. -/
theorem scale_bar_holds_seven_waves :
    7 * lambda_nm < 100000 ∧ 100000 < 8 * lambda_nm := by
  decide

/-- Inhibitor diffusion outruns the activator.
    Dv/Du = 9295/1000 = 9.295, the ratio integrated by the script. -/
def Du_milli : Int := 1000
def Dv_milli : Int := 9295

theorem inhibitor_outdiffuses_activator :
    Dv_milli - Du_milli = 8295 ∧ Dv_milli > Du_milli := by
  decide

/-- Chronoseries 1/1 at 0 Ma: the extant wavelength is the whole series. -/
def series_index : Int := 1
def series_length : Int := 1
def age_ma : Int := 0

theorem holocene_chronoseries_singleton :
    series_index = series_length ∧ age_ma = 0 := by
  decide

end MathOfBiology.Loop_3122
