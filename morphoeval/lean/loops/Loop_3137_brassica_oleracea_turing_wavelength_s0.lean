/-
  Loop 3137 — Brassica oleracea (Romanesco) Turing wavelength.
  Scale: 172 µm primordium-detail mode (tier 5_primordium_100um, detail variant).
  Age: Holocene extant, 0 Ma. Chronoseries 1/1 (no other Brassica chronospecies).
  Patch used by the script: 24 selected waves = 4128 µm.
  Nominal predecessor scale on that patch is the 100 µm tier.
  Successor cellular scale is a 15 µm epidermal cell, which does not equal the selected wave.
-/
namespace MathOfBiology.Loop_3137

def patchUm : Nat := 4128
def lambdaDetailUm : Nat := 172
def lambdaNominalTierUm : Nat := 100
def lambdaCellUm : Nat := 15

/-- The 4128 µm meristem patch holds exactly 24 waves of 172 µm. -/
theorem detail_patch_holds_24_waves :
    patchUm / lambdaDetailUm = 24 := by
  decide

/-- On the same patch the nominal 100 µm tier packs more peaks than this detail variant. -/
theorem nominal_tier_packs_more_peaks :
    patchUm / lambdaNominalTierUm > patchUm / lambdaDetailUm := by
  decide

/-- One 172 µm wave spans 11 cells of 15 µm, so the cell scale is not the selected mode. -/
theorem cell_successor_is_not_the_selected_wave :
    lambdaDetailUm / lambdaCellUm = 11 := by
  decide

end MathOfBiology.Loop_3137
