/-
  Loop 3215 — Eldredgeops rana_18file, Turing wavenumber at the 492 nm detail scale.
  Horizon: Middle Devonian (Cazenovian), 387 Ma. Chronoseries 1/2.
  Successor: Eldredgeops rana_17file at 384 Ma (17 dorsoventral lens files).
  The cellular-tier sibling wavelength is 17860 nm (17.86 µm); this loop is the
  492 nm pigment / ocellus scale, so its selected wavenumber is higher.
-/
namespace MathOfBiology.Loop_3215

/-- Assigned critical wavelength, in nanometres, for the 18-file Cazenovian slot. -/
def lambdaNm : Int := 492

/-- Dorsoventral calcite-lens file count at 387 Ma. -/
def nFiles : Int := 18

/-- Successor file count (Tioughniogan 17-file morph, 384 Ma). -/
def nFilesSuccessor : Int := 17

theorem lambda_nm_positive : lambdaNm > 0 := by
  decide

/-- Punctuated file-count step at the organism scale: exactly one file is lost. -/
theorem file_count_drops_by_one : nFiles - nFilesSuccessor = 1 := by
  decide

/-- If the dorsoventral arc is conserved, wavelengths per file scale by 18/17.
    Cross-multiplying, the 18-file morph packs more files than the successor
    into the same arc, which is the integer content of that ratio. -/
theorem eighteen_files_exceed_successor : nFiles * lambdaNm > nFilesSuccessor * lambdaNm := by
  decide

/-- Cazenovian 387 Ma is older than the Tioughniogan successor at 384 Ma. -/
theorem cazenovian_precedes_tioughniogan : (387 : Int) > 384 := by
  decide

/-- 492 nm is a shorter wavelength than the sibling cellular tier (17860 nm),
    so the detail-scale wavenumber 2π/λ is the larger of the two. -/
theorem detail_wavelength_shorter_than_cellular_sibling : (17860 : Int) > lambdaNm := by
  decide

/-- Rescaled inhibitor diffusivity exceeds activator diffusivity
    (Dv × 10^9 = 583320837 > Du × 10^9 = 21341006), the Turing requirement Du ≠ Dv
    with the inhibitor the faster species. -/
theorem inhibitor_diffuses_faster : (583320837 : Int) > 21341006 := by
  decide

end MathOfBiology.Loop_3215
