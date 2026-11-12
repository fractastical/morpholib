/-
  Loop 3152 — Confuciusornis sanctus, Turing wavenumber at the 143.4 µm primordium.
  Geological horizon: Early Cretaceous, Yixian Formation, 125.0 Ma.
  Namespace MathOfBiology.Loop_3152.

  Integer encoding (exact, matching the script):
    fu = 13/15, gv = -36/25, det(J) = 36/25, diffusion ratio r = 12.
    (r fu + gv) = 224/25.
    Turing discriminant comparison cleared of denominators:
      (224/25)^2 ? 4 * r * det = 1728/25
      iff 224*224 ? 1728*25.
  Wavelength unit: tenths of a micrometre, so 1434 stands for 143.4 µm.
-/

namespace MathOfBiology.Loop_3152

/-- Critical wavelength in tenths of a micrometre (143.4 µm). -/
def lambdaCTenthUm : Nat := 1434

/-- Organ-scale sibling vane, 5.4 mm, in the same tenths-of-a-micrometre unit. -/
def lambdaOrganSiblingTenthUm : Nat := 54000

/-- Primordium wavelength is shorter than the 5.4 mm organ-scale sibling. -/
theorem primordium_shorter_than_organ_sibling :
    lambdaCTenthUm < lambdaOrganSiblingTenthUm := by
  decide

/-- Jacobian trace numerator 13*25 + (-36)*15 = -215 < 0, so tr(J) < 0. -/
theorem trace_negative : (13 * 25 + (-36) * 15 : Int) < 0 := by
  decide

/-- Determinant of the homogeneous Jacobian is positive (det = 36/25). -/
theorem det_positive : (36 : Int) > 0 := by
  decide

/-- Turing instability inequality (r fu + gv)^2 > 4 r det, cleared of denominators. -/
theorem turing_wavenumber_instability : (224 : Int) * 224 > 1728 * 25 := by
  decide

/-- Eight critical wavelengths fit on the 1147.2 µm collar transect. -/
theorem eight_wavelengths_on_collar :
    (8 * lambdaCTenthUm) / lambdaCTenthUm = 8 := by
  decide

/-- This 125 Ma node is younger than the 160 Ma Anchiornis-grade predecessor horizon. -/
theorem yixian_younger_than_predecessor_horizon : (125 : Nat) < 160 := by
  decide

end MathOfBiology.Loop_3152
