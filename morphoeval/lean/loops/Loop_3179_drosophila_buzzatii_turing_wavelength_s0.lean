/-
  Loop 3179 — Drosophila buzzatii, Turing wavenumber at the 2.908 µm detail scale.
  Holocene (extant), 0 Ma. Subgenus Drosophila virilis–repleta–guttifera radiation.

  Wavelengths are nanometres so the comparisons are Nat arithmetic:
    λ(D. buzzatii)      = 2908 nm     = 2.908 µm   (this slot)
    λ(D. bipectinata)   = 85920 nm    = 85.92 µm   (predecessor, slot 2851)
    λ(D. cyrtoloma)     = 207600 nm   = 207.6 µm   (successor, slot 3061)
  Diffusivities are milli-µm²: D_u = 40, D_v = 1160, so D_v / D_u = 29.
  The discriminant integers are the exact Schnakenberg fractions for
  a = 3/25, b = 17/20, D_u = 1/25, D_v = 29/25:
    disc² = 410359297184929 / 588062500000000
    4 D_u D_v det = 272861 / 1562500
  and disc² > 4 D_u D_v det iff the cross-multiplied inequality below holds.
-/

namespace MathOfBiology.Loop_3179

def lambdaBuzzatiiNm : Nat := 2908
def lambdaBipectinataNm : Nat := 85920
def lambdaCyrtolomaNm : Nat := 207600
def wingChordNm : Nat := 1700000
def DuMilli : Nat := 40
def DvMilli : Nat := 1160

/-- At 0 Ma the buzzatii detail-scale wavelength is shorter than the
    predecessor bipectinata cellular ocellus / sex-comb scale. -/
theorem buzzatii_shorter_than_bipectinata :
    lambdaBuzzatiiNm < lambdaBipectinataNm := by decide

/-- The same Holocene fly is also shorter-wavelength than the picture-wing
    successor cyrtoloma (208 µm melanin patches). -/
theorem buzzatii_shorter_than_cyrtoloma :
    lambdaBuzzatiiNm < lambdaCyrtolomaNm := by decide

/-- Subcellular inhibitor/activator ratio is exactly 29, and 29 > 1,
    which is required for a diffusion-driven instability. -/
theorem diffusion_ratio_is_twenty_nine :
    DvMilli = 29 * DuMilli ∧ 29 > 1 := by decide

/-- On a fixed 1.70 mm wing chord, mode count N = L / λ is largest at 2.908 µm. -/
theorem more_modes_than_predecessor_and_successor :
    wingChordNm / lambdaBuzzatiiNm > wingChordNm / lambdaBipectinataNm
    ∧ wingChordNm / lambdaBipectinataNm > wingChordNm / lambdaCyrtolomaNm := by decide

/-- Turing discriminant for this slot's Schnakenberg Jacobian:
    (D_v f_u + D_u g_v)² > 4 D_u D_v det(J). -/
theorem turing_discriminant_positive :
    410359297184929 * 1562500 > 272861 * 588062500000000 := by decide

end MathOfBiology.Loop_3179
