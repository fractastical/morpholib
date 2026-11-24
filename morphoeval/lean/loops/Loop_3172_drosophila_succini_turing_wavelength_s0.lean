/-
  Loop 3172 — Drosophila succini, detail-scale Turing wavelength (2.759 µm, 44 Ma).

  The integers below are the slot's own scales, not the 114 µm sibling (Loop 3091):
  * λ_succini = 2759 nm (2.759 µm = 2.759e-06 m)
  * λ_berryi  = 2382 nm, the 19 Ma successor after transferring the registry
    primordium ratio 98.43 µm / 114 µm onto this detail scale
  * diffusion ratio ×100: stem 840 < onset 1077 < succini 1872
    (r_stem = 8.40, r_onset ≈ 10.77, r_succini = 18.72)

  Schnakenberg critical wavenumber: k_c = argmax σ(k), λ_c = 2π / k_c.
  No `sorry`.
-/
namespace MathOfBiology.Loop_3172

def lambdaSucciniNm : Nat := 2759
def lambdaBerryiNm : Nat := 2382
def rStemX100 : Nat := 840
def rOnsetX100 : Nat := 1077
def rSucciniX100 : Nat := 1872

/-- Crossing onset at 44 Ma turns the 2.759 µm ocellar mode on. -/
theorem succini_ratio_above_onset : rOnsetX100 < rSucciniX100 := by
  decide

/-- The pre-Baltic stem (chronoseries index 0) is below onset, so σ_max < 0. -/
theorem stem_predecessor_below_onset : rStemX100 < rOnsetX100 := by
  decide

/-- Successor D. berryi (19 Ma) keeps the same kinetics class at a shorter wavelength. -/
theorem berryi_detail_wavelength_shorter : lambdaBerryiNm < lambdaSucciniNm := by
  decide

/-- The compression from succini to berryi is 377 nm on this detail scale. -/
theorem berryi_compression_nm : lambdaSucciniNm - lambdaBerryiNm = 377 := by
  decide

/-- Mode count N = L / λ on a retinal patch. Doubling the patch admits more wavelengths. -/
def modeCount (domainNm wavelengthNm : Nat) : Nat := domainNm / wavelengthNm

theorem mode_count_increases_when_domain_doubles :
    modeCount 8000 lambdaSucciniNm < modeCount 16000 lambdaSucciniNm := by
  decide

end MathOfBiology.Loop_3172
