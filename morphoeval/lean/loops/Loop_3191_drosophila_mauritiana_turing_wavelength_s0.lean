/-
  Loop 3191 — Drosophila mauritiana, Turing wavenumber at the 1.6 µm detail scale.
  Wavelength 1637 nm (1.637 µm, 1.637e-6 m, log10 = -5.786). Geological age 0 Ma.
  Predecessor D. kikkawai cellular stripe: 93410 nm (93.41 µm).
  Compact spinose genital-arch window: 38400 nm (38.4 µm).
  Diffusion contrast D_v/D_u = 9.875, stored as 9875/1000.
  No `sorry`. Integer facts only; the floating-point dispersion solver lives in the Python script.
-/
namespace MathOfBiology.Loop_3191

/-- Detail-scale Turing wavelength for this slot, in nanometres. -/
def lambdaDetailNm : Int := 1637

/-- Predecessor D. kikkawai cellular pigment-stripe wavelength, in nanometres. -/
def lambdaKikkawaiNm : Int := 93410

/-- Reconstructed compact-lobe comparison window, in nanometres. -/
def compactLobeWindowNm : Int := 38400

/-- Inhibitor/activator diffusivity contrast scaled by 1000 (9.875 → 9875). -/
def diffusionContrastMilli : Int := 9875

theorem detail_wavelength_positive : 0 < lambdaDetailNm := by
  unfold lambdaDetailNm
  decide

/-- The detail wavelength is more than fifty times shorter than the kikkawai cellular stripe. -/
theorem detail_wavenumber_exceeds_kikkawai :
    lambdaKikkawaiNm > lambdaDetailNm * 50 := by
  unfold lambdaKikkawaiNm lambdaDetailNm
  decide

/-- The 38.4 µm window holds at least 23 detail ridges and is shorter than one kikkawai wavelength. -/
theorem compact_lobe_packs_detail_ridges :
    lambdaDetailNm * 23 < compactLobeWindowNm ∧
      compactLobeWindowNm < lambdaKikkawaiNm := by
  unfold lambdaDetailNm compactLobeWindowNm lambdaKikkawaiNm
  decide

/-- Differential diffusion: the inhibitor is faster than the activator (contrast > 1). -/
theorem inhibitor_diffuses_farther : diffusionContrastMilli > 1000 := by
  unfold diffusionContrastMilli
  decide

end MathOfBiology.Loop_3191
