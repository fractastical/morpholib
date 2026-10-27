/-
  Loop 3208 — Drosophila takahashii detail-scale Turing wavelength.
  Scale: 2.362 µm = 2362 nm (human label 2.4 µm; 2.362e-06 m; log10 = -5.627).
  Age: Holocene extant, 0 Ma. Chronoseries 37/41.
  Predecessor envelope: Drosophila suzukii, 98450 nm (98.45 µm).
  Successor envelope: Drosophila teissieri, 107000 nm (107 µm).
  The detail wavelength is not rescaled when the cellular envelope lengthens.
-/
namespace MathOfBiology.Loop_3208

def lambdaDetailNm : Int := 2362
def lambdaSuzukiiEnvelopeNm : Int := 98450
def lambdaTeissieriEnvelopeNm : Int := 107000
def domainNm : Int := 28344

/-- Twelve critical wavelengths span the periodic chart: 12 * 2362 = 28344 nm. -/
theorem twelve_wavelengths_span_the_chart :
    domainNm = 12 * lambdaDetailNm := by
  decide

/-- The 2.362 µm lattice is strictly shorter than the suzukii cellular envelope. -/
theorem detail_shorter_than_suzukii_envelope :
    lambdaDetailNm < lambdaSuzukiiEnvelopeNm := by
  decide

/-- The suzukii envelope holds 41 detail wavelengths and not 42. -/
theorem suzukii_envelope_holds_forty_one_detail_waves :
    41 * lambdaDetailNm < lambdaSuzukiiEnvelopeNm
      ∧ lambdaSuzukiiEnvelopeNm < 42 * lambdaDetailNm := by
  decide

/-- From suzukii to teissieri the cellular envelope lengthens by 8550 nm.
    The detail wavelength stays below both envelopes, so that lengthening does
    not rescale λ_c of this slot. -/
theorem successor_envelope_lengthens_without_rescaling_detail :
    lambdaTeissieriEnvelopeNm - lambdaSuzukiiEnvelopeNm = 8550
      ∧ lambdaDetailNm < lambdaSuzukiiEnvelopeNm
      ∧ lambdaDetailNm < lambdaTeissieriEnvelopeNm := by
  decide

end MathOfBiology.Loop_3208
