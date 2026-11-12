/-
  Loop 3233 — Holocene (extant, 0 Ma) Gasterosteus aculeatus.
  Cellular Turing carrier at 56 µm (tier 4_cellular_10um), not the millimetre bar envelope.
  Schnakenberg Jacobian at a = 1/5, b = 1, Dᵤ = 1, Dᵥ = 30, cleared over 75 and 625:
    trace = −58/75 < 0,  det = 108/75 > 0,  discriminant = 107296/625 > 0.
  Chronospecies cellular wavelengths in hundredths of a micrometre, same dx:
    marine predecessor 6181 > this Holocene slot 5600 > benthic successor 5223.
-/
namespace MathOfBiology.Loop_3233

/-- Trace numerator of the homogeneous Jacobian (−58/75). Stable without diffusion. -/
def traceNumerator : Int := -58

/-- Determinant numerator of the homogeneous Jacobian (108/75). -/
def detNumerator : Int := 108

/-- Numerator of (Dᵥ fᵤ + Dᵤ gᵥ)² − 4 Dᵤ Dᵥ det after clearing the denominator 625. -/
def discriminantNumerator : Int := 107296

theorem stable_without_diffusion :
    traceNumerator < 0 ∧ detNumerator > 0 := by
  decide

theorem diffusion_driven_discriminant_positive :
    discriminantNumerator > 0 := by
  decide

/-- 56 µm cellular carrier versus the organ-scale envelopes of the neighbouring slots
    (marine 2400 µm, benthic 2200 µm). -/
def marineEnvelopeUm : Nat := 2400
def benthicEnvelopeUm : Nat := 2200
def cellularCarrierUm : Nat := 56

theorem cellular_carrier_finer_than_chronoseries_envelopes :
    cellularCarrierUm < benthicEnvelopeUm ∧ benthicEnvelopeUm < marineEnvelopeUm := by
  decide

/-- Analytic cellular wavelengths (×100 µm) at fixed dx: marine a = 0.28, Holocene a = 0.20,
    benthic a = 0.14. The carrier shortens toward the benthic successor. -/
def marineCellularHundredths : Nat := 6181
def holoceneCellularHundredths : Nat := 5600
def benthicCellularHundredths : Nat := 5223

theorem cellular_wavelength_contracts_from_marine_to_benthic :
    marineCellularHundredths > holoceneCellularHundredths
      ∧ holoceneCellularHundredths > benthicCellularHundredths := by
  decide

end MathOfBiology.Loop_3233
