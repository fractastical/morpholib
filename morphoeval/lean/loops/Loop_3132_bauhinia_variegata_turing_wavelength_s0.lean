/-
  morphoeval/lean/loops/Loop_3132_bauhinia_variegata_turing_wavelength_s0.lean
  Slot #3132 — Bauhinia variegata (Orchid Tree Pod), Holocene extant (0 Ma).
  Scale: 69.12 µm = 6.912e-5 m (log10 = -4.160), tier 4_cellular_10um, detail variant.
  Law: Turing wavenumber k_c for Schnakenberg kinetics with D_v/D_u = 16.9.
  Chronoseries 1/1: there is no earlier Bauhinia chronospecies in this slot.
  The centimetre-scale valve (10_000 µm) holds 144 wavelengths of this mode and not 145.
-/
namespace MathOfBiology.Loop_3132

/-- Diffusion ratio r = D_v / D_u, scaled by 1000. Slot value 16.900. -/
def diffusionRatioMilli : Int := 16900

/-- Larger root of the Schnakenberg discriminant at (a, b) = (0.10, 0.90), scaled by 1000.
    r must exceed this root before a real unstable band exists. -/
def ratioThresholdMilli : Int := 8568

def law1Margin (r rcrit : Int) : Int := r - rcrit

theorem law1_turing_ratio_exceeds_threshold :
    law1Margin diffusionRatioMilli ratioThresholdMilli > 0 := by
  decide

/-- Gauge cancellation for the physical domain length. The 69.12 µm mode is selected
    by k_c, independent of an additive reference-frame gauge. -/
def law2Balance (domain gauge : Int) : Int := domain + gauge - gauge

theorem law2_domain_gauge_cancellation (domain gauge : Int) :
    law2Balance domain gauge = domain := by
  dsimp [law2Balance]
  omega

/-- λ = 69.12 µm stored as hundredths of a micrometre. -/
def lambdaHundredthsUm : Int := 6912

/-- One centimetre, the photographed valve scale, in the same hundredths. -/
def centimetreHundredthsUm : Int := 1000000

/-- How many critical wavelengths fit along the 1 cm scale bar. -/
def wavelengthsInCentimetre : Int := 144

theorem law3_cellular_mode_count_on_one_centimetre :
    wavelengthsInCentimetre * lambdaHundredthsUm < centimetreHundredthsUm
    ∧ (wavelengthsInCentimetre + 1) * lambdaHundredthsUm > centimetreHundredthsUm := by
  decide

/-- Holocene extant. This chronoseries has a single point, so the age is 0 Ma. -/
def holoceneAgeMa : Int := 0

theorem law4_extant_holocene_age_is_zero :
    holoceneAgeMa = 0 := by
  decide

end MathOfBiology.Loop_3132
