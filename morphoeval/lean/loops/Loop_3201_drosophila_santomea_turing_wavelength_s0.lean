/-
  Loop 3201 — Drosophila santomea detail-scale Turing wavelength.
  Scale: 2.326 µm (2.326e-6 m, log10 = -5.633), tier 3_subcellular_um. Age: 0 Ma (Holocene extant).
  Schnakenberg Jacobian at a = 0.130, b = 0.870 gives f_u = 0.74, g_v = -1, det = 1, tr = -0.26.
  With D_u = 1 and D_v = 40, h = D_v f_u + D_u g_v = 28.6 and 4 D_u D_v det = 160.
  Cleared of decimals: 286^2 > 16000 because (28.6 * 10)^2 > 160 * 100.
  Critical wavenumber identity: k_c^2 = h / (2 D_u D_v) = 28.6 / 80 = 143/400.
  Chronoseries wavelengths in nanometres (this tier only):
    predecessor D. pseudoobscura 1840 nm < successor D. sechellia 2050 nm < D. santomea 2326 nm.
  gamma scales as 1/lambda^2, so the santomea gamma is the smallest of the three.
  Proofs are closed by decide.
-/

namespace MathOfBiology.Loop_3201

def dispersionNumerator : Int := 286

def dispersionDenominator : Int := 16000

/-- Turing gap at the locked santomea Jacobian: h^2 > 4 D_u D_v det, in tenths. -/
theorem turing_determinant_gap :
    dispersionNumerator * dispersionNumerator > dispersionDenominator := by
  decide

/-- k_c^2 = h / (2 D_u D_v) = 28.6/80 = 143/400. -/
theorem critical_wavenumber_identity :
    (286 : Int) * 400 = 143 * 800 := by
  decide

/-- Equal diffusion: h = D (f_u + g_v) and f_u + g_v = -0.26 < 0, so no positive k_c. -/
theorem equal_diffusion_trace_negative :
    (74 : Int) + (-100) < 0 := by
  decide

def lambdaNmPseudoobscura : Nat := 1840

def lambdaNmSechellia : Nat := 2050

def lambdaNmSantomea : Nat := 2326

/-- At 2.326 µm and 0 Ma, santomea is longer-wavelength than both chronoseries neighbors. -/
theorem santomea_longer_than_neighbors :
    lambdaNmPseudoobscura < lambdaNmSechellia ∧ lambdaNmSechellia < lambdaNmSantomea := by
  decide

/-- Because gamma ∝ 1/λ^2, the predecessor has the largest gamma and santomea the smallest. -/
theorem gamma_ordered_by_inverse_square_wavelength :
    lambdaNmPseudoobscura * lambdaNmPseudoobscura < lambdaNmSechellia * lambdaNmSechellia
      ∧ lambdaNmSechellia * lambdaNmSechellia < lambdaNmSantomea * lambdaNmSantomea := by
  decide

end MathOfBiology.Loop_3201
