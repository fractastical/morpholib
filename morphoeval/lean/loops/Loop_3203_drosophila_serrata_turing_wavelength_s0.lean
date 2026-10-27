/-
  Loop 3203 — Drosophila serrata detail-scale Turing wavelength.
  Phenomenon: 2.123 µm (2123 nm), Holocene, 0 Ma.
  Chronoseries 32/41, Sophophora ananassae–montium clade.
  Predecessor Drosophila sechellia (2134 nm). Successor Drosophila silvestris (2102 nm).
  Integerization of the Schnakenberg Jacobian at a = 0.037, b = 1.146:
    f_u = 937447/10^6, g_v = -1399489/10^6, det J = 1399489/10^6,
    δ_serrata = 98855/10000 (1.085 × onset).
  Turing inequality (δ f_u + g_v)^2 > 4 δ det is the Nat/Int statement serrata_turing_inequality.
-/
namespace MathOfBiology.Loop_3203

def lambdaSechelliaNm : Nat := 2134
def lambdaSerrataNm : Nat := 2123
def lambdaSilvestrisNm : Nat := 2102

/-- At fixed activator diffusivity, raising δ shortens λ_c.
    Successor (2102 nm) < serrata Holocene (2123 nm) < predecessor (2134 nm). -/
theorem chronoseries_detail_wavelength_shortens :
    lambdaSilvestrisNm < lambdaSerrataNm ∧ lambdaSerrataNm < lambdaSechelliaNm := by
  decide

/-- Eight critical wavelengths span the periodic tergite patch used in the script (8 × 2123 = 16984 nm). -/
theorem tergite_patch_holds_eight_wavelengths :
    8 * lambdaSerrataNm = 16984 := by
  decide

/-- Serrata δ numerator 98855/10000 lies strictly above the integerized onset 91111/10000. -/
theorem serrata_above_turing_onset :
    (98855 : Nat) > 91111 := by
  decide

/-- Equal diffusion (δ = 1) makes the trace-diffusion coefficient D_v f_u + D_u g_v negative,
    so the band-center formula does not select a finite unstable wavenumber. -/
theorem equal_diffusion_trace_diffusion_negative :
    (937447 : Int) + (-1399489) < 0 := by
  decide

/-- Integerized Turing inequality at the serrata operating point.
    lhs = (δ_num f_u_micro + 10000 g_v_micro)^2
    rhs = 4 δ_num det_micro 10^10
    which is the common-denominator form of (δ f_u + g_v)^2 > 4 δ det. -/
theorem serrata_turing_inequality :
    ((98855 : Int) * 937447 + 10000 * (-1399489)) ^ 2
      - 4 * (98855 : Int) * 1399489 * (10 : Int) ^ (10 : Nat) > 0 := by
  decide

end MathOfBiology.Loop_3203
