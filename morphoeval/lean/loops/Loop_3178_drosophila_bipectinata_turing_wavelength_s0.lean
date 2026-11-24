/-
  Loop 3178 — Drosophila bipectinata, subcellular Turing wavelength, detail rung s0.
  Scale: 1.92 µm = 1920 nm (log10(1.92e-6) ≈ −5.717), tier 3_subcellular_um.
  Age: Holocene (Extant), 0 Ma. Chronoseries 7/41, Sophophora ananassae–montium clade.
  Predecessor Drosophila biarmipes (2400 nm) and successor Drosophila buzzatii (1536 nm)
  differ by activator diffusivity at this same age, not by geological time:
    (λ / λ_focal)² = D_u / D_u,focal.
  Namespace MathOfBiology.Loop_3178. Proofs are closed with decide.
-/
namespace MathOfBiology.Loop_3178

def lambdaFocalNm : Nat := 1920
def lambdaBiarmipesNm : Nat := 2400
def lambdaBuzzatiiNm : Nat := 1536
def domainNm : Nat := 15360

/-- Du factor 25/16 lengthens the predecessor wavelength: λ_b² · 16 = λ_f² · 25. -/
theorem biarmipes_diffusivity_lengthens_wavelength :
    lambdaBiarmipesNm * lambdaBiarmipesNm * 16 = lambdaFocalNm * lambdaFocalNm * 25 := by
  decide

/-- Du factor 16/25 shortens the successor wavelength: λ_z² · 25 = λ_f² · 16. -/
theorem buzzatii_diffusivity_shortens_wavelength :
    lambdaBuzzatiiNm * lambdaBuzzatiiNm * 25 = lambdaFocalNm * lambdaFocalNm * 16 := by
  decide

/-- At 0 Ma the ordered chronoseries is still strictly separated in wavelength. -/
theorem chronoseries_wavelength_order :
    lambdaBiarmipesNm > lambdaFocalNm ∧ lambdaFocalNm > lambdaBuzzatiiNm := by
  decide

/-- Eight focal wavelengths tile the 15.36 µm periodic patch. -/
theorem focal_domain_holds_eight_wavelengths :
    domainNm / lambdaFocalNm = 8 := by
  decide

/-- Equal diffusion sits below the Turing threshold for these kinetics.
    fu0 = 73/97 and gv0 = −9409/10000, so Du·gv + Dv·fu at Dv = Du has numerator
    73·10000 − 9409·97 < 0. -/
theorem equal_diffusion_trace_is_negative :
    (73 * 10000 : Int) - 9409 * 97 < 0 := by
  decide

end MathOfBiology.Loop_3178
