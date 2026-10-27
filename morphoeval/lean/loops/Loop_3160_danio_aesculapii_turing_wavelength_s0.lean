/-
  Loop 3160 — Danio aesculapii cellular Turing wavenumber at 20.8 µm (nominal 21 µm).
  Phenomenon size λ_c = 2.08×10⁻⁵ m, log10 = −4.682, tier 4_cellular_10um, age 0 Ma.
  Successor chronospecies Danio albolineatus is below the Turing threshold at this scale.
  Integer ledger matches models/2026-delphi10000-danio-aesculapii-turing-wavelength-s0.py.
  Jacobian entries are in units of 1/10000. Diffusivities are in units of 1/10^6.
-/
namespace MathOfBiology.Loop_3160

/-- Trace f_u + g_v = 7500/10000 + (−12544)/10000 = −0.5044. -/
def traceScaled : Int := 7500 + (-12544)

theorem trace_negative : traceScaled < 0 := by
  decide

/-- det J = f_u g_v − f_v g_u, with the same 1/10000 scaling on each entry.
    The product scaling is 1/10^8 and the sign is what the threshold needs. -/
def detScaled : Int := 7500 * (-12544) - 12544 * (-17500)

theorem determinant_positive : detScaled > 0 := by
  decide

/-- Diffusion-weighted trace D_v f_u + D_u g_v for D. aesculapii at R = D_v/D_u = 28.
    Positive, so a finite band of wavenumbers can be unstable. -/
def aesculapiiDiffusionTrace : Int := 64020722 * 7500 + 2286454 * (-12544)

theorem aesculapii_above_turing_threshold : aesculapiiDiffusionTrace > 0 := by
  decide

/-- Same activator diffusivity, inhibitor collapsed to R = 1.20 (D. albolineatus
    at this 20.8 µm scale). The diffusion-weighted trace changes sign. -/
def albolineatusDiffusionTrace : Int := 2743745 * 7500 + 2286454 * (-12544)

theorem albolineatus_below_turing_threshold : albolineatusDiffusionTrace < 0 := by
  decide

/-- Discriminant (D_v f_u + D_u g_v)² − 4 D_u D_v det J, in the scaled ledger.
    Positive iff the aesculapii parameters satisfy the Turing inequality. -/
def turingDiscriminant : Int :=
  aesculapiiDiffusionTrace * aesculapiiDiffusionTrace
    - 4 * 2286454 * 64020722 * detScaled

theorem discriminant_positive : turingDiscriminant > 0 := by
  decide

/-- A 12-wavelength cellular patch at 20.8 µm is 249.6 µm (2496 tenths of a micrometre).
    This is the domain of the s0 detail model, not the ~1 mm organ-scale sibling. -/
theorem twelve_wavelengths_span_the_cellular_patch : 12 * 208 = 2496 := by
  decide

end MathOfBiology.Loop_3160
