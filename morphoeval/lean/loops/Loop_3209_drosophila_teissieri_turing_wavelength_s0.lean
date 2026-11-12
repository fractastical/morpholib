/-
  morphoeval/lean/loops/Loop_3209_drosophila_teissieri_turing_wavelength_s0.lean
  Slot 3209 — Drosophila teissieri, detail-scale Turing wavelength.
  Scale: 1.72 µm (1.72e-6 m, log10 ≈ -5.764), tier 3_subcellular_um, variant s0.
  Age: Holocene extant, 0 Ma.
  Law: k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v), λ_c = 2π / k_c.
  Integer scaling used below (hundredths of a micrometre, or cleared denominators)
  matches models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py:
    a = 13/100, b = 91/100, u0 = 26/25, f_u = 3/4, g_v = -676/625,
    trace numerator 3*625 - 676*4 = -829 < 0,
    diffusivity ratio 17.4, domain 12 * 172 = 2064 hundredths of a micrometre.
-/
namespace MathOfBiology.Loop_3209

/-- Cleared numerator of trace(J) = f_u + g_v = 3/4 - 676/625.
    Denominator 4*625 > 0, so the sign is the numerator. -/
def traceNumerator : Int := (3 : Int) * 625 - 676 * 4

/-- Homogeneous Schnakenberg state is asymptotically stable without diffusion
    at the slot-3209 Jacobian (trace < 0). Turing instability, if it occurs,
    is diffusion-driven. -/
theorem homogeneous_trace_negative : traceNumerator < 0 := by
  decide

/-- Cleared numerator of the diffusion-weighted drive r f_u + g_v at r = 174/10.
    r f_u + g_v = 261/20 - 676/625, denominator 2500 > 0. -/
def driveNumerator : Int := (261 : Int) * 125 - 676 * 4

/-- Inhibitor diffuses fast enough that D_v f_u + D_u g_v > 0, the necessary
    sign for a finite critical wavenumber. -/
theorem diffusion_drive_positive : driveNumerator > 0 := by
  decide

/-- Equal diffusivities (r = 1) make the same drive equal to the trace, which
    is negative, so the classical Turing drive condition fails. -/
theorem equal_diffusion_drive_negative : traceNumerator < 0 := by
  decide

/-- Detail patch side in hundredths of a micrometre: 12 wavelengths of 1.72 µm. -/
def domainHundredths : Int := 12 * 172

theorem twelve_wavelength_domain : domainHundredths = 2064 := by
  decide

/-- Predecessor Drosophila takahashii cell-tier atlas stripe is 97.6 µm
    (9760 hundredths). This detail wavenumber is more than 50× finer. -/
theorem predecessor_wavenumber_ratio_exceeds_fifty :
    (9760 : Int) > 50 * 172 := by
  decide

/-- Successor Drosophila virilis atlas floor (2.72 µm = 272 hundredths) is
    longer than this 1.72 µm detail wavelength. -/
theorem successor_virilis_floor_longer : (272 : Int) > 172 := by
  decide

/-- A longer periodic patch (18 wavelengths, 30.96 µm) holds strictly more
    critical modes than the 12-wavelength detail patch. -/
theorem longer_domain_holds_more_stripes :
    ((18 * 172) : Int) / 172 > (12 * 172) / 172 := by
  decide

end MathOfBiology.Loop_3209
