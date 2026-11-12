/-
  Loop 3185 — Drosophila ficusphila longitudinal sex-comb tooth array.
  Scale: 2.072 µm (2.072e-6 m, log10 = -5.684), tier 3_subcellular_um, Holocene 0 Ma.
  Schnakenberg Jacobian at (a, b) = (1/10, 9/10), reported in milli-units:
    f_u = 800, f_v = 1000, g_u = -1800, g_v = -1000, det = 1000 (meaning 1.000).
  Diffusion ratio r = D_v / D_u = 20 on the focal species.
  Wavelengths in nanometres at this tier (scenario densities 1, 2/3, 1/2):
    ficusphila 2072 nm, eugracilis 3108 nm, grimshawi 4144 nm.
  These integers are not the 100 µm sibling slot's (60, 17) margin.
-/
namespace MathOfBiology.Loop_3185

/-- Trace of the Schnakenberg Jacobian in milli-units: f_u + g_v. -/
def trace_milli : Int := 800 + (-1000)

theorem trace_is_negative : trace_milli < 0 := by
  decide

/-- Determinant f_u g_v − f_v g_u in milli-units (equals 1000 · det, det = 1). -/
def det_milli : Int := 800 * (-1000) - 1000 * (-1800)

theorem determinant_is_positive : det_milli > 0 := by
  decide

/-- Turing diffusion inequality at ratio r, with the milli-Jacobian.
    (r f_u + g_v)² − 4 r det_milli · 1000  > 0  iff the critical mode is unstable.
    det_milli is 1_000_000 in the raw product above; the threshold uses det = 1000
    so that 4 r det matches the bos-free scaling (r=20 → 225_000_000 > 80_000_000). -/
def turing_margin (r fu gv detScale : Int) : Int :=
  (r * fu + gv) * (r * fu + gv) - 4 * r * detScale * 1000

theorem focal_ratio_is_unstable :
    turing_margin 20 800 (-1000) 1000 > 0 := by
  decide

theorem equal_diffusion_is_stable :
    turing_margin 1 800 (-1000) 1000 < 0 := by
  decide

/-- Critical wavelengths in nanometres. Packed ficusphila (2072) is shorter than
    the reduced-tooth predecessor (3108) and the picture-wing successor (4144). -/
def lam_fic_nm : Int := 2072
def lam_eug_nm : Int := 3108
def lam_grim_nm : Int := 4144

theorem focal_wavelength_is_shortest :
    lam_fic_nm < lam_eug_nm ∧ lam_fic_nm < lam_grim_nm := by
  decide

/-- Ridge count on a 40 µm (40_000 nm) window, integer division. -/
def ridge_count (lam_nm : Int) : Int := 40000 / lam_nm

theorem packed_comb_has_more_ridges :
    ridge_count lam_fic_nm > ridge_count lam_eug_nm
      ∧ ridge_count lam_fic_nm > ridge_count lam_grim_nm := by
  decide

end MathOfBiology.Loop_3185
