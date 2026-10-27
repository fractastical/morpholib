/-
  Loop 3189 — Drosophila hydei detail-scale Turing wavelength (slot s0).
  Scale: 2.344 µm = 2344 nm (displayed 2.3 µm), log10(2.344e-6 m) = -5.630, tier 3_subcellular_um.
  Age: Holocene extant, 0 Ma.
  Jacobian below is the Schnakenberg (a=0.10, b=0.90, γ=1) linearisation scaled by 10:
    f_u=0.8 → 8, f_v=1 → 10, g_u=-1.8 → -18, g_v=-1 → -10, D_u=1, D_v=20.
  Predecessor D. heteroneura 175400 nm; same-species primordium slot 117200 nm;
  successor D. kikkawai 93410 nm.
-/
namespace MathOfBiology.Loop_3189

def fu : Int := 8
def fv : Int := 10
def gu : Int := -18
def gv : Int := -10
def Du : Int := 1
def Dv : Int := 20

def lambda_nm_hydei_detail : Int := 2344
def lambda_nm_heteroneura : Int := 175400
def lambda_nm_kikkawai : Int := 93410
def lambda_nm_hydei_primordium : Int := 117200

/-- Trace of the reaction Jacobian is negative: the well-mixed state is stable. -/
theorem trace_negative : fu + gv < 0 := by
  decide

/-- Determinant f_u g_v - f_v g_u is positive (scaled by 100 relative to det = 1). -/
theorem determinant_positive : fu * gv - fv * gu > 0 := by
  decide

/-- Diffusion-driven instability: (D_v f_u + D_u g_v)² > 4 D_u D_v det. -/
theorem diffusion_driven_instability :
    (Dv * fu + Du * gv) * (Dv * fu + Du * gv) >
      4 * Du * Dv * (fu * gv - fv * gu) := by
  decide

/-- Equal diffusion fails the necessary condition h = D (f_u + g_v) > 0. -/
theorem equal_diffusion_not_turing : Du * fu + Du * gv < 0 := by
  decide

/-- Same species, coarser tier: 117.2 µm is exactly 50 × 2.344 µm. -/
theorem primordium_is_fiftyfold_coarser :
    lambda_nm_hydei_primordium = 50 * lambda_nm_hydei_detail := by
  decide

/-- Predecessor picture-wing band (175.4 µm) is more than 74× this detail wavelength. -/
theorem predecessor_wavelength_coarser :
    lambda_nm_heteroneura > 74 * lambda_nm_hydei_detail := by
  decide

/-- Successor tergite scale (93.41 µm) is more than 39× this detail wavelength. -/
theorem successor_wavelength_coarser :
    lambda_nm_kikkawai > 39 * lambda_nm_hydei_detail := by
  decide

end MathOfBiology.Loop_3189
