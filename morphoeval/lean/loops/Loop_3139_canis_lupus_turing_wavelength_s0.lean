/-
  Loop 3139 — Canis lupus (Canine, MDCK lineage) Turing wavenumber, detail scale.
  Phenomenon scale: 1.781 µm = 1.781e-6 m (tier 3_subcellular_um), Holocene extant (0 Ma).
  Chronoseries 1/1: no predecessor chronospecies is in this series.
  Integer coefficients are exact scalings of the Schnakenberg Jacobian at
  (a, b, γ, D_u, D_v) = (0.10, 1.10, 40, 0.8, 9.6):
  rates × 30 and diffusivities × 5, which preserves the sign of every
  Turing necessary condition below.
-/
namespace MathOfBiology.Loop_3139

/-- Jacobian entries × 30. f_u = 100/3, g_v = -288/5, f_v = 288/5, g_u = -220/3. -/
def fu : Int := 1000
def gv : Int := -1728
def fv : Int := 1728
def gu : Int := -2200

/-- Homogeneous steady state is linearly stable: trace < 0 and det(J) > 0. -/
theorem homogeneous_state_is_stable :
    (fu + gv) < 0 ∧ (fu * gv - fv * gu) > 0 := by
  decide

/-- Diffusivities × 5. D_v / D_u = 12, and long-range inhibition holds. -/
def du : Int := 4
def dv : Int := 48

theorem inhibitor_outruns_activator :
    dv / du = 12 ∧ (dv * fu + du * gv) > 0 := by
  decide

/-- Equal-diffusion ablation: D_v = D_u fails D_v f_u + D_u g_v > 0. -/
theorem equal_diffusion_has_no_turing_instability :
    (du * fu + du * gv) < 0 := by
  decide

/-- Detail scale versus a 2 cm coat-scale pre-pattern, on one 8-wavelength apical face.
    8 × 1781 nm = 14248 nm = 14.248 µm. Age of this chronospecies is 0 Ma. -/
def lambda_nm : Int := 1781
def waves_on_apical_face : Int := 8
def apical_face_nm : Int := waves_on_apical_face * lambda_nm
def coat_scale_nm : Int := 20000000
def age_ma : Int := 0
def chronoseries_length : Int := 1

theorem holocene_detail_scale_versus_coat :
    age_ma = 0
      ∧ chronoseries_length = 1
      ∧ apical_face_nm = 14248
      ∧ coat_scale_nm / lambda_nm > 10000 := by
  decide

end MathOfBiology.Loop_3139
