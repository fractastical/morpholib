/-
  morphoeval/lean/loops/Loop_3174_drosophila_affinis_turing_wavelength_s0.lean
  Slot 3174 — Drosophila affinis, Holocene (0 Ma), detail-tier Turing wavelength.

  Scale: λ_c = 2.444 µm = 2444 nm (2.444e-6 m, tier 3_subcellular_um).
  Chronoseries neighbors (project order, not an ancestor claim):
    predecessor Drosophila berryi  98.43 µm at 19 Ma
    successor  Drosophila americana 138.7 µm at 0 Ma

  Integer encoding used by the instability theorems (milli-units of the
  Jacobian entries and of Du, Dv in µm² per time):
    f_u = 905, g_v = -1103, det J = 1103, D_u = 60, D_v = 601
  which tracks a = 0.05, b = 1, D_v/D_u = 10, λ_c = 2.444 µm.
  The equal-diffusivity ablation sets D_v := D_u.
-/

namespace MathOfBiology.Loop_3174

def lambdaDetailNm : Int := 2444
def lambdaBerryiNm : Int := 98430
def lambdaAmericanaNm : Int := 138700

/-- The Holocene detail wavelength sits strictly below both chronoseries neighbors. -/
theorem detail_wavelength_below_chronospecies :
    lambdaDetailNm < lambdaBerryiNm ∧ lambdaDetailNm < lambdaAmericanaNm := by
  decide

def fu : Int := 905
def gv : Int := -1103
def detJ : Int := 1103

/-- Without diffusion the Schnakenberg steady state is asymptotically stable:
    trace(J) < 0 and det(J) > 0. -/
theorem homogeneous_steady_state_stable :
    fu + gv < 0 ∧ detJ > 0 := by
  decide

def Du : Int := 60
def Dv : Int := 601

/-- Reaction-diffusion trace coefficient D_v f_u + D_u g_v. -/
def rdTrace (dU dV fU gV : Int) : Int := dV * fU + dU * gV

/-- The inhibitor diffuses faster than the activator at the detail-tier ratio 10. -/
theorem inhibitor_diffuses_faster : Dv > Du := by
  decide

/-- Diffusion-driven instability at λ_c = 2.444 µm:
    the trace coefficient is positive and the discriminant
    (D_v f_u + D_u g_v)² − 4 D_u D_v det(J) is positive. -/
theorem diffusion_driven_instability_at_detail_scale :
    rdTrace Du Dv fu gv > 0 ∧
      rdTrace Du Dv fu gv * rdTrace Du Dv fu gv - 4 * Du * Dv * detJ > 0 := by
  decide

/-- Negative control: equal diffusivities (D_v = D_u) make the trace coefficient
    negative, so the Turing condition fails and no finite wavelength is selected. -/
theorem equal_diffusivity_ablation_not_turing :
    rdTrace Du Du fu gv < 0 := by
  decide

end MathOfBiology.Loop_3174
