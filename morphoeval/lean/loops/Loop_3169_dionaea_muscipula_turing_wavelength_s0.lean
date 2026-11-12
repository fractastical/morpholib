/-
  morphoeval/lean/loops/Loop_3169_dionaea_muscipula_turing_wavelength_s0.lean
  Slot 3169. *Dionaea muscipula* (Venus flytrap), Holocene (Extant), 0 Ma.
  Scale: 24 µm (2.4e-5 m, log10 = -4.620), tier 4_cellular_10um, detail variant.
  Law: Turing wavenumber k_c for an adaxial epidermal pigment / ocellus mode.
  Chronoseries 1/1 has no other age. The predecessor compared here is the
  subcellular 8 µm domain (no unstable mode); the successor is the 480 µm
  tissue patch (same k_c, twenty wavelengths instead of eight).
-/
namespace MathOfBiology.Loop_3169

/-- Law 1. Turing diffusion margin Δ = D_v * f_u + D_u * g_v.
    Integer shadow of the Schnakenberg Jacobian (f_u, g_v) = (0.8, -1)
    scaled by 5, with diffusivity ratio 18 represented as (D_u, D_v) = (5, 90). -/
def turing_margin (Du Dv fu gv : Int) : Int :=
  Dv * fu + Du * gv

theorem patterned_margin_positive :
    turing_margin 5 90 4 (-5) = 335 := by
  decide

/-- Law 2. The well-mixed steady state is stable: trace(J) = f_u + g_v < 0,
    so diffusionless perturbations decay. -/
theorem homogeneous_trace_stable :
    ((4 : Int) + (-5)) = -1 := by
  decide

/-- Law 3. Equal-diffusion ablation flips the margin negative, so the
    Turing band disappears. -/
theorem equal_diffusion_ablation :
    turing_margin 5 5 4 (-5) = -5 := by
  decide

/-- Law 4. This slot's 192 µm adaxial patch holds exactly eight 24 µm wavelengths. -/
theorem domain_holds_eight_wavelengths :
    (8 : Int) * 24 = 192 := by
  decide

/-- The 8 µm subcellular predecessor is shorter than one critical wavelength. -/
theorem predecessor_shorter_than_one_wavelength :
    (8 : Int) < 24 := by
  decide

/-- The 480 µm tissue-scale successor holds twenty wavelengths at the same k_c. -/
theorem successor_tissue_holds_twenty_wavelengths :
    (20 : Int) * 24 = 480 := by
  decide

end MathOfBiology.Loop_3169
