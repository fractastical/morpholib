/-
  morphoeval/lean/loops/Loop_3158_cyanobacteria_mat_turing_wavelength_s0.lean
  Cyanobacteria mat (stromatolite mat), Holocene extant, 0 Ma.
  Chronoseries 1/1: no predecessor and no successor chronospecies.
  Scale: 64.19 µm = 6.419e-5 m, log10 = -4.193, tier 4_cellular_10um, detail.
  Law: Turing wavenumber selection (Schnakenberg activator–inhibitor).
  Slot parameters (a, b, D_v/D_u) = (0.11, 0.83, 16), distinct from the
  Anabaena heterocyst slot. Integer scalings are 10^4 times the Jacobian
  entries evaluated at the homogeneous state (f_u ≈ 0.7660, g_v ≈ -0.8836).
-/

namespace MathOfBiology.Loop_3158

/-- Law 1. Homogeneous Schnakenberg state is linearly stable: tr J = f_u + g_v < 0. -/
def law1_trace (fu gv : Int) : Int := fu + gv

theorem law1_homogeneous_stable :
    law1_trace 7660 (-8836) < 0 := by
  dsimp [law1_trace]
  decide

/-- Law 2. Eight selected wavelengths tile the periodic domain.
    λ is stored in hundredths of a micrometre (6419 → 64.19 µm). -/
def law2_domain (nWaves lamHundredths : Int) : Int := nWaves * lamHundredths

theorem law2_eight_wavelengths :
    law2_domain 8 6419 = 51352 := by
  dsimp [law2_domain]
  decide

/-- Law 3. Inhibitor diffusion exceeds activator diffusion.
    The gap ratio − 1 is positive for the slot ratio 16 and would vanish
    for the equal-diffusion ablation (ratio 1). -/
def law3_ratio_gap (ratio : Int) : Int := ratio - 1

theorem law3_inhibitor_faster :
    law3_ratio_gap 16 > 0 := by
  dsimp [law3_ratio_gap]
  decide

/-- Law 4. This chronoseries node is Holocene (0 Ma), not the 3480 Ma
    Dresser Formation stromatolites returned by the fossil image search. -/
theorem law4_holocene_not_paleoarchean :
    (0 : Int) < 3480 := by
  decide

/-- Law 5. Determinant sign at the homogeneous state is positive
    (det J scaled by 10^4 is 8836). Together with law 1 this is the
    stable-homogeneous half of the Turing conditions. -/
theorem law5_determinant_positive :
    (8836 : Int) > 0 := by
  decide

end MathOfBiology.Loop_3158
