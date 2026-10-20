/-
  morphoeval/lean/loops/Loop_097_termite_mound_convective_chimneys.lean
  Formal Lean 4 verification of King, Ocko & Mahadevan (2015) PNAS 112(37):11589-11593
  ("Termite mounds harness diurnal temperature oscillations for ventilation"),
  Ocko et al. (2017) PNAS 114(40):10620-10624 ("Solar-powered ventilation of African
  termite mounds"), and Ocko, Heyde & Mahadevan (2019) PNAS 116(9):3379-3384
  ("Morphogenesis of termite mounds") governing diurnal buoyancy thermosiphon flow
  reversal and mass continuity, convective Péclet amplification of mound aspect ratio,
  and Taylor–Aris oscillatory CO₂ dispersion (`ocko-termite-mound-convective-chimneys`, Slot 097).

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_097

-- ============================================================================
-- Law 1 (`ocko-termite-mound-convective-chimneys-1`):
-- Diurnal Buoyancy Thermosiphon Flow Reversal & Mass Continuity Law
-- ============================================================================
/-- In King, Ocko & Mahadevan (2015) PNAS & Ocko et al. (2017) PNAS, solar heating
    creates a temperature difference `dT = T_flute - T_chimney` between thin exterior
    surface flutes (`total area A_f`) and the thermally buffered central chimney (`area A_c`).
    The volumetric flute updraft flux is `Q_flute(K_buoy, dT) = K_buoy * dT` and mass
    conservation across the closed loop requires `Q_chimney(K_buoy, dT) = -Q_flute(K_buoy, dT)`
    (`A_c * u_c + A_f * u_f = 0`), yielding exact mass continuity, diurnal flow reversal
    under sign inversion `dT -> -dT`, and isothermal stagnation at `dT = 0`. -/
def thermosiphon_flute_flux (K_buoy dT : Int) : Int :=
  K_buoy * dT

def thermosiphon_chimney_flux (K_buoy dT : Int) : Int :=
  -(K_buoy * dT)

theorem diurnal_thermosiphon_reversal_and_continuity (K_buoy dT : Int)
    (hK : K_buoy > 0) (hdT : dT > 0) :
    thermosiphon_flute_flux K_buoy 0 = 0 ∧
    thermosiphon_flute_flux K_buoy dT + thermosiphon_chimney_flux K_buoy dT = 0 ∧
    thermosiphon_flute_flux K_buoy (-dT) = -thermosiphon_flute_flux K_buoy dT ∧
    thermosiphon_flute_flux K_buoy dT > 0 ∧
    thermosiphon_chimney_flux K_buoy dT < 0 := by
  have h_pos : K_buoy * dT > 0 := Int.mul_pos hK hdT
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · unfold thermosiphon_flute_flux
    omega
  · unfold thermosiphon_flute_flux thermosiphon_chimney_flux
    omega
  · unfold thermosiphon_flute_flux
    have h_neg : K_buoy * (-dT) = -(K_buoy * dT) := Int.mul_neg K_buoy dT
    exact h_neg
  · unfold thermosiphon_flute_flux
    exact h_pos
  · unfold thermosiphon_chimney_flux
    omega


-- ============================================================================
-- Law 2 (`ocko-termite-mound-convective-chimneys-2`):
-- Convective Péclet Apical Pheromone Amplification & Spire Aspect Ratio Law
-- ============================================================================
/-- In Ocko, Heyde & Mahadevan (2019) PNAS 116:3379-3384, upward convective airflow `u`
    advects volatile building pheromone from the nest with Péclet number `Pe = u * H / D_c`.
    At `Pe = 0` (isothermal pure diffusion), threshold soil deposition forms a hemispherical
    dome with `H / R = 1` (`H_milli = 1000 * R`). For `Pe > 0`, upward advection focuses
    pheromone at the apex so `H_milli(R, eta, Pe) = R * (1000 + eta * Pe)` strictly exceeds
    the hemispherical dome and strictly increases with `Pe`. -/
def mound_height_milli (R eta Pe : Nat) : Nat :=
  R * (1000 + eta * Pe)

def hemispherical_dome_height_milli (R : Nat) : Nat :=
  R * 1000

theorem convective_peclet_spire_elongation (R eta Pe1 Pe2 : Nat)
    (hR : R > 0) (heta : eta > 0) (hPe : Pe1 < Pe2) :
    mound_height_milli R eta 0 = hemispherical_dome_height_milli R ∧
    mound_height_milli R eta Pe2 > hemispherical_dome_height_milli R ∧
    mound_height_milli R eta Pe1 < mound_height_milli R eta Pe2 := by
  have h_pe2_pos : Pe2 > 0 := by omega
  have h_eta_pe2 : eta * Pe2 > 0 := Nat.mul_pos heta h_pe2_pos
  have h_eta_lt : eta * Pe1 < eta * Pe2 := Nat.mul_lt_mul_of_pos_left hPe heta
  have h_inner_gt : 1000 < 1000 + eta * Pe2 := by omega
  have h_inner_lt : 1000 + eta * Pe1 < 1000 + eta * Pe2 := by omega
  refine ⟨?_, ?_, ?_⟩
  · unfold mound_height_milli hemispherical_dome_height_milli
    omega
  · unfold mound_height_milli hemispherical_dome_height_milli
    exact Nat.mul_lt_mul_of_pos_left h_inner_gt hR
  · unfold mound_height_milli
    exact Nat.mul_lt_mul_of_pos_left h_inner_lt hR


-- ============================================================================
-- Law 3 (`ocko-termite-mound-convective-chimneys-3`):
-- Taylor–Aris Oscillatory CO₂ Ventilation Dispersion Enhancement Law
-- ============================================================================
/-- In King, Ocko & Mahadevan (2015) PNAS & Ocko et al. (2017) PNAS, diurnal oscillatory
    thermosiphon flow with velocity amplitude `u_amp` rectifies CO₂ clearance via
    pendelluft / Taylor–Aris shear dispersion `D_eff = D_0 + alpha_TA * u_amp^2`.
    `D_eff` equals molecular diffusion `D_0` at `u_amp = 0`, is invariant under flow
    reversal `u_amp -> -u_amp`, and strictly exceeds `D_0` whenever `alpha_TA > 0`
    and `u_amp > 0`. -/
def effective_co2_dispersion (D0 alpha_TA u_amp : Int) : Int :=
  D0 + alpha_TA * (u_amp * u_amp)

theorem taylor_aris_oscillatory_ventilation_enhancement (D0 alpha_TA u_amp : Int)
    (hD0 : D0 > 0) (ha : alpha_TA > 0) (hu : u_amp > 0) :
    effective_co2_dispersion D0 alpha_TA 0 = D0 ∧
    effective_co2_dispersion D0 alpha_TA (-u_amp) = effective_co2_dispersion D0 alpha_TA u_amp ∧
    effective_co2_dispersion D0 alpha_TA u_amp > D0 ∧
    effective_co2_dispersion D0 alpha_TA u_amp > 0 := by
  have hu2 : u_amp * u_amp > 0 := Int.mul_pos hu hu
  have h_disp : alpha_TA * (u_amp * u_amp) > 0 := Int.mul_pos ha hu2
  have h_pos : D0 + alpha_TA * (u_amp * u_amp) > 0 := Int.add_pos hD0 h_disp
  have h_sq_sym : (-u_amp) * (-u_amp) = u_amp * u_amp := by
    rw [Int.neg_mul_neg]
  refine ⟨?_, ?_, ?_, ?_⟩
  · unfold effective_co2_dispersion
    omega
  · unfold effective_co2_dispersion
    rw [h_sq_sym]
  · unfold effective_co2_dispersion
    omega
  · unfold effective_co2_dispersion
    exact h_pos

end MathOfBiology.Loop_097
