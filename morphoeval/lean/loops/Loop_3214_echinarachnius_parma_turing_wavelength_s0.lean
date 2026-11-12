/-
  Loop 3214 — Echinarachnius parma, Holocene (0 Ma), Turing wavelength 60 µm.

  The chronoseries has one member, so geological age is not a parameter of the
  wavelength count. What changes is the patch length on the scale ladder:

    10 µm  predecessor tier   n = 0   (the unstable mode does not fit; pattern quenches)
    480 µm this detail patch  n = 8   (the 60 µm ocellus wavelength, eight times)
    600 µm successor tier     n = 10

  Integer kinetics mirror the Schnakenberg Jacobian at the homogeneous state
  (fu = 0.8, |gv| = 1), scaled by 5 so the trace gap stays in Nat:
  fuScaled = 4, gvAbs = 5. Ratio Dv/Du = 12 is the Holocene calibration;
  ratio 1 is the equal-diffusion ablation.
-/
namespace MathOfBiology.Loop_3214

/-- How many 60 µm wavelengths fit in a patch of length `L` micrometres. -/
def nWavelengths (L : Nat) : Nat := L / 60

/-- Dv·fu − Du·|gv| with the scaled Jacobian entries. Positive iff a Turing band exists. -/
def turingTraceGap (Du Dv fuScaled gvAbs : Nat) : Int :=
  (Dv * fuScaled : Int) - (Du * gvAbs : Int)

theorem nWavelengths_monotone (L1 L2 : Nat) (h : L1 ≤ L2) :
    nWavelengths L1 ≤ nWavelengths L2 := by
  unfold nWavelengths
  exact Nat.div_le_div_right h

/-- Age does not appear. A 1/1 Holocene series has no older chronospecies to change λ. -/
theorem wavelength_count_independent_of_age (L ageMa : Nat) :
    nWavelengths L = L / 60 := by
  unfold nWavelengths
  rfl

theorem holocene_480um_patch_holds_eight :
    nWavelengths 480 = 8 := by decide

theorem predecessor_10um_quenches :
    nWavelengths 10 = 0 := by decide

theorem successor_600um_holds_ten :
    nWavelengths 600 = 10 := by decide

theorem equal_diffusion_closes_turing (fuScaled gvAbs : Nat) (h : fuScaled < gvAbs) :
    turingTraceGap 1 1 fuScaled gvAbs < 0 := by
  unfold turingTraceGap
  omega

theorem holocene_ratio_12_opens_turing :
    turingTraceGap 1 12 4 5 > 0 := by
  unfold turingTraceGap
  decide

end MathOfBiology.Loop_3214
