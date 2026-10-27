/-
  Loop 3147 — Chrysomallon squamiferum, Turing wavenumber at the 2.0 nm detail scale.
  Holocene extant, geological age 0 Ma, chronoseries 1/1.
  Integer Jacobian in thousandths from Schnakenberg (a = 0.062, b = 1.084):
    f_u = 892, g_v = -1313, D_u : D_v = 45 : 1620 (ratio 36).
  These integers are the cleared-denominator form of
    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v),
  not a copy of the 20 nm sibling slot.
-/
namespace MathOfBiology.Loop_3147

def lambdaPm : Nat := 2000
def greigiteAPm : Nat := 988
def lambdaPredecessorPm : Nat := 20000
def seriesLength : Nat := 1
def ageMa : Nat := 0

/-- 2.0 nm sits strictly between two and three greigite cubic cells (a = 988 pm). -/
theorem greigite_double_cell_window :
    2 * greigiteAPm < lambdaPm ∧ lambdaPm < 3 * greigiteAPm := by
  decide

/-- Detail scale s0 is tenfold finer than the 20 nm nanocrystal-packing comparison scale. -/
theorem detail_scale_tenfold_finer :
    lambdaPredecessorPm = 10 * lambdaPm := by
  decide

/-- Equal diffusion cannot satisfy the trace condition for diffusion-driven instability. -/
theorem equal_diffusion_trace_stable :
    ((892 : Int) + (-1313)) < 0 := by
  decide

/-- Inhibitor diffuses faster, and the diffusion-weighted numerator of k_c^2 is positive. -/
theorem diffusion_drive_numerator_positive :
    (1620 : Int) > 45 ∧ (1620 * 892 + 45 * (-1313)) > 0 := by
  decide

/-- Division algorithm for the integer critical-wavenumber identity at this slot.
    num = D_v f_u + D_u g_v, den = 2 D_u D_v, num = den * 9 + 73755. -/
theorem kc2_division_algorithm :
    (1620 * 892 + 45 * (-1313)) = (2 * 45 * 1620) * 9 + 73755 := by
  decide

/-- Chronoseries 1/1 at 0 Ma: the Holocene species is the only node and it is extant. -/
theorem holocene_chronoseries_singleton :
    seriesLength = 1 ∧ ageMa = 0 := by
  decide

end MathOfBiology.Loop_3147
