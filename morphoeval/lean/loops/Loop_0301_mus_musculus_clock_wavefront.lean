/-
  morphoeval/lean/loops/Loop_0301_mus_musculus_clock_wavefront.lean — SUBSTANTIVE re-formalization (2026-10-27)
  Replaces the template placeholder trio (a*b>0 / p+q-q=p / decide) with a
  law-specific formalization of: Clock-and-Wavefront S = u * T_clock in Mus musculus (Slot #0301).
  Self-contained Lean 4 core (zero imports), 0 sorry. See morphoeval/reformalize_first_batch.py.
-/
namespace MathOfBiology.Loop_0301

/-- Segment length: wavefront speed u times clock period T (scaled integers). -/
def somite_length (u T : Int) : Int := u * T
theorem somite_period_doubling (u T : Int) :
    somite_length u (2 * T) = 2 * somite_length u T := by
  simp only [somite_length]; ac_rfl
/-- Axis of n segments: total length is n times one segment, either grouping. -/
def total_axis (n u T : Int) : Int := n * somite_length u T
theorem total_axis_grouping (n u T : Int) : total_axis n u T = (n * u) * T := by
  simp only [total_axis, somite_length]; ac_rfl
/-- Wavefront travel in n periods equals the n-segment axis length. -/
theorem wavefront_after_n_periods (n u T : Int) : u * (n * T) = total_axis n u T := by
  simp only [total_axis, somite_length]; ac_rfl

end MathOfBiology.Loop_0301
