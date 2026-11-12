/-
  morphoeval/lean/loops/Loop_3168_dictyostelium_discoideum_turing_wavelength_s0.lean
  Slot 3168 — Dictyostelium discoideum (social amoeba), chronoseries 1/1.
  Scale: 2.304 µm = 2304 nm (2.304e-6 m), tier 3_subcellular_um, detail variant.
  Age: Holocene (Extant), 0 Ma. No earlier fossil chronospecies in this series.
  Law: Turing wavenumber k_c for Schnakenberg kinetics on one vegetative cell.
  Predecessor regime: a 200 nm molecular domain admits zero copies of λ_c.
  Successor regime: millimetre-scale extracellular cAMP waves are a longer wavelength.
-/
namespace MathOfBiology.Loop_3168

/-- Diffusion-ratio gate for this slot's Schnakenberg Jacobian.
    f_u is carried as 709/1000 and g_v as -1210/1000, so
    δ f_u + g_v has the sign of δ * 709 - 1210. The cortical ratio δ = 24
    lies above onset. -/
def law1_diffusion_margin (delta fu_milli gv_milli : Int) : Int :=
  delta * fu_milli + gv_milli

theorem law1_cortical_ratio_exceeds_onset :
    law1_diffusion_margin 24 709 (-1210) > 0 := by
  decide

/-- Integer mode count N = L / λ_c on a periodic cortical patch. -/
def law2_mode_count (length_nm wavelength_nm : Int) : Int :=
  length_nm / wavelength_nm

theorem law2_five_modes_in_vegetative_cell :
    law2_mode_count 11520 2304 = 5 := by
  decide

theorem law2_molecular_predecessor_admits_no_mode :
    law2_mode_count 200 2304 = 0 := by
  decide

/-- Gap between a 1 mm aggregation-scale successor wavelength and λ_c = 2304 nm.
    Holocene age of this chronospecies is 0 Ma; the change tracked here is spatial. -/
def law3_scale_gap (successor_nm cortical_nm : Int) : Int :=
  successor_nm - cortical_nm

theorem law3_aggregation_successor_is_longer :
    law3_scale_gap 1000000 2304 > 0 := by
  decide

end MathOfBiology.Loop_3168
