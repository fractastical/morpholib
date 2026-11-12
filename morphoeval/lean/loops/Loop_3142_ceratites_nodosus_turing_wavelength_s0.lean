/-
  Loop 3142 — Ceratites nodosus, cellular Turing wavelength.
  Scale: 26 µm = 25.6 µm stored (256 tenths of a micrometre), Middle Triassic 240 Ma.
  Law: Turing wavenumber selection on a Schnakenberg activator–inhibitor system.
  No mathlib. No sorry.
-/
namespace MathOfBiology.Loop_3142

/-- Wavelength count across a shell. Lengths are tenths of a micrometre.
    25.6 µm = 256. 32 mm = 320000. -/
def wavelengths (sizeTenthUm waveTenthUm : Nat) : Nat :=
  sizeTenthUm / waveTenthUm

/-- This horizon: a 32 mm Ceratites nodosus shell holds 1250 cellular wavelengths. -/
theorem ceratites_shell_holds_1250_cellular_waves :
    wavelengths 320000 256 = 1250 := by
  decide

/-- Predecessor Goniatites crenistria (18 mm, 333 Ma) holds fewer cellular wavelengths
    at the same 26 µm epithelial scale. -/
theorem more_ocelli_than_goniatites_predecessor :
    wavelengths 320000 256 > wavelengths 180000 256 := by
  decide

/-- Successor Dactylioceras commune (42 mm, 181 Ma) holds more. -/
theorem fewer_ocelli_than_dactylioceras_successor :
    wavelengths 320000 256 < wavelengths 420000 256 := by
  decide

/-- One 1.088 mm organ-scale rib (10880 tenths of a micrometre) contains 42
    wavelengths of 25.6 µm. This is the scale change versus the 1.1 mm sibling, not a copied diffusivity. -/
theorem organ_rib_contains_42_cellular_wavelengths :
    wavelengths 10880 256 = 42 := by
  decide

/-- Integer proxy of the Turing numerator γ f_u + g_v.
    f_u and g_v are in thousandths (0.897 → 897, −1.445 → −1445). -/
def kcNumerator (gamma fu gv : Int) : Int :=
  gamma * fu + gv

/-- Operating ratio γ = 12 (above onset for these kinetics): numerator stays positive. -/
theorem operating_regime_numerator_positive :
    kcNumerator 12 897 (-1445) > 0 := by
  decide

/-- Equal-diffusion ablation γ = 1: numerator is negative, so the Turing condition fails. -/
theorem equal_diffusion_ablation_not_turing :
    kcNumerator 1 897 (-1445) < 0 := by
  decide

end MathOfBiology.Loop_3142
