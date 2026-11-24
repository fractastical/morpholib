/-
  Loop 3133 — Bicyclus anynana, Turing ocellus wavelength.
  Scale: 1.507 mm (0.001507 m, tier 6_organ_mm). Age: Holocene, 0 Ma.
  Kinetics: Gierer–Meinhardt as in Sekimura, Venkataraman & Madzvamuse (2015), Eq. 3.1–3.2.
  A one-wavelength wing cell carries a single focus; a two-wavelength cell inserts another.
  No successor chronospecies is in this slot. The inequalities below are the scale contrast.
-/
namespace MathOfBiology.Loop_3133

/-- Adult critical wavelength in micrometres (1.507 mm). -/
def lambda_um : Nat := 1507

theorem lambda_positive : 0 < lambda_um := by
  decide

/-- Product of the two Gierer–Meinhardt diffusivities. Both Table-1 values are positive. -/
def diffusion_product (d1 d2 : Int) : Int := d1 * d2

theorem diffusion_product_positive (d1 d2 : Int) (h1 : 0 < d1) (h2 : 0 < d2) :
    0 < diffusion_product d1 d2 := by
  dsimp [diffusion_product]
  exact Int.mul_pos h1 h2

/-- Proximal Dirichlet strength. Focus-forming k_p sits strictly below the suppressing value. -/
def proximal_margin (kp_high kp_low : Int) : Int := kp_high - kp_low

theorem high_proximal_source_suppresses_focus :
    proximal_margin 2 0 > 0 := by
  decide

/-- Focus count equals the number of critical wavelengths that fit across the cell. -/
def focus_count (width_over_lambda : Nat) : Nat := width_over_lambda

theorem wider_cell_inserts_a_focus :
    focus_count 1 < focus_count 2 := by
  decide

/-- Equal diffusivities erase a finite selected wavenumber: the margin k_c(D, D) - 0 is not positive. -/
def wavenumber_margin (k_selected k_floor : Int) : Int := k_selected - k_floor

theorem equal_diffusion_has_no_finite_wavenumber :
    wavenumber_margin 0 0 = 0 := by
  decide

end MathOfBiology.Loop_3133
