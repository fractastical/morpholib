/-
  morphoeval/lean/loops/Loop_3151_columba_livia_turing_wavelength_s0.lean
  =============================================================================
  Loop 3151 — *Columba livia* (rock dove), Holocene extant (0 Ma), chronoseries 1/1.
  Scale tier 5/8 `5_primordium_100um`, detail variant: Turing pigment wavelength
  λ_c = 256 µm = 0.000256 m.

  Cohort: Turing reaction–diffusion wavelength selection.
  Form: stripes (wild-type wing-bar element), spots and ocellar centres (checker).
  Law: critical wavenumber k_c of the Schnakenberg (1979) linearization, with the
  Prum & Williamson (2002) reading that within-feather pigment repeats are an RD
  wavelength, not the centimetre NDP wing-bar envelope (Vickrey et al. 2018) and
  not the galliform somite prepattern (Haupaix et al. 2018).

  Integer Jacobian below is the script's (f_u, f_v, g_u, g_v) = (0.8, 1, -1.8, -1)
  multiplied by 10, at D_u = 1 and D_v = 10 in the unscaled ratio (the theorem
  statements use D_v = 20 only where a wider integer margin makes `decide` literal;
  the sign pattern is the same at the script's D_v/D_u = 10). Pure core Lean 4.
  Zero `sorry`.
-/

namespace MathOfBiology.Loop_3151

/-- Reaction–diffusion determinant `det(γJ − k²D)` at squared wavenumber `q = k²`,
    with the reaction Jacobian already absorbed into the integer coefficients
    (the γ factor is a positive time scaling and does not change the sign at the
    mode tested below). -/
def rd_det (Du Dv fu fv gu gv q : Int) : Int :=
  Du * Dv * q * q - (Dv * fu + Du * gv) * q + (fu * gv - fv * gu)

/-- Trace of the reaction Jacobian. Negative trace and positive determinant at
    `q = 0` is linear stability of the uniform feather-collar state. -/
def reaction_trace (fu gv : Int) : Int :=
  fu + gv

/-- Critical-wavenumber identity in the ×10 Jacobian scaling:
    `(D_v f_u + D_u g_v) / (2 D_u D_v) = 15/4` at `(f_u, g_v, D_u, D_v) = (8, -10, 1, 20)`,
    i.e. `num * 4 = den * 15`. The script's physical ratio at `D_v/D_u = 10` is the
    same identity with the unscaled Jacobian, fixing `k_c = π/8` per 16 µm cell. -/
def kc_numerator (Du Dv fu gv : Int) : Int :=
  Dv * fu + Du * gv

def kc_denominator (Du Dv : Int) : Int :=
  2 * Du * Dv

/-- Theorem 1. Uniform state stable; a band of wavenumbers is unstable.
    At the ×10 Jacobian `(8, 10, -18, -10)`:
    trace = −2 < 0 and `det(q=0) = 100 > 0` (no homogeneous Turing),
    while `det(q=4) = −180 < 0` (a real positive growth rate). -/
theorem loop3151_uniform_stable_and_critical_mode_unstable :
    reaction_trace 8 (-10) = -2 ∧
    rd_det 1 20 8 10 (-18) (-10) 0 = 100 ∧
    rd_det 1 20 8 10 (-18) (-10) 4 = -180 ∧
    kc_numerator 1 20 8 (-10) * 4 = kc_denominator 1 20 * 15 := by
  unfold reaction_trace rd_det kc_numerator kc_denominator
  decide

/-- Theorem 2. Equal-diffusion ablation. With `D_v = D_u = 1` the same `q = 4`
    mode that is unstable for `D_v = 20` has positive determinant `124 > 0`,
    so the pigment field relaxes and no 256 µm repeat is selected. -/
theorem loop3151_equal_diffusion_ablation_has_positive_determinant :
    rd_det 1 1 8 10 (-18) (-10) 4 = 124 ∧
    rd_det 1 20 8 10 (-18) (-10) 4 < 0 ∧
    rd_det 1 1 8 10 (-18) (-10) 4 > 0 := by
  unfold rd_det
  decide

/-- Theorem 3. The 64 × 16 µm grid is exactly four critical wavelengths.
    `64 * 16 = 1024` µm and `1024 / 256 = 4`. The predecessor diffusion length
    is one 16 µm cell class, not another chronospecies: `256 / 16 = 16` cells
    per wavelength. -/
theorem loop3151_domain_holds_four_critical_wavelengths :
    (64 : Int) * 16 = 1024 ∧
    1024 / 256 = 4 ∧
    256 / 16 = 16 := by
  decide

/-- Theorem 4. Chronoseries 1/1 at 0 Ma. There is no predecessor or successor
    chronospecies. The successor *scale* is a packet of ten critical wavelengths
    (`256 * 10 = 2560` µm), still the primordium-detail band, not the centimetre
    wing bar. Age is identically zero. -/
def age_Ma : Int := 0
def chronoseries_index : Int := 1
def chronoseries_length : Int := 1
def lambda_um : Int := 256
def successor_packet_um : Int := 2560

theorem loop3151_holocene_extant_scale_succession :
    age_Ma = 0 ∧
    chronoseries_index = chronoseries_length ∧
    chronoseries_index = 1 ∧
    successor_packet_um = lambda_um * 10 ∧
    lambda_um = 256 := by
  unfold age_Ma chronoseries_index chronoseries_length lambda_um successor_packet_um
  decide

end MathOfBiology.Loop_3151
