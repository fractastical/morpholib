/-
  morphoeval/lean/loops/Loop_018_joint_interzone_clefting.lean
  Formal Lean 4 verification of Kavanagh, Shoval, Winslow, Alon, Leary, Kan & Tabin
  (2013) PNAS 110(45):18190-18195 ("Developmental bias in the evolution of phalanges",
  doi:10.1073/pnas.1315213110), Hiscock, Tschopp & Tabin (2017) Dev. Cell 41:459-465
  (doi:10.1016/j.devcel.2017.04.021), and Brunet et al. (1998) Science 280:1455-1457
  governing the phalangeal Inhibitory Cascade (IC), BMP-GDF5-Noggin Turing-wavefront
  synovial joint interzone segmentation, and Noggin/Gdf5 loss-of-function joint fusion
  (`onuff-joint-interzone-clefting`, Slot 018).

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_018

-- ============================================================================
-- Law 1 (`onuff-joint-interzone-clefting-1`):
-- Kavanagh et al. (2013) Arithmetic Cascade Midpoint & One-Third Invariant Law
-- ============================================================================
/-- In Kavanagh et al. (2013) PNAS 110:18190-18195, sequential segmentation of a
    growing chondrogenic digit ray with mesenchymal activation `A` and proximal
    condensation inhibition `I` produces an arithmetic progression of phalanx
    lengths `P_n = P_1 + (n - 1) * delta` (where `delta = P_1 * (A - I) / I`):
      `P_1 = p1`,  `P_2 = p1 + delta`,  `P_3 = p1 + 2 * delta`.
    We prove that for any initial proximal phalanx length `p1` and step `delta`:
    1. Exact affine midpoint identity: `2 * P_2 = P_1 + P_3`
       (equivalently `P_2 / P_1 = 1/2 + (1/2) * (P_3 / P_1)`).
    2. Exact middle-phalanx one-third total digit proportion:
       `3 * P_2 = P_1 + P_2 + P_3` (equivalently `P_2 / (P_1 + P_2 + P_3) = 1/3`). -/
def phalanx_len (p1 delta n : Int) : Int :=
  p1 + (n - 1) * delta

def digit_total_3 (p1 delta : Int) : Int :=
  phalanx_len p1 delta 1 + phalanx_len p1 delta 2 + phalanx_len p1 delta 3

theorem kavanagh_inhibitory_cascade_midpoint_and_third (p1 delta : Int) :
    2 * phalanx_len p1 delta 2 = phalanx_len p1 delta 1 + phalanx_len p1 delta 3 ∧
    3 * phalanx_len p1 delta 2 = digit_total_3 p1 delta ∧
    phalanx_len 150 (-53) 2 = 97 ∧
    digit_total_3 150 (-53) = 291 := by
  refine ⟨?_, ?_, by decide, by decide⟩
  · unfold phalanx_len
    omega
  · unfold digit_total_3 phalanx_len
    omega

-- ============================================================================
-- Law 2 (`onuff-joint-interzone-clefting-2`):
-- Distal PFR-FGF Tapering (`delta < 0 => P_1 > P_2 > P_3`) & GDF5-Noggin Dispersion
-- ============================================================================
/-- Under a decaying distal Phalanx-Forming Region (PFR) FGF8/Wnt growth gradient
    (`I > A`, so `delta ≤ -1`), phalanges strictly taper from proximal to distal
    (`P_1 > P_2 > P_3`), while the coupled GDF5 (`G`, `D_G = 12`) and Noggin
    (`N`, `D_N = 180`) activator-inhibitor reaction-diffusion Jacobian
    `J = [[160, -240], [190, -210]]` is linearly stable at `q^2 = 0`
    (`tr(J) = -50 < 0`, `det(J) = 12000 > 0`) and undergoes diffusion-driven
    Turing sign inversion (`Δ_disp(q^2 = 1) = -12120 < 0`) to nucleate periodic
    synovial joint interzones. -/
def gdf5_noggin_disp_det (f_G f_N g_G g_N D_G D_N q2 : Int) : Int :=
  (D_G * q2 - f_G) * (D_N * q2 - g_N) - f_N * g_G

theorem pfr_tapering_and_gdf5_noggin_turing_inversion (p1 delta : Int) (hd : delta ≤ -1) :
    phalanx_len p1 delta 1 > phalanx_len p1 delta 2 ∧
    phalanx_len p1 delta 2 > phalanx_len p1 delta 3 ∧
    (160 + (-210) : Int) = -50 ∧
    gdf5_noggin_disp_det 160 (-240) 190 (-210) 12 180 0 = 12000 ∧
    gdf5_noggin_disp_det 160 (-240) 190 (-210) 12 180 1 = -12120 := by
  refine ⟨?_, ?_, by decide, by decide, by decide⟩
  · unfold phalanx_len
    omega
  · unfold phalanx_len
    omega

-- ============================================================================
-- Law 3 (`onuff-joint-interzone-clefting-3`):
-- Noggin Null & Gdf5 Brachypodism Loss-of-Function Joint Cleft Ablation Law
-- ============================================================================
/-- In Brunet et al. (1998) Science 280:1455-1457 and Storm et al. (1994) Nature
    368:639-643, synovial joint interzone clefting requires Noggin-mediated BMP
    antagonism (`nog_dose`) and GDF5 interzone signaling (`gdf5_dose`) to exceed
    the chondrogenic fusion threshold `tau_cleft`. When `nog_dose = 0` (Noggin null),
    effective interzone clefting amplitude `nog_dose * gdf5_dose` vanishes (`= 0`),
    ablating all synovial joint clefts (`N_joints = 0`, leaving 1 continuous fused
    cartilaginous rod) compared to `N_joints = 3` (MCP, PIP, DIP) in wild type. -/
def active_synovial_joints (nog_dose gdf5_dose tau_cleft : Nat) : Nat :=
  let amp := nog_dose * gdf5_dose
  if amp ≥ 3 * tau_cleft then 3
  else if amp ≥ 2 * tau_cleft then 2
  else if amp ≥ tau_cleft then 1
  else 0

def segmented_phalanges (nog_dose gdf5_dose tau_cleft : Nat) : Nat :=
  let j := active_synovial_joints nog_dose gdf5_dose tau_cleft
  if j == 0 then 1 else j

theorem noggin_gdf5_joint_cleft_ablation (gdf5_dose : Nat) :
    active_synovial_joints 0 gdf5_dose 25 = 0 ∧
    segmented_phalanges 0 gdf5_dose 25 = 1 ∧
    active_synovial_joints 7 8 25 = 2 ∧
    active_synovial_joints 10 10 25 = 3 ∧
    segmented_phalanges 10 10 25 = 3 := by
  refine ⟨?_, ?_, by decide, by decide, by decide⟩
  · simp [active_synovial_joints]
  · simp [segmented_phalanges, active_synovial_joints]

-- ============================================================================
-- Law 4 (`onuff-joint-interzone-clefting-4`, Pass 3):
-- Reduced Biot Poroelastic Synovial Cavitation Gating Law
-- (Mow et al. 1980 biphasic cartilage; Edwards et al. 1994 & Dowthwaite et al. 1998
--  hyaluronan-driven loss of interzone cohesion; Drachman & Sokoloff 1966 and
--  Kahn et al. 2009 immobilisation / muscle-less joint fusion)
-- ============================================================================
/-- In the reduced Biot poroelastic cavitation model of
    `2016-onuff-joint-interzone-clefting.py` (`simulate_poroelastic_cavitation`),
    Gdf5+ interzone cells (`gdf5`) synthesise hyaluronan at a mechanosensitive rate
    proportional to embryonic muscle loading (`m_load`); at steady state the HA
    content is `H_ss = k_H * m_load * gdf5` and the linearised Donnan osmotic
    pore pressure is `p_ss = beta * H_ss`. The solid interzone cohesion is
    `sigma_coh = sigma_0 - kappa * gdf5`, and a synovial cavity opens iff
    `p_ss > sigma_coh`. We prove:
    1. `m_load = 0` (paralysed / muscle-less embryo) forces `p_ss = 0`, so no cavity
       opens wherever cohesion is positive, however strong the Gdf5 interzone is
       (Drachman & Sokoloff 1966; Kahn et al. 2009);
    2. `gdf5 = 0` (no interzone, e.g. the Noggin null) also forces `p_ss = 0`, so no cavity;
    3. steady pore pressure is monotone non-decreasing in muscle loading
       (the `--m-load` sweep is monotone);
    4. at the wild-type integer calibration (`beta = 1, k_H = 1, m_load = 10,
       gdf5 = 10, sigma_0 = 100, kappa = 5`, i.e. p = 100 > 50 in hundredths of
       the dimensionless script units beta_osm = 1, sigma_0 = 1, kappa_G = 0.5)
       the cavity opens, whereas at `m_load = 3` (p = 30 < 50) it does not. -/
def ha_steady (k_H m_load gdf5 : Nat) : Nat :=
  k_H * gdf5 * m_load

def pore_pressure_steady (beta k_H m_load gdf5 : Nat) : Nat :=
  beta * ha_steady k_H m_load gdf5

def interzone_cohesion (sigma_0 kappa gdf5 : Int) : Int :=
  sigma_0 - kappa * gdf5

def cavity_opens (p : Nat) (sigma_coh : Int) : Bool :=
  decide ((p : Int) > sigma_coh)

theorem poroelastic_cavitation_requires_load_and_interzone
    (beta k_H gdf5 m1 m2 : Nat) (sigma_coh : Int) (hpos : 0 < sigma_coh)
    (hm : m1 ≤ m2) :
    pore_pressure_steady beta k_H 0 gdf5 = 0 ∧
    pore_pressure_steady beta k_H m1 0 = 0 ∧
    cavity_opens (pore_pressure_steady beta k_H 0 gdf5) sigma_coh = false ∧
    pore_pressure_steady beta k_H m1 gdf5 ≤ pore_pressure_steady beta k_H m2 gdf5 ∧
    cavity_opens (pore_pressure_steady 1 1 10 10) (interzone_cohesion 100 5 10) = true ∧
    cavity_opens (pore_pressure_steady 1 1 3 10) (interzone_cohesion 100 5 10) = false := by
  refine ⟨?_, ?_, ?_, ?_, by decide, by decide⟩
  · simp [pore_pressure_steady, ha_steady]
  · simp [pore_pressure_steady, ha_steady]
  · simp [cavity_opens, pore_pressure_steady, ha_steady]
    omega
  · unfold pore_pressure_steady ha_steady
    exact Nat.mul_le_mul_left beta (Nat.mul_le_mul_left (k_H * gdf5) hm)

-- ============================================================================
-- Law 5 (`onuff-joint-interzone-clefting-5`, Pass 3):
-- Darcy Lens-Cavity Midline Maximum Law
-- ============================================================================
/-- With free drainage (`p = 0`) at the perichondrial walls `y = ±W` and a uniform
    osmotic source `s ≥ 0` across the interzone, the steady 1D Darcy pressure
    `κ p'' + s = 0` is the parabola `2 κ p(y) = s (W² − y²)`. Writing
    `darcy_lens_pressure s W y = s * (W*W − y*y)` for the scaled pressure, the
    pressure (and hence the cavitation drive `p − σ_coh`) is maximal on the digit
    midline `y = 0` and vanishes at the walls, which is why the simulated synovial
    cavity nucleates as a transverse lens centred on the ray axis. -/
def darcy_lens_pressure (s W y : Int) : Int :=
  s * (W * W - y * y)

theorem darcy_lens_cavity_midline_maximum (s W y : Int) (hs : 0 ≤ s) :
    darcy_lens_pressure s W y ≤ darcy_lens_pressure s W 0 ∧
    darcy_lens_pressure s W W = 0 ∧
    darcy_lens_pressure 3 10 0 = 300 ∧
    darcy_lens_pressure 3 10 6 = 192 := by
  refine ⟨?_, ?_, by decide, by decide⟩
  · unfold darcy_lens_pressure
    have hyy : 0 ≤ y * y := by
      rcases Int.le_total 0 y with h | h
      · exact Int.mul_nonneg h h
      · exact Int.mul_nonneg_of_nonpos_of_nonpos h h
    have h1 : W * W - y * y ≤ W * W - 0 * 0 := by
      generalize W * W = ww
      generalize y * y = yy at hyy ⊢
      omega
    exact Int.mul_le_mul_of_nonneg_left h1 hs
  · unfold darcy_lens_pressure
    generalize W * W = ww
    omega

end MathOfBiology.Loop_018
