/-
  Loop 3106 — Ambulocetus natans, Turing wavenumber at the 224 µm dermal primordium.
  Scale: 224 µm = 0.000224 m (log10 = -3.650), tier 5_primordium_100um, detail variant.
  Age: Early Eocene, Lutetian, 47.5 Ma. Lineage: Cetacean Limb-to-Flipper Fossil Series.
  This slot is the single Ambulocetus chronospecies. Predecessor Pakicetus attocki
  (shorter reconstructed wavelength, 190 µm) and successor Rodhocetus kasranii
  (longer reconstructed wavelength, 260 µm) are the comparison points, not copies
  of the 1 cm organism-scale sibling.
  Law: k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v). Integer statements below pin the
  sign of the trace-diffusion numerator and the chronoseries ordering.
-/

namespace MathOfBiology.Loop_3106

def lambdaPakicetusUm : Nat := 190
def lambdaAmbulocetusUm : Nat := 224
def lambdaRodhocetusUm : Nat := 260
def primordiumDomainUm : Nat := 1344

def du : Int := 2
def dvEqual : Int := 2
def dvTuring : Int := 40
def fu : Int := 3
def gv : Int := -6

/-- 224 µm lies strictly inside the primordium band (100 µm, 1 mm). -/
theorem detail_scale_inside_primordium_tier :
    100 < lambdaAmbulocetusUm ∧ lambdaAmbulocetusUm < 1000 := by
  decide

/-- Hydration transfer along the limb-to-flipper series lengthens the dermal wavelength. -/
theorem chronoseries_wavelength_increases :
    lambdaPakicetusUm < lambdaAmbulocetusUm ∧
    lambdaAmbulocetusUm < lambdaRodhocetusUm := by
  decide

/-- k ∝ 1/λ, so the wavelength order reverses for wavenumber: k_Pakicetus > k_Ambulocetus > k_Rodhocetus. -/
theorem wavenumber_order_from_wavelength_order :
    lambdaAmbulocetusUm * lambdaRodhocetusUm > lambdaPakicetusUm * lambdaRodhocetusUm ∧
    lambdaRodhocetusUm * lambdaPakicetusUm > lambdaAmbulocetusUm * lambdaPakicetusUm := by
  decide

/-- A 1344 µm primordium patch holds exactly six Ambulocetus wavelengths of 224 µm. -/
theorem ambulocetus_domain_holds_six_wavelengths :
    lambdaAmbulocetusUm * 6 = primordiumDomainUm := by
  decide

/-- The same patch packs more stripes in Pakicetus and fewer in Rodhocetus. -/
theorem stripe_count_decreases_as_wavelength_grows :
    primordiumDomainUm / lambdaPakicetusUm = 7 ∧
    primordiumDomainUm / lambdaAmbulocetusUm = 6 ∧
    primordiumDomainUm / lambdaRodhocetusUm = 5 := by
  decide

/-- Equal diffusivities give D(f_u + g_v) < 0, so the Turing band is closed. -/
theorem equal_diffusion_suppresses_turing_band :
    dvEqual * fu + du * gv < 0 := by
  decide

/-- A faster inhibitor opens a positive trace-diffusion numerator (here 108). -/
theorem faster_inhibitor_opens_turing_band :
    dvTuring * fu + du * gv = 108 ∧ 0 < dvTuring * fu + du * gv := by
  decide

end MathOfBiology.Loop_3106
