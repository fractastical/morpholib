# Loop Pass 2 (Remediation): Growing-Domain Turing for the Cow

**Remediates:** Pass-1 verdict — static adult-flank Turing makes 142 uniform spots; the cow has one body-scale patch.
**Changes:** `simulate_growing()` (pre-pattern on an 80×60 domain, grown twice to 160×120 with RD continuing after each growth step — Murray's scale argument), free coat polarity (pigment = low-activator phase; spots become white enclaves), and an explicit morphometric fit objective (`morphometric_distance`: log-count ratios for both phases + coverage + largest-share + diameter CVs) grid-searched over mode × γ × polarity × threshold quantile (48 candidates).
**Result:** best = growing, γ=0.25, q=0.65 — pigment patches 2 (cow 3), coverage 0.65 (cow 0.77), largest share 0.942 (cow 0.992), largest diameter 0.765 (cow 0.756), enclaves 12 (cow 12). Distance 5.22 (Pass-1 presentation) → **1.19**.
**Audit:** `2026-10-26-bos-taurus-turing-wavelength-audit-v2` — **57.6 → 66.0** (specimen-match 20 → 45).
**Honesty note:** this is a *fit* on n=1 (γ, polarity, threshold chosen against the same flank that scores them). It shows a growing Turing system *can* reach cow-like patch statistics; it does not validate that cows do this. Animation regenerated: the pattern now grows and stretches with the domain.
**Next (R6):** enclaves still smooth/round and too interior — add boundary roughness to the morphometrics and a roughening mechanism.
