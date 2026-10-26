# Loop Pass 6: Held-Out Multi-Specimen Validation (R2) + Mask Cleanup (R3, partial)

**Specimens:** 9 further Holsteins from Wikimedia Commons (PD / CC BY / CC BY-SA; titles, licences, URLs and SHA-256 in the reference JSON; flank ROIs set by hand from gridded views), total **10 animals**. Specimens 02–10 were never used in any Pass 1–5 fit.
**Mask cleanup:** 2-iteration majority filter on the Otsu mask (`clean_mask`) — cow-01 enclave roughness 11.9 → 9.7, i.e. part of the headline gap was photo speckle; the RD gap (sim ≈ 5) remains. Rectangular ROIs retained (full coat mask = R3b).
**Held-out result** (5 seeds, coverage calibrated per specimen, distance v2 on unfitted statistics):

| Model | Held-out distance (9 cows) | Wins |
|---|---:|---:|
| RD static γ=1 | 6.53 | 0/9 |
| RD growing-het | 3.60 | 2/9 |
| **Melanoblast Eden** | **3.19** | **7/9** |

Across all 10 cows: coverage 0.26–0.77 (huge breed variation), enclave roughness median **10.3**, enclave CV median **1.30** — versus RD ≈ 5 / 0.4 and Eden ≈ 9 / 0.9. The boundary signature that separated the mechanisms on cow 01 is present on animals the fits never saw.
**Audit:** v5, builder **72.5 → 73.8** (`morphometric-validation` 80 → 85: this is now validation, not just fit). Montage: `bos_taurus_specimens_montage.png`.
**Next:** R8 fresh blinded Fable re-score of this tree · R3b full coat mask · R7 CLI.
