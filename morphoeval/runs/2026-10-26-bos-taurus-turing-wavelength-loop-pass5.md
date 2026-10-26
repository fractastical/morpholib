# Loop Pass 5: Melanoblast Positive Control + Fable-Corrected Scoring

**Remediates:** Fable second-assessor findings (fit tautology, seed noise, no positive control) and R5.
**Changes:** (1) `simulate_melanoblast()` — Eden-type colonisation from a dorsal founder line + sparse scattered founders, stopped at calibrated coverage; white = never colonised. (2) `morphometric_distance_v2()` — excludes coverage and largest-patch share (set by calibration, per Fable); scores only unfitted statistics. (3) `multi_seed_comparison()` — RD-static vs RD-growing-het vs Eden, 5 seeds each at cow-matched coverage. (4) Self-test now covers Pass-2/3/5 code; Schnakenberg (1979) cited and MITF authors corrected (Edea et al., *Genes & Genomics* 39(3):285–293).

| Model (5 seeds, coverage = cow 0.77) | distance v2 | enclaves (cow 12) | enclave roughness (cow 11.9) | enclave CV (cow 0.99) | enclave median Ø (cow 0.040) |
|---|---:|---:|---:|---:|---:|
| RD static γ=1 | 4.95 ± 0.04 | 146 | 4.5 | 0.11 | 0.039 |
| RD growing-het γ=0.25 | 2.75 ± 0.45 | 14.8 | 5.3 | 0.38 | 0.122 |
| **Melanoblast Eden (founders 0.002)** | **2.03 ± 0.70** | **9.8** | **9.0** | **0.93** | **0.063** |

Seed ranges of the top two overlap (Eden 0.73–2.83, RD 2.25–3.32); founder density was selected on 3 seeds against enclave count and is disclosed as selection on a scored statistic. **Reading:** growing RD can fit patch *areas*; patch *boundaries* (roughness, size diversity) are melanoblast-like. The titular Turing model is now the control that loses on the discriminating statistics — which is the scientifically useful outcome of the loop.
**Audit:** v4, builder **68.6 → 72.5** (Fable's independent 54.4 stands as the second-assessor record). **Animation:** 3 panels — cow mask | RD coat | Eden colonisation growing.
**Next:** R2 ≥10 specimens · R3 coat mask · R8 a fresh independent re-score of the Pass-5 tree.
