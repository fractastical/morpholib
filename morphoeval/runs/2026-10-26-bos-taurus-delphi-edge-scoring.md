# Delphi edge scoring — Bos taurus follow-up path (OP-11), 2026-10-26

**Why this exists:** problem-graph edges carry Delphi path-success probabilities where a panel has scored them, and NOT SCORED where none has (no invented numbers). The bos branch's only prospective path is OP-11, so it was scored by the repo's usual 3-model Delphi panel (Fable / GPT / Gemini, blind to each other; instructed not to read the bos audits/run records).

**Proposition scored:** "Within this repo's means, OP-11 succeeds: the independent re-score confirms the published verdict's robustness AND the migration-wave variant yields a decisive statistical separation (or decisive non-separation) between growing RD and colonisation on the existing 10 specimens."

Subgoals: B1 (25) blinded re-score of the Pass 8+ tree · B2 (30) migration-wave variant implemented + scored on the same pipeline · B3 (30) separation decisive vs seed/specimen noise at n=9+1 · B4 (15) superseding assessment published either way.

| Panellist | B1 | B2 | B3 | B4 | weighted | prior | for | against | net | BF | posterior | conf | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Claude Fable 5.1 | 60 | 72 | 38 | 70 | 58.5 | 20% | +11 | −13 | −2.0 | 0.631 | 13.62% | 0.60 | ACHIEVABLE_BUT_UNDERPOWERED |
| GPT 5.6 Sol | 72 | 82 | 38 | 86 | 66.9 | 42% | +7.5 | −8.5 | −1.0 | 0.794 | 36.52% | 0.78 | PLAUSIBLE_BUT_UNDERPOWERED |
| Gemini 4 Argon | 78 | 85 | 68 | 88 | 78.6 | 60% | +8.5 | −5.5 | +3.0 | 1.995 | 74.96% | 0.78 | LIKELY_ACHIEVABLE |
| **Pooled (log-odds)** | 70.0 | 79.7 | 48.0 | 81.3 | 68.0 | 39.30% | +9.0 | −9.0 | 0.0 | 1.000 | **39.30%** | 0.72 | balanced evidence; success = possible pathway (≥15%) |

Pooling follows the Rule 30 assessment: prior = geometric-mean odds of panellist priors; dB = arithmetic means; posterior = pooled prior odds × BF. All rows satisfy the builder's self-checks (weights = 100; weighted subgoals = stated totals; prior × BF = posterior; for + against = net).

**Normalisation note:** GPT returned against-evidence as positive magnitudes (+8.5 dB, positive item dBs); converted to the panel's signed convention (−8.5) before pooling. No values changed, only signs.

**Reading:** implementation-side subgoals score high (B2 79.7, B4 81.3 — harness and publication machinery exist); the crux is B3 (48.0): at n=9–10 with per-cow difference SD ≈1.4, decisiveness needs a near-unanimous sign or pre-registered equivalence bounds. The 5.5× posterior spread (13.6–75.0%) is recorded as disagreement data, as with Prize 1. Pooled 39.30% ≥ 15% → the OP-11 edge renders green ("possible pathway") on the problem graph and MorphoSwarm; it is a panel posterior, not a measurement.

**Graph wiring:** canvas `DELPHI.bos` holds this entry; edge `f5 → b6` ("OP-11 route: re-score + migration-wave → superseding assessment") carries `prob: 39.3`; OP-11 links `delphi: bos`. Edges with no panel (including the realised bos publication edge) render NOT SCORED.
