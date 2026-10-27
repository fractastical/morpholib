# Drosophila affinis: does Bilayer Cube-Root Wrinkling Wavelength Law actually generate this form?
🔎 UNVALIDATED — promoted for scrutiny, not as a result · care score 85.0/100 · finding `miller-drosophila-affinis-elastic-buckling`

**Why you, Thomas Lecuit:** you work on exactly this (Centre National de la Recherche Scientifique, Institut de Biologie du Développement Marseille). Your relevant work: Planar polarized actomyosin contractile flows control epithelial junction remodelling. If this is the wrong expertise, reply `wrong_expert` — that is a useful answer.

## The one question (15 minutes to answer, 2 minutes to triage)
Is 'Bilayer Cube-Root Wrinkling Wavelength Law' the right generative law for this structure in Drosophila affinis — and if not, what observation kills it?

**Triage (2 min):** reply one word — `care` / `dont_care` / `kill_it` / `wrong_expert`. Every answer is recorded and credited; `kill_it` closes this finding and saves everyone time.

## Why a scientist in this field might care
- A named, falsifiable law (`Bilayer Cube-Root Wrinkling Wavelength Law`) is pinned to a real structure in *Drosophila affinis* at 0.0005277 m — it makes a quantitative claim that can simply be wrong.
- It ships with a runnable reproduction and a side-by-side specimen comparison (below), so you judge evidence, not prose.
- An active expert community exists for this system — 50 eligible, conflict-free reviewers already identified.

![Simulation vs specimen](docs/media/drosophila_affinis_elastic_buckling_comparison.png)

## Three numbers (the whole evidence case)
| Metric | Value | Source |
|---|---|---|
| AI-audit overall (MorphoEval, uncalibrated) | 96.5 / 100 | `morphoeval/audits/2026-10-22-drosophila-affinis-elastic-buckling-audit-v2.yaml` |
| Simulation-vs-specimen image fitness | None | run telemetry |
| Fit to empirical series (Pearson r / RMSE) | 0.99982 / 0.06309 | vendored reference data |

## What to do — pick your cost
1. **2-minute triage** — one word above.
2. **15-minute verdict** — answer the one question: `validated` / `partially_validated` / `refuted` / `unable_to_assess`, plus one sentence. Recorded with your name (or anonymously) in the validation loop; time-to-verdict is published with the finding.
3. **One-sitting replication** — run it yourself:
   ```bash
   python3 models/1952-miller-drosophila-affinis-elastic-buckling.py --self-test && python3 models/1952-miller-drosophila-affinis-elastic-buckling.py
   ```
   If the figure does not reproduce, that is a `refuted` verdict with evidence.

## What would kill it
The falsifier above. A single expert `refuted` + one confirming verdict closes this finding as refuted — and that closure, with its elapsed time, is part of the public record.

## Caveats, upfront
- This finding was produced and AI-audited by automated loops; **no human expert has validated it** (unless the banner above says otherwise). Scores are uncalibrated rubric ratings, not probabilities.
- Claim boundary: the code implements the cited law; whether that law is the right biology is exactly the open question.
- You are never auto-emailed again by this system; a human sent this, and only recorded replies count.

## Credit & record
Replies feed `morphoeval/validations/miller-drosophila-affinis-elastic-buckling-human-validation.json` (verdict + timing) and this promotion's funnel record. Reviewers are named on the finding unless they ask otherwise.
Promotion status so far: track=`promote_for_scrutiny`, missing for result-track: ['human validation consensus']
