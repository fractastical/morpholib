# Project Delphi panel — Turing open problems (blinded independent assessor)

You are ONE independent assessor on a three-provider Bayesian Delphi panel. Two other assessors (other model providers) are scoring the same packet in parallel. Your scores will be pooled in log-odds; the spread between assessors is part of the result. Do not try to guess what others would say.

## Blinding — read first
- Read ONLY `/tmp/turing_panel/packet.json` for the problems. It contains 10 open problems in Turing-pattern biology, each with: id, anchor (historical lineage), title, gap, done (definition of done), tool_route, deciding_data, prior_attempts, hypothesis (the statement you assess), papers.
- You MAY inspect the repository `/Users/jdietz/Documents/GitHub/infinitemorphospace` to check whether the tools/data named in a `tool_route` / `deciding_data` actually exist (e.g. `models/1952-turing-morpho.py`, `models/1981-murray-bos-taurus-turing-wavelength.py`, `models/1984-wolfram-conus-textile-rule30.py`, `wave-vector-analysis/`, `morphoeval/scout_literature.py`, `morphoeval/problem_scout_100.py`, `predictions/`).
- Do NOT open any of these — they contain a prior single-assessor scoring of the same problems and would contaminate you: `docs/TURING_OPEN_PROBLEMS.md`, `docs/turing-open-problems.json`, `morphoeval/audits/2026-10-27-turing-open-problems-assessment.yaml`, `morphoeval/kg/2026-10-27-turing-open-problems-assessment/`, `morphoeval/problem_scouts/scout-101.json`, `morphoeval/build_turing_open_problems.py`, `docs/open-problems.json`, `docs/problem-graph.html`, `docs/morphoswarm.html`, `morphoeval/problem_scouts/LEADERBOARD.md`, `morphoeval/problem_scouts/leaderboard.json`. If you accidentally see a score, record it under `exposure_disclosure`.
- You may use web search for the cited papers.
- Do not modify, create or delete any file inside the repository. Your only output is the JSON file named below.

## Two separate axes — score BOTH for every problem

### Axis 1: Resolvability (0-100, deterministic rubric)
Prices whether OUR tools, data in hand and a taste-free definition-of-done can reach a verdict HERE — NOT whether the claim is true. Score each factor 0..weight with a one-line `why`:
- tool_fit (0-25): a named repo tool, run as-is, produces the deciding evidence
- data_in_hand (0-20): deciding data already in repo, or one named public fetch away
- falsifiable_done (0-15): an exact definition-of-done a scorer can check without taste
- compute_in_repo (0-15): runs on the repo stack in hours, no wet lab / no new infrastructure
- isolation (0-10): closable as new files only, zero-conflict with parallel agents
- time_to_verdict (0-10): verdict reachable in one loop / one sitting
- not_blocked (0-5): no recorded prior attempt already refuted this exact route
Do not report a total; the ranker sums factors.

### Axis 2: Viability (provisional Bayesian, deciban ledger)
Question per problem: "Will the stated `hypothesis` reach a DECISIVE verdict (either way) if the `tool_route` is followed in this repo?" Start from prior 50%. List evidence items, each `{"db": <signed integer or half-integer decibans>, "text": "..."}`: positive db for, negative db against. Keep |db| <= 5 per item and total |net| <= 12. Posterior is computed by the aggregator from prior + net dB (10*log10 odds). Also give `confidence` (0-1, how sure you are of your own ledger) and a short SCREAMING_SNAKE `verdict` label.

## Output — exactly one file
Write `/tmp/turing_panel/<YOUR_ID>.json` where YOUR_ID is the one given in your task prompt, with this schema:
{
  "assessor_id": "<YOUR_ID>",
  "assessor_label": "<model name and provider>",
  "date": "2026-10-27",
  "exposure_disclosure": "<what, if anything, you saw that could bias you; 'none' otherwise>",
  "problems": [
    {"id": "TOP-01",
     "factors": {"tool_fit": {"score": 0, "why": ""}, "data_in_hand": {...}, "falsifiable_done": {...}, "compute_in_repo": {...}, "isolation": {...}, "time_to_verdict": {...}, "not_blocked": {...}},
     "evidence_for": [{"db": 3, "text": "..."}],
     "evidence_against": [{"db": -2, "text": "..."}],
     "confidence": 0.6,
     "verdict": "LABEL",
     "note": "<one or two sentences: the crux, and anything in the packet you think is wrong or missing>"}
    ... all 10 problems, TOP-01 .. TOP-10 ...
  ]
}
Validate it parses as JSON and has all 10 ids and all 7 factors each before finishing. In your final reply, give only: the path written, and a 10-line table id | sum-of-factors | net dB | verdict.
