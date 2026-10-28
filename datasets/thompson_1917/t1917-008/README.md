# T1917-008 — The rate of growth (relative allometry)

**Claim (Thompson 1917, Ch. III, Figs. 18–22):** Parts of an organism grow at different but definite relative rates, so organ proportions change systematically with body size (brain, heart, body; plaice weight–length).

## Files

| File | Role |
|------|------|
| `thompson_fig18_brain_body.csv` | Mammal brain vs body weights (Fig. 18 digitisation) |
| `thompson_fig19_heart_body.csv` | Mammal heart vs body weights (Fig. 19) |
| `thompson_fig22_plaice_length_weight.csv` | Plaice length vs weight (Fig. 22) |
| `pheidole_caste_head_body.csv` | Modern ant caste analogue (labelled) |
| `pheidole_megacephala_*.jpg` | CC-licensed specimen photographs |
| `t1917_008_the_rate_of_growth_reference.json` | Equations, parameters, SHA-256 locks |
| `result.json` | Loop verdict (written by claim script) |

## Model (Huxley 1932 formalisation of Thompson’s differential rates)

\[
\frac{dy}{dt} = \alpha y,\quad \frac{dx}{dt} = \beta x \Rightarrow \log y = \log b + k \log x,\quad k = \alpha/\beta
\]

Implementation: `thompson/claims/t1917-008_the-rate-of-growth.py`

Lean 4: `morphoeval/lean/loops/Thompson1917_T1917_008.lean`
