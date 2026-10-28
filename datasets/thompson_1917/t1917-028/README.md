# T1917-028 — Nautilus logarithmic spiral & septal geometry

Thompson (1917) claim **T1917-028** (Ch. XI, Figs. 304-307): in *Nautilus*, the shell spiral and septa satisfy the geometric relations Thompson diagrams (septa meeting the wall; spiral as evolute/involute).

## Files

| File | Role |
|------|------|
| `scout_manifest.json` | Step 1 verification dossier (Commons specimen + literature) |
| `t1917_028_the_logarithmic_spiral_reference.json` | Equations, parameters, SHA-256 pins |
| `nautilus_pompilius_section_specimen.jpg` | CC BY-SA 4.0 section photograph |
| `paper_citations.bib` | BibTeX |
| `result.json` | Loop verdict (written by claim script) |

## Reproduce

```bash
python3 thompson/claims/t1917-028_the-logarithmic-spiral.py --self-test
python3 thompson/claims/t1917-028_the-logarithmic-spiral.py
```
