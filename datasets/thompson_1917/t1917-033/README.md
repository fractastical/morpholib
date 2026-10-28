# T1917-033 — Thompson (1917) Phyllotaxis / Golden Divergence Angle

**Claim:** Leaf/floret divergence approximates the golden (Fibonacci) angle (~137.5°), with rational approximations 1/2, 1/3, 2/5, 3/8… and Fibonacci numbers of parastichies.

**Source:** Thompson, D.W. (1917) *On Growth and Form*, Ch. XIV, pp. 635–651, Figs. 325–327.

## Dataset

| File | Description |
|------|-------------|
| `helianthus_annuus_capitulum_specimen.jpg` | CC BY-SA 2.0 Wikimedia sunflower capitulum (SHA-256 pinned in reference JSON) |
| `helianthus_parastichy_annotated.jpg` | CC BY-SA 3.0 annotated golden-ratio seedhead |
| `t1917_033_on_leaf_arrangement_or_phyllot_reference.json` | Equations, parameter regimes, specimen digests |
| `paper_citations.bib` | BibTeX for Thompson 1917 and modern validation papers |
| `scout_manifest.json` | Step 1 verification dossier (IN_REPO dataset) |
| `result.json` | Loop verdict on the 1917 claim |

## Reproduction

```bash
python3 thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py --self-test
python3 thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py
```

Outputs: `docs/media/t1917_033_on_leaf_arrangement_or_phyllot_comparison.png`, `docs/media/t1917_033_on_leaf_arrangement_or_phyllot_growth.gif`.
