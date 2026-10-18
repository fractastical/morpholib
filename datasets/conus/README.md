# *Conus textile* Shell Pigmentation & Wolfram Rule 30 Fixtures

Reference dataset and SHA-256-pinned fixtures for [`1984-wolfram-conus-textile-rule30.py`](../../1984-wolfram-conus-textile-rule30.py), bridging **Stephen Wolfram's (1984/1986) Rule 30 cellular automaton** (and the [**Wolfram Rule 30 Prizes**](https://rule30prize.org/)), **Hans Meinhardt & Martin Klingler's (1987/1995) activator–substrate–inhibitor reaction–diffusion PDE**, and **David Raup's (1966) 3D helicospiral shell geometry** against real *Conus textile* Linnaeus, 1758 specimens.

![Conus textile pigmentation: Rule 30 vs Meinhardt PDE mapped onto a 3D Raup shell and compared against real specimens](../../docs/media/conus_textile_rule30.gif)

## Vendored Reference Fixture

| File | SHA-256 | Size | Description |
| :--- | :--- | ---: | :--- |
| `conus_textile_reference.json` | `dfcb500d87a393907a5e862cb5167f1d79591f24da0f22b446ea10fc98a2afa5` | 40,985 B | Extracted 256×140 specimen silhouette (`H/W = 1.8330`), 148×92 segmented dorsal white-tent lattice (`1,814` interior white chord runs, `CV = 0.6403`), Ling close-up chord statistics (`18,701` runs), and the first 128 center-column bits of Rule 30 (`OEIS A051023`). |
| `paper_citations.bib` | — | 1,648 B | BibTeX entries with DOIs for Wolfram (1984, 1986), Jen (1990), Ermentrout, Campbell & Oster (1986), Meinhardt & Klingler (1987), Meinhardt (1995), and Raup (1966). |

## Specimen Provenance & Licences (`--download` to `.conus_cache/`)

Running `python3 1984-wolfram-conus-textile-rule30.py --download` fetches the two original Wikimedia Commons photographs into the git-ignored `.conus_cache/` directory and verifies their SHA-256 digests:

1. **`stjohn_osu_24399204286` (`conus_textile_stjohn_ccby2.jpg`):**
   - **Source:** Wikimedia Commons [`File:Conus textile (textile cone snail) 1 (24399204286).jpg`](https://commons.wikimedia.org/wiki/File:Conus_textile_(textile_cone_snail)_1_(24399204286).jpg) (`500×897` thumbnail)
   - **Author:** James St. John (Ohio State University at Newark)
   - **Licence:** [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)
   - **Bytes / SHA-256:** `101,322` bytes · `53409ef2518c010613002d68f48f27d236ed22f924d13391d082fab804fcf3c4`
2. **`ling_commons_010` (`conus_textile_010_pd.jpg`):**
   - **Source:** Wikimedia Commons [`File:Conus textile 010.jpg`](https://commons.wikimedia.org/wiki/File:Conus_textile_010.jpg) (`1366×1024`)
   - **Author:** Richard Ling
   - **Licence:** Public domain
   - **Bytes / SHA-256:** `614,321` bytes · `fdc05730ef1b2e1aaf723e4cd8e15f07f57afc7ba13f2b21227cfec926d98760`

## Connection to the Wolfram Rule 30 Prizes (`rule30prize.org`)

Rule 30 evolves a 1D binary ring $a_{t,i} \in \{0, 1\}$ according to:

$$a_{t+1,\,i} \;=\; a_{t,\,i-1} \,\oplus\, \bigl(a_{t,\,i} \lor a_{t,\,i+1}\bigr) \;\equiv\; a_{t,\,i-1} + a_{t,\,i} + a_{t,\,i+1} + a_{t,\,i}\,a_{t,\,i+1} \pmod 2.$$

`1984-wolfram-conus-textile-rule30.py` implements and verifies three structural properties directly relevant to the [Rule 30 Prizes](https://rule30prize.org/):

1. **Left-Bijective Inverse & Erica Jen's (1990) Pair-Aperiodicity Theorem:** Because $\partial a_{t+1,i} / \partial a_{t,i-1} = 1$ over $\mathbb{F}_2$, any two adjacent columns $(a_{t,i}, a_{t,i+1})_{t \ge 0}$ uniquely determine the column to their left via $a_{t,i-1} = a_{t+1,i} \oplus (a_{t,i} \lor a_{t,i+1})$. The script verifies zero bit errors under backward spatial reconstruction across the entire light-cone and tracks the meandering phase boundary ($v_{\text{front}} \approx -0.252$ cells/step) separating the period-$2^k$ left diagonals from the chaotic right half that drives the center column $c_t = a_{t,0}$.
2. **Center-Column Equidistribution & Aperiodicity Diagnostics (Prize Problems 1 & 2):** Verifies $c_0 \dots c_{127}$ against `OEIS A051023`, checks that no period $(p, t_0)$ exists in the tested horizon, and measures the convergence of cumulative ones density $\bar{c}_T \to 1/2$ and $k$-block Shannon entropy $H_k / k \to 1$ bit/symbol.
3. **White-Triangle ("Tent") Extinction Law vs. *Conus textile* Specimens:** Under Rule 30, a horizontal run of $w$ consecutive zeros (`00...0`) erodes inward from its boundaries (`100 -> 1`, `001 -> 1`, `000 -> 0`), forming a downward-pointing white triangle whose base width $W$ follows a geometric distribution $P(W = w) = 2^{-w}$ ($CV = 1/\sqrt{2} \approx 0.7071$). The script compares the normalized white-chord distribution $w / \langle w \rangle$ across **Rule 30**, **Rule 90** (Sierpiński additive control), **Meinhardt–Klingler (1987) activator–substrate–inhibitor PDE**, and the segmented **real *Conus textile* specimens**.
4. **1D Space-Time Kymograph & Defect-Front Tooling (`conus_rule30_kymograph_defect_fronts.png`):**
   - **Left-Cone Diagonal Recursion & Jen (1990) Period-Doubling Dislocations:** Verifies $d_k(t+1) = d_{k-2}(t) \oplus (d_{k-1}(t) \lor d_k(t))$ with `0` bit errors across $k = 2 \dots 414$, extracts the exact period $P(k) = 2^{m(k)}$ and lock-in transient $T_{\text{lock}}(k) \approx 1.2441\,k + 7.89$ ($\implies v_{\text{lock}} = -0.1962$ cells/step, inward diagonal speed $0.8038 < 1$), and confirms the first five period-doubling dislocations at $k = 0$ ($P=1$), $k = 3$ ($P=2$), $k = 8$ ($P=4$), $k = 29$ ($P=8$, followed by a 371-diagonal plateau $k = 29 \dots 399$), and $k = 400$ ($P=16$, $T_{\text{lock}} = 498$), each gated by an all-zero barrier $d_{k-1}(t) \equiv 0$ with odd upstream parity $\sum d_{k-2} \equiv 1 \pmod 2$.
   - **Asymmetric Boolean Jacobian Damage-Spreading Light-Cone:** Proves and measures across a 25-seed ensemble that the right damage front moves at exact speed $v_R = +1.0000 \pm 0.0000$ cell/step ($\partial a_{t+1,i}/\partial a_{t,i-1} = 1$), while the left damage front is blocked whenever $a_{t, L(t)-1} = 1$ ($\partial a_{t+1,i}/\partial a_{t,i+1} = 1 \oplus a_{t,i}$), drifting at $v_L = -0.2604 \pm 0.1024$ cells/step.
   - **Cross-Model Morphogenetic Defect Fronts:** Compares pinned mantle-lesion wakes in Rule 30 vs. Meinhardt–Klingler (1987) PDE ($v_{\text{PDE}} = \pm 0.2285$ cells/step), autonomous cell-sorting frozen-defect kymographs (Zhang, Goldstein & Levin 2025: Bubble pinned barrier `cross_swaps = 0, inv = 205` vs. Selection non-local defect tunneling `cross_swaps = 161, inv = 0`), and bioelectric gap-junction domain-wall kymographs (Cervera, Levin & Mafé 2021: $x^*(t) = 0.500 \to 0.590$ crossing the PlanformDB bipolar threshold $x^*_b = 0.583$).

![1D Space-Time Kymograph & Defect-Front Tooling](../../docs/media/conus_rule30_kymograph_defect_fronts.png)

> **Mathematical Review Brief:** See [`docs/RULE30_DEFECT_FRONTS_MATH_REVIEW.md`](../../docs/RULE30_DEFECT_FRONTS_MATH_REVIEW.md) for the standalone mathematical review document detailing the **Kogge–Stone carry-lookahead factorization** of the diagonal cycle-transfer operator $\mathcal{T}_P$ (Theorem 1), the **time-shift equivariance proof** of why $k_1 = 3, k_2 = 8, k_3 = 29, k_4 = 400$ and the first $P=16$ barrier at $k - 1 = 53\,207$ are transient-independent (Theorem 2), and the **exhaustive 15-split / 16-leaf acyclicity theorem** proving that every branch on the $P=16$ plateau doubles to $P=32$ in $k_5 \in [87\,867,\, 894\,235]$ (Theorem 3).

## MorphoEval 7-Axis Audit & Published Nanopublications (`2026-10-18-conus-audit`)

Evaluated at commit `fb234bc` under MorphoEval Rubric `0.1.0` ([`morphoeval/audits/2026-10-18-conus-audit.yaml`](../../morphoeval/audits/2026-10-18-conus-audit.yaml) · [`morphoeval/kg/2026-10-18-conus-audit/REPORT.md`](../../morphoeval/kg/2026-10-18-conus-audit/REPORT.md)):

- **Overall Score:** **`87.3 / 100`** (`faithful`) — **Model Implementation (`30`):** `87.7` · **Biological Validation (`40`):** `85.5` · **Reproducibility (`30`):** `89.3`
- **Dimension Breakdown:**
  - `model-fidelity` (weight 20): **`89`** — [`https://w3id.org/np/RABpB9zkRHfFYrnvVIkm8CnbxALvhzWqXFV5_Z_g7r0Pg`](https://w3id.org/np/RABpB9zkRHfFYrnvVIkm8CnbxALvhzWqXFV5_Z_g7r0Pg)
  - `specimen-match` (weight 25): **`84`** — [`https://w3id.org/np/RA8NsAfvATeT-dimbe8UF-eyoz5rhuAxKiumbicZgdERE`](https://w3id.org/np/RA8NsAfvATeT-dimbe8UF-eyoz5rhuAxKiumbicZgdERE)
  - `parameter-expressivity` (weight 10): **`85`** — [`https://w3id.org/np/RAw40YodQ38R_Y955Wld9YF6ppkRjD9a8_IMQgUmxUi1c`](https://w3id.org/np/RAw40YodQ38R_Y955Wld9YF6ppkRjD9a8_IMQgUmxUi1c)
  - `morphometric-validation` (weight 15): **`88`** — [`https://w3id.org/np/RAZdTwwTpNyKpK-XRw2s8T-4nAJVRA_8_Y9a0Z3ZL8nr4`](https://w3id.org/np/RAZdTwwTpNyKpK-XRw2s8T-4nAJVRA_8_Y9a0Z3ZL8nr4)
  - `source-traceability` (weight 10): **`90`** — [`https://w3id.org/np/RAE83Ep2iotK34zGkDFZTgGwcHREUSFuB4Ylv2d_M9i4w`](https://w3id.org/np/RAE83Ep2iotK34zGkDFZTgGwcHREUSFuB4Ylv2d_M9i4w)
  - `code-traceability` (weight 8): **`89`** — [`https://w3id.org/np/RADDG5GaxvPz-TaTWZbmRfJUaZdgrqdW1qhKxTyg_Jg1Q`](https://w3id.org/np/RADDG5GaxvPz-TaTWZbmRfJUaZdgrqdW1qhKxTyg_Jg1Q)
  - `repro-package` (weight 12): **`89`** — [`https://w3id.org/np/RAS5O9oGRxbWIoFGJLcUH_43zARSJu3TrtYKCkOk8aVyc`](https://w3id.org/np/RAS5O9oGRxbWIoFGJLcUH_43zARSJu3TrtYKCkOk8aVyc)
  - `overall` composite: **`87.3`** — [`https://w3id.org/np/RACS7k6nr8UJVeKgaT0a2-7NLPISBz3SRaH6qPEcVSVGg`](https://w3id.org/np/RACS7k6nr8UJVeKgaT0a2-7NLPISBz3SRaH6qPEcVSVGg)
  - `claim` target: [`https://w3id.org/np/RA2G31b0Bg50syNWFhJ5Aj8U2-KZQ5QizNhMjRl1uSb_g`](https://w3id.org/np/RA2G31b0Bg50syNWFhJ5Aj8U2-KZQ5QizNhMjRl1uSb_g)


