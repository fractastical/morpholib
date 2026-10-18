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
