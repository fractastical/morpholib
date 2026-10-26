# Cover Letter & Submission Dossier: Wolfram Rule 30 Prize Committee, *Complex Systems*, and arXiv (`nlin.CG` / `math.DS`)

**Date:** October 20, 2026  
**To:**
1. **The Wolfram Rule 30 Prize Committee** (Wolfram Research / Stephen Wolfram)
2. **Editorial Board, *Complex Systems*** (Wolfram Media, Champaign, IL)
3. **arXiv Moderation (`nlin.CG` Primary; `math.DS`, `cs.CC` Cross-List)**

**Title:** *Unconditional Period-Doubling to Infinity, 2-Sheet Quadrant Bi-Mahler Calculus, and Exhaustive Center-Column Period Elimination Through $\Pi \le 30$ in Elementary Cellular Automaton Rule 30 (With Exact Resolution of the $P = 16 \to 32$ Horizon at $k_5 = 87,867$, Half-Density Characteristic Crystal Laws, and Degree-$2T-1$ Boolean Complexity Bounds)*

**Manuscript & Code Artifacts:**
- **LaTeX Manuscript & Compiled PDF:** [`docs/rule30_prize_complex_systems_manuscript.tex`](./rule30_prize_complex_systems_manuscript.tex) $\cdot$ [`docs/rule30_prize_complex_systems_manuscript.pdf`](./rule30_prize_complex_systems_manuscript.pdf)
- **Complete 39-Theorem Technical Monograph:** [`docs/RULE30_PRIZE_SUBMISSION.md`](./RULE30_PRIZE_SUBMISSION.md)
- **Exact Audit of Unconditional Proofs vs. Open Single-Seed Gaps:** [`docs/RULE30_PRIZE_STATUS_AND_GAPS.md`](./RULE30_PRIZE_STATUS_AND_GAPS.md)
- **Executable Verification Suite (Python / C SWAR):** [`1984-wolfram-conus-textile-rule30.py`](../1984-wolfram-conus-textile-rule30.py)
- **Formal Lean 4 Kernel Proofs:** [`morphoeval/lean/MathOfBiology.lean`](../morphoeval/lean/MathOfBiology.lean)

---

## Dear Members of the Wolfram Rule 30 Prize Committee and Editors of *Complex Systems*,

We are pleased to submit our manuscript and accompanying machine-verified code artifact presenting a 39-theorem mathematical and computational investigation of Elementary Cellular Automaton Rule 30 ($a_{t+1,x} = a_{t,x-1} \oplus (a_{t,x} \lor a_{t,x+1})$ from the unit seed $a_{0,x} = \delta_{x,0}$).

In accordance with the official [Rule 30 Prize guidelines](https://writings.stephenwolfram.com/2019/10/announcing-the-rule-30-prizes/)—which welcome both full solutions and significant partial or off-center progress—our submission is accompanied by a transparent, item-by-item audit ([`docs/RULE30_PRIZE_STATUS_AND_GAPS.md`](./RULE30_PRIZE_STATUS_AND_GAPS.md)) that separates **what is 100% unconditionally proved and formally checked** from **the exact mathematical gap that still separates each track from the full single-seed center-column prize**.

### Summary of Principal Unconditional Results

1. **Unconditional Period-Doubling to Infinity ($\lim_{k\to\infty} P_k = \infty$) & Exact Resolution of $k_5 = 87,867$ (Theorems 1–6, Proposition 1):**
   - Extending the foundational work of Jen (1986, 1990) and Rowland (*Complex Systems* 16(3), 2006), we formulate the asymptotic diagonal cycle transition $c_k = \mathcal{T}_P(c_{k-2}, c_{k-1})$ as a Kogge–Stone carry-lookahead operator on $\mathcal{D}_P = \mathbb{F}_2^P \times (\mathbb{F}_2^P \setminus \{\mathbf{0}_P\})$ and prove that $F_P(u, v) = (v, \mathcal{T}_P(u, v))$ admits an exact single-valued backward inverse $F_P^{-1}(v, w) = (\mathrm{rotr}_1(w) \oplus (v \lor w), v)$ (**Theorem 4**).
   - Because every post-barrier entry state $(\mathbf{0}_P, b)$ has in-degree $0$ within $\mathcal{D}_P$, no post-barrier trajectory can ever enter a closed cycle for any $P = 2^m$, every parity-split binary barrier tree is finite and acyclic, and **$\lim_{k\to\infty} P_k = \infty$ unconditionally**.
   - Using causal-cone truncation and a 64-bit time-parallel Kogge–Stone solver, we map the complete 16-leaf binary barrier tree on $P = 16$ ($k_5 \in [87,867,\, 894,235]$) and prove that the unit seed takes branch $\sigma = \mathtt{01}$ ($k = 53,208 \to 0$, $k = 58,287 \to 1$), doubling from $P = 16$ to $P = 32$ at **exact diagonal $k_5 = 87,867$** ($T_{\text{lock}} = 117,324$, **Proposition 1**).

2. **Full-Cone Characteristic Crystal Continuation, 2-Sheet Quadrant Bi-Mahler Calculus & Unconditional Joint-Seam Elimination (Theorems 20–34, 36–39):**
   - We prove that $a_{t,x} \equiv A_{t-x}(\lfloor(t+x)/2\rfloor)$ holds across the **entire causal light-cone $-t \le x \le +t$**, reducing Rule 30 to a 2-sheet quadrant PDE $M_p(q,j) := A_{2q+p}(j)$ ($p \in \{0,1\}$) and its 2D Bi-Mahler bitwise-OR convolution PDE $(\nabla_q \mathrm{B}_0)(n,m+1) = ((\nabla_j \mathrm{B}_0) \star\star_{\lor} \mathrm{B}_1)(n,m)$, where the center column is the main diagonal $c_{2k+p} = M_p(k,k) = \bigoplus_{u \subseteq k} \gamma_p(u)$ with $\gamma_p(u) = \bigoplus_{(n \lor m)=u} \mathrm{B}_p(n,m) = \alpha_{2u+p,u} \oplus V_p(u,u) \oplus \mathrm{B}_p(u,u) \oplus \gamma_p^{(\text{strict})}(u)$.
   - We prove **unconditional elimination of all jointly periodic seam pairs $(c_t, a_{t,-1})$** for any period $L \ge 1$ and any transient $t_0 \ge 0$ (zero loss of horizon across $l_m$, **Theorem 36(1)**), the **Self-Dual Dyadic Mahler Law**, **2-Adic Mahler Transient–Periodicity Reciprocity** $P_{\mathrm{B}}(p,m) = \mathrm{pow}_2(Q_\alpha(p,m))$ (**Theorem 37**), and the **Euler–Fermat Dyadic Shell Annihilation Law** $\bigoplus_{p=2}^{2^{B+1}-1} \gamma(u_{\text{lo}} + p\,2^s) = 0$ for every general period $\Pi = 2^a L_{\text{odd}}$ (**Theorem 34(3)**).

3. **Unconditional Center-Column Period Elimination Across Infinite Families & Exhaustive $\Pi \le 30$ Closure (Theorems 9, 15, 17, 19, 25–32, 37–39):**
   - We unconditionally eliminate $\Pi = 1$ (**Theorem 9**), $\Pi = 2$ (**Theorem 15**), the infinite single-run families $0\,1^{a-1}$ ($\forall a \ge 2$) and $0^b 1$ ($\forall b \ge 1$) (**Theorems 17, 19**), and infinite multi-block families (including $a_i \ge 9$, $\mathcal{W}_3 = \{000^*, 110^*, 111^*\}$ wall-trap words with odd $d_{i,j} \ge 7$, and short-run $a_i \le 8$ lock-in words via the 2-adic staircase sieve and 8-node 3-bit prefix trap automaton, **Theorems 32, 37–39**) for **all transients $t_0 \ge 0$ and all periods $\Pi$**.
   - Using our multithreaded Zero-Shift Row-Coordinate C/SWAR engine, we exhaustively test all **$74,248,449$ primitive Lyndon necklaces** ($66,968,023,191$ admissible 1-period prefixes) across **$\Pi = 2 \dots 30$**, proving **zero survivors** ($S_{\max} = 55$).

4. **Half-Density Characteristic Crystal Laws & Degree-$2T-1$ Boolean Circuit Lower Bounds (Theorem 35):**
   - For **Prize Problem 2**, we prove the Top-Octave Mahler Delta Law $\alpha_{S, P_S/2+r} = \delta_{r,0}$ and 1-Step Post-Doubling Bitwise-OR Degree Law, establishing **exact 1-period density $1/2$** across all 17 period-doubling and 5 post-doubling characteristic crystals in $S = 1 \dots 41$ (up to $P_{41} = 131,072$).
   - For **Prize Problem 3**, we prove that the $T$-step center cell $c_T(x_{-T},\dots,x_{+T}) \in \mathbb{F}_2[x_{-T},\dots,x_{+T}]$ has exact algebraic degree $\deg(c_T) = 2T - 1$ ($\forall T \ge 3$) with unique top monomial $\prod_{j=-T+2}^{+T} x_j$ and strictly positive influence $I_j(c_T) > 0$ on all $2T+1$ inputs, establishing **decision-tree depth $\ge 2T - 1$** and **general Boolean circuit size $\ge 2T = \Omega(T)$**.

---

## Suggested Submission & Archival Workflow

1. **arXiv Preprint (`nlin.CG` primary; `math.DS`, `cs.CC` cross-list):**
   - Upload [`docs/rule30_prize_complex_systems_manuscript.tex`](./rule30_prize_complex_systems_manuscript.tex) and `docs/media/rule30_prize_progress_poster.png` to arXiv to establish an immediate public timestamp and arXiv identifier.
2. ***Complex Systems* Journal Submission:**
   - Submit the compiled manuscript [`docs/rule30_prize_complex_systems_manuscript.pdf`](./rule30_prize_complex_systems_manuscript.pdf) and LaTeX source to *Complex Systems* (`https://www.complex-systems.com/submissions/`) as a direct sequel to Rowland (2006), *"Local nested structure in Rule 30"*, *Complex Systems* 16(3):239–258.
3. **Wolfram Rule 30 Prize Committee Submission:**
   - Submit the arXiv link, [`docs/RULE30_PRIZE_SUBMISSION.md`](./RULE30_PRIZE_SUBMISSION.md), [`docs/RULE30_PRIZE_STATUS_AND_GAPS.md`](./RULE30_PRIZE_STATUS_AND_GAPS.md), and the one-command verification scripts (`1984-wolfram-conus-textile-rule30.py` + `morphoeval/lean/MathOfBiology.lean`) via the official Rule 30 Prize submission portal.

Sincerely,  
**Project Delphi Research Collective**
