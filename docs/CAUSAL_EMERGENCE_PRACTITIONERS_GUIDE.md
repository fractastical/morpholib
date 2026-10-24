# Practitioner's Guide to Causal Emergence 1.0 (PNAS 2013) & 2.0 (*Patterns* 2026): How It Works & Where It Breaks

**Primary Author:** Joel Dietz ([Project Delphi](https://github.com/fractastical/project-delphi-ontology) · [Infinite Morphospace](https://github.com/fractastical/morpholib))  
**Executable Reference Script:** [`sorting-morphogenesis/quantify_emergence_ce2.py`](../sorting-morphogenesis/quantify_emergence_ce2.py)  
**Locked Metrics & Benchmarks (`10` Papers, `2013–2026`):** [`datasets/causal_emergence/causal_emergence_reference.json`](../datasets/causal_emergence/causal_emergence_reference.json) & [`sorting-morphogenesis/figures/quantifying_emergence_ce2_metrics.json`](../sorting-morphogenesis/figures/quantifying_emergence_ce2_metrics.json)  
**Machine-Checked Fixed-Point Algebraic Proofs (`0 sorry`):** [`morphoeval/lean/loops/Loop_causal_emergence.lean`](../morphoeval/lean/loops/Loop_causal_emergence.lean) & [`morphoeval/lean/loops/Loop_zhang_sorting_morphogenesis.lean`](../morphoeval/lean/loops/Loop_zhang_sorting_morphogenesis.lean) *(Note: these Lean 4 modules verify the exact integer/fixed-point ring identities of the decompositions and discrete GRN transitions; continuous Shannon entropies $\log_2 p$ are computed in Python/NumPy).*

![Practitioner's 5-Step Recipe for Applying Causal Emergence 1.0 & 2.0 in Practice](./media/substack_ce2_practitioner_workflow.png)

---

## 1. Glossary & Core Concepts (Read This First)

Before writing a single equation, here is a plain-language glossary of every symbol and concept used in Erik Hoel's Causal Emergence literature (`2013–2026`):

| Term / Symbol | Plain-English Meaning | Mathematical Definition / Range |
| :--- | :--- | :--- |
| **Microstate Space ($S_{\text{micro}}, N$)** | The finest-grained list of states of a discrete dynamical system (e.g., all $N = n!$ permutations of an array of $n$ sorting cells, or all $N = 2^g$ binary states of a $g$-gene regulatory network). | Discrete index set $\{1, \dots, N\}$ with bit-capacity $\log_2 N$. |
| **Transition Probability Matrix ($\mathrm{TPM}, T$)** | An $N \times N$ table where row $i$, column $j$ gives the probability of landing in state $j$ at step $t+1$ if you force (intervene on) the system to start in state $i$ at step $t$. | $T_{ij} = P(S_{t+1} = j \mid \mathrm{do}(S_t = i))$, with $T_{ij} \ge 0$ and $\sum_{j=1}^N T_{ij} = 1$. |
| **Coarse-Graining Map ($g: S_{\text{micro}} \to S_{\text{macro}}$)** | A "ruler" or grouping rule that lumps $N$ microstates into $M < N$ macro-buckets $U_1, \dots, U_M$ (where $U_a = \{i : g(i) = a\}$). *(We use $g$ for a single projection map and $\Pi = (\pi^{(0)}, \dots, \pi^{(K)})$ for an ordered hierarchy of partitions).* | Surjective map onto $M$ buckets ($2 \le M \le N$). |
| **Dynamic Markov Lumpability** | A partition $g$ is *dynamically lumpable* (Kemeny & Snell 1960) if every microstate $i$ inside bucket $U_a$ has the *exact same* total transition probability $\sum_{j \in U_b} T_{ij}$ to every target bucket $U_b$. | If lumpable, the macro-chain reproduces the projected micro-dynamics with zero path-history dependence. |
| **Garden-of-Eden State** | A state $j$ with column sum $\sum_{i=1}^N T_{ij} = 0$ (zero incoming transitions). It can never be reached naturally by the dynamics, only by an external `do()` intervention. | Critical pitfall when applying uniform `do()` interventions to dissipative biological networks. |
| **Determinism ($\mathrm{det}(T)$)** | How reliably the current state predicts the next state ($1$ minus average row entropy normalized by $\log_2 N$). Measures **causal sufficiency** in entropic form. | $\mathrm{det}(T) = 1 - \frac{\langle H(T_{i,\cdot})\rangle}{\log_2 N} \in [0, 1]$. |
| **Degeneracy ($\mathrm{deg}(T)$)** | How much different starting states collapse into the same outputs ($1$ minus the entropy of the average row $\bar{u} = \frac{1}{N}\sum_i T_{i,\cdot}$ normalized by $\log_2 N$). Measures **$1 - \text{causal necessity}$** in entropic form. | $\mathrm{deg}(T) = 1 - \frac{H(\bar{u})}{\log_2 N} \in [0, 1]$. |
| **CE 2.0 Entropic Effectiveness ($\mathrm{CP}_{\text{eff}}(T)$)** | The intensive **Causal Primitive** used in Hoel (2013, 2026): Determinism minus Degeneracy. **By the concavity of Shannon entropy ($H(\bar{u}) \ge \langle H(T_{i,\cdot})\rangle$), $\mathrm{CP}_{\text{eff}}(T)$ is strictly non-negative ($\in [0, 1]$) for any valid TPM.** | $\mathrm{CP}_{\text{eff}}(T) = \mathrm{det}(T) - \mathrm{deg}(T) = \frac{H(\bar{u}) - \langle H(T_{i,\cdot})\rangle}{\log_2 N} \in [0, 1]$. |
| **CE 2.0 Probabilistic Sufficiency-Necessity ($\mathrm{CP}_{\text{sn}}(T)$)** | An alternative intensive Causal Primitive studied in Comolatti & Hoel (2025) and Hoel (2026) based directly on Eells (1991) probabilistic causation $\mathrm{suff} + \mathrm{nec} - 1$ rather than Shannon entropies. | $\mathrm{suff} = P(e \mid c)$, $\mathrm{nec} = 1 - P(e \mid \neg c)$; $\mathrm{CP}_{\text{sn}} = \mathrm{suff} + \mathrm{nec} - 1 \in [0, 1]$ *(differs numerically from $\mathrm{CP}_{\text{eff}}$!)*. |
| **CE 2.0 Gain ($\Delta\mathrm{CP}$)** | The difference in intensive causal power between a macro-scale ($M$ states) and micro-scale ($N$ states). Unlike $\mathrm{CP}(T) \in [0, 1]$, the *difference* $\Delta\mathrm{CP}$ can be positive or negative. | $\Delta\mathrm{CP} = \mathrm{CP}(T_M) - \mathrm{CP}(T_m) \in [-1, 1]$. |
| **CE 1.0 Effective Information ($\mathrm{EI}(T)$)** | Hoel, Albantakis & Tononi's (2013) extensive measure of causal power in bits: the mutual information $I(I_D; E_D)$ under a uniform intervention distribution $I_D = \mathcal{U}(1..N)$. | $\mathrm{EI}(T) = \mathrm{CP}_{\text{eff}}(T)\,\log_2 N = H(\bar{u}) - \langle H(T_{i,\cdot})\rangle \in [0, \log_2 N]\text{ bits}$. |
| **CE 2.0 Emergent Complexity ($\mathrm{EC}(\Pi)$)** | Hoel's (2026) measure (in bits or nats) of how widely causal power is distributed across multiple scales along a partition hierarchy $\Pi = (\pi^{(0)}, \dots, \pi^{(K)})$. | Shannon entropy $H(\mathbf{w}) = -\sum_k w_k \log_2 w_k$ of the normalized positive step-wise causal gains $w_k = \Delta\mathrm{CP}_k / \sum_j \Delta\mathrm{CP}_j$. |

---

## 2. Hand-Worked Toy Example First: Hoel's (2026) 8-State Two-Block System (`fig4_block_to_diagonal_50steps`, Step 0)

Before looking at code or large state spaces, work through this exact 8-state toy system by hand (`datasets/causal_emergence/causal_emergence_reference.json` $\to$ `hoel_2026_ce2_pymergence_benchmarks.fig4_block_to_diagonal_50steps` at `step0`, and `Loop_causal_emergence.lean` Law 6 `hoel_2026_ce2_apportioning_and_block_divergence`). It shows in three lines why CE 1.0 ($\Delta\mathrm{EI}$) and CE 2.0 ($\Delta\mathrm{CP}$) can give completely different answers.

### The Micro-System ($N = 8$ States, $\log_2 N = 3\text{ bits}$)
Divide $N = 8$ microstates into two 4-state blocks: $B_0 = \{0, 1, 2, 3\}$ and $B_1 = \{4, 5, 6, 7\}$.
* Suppose any state in $B_0$ transitions with equal probability ($1/4$ each) to the 4 states in $B_1$:
  $$T_{i,\cdot} = [0, 0, 0, 0, \tfrac{1}{4}, \tfrac{1}{4}, \tfrac{1}{4}, \tfrac{1}{4}] \quad \text{for } i \in B_0.$$
* And any state in $B_1$ transitions with equal probability ($1/4$ each) to the 4 states in $B_0$:
  $$T_{i,\cdot} = [\tfrac{1}{4}, \tfrac{1}{4}, \tfrac{1}{4}, \tfrac{1}{4}, 0, 0, 0, 0] \quad \text{for } i \in B_1.$$

Now compute the micro-scale metrics by hand:
1. **Row Entropy:** Every row has four $\tfrac{1}{4}$ entries, so $H(T_{i,\cdot}) = -4 \times \bigl(\tfrac{1}{4}\log_2 \tfrac{1}{4}\bigr) = \mathbf{2.0000\text{ bits}}$.
2. **Micro Determinism:** $\mathrm{det}_m = 1 - \frac{2.0}{\log_2 8} = 1 - \frac{2}{3} = \mathbf{\tfrac{1}{3} \approx 0.3333}$.
3. **Average Row $\bar{u}$ & Micro Degeneracy:** Averaging all 8 rows gives $\bar{u} = [\tfrac{1}{8}, \dots, \tfrac{1}{8}]$, so $H(\bar{u}) = \log_2 8 = 3.0\text{ bits}$ and $\mathrm{deg}_m = 1 - \frac{3.0}{3.0} = \mathbf{0.0000}$.
4. **Micro CE 2.0 ($\mathrm{CP}_{\text{eff},m}$) and CE 1.0 ($\mathrm{EI}_m$):**
   $$\mathrm{CP}_{\text{eff},m} = \tfrac{1}{3} - 0 = \mathbf{0.3333}, \qquad \mathrm{EI}_m = \mathrm{CP}_{\text{eff},m} \times \log_2 8 = \tfrac{1}{3} \times 3 = \mathbf{1.0000\text{ bit}}.$$

### The Macro-System ($M = 2$ Blocks, $\log_2 M = 1\text{ bit}$)
Now coarse-grain onto the two blocks $B_0$ and $B_1$:
$$T_M = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}.$$
At the macro-scale, the system is a **$100\%$ deterministic toggle** ($B_0 \leftrightarrow B_1$)!
1. **Macro Row Entropy & Determinism:** $H(T_{M,a,\cdot}) = 0\text{ bits} \implies \mathrm{det}_M = 1 - 0 = \mathbf{1.0000}$.
2. **Macro Degeneracy:** $\bar{u}_M = [\tfrac{1}{2}, \tfrac{1}{2}] \implies H(\bar{u}_M) = 1.0\text{ bit} \implies \mathrm{deg}_M = 1 - 1 = \mathbf{0.0000}$.
3. **Macro CE 2.0 ($\mathrm{CP}_{\text{eff},M}$) and CE 1.0 ($\mathrm{EI}_M$):**
   $$\mathrm{CP}_{\text{eff},M} = \mathbf{1.0000}, \qquad \mathrm{EI}_M = 1.0000 \times \log_2 2 = \mathbf{1.0000\text{ bit}}.$$

### Compare the Verdict:
* **CE 1.0 ($\Delta\mathrm{EI}$):** $\mathrm{EI}_M - \mathrm{EI}_m = 1.0000 - 1.0000 = \mathbf{0.0000\text{ bits}}$ (detects *zero* emergence because the $3\times$ jump in determinism is exactly canceled out by the $3\times$ reduction in bit capacity $\log_2 2 / \log_2 8 = 1/3$). *(Note: in fixed-point $10^4$-scaled integer arithmetic in `Loop_causal_emergence.lean` Law 6, $10000 \times 1 - 3333 \times 3 = 1$, i.e. $10^{-4}\text{ bits}$ from rounding $1/3 \approx 0.3333$).*
* **CE 2.0 ($\Delta\mathrm{CP}_{\text{eff}}$):** $\mathrm{CP}_{\text{eff},M} - \mathrm{CP}_{\text{eff},m} = 1.0000 - 0.3333 = \mathbf{+0.6667}$ (detects a $+66.7\%$ gain in intensive causal determinism).

---

## 3. The 5-Step Mathematical Procedure (And Key Theoretical Distinctions)

To apply Causal Emergence to any discrete system, follow the 5 steps illustrated in the workflow diagram at the top of this guide:

### Step 1 — Build the Interventional Micro-TPM $T_m$ ($N \times N$)
Enumerate all $N$ microstates and record the single-step transition probabilities $T_{m,ij} = P(S_{t+1} = j \mid \mathrm{do}(S_t = i))$ under uniform interventions $I_D = \mathcal{U}(1..N)$.

### Step 2 — Pre-Register Competing Candidate Macro-Maps $g: \{1..N\} \to \{1..M\}$
Define your candidate coarse-graining maps $g$ (each partitioning $\{1..N\}$ into $M$ non-empty buckets $U_1, \dots, U_M$ with $2 \le M \le N$) *before* computing scores, so you do not overfit by searching over the exponentially large Bell number $B_N$ of partitions.

### Step 3 — Coarse-Grain $T_m$ into $T_M$ ($M \times M$) via Uniform Within-Bucket Weighting
Under Hoel's (2013, 2026) interventional definition, intervening on macro-state $U_a$ means intervening uniformly ($w_i = 1/|U_a|$) across all microstates $i \in U_a$ and summing the landing probabilities over $j \in U_b$:
$$T_{M,ab} \;=\; \frac{1}{|U_a|}\sum_{i \in U_a}\sum_{j \in U_b} T_{m,ij}.$$
* **Important Subtlety on Markovianity vs. Dynamic Lumpability:** The matrix $T_M$ constructed by the formula above is a valid row-stochastic $M \times M$ Markov matrix by construction. However, **only when the partition $g$ is dynamically lumpable** (i.e., when the row sum $\sum_{j \in U_b} T_{m,ij}$ is identical for every microstate $i \in U_a$) does iterating $T_M^t$ faithfully reproduce the projected multi-step micro-dynamics! When $g$ is *not* lumpable, different microstates inside the same bucket $U_a$ push the system toward different future buckets, inflating the row entropy $H(T_{M,a,\cdot})$ and lowering macro-determinism $\mathrm{det}(T_M)$.

### Step 4 — Compute $\mathrm{det}$, $\mathrm{deg}$, $\mathrm{CP}_{\text{eff}} \in [0, 1]$, $\Delta\mathrm{CP} \in [-1, 1]$, and the Exact $\Delta\mathrm{EI}$ Split

1. **Concavity Proof That $\mathrm{CP}_{\text{eff}}(T) \in [0, 1]$:**
   For any $N \times N$ row-stochastic matrix $T$ with $N \ge 2$, the Shannon entropy function $H(p) = -\sum_j p_j \log_2 p_j$ is strictly concave on the probability simplex. By Jensen's inequality applied to the uniform mixture $\bar{u} = \frac{1}{N}\sum_{i=1}^N T_{i,\cdot}$:
   $$H(\bar{u}) \;=\; H\!\left(\frac{1}{N}\sum_{i=1}^N T_{i,\cdot}\right) \;\ge\; \frac{1}{N}\sum_{i=1}^N H(T_{i,\cdot}) \;=\; \langle H(T_{i,\cdot})\rangle.$$
   Dividing by $\log_2 N > 0$ proves that for every valid TPM:
   $$\mathrm{CP}_{\text{eff}}(T) \;=\; \mathrm{det}(T) - \mathrm{deg}(T) \;=\; \frac{H(\bar{u}) - \langle H(T_{i,\cdot})\rangle}{\log_2 N} \;\in\; [0, 1].$$
   * $\mathrm{CP}_{\text{eff}}(T) = 0$ if and only if all rows of $T$ are identical ($T_{i,\cdot} = \bar{u}$ for all $i$, zero causal information).
   * $\mathrm{CP}_{\text{eff}}(T) = 1$ if and only if $T$ is a permutation matrix ($\langle H(T_{i,\cdot})\rangle = 0$ and $H(\bar{u}) = \log_2 N$).
   * Only the **inter-scale difference** $\Delta\mathrm{CP} = \mathrm{CP}_{\text{eff}}(T_M) - \mathrm{CP}_{\text{eff}}(T_m)$ ranges over $[-1, 1]$.

2. **Why Entropic $\mathrm{CP}_{\text{eff}}$ and Probabilistic $\mathrm{CP}_{\text{sn}}$ Are Distinct Variants (Not Identical!):**
   In Comolatti & Hoel (2025, *Entropy* `27:825`) and Hoel (2026, *Patterns* `7:101472`), the authors compare two distinct ways of defining the intensive Causal Primitive:
   * **Probabilistic Sufficiency + Necessity ($\mathrm{CP}_{\text{sn}}$, Eells 1991; Cheng 1997):** For a specific transition $c \to e$, sufficiency is $\mathrm{suff}(c, e) = P(e \mid \mathrm{do}(c)) = T_{ce}$, and necessity under a uniform counterfactual background is $\mathrm{nec}(c, e) = 1 - P(e \mid \mathrm{do}(C \setminus c)) = 1 - \frac{1}{N-1}\sum_{k \ne c} T_{ke}$. Eells's causal power is $\mathrm{suff} + \mathrm{nec} - 1$ (verified algebraically in `Loop_causal_emergence.lean` Law 5 `comolatti_hoel_2025_causal_consilience`). Averaging over active transitions gives $\mathrm{CP}_{\text{sn}}(T)$.
   * **Entropic Effectiveness ($\mathrm{CP}_{\text{eff}} = \mathrm{det} - \mathrm{deg}$):** Replaces raw transition probabilities with $\log_2 N$-normalized Shannon entropies, where $\mathrm{det}$ plays the *analogous* role to sufficiency and $1 - \mathrm{deg}$ plays the *analogous* role to necessity.
   * **They are NOT numerically equal:** For example, on a $2 \times 2$ uniform noise row $[\tfrac{1}{2}, \tfrac{1}{2}]$, entropic $\mathrm{det} = 0$, whereas probabilistic $P(e \mid c) = 0.5$. In our reference JSON (`hoel_2026_ce2_pymergence_benchmarks.fig2_8state_4selfloop_1block`), on the exact same 8-state matrix, $\mathrm{CP}_{\text{eff},\text{micro}} = 0.6667$ ($\Delta\mathrm{CP}_{\text{eff}} = +0.3333$) whereas $\mathrm{CP}_{\text{sn},\text{micro}} = 0.5893$ ($\Delta\mathrm{CP}_{\text{sn}} = +0.4107$). Always specify whether you are reporting $\mathrm{CP}_{\text{eff}}$ or $\mathrm{CP}_{\text{sn}}$!

3. **Why $\mathrm{EI}$ Alone Can Mislead When $M \ll N$ (While Succeeding on Other $M \ll N$ Systems):**
   From $\mathrm{EI}(T) = \mathrm{CP}_{\text{eff}}(T)\log_2 N$, any coarse-graining $N \to M$ obeys the exact algebraic decomposition (`Loop_causal_emergence.lean` Law 1 `hoel_2013_ei_det_minus_deg_and_size_split`):
   $$\Delta\mathrm{EI} \;=\; \mathrm{EI}_M - \mathrm{EI}_m \;=\; \underbrace{(\mathrm{CP}_{\text{eff},M} - \mathrm{CP}_{\text{eff},m})\log_2 M}_{\Delta I_{\text{eff}} \text{ (Effectiveness Gain)}} \;+\; \underbrace{\mathrm{CP}_{\text{eff},m}(\log_2 M - \log_2 N)}_{\Delta I_{\text{size}} \text{ (Size Penalty } <\,0\text{)}}.$$
   * **When CE 1.0 ($\Delta\mathrm{EI} > 0$) succeeds at $M \ll N$:** If the micro-scale is very noisy or highly degenerate ($\mathrm{CP}_{\text{eff},m}$ is small, e.g., $\mathrm{CP}_{\text{eff},m} = 0.2871$ in Hoel 2013 Fig. 2 with $16 \to 4$ states, or $\mathrm{CP}_{\text{eff},m} = 0.4056$ in Fig. 4 with $64 \to 8$ states), then $\Delta I_{\text{eff}} = +0.978\text{ bits}$ or $+1.783\text{ bits}$ easily outweighs the size penalty $\Delta I_{\text{size}} = -0.574\text{ bits}$ or $-1.217\text{ bits}$, giving positive $\Delta\mathrm{EI} = +0.403\text{ bits}$ and $+0.566\text{ bits}$.
   * **When CE 1.0 ($\Delta\mathrm{EI} < 0$) is blinded by size:** If the micro-scale is *already* fairly deterministic ($\mathrm{CP}_{\text{eff},m} = 0.8496$ on $S_6$ sorting), even jumping to a $100\%$ deterministic macro-law ($\mathrm{CP}_{\text{eff},M} = 0.9687$ on $I(v)$) can only gain $+0.1191$ in $\mathrm{CP}_{\text{eff}}$, so $\Delta I_{\text{eff}} = +0.476\text{ bits}$ is swamped by $\Delta I_{\text{size}} = 0.8496 \times (4.0 - 9.492) = -4.666\text{ bits}$ ($\Delta\mathrm{EI} = -4.189\text{ bits}$). Normalization is a modeling choice: use $\mathrm{EI}$ if you care about total bits of control capacity; use $\mathrm{CP}_{\text{eff}}$ if you care about per-step causal reliability.

### Step 5 — Multi-Scale Path Apportioning & Emergent Complexity ($\mathrm{EC}$)
Given an ordered hierarchy of coarse-grainings $\Pi = (\pi^{(0)}, \pi^{(1)}, \dots, \pi^{(K)})$ from micro ($\pi^{(0)}$) to macro ($\pi^{(K)}$):
1. **Telescoping Path Sum:** The step-wise increments $\Delta\mathrm{CP}_k = \mathrm{CP}(\pi^{(k)}) - \mathrm{CP}(\pi^{(k-1)})$ satisfy the algebraic identity $\mathrm{CP}(\pi^{(0)}) + \sum_{k=1}^K \Delta\mathrm{CP}_k = \mathrm{CP}(\pi^{(K)})$.
2. **Emergent Complexity ($\mathrm{EC}(\Pi)$, Hoel 2026 Eq. 8):** Telescoping alone is a tautology; to measure *how distributed* causal power is across scales, Hoel (2026) defines the non-negative causal contribution vector $\mathbf{c} = (c_0, c_1, \dots, c_K)$ where $c_0 = \mathrm{CP}(\pi^{(0)})$ and $c_k = \max(0, \Delta\mathrm{CP}_k)$, normalizes it to a probability distribution $w_k = c_k / \sum_{j=0}^K c_j$, and defines **Emergent Complexity ($\mathrm{EC}$)** as its Shannon entropy:
   $$\mathrm{EC}(\Pi) \;=\; -\sum_{k=0}^{K} w_k \log_2 w_k \quad \text{(in bits, or } -\sum_{k=0}^{K} w_k \ln w_k \text{ in nats)}.$$
   * **Hand-Worked $\mathrm{EC}$ Example on $S_6$ Sorting:**
     - **2-Level Path $\Pi_2 = (S_6 \to I(v))$:** $c_0 = 0.8496$, $c_1 = \Delta\mathrm{CP}_I = 0.1191$, sum $= 0.9687$. The normalized weights are $w_0 = 0.8496 / 0.9687 = 0.87705$ and $w_1 = 0.1191 / 0.9687 = 0.12295$. Thus:
       $$\mathrm{EC}(\Pi_2) = -\bigl(0.87705\log_2 0.87705 + 0.12295\log_2 0.12295\bigr) = \mathbf{0.5378\text{ bits}}$$
       (matching `ce2_emergent_complexity_bits: 0.5378` in `quantifying_emergence_ce2_metrics.json`).
     - **3-Level Path $\Pi_3 = (S_6 \to \text{Meso } (I,\text{head}) \to I(v))$:** $c_0 = 0.8496$, $c_1 = 0.0477$, $c_2 = 0.0714$, sum $= 0.9687$. The normalized weights are $\mathbf{w} = (0.87705, 0.04924, 0.07371)$, yielding:
       $$\mathrm{EC}(\Pi_3) = -\sum_{k=0}^2 w_k \log_2 w_k = \mathbf{0.6572\text{ bits}}$$
       (matching `ce2_emergent_complexity_3bin_path_bits: 0.6572` in `quantifying_emergence_ce2_metrics.json`).
   * **Reproducibility Note on Hoel (2026) Fig. 3d (`PyMergence` vs. Paper Text):** In `datasets/causal_emergence/causal_emergence_reference.json` (`fig3d_mesoscale_pymergence_8x8`), running the exact $8 \times 8$ matrix from Hoel's `PyMergence` repository yields $\Delta\mathrm{CP}_{\text{eff}} = 0.2336$ and $\mathrm{EC} = 2.117\text{ bits}$, whereas the printed text of Hoel (2026) Fig. 3d reports $\mathrm{CE} = 0.13$ and $\mathrm{EC} = 2.56\text{ bits}$ (due to different path-slicing/logarithm conventions between the manuscript figure and the released notebook).

---

## 4. Validated Python Reference Implementation (~60 Lines of NumPy)

Here is a complete, defensive NumPy implementation that validates row-stochasticity, guards against the degenerate $M \le 1$ single-bucket trap (returning $\mathrm{CP}_{\text{eff}} = 0.0$ rather than $1.0$ when $n \le 1$, so collapsing a system to 1 state never produces spurious emergence!), and computes $\mathrm{det}$, $\mathrm{deg}$, $\mathrm{CP}_{\text{eff}}$, $\mathrm{EI}$, and $\mathrm{EC}$:

```python
import numpy as np
from typing import Sequence

def compute_ce1_and_ce2(tpm: np.ndarray, tol: float = 1e-6) -> dict:
    """
    Compute Erik Hoel's Causal Emergence 1.0 (PNAS 2013) & 2.0 (Patterns 2026)
    entropic primitives for an N x N row-stochastic Transition Probability Matrix.
    """
    tpm = np.asarray(tpm, dtype=float)
    if tpm.ndim != 2 or tpm.shape[0] != tpm.shape[1]:
        raise ValueError(f"TPM must be square 2D array, got shape {tpm.shape}")
    n = tpm.shape[0]
    if n <= 1:
        # Guard: a 1-state partition has log2(1) = 0 and transmits 0 bits of causal info
        return {"n_states": int(n), "log2_n": 0.0, "determinism": 0.0, "degeneracy": 1.0, "cp_eff": 0.0, "EI_bits": 0.0}
    if np.any(tpm < -tol) or not np.allclose(tpm.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Each row of TPM must be a valid probability distribution summing to 1.0")

    log2_n = float(np.log2(n))
    p_clip = np.clip(tpm, 1e-15, 1.0)

    # 1. Determinism: 1 - <H(row_i)> / log2(N) in [0, 1]
    row_entropies = -np.sum(tpm * np.log2(p_clip), axis=1)
    det = 1.0 - float(np.mean(row_entropies)) / log2_n

    # 2. Degeneracy: 1 - H(<row>) / log2(N) in [0, 1]
    u_e = np.mean(tpm, axis=0)
    u_clip = np.clip(u_e, 1e-15, 1.0)
    eff_entropy = float(-np.sum(u_e * np.log2(u_clip)))
    deg = 1.0 - eff_entropy / log2_n

    # 3. CE 2.0 Entropic Effectiveness CP_eff in [0, 1] & CE 1.0 EI in bits
    cp_eff = max(0.0, det - deg)
    ei_bits = cp_eff * log2_n
    return {
        "n_states": int(n),
        "log2_n": round(log2_n, 4),
        "determinism": round(det, 4),
        "degeneracy": round(deg, 4),
        "cp_eff": round(cp_eff, 4),
        "EI_bits": round(ei_bits, 4),
    }

def coarse_grain_tpm(tpm_micro: np.ndarray, macro_labels: Sequence[int]) -> np.ndarray:
    """
    Coarse-grain an N x N micro-TPM onto M macro-states via uniform within-bucket
    interventional weighting (O(N^2) dense vectorized aggregation).
    """
    tpm_micro = np.asarray(tpm_micro, dtype=float)
    n = tpm_micro.shape[0]
    if len(macro_labels) != n:
        raise ValueError(f"Expected {n} macro_labels, got {len(macro_labels)}")
    _, inv = np.unique(macro_labels, return_inverse=True)
    m = int(inv.max()) + 1
    # One-hot membership matrix P of shape (N, M)
    P = np.zeros((n, m), dtype=float)
    P[np.arange(n), inv] = 1.0
    bucket_sizes = P.sum(axis=0, keepdims=True).T  # shape (M, 1)
    # T_M = (P^T / |U_a|) @ T_m @ P
    return (P.T / bucket_sizes) @ tpm_micro @ P
```

---

## 5. Worked Empirical Case Studies (And What They Reveal)

### Case Study 1 — Autonomous Sorting Morphogenesis Across $S_3, S_4, S_5, S_6$ ($N = 6 \dots 720$)
In Zhang, Goldstein & Levin (2025), $n$ integers sort themselves via local Bubble swaps. On the symmetric group $S_n$, we compare two macro-variables:
* **Coxeter Pairwise Inversions $I(v) = \#\{(i < j) : v_i > v_j\}$** ($M = 1 + \binom{n}{2}$ levels), and
* **Adjacent Monotonicity Error $E(v) = \#\{i : v_i > v_{i+1}\}$** ($M = n$ levels),
along with an intermediate **Meso-Scale $(I(v), \mathbf{1}[v_0 > n/2])$** that tracks both $I(v)$ and whether the head element is in the upper half of keys.

Here is the complete table across $S_3, S_4, S_5, S_6$ from `sorting-morphogenesis/figures/quantifying_emergence_ce2_metrics.json`:

| Group ($N = n!$) | Micro $\mathrm{CP}_{\text{swap}}$ | Meso $\mathrm{CP}_{\text{swap}}$ ($\Delta\mathrm{CP}_{\text{micro}\to\text{meso}}$) | Macro $I(v)$ $\mathrm{CP}_{\text{swap}}$ ($\mathrm{det}_I$) | Active-Swap $\Delta\mathrm{CP}_I$ | **Like-for-Like Raw $\Delta\mathrm{CP}_{I,\text{raw}}$** | Macro $E(v)$ $\mathrm{CP}_{\text{swap}}$ ($\Delta\mathrm{CP}_E$) | CE 1.0 $\Delta\mathrm{EI}_I$ ($\Delta I_{\text{eff}} + \Delta I_{\text{size}}$) | $\mathrm{EC}$ (`2-bin` / `3-bin`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$S_3$ ($N = 6$)** | `0.6934` | `0.5904` (**`-0.1030`**) | `0.7500` (`1.000`) | **`+0.0566`** | **`-0.0142`** (`0.5591 - 0.5733`) | `0.4206` (**`-0.2728`**) | `-0.2925b` (`+0.1132 - 0.4056`) | `0.3860b` / `0.6698b` |
| **$S_4$ ($N = 24$)** | `0.8045` | `0.7959` (**`-0.0086`**) | `0.8982` (`1.000`) | **`+0.0937`** | **`-0.0299`** (`0.6688 - 0.6987`) | `0.5059` (**`-0.2986`**) | `-1.1671b` (`+0.2631 - 1.4301`) | `0.4825b` / `0.4825b` |
| **$S_5$ ($N = 120$)** | `0.8329` | `0.8652` (`+0.0323`) | `0.9474` (`1.000`) | **`+0.1145`** | **`-0.0417`** (`0.7272 - 0.7689`) | `0.5573` (**`-0.2756`**) | `-2.4749b` (`+0.3961 - 2.8714`) | `0.5318b` / `0.6426b` |
| **$S_6$ ($N = 720$)** | `0.8496` | `0.8973` (`+0.0477`) | `0.9687` (`1.000`) | **`+0.1191`** | **`-0.0501`** (`0.7629 - 0.8130`) | `0.5971` (**`-0.2525`**) | `-4.1890b` (`+0.4764 - 4.6659`) | `0.5378b` / `0.6572b` |

**Three Critical Lessons from This Table:**
1. **Why Active-Swap $I(v)$ Is Analytic:** On active swaps, every legal Bubble swap decreases $I(v)$ by $\Delta I = -1$, so $T_{M,I}$ is an exact countdown matrix (`k → k-1` for $k \ge 1$, and `0 → 0`). Thus $\mathrm{det}_I = 1.0000$, and only the absorbing boundary (`1 → 0` and `0 → 0` sharing column `0`) creates a tiny degeneracy $\mathrm{deg}_I = \frac{2/M}{\log_2 M} = \frac{2/16}{4} = 0.03125$, giving $\mathrm{CP}_{\text{eff},I} = 1 - 0.03125 = 0.96875$ (`EI = 3.875 bits`) in closed form!
2. **Look at the Meso Column on $S_3$ and $S_4$:** On $S_3$ and $S_4$, the intermediate meso-partition $(I(v), \text{head})$ actually **loses** causal power ($\Delta\mathrm{CP}_{\text{micro}\to\text{meso}} = -0.1030$ and $-0.0086$) because splitting inversion levels by the head element introduces non-lumpable branching noise when $n$ is small, before turning positive ($+0.0323$ and $+0.0477$) on $S_5$ and $S_6$.
3. **Look at the Like-for-Like Raw Step Column ($\Delta\mathrm{CP}_{I,\text{raw}} < 0$):** Why is $\Delta\mathrm{CP}_I = +0.1191 > 0$ on active swaps (`tpm_swap`), yet **negative (`-0.0501` on $S_6$)** when no-op self-loops are included (`macro_I_raw` `0.7629` vs. `micro_raw` `0.8130`)? Because on raw proposal steps, each macro-level $I = k$ branches between two outcomes: stay at $k$ (no swap) or drop to $k-1$ (swap). That binary choice has $\sim 0.89\text{ bits}$ of row entropy, which CE 2.0 divides by $\log_2 16 = 4.0\text{ bits}$ ($\mathrm{det}_{I,\text{raw}} = 0.7776$), whereas at the micro-scale ($N = 720$), $1.70\text{ bits}$ of row entropy is divided by $\log_2 720 = 9.49\text{ bits}$ ($\mathrm{det}_{m,\text{raw}} = 0.8210$)! Calling `CP(macro_I_swap) - CP(micro_raw) = +0.1557` "raw emergence" mixes temporal coarse-graining (filtering out no-op self-loops) with spatial coarse-graining. **Spatial coarse-graining onto $I(v)$ alone only achieves positive $\Delta\mathrm{CP}$ when paired with event-driven (active-swap) time steps.**

### Case Study 2 — Hoel & Levin (2020) 5-Gene Mammalian Cardiac GRN (`32 → 18` States)
In `datasets/causal_emergence/causal_emergence_reference.json` (`hoel_levin_2020_biological_grn_benchmarks.cardiac_grn_5gene_herrmann2012`) and `Loop_causal_emergence.lean` Law 4 (`hoel_levin_2020_cardiac_grn_macronodes`):
* **Exact Numbers:** The deterministic 5-gene Boolean network (`exogen_canWnt_II`, `Foxc1_2`, `Fgf8`, `canWnt`, `Isl1`; $N = 32, \log_2 N = 5.0\text{ bits}$) has micro-scale $\mathrm{det}_{\text{micro}} = 5.0\text{ bits}$ ($1.0000$ normalized), $\mathrm{deg}_{\text{micro}} = 2.2111\text{ bits}$ ($0.4422$ normalized), $\mathrm{EI}_{\text{micro}} = 2.7889\text{ bits}$, and $\mathrm{CP}_{\text{eff,micro}} = 2.7889 / 5.0 = \mathbf{0.5578}$.
* Grouping the 8 microstates with `(exogen_canWnt_II = 1, Foxc1_2 = 0)` into macro-node $\mu_1$ (all 8 transition deterministically to `01011`) and the 8 microstates with `(exogen_canWnt_II = 1, Foxc1_2 = 1)` into macro-node $\mu_2$ (all 8 transition deterministically to `01111`) reduces the state space from $N = 32 \to M = 18$ states ($\log_2 18 = 4.1699\text{ bits}$), lowering degeneracy from $2.2111 \to 1.2642\text{ bits}$ and raising $\mathrm{EI}_{\text{macro}}$ to $\mathbf{2.9058\text{ bits}}$ ($\Delta\mathrm{EI} = \mathbf{+0.1169\text{ bits}}$, with derived $\mathrm{CP}_{\text{eff,macro}} = 2.9058 / 4.1699 = \mathbf{0.6968}$, $\Delta\mathrm{CP}_{\text{eff}} = \mathbf{+0.1390}$).

---

## 6. Five Fundamental Limitations & Criticisms of Causal Emergence in Practice

Why should a mathematician, physicist, or biologist treat Causal Emergence with healthy skepticism rather than as an automatic discovery engine? Here are the five most important structural limitations:

1. **The "Garden-of-Eden" Artificial Intervention Bias (Illustrated by Case Study 2 Above!):**
   Look closely at the update rule for `exogen_canWnt_II` (`eW`) in the 5-gene cardiac GRN (`Loop_causal_emergence.lean` L140): `neW = 0` (`false`). That means `exogen_canWnt_II` is an **external transient input** that resets to `0` after step 1 and can *never* be turned back on by the network! Consequently, **all 16 states with `eW = 1` have in-degree 0 (they are unreachable "Garden-of-Eden" states)**. The entire $\Delta\mathrm{EI} = +0.1169\text{ bit}$ "emergence" in Case Study 2 comes from collapsing 16 unreachable input states into 2 macro-nodes under uniform `do()` interventions. As critics (e.g., Dewhurst 2021; Eberhardt & Lee 2022) have noted, injecting a uniform `do()` distribution across states that the physical system can never visit manufactures degeneracy that is a property of the *intervention protocol*, not the autonomous biology.
2. **Still Sensitive to $\log_2 N$ Normalization Whenever $\mathrm{det}_M < 1$ (Illustrated by Raw vs. Swap Steps in Case Study 1!):**
   While CE 2.0 fixes CE 1.0's failure on $100\%$ deterministic macro-chains ($\mathrm{det}_M = 1 \implies \langle H\rangle / \log_2 M = 0$), dividing non-zero row entropy $\langle H\rangle > 0$ by $\log_2 M$ still penalizes small $M \ll N$ macro-spaces whenever the macro-chain has self-loops or residual stochasticity (which is why like-for-like raw $\Delta\mathrm{CP}_{I,\text{raw}} = -0.0501 < 0$ on $S_6$!).
3. **Exponential $N \times N$ State-Space Explosion:**
   Building an exact $N \times N$ interventional TPM requires $O(N^2)$ transitions. For $n = 6$ sorting cells, $N = 6! = 720$; for $n = 7$, $N = 5,040$ ($2.5 \times 10^7$ entries); and for $n = 100$ sorting cells (the actual size used in Zhang, Goldstein & Levin 2025), $N = 100! \approx 9.3 \times 10^{157}$—making exact TPM construction physically impossible. On real high-dimensional systems, you must already know the analytic macro-coordinate ($I(v)$) to sample it.
4. **Analytic Redundancy When a Lyapunov Function Is Already Known:**
   In sorting morphogenesis, you do not need a $720 \times 720$ Markov matrix to know that $I(v)$ is a $100\%$ deterministic macro-coordinate: a 2-line proof (`Loop_zhang_sorting_morphogenesis.lean` Theorem 3 `adjacent_swap_strictly_decreases_inversions_window3`) proves $\Delta I = -1$ for all $n \ge 2$. Causal Emergence is a post-hoc information-theoretic thermometer that scores candidate coordinates after you propose them; it does not replace domain-specific mathematical analysis.
5. **Bell-Number Partition Search Overfitting ($B_N$):**
   The number of ways to partition $N$ microstates into macro-buckets is the Bell number $B_N$ ($B_{16} \approx 10^{10}$, $B_{720} > 10^{1000}$). If you use greedy optimization, simulated annealing, or spectral clustering (Griebenow, Klein & Hoel 2019) to search for a partition that maximizes $\Delta\mathrm{EI}$ or $\Delta\mathrm{CP}$ on a single TPM without pre-registering the partition or testing on held-out dynamics, you can easily find ad-hoc state groupings that score $\Delta\mathrm{CP} > 0$ by chance but have zero physical meaning.

---

## 7. References

1. **Hoel, E. P., Albantakis, L., & Tononi, G. (2013).** *Quantifying causal emergence shows that macro can beat micro.* **PNAS**, 110(49), 19790–19795. [doi:10.1073/pnas.1314922110](https://doi.org/10.1073/pnas.1314922110)
2. **Hoel, E. P., Albantakis, L., Marshall, W., & Tononi, G. (2016).** *Can the macro beat the micro? Integrated information across spatiotemporal scales.* **Neuroscience of Consciousness**, 2016(1), niw012. [doi:10.1093/nc/niw012](https://doi.org/10.1093/nc/niw012)
3. **Hoel, E. P. (2017).** *When the Map Is Better Than the Territory.* **Entropy**, 19(5), 188. [doi:10.3390/e19050188](https://doi.org/10.3390/e19050188)
4. **Griebenow, R., Klein, B., & Hoel, E. P. (2019).** *Finding the right scale of a network: efficient identification of causal emergence through spectral clustering.* **arXiv:1908.07565**.
5. **Klein, B., & Hoel, E. P. (2020).** *The Emergence of Informative Higher Scales in Complex Networks.* **Complexity**, 2020, 8932526. [doi:10.1155/2020/8932526](https://doi.org/10.1155/2020/8932526)
6. **Hoel, E. P., & Levin, M. (2020).** *Emergence of informative higher scales in biological systems: a computational toolkit for optimal prediction and control.* **Communicative & Integrative Biology**, 13(1), 108–118. [doi:10.1080/19420889.2020.1802914](https://doi.org/10.1080/19420889.2020.1802914)
7. **Dewhurst, J. (2021).** *Causal emergence and the interventionist conception of causation.* **Philosophy of Science** / commentary on uniform intervention priors in Effective Information.
8. **Eberhardt, F., & Lee, J. (2022).** *Causal Emergence: When is a macro-variable really better?* Critique of uniform `do()` distributions and partition search in coarse-grained Markov chains.
9. **Comolatti, R., & Hoel, E. P. (2025).** *Consilience in Causation: Causal Emergence Is Found Across Measures of Causation.* **Entropy**, 27(8), 825. [doi:10.3390/e27080825](https://doi.org/10.3390/e27080825)
10. **Hoel, E. P. (2026).** *Quantifying emergent complexity (Causal Emergence 2.0).* **Patterns**, 7(1), 101472 (`arXiv:2503.13395`, `PyMergence` Zenodo [doi:10.5281/zenodo.17210078](https://doi.org/10.5281/zenodo.17210078)). [doi:10.1016/j.patter.2025.101472](https://doi.org/10.1016/j.patter.2025.101472)
