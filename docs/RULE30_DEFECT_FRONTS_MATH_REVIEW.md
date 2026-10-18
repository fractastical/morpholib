# Mathematical Review Brief: Left-Cone Diagonal Carry-Lookahead Algebra, All-Zero Barrier Trees & 1D Space-Time Defect Fronts in Rule 30

**Executable Verification Script:** [`1984-wolfram-conus-textile-rule30.py`](../1984-wolfram-conus-textile-rule30.py) (`python3 1984-wolfram-conus-textile-rule30.py --self-test-only`)  
**Exported Data Table:** [`conus_rule30_defect_fronts.csv`](../conus_rule30_defect_fronts.csv) · [`datasets/conus/conus_textile_reference.json`](../datasets/conus/conus_textile_reference.json)  
**Primary Figures:** [`docs/media/conus_rule30_kymograph_defect_fronts.png`](media/conus_rule30_kymograph_defect_fronts.png) · [`docs/media/conus_textile_rule30.gif`](media/conus_textile_rule30.gif)

![1D Space-Time Kymograph & Defect-Front Tooling](media/conus_rule30_kymograph_defect_fronts.png)

---

## Executive Summary for Mathematical Reviewers: *What Is Classical vs. What Is Novel Here?*

This document is structured so a mathematician in dynamical systems, symbolic dynamics, or cellular automata can immediately distinguish **known classical theorems** from the **new algebraic formulations, computational theorems, and cross-model unifications** introduced in this repository.

### 1. What Is Classical (Known in the Literature)
- **Wolfram (1983, 1984, 2002; 2019 Rule 30 Prizes):** Defines Elementary Cellular Automaton Rule 30 over $\mathbb{F}_2$, notes the coexistence of a regular periodic/period-doubling left half with a chaotic right half on the single-seed light-cone, and poses three open prize problems on the center column $a_{t,0}$ ([OEIS A051023](https://oeis.org/A051023)).
- **Erica Jen (1990, *Physica D* 45:3–18):** Proves left-bijective spatial invertibility $a_{t,i-1} = a_{t+1,i} \oplus (a_{t,i} \lor a_{t,i+1})$, introduces left-cone diagonal coordinates $d_k(t) = a_{t,-t+k}$, and proves that temporal period doubling $2^m \to 2^{m+1}$ along diagonal $k$ requires an asymptotically all-zero diagonal $d_{k-1}(t) \equiv 0$ paired with an odd-parity upstream cycle $d_{k-2}(t)$.
- **Meinhardt & Klingler (1987, *J. Theor. Biol.* 126:63–89):** Formulates continuous 1D mantle-edge activator–substrate–hormone reaction–diffusion PDEs for *Conus textile* tent patterns and V-shaped lesion recovery wakes.

### 2. What Is Newly Formulated, Proved & Verified Here
1. **Kogge–Stone Carry-Lookahead Factorization of the Diagonal Cycle-Transfer Operator ($\mathcal{O}(\log_2 P)$ per diagonal, Theorem 1):**
   - We rewrite the asymptotic diagonal recurrence $d_k(t+1) = d_{k-2}(t) \oplus (d_{k-1}(t) \lor d_k(t))$ on the cyclic ring $\mathbb{F}_2[\mathbb{Z}/P\mathbb{Z}]$ ($P = 2^m$) in **generate–propagate form**:
     $$d_k(t) \;=\; g_0(t) \;\oplus\; p_0(t)\,d_k(t-1), \qquad g_0 = \mathrm{rotl}_1(d_{k-2} \oplus d_{k-1}), \quad p_0 = \mathrm{rotl}_1(\overline{d_{k-1}}).$$
   - Whenever $d_{k-1} \ne \mathbf{0}_P$, the cyclic product of the propagate mask vanishes ($\prod_{t=0}^{P-1} p_0(t) = 0$), so **$m = \log_2 P$ rounds of Kogge–Stone parallel-prefix doubling** ($s = 1, 2, 4, \dots, 2^{m-1}$) annihilate $p_m = \mathbf{0}_P$ and yield the unique periodic orbit $c_k = g_m \in \mathbb{F}_2^P$ in closed form without simulating the 2D cellular automaton.
   - In this algebraic language, **Jen's All-Zero Barrier ($d_{k-1} = \mathbf{0}_P$) is simply a full carry-propagate ring ($p_0 = \mathbf{1}_P$)** where $p_m = \mathbf{1}_P$ fails to vanish and $g_m = \bigl(\bigoplus_{t=0}^{P-1} d_{k-2}(t)\bigr)\mathbf{1}_P$ is the total loop parity.
2. **Transient-Independence of the First Four Doublings ($k = 3, 8, 29, 400$) vs. The Even-Parity Binary Bifurcation Tree on the $P = 16$ Plateau ($k \ge 53\,208$, Theorems 2 & 3):**
   - **Why $k_1 = 3, k_2 = 8, k_3 = 29, k_4 = 400$ and the first $P=16$ barrier at $k = 53\,207$ are transient-independent (Theorem 2):** At every odd-parity barrier ($d_{k-1} = \mathbf{0}_P$, $\mathrm{wt}(d_{k-2}) \equiv 1 \pmod 2$), the two possible choices of the transient phase bit $\sigma = d_k(t_0) \in \{0, 1\}$ produce period-$2P$ orbits that are **conjugate under the half-period time shift $\tau_P: t \mapsto t + P \pmod{2P}$**. Because $\tau_P$ fixes $d_{k-2}$ and $d_{k-1}$ and commutes with the cycle-transfer operator $\mathcal{T}_{2P}$, **finite-time transient chaos cannot alter subsequent barrier positions** as long as only odd-parity barriers are encountered.
   - **Discovery of the First Even-Parity Silent Barrier at $k - 1 = 53\,207$ and Proof of Acyclicity on the $P=16$ Plateau (Theorem 3):** On the $P = 16$ plateau starting at $(c_{399}, c_{400}) = (\mathbf{0}_{16}, \mathtt{0x27d8})$, the first all-zero barrier occurs at **$k - 1 = 53\,207$** ($\Delta k = 52\,808$), where the upstream cycle $c_{53206} = \mathtt{0xca30}$ has **even Hamming weight** $\mathrm{wt}(c_{53206}) = 6 \equiv 0 \pmod 2$. Here the period **does not double** ($P(53208) = 16$), and the two transient choices $\sigma \in \{0, 1\}$ split into **bitwise-complementary period-16 orbits in distinct $\mathbb{Z}/16\mathbb{Z}$-shift classes** ($\mathtt{0x8c20}$ of weight $3$ vs. $\mathtt{0x73df}$ of weight $13$).
   - By exhaustive Kogge–Stone traversal (1.2 seconds in [`1984-wolfram-conus-textile-rule30.py`](../1984-wolfram-conus-textile-rule30.py)), we prove that the resulting binary decision tree on $(\mathbb{F}_2^{16})^2$ is **finite and acyclic**, with **15 internal even-parity split nodes** and **16 odd-parity $P = 16 \to 32$ leaf nodes** spanning $k_5 \in [87\,867,\, 894\,235]$. Thus, **every branch is unconditionally guaranteed to double to period $P = 32$** without ever entering a spatial cycle in $k$.
3. **Cross-Paradigm 1D Space-Time Defect-Front Synthesis:**
   - Unifies four 1D morphogenetic front regimes in a single mathematical taxonomy: (i) ballistic/directed-percolation $\mathbb{F}_2$ Jacobian fronts in Rule 30, (ii) excitable wave-branching recovery fronts in the Meinhardt–Klingler PDE, (iii) local pinning vs. non-local defect tunneling in autonomous cell sorting, and (iv) overdamped diffusive domain walls in antagonistic bioelectric reaction–diffusion systems.

---

## 1. Exact Algebraic Setup & Left-Cone Diagonal Coordinates

Let $a_{t,i} \in \mathbb{F}_2 = \{0, 1\}$ ($t \in \mathbb{Z}_{\ge 0}$, $i \in \mathbb{Z}$) evolve under Elementary Cellular Automaton **Rule 30** from the single-site seed $a_{0,i} = \delta_{i,0}$:

$$a_{t+1,\,i} \;=\; f(a_{t,i-1},\, a_{t,i},\, a_{t,i+1}) \;=\; a_{t,i-1} \,\oplus\, \bigl(a_{t,i} \lor a_{t,i+1}\bigr) \;\equiv\; a_{t,i-1} + a_{t,i} + a_{t,i+1} + a_{t,i}a_{t,i+1} \pmod 2.$$

### 1.1 Exact $\mathbb{F}_2$ Boolean Jacobian & Asymmetric Damage Cone
For any Boolean function $f: \mathbb{F}_2^3 \to \mathbb{F}_2$, the partial Boolean derivative with respect to input $x_j$ is $\frac{\partial f}{\partial x_j} = f(\dots, x_j \oplus 1, \dots) \oplus f(\dots, x_j, \dots)$. For Rule 30:

$$\frac{\partial f}{\partial l} \;=\; 1, \qquad \frac{\partial f}{\partial c} \;=\; 1 \oplus r \;=\; \bar{r}, \qquad \frac{\partial f}{\partial r} \;=\; 1 \oplus c \;=\; \bar{c}.$$

- **Rightward perturbation front ($v_R = +1$):** Because $\partial f / \partial l = 1$ identically on $\mathbb{F}_2^3$, any bit flip at $(t, i)$ flips $(t+1, i+1)$ with probability $1$ whenever $(t, i)$ is the rightmost damaged site. Thus the right edge of the damage cone $\Delta a_{t,i} = a_{t,i} \oplus a'_{t,i}$ propagates deterministically at the light-cone bound:
  $$v_R \;=\; +1.0000 \pm 0.0000\text{ cells/step}.$$
- **Leftward perturbation front ($v_L \approx -0.260$):** Because $\partial f / \partial r = \bar{c}$, a bit flip at $(t, i)$ propagates leftward to $(t+1, i-1)$ **if and only if** $a_{t,i-1} = 0$. Whenever $a_{t,i-1} = 1$, the `OR` gate saturates ($1 \lor r = 1$) and blocks the perturbation. Across a 25-seed random ensemble (`compute_damage_cone_ensemble()`), the left damage front drifts sub-ballistically at:
  $$v_L \;=\; -0.2604 \pm 0.1024\text{ cells/step}.$$

### 1.2 Left-Cone Diagonal Coordinate Transformation
Define the $k$-th left-cone diagonal ($k \in \mathbb{Z}_{\ge 0}$) at time $t \ge \lceil k/2 \rceil$ by measuring inward from the left boundary $i = -t$:

$$d_k(t) \;=\; a_{t,\, -t + k} \qquad \Longleftrightarrow \qquad i \;=\; -t + k.$$

Substituting $i = -(t+1) + k = -t + (k-1)$ into Rule 30 yields the **autonomous triangular diagonal recurrence**:

$$\boxed{\;d_k(t+1) \;=\; d_{k-2}(t) \,\oplus\, \bigl(d_{k-1}(t) \lor d_k(t)\bigr), \qquad k \ge 2,\;\; t \ge k - 1,\;}$$

with boundary diagonals $d_0(t) \equiv 1$ ($t \ge 0$) and $d_1(t) \equiv 1$ ($t \ge 1$).

---

## 2. Theorem 1: Kogge–Stone Carry-Lookahead Factorization of the Diagonal Cycle-Transfer Operator

Suppose diagonals $k-2$ and $k-1$ have locked into periodic orbits $c_{k-2}, c_{k-1} \in \mathbb{F}_2^P$ of period $P = 2^m$ for all $t \ge T_{\text{lock}}(k-1)$. Over $\mathbb{F}_2$, the binary `OR` operation satisfies $b \lor x = b \oplus \bar{b}\,x$, where $\bar{b} = 1 \oplus b$. Shifting the time index by $1$ on the cyclic group $\mathbb{Z}/P\mathbb{Z}$, the diagonal recurrence becomes a first-order linear non-homogeneous recurrence over $\mathbb{F}_2$:

$$d_k(t) \;=\; g_0(t) \;\oplus\; p_0(t)\, d_k(t-1),$$

where the **generate** and **propagate** vectors $g_0, p_0 \in \mathbb{F}_2^P$ are defined by:

$$g_0(t) \;=\; c_{k-2}(t-1) \oplus c_{k-1}(t-1), \qquad p_0(t) \;=\; \overline{c_{k-1}(t-1)} \;=\; 1 \oplus c_{k-1}(t-1).$$

> **Theorem 1 (Kogge–Stone Cycle-Transfer Operator & Carry-Lookahead Dichotomy).**  
> Let $\mathrm{rotl}_s: \mathbb{F}_2^P \to \mathbb{F}_2^P$ denote the cyclic shift $(\mathrm{rotl}_s x)(t) = x(t - s \bmod P)$. Define the $m = \log_2 P$ **Kogge–Stone parallel-prefix doubling sequence** $(g_r, p_r) \in \mathbb{F}_2^P \times \mathbb{F}_2^P$ for $r = 0, 1, \dots, m-1$ with stride $s = 2^r$:
> $$g_{r+1} \;=\; g_r \,\oplus\, \bigl(p_r \land \mathrm{rotl}_{2^r}(g_r)\bigr), \qquad p_{r+1} \;=\; p_r \land \mathrm{rotl}_{2^r}(p_r).$$
> After $m = \log_2 P$ doubling steps, every bit of $p_m$ equals the cyclic conjunction $\Pi = \bigwedge_{t=0}^{P-1} p_0(t) = \bigwedge_{t=0}^{P-1} \overline{c_{k-1}(t)}$, yielding an exact dichotomy:
>
> 1. **Synchronized Regime ($c_{k-1} \ne \mathbf{0}_P \iff \Pi = 0$):**  
>    The final propagate vector vanishes identically ($p_m = \mathbf{0}_P$), and for **every** initial transient state, $d_k(t)$ locks (within at most $P$ steps after $T_{\text{lock}}(k-1)$) onto the **unique** periodic orbit of period dividing $P$:
>    $$c_k \;=\; \mathcal{T}_P(c_{k-2},\, c_{k-1}) \;=\; g_m \;\in\; \mathbb{F}_2^P.$$
> 2. **All-Zero Barrier Regime ($c_{k-1} = \mathbf{0}_P \iff \Pi = 1$):**  
>    The final propagate vector is all-ones ($p_m = \mathbf{1}_P$), and for every $t \ge T_{\text{lock}}(k-1)$, moving once around the period-$P$ cycle shifts $d_k$ by the total Hamming parity $\Omega = \bigoplus_{t=0}^{P-1} c_{k-2}(t) = \mathrm{wt}(c_{k-2}) \bmod 2$:
>    $$d_k(t + P) \;=\; d_k(t) \,\oplus\, \Omega.$$
>    - **2a. Odd-Parity Dislocation ($\Omega = 1$):** $d_k(t + P) = \overline{d_k(t)}$ and $d_k(t + 2P) = d_k(t)$, so the minimal period **doubles** from $P = 2^m$ to $2P = 2^{m+1}$.
>    - **2b. Even-Parity Silent Barrier ($\Omega = 0$):** $d_k(t + P) = d_k(t)$, so the period **remains $P = 2^m$**, and the two choices of transient phase bit $\sigma = d_k(t_0) \in \{0, 1\}$ yield two bitwise-complementary period-$P$ orbits $c_k^{(0)}$ and $c_k^{(1)} = \overline{c_k^{(0)}}$.

*Proof.* By induction on $r$, expanding $d_k(t) = g_0(t) \oplus p_0(t)\,d_k(t-1)$ across $2^r$ steps gives $d_k(t) = g_r(t) \oplus p_r(t)\,d_k(t - 2^r)$ for all $t \ge T_{\text{lock}}(k-1) + 2^r$. Setting $r = m$ ($2^m = P$) yields:
$$d_k(t) \;=\; g_m(t) \;\oplus\; p_m(t)\, d_k(t - P), \qquad p_m(t) \;=\; \prod_{\tau=0}^{P-1} \overline{c_{k-1}(\tau)} \;=\; \Pi.$$
If $c_{k-1} \ne \mathbf{0}_P$, at least one factor $\overline{c_{k-1}(\tau)} = 0$, so $\Pi = 0$, $p_m = \mathbf{0}_P$, and $d_k(t) = g_m(t \bmod P)$ is independent of $d_k(t-P)$. If $c_{k-1} = \mathbf{0}_P$, then $p_0 = \mathbf{1}_P$ and $g_0 = \mathrm{rotl}_1(c_{k-2})$, so $p_m = \mathbf{1}_P$ and $g_m(t) = \bigoplus_{\tau=0}^{P-1} c_{k-2}(\tau) = \Omega$, giving $d_k(t) = \Omega \oplus d_k(t-P)$. $\square$

---

## 3. Theorem 2 & Theorem 3: Transient-Independence Through $k = 53\,207$ vs. The Acyclic Bifurcation Tree on $P = 16$

A natural question for a mathematician reviewing Theorem 1 is:
> *"At each all-zero barrier $c_{k-1} = \mathbf{0}_P$, the asymptotic orbit $c_k$ depends on a 1-bit initial condition $\sigma = d_k(t_0) \in \{0, 1\}$ determined by the finite-time transient $t < T_{\text{lock}}(k)$. Why doesn't this transient bit change the location of subsequent period-doubling dislocations $k_1, k_2, k_3, k_4$?"*

The answer reveals a sharp structural distinction between **odd-parity barriers** ($\Omega = 1$) and **even-parity barriers** ($\Omega = 0$):

> **Theorem 2 (Time-Shift Equivariance at Odd-Parity Barriers).**  
> Suppose $c_{k-1} = \mathbf{0}_P$ is an odd-parity barrier ($\Omega = \mathrm{wt}(c_{k-2}) \bmod 2 = 1$), so $P \to 2P$. Let $c_k^{(0)}, c_k^{(1)} \in \mathbb{F}_2^{2P}$ denote the two period-$2P$ orbits corresponding to $\sigma = 0$ and $\sigma = 1$. Then:
> 1. Since $d_k(t + P) = d_k(t) \oplus 1$, shifting time by $P$ steps ($\tau_P: t \mapsto t + P \pmod{2P}$) inverts every bit of $c_k$, so **$c_k^{(1)} = \overline{c_k^{(0)}} = \tau_P\bigl(c_k^{(0)}\bigr)$**.
> 2. Because $c_{k-2}$ and $c_{k-1} = \mathbf{0}$ have period dividing $P$, they are fixed by $\tau_P$: $\tau_P(c_{k-2}) = c_{k-2}$ and $\tau_P(c_{k-1}) = c_{k-1}$.
> 3. Since the Kogge–Stone operator $\mathcal{T}_{2P}$ is equivariant under cyclic time shifts ($\mathcal{T}_{2P}(\tau_P x, \tau_P y) = \tau_P \mathcal{T}_{2P}(x, y)$), **both branches $\sigma \in \{0, 1\}$ generate identical subsequent orbits up to the global time shift $\tau_P$** ($c_j^{(1)} = \tau_P c_j^{(0)}$ for all $j \ge k$ up to the next all-zero barrier). Since both the all-zero condition ($c = \mathbf{0}_{2P}$) and Hamming weight ($\mathrm{wt}(c)$) are invariant under $\tau_P$, **the transient bit $\sigma$ at an odd-parity barrier has zero effect on the index $k$ or parity $\Omega$ of the next all-zero barrier.**

### 3.1 Exact Table of the First Five Transient-Independent Barriers ($k \le 53\,207$)

Applying $\mathcal{T}_P$ purely algebraically starting from $c_0 = (1), c_1 = (1)$ (verified with `0` bit errors against the 2D light-cone in `extract_diagonal_period_cascade()`):

| Barrier Diagonal $k-1$ | Dislocation Diagonal $k$ | Incoming Period $P$ | Upstream Cycle $c_{k-2}$ (hex / bits) | Upstream Weight $\mathrm{wt}(c_{k-2})$ | Parity $\Omega$ | Outgoing Period $P(k)$ | Lock-in Step $T_{\text{lock}}(k)$ (Unit Seed) |
| ---: | ---: | ---: | :--- | ---: | :---: | ---: | ---: |
| — | **$k_0 = 0$** | $1$ | — | — | — | **$1$** ($2^0$) | $0$ |
| $k-1 = 2$ | **$k_1 = 3$** | $1$ | `1` | $1$ | **Odd ($1$)** | **$2$** ($2^1$) | $2$ |
| $k-1 = 7$ | **$k_2 = 8$** | $2$ | `10` | $1$ | **Odd ($1$)** | **$4$** ($2^2$) | $2$ |
| $k-1 = 28$ | **$k_3 = 29$** | $4$ | `1011` | $3$ | **Odd ($1$)** | **$8$** ($2^3$) | $30$ |
| $k-1 = 399$ | **$k_4 = 400$** | $8$ | `01100001` (`0x86`) | $3$ | **Odd ($1$)** | **$16$** ($2^4$) | $498$ |
| **$k-1 = 53\,207$** | **$k = 53\,208$** | $16$ | `0xca30` (`1100101000110000`) | **$6$** | **Even ($0$)** | **Stays $16$** (Split!) | — |

### 3.2 What Happens at $k - 1 = 53\,207$? (Theorem 3: The $P = 16$ Even-Parity Binary Tree)

At $k - 1 = 53\,207$ (after $\Delta k = 52\,808$ period-16 diagonals beyond $k_4 = 400$), the $P = 16$ plateau hits its first all-zero barrier $c_{53207} = \mathbf{0}_{16}$, **and for the first time in Rule 30, the upstream cycle $c_{53206} = \mathtt{0xca30}$ has even weight ($\mathrm{wt} = 6$, $\Omega = 0$)**!

Why does this matter?
- Because $\Omega = 0$, the period at $k = 53\,208$ **does not double** ($P = 16$).
- Worse, the two choices of the transient bit $\sigma \in \{0, 1\}$ at $k = 53\,208$ produce two bitwise-complementary period-16 cycles:
  $$c_{53208}^{(0)} \;=\; \mathtt{0x8c20} \quad (\mathrm{wt} = 3), \qquad c_{53208}^{(1)} \;=\; \overline{c_{53208}^{(0)}} \;=\; \mathtt{0x73df} \quad (\mathrm{wt} = 13).$$
  Because $\mathrm{wt}\bigl(c_{53208}^{(0)}\bigr) = 3 \ne 13 = \mathrm{wt}\bigl(c_{53208}^{(1)}\bigr)$, **$c_{53208}^{(0)}$ and $c_{53208}^{(1)}$ cannot be related by any time shift!** Here, 1 bit of finite-time transient information from the chaotic interior genuinely splits the future evolution of the periodic crystal into two non-conjugate trajectories.

Does either branch get trapped in a spatial cycle of period-16 diagonals (which would halt period doubling at $P = 16$ forever), or do both branches eventually hit an odd-parity barrier and double to $P = 32$?

> **Theorem 3 (Finite Acyclic Binary Tree of $P = 16 \to 32$ Dislocations).**  
> Starting from the canonical period-16 state $(c_{399}, c_{400}) = (\mathbf{0}_{16}, \mathtt{0x27d8})$ at $k_4 = 400$, the state-space trajectory under $\mathcal{T}_{16}$ with arbitrary binary branching $\sigma \in \{0, 1\}$ at each even-parity barrier ($c_{k-1} = \mathbf{0}_{16}, \Omega = 0$) forms a **finite, cycle-free full binary tree** with:
> - **15 internal even-parity split nodes** (beginning at root $k = 53\,208$), and
> - **16 odd-parity leaf nodes** ($\Omega = 1$), at every one of which the period doubles from $P = 16$ to $P = 32$.
>
> Consequently, **for every sequence of transient bits $\sigma \in \{0, 1\}^{*}$**, the $P = 16$ plateau is guaranteed to terminate and double to $P = 32$ at one of the 16 leaf diagonals $k_5 \in [87\,867,\, 894\,235]$:

| Binary Branch Path $\sigma \in \{0,1\}^*$ | Leaf Dislocation Diagonal $k_5$ ($P = 16 \to 32$) | Total $P = 16$ Plateau Width ($k_5 - 400$) | Upstream Odd Weight $\mathrm{wt}(c_{k_5-2})$ |
| :--- | ---: | ---: | ---: |
| `01` (shortest branch) | **`87,867`** | `87,467` | $5$ |
| `110` | `183,184` | `182,784` | $11$ |
| `11100` | `196,189` | `195,789` | $11$ |
| `00` | `229,338` | `228,938` | $7$ |
| `11111` | `253,537` | `253,137` | $11$ |
| `11101` | `271,596` | `271,196` | $5$ |
| `10` | `291,257` | `290,857` | $5$ |
| `11110101` | `527,724` | `527,324` | $9$ |
| `111100` | `551,910` | `551,510` | $7$ |
| `11110111` | `555,813` | `555,413` | $7$ |
| `1111011011` | `575,211` | `574,811` | $9$ |
| `111101000` | `634,886` | `634,486` | $13$ |
| `111101001` | `645,655` | `645,255` | $7$ |
| `111101100` | `667,052` | `666,652` | $9$ |
| `11110110100` | `770,532` | `770,132` | $11$ |
| `11110110101` (longest branch) | **`894,235`** | `893,835` | $11$ |

*Computational Proof.* Verified exhaustively in `1.2 s` by `verify_koggestone_p16_barrier_tree()` in [`1984-wolfram-conus-textile-rule30.py`](../1984-wolfram-conus-textile-rule30.py). $\square$

---

## 4. Physical Lock-In Velocity & Connection to Wolfram's Rule 30 Prize #1

Why doesn't the periodic left crystal eventually invade the center column $i = 0$ and force $a_{t,0}$ to become periodic (Rule 30 Prize Problem #1)?

1. **Condition for Center-Column Capture:** Since $i = -t + k$, the center column $i = 0$ corresponds to the diagonal $k(t) = t$. For the periodic crystal to reach $i = 0$, the lock-in time $T_{\text{lock}}(k)$ would have to satisfy $\limsup_{k\to\infty} T_{\text{lock}}(k)/k \le 1$.
2. **Empirical & Structural Delay Slope ($\mathrm{d}T_{\text{lock}}/\mathrm{d}k > 1$):** Across the 371-diagonal period-8 plateau ($k \in [30, 400]$), least-squares linear regression gives:
   $$T_{\text{lock}}(k) \;\approx\; 1.2441\,k \;+\; 7.89 \qquad \Longrightarrow \qquad v_{\text{diag}} \;=\; \frac{\mathrm{d}k_{\text{lock}}}{\mathrm{d}t} \;=\; \frac{1}{1.2441} \;=\; 0.8038 \;<\; 1\text{ diagonals/step}.$$
   In physical space $i = k - t$, the phase boundary between the periodic crystal and the chaotic right interior therefore recedes **leftward** at velocity:
   $$v_{\text{lock}} \;=\; v_{\text{diag}} - 1 \;=\; -0.1962\text{ cells/step}.$$
   Why is $\mathrm{d}T_{\text{lock}}/\mathrm{d}k > 1$? Whenever $c_{k-1}(t) = 0$ across a run of $r$ consecutive time steps (where $p_0(t) = 1$), $d_k$ cannot synchronize immediately at step $T_{\text{lock}}(k-1) + 1$; it must wait an additional synchronization delay $\Delta T \ge 1$ until the next `1` in $c_{k-1}$ arrives to clear the carry-propagate chain.

---

## 5. Open Mathematical Problems Suggested by This Formulation

We highlight four concrete open problems suitable for mathematical collaboration, each tracked in a dedicated GitHub issue:

1. **Problem A — General Acyclicity of the Kogge–Stone Graph $\mathcal{G}_{2^m}$ ([Issue #14](https://github.com/fractastical/morpholib/issues/14)):**  
   On the state space $\mathcal{S}_P = \mathbb{F}_2^P \times \mathbb{F}_2^P$ ($P = 2^m$), does the deterministic Kogge–Stone map $(u, v) \mapsto (v, \mathcal{T}_P(u, v))$ (for $v \ne \mathbf{0}_P$) contain **any** directed cycles that avoid the absorbing barrier $v = \mathbf{0}_P$ and are reachable from the period-doubled entry states $(0, c_{k_m})$? Theorem 3 proves that for $m = 0, 1, 2, 3, 4$ ($P \le 16$), every reachable trajectory hits $v = \mathbf{0}_P$ and exits to $2P = 32$ in at most $11$ even-parity splits. Extending this non-cycling proof to $P = 32$ and general $m$ would provide a purely finite-algebraic proof of unbounded period doubling on the left cone without requiring infinite-lattice ergodic assumptions.
2. **Problem B — Lower Bound on Mean Synchronization Delay $\langle \Delta T_{\text{lock}} \rangle > 1$ ([Issue #15](https://github.com/fractastical/morpholib/issues/15)):**  
   By Theorem 1, the extra lock-in delay $\Delta T_{\text{lock}}(k) = T_{\text{lock}}(k) - T_{\text{lock}}(k-1)$ at diagonal $k$ is governed by the waiting time to the next `1` in $c_{k-1}(t)$ (i.e., the run-length distribution of `1`s in the propagate mask $p_0 = \overline{c_{k-1}}$). Proving rigorously that $\liminf_{K\to\infty} \frac{1}{K}\sum_{k=1}^K \Delta T_{\text{lock}}(k) > 1$ would prove unconditionally that the periodic left phase boundary recedes leftward ($v_{\text{lock}} < 0$) and can never capture the center column $i = 0$.
3. **Problem C — Exact Selection of the True Unit-Seed Path in the $P=16$ Binary Tree ([Issue #16](https://github.com/fractastical/morpholib/issues/16)):**  
   By Theorem 3, the unit-seed Rule 30 light-cone must follow one of the 16 root-to-leaf paths in Table 3.2 (entering $P = 32$ at one of the 16 values $k_5 \in [87\,867,\, 894\,235]$). Determining which branch bit $\sigma_1 \in \{0, 1\}$ the single-cell seed selects at the first even-parity split $k = 53\,208$ requires evaluating $d_{53208}(t)$ just past $t = T_{\text{lock}}(53208) \approx 1.244 \times 53\,208 \approx 66\,200$ steps — a concrete finite bit-sliced computation ($\approx 2.2 \times 10^9$ cell updates) that pins down the exact path in our 16-leaf tree.
4. **Problem D — Exact 2D Tent Slope Asymmetry & Cross-Model 1D Defect-Front Universality ([Issue #17](https://github.com/fractastical/morpholib/issues/17)):**  
   Quantify the left-slope vs. right-slope erosion asymmetry of white triangles in Rule 30 (`100 -> 1` deterministic left slope vs. `001 -> 1` exterior-state-modulated right slope) against the symmetric excitable wave-branching slopes of the Meinhardt–Klingler (1987) PDE and cylindrical UV unwrappings of *Conus textile* shells.

---

## 6. Cross-Model Summary Table: Four Universality Classes of 1D Morphogenetic Defect Fronts

| Regime & Script | State Space on 1D Lattice/Domain | Governing Evolution / Front Equation | Left / Right Front Velocities | Topological / Biological Invariant |
| :--- | :--- | :--- | :--- | :--- |
| **1. Wolfram Rule 30 Diagonal & Damage Fronts** (`1984-wolfram-conus-textile-rule30.py`) | $a_{t,i} \in \mathbb{F}_2$, diagonal $d_k(t) = a_{t,-t+k}$ | $d_k = g_0 \oplus p_0\,d_k^{(-1)}$ (Kogge–Stone $\mathcal{T}_P$); $\partial f/\partial l = 1, \partial f/\partial r = \bar{c}$ | $v_{\text{lock}} = -0.1962$, $v_L = -0.2604 \pm 0.1024$, $v_R = +1.0000$ cells/step | Dislocations at $k \in \{0,3,8,29,400\}$; 15-split/16-leaf tree on $P=16$ ($k_5 \in [87\,867, 894\,235]$) |
| **2. Meinhardt–Klingler Mantle Lesion Scar** (`1984-wolfram-conus-textile-rule30.py`) | $(a, s, b) \in \mathbb{R}_{\ge 0}^3$ on $x \in [0, L]$ | $\partial_t a = \frac{\rho s(a^2/(1+\kappa a^2)+\rho_0)}{b} - \mu_a a + D_a \partial_x^2 a$ | $v_{\text{PDE}} = \pm 0.2285$ cells/step | Self-healing V-shaped dorsal tent scar via substrate accumulation in lesion gap |
| **3. Autonomous Cell-Sorting Frozen Defect** (`sorting-morphogenesis/`) | Permutation $v_{t,i} \in \{1\dots N\}$ with pinned defect $i_{\text{def}} = 18$ | Local Bubble ($|j-i|=1$) vs. Non-Local Selection (`ideal_position` probe) | $v_{\text{tunnel}}^{\text{Bubble}} = 0$, $v_{\text{tunnel}}^{\text{Sel}} = 1.7889$ swaps/step | Bubble pinned (`0` cross-swaps, `inv=205`) vs. Selection defect tunneling (`161` cross-swaps, `inv=0`) |
| **4. Bioelectric Antagonistic Domain Wall** (`2021-Cervera–Levin–Mafe.py`) | $(m_1, m_2) \in \mathbb{R}_{\ge 0}^2$ on $x \in [0, 1]$ | $\partial_t m_i = D_i \partial_x^2 m_i - k_i m_i - k_{12}m_1 m_2$ ($D_2 = f_2 D_1$) | Overdamped relaxation $x^*(0)=0.500 \to x^*_{\infty}=0.590$ | Crosses posterior pharynx boundary $x^*_b = 0.583$ at $f_2 \le 0.420 \to$ bipolar two-headed worm (`#2`) |

---

## 7. One-Command Reproducibility for Reviewers

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run all algebraic, Kogge-Stone barrier-tree, and morphometric self-tests (~2.5 s)
MPLCONFIGDIR="$PWD/.mplcache" python3 1984-wolfram-conus-textile-rule30.py --self-test-only
```
