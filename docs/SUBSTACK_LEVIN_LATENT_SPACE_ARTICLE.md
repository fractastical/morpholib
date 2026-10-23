# Between Galvani’s Frogs and Plato’s Ghosts: Testing Michael Levin’s "Latent Space" Claims with Code, Math, and 10,303 Fish

*In a sea of internet debates between "Platonic Ingression" and reductionist blog critics, an internet philosopher who can code doesn’t have to guess—because every single core claim can be tested with tools at our fingertips.*

---

For some time I have been participating actively in documenting the scientific potential of bioelectricity (which goes back to Galvani’s frog experiments) and the various fascinating, though usually hypothetical, metaphysical concepts that often travel side by side with it ([bioelectricity.net](https://bioelectricity.net)). As a historian of math, it doesn’t bother me that sci-fi concepts (such as *psi*) and real science can develop side by side—historically, Kepler’s astrophysics and Newton’s alchemy grew out of the same soil—but I’m equally aware that religious effects can take over and obliterate free critical thinking (including in Buddhist traditions that bind one to absolute obedience to a guru).

When I was working at the California Institute of Machine Consciousness in early 2026, my then-employer Joscha Bach would often elatedly insert concepts of *morphogenesis* into conversations. This term means, roughly, how things get to be their current shape, and tends to be a pretty mathy set of explanations.

When board members of the organization asked me to explain it, I did so as follows:

> *"As someone with a computational background leaning into bio, Michael Levin has taken the scalpel down to the smallest component parts, worked on new imaging techniques, and tried to understand how cells take their form, which is related to newly observable phenomena via the electrical communication that happens between cells. This then is used to explain various other previously unexplainable phenomena."*

This satisfied her, but she said that until I explained it that way, most of the Mike Levin orbit concepts sounded instead impermeable or much like "woo science."

In my case, I was fortunately aided by Mike Levin himself, who gave me a list of articles to read in which order to get started on learning the space, to which I added a massive pile of his papers, which I synthesized with a knowledge graph ([Project Delphi](https://github.com/fractastical/morpholib)), which then became an independent product (**MorphoEval**) that now is able to autonomously build out the math of biology.

![](https://fractastical.github.io/morpholib/media/all_forms_montage.gif)
*Fifteen flagship animated models of biological form (1917–2025) in our open-source [Infinite Morphospace](https://github.com/fractastical/morpholib) repository—audited by independent AI referees from competing labs and formally checked in the Lean 4 theorem prover.*

Although none of this required "Platonism," it was partially the concept of validating the concepts of what mathematicians called the **"morphospace"** and Levin calls the **"latent Platonic space"** that motivated the creation of these tools—and which now allows agents to explore math more directly and which has already led to notable discoveries.

As someone who needs sizable amounts of funding to conduct his experiments (including 400+ papers in the last decade out of his lab), he is probably more sensitive than I am to the language that makes funders want to fund. However, his more recent set of terms choosing "Platonism" and other terms like "ingress" (*Ingressing Minds*, 2025/2026) have set off a large set of internet debates, in which I inadvertently got involved when *Sweet Rationalism* (Nathan Sweet) suggested my method for validation may be more comprehensive and better than Levin’s own.

Although we have briefly discussed these things in person, I do not know, at present, all of the experiments Mike Levin has underway. What I know is mostly limited to the core arguments he brings in his most recent paper, which are the **D’Arcy Thompson-inspired version of the morphology of math** and the **Zhang, Goldstein & Levin (2025) paper regarding emergent phenomena in Bubble, Insertion, and Selection sort** (alongside the classic bioelectric flatworm, Picasso tadpole, and Xenobot models).

In both cases, **precise language of what is claimed is what is most important**, and, in a sea of definitions, a place where an internet philosopher who can code might be most useful. It also turns out that despite a bunch of folks who are "pro" and "against" on the internet, **all of the main claims can be validated with tools that everyone has at their fingertips.**

Let’s go through them claim by claim.

---

## Claim 1: "No Error-Handling Code Was Added" to Self-Sorting Cells (Zhang, Goldstein & Levin 2025)

In *Adaptive Behavior* (2025) and Section 6.3 / Figure 14 of *Ingressing Minds* (2026), Taining Zhang, Adam Goldstein, and Michael Levin asked a clever question: **What if there is no central controller (no "gym teacher") sorting an array of numbers from 1 to 100?** What if each number is an independent thread (an autonomous "cell") that can only look at its neighbors and decide on its own whether to swap places?

They gave the cells three classical computer-science "personalities" (**algotypes**):
* **Bubble cells (B):** Look at your immediate left or right neighbor (`p − 1` or `p + 1`). If you are out of order, swap!
* **Insertion cells (I):** Look at everyone to your left (`0 .. p − 1`). Only step left with `p − 1` if everyone to your left is already in order.
* **Selection cells (S):** Keep a mental finger (`ideal_position` / `target` pointer) starting at slot `0`, then `1`, then `2`... If your number is smaller than whoever is standing at `target`, jump straight there and swap with them; otherwise move your finger one slot to the right.

![](https://fractastical.github.io/morpholib/media/zhang_13yo_fig1_three_personalities.png)

Before testing the philosophical claims, we ported their exact GitHub commit ([`Zhangtaining/cell_research@1fd2bd59`](https://github.com/Zhangtaining/cell_research/tree/1fd2bd5921c1f6b423a71f691d5189106a8a1020)) into **four independent languages: Python 3, C++20, JavaScript (Node.js), and the Lean 4 theorem prover** (`0 sorry`).

First, credit where credit is due: **all 14 of the empirical curves published in Zhang, Goldstein & Levin (2025) replicate 100%**. Across Python, C++20, and Node.js, our ports produce bit-for-bit identical space-time kymographs (`0` cell mismatches) and match every benchmark in the paper.

![](https://fractastical.github.io/morpholib/media/zhang_sorting_morphogenesis.gif)

![](https://fractastical.github.io/morpholib/media/zhang_multilang_port_verification.png)

Now look at the first philosophical claim Levin makes about this code in *Ingressing Minds* (Section 6.3, p. 22) when a few cells in the array are "frozen" (broken and refusing to move):

> *"The critical thing is that we did not add steps to the algorithm to check whether a swap had been completed, or to do specific things if an operation failed."*

**What We Found When We Audited the Code (Panel A below):**
For **Bubble** sort, Levin’s statement is 100% true (while **Selection** sort advances its `target` pointer past frozen cells in `SelectionSortCell.cpython-310.pyc`).
**However, for Insertion Sort (`InsertionSortCell.py`, lines 68–82), the claim is false.**

Remember Insertion sort’s rule: a cell at position `p` refuses to move left unless the entire prefix `0 .. p − 1` to its left is already sorted. If a single frozen cell wearing `#99` sits near the front of the line, then the left side is *never* sorted—which under pure Insertion sort would permanently paralyze every cell to the right of `#99`!

When we opened [`InsertionSortCell.py` (lines 68–82)](https://github.com/Zhangtaining/cell_research/blob/1fd2bd5921c1f6b423a71f691d5189106a8a1020/modules/multithread/InsertionSortCell.py#L68-L82), we found **3 lines of explicit broken-cell exception handling**:

```python
for i in range(self.current_position[0]):
    if self.cells[i].status == CellStatus.FREEZE:
        prev = -1   # <-- Explicit exception handler resetting prefix check at frozen cells!
        continue
    if self.cells[i].value < prev:
        return False
    prev = self.cells[i].value
```

When we deleted those 3 lines of code (`without_explicit_freeze_reset` in **Panel A** below, red bars, and formally proved in Lean 4 Theorem 2), Insertion sort’s final error (`n = 60`, `f = 3` frozen cells) **jumped 4.5× higher (from 4.90 to 21.95)** as all cells behind the defect froze!

![](https://fractastical.github.io/morpholib/media/zhang_levin_platonic_claims_audit.png)

* **Verdict on Claim 1 — Partially Refuted:** Bubble sort genuinely gets defect tolerance for free from decentralized execution; Selection advances its target pointer past frozen cells; and Insertion sort relies on an explicit 3-line `if status == FREEZE: prev = -1` exception handler.

---

## Claim 2: "Delayed Gratification" — Do Sorting Cells Really Go Against Their Gradient to Navigate Around a Barrier?

In *Ingressing Minds* (Section 6.3 & Figure 14A–B), Levin compares the sorting array to an agent walking around a physical wall—contrasting it with two magnets separated by a piece of wood, which get stuck forever because they cannot temporarily move *against* their magnetic gradient. Because the paper’s sorting error score (`monotonicity_error`, `E(v)`) temporarily ticks *upward* before dropping lower (`Delayed Gratification > 0`), Levin argues the algorithm is *"temporarily going against its gradient to recoup greater gains later."*

**What We Found (`Panel B` above & Illustrated Guide 2 below):**
We tested two mathematical properties of the trajectories—and proved both in Lean 4:

1. **Zero Bubble or Insertion cells ever cross an immovable barrier:**
   In Bubble and Insertion sort, cells only swap with immediate neighbors `(p, p + 1)`. If an immovable frozen cell sits at index `d = 20`, **literally 0 cells from the left of spot 20 ever cross to the right of spot 20** (Lean 4 Theorem 1: `bubble_adjacent_swap_preserves_left_partition`). The left half and right half sort in complete isolation! Selection sort only bypasses spot 20 because its `target` pointer is given global array access to teleport across indices.

2. **Why did the error score go up? The Optical Illusion of the Two Rulers:**
   Why did `monotonicity_error` `E(v)` tick uphill 47 times in **Panel B** (blue jagged curve)? Because there are **two different mathematical rulers** for measuring how scrambled a line of numbers is:
   * **Ruler 1 — Total Out-of-Order Pairs `I(v)` (The True Energy):** Count *every* pair of cells `(i < j)` in the whole line where a larger number sits to the left of a smaller number (the Coxeter word length on the symmetric group `S_n`).
   * **Ruler 2 — Neighbor Drops `E(v)` (The Paper’s Ruler):** Only count how many times a cell is larger than the cell *immediately* to its right (`v_i > v_{i+1}`).

   Look at a tiny line of 4 cells: **`[2, 4, 1, 3]`**:
   * Under **Ruler 1 (`I`)**, there are **3** out-of-order pairs: `(2,1)`, `(4,1)`, and `(4,3)`.
   * Under **Ruler 2 (`E`)**, there is only **1** neighbor drop: `4 → 1` (since `2 < 4` and `1 < 3`).
   * Now let Bubble sort swap the out-of-order pair `4 ↔ 1` to get **`[2, 1, 4, 3]`**.
   * What happened to **Ruler 1 (`I`)**? It went **strictly DOWN from `3 → 2`**! In fact, **every single Bubble swap in history reduces `I(v)` by exactly `−1` (`0` uphill steps out of 192 swaps—the yellow straight line in Panel B!)**.
   * Why did **Ruler 2 (`E`)** go **UP from `1 → 2`** (`2 > 1` and `4 > 3`)? Because swapping `4` and `1` broke one tall cliff (`4 → 1`) into two smaller steps (`2 → 1` and `4 → 3`)! This happens **even when there are zero frozen cells (`f = 0`, where `DG = 0.38–0.43`)**.

![](https://fractastical.github.io/morpholib/media/zhang_13yo_fig2_two_rulers_illusion.png)

### The 13-Year-Old’s Guide to "Causal Emergence 2.0": Why Zooming Out to the Right Scoreboard Beats Watching Individual Pixels!

Wait—how can we *prove* mathematically that **Ruler 1 (`I(v)`)** is the "real" law of the sorting line and **Ruler 2 (`E(v)`)** is a blurry illusion? That is where **Erik Hoel’s *Causal Emergence 2.0*** (*Patterns* 2026, upgrading his classic 2013 *PNAS* paper) comes in.

Here is the 13-year-old version of how Causal Emergence works (**Illustrated Guide 5** below):

1. **Level 1 — The "Zoomed-In" Micro-View (Watching Exact Lines of Kids = A Noisy Coin Flip!):**
   Suppose the kids are standing in the line **`[2, 4, 1, 3]`** (**Box 1** below). Because kids wake up at random to look at their neighbors, we don't know which kid will move first:
   * **50% chance:** Kid `#4` swaps with `#1` → the next line is **`[2, 1, 4, 3]`**.
   * **50% chance:** Kid `#4` swaps with `#3` → the next line is **`[2, 4, 3, 1]`**.
   If you are trying to predict the *exact* line of kids at the next step, it’s a coin toss! Across all 720 possible ways to line up 6 kids (`S_6`), the zoomed-in micro-view is only **86.9% predictable** (its Causal Score is **`CP = 0.850`**).

2. **Level 2 — Zooming Out to a "Good" Macro-Scoreboard (Ruler 1: Total Out-of-Order Pairs `I(v)` = 100% Predictable!):**
   Now stop squinting at which exact kid moved, and just look at **Ruler 1 (`I`)** on the scoreboard (**Box 2** below):
   * In **`[2, 4, 1, 3]`**, we start at **`I = 3`**.
   * If the kids branch to Line A (`[2, 1, 4, 3]`), what is `I`? **`I = 2`**!
   * If the kids branch to Line B (`[2, 4, 3, 1]`), what is `I`? **Also `I = 2`**!
   * **Every single branch lands on the exact same score (`3 → 2 → 1 → 0`)!**
   By zooming out to Ruler 1, the coin-flip noise completely disappears and predictability jumps from **86.9% to 100.0%** (`CP` rises from `0.850 → 0.969`, giving a **Positive Causal Emergence 2.0 boost of `ΔCP = +0.1191 > 0`**). In plain English: **the zoomed-out scoreboard is MORE deterministic than the zoomed-in pixels!**

3. **Level 3 — Zooming Out to a "Bad" Macro-Scoreboard (Ruler 2: Neighbor Drops `E(v)` = A Blurry Mess!):**
   What happens if we zoom out using the paper’s ruler, **Ruler 2 (`E`)** (**Box 3** below)?
   Ruler 2 puts `[2, 4, 1, 3]`, `[1, 4, 2, 3]`, and `[1, 3, 2, 4]` into the *exact same bucket* (`E = 1 neighbor drop`). But from that bucket, one swap can make `E` **go UP to `E = 2`** (`[2, 1, 4, 3]`), **stay flat at `E = 1`** (`[1, 2, 4, 3]`), or **drop to `E = 0`** (`[1, 2, 3, 4]`)!
   Instead of cleaning up the coin-flip noise, Ruler 2 scrambles the future: predictability crashes from **86.9% down to 72.1%**, giving **Negative Causal Emergence (`ΔCP = −0.2525 < 0`)**.

4. **Why Erik Hoel Needed "Causal Emergence 2.0" (2026) to Fix "Causal Emergence 1.0" (2013) (`Box 4` below):**
   Here is a cool math bonus: in Hoel’s original 2013 formula (**CE 1.0**), a scoreboard got points for *both* accuracy AND having lots of possible numbers. Because Ruler 1 compresses `720` different lines of kids into just `16` scores (`I = 0..15`), the old 2013 formula slapped Ruler 1 with a huge **`−4.66 bit` "small-scoreboard penalty"** that hid its **`+0.48 bit` accuracy gain** (`Total ΔEI = −4.19 bits`, making CE 1.0 miss the law!). Hoel’s brand-new 2026 formula (**CE 2.0**) measures pure cause-and-effect reliability per step (`ΔCP = +0.1191 > 0`), immediately spotting that Ruler 1 is a 100% deterministic law!

![](https://fractastical.github.io/morpholib/media/zhang_13yo_fig5_causal_emergence_explained.png)

![](https://fractastical.github.io/morpholib/media/zhang_hoel_quantifying_emergence_ce2.png)

* **Verdict on Claim 2 — Optical Illusion of the Ruler:** The sorting cells never move against their dynamical gradient. They slide 100% deterministically downhill on their true Lyapunov energy `I(v)` (`ΔCP = +0.1191 > 0`) at every single step, while the observer watches them through a blurry, negative-emergence ruler `E(v)` (`ΔCP = −0.2525 < 0`).

---

## Claim 3: "Spontaneous Algotype Clustering" — Why Levin’s Phenomenon Is 100% Real, and the CS Blog Critics Were Wrong Twice!

Here is where our audit caught the *critics* making an even bigger mathematical mistake than the dualists.

In Figure 8 of Zhang et al. (and Figure 14C of *Ingressing Minds*), when you mix 50% **Bubble (`B`)** cells and 50% **Selection (`S`)** cells in the same line, cells of the same personality spontaneously group together (`aggregation > 0.50`)—even though **zero lines of code ever inspect `neighbor.algotype`**.

Levin called this an unprogrammed "side-quest" from Platonic space.
In response, computer-science bloggers (including Nathan Sweet at *Sweet Rationalism*) argued that this clustering is nothing more than a 1960s textbook property: **duplicate-key sort stability** (when multiple cells have the same number, Bubble sort is "stable" and doesn't swap equal keys).

When we ran the ablations (**Panel C** of both figures above, and Illustrated Guide 3 below), we proved that **the CS blog critics were wrong twice—both on unique numbers AND on duplicate numbers:**

1. **Critics’ Mistake #1 — Paper Fig. 8A Has ZERO Duplicate Numbers!**
   In Paper Figure 8A (**Panel C**, blue curve), every single cell has a **strictly unique number from `1` to `n` (`0` duplicate keys)**. When there are no duplicate keys, "duplicate-key sort stability" is mathematically impossible! Yet `B + S` aggregation still rises from `0.50` to a peak of **`0.624–0.640`** mid-sort before relaxing to `0.501` at the end.
   **Why?** Because **Selection cells move like sports cars and Bubble cells move like pedestrians!** Selection cells use their `target` pointer to jump small numbers straight to the left front of the line (`0, 1, 2...`) in `O(1)` swaps, shoving a contiguous wave of slow `O(n²)` Bubble cells into the right half of the line.

2. **Critics’ Mistake #2 — The `C = 2` "Anti-Clustering" Smoking Gun on Duplicate Numbers!**
   What about Paper Figure 8D, where there *are* duplicate numbers (`C` copies of each key)? Think carefully: at the start of the sort, the `B` and `S` labels are shuffled randomly (`aggregation = 0.50`). If textbook "sort stability" simply preserved the initial relative order of equal keys, then **final aggregation at the end of the sort would stay at `0.50` for every `C`!**
   Instead, when we swept `C = 1, 2, 4, 8, 10, 20, 40` copies per number, we discovered that because Selection cells scan `target` from `0` upward, **66.2% (at `C = 2`) to 75.0% (at `C = 10`) of Selection cells pack into the LEFT half of every number’s group (`S...S | B...B`)**:
   * When there are **`C = 2` copies of each number** (one `S` and one `B` wearing `#1`, one `S` and one `B` wearing `#2`, etc.), putting `S` on the left and `B` on the right of each pair creates the alternating pattern **`S B | S B | S B`**—which drives clustering **below random to `0.477` (Anti-Clustering!)**!
   * When there are **`C = 10` copies of each number** (`Fig. 8D`), putting 5 `S` cells on the left and 5 `B` cells on the right of each group creates **`SSSSS | BBBBB`**, driving permanent clustering **up to `0.659`**! (And when we change `<` to `<=` on ties in **Panel C**, red dashed curve, the permanent clustering collapses back to `0.497–0.514`.)

![](https://fractastical.github.io/morpholib/media/zhang_13yo_fig3_sports_cars_vs_pedestrians.png)

* **Verdict on Claim 3 — Empirically Vindicated (via Kinematic Phase Separation):** The clustering is 100% real and refutes the blog critics’ "textbook stability" dismissal—yet it is also 100% produced and predicted by the kinematic transport equations of the two algorithms without needing non-physical ghosts. ([Try the live `C = 2` simulator in your browser here!](https://fractastical.github.io/morpholib/bioelectricity/sorting-morphogenesis.html?tab=sim&scenario=c2_anticluster&autoplay=1))

---

## Claim 4: D’Arcy Thompson’s (1917) Generative Morphospace Across 10,303 Fish Species

The second major pillar Levin invokes for his latent space is **D’Arcy Wentworth Thompson’s (1917)** famous coordinate transformations from *On Growth and Form*: the idea that nature doesn't redesign every bone from scratch when evolving a parrotfish (*Scarus*) into an angelfish (*Pomacanthus*) or a pufferfish (*Diodon*) into an ocean sunfish (*Mola mola*), but instead turns a few global **"growth knobs"** that bend the 2D coordinate grid of the body like a sheet of rubber.

![](https://fractastical.github.io/morpholib/guide_step1_1917_rubber_sheet.png)

For the last 30 years, evolutionary biologists tried to test Thompson’s idea on computer landmark databases—and concluded he was mostly wrong across fish families. **Why did their tests fail?**

When we audited the morphometrics literature across **three independent datasets scaling up to 10,303 fish species (`440` families)**, we uncovered two methodological traps that had hidden Thompson’s law for a century:

1. **Trap #1 — The "17-Dot Head Seesaw":**
   Standard fish databases (Claverie & Wainwright 2014; Friedman et al. 2019) don't trace the whole outline of the fish—they measure **17 dots on the skeleton**, with **9 of those 17 dots (53%) packed inside the head and jaws** and **0% on the outer dorsal, anal, or tail fins**. Because jaws evolve rapidly for diet while the body evolves for swimming, fitting a global rubber sheet to those 17 dots lets the 9 head dots pull the back of the sheet the wrong way—making the warped fish silhouette look **worse than no warp at all** (`40.6% → 56.2%` mismatch on *Scarus* → *Pomacanthus*)!

![](https://fractastical.github.io/morpholib/guide_step2_17dot_head_trap.png)

![](https://fractastical.github.io/morpholib/media/thompson_1917_silhouette_paradox_summary.gif)

2. **Trap #2 — The "Plain Box Stretch" Illusion vs. True Thompson Curves:**
   When we switch to a **600-point whole-body outline**, if you only ask a computer to overlap two *unlabeled black shadows*, a trivial **Rectangular Box Stretch** (stretching height vs. width with *zero* slant or curve) can "cheat" by stretching the middle of a pufferfish’s belly straight up to cover the back fins of an ocean sunfish!
   **Where Thompson’s curved and slanted grids (`70° Oblique Shear`, `Coaxial Circles`, `Bilinear Fan`, `Radial Flare`) are genuinely required** is when you force the grid to match **true homologous perimeter anatomy** (dorsal fin base & tip to dorsal fin base & tip, anal fin to anal fin, without the 9 internal skull dots): there, a plain Box Stretch fails (**`2.4%`** match on the swept-back fin tips of the angelfish), and **Thompson’s curved grids jump by `+13%` to `+48%` (`15×` better)** while staying **100% smooth and un-folded on `6 out of 6` pairs** (whereas standard 600-point Thin-Plate Splines fold and crumple the interior tissue grid on `6 out of 6` pairs!).

![](https://fractastical.github.io/morpholib/guide_step3_box_cheat_vs_curves.png)

3. **The Two Rules of Fish Evolution Across 3 Datasets (`791 → 6,614 → 10,303` Species) & Our Blind Test on 5,180 Unseen Species:**
   Testing three independent datasets (**791**, **6,614**, and **10,303** species across `440` families and `987` genera) revealed that fish evolution obeys two orthogonal rules at once:
   * **Rule 1 — The Outside Rubber Sheet ("How the Fish Swims"):** Over 150 million years across 440 families, whole-body evolution follows D’Arcy Thompson’s 8-knob coordinate highway with **93.5% to 98.9% alignment (`θ = 20.7° to 8.5°`)**, capturing **92.0% of the theoretical 8-parameter Relative-Warp variance ceiling** without a single data-trained eigenvector!
   * **Rule 2 — Inside the Head ("What the Fish Eats"):** When closely related sister species live side-by-side on the same coral reef, they share the same swimming body, so they **rapidly rewire their jaws, snouts, and eyes 3.3× to 9.3× faster than their bodies** (`85.4%` of within-genus variance; diet shift `p = 9.2 × 10⁻⁵`) to partition food without fighting over the same meal!
   * **Pre-Registered Blind Test on 5,180 Unseen Species (`4/7` Hypotheses Supported):** Before opening the Smithsonian’s **FishShapes v1** dataset (`5,180` never-seen species), we locked our predictions by SHA-256 hash on the production Nanopub network ([`RAilN4nbRHp3e3ZHqlbIgeL3UGPRV_KtQHtJD3gzVVdqw`](https://registry.knowledgepixels.com/np/RAilN4nbRHp3e3ZHqlbIgeL3UGPRV_KtQHtJD3gzVVdqw)). Within the **~50 Myr modularity horizon** (`n = 1,122` species), both the full 30-D tree predictor (`p = 10⁻¹⁰³`, `r = 0.87`, `92.6%` empirical coverage) and the 7-mode Thompson predictor (`p = 10⁻⁸¹`) crushed the clade consensus—while our pre-registered scoring protocol also honestly caught that on a single noisy calliper ratio, full Gaussian tree conditioning only beats copying the single nearest relative by 4% (`p = 0.061`).

![](https://fractastical.github.io/morpholib/guide_step4_outside_vs_inside_rules.png)

* **Verdict on Claim 4 — Strongly Confirmed:** Across 10,303 species, Thompson’s and Levin’s core thesis—that macroevolution navigates a low-dimensional geometric coordinate manifold while local modules handle rapid ecological competition—is a quantitative law of vertebrate evolution.

---

## Claims 5, 6 & 7: How Bioelectric & Biomechanical Attractors Physically Implement the Latent Space (Planarians, Picasso Tadpoles & Xenobots)

How does a living tissue actually store a target coordinate in that morphospace? In our repository, we also built and audited mathematical reproductions of Levin’s three most famous wet-lab discoveries—testing whether standard biophysical PDEs and ODEs genuinely produce those attractor behaviors (while explicitly documenting reduced-model parameter assumptions in our MorphoEval Pass 3 referee audits):

* **Claim 5 — Bioelectric Target Memory in Two-Headed Planarians (Beane et al. 2011; Pietak & Levin 2016/2017; Cervera, Levin & Mafé 2021):**
  In the lab, blocking gap junctions in a cut planarian flatworm with 8-octanol for 48 hours causes it to regenerate **two heads (`2H0T`)**—and weeks after the drug is washed out, cutting the worm again in plain water *still* produces two heads with zero DNA change.
  In our 1D PDE and 405-cell 2D Voronoi Scharfetter–Gummel electrodiffusion models (`2017-pietak-bioelectric-morphogen-coupling.py`), positively charged serotonin (`5-HT⁺`, `z = +1`) electrophoreses down the resting voltage gradient (`−20 mV` head to `−60 mV` tail) by the exact Boltzmann ratio `c_post / c_ant = exp(40 mV / 25.85 mV) ≈ 4.70`. Crucially, **gap-junction conductance `G_gj` cancels out of the zero-flux equilibrium ratio!** Once transient 8-octanol blockade lets the posterior pole flip to `−20 mV` (`2H0T`), washing out the drug leaves `ΔV = 0` and permanently locks the worm in the two-headed attractor basin.

![](https://fractastical.github.io/morpholib/media/cervera_levin_mafe_gj_block.gif)

* **Claim 6 — Closed-Loop "Picasso Tadpole" Homeostasis (Vandenberg, Adams & Levin 2012; Pezzulo & Levin 2015; Pinet et al. 2019):**
  When tadpole eyes, jaws, and nostrils are surgically scrambled at stage 45, they don't follow hardwired displacement vectors (which would preserve 100% of the scrambling error at metamorphosis). Instead, they move along novel paths with speed proportional to defect distance until they reach the normal froglet face—unless MMP tissue remodeling is blocked by GM6001. Our Lean-verified Procrustes / Variational Free Energy Lyapunov models (`dV/dt ≤ 0`, coupled in Pass 3 to 3D Maxwell viscoelastic cartilage rods and MMP-2/MMP-9 zymography fields) reproduce both normal convergence and GM6001 arrest.

![](https://fractastical.github.io/morpholib/media/xenopus_craniofacial_remap_growth.gif)

* **Claim 7 — Xenobots & Kinematic Self-Replication Without Genetic Change (Kriegman, Blackiston, Levin & Bongard 2020, 2021 *PNAS*):**
  Wild-type frog skin and cardiac cells sculpted into a **C-shaped ("Pac-Man") mouth concavity (`θ ≈ 105°`)** use cilia-driven Stokes flow to sweep loose stem cells into piles that compact into 3–4 generations of daughter Xenobots. Both our 8×8×7 3D voxel mass-spring lattice and our 2D regularized-Stokeslet squirmer fluid simulation confirm that `θ ≈ 105°` is a genuine hydrodynamic capture optimum (`3–4` daughter generations vs. `1` for a sphere and `0` for a flat slab).

![](https://fractastical.github.io/morpholib/media/xenobot_evolutionary_biobots_growth.gif)

---

## The Final Scoreboard (Copy-Paste Summary)

* **Claim 1 — Sorting Array Error Tolerance ("No error-handling code was added"):**
  **PARTIALLY REFUTED.** True for Bubble sort (while Selection advances its target pointer past frozen cells), but **false for Insertion sort**, which contains an explicit 3-line `if status == FREEZE: prev = -1` reset in `InsertionSortCell.py` (L74–76). Deleting those 3 lines spikes Insertion sort’s error **4.5× higher (`4.90 → 21.95`)**.
* **Claim 2 — Sorting "Delayed Gratification" ("Going against the gradient around a barrier"):**
  **OPTICAL ILLUSION OF THE RULER.** Proved in Lean 4 that **`0` cells ever cross an immovable barrier** in Bubble or Insertion sort, and every Bubble swap goes **strictly downhill (`ΔI = −1`, `0` uphill steps)** on the true pairwise inversion potential `I(v)` (**Hoel Causal Emergence 2.0 = `+0.119`**). Apparent uphill ticks happen only on the non-Markovian neighbor-drop ruler `E(v)` (**CE 2.0 = `−0.253`**).
* **Claim 3 — Chimeric Algotype Clustering ("Clustering without friend-sensing code"):**
  **EMPIRICALLY VINDICATED (AND BLOG CRITICS REFUTED TWICE).** (1) Paper Fig. 8A has **strictly unique keys (`0` duplicates)** yet peaks at **`0.624–0.640`** via kinematic transport-speed waves (`O(1)` Selection sports cars vs. `O(n²)` Bubble pedestrians); (2) on duplicate keys, **`C = 2` copies per key produces Anti-Clustering (`0.477 < 0.50`, `S B | S B`)**, proving textbook sort stability is not the cause!
* **Claim 4 — D’Arcy Thompson’s Low-Dimensional Generative Morphospace:**
  **STRONGLY CONFIRMED ACROSS 10,303 SPECIES.** Standard 17-dot tests failed only because 53% of the dots sit in the skull. On homologous perimeter outlines, Thompson’s 8-knob grids win by **`+13%` to `+48%`** (`100%` un-folded) and align with **150 Myr of fish body-plan evolution at `93.5%–98.9%`** (blind test `p = 10⁻¹⁰³` on `1,122` unseen species within 50 Myr).
* **Claims 5–7 — Bioelectric Memory (`2H0T` Planarians), Picasso Tadpoles & Xenobots:**
  **STRONGLY CONFIRMED AS DYNAMICAL ATTRACTORS.** Zero-flux Nernst–Planck Boltzmann electrophoresis (`exp(40/25.85) ≈ 4.70`), closed-loop Lyapunov free-energy descent (`dV/dt ≤ 0`), and `θ = 105°` Stokes squirmer capture turn "latent morphospace" from a philosophical metaphor into testable differential equations.

---

## How We Built and Audited This: Methods, Models, Knowledge-Graph Queries & Code Provenance

How do you keep yourself honest when testing high-profile scientific claims—especially when both the "Platonic Space" camp and the "Nothing New Here" blog critics have strong priors?

Just as we did for our D’Arcy Thompson study, we documented the entire investigation in a **4-Page Illustrated Magazine Edition Methods & Provenance PDF** ([`docs/levin_latent_space_methods_and_provenance.pdf`](https://fractastical.github.io/morpholib/levin_latent_space_methods_and_provenance.pdf) and [`docs/LEVIN_LATENT_SPACE_METHODS.md`](https://github.com/fractastical/morpholib/blob/main/docs/LEVIN_LATENT_SPACE_METHODS.md)) that answers five concrete questions so anyone can understand and re-run our experiments:

![](https://fractastical.github.io/morpholib/media/levin_methods_page_1.png)
*Page 1 of our 4-Page Methods & Provenance Guide: Cross-provider AI referee roles (Google DeepMind, Anthropic, OpenAI + human PI/domain critiques) and all 24 mathematical, algorithmic, and biophysical models fitted across Claims 1–7.*

![](https://fractastical.github.io/morpholib/media/levin_methods_page_2.png)
*Page 2 of our 4-Page Methods & Provenance Guide: Line-by-line upstream code traceability (`[U1]–[U6]` from `Zhangtaining/cell_research@1fd2bd59`), the 3-ablation Platonic Space audit, and Erik Hoel’s Causal Emergence 1.0 & 2.0 on `S_3..S_6`.*

1. **Which AI/LLM Models and Human Reviewers Were Used (3-Company Blinded Referee Architecture):**
   To prevent an AI coding agent from grading its own homework, every simulation script and mathematical proof in our repository is built and audited inside our **12-Step MorphoEval Closed-Loop Protocol** using competing model families:
   * **Google DeepMind (`gemini-4-argon`):** Ported `Zhangtaining/cell_research@1fd2bd59` across **Python 3, C++20, Node.js, and Lean 4** (`0` cell mismatches); built the 3-ablation Platonic Space audit (`audit_platonic_space_claims.py`) and the `S_3..S_6` Markov-chain **Causal Emergence 2.0** analyzer (`quantify_emergence_ce2.py`); and ran the blinded 2nd-assessor pass (`90.8/100` on Thompson) and the 10,303-species Thompson vs. competition synthesis.
   * **Anthropic (`claude-fable-5-1` & `claude-sonnet-5-5-high`):** Conducted primary MorphoEval referee audits across the repository; authored the `pk:evidenceAgainst` remediation items in `2026-10-17-thompson-audit.yaml` (L330–337) that sparked our 600-point outline breakthrough; built the 5,180-species `FishShapes v1` blind preregistration/scoring pipeline; and built & audited the Pass 3 2D/3D PDE and fluid extensions for Planarians (`Slot 081/082`), Picasso Tadpoles (`Slot 085`), and Xenobots (`Slot 083`).
   * **OpenAI (`gpt-5.6-sol`):** Acted as an independent third-company referee on `2021-Cervera–Levin–Mafe.py` (`2026-10-17-cervera-audit-v2.yaml`, scoring `44.9 → 55.6/100`), catching an unused `f3` loop variable, a hard-coded `0.58` pharynx threshold, a bisection clipping bug in `f2_for_shift`, and a wrong PMID (`34418508 → 34411690`), which directly drove us to build the 63-series `PlanformDB 2.5.0` evidence extractor (`models/planform_bioelectric.py`).
   * **Human Principal Investigator & External Critiques:** **Joel Dietz** (PI, Project Delphi) synthesized the initial Levin-lab knowledge graph, rejected 17-dot stick figures (exposing the `9/17` skull-dot seesaw), required the 4-parameter Box-Stretch vs. Thompson Warp deconfounding audit, and initiated the 3-ablation + Hoel CE 2.0 audit after **Nathan Sweet** (*Sweet Rationalism*) critiqued Levin’s sorting paper; **Prof. Peter C. Wainwright** (UC Davis) prompted the 600-point Curve-TPS/EFD benchmark and convergent trajectory tests.

2. **How We Did the Sorting & Causal Emergence Experiments (Claims 1–3):**
   * **Line-by-Line Upstream Code Traceability (`[U1]–[U6]`):** We audited [`Zhangtaining/cell_research@1fd2bd59`](https://github.com/Zhangtaining/cell_research/tree/1fd2bd5921c1f6b423a71f691d5189106a8a1020) and pinned all 6 upstream rules: `[U1]` `BubbleSortCell.py` (L38–74), `[U2]` `InsertionSortCell.py` (`is_enable_to_move` L68–82, containing the explicit `if status == FREEZE: prev = -1` reset at L74–76), `[U3]` `SelectionSortCell.py` (L41–88) & `MultiThreadCell.py` (`swap` L104–118), `[U4]` `SelectionSortCell.cpython-310.pyc` (compiled Python 3.10 bytecode rule advancing `target` past passive frozen cells), `[U5]` `delay_gratification_analysis_for_not_move.py` (L24–89), and `[U6]` `cell_type_aggregation_analysis.py` (L518–538).
   * **Bit-Exact 4-Language Ports:** We implemented those exact rules in **Python 3**, **C++20** (`clang++ -O3`, running all `n = 100` benchmarks in `0.8 s`), and **JavaScript (Node.js)** sharing a 64-bit `SplitMix64` PRNG (`verify_multilang_ports.py`), plus 3 concurrency schedulers (random-async, synchronous maximal matching, and literal OS `threading.Lock` threads).
   * **Exact Markov TPMs & Hoel’s Causal Emergence 2.0:** In `quantify_emergence_ce2.py`, we constructed the exact `N × N` Transition Probability Matrices on the symmetric groups `S_3, S_4, S_5, S_6` (`N = 6, 24, 120, 720` permutations) and computed both Hoel’s (2013 *PNAS*) `ΔEI = ΔI_eff + ΔI_size` and Hoel’s (2026 *Patterns* `7:101472`) intensive Causal Primitive `CP(π) = det(π) − deg(π)` on `I(v)` vs. `E(v)`.

![](https://fractastical.github.io/morpholib/media/levin_methods_page_3.png)
*Page 3 of our 4-Page Methods & Provenance Guide: Machine-checked Lean 4 theorems (`0 sorry`), Knowledge-Graph `pk:evidenceAgainst` remediation loops, and Pass 3 2D Voronoi electrodiffusion & 3D Xenobot Stokeslet squirmer simulations.*

![](https://fractastical.github.io/morpholib/media/levin_methods_page_4.png)
*Page 4 of our 4-Page Methods & Provenance Guide: SHA-256 cryptographic data provenance across all 7 claims, Pass 3 3D Maxwell cartilage & MMP zymography for Picasso tadpoles, and the 12-command reproducibility map.*

3. **How We Did the 10,303-Species D’Arcy Thompson & Blind FishShapes Experiments (Claim 4):**
   * **16 Coordinate, Morphospace & Phylogenetic Models:** We benchmarked 11 2D coordinate laws—including a **4-parameter Orthotropic Rectangular Box Stretch (`x' = ax + t_x, y' = dy + t_y`, zero skew, zero curve)** to deconfound unlabeled shadow matching from true homologous perimeter warping—alongside a baseline-anchored bidirectional Chamfer contour solver (`cKDTree`), `600 × 600` Jacobian bijectivity checks (`det DΦ > 0`), Rohlf’s (1993) Relative Warps under Leave-One-Family-Out CV, Schluter’s (1996) `G` vs. `R` covariance alignment across **791**, **6,614**, and **10,303** teleost species (`440` families, `69,682` congeneric pairs), and 5 phylogenetic GLS process models (where Pagel’s `λ = 0.959` won by `ΔAICc > 2,400`).
   * **Cryptographic Pre-Registration Before Unblinding:** We generated predictions for **10,888 unsampled species** on the Rabosky et al. (2018) time tree, locked their SHA-256 hashes (`dac08493`), and published the signed pre-registration Nanopublication ([`RAilN4nbRHp3e3ZHqlbIgeL3UGPRV_KtQHtJD3gzVVdqw`](https://registry.knowledgepixels.com/np/RAilN4nbRHp3e3ZHqlbIgeL3UGPRV_KtQHtJD3gzVVdqw)) to the production Nanopub registry *before* downloading or opening the Smithsonian’s **FishShapes v1** dataset (`5,180` matched unseen species), followed by the post-unblinding outcome Nanopub ([`RAJancwGOPEkHzBoZXGv_vu_snW0Lff2FktE85Me1P-fI`](https://registry.knowledgepixels.com/np/RAJancwGOPEkHzBoZXGv_vu_snW0Lff2FktE85Me1P-fI)).

4. **How We Did the Bioelectric Planarian, Picasso Tadpole & Xenobot Experiments (Claims 5–7):**
   * **Claim 5 (Planarians, `Slots 081/082` & `Cervera 2021`):** Coupled 1D bistable `V_mem` + Bell-form innexin gap-junction PDEs and a **2D finite-volume planarian body mesh** (`2,896` in-mask cells with slit/netrin axon guidance and `1,500` stochastic neoblasts in `2011-beane-planarian-hk-atpase-anterior.py`) with a **405-cell / 1,114-edge 2D Voronoi Scharfetter–Gummel electrodiffusion network** (`2017-pietak-bioelectric-morphogen-coupling.py`) co-transporting `5-HT⁺ (z = +1)`, `IP₃ (z = −3)`, and `Ca²⁺ (z = +2)` across `n = 256` stochastic Euler–Maruyama blastemas.
   * **Claim 6 (Picasso Tadpoles & Limb VFE, `Slots 084/085`):** Coupled 2D/3D Procrustes relational spring-target attractors (`dV/dt ≤ 0`) and Active-Inference Variational Free Energy gradient flows (`dF/dt ≤ 0`, `2015-pezzulo-bioelectric-target-morphology-memory.py`) to **5 3D Maxwell viscoelastic cartilage rods** (Meckel’s, ceratohyal, palatoquadrate) and spatially resolved **MMP-2/MMP-9 reaction–diffusion zymography fields** (`2019-bischof-xenopus-craniofacial-remap.py`).
   * **Claim 7 (Xenobots, `Slot 083`):** Coupled an **8×8×7 3D voxel mass-spring lattice** (`456` nodes, `3,085` springs with Coulomb ground contact and explicit per-node Stokes drag) to a **2D force- and torque-free Cortez (2001) regularized-Stokeslet ciliary squirmer solver** (`2020-kriegman-xenobot-evolutionary-biobots.py`) advecting finite-size stem cells into the `θ = 105°` C-mouth pocket.

5. **12 Deterministic Commands Anyone Can Run to Verify Every Result:**
   Clone [`https://github.com/fractastical/morpholib`](https://github.com/fractastical/morpholib) and run any of these 12 self-contained commands from the repository root:
   ```bash
   # Claims 1-3: 4-Language Bit-Exact Verifier, 3-Ablation Audit, Hoel CE 2.0 & Lean 4 Proofs
   python3 sorting-morphogenesis/verify_multilang_ports.py
   python3 sorting-morphogenesis/audit_platonic_space_claims.py
   python3 sorting-morphogenesis/quantify_emergence_ce2.py
   python3 sorting-morphogenesis/validate_kymograph_and_paper.py --n 100 --reps 15
   lean morphoeval/lean/loops/Loop_zhang_sorting_morphogenesis.lean

   # Claim 4: D'Arcy Thompson 600-pt Silhouette Paradox, 4p Box Audit, 10,303 Fishes & Blind Test
   python3 thompson/1917-thompson-silhouette-area-paradox.py && python3 thompson/run_box_vs_warping_audit.py
   python3 thompson/run_three_dataset_validation.py
   python3 thompson/score_blind_predictions.py

   # Claims 5-7: Bioelectric Planarians, Picasso Tadpoles, Xenobots & Core Lean 4 Proofs
   python3 models/2011-beane-planarian-hk-atpase-anterior.py --self-test
   python3 models/2017-pietak-bioelectric-morphogen-coupling.py --self-test
   python3 models/2019-bischof-xenopus-craniofacial-remap.py --self-test
   python3 models/2020-kriegman-xenobot-evolutionary-biobots.py --self-test
   lean morphoeval/lean/MathOfBiology.lean
   ```

---

## Conclusion: Why We Don't Need "Platonic Ghosts" When the Math Is This Good

When we apply precise definitions and open-source code verification to the "Levin orbit," we get a result that is far more interesting than either uncritical guru-worship or dismissive blog snark:

1. **Where "Weak Structural Platonism" (The Mathematical Free Lunch) Is 100% Justified:**
   Living systems genuinely get "more out than they put in" from the mathematical geometry of their state spaces. Evolution does not need to write a separate gene for every anatomical detail because **the Coxeter lattice of permutations (`S_n`)** guarantees local swaps converge without local minima on `I(v)`; **kinematic speed differentials** automatically create phase-separated cell domains; **Nernst–Planck electrodiffusion** creates rewritable bistable voltage memory; and **D’Arcy Thompson’s 8-knob coordinate fields** channel 150 million years of vertebrate body-plan evolution.
2. **Where "Strong Dualist Ingress" Is Unnecessary:**
   Describing Delayed Gratification or chimeric clustering as *"side-quests—not predicted, explained, or produced by that medium"* is where the language overreaches. Once you read line 74 of `InsertionSortCell.py`, measure permutations with their causally emergent Lyapunov potential `I(v)` (`ΔCP = +0.1191 > 0`) instead of the lossy neighbor ruler `E(v)` (`ΔCP = −0.2525 < 0`), and test duplicate multiplicity at `C = 2` (`S B | S B`), **100% of the behavior is predicted, explained, and produced by the local transition matrix of the code.**

We don't need non-physical ghosts beaming into the machine, and we don't need reductionist dismissals that wave away real emergent phenomena without checking the math. All the tools to test the mathematics of life are sitting right at our fingertips.

---

### Interactive Labs, Papers & Code for Readers
* **Play with the Live Sorting Morphogenesis, "Two Rulers" & `C = 2` Anti-Clustering Simulator:** [https://fractastical.github.io/morpholib/bioelectricity/sorting-morphogenesis.html](https://fractastical.github.io/morpholib/bioelectricity/sorting-morphogenesis.html)
* **Explore the 10,303-Species D’Arcy Thompson & Bioelectric Morphospace Sandbox:** [https://fractastical.github.io/morpholib/bioelectricity/](https://fractastical.github.io/morpholib/bioelectricity/)
* **Read the 8-Page Sorting Morphogenesis & Causal Emergence 2.0 Paper (PDF):** [https://fractastical.github.io/morpholib/zhang_levin_sorting_morphogenesis_manuscript.pdf](https://fractastical.github.io/morpholib/zhang_levin_sorting_morphogenesis_manuscript.pdf)
* **Read the 28-Page D’Arcy Thompson 1917 & 10,303-Species Paper (PDF):** [https://fractastical.github.io/morpholib/thompson_1917_fish_transformations_manuscript.pdf](https://fractastical.github.io/morpholib/thompson_1917_fish_transformations_manuscript.pdf)
* **Open-Source Repository (Python 3, C++20, Node.js, Lean 4 Proofs & Signed Nanopublications):** [https://github.com/fractastical/morpholib](https://github.com/fractastical/morpholib)
