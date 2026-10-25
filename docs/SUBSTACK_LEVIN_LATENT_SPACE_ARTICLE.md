# Why Michael Levin Is 81.6% Right: Testing His "Latent Platonic Space" Claims with Code, Math, and 10,303 Fish (Between Galvani’s Frogs and Plato’s Ghosts)

*In a sea of internet debates between "Platonic Ingression" and reductionist blog critics, an internet philosopher who can code doesn’t have to guess—because every single core claim can be tested with tools at our fingertips (`92.4%` right on structural math, `0.0%` on non-physical ghosts).*

---

For some time I have been participating actively in documenting the scientific potential of bioelectricity (which goes back to Galvani’s frog experiments; see [Bioelectricity Foundations](https://bioelectricity.net/bioelectricity-foundations)) and the various fascinating, though usually hypothetical, metaphysical concepts that often travel side by side with it ([bioelectricity.net](https://bioelectricity.net)). As a historian of math, it doesn’t bother me that sci-fi concepts (such as *psi*) and real science can develop side by side—historically, Kepler’s astrophysics and Newton’s alchemy grew out of the same soil—but I’m equally aware that religious effects can take over and obliterate free critical thinking (including in Buddhist traditions that bind one to absolute obedience to a guru).

When I was working at the California Institute of Machine Consciousness in early 2026, my then-employer Joscha Bach would often elatedly insert concepts of *morphogenesis* into conversations. This term means, roughly, how things get to be their current shape, and tends to be a pretty mathy set of explanations.

When a board member of the organization asked me to explain it, I did so as follows:

> *"As someone with a computational background leaning into bio, Michael Levin has taken the scalpel down to the smallest component parts, worked on new imaging techniques, and tried to understand how cells take their form, which is related to newly observable phenomena via the electrical communication that happens between cells. This then is used to explain various other previously unexplainable phenomena."*

This satisfied her, but she said that until I explained it that way, most of the Mike Levin orbit concepts sounded instead impermeable or much like "woo science."

In my case, I was fortunately aided by Mike Levin himself, who gave me a list of articles to read in which order to get started on learning the space, to which I added a massive pile of his papers, which I synthesized with a knowledge graph ([Project Delphi](https://github.com/fractastical/project-delphi-ontology)). That knowledge graph then evolved into a three-stage series of open-source verification engines—starting with **NeuroEval** ([doi:10.5281/zenodo.20622878](https://doi.org/10.5281/zenodo.20622878), evaluating data transparency and neural-coding rigor in computational and biohybrid/wetware neuroscience papers), followed by **BioEval** ([doi:10.5281/zenodo.20567719](https://doi.org/10.5281/zenodo.20567719), evaluating bioinformatics and computational-biology papers), and finally culminating in **MorphoEval** ([`fractastical/morpholib`](https://github.com/fractastical/morpholib) and our **Adversarial Epistemic Loop / AEL** protocol), which turns that auditing machinery inward to autonomously build out, audit, and formally prove the math of biology in executable code and Lean 4.

![](https://fractastical.github.io/morpholib/media/all_forms_montage.gif)
*Fifteen flagship animated models of biological form (1917–2025) in our open-source [Infinite Morphospace](https://github.com/fractastical/morpholib) repository—audited by independent AI referees from competing labs and formally checked in the Lean 4 theorem prover.*

Although none of this required "Platonism," it was partially the concept of validating the concepts of what mathematicians called the **"morphospace"** and Levin calls the **"latent Platonic space"** that motivated the creation of these tools—and which now allows agents to explore math more directly and which has already led to notable discoveries.

As someone who needs sizable amounts of funding to conduct his experiments (including 400+ papers in the last decade out of his lab), he is probably more sensitive than I am to the language that makes funders want to fund. However, his more recent set of terms choosing "Platonism" and other terms like "ingress" (*Ingressing Minds*, [2025](https://doi.org/10.31234/osf.io/5g2xj_v3)/[2026](https://doi.org/10.3390/philosophies11050161)) have set off a large set of internet debates, in which I inadvertently got involved when *Sweet Rationalism* (Nathan Sweet) suggested my method for validation may be more comprehensive and better than Levin’s own.

Although we have briefly discussed these things in person, I do not know, at present, all of the experiments Mike Levin has underway. What I know is mostly limited to the core arguments he brings in his most recent paper, which are the **D’Arcy Thompson-inspired version of the morphology of math** ([Thompson 1917](https://doi.org/10.5962/bhl.title.11332)) and the **[Zhang, Goldstein & Levin (2025)](https://doi.org/10.1177/10597123241269740) paper regarding emergent phenomena in Bubble, Insertion, and Selection sort** (alongside the classic bioelectric flatworm, Picasso tadpole, and Xenobot models).

In both cases, **precise language of what is claimed is what is most important**, and, in a sea of definitions, a place where an internet philosopher who can code might be most useful. It also turns out that despite a bunch of folks who are "pro" and "against" on the internet, **all of the main claims can be validated with tools that everyone has at their fingertips.**

Let’s go through them claim by claim.

---

## Claim 1: "No Error-Handling Code Was Added" to Self-Sorting Cells ([Zhang, Goldstein & Levin 2025](https://doi.org/10.1177/10597123241269740))

In [*Adaptive Behavior* (2025)](https://doi.org/10.1177/10597123241269740) and Section 6.3 / Figure 14 of [*Ingressing Minds* (2026)](https://doi.org/10.3390/philosophies11050161), Taining Zhang, Adam Goldstein, and Michael Levin asked a clever question: **What if there is no central controller (no "gym teacher") sorting an array of numbers from 1 to 100?** What if each number is an independent thread (an autonomous "cell") that can only look at its neighbors and decide on its own whether to swap places?

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
   * What happened to **Ruler 1 (`I`)**? It went **strictly DOWN from `3 → 2`**! In fact, **every single Bubble swap in history reduces `I(v)` by exactly `1` (`ΔI = −1`, `0` uphill steps out of 192 swaps—the yellow straight line in Panel B!)**.
   * Why did **Ruler 2 (`E`)** go **UP from `1 → 2`** (`2 > 1` and `4 > 3`)? Because swapping `4` and `1` broke one tall cliff (`4 → 1`) into two smaller steps (`2 → 1` and `4 → 3`)! This happens **even when there are zero frozen cells (`f = 0`, where `DG = 0.38–0.43`)**.

![](https://fractastical.github.io/morpholib/media/zhang_13yo_fig2_two_rulers_illusion.png)

* **Verdict on Claim 2 — Optical Illusion of the Ruler:** The sorting cells never move against their dynamical gradient. They slide 100% deterministically downhill on their true Lyapunov energy `I(v)` (`ΔI = −1` on every single swap), while the observer watches them through a blurry neighbor-drop ruler `E(v)`.¹

---

## Claim 3: "Spontaneous Algotype Clustering" — Why Levin’s Phenomenon Is 100% Real, and the CS Blog Critics Were Wrong Twice!

Here is where our audit caught the *critics* making an even bigger mathematical mistake than the dualists.

In Figure 8 of Zhang et al. (and Figure 14C of *Ingressing Minds*), when you mix 50% **Bubble (`B`)** cells and 50% **Selection (`S`)** cells in the same line, cells of the same personality spontaneously group together (`aggregation > 0.50`)—even though **zero lines of code ever inspect `neighbor.algotype`**.

Levin called this an unprogrammed "side-quest" from Platonic space.
In response, computer-science bloggers (including Nathan Sweet at *Sweet Rationalism*) argued that this clustering is nothing more than a 1960s textbook property: **duplicate-key sort stability** (when multiple cells have the same number, Bubble sort is "stable" and doesn't swap equal keys).

When we ran the ablations (**Panel C** of the audit figure above, and Illustrated Guide 3 below), we proved that **the CS blog critics were wrong twice—both on unique numbers AND on duplicate numbers:**

1. **Critics’ Mistake #1 — Paper Fig. 8A Has ZERO Duplicate Numbers!**
   In Paper Figure 8A (**Panel C**, blue curve), every single cell has a **strictly unique number from `1` to `n` (`0` duplicate keys)**. When there are no duplicate keys, "duplicate-key sort stability" is mathematically impossible! Yet `B + S` aggregation still rises from `0.50` to a peak of **`0.624–0.640`** mid-sort before relaxing to `0.501` at the end.
   **Why?** Because **Selection cells move like sports cars and Bubble cells move like pedestrians!** Selection cells use their `target` pointer to jump small numbers straight to the left front of the line (`0, 1, 2...`) in `O(1)` swaps, shoving a contiguous wave of slow `O(n²)` Bubble cells into the right half of the line.

2. **Critics’ Mistake #2 — The `C = 2` "Anti-Clustering" Smoking Gun on Duplicate Numbers!**
   What about Paper Figure 8D, where there *are* duplicate numbers (`C` copies of each key)? Think carefully: at the start of the sort, the `B` and `S` labels are shuffled randomly (`aggregation = 0.50`). If textbook "sort stability" simply preserved the initial relative order of equal keys, then **final aggregation at the end of the sort would stay at `0.50` for every `C`!**
   Instead, when we swept `C = 1, 2, 4, 8, 10, 20, 40` copies per number, we discovered that because Selection cells scan `target` from `0` upward, **66.2% (at `C = 2`) to 75.0% (at `C = 10`) of Selection cells pack into the LEFT half of every number’s group (`S...S | B...B`)**:
   * When there are **`C = 2` copies of each number** (one `S` and one `B` wearing `#1`, one `S` and one `B` wearing `#2`, etc.), putting `S` on the left and `B` on the right of each pair creates the alternating pattern **`S B | S B | S B`**—which drives clustering **below random to `0.477` (Anti-Clustering!)**!
   * When there are **`C = 10` copies of each number** (`Fig. 8D`), putting 5 `S` cells on the left and 5 `B` cells on the right of each group creates **`SSSSS | BBBBB`**, driving permanent clustering **up to `0.659`**! (And when we change `<` to `<=` on ties in **Panel C**, red dashed curve, the permanent clustering collapses back to `0.497–0.514`.)

![](https://fractastical.github.io/morpholib/media/zhang_13yo_fig3_sports_cars_vs_pedestrians.png)

And here are the real computer-experiment curves side by side—showing **(A)** the smooth downhill slide on Ruler 1 (`I(v)`, `0` uphill steps) vs. the 47 illusory bumps on Ruler 2 (`E(v)`), and **(B)** how the `C = 2` Anti-Clustering dip (`0.477 < 0.50`) and `C = 10` Clique peak (`0.659`) refute the blog critics’ flat `0.50` prediction:

![](https://fractastical.github.io/morpholib/media/zhang_13yo_fig4_real_data_curves.png)

* **Verdict on Claim 3 — Empirically Vindicated (via Kinematic Phase Separation):** The clustering is 100% real and refutes the blog critics’ "textbook stability" dismissal—yet it is also 100% produced and predicted by the kinematic transport equations of the two algorithms without needing non-physical ghosts. ([Try the live `C = 2` Sorting Morphogenesis Simulator on bioelectricity.net here!](https://bioelectricity.net/sorting-morphogenesis))

---

## Claim 4: D’Arcy Thompson’s (1917) Generative Morphospace Across 10,303 Fish Species

The second major pillar Levin invokes for his latent space is **D’Arcy Wentworth Thompson’s (1917)** famous coordinate transformations from *On Growth and Form*: the idea that nature doesn't redesign every bone from scratch when evolving a parrotfish (*Scarus*) into an angelfish (*Pomacanthus*) or a pufferfish (*Diodon*) into an ocean sunfish (*Mola mola*), but instead turns a few global **"growth knobs"** that bend the 2D coordinate grid of the body like a sheet of rubber.

![](https://fractastical.github.io/morpholib/guide_step1_1917_rubber_sheet.png)

For the last 30 years, evolutionary biologists tried to test Thompson’s idea on computer landmark databases—and concluded he was mostly wrong across fish families. **Why did their tests fail?**

When we audited the morphometrics literature across **three independent datasets scaling up to 10,303 fish species (`440` families)**, we uncovered two methodological traps that had hidden Thompson’s law for a century:

1. **Trap #1 — The "17-Dot Head Seesaw":**
   Standard fish databases ([Claverie & Wainwright 2014](https://doi.org/10.1111/evo.12502); [Friedman et al. 2019](https://doi.org/10.1093/sysbio/syz029)) don't trace the whole outline of the fish—they measure **17 dots on the skeleton**, with **9 of those 17 dots (53%) packed inside the head and jaws** and **0% on the outer dorsal, anal, or tail fins**. Because jaws evolve rapidly for diet while the body evolves for swimming, fitting a global rubber sheet to those 17 dots lets the 9 head dots pull the back of the sheet the wrong way—making the warped fish silhouette look **worse than no warp at all** (`40.6% → 56.2%` mismatch on *Scarus* → *Pomacanthus*)!

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

* **Verdict on Claim 4 — Strongly Confirmed:** Across 10,303 species, Thompson’s and Levin’s core thesis—that macroevolution navigates a low-dimensional geometric coordinate manifold while local modules handle rapid ecological competition—is a quantitative law of vertebrate evolution. ([Explore the interactive 10,303-Species Fish Morphospace on bioelectricity.net here!](https://bioelectricity.net/bioelectric-morphospace))

---

## Claims 5, 6 & 7: How Bioelectric & Biomechanical Attractors Physically Implement the Latent Space (Planarians, Picasso Tadpoles & Xenobots)

How does a living tissue actually store a target coordinate in that morphospace? In our repository, we also built and audited mathematical reproductions of Levin’s three most famous wet-lab discoveries—testing whether standard biophysical PDEs and ODEs genuinely produce those attractor behaviors (while explicitly documenting reduced-model parameter assumptions in our MorphoEval Pass 3 referee audits):

* **Claim 5 — Bioelectric Target Memory in Two-Headed Planarians ([Oviedo et al. 2010](https://doi.org/10.1016/j.ydbio.2009.12.012); [Beane et al. 2011](https://doi.org/10.1016/j.chembiol.2010.11.012); [Pietak & Levin 2016](https://doi.org/10.3389/fphys.2016.00178)/[2017](https://doi.org/10.1098/rsif.2017.0425); [Cervera, Levin & Mafé 2021](https://doi.org/10.1016/j.biosystems.2021.104511)):**
  In the lab, blocking gap junctions in a cut planarian flatworm with 8-octanol for 48 hours causes it to regenerate **two heads (`2H0T`)**—and weeks after the drug is washed out, cutting the worm again in plain water *still* produces two heads with zero DNA change. (For scale across the literature, our exact McKay IR isomorphism census of [Lobo, Malone & Levin’s 2013 `PlanformDB 2.5.0`](https://doi.org/10.1093/bioinformatics/btt088) shows that its `369` recorded planarian morphologies reduce to `38` exact region-graph topologies across `12` unlabelled tree skeletons, with wild-type `1H1T` first at `n = 253` and bipolar double-head `2H0T` second at `n = 16`.)
  In our 1D PDE and 405-cell 2D Voronoi Scharfetter–Gummel electrodiffusion models (`2017-pietak-bioelectric-morphogen-coupling.py`), positively charged serotonin (`5-HT⁺`, `z = +1`) electrophoreses down the resting voltage gradient (`−20 mV` head to `−60 mV` tail) by the exact Boltzmann ratio `c_post / c_ant = exp(40 mV / 25.85 mV) ≈ 4.70`. Crucially, **gap-junction conductance `G_gj` cancels out of the zero-flux equilibrium ratio!** Once transient 8-octanol blockade lets the posterior pole flip to `−20 mV` (`2H0T`), washing out the drug leaves `ΔV = 0` and permanently locks the worm in the two-headed attractor basin.

![](https://fractastical.github.io/morpholib/media/cervera_levin_mafe_gj_block.gif)

* **Claim 6 — Closed-Loop "Picasso Tadpole" Homeostasis ([Vandenberg, Adams & Levin 2012](https://doi.org/10.1016/j.ydbio.2012.01.027); [Pezzulo & Levin 2015](https://doi.org/10.1098/rsif.2015.0772); [Pinet et al. 2019](https://doi.org/10.1242/dev.175950)):**
  When tadpole eyes, jaws, and nostrils are surgically scrambled at stage 45, they don't follow hardwired displacement vectors (which would preserve 100% of the scrambling error at metamorphosis). Instead, they move along novel paths with speed proportional to defect distance until they reach the normal froglet face—unless MMP tissue remodeling is blocked by GM6001. Our Lean-verified Procrustes / Variational Free Energy Lyapunov models (`dV/dt ≤ 0`, coupled in Pass 3 to 3D Maxwell viscoelastic cartilage rods and MMP-2/MMP-9 zymography fields) reproduce both normal convergence and GM6001 arrest.

![](https://fractastical.github.io/morpholib/media/xenopus_craniofacial_remap_growth.gif)

* **Claim 7 — Xenobots & Kinematic Self-Replication Without Genetic Change ([Kriegman, Blackiston, Levin & Bongard 2020](https://doi.org/10.1073/pnas.1910837117), [2021 *PNAS*](https://doi.org/10.1073/pnas.2112672118)):**
  In Kriegman et al. (2021), when ciliated *Xenopus laevis* embryonic skin (animal-cap ectoderm) spheroids are sculpted into a **C-shaped ("Pac-Man") mouth concavity (`θ ≈ 105°`)** and placed in a Petri dish filled with dissociated individual frog embryonic cells, their beating cilia generate fluid currents that sweep the dissociated cells into the mouth pocket; once enough cells are pushed together, natural cadherin cell-cell adhesion bonds them into a new, motile daughter Xenobot (`3–4` generations of kinematic self-replication). Both our 8×8×7 3D voxel mass-spring lattice and our 2D regularized-Stokeslet squirmer fluid simulation confirm that `θ ≈ 105°` is a genuine hydrodynamic capture optimum (`3–4` daughter generations vs. `1` for a sphere and `0` for a flat slab).

![](https://fractastical.github.io/morpholib/media/xenobot_evolutionary_biobots_growth.gif)

---

## The Final Scoreboard: How Right Is Michael Levin? (`81.6%` Across the 3 Equal `33.3%` Pillars — `92.4%` on Structural Math, `0.0%` on Non-Physical Ghosts)

![](https://fractastical.github.io/morpholib/media/substack_final_scoreboard.png)

To give a concrete, transparent answer to *"How right is Michael Levin?"*, we gave **each of the three major pillars of this article an equal `33.3%` (`1/3`) weight**:
1. **Pillar I — Self-Sorting Cells (Claims 1–3 = `33.3%` total, `11.1%` each)**
2. **Pillar II — D’Arcy Thompson’s Generative Morphospace (Claim 4 = `33.3%` total)**
3. **Pillar III — Bioelectric & Xenobot Attractors (Claims 5–7 = `33.3%` total, `11.1%` each)**

Here is how the subcomponents add up across the three equal pillars:

* **Pillar I — Self-Sorting Cells (`18.5 / 33.3 pts = 55.6%` of Pillar I):**
  * **Claim 1 — Sorting Error Tolerance (`7.4 / 11.1 pts = 66.7%` — PARTIALLY REFUTED):** Bubble (`+3.7 / 3.7`) and Selection (`+3.7 / 3.7`) work without error-handling code, while Insertion (`+0.0 / 3.7`) relies on an explicit 3-line `if status == FREEZE: prev = -1` reset in `InsertionSortCell.py` (L74–76; `4.5×` error jump when removed).
  * **Claim 2 — Sorting "Delayed Gratification" (`1.5 / 11.1 pts = 13.5%` — OPTICAL ILLUSION OF THE RULER):** The neighbor-drop curve `E(v)` genuinely ticks up 47 times (`+1.5 / 3.7`), but Lean 4 Theorem 1 proves `0` Bubble/Insertion cells cross an immovable barrier (`+0.0 / 3.7`) and every Bubble swap goes strictly downhill (`ΔI = −1`, `0/192` uphill steps, `+0.0 / 3.7` dualism) on the true pairwise inversion energy `I(v)`.¹
  * **Claim 3 — Secret Clique Clustering (`9.6 / 11.1 pts = 86.5%` — EMPIRICALLY VINDICATED & CRITICS REFUTED TWICE):** Strictly unique keys (`Fig. 8A`, `+4.8 / 4.8`) and `C = 2` anti-clustering vs. `C = 10` cliques (`+4.8 / 4.8`) are 100% real and refute the CS blog critics twice, while `+0.0 / 1.5` pts go to non-physical dualism.
* **Pillar II — D’Arcy Thompson’s Generative Morphospace (`32.0 / 33.3 pts = 96.0%` of Pillar II):**
  * **Claim 4 — 600-Point Homologous Fin Outlines (`+16.7 / 16.7 pts`) & 10,303-Species / 5,180-Blind-Test Validation (`+15.3 / 16.6 pts` — STRONGLY CONFIRMED):** Standard 17-dot tests failed only because 53% of the dots sit in the skull. On homologous perimeter outlines, Thompson’s 8-knob grids win by **`+13%` to `+48%`** (`100%` un-folded) and align with **150 Myr of fish body-plan evolution at `93.5%–98.9%`** across 440 families (`p = 10⁻¹⁰³` in our pre-registered blind test; losing `1.3 pts` only because `H2` nearest-neighbor margin was `4%`, `p = 0.061`).
* **Pillar III — Bioelectric Memory (`2H0T` Planarians), Picasso Tadpoles & Xenobots (`31.1 / 33.3 pts = 93.3%` of Pillar III):**
  * **Claim 5 — Planarian `2H0T` Bioelectric Memory (`10.55 / 11.1 pts = 95%`):** Zero-flux Nernst–Planck Boltzmann ratio (`exp(40/25.85) ≈ 4.70`) cancels `G_gj` at equilibrium and locks permanent two-headed memory after washout (`+0.0 / 0.55` for dualist extras).
  * **Claim 6 — Picasso Tadpole Homeostasis (`10.55 / 11.1 pts = 95%`):** Closed-loop Procrustes / Variational Free Energy Lyapunov descent (`dV/dt ≤ 0`) coupled to 3D Maxwell cartilage & MMP zymography reproduces face normalization and GM6001 arrest (`+0.0 / 0.55` for dualist extras).
  * **Claim 7 — Xenobot Kinematic Self-Replication (`10.0 / 11.1 pts = 90%`):** 8×8×7 3D voxel lattice and 2D regularized-Stokeslet squirmer confirm `θ ≈ 105°` C-mouth concavity as the hydrodynamic capture optimum (`3–4` daughter generations; `+0.0 / 1.1` for unmodeled 3D continuum/dualist extras).
* **How the 3 Equal Pillars Add Up (`18.5 + 32.0 + 31.1 = 81.6 / 100 pts = 81.6% Overall`):**
  * **3-Pillar Composite Score: `(55.6% + 96.0% + 93.3%) / 3 = 81.6%`**
  * **On Empirical Phenomena & Structural Mathematical Free Lunches (`D + P_structural`): `81.6 / 88.3 pts = 92.4% RIGHT!`**
  * **On Strong Dualist "Non-Physical Ingression" (`P_dualist`): `0.0 / 11.7 pts = 0.0% NEEDED!`** *(where the `11.7` pts of dualist/overstated claims are `3.7` in Claim 1 Insertion + `7.4` in Claim 2 barrier/gradient + `0.6` across Claims 3–7).*

---

## How We Built and Audited This: From NeuroEval → BioEval → MorphoEval (The Adversarial Epistemic Loop)

How do you keep yourself honest when testing high-profile scientific claims—especially when both the "Platonic Space" camp and the "Nothing New Here" blog critics have strong priors?

The methodology used in this article—which we call the **Adversarial Epistemic Loop (AEL)**—was developed across three successive evaluation systems in Project Delphi:
1. **Stage 1 — NeuroEval ([Zenodo `10.5281/zenodo.20622878`](https://doi.org/10.5281/zenodo.20622878)):** Built first to evaluate data transparency, reproducibility, and neural-coding rigor in computational and biohybrid/wetware neuroscience papers.
2. **Stage 2 — BioEval ([Zenodo `10.5281/zenodo.20567719`](https://doi.org/10.5281/zenodo.20567719)):** Extended that rubric to score bioinformatics and computational-biology papers for reproducibility, statistical calibration, and information-theoretic rigor.
3. **Stage 3 — MorphoEval ([`fractastical/morpholib`](https://github.com/fractastical/morpholib)):** Turned the evaluation machinery from *grading published papers* to *generating, executing, and auditing the mathematical models themselves* in a 12-step closed loop—where every verbal claim is translated into multi-language executable code (**Python 3, C++20, Node.js**) and machine-checked **Lean 4** theorems (`0 sorry`), locked against **SHA-256** empirical datasets, audited blind by **three competing AI model families** (Google DeepMind’s `gemini-4-argon`, Anthropic’s `claude-fable-5-1` / `claude-sonnet-5-5`, and OpenAI’s `gpt-5.6-sol`), and pre-registered as cryptographically signed **Nanopublications** before unblinding held-out species.

📄 **Want the full line-by-line code traceability tables, all 24 mathematical/biophysical equations, SHA-256 dataset locks, and the 12 terminal commands to re-run every experiment?** Read our standalone **[4-Page Illustrated Methods & Provenance PDF (`levin_latent_space_methods_and_provenance.pdf`)](https://fractastical.github.io/morpholib/levin_latent_space_methods_and_provenance.pdf)** (and **[Markdown Methods Guide (`docs/LEVIN_LATENT_SPACE_METHODS.md`)](https://github.com/fractastical/morpholib/blob/main/docs/LEVIN_LATENT_SPACE_METHODS.md)**).

### Transparency Heatmap: How Much of This Article and Pipeline Was Written by AI vs. Human?

Because our **Adversarial Epistemic Loop (AEL)** pairs human philosophical direction and domain skepticism with multi-agent code generation, formal theorem proving, and cross-provider refereeing, we also audited the exact **Human vs. AI-Agent contribution** across all 8 sections of this article and all 5 workflow tasks:

![](https://fractastical.github.io/morpholib/media/substack_ai_authorship_heatmap.png)
*Methods Transparency Heatmap: Breakdown of AI-Agent vs. Human PI (Joel Dietz) contribution across all 8 sections of this article and 5 workflow dimensions (Hypothesis & Direction, Prose Drafting, Simulation & Math Code, Lean 4 & Statistical Proofs, and Blinded Referee Auditing).*

* **Hypothesis & Philosophical Direction (`80.6%` Human / `19.4%` AI):** The personal narrative (Galvani, Joscha Bach at CIMC, the board member’s "woo science" question, Mike Levin’s reading list, the Foresight Institute metascience reflection), the visual skepticism that exposed the `9/17` skull-dot seesaw trap and the 4-parameter Box-Stretch illusion, and the `D` vs. `D + P` framing were driven by the human author (**Joel Dietz**, `85%–95%` human in Sections 1, 5, 7, and 8).
* **Article Prose Word Count (`38.4%` Direct Human Writing & Edits / `61.6%` AI-Assisted Technical Synthesis):** Section 1 (the personal introduction) is **`92%` verbatim human prose**, and Section 8 (the `D` vs. `D + P` synthesis, Foresight metascience reflection, and conclusion) is **`52%` human prose**, while the technical claim walkthroughs and methods summaries (Sections 2–7) were `62%–78%` drafted by `gemini-4-argon` and `claude-fable-5-1` from the repository’s locked JSON/YAML audit files and edited by the human author.
* **Multi-Language Simulation Code, Lean 4 Proofs & Referee Audits (`85.8%` AI Agents / `14.2%` Human):** The C++20, Node.js, and Python ports, the 2D/3D PDE and Stokeslet fluid solvers, the 25+ kernel-checked **Lean 4** theorems (`0 sorry`), and the blinded cross-provider referee audits (`claude-fable-5-1`, `claude-sonnet-5-5-high`, `gemini-4-argon`, and OpenAI’s `gpt-5.6-sol`) were `85%–96%` executed by the AI agents under deterministic `--self-test` gates.

---

## Answering the "Decisive Comparison (`D` vs. `D + P`)", "Legacy P1 Category" & "Frozen Update Rule" Critique

In a thoughtful follow-up to my earlier comments, Nathan Sweet (*Sweet Rationalism*) laid out what he calls the **decisive-comparison requirement** for any predictive morphospace language:

> *"Joel Dietz points in the right direction by proposing a predictive morphospace language connecting bioelectric patterns to downstream gene-expression cascades and morphology. A knowledge graph, however, can encode `D` or `D + P`. Platonic ingress begins earning evidence only when at least one `P`-specific relation improves held-out prediction, intervention, compression, or comparative model performance beyond the best `D`-only account.*
>
> *The decisive experiment would make `P` itself measurable at the edges: specify its hard core and bridge principle before observation; define a `P`-linked variable, topology, relation, or invariant independently of the residual; identify outcomes that materially separate `D + P` from serious rivals; use negative controls and held-out embodiments; replicate the advantage independently. Most important, freeze the update rule. A failed bridge must cost `P` something. A repair must buy fresh risk."*

And on the legacy `P1` metadata category in Project Delphi:

> *"The publication category is not the experimental contrast. Project Delphi’s legacy `P1` example makes this distinction tangible. The record proposes perturbing or removing tissue and measuring anatomical restoration, while its metadata places the proposal in a Platonic-forms category. That label can organize the proposal’s intellectual setting. Recovery evidence first bears on the developmental capacity being tested; an independent source of forms earns further credit through a comparison that distinguishes its contribution from regulatory and organizational alternatives."*

**We agree 100% with every single one of those criteria—and in fact, our repository’s entire architecture was built to enforce that exact `D` vs. `D + P` protocol, including the frozen update rule.**

![](https://fractastical.github.io/morpholib/media/zhang_13yo_fig6_decisive_comparison_d_vs_dp.png)
*Illustrated Guide 6: (Rule 1, Top) Testing `D` (the 6 lines of local code alone) against `D + P` (adding a "Platonic Ghost" sticker), showing `0.00` extra swaps predicted by `P`; and (Rule 2, Bottom) locking predictions in a digital safe before unblinding new species—and taking losses honestly when a bridge fails on hornet wings.*

Let’s walk through Nathan’s criteria one by one against the actual experiments in this repository:

1. **Why "The Publication Category (`P1`) Is Not the Experimental Contrast":**
   In early Project Delphi nanopublication schemas (`P1`), placing a regeneration experiment inside a `Platonic-forms` collection was a **bibliographic filing-cabinet tag** (grouping claims inspired by Levin’s papers), *not* an evidentiary proof that anatomical recovery comes from outside physics. As our current [`README.md`](https://github.com/fractastical/morpholib) and 4-tier claim-boundary catalog (`sorting-morphogenesis/catalog.json`) explicitly enforce, **recovery evidence first credits the dynamical/regulatory mechanism `D` (`1H1T`/`2H0T` Nernst–Planck bistability or Procrustes Lyapunov descent `dV/dt ≤ 0`)**. Putting a "Platonic" metadata sticker on a differential equation never gives `P` a free ride on `D`’s empirical success.

2. **Separating `P_structural` (Mathematical Invariants Defined Independently of the Residual) from `P_dualist` ("Non-Physical Side-Quests"):**
   Nathan writes that `P` begins earning evidence only when we *"define a `P`-linked variable, topology, relation, or invariant independently of the residual"* that *"improves held-out prediction, intervention, compression, or comparative model performance beyond the best `D`-only account."*
   Look at **Rule 1 (top row)** of Illustrated Guide 6 above. Once we make that definition precise, `P` splits into two completely different hypotheses with opposite experimental outcomes:
   * **`D + P_dualist` (Strong Dualist Ingression — "Side-Quests Not Produced by the Medium"): FAILS (`0.00` Incremental Standing).**
     In *Ingressing Minds* (Sec. 6.3), Levin claims that Delayed Gratification (`DG > 0`) and chimeric algotype clustering are *"side-quests—not predicted, explained, or produced by that medium."* When we test `D + P_dualist` against `D` alone (the 6 local cell rules `[U1]–[U6]` in `Zhangtaining/cell_research@1fd2bd59`), our bit-exact ports across Python 3, C++20, Node.js, and Lean 4 prove that **`P(D + P_dualist) − P(D) = 0` (`0` cell mismatches; published to the production Nanopub network as [`RAH6FDLaacsq28cjx2NVRYRANqOThwHaHwE58ByxqIWI8`](https://registry.knowledgepixels.com/np/RAH6FDLaacsq28cjx2NVRYRANqOThwHaHwE58ByxqIWI8) and [`RA_mMJXveSMNlxIjYu_oTtIY233QfJnNCf47yRLfInMuE`](https://registry.knowledgepixels.com/np/RA_mMJXveSMNlxIjYu_oTtIY233QfJnNCf47yRLfInMuE))**. Adding a non-physical ingression edge improves neither prediction, nor intervention, nor compression by a single bit.
   * **`D + P_structural` (Structural Mathematical Invariants Defined Independently of the Residual): EARNS REAL COMPRESSION & OUT-OF-SAMPLE STANDING.**
     What if, instead of a dualist ghost, the "latent pattern" is a **mathematical invariant or symmetry group defined strictly a priori (independently of any fitted residual)**?
     * **In Sorting Arrays:** The Coxeter pairwise inversion potential `I(v)` on the symmetric group `S_n` is defined purely from the algebra of permutations (zero fitted parameters). Coarse-graining the 720 micro-states of `S_6` onto `I(v)` **compresses the state space `45×` (`720 → 16` states)** while **increasing Markov determinism from `86.9%` to `100.0%` (`ΔI = −1` on every single swap)**!
     * **In Fish Morphospace:** D’Arcy Thompson’s 7 linearized tangent coordinate modes (`aspect`, `shear`, `2 fans`, `radial bow`, `2 coaxial Möbius modes`) are defined analytically from 1917 geometry—*completely independently of any species residuals*. Projecting 30-D landmark space onto those 7 analytical modes **compresses the evolutionary covariance matrix from `465` parameters to `28`**, lowers the matrix condition number **`10×` (`κ: 201 → 20`, or `>1,000×` on small clades)**, and when fitted solely to the 600-point outer silhouette, **predicts 14 held-out interior skeletal bones `2.4×` more accurately** than a 600-point Thin-Plate Spline (`0.045` vs. `0.107` Procrustes RMS, while keeping `100%` of the interior tissue grid un-folded vs. `0%` for TPS)!
     * **Cautionary Counter-Example (When a Generative Grammar Does *Not* Earn `P_structural` Credit):** We also tested whether an unconstrained graph-rewrite grammar on [`PlanformDB 2.5.0`](https://doi.org/10.1093/bioinformatics/btt088) ([`models/2026-wolfram-planform-multiway-rewriting.py`](https://github.com/fractastical/morpholib/blob/main/models/2026-wolfram-planform-multiway-rewriting.py), audited by `claude-fable-5-1-thinking-high`) could earn `P_structural` credit. It cannot: a 3-rule leaf-addition and vertex-recoloring grammar ($\mathcal{B}_3 = \{L_1, D_1, D_2\}$) generates *every* possible 3-colored forest (`754,520` canonical trees at depth 12 to hit `37/38` observed PlanformDB classes, a **precision of `0.0049%`**), so the grammar itself compresses nothing and predicts nothing out of sample—the biological target (`1H1T` vs. `2H0T`) is selected entirely by the bioelectric dynamics `D`.

3. **Freezing the Update Rule Before Observation (`"A Failed Bridge Must Cost Something"`):**
   Nathan’s sharpest requirement is: *"Specify its hard core and bridge principle before observation... use negative controls and held-out embodiments... Most important, freeze the update rule. A failed bridge must cost `P` something."*
   Look at **Rule 2 (bottom row)** of Illustrated Guide 6:
   * **Frozen Before Observation:** Before downloading the Smithsonian’s **FishShapes v1** dataset (`16,523` specimens, `5,180` unseen target species), we froze the 7-mode Thompson bridge, the 10,888 species predictions, the 7 hypotheses (`H1–H7`), and the exact scoring script (`thompson/score_blind_predictions.py`) by SHA-256 hash (`commit dac08493`) and published them to the immutable Nanopublication network ([`RAilN4nbRHp3e3ZHqlbIgeL3UGPRV_KtQHtJD3gzVVdqw`](https://registry.knowledgepixels.com/np/RAilN4nbRHp3e3ZHqlbIgeL3UGPRV_KtQHtJD3gzVVdqw)).
   * **Where Failure Cost Confidence (Published in Outcome Nanopub [`RAJancwGOPEkHzBoZXGv_vu_snW0Lff2FktE85Me1P-fI`](https://registry.knowledgepixels.com/np/RAJancwGOPEkHzBoZXGv_vu_snW0Lff2FktE85Me1P-fI)):**
     1. **`H2` Failed on 1D Calliper Depth:** While the bridge crushed the consensus within 50 Myr (`H1`, `p = 10⁻¹⁰³`, `r = 0.87`, `92.6%` coverage), it only beat copying the single nearest relative by `4%` (`0.0383` vs. `0.0400`, `p = 0.061`), so **`H2` was formally rejected**.
     2. **`H6` Showed the Exact Cost of 7-Mode Compression:** On that 1D ratio, the 7-mode Thompson bridge (`0.0588` error) lost to the unconstrained 30-D shape space (`0.0383`) while beating the consensus (`0.0927`, `p = 10⁻⁸¹`)—quantifying the exact trade-off between a 7-mode geometric invariant and local 30-D degrees of freedom.
     3. **Negative-Control Clade Falsified on Hornet Wings (`0/7` Passed):** When we locked and ran the exact same 7-mode Thompson bridge (`W1–W7`, commit `330ea961`) on a held-out non-vertebrate embodiment—**1,027 *Vespinae* hornet wings across 30 species** (`thompson/portfolio/groups/vespinae_wings.py`)—**`0 out of 7` hypotheses passed** (`38.8%` variance in Thompson-7 vs. `95.5%` in PCA-7), proving that insect wing venation is **not** governed by a global Thompson coordinate sheet!

4. **"A Repair Must Buy Fresh Risk" — And the Metascience & Psychological Power of Failing Fast in Public:**
   What happened after the global Thompson bridge failed (`0/7`) on hornet wings and (`4/9`) on bee wings? Under our protocol, **you are not allowed to silently tweak parameters on the same dataset**. Instead:
   * Every revision must publish a new signed Nanopublication linked via `npx:supersedes` (so the failed prediction stays permanently in the public graph),
   * To explain *why* hornet and bee wings diverged from Thompson’s law, we first **pre-registered 8 causal hypotheses (`C1–C8`: `P`-drift null, allometric alignment, digitizer artifact, and PGLS allometry) in commit `1da6218f` *before* running the causal decomposition in commit `ae3b36b6`** (`5/13` supported, ruling out digitizer noise and simple body-size allometry while isolating the failure to interior cross-vein junctions), and
   * We then locked a third-round biomechanical & phylogenetic overhaul (**`D1–D8` in commit `838a7fe1`**, scored in `thompson/portfolio/groups/vespinae_venation_mechanics.py` and signed as Nanopub [`RAhDgbkGRriVeuYJ4E85P3d_vZJlrMYYeU-8jIqRD2pqQ`](https://registry.knowledgepixels.com/np/RAhDgbkGRriVeuYJ4E85P3d_vZJlrMYYeU-8jIqRD2pqQ), **`7/8` supported, Brier `0.131`**): replacing OpenTree's 15-way *Vespa* polytomy with the resolved Total-Evidence phylogeny of Perrard et al. (2013) restored strong phylogenetic signal (**Pagel's `λ = 0.533 → 0.817` in Thompson-7 and `0.670 → 0.946` in full space**) and cut leave-one-species-out error by **37.5% (`0.0064 → 0.0040`, `p < 10⁻⁵`)**; partitioning the wing into its **11 outer perimeter/spar landmarks** versus **8 interior flexion/cross-vein junctions** proved that the outer wing frame is **`65.8%` Thompsonian (exceeding fish bodies at `57.9%`!)**, whereas global least-squares drags the antiparallel prestigma (`LM 12–5`, `r = −0.80`) and basal `M` vein (`LM 13–14`, `r = −0.89`) in the wrong direction (`R² = −0.34` and `−0.52`, proved universally in `morphoeval/lean/loops/Loop_perrard_vespinae_wing_mechanics.lean`); and while our pre-registered isotropic Laplacian truss (`42.6%`, `D4` ✗) missed its `+12` pt hurdle, replacing the 4 isotropic modes with the 4 anatomical antiparallel vein-dipole modes lifted captured variance to **`64.2%`**!

   At a recent **Foresight Institute Metascience event**, there was a great discussion on **celebrating failure in science**—and why building infrastructure that lets researchers **fail fast and publicly in hypothesis generation** is not just a methodological upgrade, but a massive **psychological benefit**.
   In traditional academia (and especially in charismatic "guru" orbits), a scientist’s ego and funding get fused to a single grand narrative: admitting a hypothesis failed feels like a career death sentence, so people retreat into unfalsifiable language where nothing is ever allowed to be wrong.
   By contrast, when your hypotheses are locked as machine-readable Nanopublications and tested in minutes against SHA-256-pinned datasets, **public failure stops being a personal humiliation and becomes a badge of honor**. Seeing our 17-dot fit fail (`40.6% → 56.2%` mismatch), seeing `H2` fail on 1D calliper depth (`p = 0.061`), or seeing `0/7` Thompson predictions pass on hornet wings wasn't a disaster—it was liberating! It freed us psychologically from defending bad assumptions and pointed us immediately toward the real discoveries (the `9/17` skull-dot seesaw trap, the `I(v)` vs. `E(v)` two-rulers illusion, and the `C = 2` anti-clustering smoking gun).

That is the exact synthesis where coding philosophers and empirical biologists can shake hands: **a "latent space" is not a mystical ghost that gets credit from metadata labels (`P1`), nor is it an unfalsifiable story. When you define a mathematical invariant (`I(v)` or Thompson’s 8-knob coordinate group) independently of the residual, lock its predictions by SHA-256 before seeing held-out species, celebrate when failed bridges (`H2` or `Vespinae` wings) visibly fail in public, and force every repair to buy fresh pre-registered risk, you turn metaphysics into hard science.**

---

## Conclusion: Why Michael Levin Is 81.6% Right (92.4% on Structural Math, 0.0% on Non-Physical Ghosts)

In a private exchange with Levin, I cited my other article on Hellenic Tantra and noted the longstanding Hellenic practice—running through Eleusis, Delphi, and Plato’s own dialogues (*Seventh Letter*, 341c–d)—of a nonverbal, ritualized encounter with other forms of knowing that were not even allowed to be scribbled down. Re-reading D’Arcy Thompson’s *On Growth and Form*, I was struck by the explicit influence of Aristotle’s *History of Animals*—and when I went back to Thompson’s own 1910 Oxford translation and prefatory note to that work (where he shows how Aristotle’s biology grew out of direct observation at a specific place, the lagoon of Pyrrha on Lesbos), I realized that this "direct knowing," rooted in specific places, is what actually grounds and enforces epistemic rigor. It is what allows us to formulate things that are genuine mathematical laws rather than the "would-be" or "almost" laws we so often find in the contemporary literature of emergence. That is also the wisdom of tradition and of standing on the shoulders of giants: Thompson stated his coordinate transformations as conjecture and presumably yearned for the proof—and here we stand, with modern computational tools, holding something that approaches a true law of biological form.

When we apply precise definitions, a `3 × 33.3%` equal-pillar rubric, and open-source code verification to the "Levin orbit," we get a result (`81.6%` overall: `18.5 / 33.3` on Sorting Claims 1–3, `32.0 / 33.3` on D’Arcy Thompson’s Morphospace in Claim 4, and `31.1 / 33.3` on Bioelectric & Biobot Attractors in Claims 5–7) that is far more interesting than either uncritical guru-worship or dismissive blog snark:

1. **Where "Weak Structural Platonism" (The Mathematical Free Lunch) Is 92.4% Vindicated (`81.6 / 88.3 pts`):**
   Living systems genuinely get "more out than they put in" from the mathematical geometry of their state spaces. Evolution does not need to write a separate gene for every anatomical detail because **the Coxeter lattice of permutations (`S_n`)** guarantees local swaps converge without local minima on `I(v)` (Claims 1–2); **kinematic speed differentials** automatically create phase-separated cell domains (Claim 3); **D’Arcy Thompson’s 8-knob coordinate fields** channel 150 million years of vertebrate body-plan evolution (Claim 4); and **Nernst–Planck electrodiffusion, Procrustes Lyapunov descent, and Stokes squirmer flow** create rewritable bistable voltage memory, craniofacial homeostasis, and kinematic self-replication (Claims 5–7).
2. **Where "Strong Dualist Ingress" Is 0.0% Needed (`0.0 / 11.7 pts`):**
   Describing Delayed Gratification or chimeric clustering as *"side-quests—not predicted, explained, or produced by that medium"* is where the language overreaches. Once you read line 74 of `InsertionSortCell.py`, measure permutations with their true pairwise inversion Lyapunov potential `I(v)` (`ΔI = −1` on 192/192 swaps, `0` uphill steps¹) instead of the lossy neighbor ruler `E(v)`, and test duplicate multiplicity at `C = 2` (`S B | S B`), **100% of the behavior is predicted, explained, and produced by the local transition matrix of the code.**

We don't need non-physical ghosts beaming into the machine, and we don't need reductionist dismissals that wave away real emergent phenomena without checking the math. All the tools to test the mathematics of life are sitting right at our fingertips.

---

### Footnotes
* **¹ On Erik Hoel’s "Causal Emergence" (2013/2026):** We also tested both rulers using Erik Hoel et al.’s Causal Emergence 1.0 (*PNAS* 2013) and 2.0 (*Patterns* 2026) formalization—a popular framework in the Tufts/Levin orbit for scoring macro-variables—where Ruler 1 `I(v)` scores `100%` determinism (`ΔCP = +0.1191 > 0` on active swaps) and Ruler 2 `E(v)` drops to `72.1%` (`ΔCP = −0.2525 < 0`). However, as we document in our **[Critical Audit of Causal Emergence 1.0 & 2.0 (`docs/CAUSAL_EMERGENCE_PRACTITIONERS_GUIDE.md`)](https://github.com/fractastical/morpholib/blob/main/docs/CAUSAL_EMERGENCE_PRACTITIONERS_GUIDE.md)**, neither we nor our Anthropic referee (`claude-sonnet-5-5-high`, which gave our initial CE write-up a `44/100` before we added the failure-mode analysis) are very convinced by the Causal Emergence formalization in practice: it suffers from `log₂(N)` normalization sensitivity that flips sign when no-op self-loops are included (`ΔCP = −0.0501`), manufactures artificial emergence from unreachable "Garden-of-Eden" states under uniform `do()` interventions, explodes at `N = 100! ≈ 10¹⁵⁸`, and is redundant when 2 lines of elementary combinatorics already prove `ΔI = −1` on every swap.

---

### Interactive Labs, Papers & Code for Readers
* **Bioelectricity Foundations (Galvani to Modern V_mem Prepatterns):** [https://bioelectricity.net/bioelectricity-foundations](https://bioelectricity.net/bioelectricity-foundations)
* **Play with the Live Sorting Morphogenesis, "Two Rulers" & `C = 2` Anti-Clustering Simulator:** [https://bioelectricity.net/sorting-morphogenesis](https://bioelectricity.net/sorting-morphogenesis)
* **Explore the 10,303-Species D’Arcy Thompson & Bioelectric Fish Morphospace:** [https://bioelectricity.net/bioelectric-morphospace](https://bioelectricity.net/bioelectric-morphospace)
* **Read Our Critical Audit of Causal Emergence 1.0 & 2.0 ("How It Works & Where It Breaks"):** [https://github.com/fractastical/morpholib/blob/main/docs/CAUSAL_EMERGENCE_PRACTITIONERS_GUIDE.md](https://github.com/fractastical/morpholib/blob/main/docs/CAUSAL_EMERGENCE_PRACTITIONERS_GUIDE.md)
* **Read the 4-Page Illustrated Methods & Provenance Guide (PDF):** [https://fractastical.github.io/morpholib/levin_latent_space_methods_and_provenance.pdf](https://fractastical.github.io/morpholib/levin_latent_space_methods_and_provenance.pdf)
* **Read the 8-Page Sorting Morphogenesis & Causal Emergence 2.0 Paper (PDF):** [https://fractastical.github.io/morpholib/zhang_levin_sorting_morphogenesis_manuscript.pdf](https://fractastical.github.io/morpholib/zhang_levin_sorting_morphogenesis_manuscript.pdf)
* **Read the 28-Page D’Arcy Thompson 1917 & 10,303-Species Paper (PDF):** [https://fractastical.github.io/morpholib/thompson_1917_fish_transformations_manuscript.pdf](https://fractastical.github.io/morpholib/thompson_1917_fish_transformations_manuscript.pdf)
* **Signed Production Nanopublications (Immutable Trusty URIs):**
  * **Sorting Morphogenesis, 3-Ablation Platonic Audit & Hoel CE 2.0 Claim Nanopub:** [`RAH6FDLaacsq28cjx2NVRYRANqOThwHaHwE58ByxqIWI8`](https://registry.knowledgepixels.com/np/RAH6FDLaacsq28cjx2NVRYRANqOThwHaHwE58ByxqIWI8) (Overall Assessment: [`RA_mMJXveSMNlxIjYu_oTtIY233QfJnNCf47yRLfInMuE`](https://registry.knowledgepixels.com/np/RA_mMJXveSMNlxIjYu_oTtIY233QfJnNCf47yRLfInMuE); Morphometric & CE 2.0 Validation: [`RAdFmYq_oVoHbK2eclzYgRWXJVhSNdiEQBfxp4nhvkzJk`](https://registry.knowledgepixels.com/np/RAdFmYq_oVoHbK2eclzYgRWXJVhSNdiEQBfxp4nhvkzJk))
  * **D’Arcy Thompson FishShapes v1 Pre-Unblinding Preregistration Nanopub:** [`RAilN4nbRHp3e3ZHqlbIgeL3UGPRV_KtQHtJD3gzVVdqw`](https://registry.knowledgepixels.com/np/RAilN4nbRHp3e3ZHqlbIgeL3UGPRV_KtQHtJD3gzVVdqw) (Post-Unblinding Outcome: [`RAJancwGOPEkHzBoZXGv_vu_snW0Lff2FktE85Me1P-fI`](https://registry.knowledgepixels.com/np/RAJancwGOPEkHzBoZXGv_vu_snW0Lff2FktE85Me1P-fI))
* **Project Delphi Knowledge-Graph Ontology:** [https://github.com/fractastical/project-delphi-ontology](https://github.com/fractastical/project-delphi-ontology)
* **Open-Source MorphoEval & Infinite Morphospace Repository (Python 3, C++20, Node.js, Lean 4 Proofs & Signed Nanopublications):** [https://github.com/fractastical/morpholib](https://github.com/fractastical/morpholib)
* **Errata & Cross-Provider Referee Audit Log (`docs/ERRATA_AND_REFEREE_AUDIT_LOG.md`):** [https://github.com/fractastical/morpholib/blob/main/docs/ERRATA_AND_REFEREE_AUDIT_LOG.md](https://github.com/fractastical/morpholib/blob/main/docs/ERRATA_AND_REFEREE_AUDIT_LOG.md)

---

### Primary Literature & References (with DOIs)

1. **Beane, W. S., Morokuma, J., Adams, D. S., & Levin, M. (2011).** A chemical genetics approach reveals H,K-ATPase-mediated membrane voltage is required for planarian head regeneration. *Chemistry & Biology* 18(1):77–89. [doi:10.1016/j.chembiol.2010.11.012](https://doi.org/10.1016/j.chembiol.2010.11.012)
2. **Bookstein, F. L. (1989).** Principal warps: thin-plate splines and the decomposition of deformations. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 11(6):567–585. [doi:10.1109/34.24792](https://doi.org/10.1109/34.24792)
3. **Cervera, J., Levin, M., & Mafé, S. (2021).** Morphology changes induced by intercellular gap junction blocking: a reaction–diffusion mechanism. *BioSystems* 209:104511. [doi:10.1016/j.biosystems.2021.104511](https://doi.org/10.1016/j.biosystems.2021.104511)
4. **Claverie, T., & Wainwright, P. C. (2014).** A morphospace for reef fishes: elongation is the dominant axis of body shape evolution. *Evolution* 68(12):3403–3418. [doi:10.1111/evo.12502](https://doi.org/10.1111/evo.12502)
5. **Dietz, J. (2026a).** *BioEval: Bioinformatics & Computational-Biology Paper Evaluation Rubric and Nanopublication Pipeline*. Zenodo. [doi:10.5281/zenodo.20567719](https://doi.org/10.5281/zenodo.20567719)
6. **Dietz, J. (2026b).** *NeuroEval: Evaluating Data Transparency, Reproducibility, and Neural-Coding Rigor in Computational and Biohybrid/Wetware Neuroscience Papers*. Zenodo. [doi:10.5281/zenodo.20622878](https://doi.org/10.5281/zenodo.20622878)
7. **Friedman, S. T., Price, S. A., Hoey, A. S., & Wainwright, P. C. (2019).** Body shape diversification along the benthic–pelagic axis in marine fishes. *Systematic Biology* 69(1):111–126. [doi:10.1093/sysbio/syz029](https://doi.org/10.1093/sysbio/syz029)
8. **Hoel, E. P., Albantakis, L., & Tononi, G. (2013).** Quantifying causal emergence shows that macro can beat micro. *Proceedings of the National Academy of Sciences USA* 110(49):19790–19795. [doi:10.1073/pnas.1314922110](https://doi.org/10.1073/pnas.1314922110)
9. **Hoel, E. P. (2026).** Quantifying emergent complexity (Causal Emergence 2.0). *Patterns* 7(1):101472. [doi:10.1016/j.patter.2025.101472](https://doi.org/10.1016/j.patter.2025.101472)
10. **Kriegman, S., Blackiston, D., Levin, M., & Bongard, J. (2020).** A scalable pipeline for designing reconfigurable organisms. *Proceedings of the National Academy of Sciences USA* 117(4):1853–1859. [doi:10.1073/pnas.1910837117](https://doi.org/10.1073/pnas.1910837117)
11. **Kriegman, S., Blackiston, D., Levin, M., & Bongard, J. (2021).** Kinematic self-replication in reconfigurable organisms. *Proceedings of the National Academy of Sciences USA* 118(49):e2112672118. [doi:10.1073/pnas.2112672118](https://doi.org/10.1073/pnas.2112672118)
12. **Levin, M. (2025/2026).** Ingressing minds: causal, non-physical patterns in-form natural, synthetic, and hybrid embodiments. *PsyArXiv* (`v3`, 2025) [doi:10.31234/osf.io/5g2xj_v3](https://doi.org/10.31234/osf.io/5g2xj_v3) & *Philosophies* 11(5):161 (2026). [doi:10.3390/philosophies11050161](https://doi.org/10.3390/philosophies11050161)
13. **Lobo, D., Malone, T. J., & Levin, M. (2013).** Planform: an application and database of graph-encoded planarian regenerative experiments. *Bioinformatics* 29(8):1098–1100. [doi:10.1093/bioinformatics/btt088](https://doi.org/10.1093/bioinformatics/btt088) (and *Biology Open* 2(2):156–169, [doi:10.1242/bio.20123400](https://doi.org/10.1242/bio.20123400))
14. **Lobo, D., Feldman, E. B., Shah, M., Malone, T. J., & Levin, M. (2014).** Limbform: a functional ontology-based database of limb regeneration experiments. *Bioinformatics* 30(24):3598–3600. [doi:10.1093/bioinformatics/btu582](https://doi.org/10.1093/bioinformatics/btu582)
15. **Lobo, D., Solano, M., Bubenik, G. A., & Levin, M. (2014).** A linear-encoding model explains the variability of the target morphology in regeneration. *Journal of The Royal Society Interface* 11(92):20130918. [doi:10.1098/rsif.2013.0918](https://doi.org/10.1098/rsif.2013.0918)
16. **Oviedo, N. J., Morokuma, J., Walentek, P., et al., & Levin, M. (2010).** Long-range neural and gap junction protein-mediated cues control polarity during planarian regeneration. *Developmental Biology* 339(1):188–199. [doi:10.1016/j.ydbio.2009.12.012](https://doi.org/10.1016/j.ydbio.2009.12.012)
17. **Perrard, A., Pickett, K. M., Villemant, C., Kojima, J., & Carpenter, J. (2013).** Phylogeny of hornets: a total evidence approach (Hymenoptera, Vespidae, Vespinae, *Vespa*). *Journal of Hymenoptera Research* 32:1–15. [doi:10.3897/JHR.32.4685](https://doi.org/10.3897/JHR.32.4685)
18. **Pezzulo, G., & Levin, M. (2015).** Re-membering the body: applications of computational neuroscience to the top-down control of regeneration of limbs and other complex organs. *Integrative Biology* 7(12):1487–1517. [doi:10.1039/c5ib00221d](https://doi.org/10.1039/c5ib00221d)
19. **Pietak, A., & Levin, M. (2016/2017).** Exploring instructive physiological signaling with the Bioelectric Tissue Simulation Engine (BETSE). *Frontiers in Physiology* 7:178 ([doi:10.3389/fphys.2016.00178](https://doi.org/10.3389/fphys.2016.00178)) & Bioelectric gene and reaction networks. *Journal of The Royal Society Interface* 14(134):20170425 ([doi:10.1098/rsif.2017.0425](https://doi.org/10.1098/rsif.2017.0425)).
20. **Pinet, K., Deolankar, M., Leung, B., & McLaughlin, K. A. (2019).** Adaptive correction of craniofacial defects in pre-metamorphic *Xenopus laevis* tadpoles involves thyroid hormone-independent tissue remodeling. *Development* 146(14):dev175950. [doi:10.1242/dev.175950](https://doi.org/10.1242/dev.175950)
21. **Rabosky, D. L., Chang, J., Title, P. O., et al. (2018).** An inverse latitudinal gradient in speciation rate for marine fishes. *Nature* 559(7714):392–395. [doi:10.1038/s41586-018-0273-1](https://doi.org/10.1038/s41586-018-0273-1)
22. **Thompson, D. W. (1917).** *On Growth and Form*. Cambridge University Press. [doi:10.5962/bhl.title.11332](https://doi.org/10.5962/bhl.title.11332) (and 1910 Oxford translation of Aristotle’s *Historia Animalium*, [doi:10.5962/bhl.title.147382](https://doi.org/10.5962/bhl.title.147382))
23. **Vandenberg, L. N., Adams, D. S., & Levin, M. (2012).** Normalized shape and location of perturbed craniofacial structures in the *Xenopus* tadpole reveal an innate ability to achieve correct morphology. *Developmental Dynamics* 241(5):863–878. [doi:10.1002/dvdy.23770](https://doi.org/10.1002/dvdy.23770)
24. **Zhang, T., Goldstein, A., & Levin, M. (2025).** Classical sorting algorithms as a model of morphogenesis: self-sorting arrays reveal unexpected competencies in a minimal model of basal intelligence. *Adaptive Behavior* 33(1):25–54. [doi:10.1177/10597123241269740](https://doi.org/10.1177/10597123241269740)

