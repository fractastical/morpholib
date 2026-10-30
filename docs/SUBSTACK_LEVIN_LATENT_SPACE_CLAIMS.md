# The Seven Claims, Walked Through

*Companion to [Why Michael Levin Is 77.6++% Validated on Platonic Ingress](https://fractastical.github.io/morpholib/substack_levin_latent_space_article.html). The scores are locked. This page is the audit that produced them.*

*Sixteen plates. Each caption names the script, dataset, or engraving it was built from. Return to the [main article](https://fractastical.github.io/morpholib/substack_levin_latent_space_article.html) for the score, the dualism essay, and the conclusion.*

---

## Claim 1: "No Error-Handling Code Was Added" to Self-Sorting Cells ([Zhang, Goldstein & Levin 2025](https://doi.org/10.1177/10597123241269740))

In [*Adaptive Behavior* (2025)](https://doi.org/10.1177/10597123241269740) and Section 6.3 / Figure 14 of [*Ingressing Minds* (2026)](https://doi.org/10.3390/philosophies11050161), Taining Zhang, Adam Goldstein, and Michael Levin asked a clever question: **What if there is no central controller (no "gym teacher") sorting an array of numbers from 1 to 100?** What if each number is an independent thread (an autonomous "cell") that can only look at its neighbors and decide on its own whether to swap places?

They gave the cells three classical computer-science "personalities" (**algotypes**):
* **Bubble cells (B):** Look at your immediate left or right neighbor (`p − 1` or `p + 1`). If you are out of order, swap!
* **Insertion cells (I):** Look at everyone to your left (`0 .. p − 1`). Only step left with `p − 1` if everyone to your left is already in order.
* **Selection cells (S):** Keep a mental finger (`ideal_position` / `target` pointer) starting at slot `0`, then `1`, then `2`... If your number is smaller than whoever is standing at `target`, jump straight there and swap with them; otherwise move your finger one slot to the right.

![](https://fractastical.github.io/morpholib/media/zhang_13yo_fig1_three_personalities.png)

*Plate I — The three algotypes as a line of numbered jerseys: Bubble (neighbor swap), Insertion (wait for a sorted left line, plus the* `prev = -1` *frozen-cell reset), and Selection (a long-range* `target` *pointer that jumps clean over a frozen brick-wall cell). Provenance: [`sorting-morphogenesis/render_13yo_illustrations.py`](https://github.com/fractastical/morpholib/blob/main/sorting-morphogenesis/render_13yo_illustrations.py), Illustrated Guide 1.*

Before testing the philosophical claims, we ported [their exact GitHub commit](https://github.com/Zhangtaining/cell_research/tree/1fd2bd5921c1f6b423a71f691d5189106a8a1020) into **four independent languages: Python 3, C++20, JavaScript (Node.js), and the Lean 4 theorem prover** (`0 sorry`).

First, credit where credit is due: **all 14 of the empirical curves published in Zhang, Goldstein & Levin (2025) replicate 100%**. Across Python, C++20, and Node.js, our ports produce bit-for-bit identical space-time kymographs (`0` cell mismatches) and match every benchmark in the paper.

![](https://fractastical.github.io/morpholib/media/zhang_sorting_morphogenesis.gif)

*Plate II — Space-time kymograph of a self-sorting array: each row is one time step, each column one cell, colour its value; the frozen defects are the vertical scars the sort must work around. Provenance: [`sorting-morphogenesis/run_experiments.py`](https://github.com/fractastical/morpholib/blob/main/sorting-morphogenesis/run_experiments.py), validated against the paper’s figures by [`validate_kymograph_and_paper.py`](https://github.com/fractastical/morpholib/blob/main/sorting-morphogenesis/validate_kymograph_and_paper.py).*

![](https://fractastical.github.io/morpholib/media/zhang_multilang_port_verification.png)

*Plate III — Four-language port verification. Python 3, C++20, and Node.js produce bit-for-bit identical kymographs (`0` cell mismatches) against the upstream commit; the Lean 4 column lists the theorems machine-checked with `0 sorry`. Provenance: [`sorting-morphogenesis/verify_multilang_ports.py`](https://github.com/fractastical/morpholib/blob/main/sorting-morphogenesis/verify_multilang_ports.py).*

Now look at the first philosophical claim Levin makes about this code in *Ingressing Minds* (Section 6.3, p. 22) when a few cells in the array are "frozen" (broken and refusing to move):

> *"The critical thing is that we did not add steps to the algorithm to check whether a swap had been completed, or to do specific things if an operation failed."*

**What we found in the code (Panel A):**
For **Bubble** sort, Levin’s statement is true. The only place [`BubbleSortCell.py`](https://github.com/Zhangtaining/cell_research/blob/1fd2bd5921c1f6b423a71f691d5189106a8a1020/modules/multithread/BubbleSortCell.py) looks at a `FREEZE` flag is to decline a swap with a frozen neighbor. The frozen cell is a brick wall, and the sorting rule is unchanged. **Selection** sort ([`SelectionSortCell.py`](https://github.com/Zhangtaining/cell_research/blob/1fd2bd5921c1f6b423a71f691d5189106a8a1020/modules/multithread/SelectionSortCell.py)) is the same: its pointer jumps straight across an immovable defect, and the one `FREEZE` branch in the upstream code only nudges that pointer past a frozen target so it does not get stuck. A skip, not a rewrite (rule `[U4]` in our port guide).

**Insertion sort is the counterexample.** Here the authors did not skip the frozen cell. They rewrote the algorithm’s own correctness test so it could see through one.

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

*Plate IV — The three-ablation Platonic-claims audit. Panel A: Insertion sort’s final error with and without the explicit* `FREEZE` *reset (`4.90 → 21.95`). Panel B: the two rulers,* `I(v)` *(strictly downhill, `0/192` uphill steps) versus* `E(v)` *(47 illusory uphill ticks). Panel C: algotype aggregation on unique keys and on duplicate keys with strict versus non-strict ties. Provenance: [`sorting-morphogenesis/audit_platonic_space_claims.py`](https://github.com/fractastical/morpholib/blob/main/sorting-morphogenesis/audit_platonic_space_claims.py); metrics locked in [`figures/platonic_claims_ablation_metrics.json`](https://github.com/fractastical/morpholib/blob/main/sorting-morphogenesis/figures/platonic_claims_ablation_metrics.json).*

> **Verdict — Claim 1: Partially Refuted.** Bubble and Selection sort genuinely get defect tolerance for free from decentralized execution (their only frozen-cell logic is a swap-precondition skip that treats a frozen cell as a wall); Insertion sort relies on an explicit 3-line `if status == FREEZE: prev = -1` exception handler that rewrites its own prefix-sorted test to see through that wall.

---

## Claim 2: "Delayed Gratification"

In *Ingressing Minds* (Section 6.3 and Figure 14A–B), Levin compares the sorting array to an agent walking around a wall. Two magnets separated by a piece of wood get stuck forever, because they cannot move against their gradient even for a moment. The paper’s error score ticks up before it comes down, and from that he argues the algorithm is *"temporarily going against its gradient to recoup greater gains later."*

Two things are wrong with that reading. Both are proved in Lean 4.

**No cell walks around the wall.** Bubble and Insertion cells only swap with the neighbor beside them. Freeze a cell at spot 20 and nothing to its left ever reaches its right; the two halves sort in isolation (Theorem 1, `bubble_adjacent_swap_preserves_left_partition`). Only Selection sort gets past, and only because its pointer is allowed to jump anywhere in the array.

**The rising score is the wrong ruler.** There are two ways to measure how scrambled a line is.

* **Ruler 1, `I(v)`, counts every pair** that is out of order anywhere in the line. This is the real energy. Every Bubble swap in the experiment lowers it by exactly 1: zero uphill steps out of 192 (the straight line in Panel B).
* **Ruler 2, `E(v)`, counts only neighbors.** It is the ruler the paper uses, and it can rise while the line gets closer to sorted. That is the jagged curve, 47 bumps.

Four cells are enough to see it. Start with `[2, 4, 1, 3]`. Ruler 1 sees three out-of-order pairs, `(2,1)`, `(4,1)`, and `(4,3)`. Ruler 2 sees one neighbor drop, `4 → 1`. Swap `4` with `1` and the line is `[2, 1, 4, 3]`. Ruler 1 falls from 3 to 2. Ruler 2 rises from 1 to 2, because one tall cliff broke into two small steps. No frozen cells are required.

![](https://fractastical.github.io/morpholib/media/zhang_13yo_fig2_two_rulers_illusion.png)

*Plate V — The two rulers, step by step, on `[2, 4, 1, 3] → [2, 1, 4, 3]`: every out-of-order pair drawn as an arch above (Ruler 1,* `I`*: `3 → 2`, strictly downhill) and the neighbor-step cliffs below (Ruler 2,* `E`*: `1 → 2`, the paper’s ruler going up while the cells move closer to sorted). Provenance: [`sorting-morphogenesis/render_13yo_illustrations.py`](https://github.com/fractastical/morpholib/blob/main/sorting-morphogenesis/render_13yo_illustrations.py), Illustrated Guide 2.*

> **Verdict — Claim 2: Optical Illusion of the Ruler.** The sorting cells never move against their dynamical gradient. They slide 100% deterministically downhill on their true Lyapunov energy `I(v)` (`ΔI = −1` on every single swap), while the observer watches them through a blurry neighbor-drop ruler `E(v)`.¹

---

## Claim 3: "Spontaneous Algotype Clustering"

Here the audit caught the critics in a bigger mistake than the dualists.

Mix Bubble cells and Selection cells half and half, in Figure 8 of Zhang et al. and Figure 14C of *Ingressing Minds*, and cells of the same personality group together — even though no line of code ever looks at a neighbor’s type. Levin called this an unprogrammed side-quest. Computer-science bloggers, Nathan Sweet among them, said it was a 1960s textbook fact: bubble sort does not swap equal keys, so duplicates stay in their original order.

Both halves of that rebuttal fail. Panel C is the measurement; Plate VI is the picture.

**Figure 8A has no duplicate numbers.** Every cell holds a unique value, so "duplicate-key stability" cannot explain anything, and the cells still cluster: aggregation rises from `0.50` to `0.624–0.640` mid-sort, then relaxes to `0.501`. The reason is speed. Selection cells jump small numbers to the front in a handful of swaps. Bubble cells shuffle one neighbor at a time. Sports cars pack the left; pedestrians get pushed into a wave on the right.

**On duplicate numbers, the critics predict the wrong direction.** If stability merely froze the starting order of equal keys, aggregation would finish at `0.50` no matter how many copies each number had. It does not. Selection scans from the left, so it takes the left half of every group of equals — `66%` of Selection cells at two copies, `75%` at ten. Two copies gives `S B | S B`, which is *less* clustered than chance (`0.477`). Ten copies gives `SSSSS | BBBBB`, a permanent clique at `0.659`. Allow equal keys to swap and the clique collapses back to chance (`0.497–0.514`, the dashed curve in Panel C).

![](https://fractastical.github.io/morpholib/media/zhang_13yo_fig3_sports_cars_vs_pedestrians.png)

*Plate VI — Sports cars versus pedestrians. Top: on unique numbers, fast Selection cells teleport left and compress slow Bubble cells into a contiguous wave. Bottom: on duplicate numbers, Selection packs the left half of every plateau, giving* `S B | S B` *anti-clustering at `C = 2` and* `SSSSS | BBBBB` *cliques at `C = 10`. Provenance: [`sorting-morphogenesis/render_13yo_illustrations.py`](https://github.com/fractastical/morpholib/blob/main/sorting-morphogenesis/render_13yo_illustrations.py), Illustrated Guide 3.*

And here are the real computer-experiment curves side by side—showing **(A)** the smooth downhill slide on Ruler 1 (`I(v)`, `0` uphill steps) vs. the 47 illusory bumps on Ruler 2 (`E(v)`), and **(B)** how the `C = 2` Anti-Clustering dip (`0.477 < 0.50`) and `C = 10` Clique peak (`0.659`) refute the blog critics’ flat `0.50` prediction:

![](https://fractastical.github.io/morpholib/media/zhang_13yo_fig4_real_data_curves.png)

*Plate VII — The measured curves. (A) Ruler 1* `I(v)` *slides smoothly downhill while Ruler 2* `E(v)` *shows its 47 illusory bumps. (B) Final `B + S` aggregation across copies-per-key `C`: the `C = 2` anti-clustering dip (`0.477`) and the `C = 10` clique peak (`0.659`) against the critics’ flat `0.50` prediction. Provenance: [`sorting-morphogenesis/render_13yo_illustrations.py`](https://github.com/fractastical/morpholib/blob/main/sorting-morphogenesis/render_13yo_illustrations.py), Illustrated Guide 4, from [`figures/platonic_claims_ablation_metrics.json`](https://github.com/fractastical/morpholib/blob/main/sorting-morphogenesis/figures/platonic_claims_ablation_metrics.json).*

> **Verdict — Claim 3: Empirically Vindicated (via Kinematic Phase Separation).** The clustering is 100% real and refutes the blog critics’ "textbook stability" dismissal—yet it is also 100% produced and predicted by the kinematic transport equations of the two algorithms without needing non-physical ghosts. ([Try the live `C = 2` Sorting Morphogenesis Simulator on bioelectricity.net here!](https://bioelectricity.net/sorting-morphogenesis))

---

## Claim 4: D’Arcy Thompson’s Morphospace Across 10,303 Fish

![](https://fractastical.github.io/morpholib/media/thompson_1906_moffat_portrait.jpg)

*Plate VIII — D’Arcy Wentworth Thompson (1860–1948) around 1906, a decade before* On Growth and Form. *Cabinet photograph by Moffat, Edinburgh. Provenance: University of St Andrews Libraries and Museums, ms48534-ph-2-5 (CC0, public domain).*

The second major pillar Levin invokes for his latent space is **D’Arcy Wentworth Thompson’s (1917)** famous coordinate transformations from *On Growth and Form*: the idea that nature doesn't redesign every bone from scratch when evolving a parrotfish (*Scarus*) into an angelfish (*Pomacanthus*) or a pufferfish (*Diodon*) into an ocean sunfish (*Mola mola*), but instead turns a few global **"growth knobs"** that bend the 2D coordinate grid of the body like a sheet of rubber.

![](https://fractastical.github.io/morpholib/guide_step1_1917_rubber_sheet.png)

*Plate IX — The still of the four families that open the main article. Thompson’s coordinate grids redrawn from* On Growth and Form *Figs. 373–382: oblique shear (Scarus → Pomacanthus), coaxial circles, the bilinear fan, and the radial flare (Diodon → Mola mola). Provenance: [`thompson/generate_13yo_visual_guide.py`](https://github.com/fractastical/morpholib/blob/main/thompson/generate_13yo_visual_guide.py), step 1.*

For the last 30 years, evolutionary biologists tried to test Thompson’s idea on computer landmark databases—and concluded he was mostly wrong across fish families. **Why did their tests fail?**

When we audited the morphometrics literature across **three independent datasets scaling up to 10,303 fish species (`440` families)**, we uncovered two methodological traps that had hidden Thompson’s law for a century:

1. **Trap #1 — The "17-Dot Head Seesaw":**
   Standard fish databases ([Claverie & Wainwright 2014](https://doi.org/10.1111/evo.12502); [Friedman et al. 2019](https://doi.org/10.1093/sysbio/syz029)) don't trace the whole outline of the fish—they measure **17 dots on the skeleton**, with **9 of those 17 dots (53%) packed inside the head and jaws** and **0% on the outer dorsal, anal, or tail fins**. Because jaws evolve rapidly for diet while the body evolves for swimming, fitting a global rubber sheet to those 17 dots lets the 9 head dots pull the back of the sheet the wrong way—making the warped fish silhouette look **worse than no warp at all** (`40.6% → 56.2%` mismatch on *Scarus* → *Pomacanthus*)!

![](https://fractastical.github.io/morpholib/guide_step2_17dot_head_trap.png)

*Plate X — The 17-dot head seesaw. Nine of the seventeen standard skeletal landmarks sit in the head and jaws and none on the outer fins, so a global rubber sheet fitted to them is pulled the wrong way at the back of the body. Provenance: [`thompson/generate_13yo_visual_guide.py`](https://github.com/fractastical/morpholib/blob/main/thompson/generate_13yo_visual_guide.py), step 2; landmarks from Claverie & Wainwright (2014) / Friedman et al. (2019).*

![](https://fractastical.github.io/morpholib/media/thompson_1917_silhouette_paradox_summary.gif)

*Plate XI — The silhouette paradox, animated: fitting the 17 skull-heavy landmarks makes the warped Scarus outline a worse match to Pomacanthus than no warp at all (`40.6% → 56.2%` mismatch), while fitting the 600-point outline directly recovers Thompson’s grid. Provenance: [`thompson/generate_summary_animation_gif.py`](https://github.com/fractastical/morpholib/blob/main/thompson/generate_summary_animation_gif.py), from [`thompson/1917-thompson-silhouette-area-paradox.py`](https://github.com/fractastical/morpholib/blob/main/thompson/1917-thompson-silhouette-area-paradox.py).*

2. **Trap #2 — The "Plain Box Stretch" Illusion vs. True Thompson Curves:**
   When we switch to a **600-point whole-body outline**, if you only ask a computer to overlap two *unlabeled black shadows*, a trivial **Rectangular Box Stretch** (stretching height vs. width with *zero* slant or curve) can "cheat" by stretching the middle of a pufferfish’s belly straight up to cover the back fins of an ocean sunfish!
   **Where Thompson’s curved and slanted grids (`70° Oblique Shear`, `Coaxial Circles`, `Bilinear Fan`, `Radial Flare`) are genuinely required** is when you force the grid to match **true homologous perimeter anatomy** (dorsal fin base & tip to dorsal fin base & tip, anal fin to anal fin, without the 9 internal skull dots): there, a plain Box Stretch fails (**`2.4%`** match on the swept-back fin tips of the angelfish), and **Thompson’s curved grids jump by `+13%` to `+48%` (`15×` better)** while staying **100% smooth and un-folded on `6 out of 6` pairs** (whereas standard 600-point Thin-Plate Splines ([Bookstein 1989](https://doi.org/10.1109/34.24792)) fold and crumple the interior tissue grid on `6 out of 6` pairs!).

![](https://fractastical.github.io/morpholib/guide_step3_box_cheat_vs_curves.png)

*Plate XII — The box-stretch cheat versus true Thompson curves. On unlabeled shadows a plain rectangular stretch can fake a match by pushing a pufferfish belly up over a sunfish fin; on homologous perimeter anatomy (fin base to fin base, fin tip to fin tip) it collapses to a `2.4%` match while the shear, coaxial, fan, and flare grids gain `+13%` to `+48%` and stay un-folded. Provenance: [`thompson/generate_13yo_visual_guide.py`](https://github.com/fractastical/morpholib/blob/main/thompson/generate_13yo_visual_guide.py), step 3.*

3. **Two rules, and a test we could not un-fail:**
   Three datasets — 791, then 6,614, then 10,303 species, across 440 families — say the same thing twice.

   **The body swims along Thompson’s sheet.** Over 150 million years, on the tree of [Rabosky et al. (2018)](https://doi.org/10.1038/s41586-018-0273-1), the direction of whole-body change lines up with his coordinate grids at **93.5% to 98.9%**. (This article uses two counts for those grids. Each 1917 family has 8 knobs; once position, rotation, and size are removed, the shared terms are the 7 modes used for prediction.) Alignment is not variance explained. On the standard 17-landmark data those 7 modes capture **57.9%** of between-species shape variance, against **86.2%** for a 7-component PCA trained on the data. Thompson’s grids are the highway, not the whole map, and 57.9% is the number every later clade is held to.

   **The head eats.** Sister species on the same reef share a swimming body and rewire jaws, snouts, and eyes **3.3× to 9.3×** faster than their bodies.

   **The blind test.** Before opening the Smithsonian’s FishShapes v1 — 5,180 species we had not seen — we locked the predictions in a [signed pre-registration](https://registry.knowledgepixels.com/np/RAilN4nbRHp3e3ZHqlbIgeL3UGPRV_KtQHtJD3gzVVdqw). Inside a 50-million-year horizon the Thompson predictor beat the clade consensus by a ridiculous margin (`p = 10⁻¹⁰³`). It did not beat the simplest rival. Copying the single nearest relative won by 4% on one noisy measurement (`p = 0.061`), and that hypothesis was rejected. The loss is in the same record as the win.

![](https://fractastical.github.io/morpholib/guide_step4_outside_vs_inside_rules.png)

*Plate XIII — The two rules of fish evolution. Outside: whole-body change across 440 families runs along Thompson’s coordinate highway (`93.5%–98.9%` alignment). Inside: sister species on the same reef rewire jaws, snouts, and eyes `3.3×–9.3×` faster than their bodies. Provenance: [`thompson/generate_13yo_visual_guide.py`](https://github.com/fractastical/morpholib/blob/main/thompson/generate_13yo_visual_guide.py), step 4; datasets of 791, 6,614, and 10,303 species.*

> **Verdict — Claim 4: Strongly Confirmed.** Across 10,303 species, Thompson’s and Levin’s core thesis—that macroevolution navigates a low-dimensional geometric coordinate manifold while local modules handle rapid ecological competition—is a quantitative law of vertebrate evolution. ([Explore the interactive 10,303-Species Fish Morphospace on bioelectricity.net here!](https://bioelectricity.net/bioelectric-morphospace))

---

## Claims 5–7: Planarians, Picasso Tadpoles, and Xenobots

How does a living tissue actually store a target coordinate in that morphospace? The three wet-lab results are valid. A planarian remembers a two-headed target after the drug is gone, a scrambled tadpole face moves back to the froglet layout along paths it was never shown, and a sculpted clump of frog skin assembles daughters with no change in its genome. What we built afterwards is a set of reduced equations, to test whether ordinary physics already produces those attractors. The equations reproduce them. The points held back below are about what the equations leave out, not about whether the experiments happened:

### Claim 5 — Two-headed planarians remember a voltage ([Oviedo et al. 2010](https://doi.org/10.1016/j.ydbio.2009.12.012); [Beane et al. 2011](https://doi.org/10.1016/j.chembiol.2010.11.012); [Pietak & Levin 2016](https://doi.org/10.3389/fphys.2016.00178)/[2017](https://doi.org/10.1098/rsif.2017.0425); [Cervera, Levin & Mafé 2021](https://doi.org/10.1016/j.biosystems.2021.104511))

Block the gap junctions of a cut planarian for 48 hours and it grows two heads. Wash the drug out, wait weeks, cut it again in plain water, and it grows two heads again. Nothing in the DNA changed.

The model says why, in one sentence. Serotonin is positively charged, so it drifts down the voltage from head (`−20 mV`) to tail (`−60 mV`) by a fixed factor of about `4.7`, and gap-junction conductance cancels out of that equilibrium ([`the model`](https://github.com/fractastical/morpholib/blob/main/models/2017-pietak-bioelectric-morphogen-coupling.py)). Hold the block long enough for the tail to flip to a head voltage, wash the drug out, and the worm is stuck in the two-headed basin.

![](https://fractastical.github.io/morpholib/media/cervera_levin_mafe_gj_block.gif)

*Plate XIV — Two-headed memory in a reaction–diffusion planarian. Transient gap-junction blockade lets the posterior pole flip from `−60 mV` to `−20 mV`; after washout the symmetric* `2H0T` *state is a stable attractor that survives re-cutting with no genetic change. Provenance: [`models/2021-Cervera–Levin–Mafe.py`](https://github.com/fractastical/morpholib/blob/main/models/2021-Cervera–Levin–Mafe.py) (1D) and [`models/2017-pietak-bioelectric-morphogen-coupling.py`](https://github.com/fractastical/morpholib/blob/main/models/2017-pietak-bioelectric-morphogen-coupling.py) (405-cell 2D Voronoi Scharfetter–Gummel).*

### Claim 6 — The Picasso tadpole finds its face ([Vandenberg, Adams & Levin 2012](https://doi.org/10.1016/j.ydbio.2012.01.027); [Pezzulo & Levin 2015](https://doi.org/10.1098/rsif.2015.0772); [Pinet et al. 2019](https://doi.org/10.1242/dev.175950))

Scramble a tadpole’s eyes, jaws, and nostrils and they do not crawl back along the paths they were pushed. They take new paths, faster the further they are from home, until the froglet face is normal. Block the enzymes that remodel tissue and they stop. A model that treats the normal face as a target the tissue descends toward (`dV/dt ≤ 0`) reproduces both the correction and the arrest.

![](https://fractastical.github.io/morpholib/media/xenopus_craniofacial_remap_growth.gif)

*Plate XV — The Picasso tadpole corrects its face. Scrambled eyes, jaws, and nostrils move along novel paths with speed proportional to their defect until the normal froglet layout is reached (`dV/dt ≤ 0`), and arrest when MMP remodeling is blocked by GM6001. Provenance: [`models/2019-bischof-xenopus-craniofacial-remap.py`](https://github.com/fractastical/morpholib/blob/main/models/2019-bischof-xenopus-craniofacial-remap.py), Pass 3 with 3D Maxwell cartilage rods and MMP-2/MMP-9 zymography fields.*

### Claim 7 — Xenobots copy themselves with no genetic change ([Kriegman, Blackiston, Levin & Bongard 2020](https://doi.org/10.1073/pnas.1910837117), [2021](https://doi.org/10.1073/pnas.2112672118))

Sculpt ciliated frog-skin spheroids into a C-shaped mouth of about `105°` and set them in a dish of loose cells. The cilia make currents, the currents sweep cells into the mouth, adhesion sticks those cells into a daughter, and the daughter does it again — three or four generations. A sphere manages one. A flat slab manages none. Both of our fluid simulations find the same angle: `105°` is where capture works.

![](https://fractastical.github.io/morpholib/media/xenobot_evolutionary_biobots_growth.gif)

*Plate XVI — Kinematic self-replication. A C-shaped ciliated spheroid (`θ ≈ 105°`) sweeps dissociated cells into its mouth pocket until adhesion binds them into a motile daughter, for `3–4` generations versus `1` for a sphere and `0` for a slab. Provenance: [`models/2020-kriegman-xenobot-evolutionary-biobots.py`](https://github.com/fractastical/morpholib/blob/main/models/2020-kriegman-xenobot-evolutionary-biobots.py), 8×8×7 voxel lattice and 2D regularized-Stokeslet squirmer.*

> **Verdict — Claims 5–7: The laboratory results are valid.** The two-headed memory, the face that finds itself, and the kinematic self-replication are published experimental results. A standard biophysical model reproduces each one, with a named attractor: bistable voltage memory, descent toward a target face, a hydrodynamic capture optimum. The `−2.2` points across the three are the gaps in our reduced equations. They are not a deduction against the experiments.

---
