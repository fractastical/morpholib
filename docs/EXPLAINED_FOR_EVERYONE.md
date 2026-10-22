# The Visual Guide for Everyone: How Nature Builds 10,000 Fishes & Sea Shells

**Primary Author:** Joel Dietz ([Infinite Morphospace Project](https://github.com/fractastical/morpholib))  
📄 **[Download the 5-Page Illustrated Visual Guide PDF (`docs/thompson_1917_visual_guide_for_everyone.pdf`)](./thompson_1917_visual_guide_for_everyone.pdf)** ([LaTeX source](./thompson_1917_visual_guide_for_everyone.tex))

This page (and the companion 6-page PDF above) explains our discoveries at a **smart 13-year-old level** using five separate, full-width, large-font visual diagrams with zero overlapping text and zero dense academic jargon. (For the full mathematical equations, Lean 4 proofs, and referee audits, see **[`docs/KEY_DISCOVERIES.md`](./KEY_DISCOVERIES.md)** and **[`docs/thompson_1917_fish_transformations_manuscript.pdf`](./thompson_1917_fish_transformations_manuscript.pdf)**.)

---

## Step 1 — D'Arcy Thompson's 1917 "Rubber Graph-Paper" Idea

In 1917, a Scottish scientist named **D'Arcy Thompson** drew a famous picture: he placed a drawing of a normal fish on a sheet of graph paper, bent the grid lines like a **sheet of rubber**, and showed that the stretched drawing turned into a totally different fish (like a tall angelfish or a giant round ocean sunfish).

He believed nature doesn't redesign every single bone from scratch when a new species evolves—instead, an embryo just turns a few global **"growth knobs"** that stretch, slant, or curve the whole body at once:

![Step 1 — D'Arcy Thompson's 1917 Rubber Graph-Paper Idea](./guide_step1_1917_rubber_sheet.png)

---

## Step 2 — Mistake #1: Why Computers Thought Thompson Was Wrong ("The 17-Dot Head Trap")

For the last 30 years, evolutionary biologists tried to test Thompson's idea using computer data—and concluded he was mostly wrong! **Why did their computer tests fail?**

Because instead of tracing the **whole outline of the fish**, standard fish databases only measure **17 dots on the skeleton**:
* **9 of those 17 dots ($53\%$) are crammed inside the head and jaws** (yellow circles in **Panel A** below).
* The other **8 dots ($47\%$)** only mark where the bases of the fins touch the body wall (green squares in **Panel A**).
* **Zero dots ($0\%$) are on the actual outer fins or tail!**

![Step 2 — Mistake #1: The 17-Dot Head Seesaw Trap](./guide_step2_17dot_head_trap.png)

> **Why It Failed:** Imagine trying to tailor a full-body wetsuit by measuring **9 points on someone's face** and **0 points on their arms or legs**. Because heads and fins evolve differently across fish families, forcing the rubber sheet to fit the 9 head dots acts like a heavy seesaw and swings the back of the sheet in the wrong direction (`56.2%` mismatch—worse than not stretching the fish at all)!

---

## Step 3 — Mistake #2: Why "Just Stretching a Box" Fails True Anatomy, and Why Thompson's Curves Win

When we switch from 17 skeleton dots to a **600-point outline** of the whole fish, we have to be careful of a second trap: *How do we know Thompson's **curved and slanted grids** (circles, fans, and flares) are really needed, instead of **just stretching a plain rectangle taller or wider ("Plain Box Stretch")**?*

![Step 3 — Mistake #2: Plain Box Stretch vs. Thompson's Curved Grids](./guide_step3_box_cheat_vs_curves.png)

1. **The "Shadow Cheat":** If you only ask a computer to overlap two **unlabeled black shadows**, a plain rectangular **Box Stretch** (no curves, no slant) can "cheat": on a pufferfish $\to$ ocean sunfish, it stretches the *middle of the pufferfish's belly* straight up and down to cover the *back fins* of the sunfish! And on a parrotfish $\to$ angelfish (**Panel A** above), a Box Stretch lifts the *middle of the back* straight up (red `X`), completely missing the swept-back fin tips of the angelfish (`only 2.4% match!`).
2. **Why Thompson's Curved Grids Win:** Once you tell the computer which part of the outline is the **Dorsal Fin Tip**, which part is the **Anal Fin Tip**, and where the fins attach along the back and belly (without the 9 internal skull dots), **a plain Box Stretch fails**, and **Thompson's curved and slanted grids jump by $+13\%$ to $+48\%$** (**Panel B** above, $15\times$ more accurate on the angelfish fin tips)!

---

## Step 4 — The Big Picture Across 10,303 Fish Species: Two Simple Rules of Evolution

To be sure our answer wasn't a fluke of just a few fish, we tested **three independent datasets** totaling **10,303 fish species across 987 genera and 440 families**. We found that fish evolution follows **two simple rules at the same time**:

![Step 4 — The Two Rules of Fish Evolution Across 10,303 Species](./guide_step4_outside_vs_inside_rules.png)

* **Rule 1 — The OUTSIDE Silhouette (How the Fish Swims):** Water doesn't flow around an invisible internal skeleton; it flows around the **whole outer silhouette** of the body and fins acting like an airplane wing. Over **150 million years across 440 fish families**, whole-body evolution follows D'Arcy Thompson's 8-knob rubber sheet (**$93.5\%$ to $98.9\%$ alignment!**). Even unrelated fish families separated by 100 million years take the **exact same Thompson highway** when they evolve to swim in the same kind of water.
* **Rule 2 — INSIDE the Head (What the Fish Eats):** Why did the 9 head dots break the global rubber sheet? Because when closely related **sister species live side-by-side on the same coral reef**, they already have the same swimming body—which means they would be fighting over the exact same food! To avoid competing, sister species rapidly rewire their **jaws, snouts, and eyes $3.3\times\text{ to }9.3\times$ faster than their bodies** ($85.4\%$ of within-genus evolution): one cousin evolves jaws to crunch coral, another sucks in tiny plankton, and another scrapes algae!

---

## Step 5 — The Final Scoreboard in Four Simple Numbers

![Step 5 — The Four-Number Scoreboard Across 10,303 Species](./guide_step5_10000_fishes_scoreboard.png)

| What We Tested | Plain Box / Baseline | Our Discovery | What It Means in Plain English |
| :--- | :---: | :---: | :--- |
| **1. Angelfish Swept Fin Tips** | `2.4%` *(Box Fails)* | **`35.3%` (`15×` Better)** | You need Thompson's curved circles to sweep the fin tips backward. |
| **2. Ocean Sunfish Fin Bases** | `56.0%` *(Box Only)* | **`97.2%` (`+41%` Jump)** | You need Thompson's exponential flare to match the giant tail-fins. |
| **3. 150-Myr Body Highway** | `4.6%` *(Local Noise)* | **`98.9%` *(Follows Grid!)* ** | Across 426 families, whole body plans evolve along Thompson's grid. |
| **4. Reef Sister-Species Jaws** | `14.6%` *(Slow Body)* | **`85.4%` (`9.3×` Jaw Speed!)** | Across 10,303 species, neighbors on the same reef rapidly rewire their jaws to share food. |

---

## Step 6 — The Real Archival Museum Plates & Field Photographs (All 9 of Thompson's 1917 Fishes)

Here are the actual historical museum lithographs (Monaco 1896 & 1911 deep-sea expeditions, US National Museum 1896, Bloch 1796, British Museum 1887) and NOAA/IFREMER field specimen photographs for all nine of D'Arcy Thompson's 1917 fishes, showing the extracted **600-point whole-body outline (cyan)** alongside the **9 gold head dots** and **8 pink fin-base dots**:

![Archival Museum Plates & Field Specimen Photographs for All Nine 1917 Taxa](./thompson_1917_archival_field_plates.png)
