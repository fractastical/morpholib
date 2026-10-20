# Scleractinian Coral Calice Septa Insertion & Advection-Diffusion Morphogenesis Reference Dataset (`datasets/coral_calice_septa_insertion/`)

## Overview

This dataset vendors the empirical reference measurements, governing mathematical equations, literature citations, and biological specimen imagery for **Slot 074 (`slot-074-coral-calice-septa-insertion`)** of the 100-Loop Infinite Morphospace & MorphoEval Campaign, modeled by `2013-thomas-coral-calice-septa-insertion.py`.

## Primary Literature Sources

1. **Milne-Edwards, H. & Haime, J. (1848).** *Recherches sur les polypiers. Premier mémoire: Observations sur la structure et le développement des polypiers en général.* Annales des Sciences Naturelles, Zoologie (Série 3) **9**: 37–89. DOI: [`10.5962/bhl.title.11574`](https://doi.org/10.5962/bhl.title.11574)
   - Establishes the foundational **Milne-Edwards & Haime hexameral septal insertion law** for Scleractinian stony corals (Hexacorallia): 6 primary protosepta ($S_1$, $N_1 = 6$) partition the calice into 6 primary exocoelic/entocoelic chambers, and each subsequent developmental cycle $k \ge 2$ inserts $n_k = 6 \cdot 2^{k-2}$ new septa simultaneously at the midpoints of all existing interseptal loculi once the expanding calice perimeter $2\pi R(t)$ exceeds $N_k \lambda_s$, yielding the geometric doubling progression $N_{\text{hex}}(k) = 6 \cdot 2^{k-1} \in \{6, 12, 24, 48, 96, 192\}$ with strictly decreasing radial reach $L_1 > L_2 > L_3 > L_4 > L_5$.
2. **Kunth, A. (1869).** *Beiträge zur Kenntniss fossiler Korallen. 2. Das Wachsthumsgesetz der Zoantharia rugosa und über Calceola sandalina.* Zeitschrift der Deutschen Geologischen Gesellschaft **21**: 647–688. DOI: [`10.1127/zdgg/21/1869/647`](https://doi.org/10.1127/zdgg/21/1869/647)
   - Formulates **Kunth's Law** for Paleozoic Rugosa (Tetracorallia): following 6 initial protosepta (cardinal, counter, 2 alar, 2 counter-lateral), metasepta are inserted serially in only 4 insertion zones (2 cardinal-side and 2 alar fossulae), producing the arithmetic progression $N_{\text{rugosa}}(m) = 6 + 4m \in \{6, 10, 14, 18, 22, \dots\}$ and pinnate bilateral symmetry contrasting sharply with Scleractinian cyclic doubling.
3. **Kaandorp, J. A. & Kübler, J. E. (2001).** *The Algorithmic Beauty of Seaweeds, Sponges and Corals.* Springer-Verlag, Berlin. DOI: [`10.1007/978-3-662-04339-4`](https://doi.org/10.1007/978-3-662-04339-4)
   - Develops boundary-layer nutrient/bicarbonate diffusion-limited accretion models for sessile marine calcifiers.
4. **Chindapol, N., Kaandorp, J. A., Cronemberger, C., Mass, T. & Genin, A. (2013).** *Modelling Growth and Form of the Scleractinian Coral Pocillopora verrucosa and the Influence of Hydrodynamics.* PLoS Computational Biology **9**(1): e1002849. DOI: [`10.1371/journal.pcbi.1002849`](https://doi.org/10.1371/journal.pcbi.1002849)
   - Couples Navier-Stokes laminar flow ($\mathbf{u}$) with advection-diffusion of dissolved inorganic carbon / calcification precursors ($\mathbf{u}\cdot\nabla c = D\nabla^2 c$, Péclet number $\text{Pe} = U L / D$), demonstrating that purely diffusive radial symmetry at $\text{Pe} = 0$ breaks into upstream-enhanced skeletal accretion as thin diffusive boundary layers on the upstream face ($\theta = 0$) steepen concentration gradients $|\nabla c \cdot \mathbf{n}|$ relative to the downstream wake ($\theta = \pi$).
5. **Cairns, S. D. & Kitahara, M. V. (2012).** *An illustrated key to the genera and subgenera of the Recent azooxanthellate Scleractinia (Cnidaria, Anthozoa), with an attached glossary.* ZooKeys **227**: 1–47. DOI: [`10.3897/zookeys.227.3612`](https://doi.org/10.3897/zookeys.227.3612)
   - Provides the high-resolution taxonomic monograph micrograph of *Caryophyllia (Caryophyllia) quadragenaria* vendored in this dataset.

## Vendored Files & SHA-256 Provenance

| File | Description | License | SHA-256 |
| :--- | :--- | :--- | :--- |
| `scleractinian_coral_septa_specimen.jpg` | *Caryophyllia (Caryophyllia) quadragenaria* calice (top view) & costate trochoid corallum (side view), ZooKeys 227:1-47 (442x987 JPEG) | CC BY 3.0 | `904035c93684336102acb95bced6a16c1dd720e46092b19c0331837698c6bffa` |
| `coral_calice_septa_insertion_reference.json` | Governing equations, Milne-Edwards vs Kunth tables, Péclet sweep benchmarks, and specimen calibration | CC BY 4.0 | Verified at runtime |
| `scout_manifest.json` | Step 1 OpenAlex & Wikimedia Commons scout dossier for Slot 074 | CC BY 4.0 | `325761c7cb0d58db893172b965c87bf252c244e60c99ffcfb73e418483e41b58` |
| `paper_citations.bib` | BibTeX citations for Milne-Edwards & Haime (1848), Kunth (1869), Kaandorp & Kübler (2001), Chindapol et al. (2013), and Cairns & Kitahara (2012) | CC BY 4.0 | `156b2c3cca2ef939f13d198c6640bd82ba2108c1ef481d8a8da0868522173eff` |
