# Mollusc Aperture Spine Buckling & Hierarchical Fronding Reference Dataset

## Overview

This dataset vendors the mathematical, biomechanical, and empirical morphometric reference parameters for **Slot 077 (`slot-077-mollusc-aperture-spine-buckling`)**, modeling hierarchical aperture spine folding, tubular vs. open-gutter loop self-contact closure, secondary/tertiary fractal frond serrations (*Chicoreus ramosus*), and episodic triradiate varix comb periodicity (*Murex pecten*) in muricid gastropods:

1. **Moulton, D. E., Goriely, A., & Chirat, R. (2015).** *The role of mechanics in the growth and form of mollusc shells: hierarchical spine folding and secondary fractal serrations.* Journal of Theoretical Biology, 364, 420–430. DOI: [`10.1016/j.jtbi.2014.09.021`](https://doi.org/10.1016/j.jtbi.2014.09.021)
2. **Chirat, R., Moulton, D. E., & Goriely, A. (2013).** *Mechanical basis of morphogenesis and convergent evolution of spiny seashells.* Proceedings of the National Academy of Sciences, 110(15), 6015–6020. DOI: [`10.1073/pnas.1220443110`](https://doi.org/10.1073/pnas.1220443110)
3. **Moulton, D. E., Goriely, A., & Chirat, R. (2012).** *Mechanical growth and morphogenesis of seashells.* Journal of Theoretical Biology, 311, 69–79. DOI: [`10.1016/j.jtbi.2012.07.009`](https://doi.org/10.1016/j.jtbi.2012.07.009)
4. **Rudraraju, S., Moulton, D. E., Chirat, R., & Goriely, A. (2019).** *A computational framework for the morpho-elastic development of molluskan shells by surface and volume growth.* PLoS Computational Biology, 15(7), e1007213. DOI: [`10.1371/journal.pcbi.1007213`](https://doi.org/10.1371/journal.pcbi.1007213)

## Governing Equations

1. **Multi-Generational Hierarchical Wavenumber Cascade (`moulton_secondary_frond_cascade`):**
   Along the generative aperture margin or along the arched crest of an already buckled primary spine, an elastic mantle strip of thickness $h_i$ (flexural rigidity $D_i = E h_i^3 / 12$) adhered to the calcified shell edge (foundation stiffness $K_s$) buckles with characteristic wavenumber:
   $$k_i^* = \left(\frac{3 K_s}{E h_i^3}\right)^{1/4} \propto h_i^{-3/4}$$
   When distal mantle thinning reduces thickness from primary ($h_1$) to secondary ($h_2 < h_1$) and tertiary ($h_3 < h_2$) margins, the wavenumber cascades upward:
   $$\frac{k_2^*}{k_1^*} = \left(\frac{h_1}{h_2}\right)^{3/4} \approx 3.8 \quad (h_1/h_2 \approx 5.93), \qquad k_3^* > k_2^* > k_1^*$$
   generating the fractal-like secondary and tertiary frond serrations of *Chicoreus ramosus*.

2. **Kirchhoff–Clebsch Elastica Loop Folding & Tubular Spine Closure Criterion:**
   Under incremental mantle growth stretch $\gamma > \gamma_c$, the delaminated mantle loop satisfies the planar Kirchhoff elastica with unilateral foundation resistance:
   $$E I \frac{\mathrm{d}^2\theta}{\mathrm{d}S^2} + F_x \sin\theta - F_y \cos\theta - K_u(y) = 0$$
   When excess growth exceeds the folding self-contact threshold ($\gamma > \gamma_{\mathrm{fold}}$), the neck gap $\Delta x_{\mathrm{neck}}(\gamma) \le 0$ closes at the base to seal a hollow tubular spine (*Murex pecten*), whereas distal tip softening or lower base compression leaves an open gutter or flared frond apex (*Chicoreus ramosus*).

3. **Episodic Triradiate Varix Growth Cycle Law:**
   Across whorl accretion angle $\phi \in [0, 2\pi N_{\mathrm{whorls}}]$, mantle growth excess oscillates episodically with three rest-period varix bursts per whorl ($\Delta\phi_{\mathrm{varix}} = 120^\circ = 2\pi/3$):
   $$A_{\mathrm{spine}}(\gamma) = \begin{cases} 0 & \text{if } \gamma(\phi) \le \gamma_c \quad \text{(smooth inter-varix whorl)}, \\ A_0 \sqrt{\gamma(\phi) - \gamma_c} & \text{if } \gamma(\phi) > \gamma_c \quad \text{(spiny varix burst)}. \end{cases}$$

## Vendored Specimen

- `murex_pecten_aperture_specimen.jpg`: Venus comb murex (*Murex pecten*, family Muricidae) shell specimen exhibiting triradiate body-whorl varices and 16 prominent periodic hollow tubular primary spines (plus interspersed secondary spines) along the straight siphonal canal.
  - **Source:** Wikimedia Commons [`File:Murex pecten (Venus comb murex snail) 2 (24962834576).jpg`](https://commons.wikimedia.org/wiki/File:Murex_pecten_(Venus_comb_murex_snail)_2_(24962834576).jpg) (James St. John)
  - **License:** CC BY 2.0
  - **Dimensions:** `343 x 800` px
