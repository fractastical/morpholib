# Dataset: Anisotropic Wing-Geometry Gierer–Meinhardt Reaction–Diffusion & Batesian Mimicry Polymorphism (`papilio_wing_stripe_mimicry`)

## Primary Literature & DOIs

1. **Sekimura T, Madzvamuse A, Wathen AJ, Maini PK (2000)**.
   *A model for colour pattern formation in the butterfly wing of Papilio dardanus.* **Proceedings of the Royal Society B: Biological Sciences** 267(1446):851–859.
   DOI: [`10.1098/rspb.2000.1081`](https://doi.org/10.1098/rspb.2000.1081)
2. **Madzvamuse A, Thomas RDK, Sekimura T, Maini PK (2002)**.
   *A numerical approach to the study of spatial pattern formation in the ligaments of arcoid bivalves and on the wing of the butterfly Papilio dardanus.* **Journal of Mathematical Biology** 44(5):489–509.
   DOI: [`10.1007/s002850100133`](https://doi.org/10.1007/s002850100133)
3. **Timmermans MJTN, Baxter SW, Clark R, ..., Thompson MJ, Jiggins CD, Vogler AP (2014)**.
   *Comparative genomics of the mimicry switch in Papilio dardanus.* **Nature** 507(7490):96–99.
   DOI: [`10.1038/nature12979`](https://doi.org/10.1038/nature12979)
4. **Gierer A, Meinhardt H (1972)**.
   *A theory of biological pattern formation.* **Kybernetik** 12(1):30–39.
   DOI: [`10.1007/BF00289234`](https://doi.org/10.1007/BF00289234)
5. **Nijhout HF (1991)**.
   *The Development and Evolution of Butterfly Wing Patterns.* **Smithsonian Institution Press**, Washington, DC.
   DOI: [`10.5962/bhl.title.119515`](https://doi.org/10.5962/bhl.title.119515)

## Governing Mathematical & Developmental Framework

Female *Papilio dardanus* (the African mocker swallowtail) exhibit striking female-limited Batesian mimicry polymorphism controlled by the *H* (*engrailed*) locus (`Timmermans/Thompson et al. 2014`), producing distinct wing colour patterns (*cenea*, *hippocoonides*, *trophonius*, *planemoides*) that mimic unpalatable Danainae (*Amauris*) and Acraeinae (*Bematistes*) models.

Sekimura, Madzvamuse, Wathen & Maini (2000) and Madzvamuse et al. (2002) demonstrated that all four major female mimetic morphs emerge from a single 2D anisotropic Gierer–Meinhardt activator–inhibitor reaction–diffusion system on the curved butterfly wing domain $\Omega \subset \mathbb{R}^2$:

$$\frac{\partial u}{\partial t} = \gamma \left(a - b u + \frac{u^2}{v(1 + \kappa u^2)}\right) + \nabla \cdot (\mathbf{D}_u \nabla u), \qquad \frac{\partial v}{\partial t} = \gamma (u^2 - v) + \nabla \cdot (\mathbf{D}_v \nabla v),$$

where:
1. **Homogeneous Steady State & Turing Instability (`Sekimura et al. 2000 Eqs. 2.1–2.4`):**
   With zero activator saturation ($\kappa = 0$), the unique positive homogeneous steady state satisfies $v_0 = u_0^2$ and $u_0 = (a + 1)/b$. The Jacobian $\mathbf{J}_0 = \begin{pmatrix} f_u & f_v \\ g_u & g_v \end{pmatrix}$ has entries $f_u = \frac{b(1 - a)}{a + 1}$, $f_v = -\left(\frac{b}{a+1}\right)^2$, $g_u = \frac{2(a+1)}{b}$, and $g_v = -1$, satisfying $\text{tr}(\mathbf{J}_0) = f_u + g_v < 0$, $\det(\mathbf{J}_0) = b > 0$, and diffusion-driven Turing instability when $d f_u + g_v > 2\sqrt{d \det(\mathbf{J}_0)}$.
2. **Anisotropic Vein-Directed Diffusivity Tensor (`Sekimura et al. 2000 Section 2(b)`):**
   Wing tracheae and veins induce directional diffusivity $\mathbf{D}_u = \text{diag}(D_{u\parallel}, D_{u\perp}) = D_{u\parallel} \text{diag}(1, \epsilon)$ with anisotropy ratio $\epsilon = D_\perp / D_\parallel$, selecting between proximodistal (longitudinal) stripes when $\epsilon \ll 1$ and anteroposterior (transverse) bands (*planemoides*) when $\epsilon \ge 1$.
3. **Three-Parameter Morphospace Switching Across Mimetic Morphs (`Sekimura et al. 2000 Table 1, Figs. 3–6`):**
   Varying (i) the melanin pigmentation threshold $u_{\text{th}}$ ($u(\mathbf{x}) > u_{\text{th}} \Rightarrow$ melanic black scale; $u(\mathbf{x}) \le u_{\text{th}} \Rightarrow$ light/coloured patch), (ii) the proximal-to-distal boundary gradient slope $\alpha_{\text{bc}}$ along anterior/posterior wing margins, and (iii) the aspect/anisotropy ratio $\epsilon = D_\perp / D_\parallel$ transitions the pattern across *hippocoonides* (broad diagonal white patch) $\to$ *trophonius* (central brick-red/orange median patch with isolated subapical spot) $\to$ *cenea* (fragmented discal/submarginal spots on melanic ground) $\to$ *planemoides* (transverse median band).

## Vendored Files

- `scout_manifest.json` — Literature & specimen discovery manifest for Slot 099.
- `paper_citations.bib` — BibTeX citations with DOIs for Sekimura et al. (2000), Madzvamuse et al. (2002), Timmermans/Thompson et al. (2014), Gierer & Meinhardt (1972), and Nijhout (1991).
- `papilio_wing_stripe_mimicry_reference.json` — Quantitative reference tables (Gierer–Meinhardt kinetic parameters, 4 female mimetic morph specifications, 6-point threshold $u_{\text{th}}$ melanic area sweep, 5-point anisotropy $\epsilon$ mode-switching sweep, negative controls, and SHA-256 digests).
- `papilio_dardanus_cenea_specimen.jpg` — CC BY-SA 4.0 photograph of a female *Papilio dardanus cenea* mimetic morph in Maputo, Mozambique (`File:Mocker swallowtail (Papilio dardanus cenea) female form cenea Maputo.jpg`, 960×640 px, Charles J. Sharp).
