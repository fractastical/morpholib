# *Drosophila novamexicana* — 1.9 µm Turing wavelength (slot 3195, detail s0)

Holocene (extant), **0 Ma**. Characteristic scale **1.9 µm** (`1.901e-06 m`, log10 = −5.721, tier 3/8 `3_subcellular_um`).

This slot is the subcellular detail variant of the Turing pigment-stripe / ocellus wavelength for the Southwestern desert riparian fly (*Drosophila novamexicana*), not the 141 µm tergite-stripe slot. The question at this scale is which reaction–diffusion wavenumber \(k_c\) selects a ~1.9 µm pigment-granule spacing on the abdominal cuticle, and how that wavenumber differs from the chronoseries neighbors *D. mojavensis* (predecessor) and *D. orena* (successor).

## Model

Schnakenberg (1979) trimolecular kinetics

\[
f = a - u + u^{2}v, \qquad g = b - u^{2}v
\]

with Turing’s (1952) dispersion relation and Murray’s (1981) reaction-rate scale \(\gamma\):

\[
\partial_t u = \gamma f(u,v) + D_u\nabla^{2}u, \qquad
\partial_t v = \gamma g(u,v) + D_v\nabla^{2}v.
\]

The critical wavenumber satisfies

\[
k_c^{2} = \gamma\frac{D_v f_u + D_u g_v}{2 D_u D_v}, \qquad \lambda_c = 2\pi/k_c.
\]

Shared diffusion ratio \(\delta = D_v/D_u = 24\), \(a = 0.10\), \(b = 0.90\). Species differ only in \(\gamma\):

| Chronoseries neighbor | \(\gamma\) | \(\lambda_c\) |
| --- | ---: | ---: |
| *D. mojavensis* (predecessor, darker cactophilic cuticle) | 1.79 | 1.421 µm |
| ***D. novamexicana* (this slot, pale ebony/tan abdomen)** | **1.00** | **1.901 µm** |
| *D. orena* (successor, melanogaster-subgroup endemic) | 2.60 | 1.179 µm |

\(\lambda_c(\gamma) = 1.901\,\mu\mathrm{m}/\sqrt{\gamma}\). The pale cis-regulatory state (reduced abdominal melanin reaction rate) lengthens the selected wavelength relative to both neighbors. These \(\gamma\) values are a reconstruction at the assigned 1.9 µm scale. They are not measurements copied from the 96 µm *D. mojavensis* or 89 µm *D. orena* sibling slots, and they are not reported as such by Wittkopp et al. (2009) or Lamb et al. (2020).

## Sources

- Turing 1952, *Phil. Trans. R. Soc. B* 237:37–72. DOI `10.1098/rstb.1952.0012`
- Schnakenberg 1979, *J. Theor. Biol.* 81:389–400. DOI `10.1016/0022-5193(79)90042-0`
- Murray 1981, *J. Theor. Biol.* 88:161–199. DOI `10.1016/0022-5193(81)90334-9`
- Wittkopp, Stewart, Arnold et al. 2009, *Science* 326:540–544. DOI `10.1126/science.1176980`
- Lamb, Wang, Simmer, Chung & Wittkopp 2020, *Front. Ecol. Evol.* 8:184. DOI `10.3389/fevo.2020.00184`

Wikimedia Commons returned no CC-licensed photograph of *D. novamexicana*. The vendored specimen is the repository species-gallery asset `Drosophila_novamexicana.jpg` (tan optical-density field), SHA-256 pinned in the reference JSON. It is the comparison image for this slot, not a scanning-electron micrograph of pigment granules.

## Run

```bash
python3 models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py
```

Formalization: `morphoeval/lean/loops/Loop_3195_drosophila_novamexicana_turing_wavelength_s0.lean` (`MathOfBiology.Loop_3195`).
