# Danio margaritatus — 13.39 µm Turing wavelength (slot 3162)

Holocene (extant), geological age **0 Ma**. Phenomenon scale **1.339e-05 m** (13.39 µm, log10 = −4.873), tier `4_cellular_10um` (4/8), detail variant.

This is chronospecies **3/5** in the extant Danio set used by this loop (aesculapii → albolineatus → **margaritatus** → nigrofasciatus → rerio), lineage Danionin Pigment Evolution Series. The predecessor is *Danio albolineatus* (uniform interstripe). The successor is *Danio nigrofasciatus* (broken ventral stripes). Age does not change between them. What changes at this cellular rung is whether a Turing mode is unstable and which wavelength it selects.

The focal law is the Turing critical wavenumber of a Schnakenberg activator–substrate system,

```
k_c^2 = gamma (D_v f_u + D_u g_v) / (2 D_u D_v),    lambda_c = 2 pi / k_c
```

with zero-flux boundaries on the cellular patch. For *D. margaritatus*, parameters are chosen so that `lambda_c` is exactly 1.339e-05 m and the mode is an isotropic spot (pearl / ocellus) pattern. For *D. albolineatus* the diffusion ratio sits below the Turing threshold, so `sigma(k) < 0` at every wavenumber. For *D. nigrofasciatus* the selected wavelength is longer (17.80 µm) and inhibitor diffusion is biased along the anterior–posterior axis.

The pinned photograph is a CC BY-SA 3.0 lateral view (Cisamarc, Wikimedia Commons, `File:Danio margaritatus (zoom).jpg`). Adult pearl spacing on that plate is hundreds of micrometres once the 18 mm catalog length is applied to the pearl-field axis; the 13.39 µm mode is the elementary cellular wavelength and is not resolved by the photograph.

Run:

```bash
python3 models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py
```
