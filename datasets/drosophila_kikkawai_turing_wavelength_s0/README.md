# Drosophila kikkawai — Turing wavelength, detail scale (slot 3190)

Holocene (Extant), **0 Ma**. Characteristic scale **2.3 µm** (`2.343e-06 m`, `log10 = -5.630`, tier 3/8 `3_subcellular_um`, variant `s0`).

This slot is the subcellular carrier of tergite / ocellus pigment in *Drosophila kikkawai* (ananassae–montium clade; Drosophila chronoseries 19/41). It is not the 93 µm cellular-band slot (`yassin-drosophila-kikkawai-turing-wavelength`).

## What is modelled

Schnakenberg (1979) activator–substrate kinetics on a periodic patch, with the Turing wavenumber \(k_c\) chosen so the most unstable wavelength is 2.343 µm:

\[
\partial_t u = D_u \nabla^2 u + a - u + u^2 v, \qquad
\partial_t v = D_v \nabla^2 v + b - u^2 v.
\]

Focal parameters (not copied from the predecessor or successor slots): \(a = 0.070\), \(b = 0.940\), \(D_u = 0.020\), \(D_v = 1.364\) (ratio 68.2). Predecessor *D. hydei* is given a longer-range inhibitor (\(D_v = 4.470\), \(\lambda \approx 3.056\,\mu\mathrm{m}\)). Successor *D. mauritiana* is given a shorter-range inhibitor (\(D_v = 0.413\), \(\lambda \approx 1.820\,\mu\mathrm{m}\)). At fixed focal diffusivities the major-gene axis of Gibert, Moreteau & David (1999) is reconstructed as a shift in \(b\): light morph \(b = 0.710\) lengthens \(\lambda\), dark morph \(b = 1.180\) shortens it.

## What is not claimed

Gibert et al. (1999) measured phenotypic plasticity of abdominal pigmentation (major gene × sex × segment × temperature). They did not publish a 2.3 µm wavelength. Yassin et al. (2016, *Current Biology*) identified *pdm3* as a hotspot for female-limited colour dimorphism; this script does not re-identify that locus. The 2.343 µm length is the MorphoEval detail-tier assignment for this slot. Wikimedia Commons had no *D. kikkawai* photograph on 2026-10-27 (the token “kikkawai” hit unrelated butterfly plates). The pinned plate is the repository gallery image, SHA-256 `1025334f37e7770aefc4092e3b6ee7b0e3c86c6df76def8da35bcfd263fe8518`.

## Run

```bash
python3 models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py
```

Outputs: `docs/media/drosophila_kikkawai_turing_wavelength_s0_comparison.png` and `docs/media/drosophila_kikkawai_turing_wavelength_s0_growth.gif`.
