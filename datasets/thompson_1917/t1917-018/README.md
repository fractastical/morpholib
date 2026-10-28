# `datasets/thompson_1917/t1917-018/` — Thompson (1917) claim T1917-018: cells meet three-at-a-time at ~120°

Reference package for `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py`
(loop `2026-10-28-t1917-018-the-forms-of-tissues-or-cell-a-loop`, slot 213 of the Thompson 1917 claim campaign).

**Claim (Thompson 1917, Ch. VII, pp. 293–345, Figs. 100–135):** *"Cells in contact, like soap films, meet
three-at-a-time at ~120 degrees and in other respects obey Plateau's laws of film configuration."*

## Step 1 — dataset verification (not a search)

| Candidate | What it is | Match? |
|---|---|---|
| **doi:10.5061/dryad.mn8cs** (pre-scouted Dryad hit) | Fura-2 ratiometric Ca²⁺ JPEG sequences (red/green/ratio) of cultured normal human urothelial monolayers under ATP or scratch wounding (2 zips, 568 MB + 604 MB) | **DATASET_MISMATCH** — cytosolic calcium frames carry no junction (cell-boundary) label, so tri-cellular junction angles cannot be scored. Byte download was additionally blocked (Anubis anti-bot challenge, HTTP 403/401) from the cloud VM; the Dryad API metadata and Dryad's own MD5 digests are vendored in `dryad_mn8cs_*_metadata.json`. Not used for the verdict. |
| **Thompson's own 1917 figures** (sanctioned fallback) | IA scan `ongrowthform1917thom`, jp2 leaf = printed page + 20: Fig. 103 *Parenchyma of Maize* (p. 298, leaf 318) and Fig. 124 *Epidermis of Girardia* (p. 321, leaf 341) | Match (these are the claim's own exemplars). Leaves are vendored byte-identical; figure crops are PNGs with crop boxes recorded. |
| **Real specimens (Wikimedia Commons, CC BY-SA 4.0)** | Human corneal endothelium in situ (isotropic monolayer, 1290×1031 px, ~150 cells) and *Allium cepa* bulb epidermis (anisotropic/elongated cells, 1600 px derivative of a 5113 px original) | Match in variable (visible cell-border network, scorable tri-junctions); the claim is species-agnostic. Flagged as **modern specimens**, not Thompson's figures. |

## Contents and SHA-256 digests

| File | SHA-256 | Notes |
|---|---|---|
| `human_corneal_endothelium_specimen.jpg` | `9daed5fb546e11bfdc60df0c8d9665515a93498b5f222befc8ad4bd9dfa6fb5a` | byte-identical to Commons original (SHA-1 `e0abd0d9…`); author Zhiguo.he, CC BY-SA 4.0 |
| `allium_cepa_epidermis_specimen.jpg` | `12d2cff76378adae98140789f36970c5dd48a1bc6a3152c7e27d84f3ba62f3fc` | 1600 px LANCZOS derivative; original SHA-256 `6430f011…`, Commons SHA-1 `4a2a8512…`; author Juan Carlos Fonseca Mata, CC BY-SA 4.0 |
| `ongrowthform1917thom_0318.jp2` | `fb329d9ddbe640d8579f5b7bbfa6433df4aa7c98960ee3df3b8615f025cfe693` | printed p. 298 (Figs. 102–103), public domain |
| `ongrowthform1917thom_0341.jp2` | `38d6d1a9bc86ad902e9c4574800239aaf371b2c55aadfd577c3e87893d08b18a` | printed p. 321 (Fig. 124), public domain |
| `thompson1917_fig103_maize_parenchyma.png` | `d1c3c3aa92d2132124b2fd582cc8d5f30382ebc1560476c78aa63f7556433373` | crop (0.46, 0.39, 0.93, 0.615) of leaf 318 |
| `thompson1917_fig124_girardia_epidermis.png` | `3a33c7b1b82aad70cb10d5f19ca4eca2ca2697ea501f7833dc0ead7cc2790d03` | crop (0.17, 0.445, 0.75, 0.55) of leaf 341 |
| `dryad_mn8cs_dataset_metadata.json` | `fe375646e7bd74d9a09b1a9b6e593b8a4078478d761a5fcccd0dcc7a8846f5fd` | Dryad API v2 dataset record |
| `dryad_mn8cs_files_metadata.json` | `ae5c993f1a9031474c5b4d073fe779071393b55b28397c6b7353c92841cb4a3f` | Dryad API v2 file list with MD5s |
| `t1917_018_the_forms_of_tissues_or_cell_a_reference.json` | — | equations E1–E5, state variables, boundary conditions, parameter-regime table, Thompson quotes with page numbers, all digests above |
| `scout_manifest.json` | — | Step 1 dossier (OpenAlex + Commons + Dryad verification) |
| `paper_citations.bib` | — | BibTeX for every source |
| `result.json` | — | the verdict (written in Step 12) |

## The model being tested

* **E1 (Lami / Thompson Fig. 113):** three films of tensions \(T_1,T_2,T_3\) balance when
  \(T_1/\sin\alpha_1 = T_2/\sin\alpha_2 = T_3/\sin\alpha_3\); equal tensions ⇒ 120° each.
* **E2 (Plateau / Lamarle; Taylor 1976):** minimal partitions have only three-fold vertices at 120° in 2D
  (tetrahedral 109°28′ edges in 3D); a four-fold vertex is unstable (\(1+\sqrt3 < 2\sqrt2\)).
* **E3 (identity):** \(\alpha+\beta+\gamma = 360°\) at every planar tri-junction, so the *mean* angle is 120°
  by construction — the test must score the **dispersion** about 120° and the **absence of four-fold vertices**.
* **E5 (simulation):** vertex energy \(E=\sum_e \Lambda(\phi_e)\,l_e + \sum_c \tfrac{K}{2}(A_c-A_0)^2\) with
  \(\Lambda(\phi)=\Lambda_0(1+\varepsilon\cos 2\phi)\); \(\varepsilon=0\) is the soap-froth limit,
  \(\varepsilon>0\) is Thompson's "asymmetrical component arising from growth or traction" (Fig. 124).

A-priori tolerance for "~120°": **±15°** (median |θ − 120°|), with ≥ 90 % three-fold vertices for Plateau's
topological law. Both thresholds were fixed before any specimen was measured.
