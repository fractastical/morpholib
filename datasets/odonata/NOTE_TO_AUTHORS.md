# Draft note to Eshghi, Buesch & Gorb on Supplementary Table S1 (not yet sent)

Remediation item from the MorphoEval audit `2026-10-14-odonata-audit`
("Report the discrepancies (#7, #23/#24) to the authors via the Zenodo record").
The text below is a draft for the repository owner to send — via the contact
on the Zenodo record https://doi.org/10.5281/zenodo.19899648 or the
corresponding author of the paper — after checking it against the current
version of the record (a newer version may already correct these rows).
Nothing in it has been verified beyond our own re-segmentation, so it is
phrased as a question.

---

Subject: Supplementary Table S1 of "Critical Crack Length as an Evolutionary Constant in Odonata Wing Venation" — three rows that may not match their drawings

Dear Dr Eshghi, Dr Buesch and Prof. Gorb,

Thank you for releasing the supplementary data and the fossil venation
drawings under CC BY (Zenodo 10.5281/zenodo.19899648). We have been using them
in an open re-analysis (https://github.com/fractastical/morpholib,
script `2026-eshghi-odonata-wing-cells.py`) that re-segments every fossil
drawing in `Supplementary Images/Fossil Wing Images/` from the pixels
(skeletonised vein ink, enclosed regions as cells, scale from each drawing's
own scale bar) and compares the result with Table S1 row by row. For 66 of the
69 wings the agreement is close (median cell-count difference 0 %, 90 % of
wings within 2 %; median cell length within ~3 %, ours systematically slightly
longer because our cells are bounded by vein mid-lines). Three rows stand out,
and we wanted to ask whether they are what you intended:

1. **#7 *Euarchistigma peterknobli* 1 (Right Wing).** Table S1 gives wing
   length 24.7 mm, 282 cells, median cell length 0.76 mm, largest cell
   1.80 mm. The drawing of that name has a 10 mm scale bar of 247 px, which
   makes the drawn wing 61.2 mm long; we count 829 cells with median 1.15 mm
   and largest cell 7.49 mm. Scaling cannot reconcile the two (the cell count
   is independent of scale), so the drawing and the table row appear to
   describe different wings. Could the row or the image have been swapped
   with another specimen?

2. **#23 and #24 *Paracordulagomphus aberrans* 1 (Right Hindwing / Left
   Forewing).** Table S1 lists 480 cells (median 0.556 mm) for #23 and 530
   cells (median 0.661 mm) for #24. From the drawings we count 530 cells
   (median 0.675 mm) for the image named #23 and 484 cells (median 0.579 mm)
   for the one named #24 — i.e. our values for each image match the table row
   of the other. The two rows may be swapped relative to the image names
   (or the two images may be).

Everything else we checked agrees within the tolerance one would expect from
two segmentations of the same drawings, including the *Cratostenophlebia
schwickerti* wings (#8–#11, ~3000 cells each), where we count 6–10 % more cells
than Table S1 because our skeleton splits some cells in the densest venation
that were presumably merged by hand.

Our per-wing comparison is in
`datasets/odonata/fossil_wing_resegmentation.csv` (columns `n_cells` /
`table_cell_count`, `cell_len_median_mm` / `table_cell_len_median_mm`, with
the scale source for each drawing), and the method is described in
`datasets/odonata/README.md` in the same repository. If the rows are indeed
transposed, a corrected version of the Zenodo record would let downstream
users (including us) cite it instead of carrying a local fix; if we have
misread the images, we would be glad to correct our comparison.

With thanks and best regards,

[name]
