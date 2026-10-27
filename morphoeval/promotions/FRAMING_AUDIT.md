# Framing audit — is the promotion falsifier right for each problem?

Generated 2026-10-27T14:40:10Z. Nodes audited: **36** (problem graph OPs, prediction followups, registered predictions, live arguments).

Generic template currently used by `promote_findings.py`:
> Is '<law>' the right generative law for this structure in <organism> — and if not, what observation kills it?

**Fits as-is: 0. Would be misframed by it: 35. Do-not-promote: 1.**

## Verdict counts

| Framing verdict | n |
|---|---|
| adjudicate_camps | 7 |
| blind_locked_use_native_verbatim | 14 |
| do_not_promote | 1 |
| misframed_use_native | 14 |

## Problem types found

| Type | n | Correct ask | Audience |
|---|---|---|---|
| failure_autopsy | 1 | Why did the method fail in exactly these clades — landmark scheme, tree quality, or a real limit? Pick one and | systematists / morphometricians who work on the failing clades |
| formal_proof | 3 | Is the stated gap exactly right, and is the proposed route (proof / counterexample / barrier result) known-dea | mathematicians / theoretical CS (symbolic dynamics, complexity, Lean formalisation) |
| independent_rescore | 1 | Blinded re-score of the named assessment tree with the current pipeline, published as a superseding assessment | independent assessor with no prior involvement (second-company / external auditor) |
| live_argument | 7 | Which camp does current evidence favour, and does the proposed settling measurement really discriminate betwee | representatives of each camp (COI: each is a stakeholder — balance, do not exclude) |
| missing_dataset | 6 | Do you have, or can you point to, exactly this series? The deliverable is data, not a verdict — the pipeline a | labs that hold or can generate the named measurement; data curators |
| preregistered_blind_test | 3 | Before unblinding: does this pre-registration actually separate the hypotheses, and is the confound named in t | experimentalists on the organism/assay + one methods/statistics reviewer |
| registered_prediction | 14 | Does the registered falsifier (quoted verbatim, locked before scoring) hold or fail on the evidence you know?  | experimentalists on the named organism/pathway |
| structural_debt | 1 | — | internal (builder loop) |

## Per node

### OP-01 — formal_proof · `misframed_use_native`
*Prize 1 — close the transparent-gate loophole (short runs 1≤a_i≤7)*  
status: `open` · pooled posterior 11.52%

- **Native falsifier (use this):** A proof (ideally Lean-checked) that for all t0, Π and short-run words, some D1(t)=1 lands on {c_t=1} — or a counterexample word, which would be just as valuable.
- **Correct ask:** Is the stated gap exactly right, and is the proposed route (proof / counterexample / barrier result) known-dead in your field? Point to the obstruction or the paper.
- **Audience:** mathematicians / theoretical CS (symbolic dynamics, complexity, Lean formalisation)
- **Promotion:** promote_for_scrutiny only to the math audience; biologists are the wrong experts. Posterior <15% → frame as 'barrier/negative result wanted', never as a result. (pooled Delphi posterior 11.52% — ask for the barrier/negative result, not the proof)

### OP-02 — formal_proof · `misframed_use_native`
*Prize 2 — diagonal-sample decorrelation along j=⌊S/2⌋*  
status: `open` · pooled posterior 2.033%

- **Native falsifier (use this):** A discrepancy bound for S -> A_S(floor(S/2)) transferring crystal density to the center column, or a proof that such a bound fails.
- **Correct ask:** Is the stated gap exactly right, and is the proposed route (proof / counterexample / barrier result) known-dead in your field? Point to the obstruction or the paper.
- **Audience:** mathematicians / theoretical CS (symbolic dynamics, complexity, Lean formalisation)
- **Promotion:** promote_for_scrutiny only to the math audience; biologists are the wrong experts. Posterior <15% → frame as 'barrier/negative result wanted', never as a result. (pooled Delphi posterior 2.033% — ask for the barrier/negative result, not the proof)

### OP-03 — formal_proof · `misframed_use_native`
*Prize 3 — fixed-seed specialisation transfer*  
status: `open` · pooled posterior 0.037%

- **Native falsifier (use this):** A formal barrier / relativisation result for the fixed-seed problem, or a specified computational model in which the bound is provable.
- **Correct ask:** Is the stated gap exactly right, and is the proposed route (proof / counterexample / barrier result) known-dead in your field? Point to the obstruction or the paper.
- **Audience:** mathematicians / theoretical CS (symbolic dynamics, complexity, Lean formalisation)
- **Promotion:** promote_for_scrutiny only to the math audience; biologists are the wrong experts. Posterior <15% → frame as 'barrier/negative result wanted', never as a result. (pooled Delphi posterior 0.037% — ask for the barrier/negative result, not the proof)

### OP-04 — preregistered_blind_test · `misframed_use_native`
*Prediction beyond the ~50 Myr horizon*  
status: `partial`

- **Native falsifier (use this):** A pre-registered deep-horizon test (registered before unblinding, as in the FishShapes protocol) that beats the clade-prior baseline beyond 50 Myr, or a sharper null.
- **Correct ask:** Before unblinding: does this pre-registration actually separate the hypotheses, and is the confound named in the gap really removed by the design? (Design review, not verdict.)
- **Audience:** experimentalists on the organism/assay + one methods/statistics reviewer
- **Promotion:** Never promote the outcome before the registered test is scored; promote the pre-registration for scrutiny. A negative result is promoted as a result.

### OP-05 — missing_dataset · `misframed_use_native`
*Within-species bee P-matrix (the one missing input)*  
status: `partial`

- **Native falsifier (use this):** A within-species bee forewing landmark dataset with replicates, run through the registered C1-C8 pipeline without changing its hypotheses.
- **Correct ask:** Do you have, or can you point to, exactly this series? The deliverable is data, not a verdict — the pipeline and hypotheses are already locked.
- **Audience:** labs that hold or can generate the named measurement; data curators
- **Promotion:** Promote as a data request, with the locked pipeline attached. Closing = supplying the series, not re-arguing the prediction.

### OP-06 — structural_debt · `do_not_promote`
*Flagship structural debt: growing-domain Turing, autonomous Meinhardt, C. elegans voxel calibration*  
status: `solved`

- **Native falsifier (use this):** Each sub-item closed by a self-test that fails on the current code and passes on the fix, with the audit score re-derived, not asserted.
- **Correct ask:** — do not promote
- **Audience:** internal (builder loop)
- **Promotion:** Solved or internal engineering debt — not a scientist-facing finding.

### OP-07 — preregistered_blind_test · `misframed_use_native`
*Xenopus neighbour transfer: kill the orientation confound*  
status: `refuted`

- **Native falsifier (use this):** A v2 pre-registration using only ground-truth-geometry videos, scored by the hash-locked contract, whichever way it comes out.
- **Correct ask:** Before unblinding: does this pre-registration actually separate the hypotheses, and is the confound named in the gap really removed by the design? (Design review, not verdict.)
- **Audience:** experimentalists on the organism/assay + one methods/statistics reviewer
- **Promotion:** Never promote the outcome before the registered test is scored; promote the pre-registration for scrutiny. A negative result is promoted as a result.

### OP-08 — preregistered_blind_test · `misframed_use_native`
*Pixel v2: image-proximity (mirror) neighbour prediction*  
status: `partial`

- **Native falsifier (use this):** Pre-registered pixel-v2 scoring on videos where poke position separates the mirror and headward hypotheses.
- **Correct ask:** Before unblinding: does this pre-registration actually separate the hypotheses, and is the confound named in the gap really removed by the design? (Design review, not verdict.)
- **Audience:** experimentalists on the organism/assay + one methods/statistics reviewer
- **Promotion:** Never promote the outcome before the registered test is scored; promote the pre-registration for scrutiny. A negative result is promoted as a result.

### OP-11 — independent_rescore · `misframed_use_native`
*Cow Turing follow-ups: independent re-score + migration-wave colonisation variant*  
status: `refuted` · pooled posterior 39.3%

- **Native falsifier (use this):** A blinded independent re-score of the Pass 8+ tree (audit R11), plus a neural-crest migration-wave colonisation variant scored on the same coat-masked, scale-matched statistics — published as a superseding assessment either way.
- **Correct ask:** Blinded re-score of the named assessment tree with the current pipeline, published as a superseding assessment whichever way it comes out.
- **Audience:** independent assessor with no prior involvement (second-company / external auditor)
- **Promotion:** Promote as an audit request; result-track only after the superseding audit lands.

### OP-10 — failure_autopsy · `misframed_use_native`
*Portfolio residue: clades where Thompson-7 failed (Leporidae 0/3, Vespinae 0/7 pre-venation)*  
status: `partial`

- **Native falsifier (use this):** A per-clade failure autopsy in the style of the why-wings decomposition, with at least one failure converted into a correct, pre-registered prediction.
- **Correct ask:** Why did the method fail in exactly these clades — landmark scheme, tree quality, or a real limit? Pick one and name the test that would confirm it.
- **Audience:** systematists / morphometricians who work on the failing clades
- **Promotion:** Promote the failures themselves (negative results are results); the ask is a diagnosis.

### ZF-HH-01:followup — missing_dataset · `misframed_use_native`
*Missing series for ZF-HH-01 (Danio rerio, morphogen_condensation)*  
status: `partially_resolved`

- **Native falsifier (use this):** Lipidation is functional in zebrafish Shh signalling: 'New chemical probes targeting cholesterylation of Sonic Hedgehog in human cells and zebrafish' (2014, doi:10.1039/C4SC01600A), and dual-lipid release biology reviewed in doi:10.1002/bies.202100133. Bound established: wild-type ShhNp is dually lipidated and tethered (repo state), and cholesterylation is required for normal signalling in zebrafish. NOT established from sources extracted: the specific registered gradient-shape claim (distal ptc-domain expansion + reduced near-source peak upon lipidation loss) in a zebrafish tissue assay. Miss
- **Correct ask:** Do you have, or can you point to, exactly this series? The deliverable is data, not a verdict — the pipeline and hypotheses are already locked.
- **Audience:** labs that hold or can generate the named measurement; data curators
- **Promotion:** Promote as a data request, with the locked pipeline attached. Closing = supplying the series, not re-arguing the prediction.

### ZF-CLK-01:followup — missing_dataset · `misframed_use_native`
*Missing series for ZF-CLK-01 (Danio rerio, clock_wavefront)*  
status: `partially_resolved`

- **Native falsifier (use this):** Bound established: the proportional architecture (clock period sets somite size at a given wavefront speed) is the standard, literature-supported segmentation-clock framework (zebrafish clock-mechanism studies, e.g. doi:10.1371/journal.pbio.0050150; core-circuit dynamics doi:10.1371/journal.pbio.1001364). NOT established: a paired dataset (measured T change + measured S change, wavefront controlled) sufficient to test the registered x0.80/x0.90/x1.10/x1.20 multipliers within 5 points. Missing data: paired her1-period and somite-length measurements under a controlled-wavefront perturbation, tab
- **Correct ask:** Do you have, or can you point to, exactly this series? The deliverable is data, not a verdict — the pipeline and hypotheses are already locked.
- **Audience:** labs that hold or can generate the named measurement; data curators
- **Promotion:** Promote as a data request, with the locked pipeline attached. Closing = supplying the series, not re-arguing the prediction.

### ZF-VTX-01:followup — missing_dataset · `misframed_use_native`
*Missing series for ZF-VTX-01 (Danio rerio, vertex_packing)*  
status: `partially_resolved`

- **Native falsifier (use this):** Bound established: 3.813 is the established vertex-model jamming value carried by the dataset law. NOT established: a zebrafish (EVL/epiboly) measured shape-index transition inside 3.76-3.87; the targeted literature search returned no zebrafish shape-index transition dataset. Missing data: cell shape-index series across a zebrafish epithelial jamming/unjamming transition (measurable from published epiboly images, not yet tabulated).
- **Correct ask:** Do you have, or can you point to, exactly this series? The deliverable is data, not a verdict — the pipeline and hypotheses are already locked.
- **Audience:** labs that hold or can generate the named measurement; data curators
- **Promotion:** Promote as a data request, with the locked pipeline attached. Closing = supplying the series, not re-arguing the prediction.

### DM-MUR-NEG-01:followup — missing_dataset · `misframed_use_native`
*Missing series for DM-MUR-NEG-01 (Drosophila melanogaster, branching_tree)*  
status: `partially_resolved`

- **Native falsifier (use this):** Bound established: insect tracheal systems are not generically Murray-optimal (ant tracheal branching violates network-transport assumptions, doi:10.1371/journal.pcbi.1007853; Murray compliance varies by insect system, doi:10.1186/s12915-021-01130-0), consistent with the negative prediction. NOT established: a Drosophila tracheal parent/daughter diameter series sufficient to fit an exponent against the registered 2.8-3.2 band. Missing data: tabulated Drosophila tracheal diameters across branch orders.
- **Correct ask:** Do you have, or can you point to, exactly this series? The deliverable is data, not a verdict — the pipeline and hypotheses are already locked.
- **Audience:** labs that hold or can generate the named measurement; data curators
- **Promotion:** Promote as a data request, with the locked pipeline attached. Closing = supplying the series, not re-arguing the prediction.

### XT-CLK-01:followup — missing_dataset · `misframed_use_native`
*Missing series for XT-CLK-01 (Xenopus tropicalis, clock_wavefront)*  
status: `partially_resolved`

- **Native falsifier (use this):** Bound established: Xenopus somitogenesis uses a her/hes segmentation clock with a wavefront, so the proportional S~T architecture is the applicable framework (cross-species replication target of ZF-CLK-01). NOT established: paired Xenopus period-manipulation + somite-length measurements sufficient to test the x0.80-x1.20 multipliers within 5 points. Missing data: paired clock-period and somite-length series under controlled wavefront conditions in Xenopus.
- **Correct ask:** Do you have, or can you point to, exactly this series? The deliverable is data, not a verdict — the pipeline and hypotheses are already locked.
- **Audience:** labs that hold or can generate the named measurement; data curators
- **Promotion:** Promote as a data request, with the locked pipeline attached. Closing = supplying the series, not re-arguing the prediction.

### ZF-TUR-01 — registered_prediction · `blind_locked_use_native_verbatim`
*BMP patterning in the zebrafish gastrula is NOT free differential diffusion of BMP vs Chordin: Chordin's effective mobility in the embryo wi*  
status: `confirmed`

- **Native falsifier (use this):** Direct mobility measurements (e.g. FRAP of tagged BMP/Chordin in gastrula) showing both species freely diffusing with D ratio ~0.5 AND normal dorsoventral patterning.
- **Correct ask:** Does the registered falsifier (quoted verbatim, locked before scoring) hold or fail on the evidence you know? If 'partially_resolved', can you supply the named missing series?
- **Audience:** experimentalists on the named organism/pathway
- **Promotion:** The falsifier field is LOCKED (blind discipline) — promotion must quote it, never paraphrase or replace it. 'confirmed' outcomes are result-track candidates once a human expert agrees with the resolution. Outcome 'confirmed' → candidate for promote_as_result after one human expert concurs.

### ZF-TUR-02 — registered_prediction · `blind_locked_use_native_verbatim`
*Blocking tolloid cleavage of Chordin (more intact Chordin, less BMP release) contracts the ventral pSmad1/5 signalling domain and dorsalizes*  
status: `confirmed`

- **Native falsifier (use this):** Tolloid loss/cleavage-resistant Chordin producing ventralization or an expanded pSmad domain.
- **Correct ask:** Does the registered falsifier (quoted verbatim, locked before scoring) hold or fail on the evidence you know? If 'partially_resolved', can you supply the named missing series?
- **Audience:** experimentalists on the named organism/pathway
- **Promotion:** The falsifier field is LOCKED (blind discipline) — promotion must quote it, never paraphrase or replace it. 'confirmed' outcomes are result-track candidates once a human expert agrees with the resolution. Outcome 'confirmed' → candidate for promote_as_result after one human expert concurs.

### ZF-HH-01 — registered_prediction · `blind_locked_use_native_verbatim`
*ShhA signalling is short-range and membrane-associated despite a small signalling product, because ShhNp is dually lipidated; disrupting lip*  
status: `partially_resolved`

- **Native falsifier (use this):** Lipidation-deficient ShhA producing an unchanged or steeper/higher-peak gradient in zebrafish neural tube/otic assays.
- **Correct ask:** Does the registered falsifier (quoted verbatim, locked before scoring) hold or fail on the evidence you know? If 'partially_resolved', can you supply the named missing series?
- **Audience:** experimentalists on the named organism/pathway
- **Promotion:** The falsifier field is LOCKED (blind discipline) — promotion must quote it, never paraphrase or replace it. 'confirmed' outcomes are result-track candidates once a human expert agrees with the resolution. Outcome 'partially_resolved' → promote as a DATA REQUEST for the named missing series.

### ZF-CLK-01 — registered_prediction · `blind_locked_use_native_verbatim`
*Somite length scales in direct proportion to segmentation-clock period at fixed wavefront speed (S = u * T_clock), including for period chan*  
status: `partially_resolved`

- **Native falsifier (use this):** A verified clock-period change (live her1 reporter) with wavefront speed controlled, producing somite-length change of opposite sign or <50% of the proportional expectation.
- **Correct ask:** Does the registered falsifier (quoted verbatim, locked before scoring) hold or fail on the evidence you know? If 'partially_resolved', can you supply the named missing series?
- **Audience:** experimentalists on the named organism/pathway
- **Promotion:** The falsifier field is LOCKED (blind discipline) — promotion must quote it, never paraphrase or replace it. 'confirmed' outcomes are result-track candidates once a human expert agrees with the resolution. Outcome 'partially_resolved' → promote as a DATA REQUEST for the named missing series.

### ZF-VTX-01 — registered_prediction · `blind_locked_use_native_verbatim`
*Zebrafish epithelial tissue assigned the vertex-packing law undergoes its jamming/unjamming transition at cell shape index p0* = 3.813.*  
status: `partially_resolved`

- **Native falsifier (use this):** Measured transition in a zebrafish epithelium (e.g. enveloping layer during epiboly) outside 3.7-3.9, or no transition in shape index at all.
- **Correct ask:** Does the registered falsifier (quoted verbatim, locked before scoring) hold or fail on the evidence you know? If 'partially_resolved', can you supply the named missing series?
- **Audience:** experimentalists on the named organism/pathway
- **Promotion:** The falsifier field is LOCKED (blind discipline) — promotion must quote it, never paraphrase or replace it. 'confirmed' outcomes are result-track candidates once a human expert agrees with the resolution. Outcome 'partially_resolved' → promote as a DATA REQUEST for the named missing series.

### DM-WG-01 — registered_prediction · `blind_locked_use_native_verbatim`
*Wingless spread requires Porcupine O-palmitoleoylation at Ser239: blocking it (porcupine loss or Wg Ser239 substitution) collapses Wg protei*  
status: `confirmed`

- **Native falsifier (use this):** Porcupine/Wg-lipidation mutants with a wild-type-width Wg gradient and target domain in wing disc.
- **Correct ask:** Does the registered falsifier (quoted verbatim, locked before scoring) hold or fail on the evidence you know? If 'partially_resolved', can you supply the named missing series?
- **Audience:** experimentalists on the named organism/pathway
- **Promotion:** The falsifier field is LOCKED (blind discipline) — promotion must quote it, never paraphrase or replace it. 'confirmed' outcomes are result-track candidates once a human expert agrees with the resolution. Outcome 'confirmed' → candidate for promote_as_result after one human expert concurs.

### DM-HH-01 — registered_prediction · `blind_locked_use_native_verbatim`
*The Hh signalling species is the 173 aa dually-lipidated N-product, not the 471 aa precursor: cholesterol-deficient HhN spreads farther from*  
status: `confirmed`

- **Native falsifier (use this):** Cholesterol-deficient HhN showing a shorter range or higher near-source peak than wild type in wing disc.
- **Correct ask:** Does the registered falsifier (quoted verbatim, locked before scoring) hold or fail on the evidence you know? If 'partially_resolved', can you supply the named missing series?
- **Audience:** experimentalists on the named organism/pathway
- **Promotion:** The falsifier field is LOCKED (blind discipline) — promotion must quote it, never paraphrase or replace it. 'confirmed' outcomes are result-track candidates once a human expert agrees with the resolution. Outcome 'confirmed' → candidate for promote_as_result after one human expert concurs.

### DM-DPP-01 — registered_prediction · `blind_locked_use_native_verbatim`
*The diffusible Dpp species in wing disc is the annotated 132 aa mature chain (residues 457-588), so gradient models using the 588 aa precurs*  
status: `confirmed`

- **Native falsifier (use this):** Evidence that uncleaved 588 aa Dpp precursor is the extracellular signalling species in wing disc.
- **Correct ask:** Does the registered falsifier (quoted verbatim, locked before scoring) hold or fail on the evidence you know? If 'partially_resolved', can you supply the named missing series?
- **Audience:** experimentalists on the named organism/pathway
- **Promotion:** The falsifier field is LOCKED (blind discipline) — promotion must quote it, never paraphrase or replace it. 'confirmed' outcomes are result-track candidates once a human expert agrees with the resolution. Outcome 'confirmed' → candidate for promote_as_result after one human expert concurs.

### DM-CLK-NEG-01 — registered_prediction · `blind_locked_use_native_verbatim`
*NEGATIVE: the clock-wavefront law assigned to Drosophila will FAIL for blastoderm segmentation -- Drosophila is a long-germ embryo with no t*  
status: `confirmed`

- **Native falsifier (use this):** Demonstration of a segmentation-clock oscillation wavefront that sequentially specifies Drosophila blastoderm stripes.
- **Correct ask:** Does the registered falsifier (quoted verbatim, locked before scoring) hold or fail on the evidence you know? If 'partially_resolved', can you supply the named missing series?
- **Audience:** experimentalists on the named organism/pathway
- **Promotion:** The falsifier field is LOCKED (blind discipline) — promotion must quote it, never paraphrase or replace it. 'confirmed' outcomes are result-track candidates once a human expert agrees with the resolution. Outcome 'confirmed' → candidate for promote_as_result after one human expert concurs.

### DM-MUR-NEG-01 — registered_prediction · `blind_locked_use_native_verbatim`
*NEGATIVE / discriminating: Drosophila tracheal branching, assigned Murray's law, will not obey the fluid-optimal exponent: parent/daughter d*  
status: `partially_resolved`

- **Native falsifier (use this):** Tracheal diameter measurements across branch orders fitting exponent 3.0 within 2.8-3.2 with small residuals.
- **Correct ask:** Does the registered falsifier (quoted verbatim, locked before scoring) hold or fail on the evidence you know? If 'partially_resolved', can you supply the named missing series?
- **Audience:** experimentalists on the named organism/pathway
- **Promotion:** The falsifier field is LOCKED (blind discipline) — promotion must quote it, never paraphrase or replace it. 'confirmed' outcomes are result-track candidates once a human expert agrees with the resolution. Outcome 'partially_resolved' → promote as a DATA REQUEST for the named missing series.

### XT-BMP-01 — registered_prediction · `blind_locked_use_native_verbatim`
*Noggin (196 aa secreted chain) patterns Xenopus dorsoventral axis by binding BMP4, not by diffusing as a Turing inhibitor: ventral Noggin mi*  
status: `confirmed`

- **Native falsifier (use this):** Noggin misexpression with no dorsalizing effect, or Noggin loss with dorsalization, in Xenopus embryos.
- **Correct ask:** Does the registered falsifier (quoted verbatim, locked before scoring) hold or fail on the evidence you know? If 'partially_resolved', can you supply the named missing series?
- **Audience:** experimentalists on the named organism/pathway
- **Promotion:** The falsifier field is LOCKED (blind discipline) — promotion must quote it, never paraphrase or replace it. 'confirmed' outcomes are result-track candidates once a human expert agrees with the resolution. Outcome 'confirmed' → candidate for promote_as_result after one human expert concurs.

### XT-SHH-01 — registered_prediction · `blind_locked_use_native_verbatim`
*Xenopus Shh is dually lipidated like its zebrafish and fly orthologs, even though the current UniProt entry (A0A803JKD3, unreviewed) annotat*  
status: `confirmed`

- **Native falsifier (use this):** Reviewed Xenopus tropicalis Shh annotation or experiment showing an unlipidated, freely diffusing N-product.
- **Correct ask:** Does the registered falsifier (quoted verbatim, locked before scoring) hold or fail on the evidence you know? If 'partially_resolved', can you supply the named missing series?
- **Audience:** experimentalists on the named organism/pathway
- **Promotion:** The falsifier field is LOCKED (blind discipline) — promotion must quote it, never paraphrase or replace it. 'confirmed' outcomes are result-track candidates once a human expert agrees with the resolution. Outcome 'confirmed' → candidate for promote_as_result after one human expert concurs.

### XT-DQ-01 — registered_prediction · `blind_locked_use_native_verbatim`
*DATA-QUALITY: the UniProt accession currently linked for Xenopus bmp4 (Q28FI5, 69 aa) is a fragment, not the BMP4 precursor: it will not mat*  
status: `confirmed`

- **Native falsifier (use this):** Q28FI5 shown to be the complete, reviewed Xenopus tropicalis BMP4 precursor.
- **Correct ask:** Does the registered falsifier (quoted verbatim, locked before scoring) hold or fail on the evidence you know? If 'partially_resolved', can you supply the named missing series?
- **Audience:** experimentalists on the named organism/pathway
- **Promotion:** The falsifier field is LOCKED (blind discipline) — promotion must quote it, never paraphrase or replace it. 'confirmed' outcomes are result-track candidates once a human expert agrees with the resolution. Outcome 'confirmed' → candidate for promote_as_result after one human expert concurs.

### XT-CLK-01 — registered_prediction · `blind_locked_use_native_verbatim`
*Xenopus tropicalis somitogenesis obeys the same proportional clock-wavefront relation as zebrafish: somite length changes in proportion to s*  
status: `partially_resolved`

- **Native falsifier (use this):** Xenopus period manipulations (hes-gene perturbation/temperature) with controlled wavefront speed producing non-proportional or opposite somite-length change.
- **Correct ask:** Does the registered falsifier (quoted verbatim, locked before scoring) hold or fail on the evidence you know? If 'partially_resolved', can you supply the named missing series?
- **Audience:** experimentalists on the named organism/pathway
- **Promotion:** The falsifier field is LOCKED (blind discipline) — promotion must quote it, never paraphrase or replace it. 'confirmed' outcomes are result-track candidates once a human expert agrees with the resolution. Outcome 'partially_resolved' → promote as a DATA REQUEST for the named missing series.

### LA-01 — live_argument · `adjudicate_camps`
*How is the zebrafish BMP dorsoventral gradient formed: Chordin-mediated shuttling, regulated free diffusion/source-sink, or ECM/HSPG-shaped diffusion?*  
status: `partly_settled`

- **Native falsifier (use this):** Paired FRAP/effective-mobility measurements of tagged BMP and Chordin plus pSmad1/5 gradient ranges in at least two Danio species of different embryo size, under identical staging.
- **Correct ask:** Which camp does current evidence favour, and does the proposed settling measurement really discriminate between the camps? (Adjudication, not a kill-one-law question.)
- **Audience:** representatives of each camp (COI: each is a stakeholder — balance, do not exclude)
- **Promotion:** Promote as adversarial-collaboration proposal; settled halves must not be re-litigated. Settled part exists — promote ONLY the remaining live part.
- **Camps:** Shuttling essential, Regulated diffusion / source-sink sufficient, ECM/HSPG-shaped

### LA-02 — live_argument · `adjudicate_camps`
*Are zebrafish stripes (and fish pigment patterns generally) a Turing reaction-diffusion system, or a cell-interaction network that only mimics one?*  
status: `live`

- **Native falsifier (use this):** Stripe wavelength and regeneration kinetics in cell-interaction mutants (leopard/gja5b, pfeffer/csf1ra) compared quantitatively against both model classes fitted on the same fish; the Turing account must predict the wavelength shift from a kinetic parameter, the cell account from interaction ranges.
- **Correct ask:** Which camp does current evidence favour, and does the proposed settling measurement really discriminate between the camps? (Adjudication, not a kill-one-law question.)
- **Audience:** representatives of each camp (COI: each is a stakeholder — balance, do not exclude)
- **Promotion:** Promote as adversarial-collaboration proposal; settled halves must not be re-litigated.
- **Camps:** Turing / reaction-diffusion, Cell-interaction network

### LA-03 — live_argument · `adjudicate_camps`
*Is vertebrate digit patterning a BMP-SOX9-WNT Turing system modulated by Hox dose, or a positional-gradient system in which Turing-like periodicity is secondary*  
status: `live`

- **Native falsifier (use this):** Digit wavelength (condensation spacing) as a function of distal Hox dose in allelic series, tested against the BSW prediction that wavelength -- not just digit identity -- shifts continuously with Hox-modulated parameters.
- **Correct ask:** Which camp does current evidence favour, and does the proposed settling measurement really discriminate between the camps? (Adjudication, not a kill-one-law question.)
- **Audience:** representatives of each camp (COI: each is a stakeholder — balance, do not exclude)
- **Promotion:** Promote as adversarial-collaboration proposal; settled halves must not be re-litigated.
- **Camps:** Turing (BSW) primary, Positional / Hox primary

### LA-04 — live_argument · `adjudicate_camps`
*How is Hedgehog transported: extracellular diffusion (with Dispatched/Scube hand-off) or cytoneme-mediated delivery?*  
status: `live`

- **Native falsifier (use this):** Gradient range after cytoneme disruption that leaves secretion intact (and vice versa), in the same tissue, with the lipidation state of the transported species verified from sequence/protein state.
- **Correct ask:** Which camp does current evidence favour, and does the proposed settling measurement really discriminate between the camps? (Adjudication, not a kill-one-law question.)
- **Audience:** representatives of each camp (COI: each is a stakeholder — balance, do not exclude)
- **Promotion:** Promote as adversarial-collaboration proposal; settled halves must not be re-litigated.
- **Camps:** Extracellular diffusion / hand-off, Cytoneme delivery

### LA-05 — live_argument · `adjudicate_camps`
*Does somite size scale with segmentation-clock period via wavefront speed (clock-and-wavefront), and what sets the scaling across species?*  
status: `live`

- **Native falsifier (use this):** Paired clock-period and somite-length measurements with PSM length/wavefront speed recorded in the same embryos, across at least the fish and frog systems the repo holds.
- **Correct ask:** Which camp does current evidence favour, and does the proposed settling measurement really discriminate between the camps? (Adjudication, not a kill-one-law question.)
- **Audience:** representatives of each camp (COI: each is a stakeholder — balance, do not exclude)
- **Promotion:** Promote as adversarial-collaboration proposal; settled halves must not be re-litigated.
- **Camps:** Clock-and-wavefront scaling, PSM-growth / multi-factor control

### LA-06 — live_argument · `adjudicate_camps`
*How does the Dpp/BMP gradient scale with the growing Drosophila wing disc: degradation-mediated scaling, expansion-repression, or growth-coupled production?*  
status: `live`

- **Native falsifier (use this):** Dpp gradient length, disc size, Dpp turnover and candidate-expander expression measured across growth stages/genotypes in one quantitative series; the two camps predict different couplings between turnover and length.
- **Correct ask:** Which camp does current evidence favour, and does the proposed settling measurement really discriminate between the camps? (Adjudication, not a kill-one-law question.)
- **Audience:** representatives of each camp (COI: each is a stakeholder — balance, do not exclude)
- **Promotion:** Promote as adversarial-collaboration proposal; settled halves must not be re-litigated.
- **Camps:** Degradation / kinetics scaling, Expansion-repression / feedback scaling

### LA-07 — live_argument · `adjudicate_camps`
*Is the bioelectric (Vmem) prepattern instructive for Xenopus craniofacial/orientation patterning -- upstream information that biochemical gradients read out -- *  
status: `live`

- **Native falsifier (use this):** Blinded, pre-registered ion-channel misexpression with BMP-pathway transcriptional readout and craniofacial scoring, where the bioelectric account predicts the readout change in advance and the conventional account predicts none without direct pathway manipulation. The repo's own blind-test contract (hash-locked, scorer refuses on mismatch) is the required format, given its earlier negative blind results.
- **Correct ask:** Which camp does current evidence favour, and does the proposed settling measurement really discriminate between the camps? (Adjudication, not a kill-one-law question.)
- **Audience:** representatives of each camp (COI: each is a stakeholder — balance, do not exclude)
- **Promotion:** Promote as adversarial-collaboration proposal; settled halves must not be re-litigated.
- **Camps:** Instructive bioelectric prepattern, Conventional-pathway primacy

## Thompson 1917 claim/counterclaim nanopubs

59 historical claims, relations {'NOT_TESTABLE': 3, 'QUALIFIED': 9, 'PARTIALLY_REFUTED': 5, 'SUPPORTED': 6, 'SUPERSEDED': 4, 'MIXED': 2, 'NO_DIRECT_TEST': 30} — **do_not_promote**. Historical claim + literature verdict records (signed nanopubs). Not open problems; promoting them as findings would misrepresent settled literature as new results. Only 'repo_2026_status_separate' items that are OPEN re-enter via the problem graph.
