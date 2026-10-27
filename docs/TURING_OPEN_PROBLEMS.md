# Turing open problems — forward from 1952

Snapshot 2026-10-27. Source of truth: `morphoeval/build_turing_open_problems.py` (regenerate everything with it). Problem sourcing here runs *outward* from one historical model to what the field still calls open (Krause, Gaffney, Maini & Klika 2021 theme issue; Landge et al. 2020; Brückner 2026), then prices each problem on the two axes this repo keeps separate:

- **Resolvability (0-100)** — the deterministic Problem-Scout rubric `{"tool_fit": 25, "data_in_hand": 20, "falsifiable_done": 15, "compute_in_repo": 15, "isolation": 10, "time_to_verdict": 10, "not_blocked": 5}`: can *our* tools, data in hand and a taste-free definition-of-done reach a verdict? Not truth odds.
- **Viability (%)** — a **provisional single-assessor** Delphi posterior (Claude Fable 5.1 Thinking High (single assessor, provisional); prior 50%, signed deciban ledger, same discipline as `predictions/round2_live_arguments.json`). NOT a multi-agent panel; a Fable/GPT/Gemini panel pass may supersede it and may not edit it. Colour rule as the graph: green only >70%.

Problems with resolvability >= 70 (RESOLVE NOW band) enter the MorphoSwarm queue as OP-12..OP-16; all ten enter the problem graph as a band under the Thompson claims.

| Rank | ID | Problem | Anchor | Resolvability | Band | Viability | Net dB | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | TOP-03 | The fine-tuning objection: replicate the two-node vs three-node Turing parameter-volume ratio and the 1%-uncertainty retention claim | Murray 1982 -> Scholes 2019 -> Tica 2024 | **88** | RESOLVE NOW | 79.9% | +6 | LIKELY_REPRODUCIBLE_ONE_LOOP |
| 2 | TOP-02 | The diffusion-ratio problem: which molecularly identified Turing systems can be two-species at all? | Gierer-Meinhardt 1972 -> Marcon 2016 -> 2025 'fundamental limits' | **78** | RESOLVE NOW | 55.7% | +1 | DECIDABLE_TABLE_MOSTLY_UNMEASURED_ROWS |
| 3 | TOP-07 | When does continuous reaction-diffusion collapse to a discrete automaton? Meinhardt mantle PDE vs the Lean-proved Rule 30 gate | Meinhardt 1995 -> Boettiger-Ermentrout-Oster 2009 -> Manukyan 2017 | **78** | RESOLVE NOW | 50.0% | +0 | DECIDABLE_BUT_OBJECT_NOT_YET_AUTONOMOUS |
| 4 | TOP-01 | Mechanism identification: a pre-registered statistic that separates reaction-diffusion from colonisation, phase separation and buckling on static patterns | Turing 1952 -> Murray 1981 -> OP-11 (2026) | **77** | RESOLVE NOW | 55.7% | +1 | ACHIEVABLE_IN_TWO_LOOPS_HARD_CASE_KNOWN |
| 5 | TOP-09 | Turing's neglected case: is there a wave (Turing-Hopf) dispersion signature in the embryonic Ca2+ wave field? | Turing 1952 Sec. 10 (oscillatory case) -> Ca2+ wave-vector pipeline | **71** | RESOLVE NOW | 38.7% | -2 | TESTABLE_NOW_LIKELY_NULL |
| 6 | TOP-04 | Growing domains: an independent ontogenetic stripe-count series to test the insertion law blind | Kondo-Asai 1995 -> Crampin-Gaffney-Maini 1999 -> Krause 2021 | **66** | RESOLVABLE WITH WORK | 33.4% | -3 | BLOCKED_ON_DATA_NOT_ON_TOOLS |
| 7 | TOP-06 | Reproducible topology: why five digits every time when RD from noise gives defective labyrinths | Sheth 2012 -> Raspopovic 2014 -> Green-Sharpe 2015 | **60** | RESOLVABLE WITH WORK | 28.5% | -4 | MODEL_MUST_BE_BUILT_FIRST |
| 8 | TOP-05 | Scaling: does biological pattern wavelength scale with body size across taxa, against Turing's intrinsic wavelength? | Turing 1952 (intrinsic wavelength) -> Werner 2015 (scaling) | **54** | RESOLVABLE WITH WORK | 16.6% | -7 | DATA_ASSET_IS_AN_ARTEFACT |
| 9 | TOP-10 | Ecology-scale Turing: fairy circles as scale-dependent-feedback RD vs termite engineering | Klausmeier 1999 -> Getzin 2016 -> Tarnita 2017 | **51** | RESOLVABLE WITH WORK | 28.5% | -4 | OUTSIDE_THE_REPO_AND_LIKELY_INDECISIVE |
| 10 | TOP-08 | Synthetic realisation: a robust, deterministic periodic Turing pattern in eukaryotic cells | Sekine 2018 -> Tica 2024 | **37** | HARD HERE | 11.2% | -9 | NOT_WITH_OUR_TOOLS |

## Each problem

### TOP-03 — The fine-tuning objection: replicate the two-node vs three-node Turing parameter-volume ratio and the 1%-uncertainty retention claim

*Anchor:* Murray 1982 -> Scholes 2019 -> Tica 2024

**Gap.** Turing instability is topologically common but occupies tiny parameter volumes: 0.00035% of sampled space for the classical two-node circuit vs 0.022% for the best three-node topology (Tica et al. 2024, Latin-hypercube linear stability), with 33% retention under 1% parameter uncertainty around a Turing I point. These numbers decide whether the fine-tuning objection to biological Turing systems stands; they have not been independently reproduced.

**Definition of done.** A pre-registered Latin-hypercube linear-stability sweep on the repo's two-node kinetics and a three-node immobile-node topology within published parameter bounds, reporting both Turing fractions and the 1%/5% local-retention fractions; verdict = within one order of magnitude of the published ratio (>=10x three-node over two-node) or not.

**Tool route.** models/1952-turing-morpho.py (Jacobian / dispersion code, Pearson Saddle-Node & Hopf curves, Lean d_c); new sweep script under morphoeval/ or models/; registry row for the ratio before running.

**Deciding data.** Published fractions (Tica 2024 Fig. 2A, Table S5 bounds; Scholes 2019 atlas) — no biological data needed.

**Prior attempts.** scout-034 proposed a 12-class Pearson (F,k) sweep, not a robustness fraction; nothing recorded on parameter volume.

**Resolvability 88/100 — RESOLVE NOW.** tool_fit 20/25: linear stability + Lean d_c already in the flagship script; a sweep driver is all that is missing; data_in_hand 18/20: target numbers and parameter bounds are published; nothing to fetch but a supplementary table; falsifiable_done 14/15: order-of-magnitude ratio and retention fractions are numbers a scorer reads off; compute_in_repo 13/15: 10^6-10^7 LHS Jacobian evaluations in numpy, under an hour; isolation 10/10: one new script + registry row; time_to_verdict 8/10: one loop; not_blocked 5/5: no prior attempt

**Viability 79.9% (prior 50%, +9 dB for, -3 dB against, net +6 dB, BF 3.9811, confidence 0.7) — LIKELY_REPRODUCIBLE_ONE_LOOP.**

*Hypothesis scored:* The repo's sweep reproduces the published two-node vs three-node Turing-fraction ratio within one order of magnitude and the >=10x ordering, with the pre-registered numbers scored.

- [+4 dB] Pure deterministic computation; linear stability of small Jacobians is exactly what the flagship already does and Lean already checks
- [+3 dB] Published target numbers and bounds exist to score against
- [+2 dB] Pearson/Turing LSA code and the d_c theorem are already in the repo
- [-2 dB] Fractions are sensitive to the exact parameter bounds and sampling conventions (Table S5); mismatch shifts them by orders of magnitude
- [-1 dB] LHS undersamples high-dimensional parameter space (the authors say so); 'within one order of magnitude' may still fail on sampling noise

*Sources:* [The Chemical Basis of Morphogenesis (1952)](https://doi.org/10.1098/rstb.1952.0012); [A Comprehensive Network Atlas Reveals That Turing Patterns Are Common but Not Robust (2019)](https://doi.org/10.1016/j.cels.2019.07.007); [High-throughput mathematical analysis identifies Turing networks for patterning with equally diffusing signals (2016)](https://doi.org/10.7554/eLife.14022); [A three-node Turing gene circuit forms periodic spatial patterns in bacteria (2024)](https://doi.org/10.1016/j.cels.2024.11.002); [Fluctuation-driven Turing patterns (2011)](https://doi.org/10.1103/PhysRevE.84.011112)

### TOP-02 — The diffusion-ratio problem: which molecularly identified Turing systems can be two-species at all?

*Anchor:* Gierer-Meinhardt 1972 -> Marcon 2016 -> 2025 'fundamental limits'

**Gap.** Two-species activator-inhibitor models need D_inh/D_act above the Lean-checked threshold (d_c = 3.93 for the repo's Gray-Scott Jacobian; ~4-10 generally), yet measured extracellular diffusivities rarely show it; three-node networks with an immobile node remove the requirement (Marcon 2016) and extracellular wiring changes behaviour qualitatively (2025 bioRxiv). Nobody has tabulated, system by system, which of the molecularly supported cases (WNT/DKK, FGF/Shh palate, Bmp-Sox9-Wnt digits, GDF5/NOG joints, EDAR/WNT/BMP ridges, Nodal/Lefty) actually satisfies the two-species condition with measured D.

**Definition of done.** A registered table of >=6 molecular Turing systems giving measured or bounded D_act, D_inh (with DOI), run through the repo's dispersion relation, classifying each as two-node-feasible / needs immobile third node / D unmeasured; the count in each class pre-registered before the literature values are looked up.

**Tool route.** python3 morphoeval/scout_literature.py --query '<morphogen> diffusion coefficient FRAP FCS' per system; models/1952-turing-morpho.py dispersion (Lean turing_differential_diffusion_instability, d_c = 3.9271); predictions/ registry row for the pre-registered counts.

**Deciding data.** Public literature only. Known in hand: Nodal/Lefty D measured (Mueller 2012); GDF5/NOG model needs no differential diffusion (Grall 2024). Others likely unmeasured — that is itself the finding.

**Prior attempts.** None in repo. scout-063 ZF-TUR-01 kernel-checked a size-prior D ratio (D_inh/D_act < 1 for chd/bmp2b) — a related negative.

**Resolvability 78/100 — RESOLVE NOW.** tool_fit 19/25: scout_literature.py + the existing dispersion code and Lean d_c decide each row; data_in_hand 10/20: all values are one public fetch away, but most D's are probably unmeasured; falsifiable_done 13/15: pre-registered class counts; the 'unmeasured' class is checkable by absence of a DOI; compute_in_repo 14/15: linear stability per row is instantaneous; isolation 10/10: one new table + registry row; time_to_verdict 7/10: one sitting of literature extraction; not_blocked 5/5: no prior attempt

**Viability 55.7% (prior 50%, +6 dB for, -5 dB against, net +1 dB, BF 1.2589, confidence 0.65) — DECIDABLE_TABLE_MOSTLY_UNMEASURED_ROWS.**

*Hypothesis scored:* The census reaches a classification for >=6 systems, with the pre-registered class counts scored, and at least one system falls in each of the three classes.

- [+3 dB] d_c is Lean-checked and the dispersion code exists; each row is a one-line computation once D's are in hand
- [+2 dB] Nodal/Lefty D's are measured (Mueller 2012) and GDF5/NOG is published as not needing differential diffusion (Grall 2024): two rows are already decidable
- [+1 dB] Literature route is the repo's own scout_literature.py
- [-3 dB] For WNT/DKK, FGF/Shh, EDAR/BMP the extracellular D's are not measured: the majority verdict may be 'unmeasured', informative but thin
- [-2 dB] Effective D depends on receptor binding and sequestration (2025 'fundamental limits'), so a raw D ratio is not the deciding quantity for the biology, only for the two-species model class

*Sources:* [A theory of biological pattern formation (1972)](https://doi.org/10.1007/BF00289234); [Differential diffusivity of Nodal and Lefty underlies a reaction-diffusion patterning system (2012)](https://doi.org/10.1126/science.1221920); [High-throughput mathematical analysis identifies Turing networks for patterning with equally diffusing signals (2016)](https://doi.org/10.7554/eLife.14022); [WNT and DKK determine hair follicle spacing through a reaction-diffusion mechanism (2006)](https://doi.org/10.1126/science.1130088); [Periodic stripe formation by a Turing mechanism operating at growth zones in the mammalian palate (2012)](https://doi.org/10.1038/ng.1090); [Digit patterning is controlled by a Bmp-Sox9-Wnt Turing network modulated by morphogen gradients (2014)](https://doi.org/10.1126/science.1252960); [Self-organized BMP signaling dynamics underlie the development and evolution of digit segmentation patterns in birds and mammals (2024)](https://doi.org/10.1073/pnas.2304470121); [The developmental basis of fingerprint pattern formation and variation (2023)](https://doi.org/10.1016/j.cell.2023.01.015); [Fundamental limits on patterning by Turing-like reaction-diffusion mechanisms (2025)](https://doi.org/10.1101/2025.05.27.656324)

### TOP-07 — When does continuous reaction-diffusion collapse to a discrete automaton? Meinhardt mantle PDE vs the Lean-proved Rule 30 gate

*Anchor:* Meinhardt 1995 -> Boettiger-Ermentrout-Oster 2009 -> Manukyan 2017

**Gap.** Ocellated lizard scales behave as a cellular automaton emerging from RD on a thick-skin lattice (Manukyan 2017), and the repo has Lean-proved that the Ermentrout-Oster neural mantle gate is bit-for-bit Rule 30 — but whether the repo's Meinhardt-Klingler Conus PDE, discretised at the mantle, induces a Rule 30-class local rule has never been measured. It is the bridge between the Rule 30 programme and the RD programme, and the RD-vs-neural shell-pattern debate is still molecularly unresolved.

**Definition of done.** From a Meinhardt PDE kymograph at the mantle edge, extract the empirical 3-neighbourhood -> next-state transducer table (binary threshold pre-registered), report the fraction of transitions agreeing with Rule 30 vs Rule 90 vs best-fit rule, plus the 2D ANF-carry density and wall chirality already used to separate R30 from R90 (node r0); verdict thresholds locked before running.

**Tool route.** models/1984-wolfram-conus-textile-rule30.py (Meinhardt-Klingler PDE + Rule 30 + kymograph); morphoeval/lean/ Ermentrout-Oster gate theorem; measure_outputs.py for ANF carry / chirality; registry row.

**Deciding data.** All in repo (PDE, CA, Lean gate, 2D observables).

**Prior attempts.** scout-003 (Rule30 <-> Ermentrout-Oster gate chain traceability); scout-034/038/039 (stochastic activator injection debt, OP-06b). None extracted the transducer table.

**Resolvability 78/100 — RESOLVE NOW.** tool_fit 18/25: PDE, CA, Lean gate and 2D observables all exist; only the transducer extraction is new; data_in_hand 17/20: everything is generated in repo; falsifiable_done 12/15: rule-agreement fractions are numbers; the binarisation threshold is the taste risk, so it is pre-registered; compute_in_repo 13/15: 1D PDE kymographs are seconds; isolation 9/10: new analysis script + registry row; time_to_verdict 6/10: one loop if the PDE is used as-is; not_blocked 3/5: OP-06b: the PDE only patterns with an unsourced stochastic injection, so the object reduced is not autonomous

**Viability 50.0% (prior 50%, +6 dB for, -6 dB against, net +0 dB, BF 1.0, confidence 0.6) — DECIDABLE_BUT_OBJECT_NOT_YET_AUTONOMOUS.**

*Hypothesis scored:* The extracted transducer table from the Meinhardt mantle PDE agrees with Rule 30 above the locked threshold on both 1D transitions and 2D ANF/chirality observables.

- [+3 dB] Both objects are in the repo and the gate = Rule 30 identity is Lean-proved
- [+2 dB] Transducer-table extraction from a kymograph is mechanical
- [+1 dB] Manukyan 2017 is a published template for RD -> CA reduction
- [-3 dB] The Conus PDE patterns only with the unsourced stochastic injection (OP-06b): any reduction is of a non-autonomous object until that is fixed
- [-2 dB] Continuous -> binary thresholding can manufacture or destroy the rule match
- [-1 dB] 1D chords cannot separate R30 from R90 (KS 0.2623 vs 0.2627, node r0); the 2D observables must carry the verdict

*Sources:* [The neural origins of shell structure and pattern in aquatic mollusks (2009)](https://doi.org/10.1073/pnas.0810311106); [A living mesoscopic cellular automaton made of skin scales (2017)](https://doi.org/10.1038/nature22031); [The Chemical Basis of Morphogenesis (1952)](https://doi.org/10.1098/rstb.1952.0012)

### TOP-01 — Mechanism identification: a pre-registered statistic that separates reaction-diffusion from colonisation, phase separation and buckling on static patterns

*Anchor:* Turing 1952 -> Murray 1981 -> OP-11 (2026)

**Gap.** Pattern phenomenology does not identify mechanism: zebrafish stripes are cell-contact mediated (Nakamasu 2009), feather buds mechanical first (Shyer 2017), Holstein spotting indistinguishable between growing RD and Eden colonisation at p=0.91 (OP-11). No registered discriminator exists that, given only an adult pattern, says which class of generator made it with a stated error rate.

**Definition of done.** A hash-locked discriminator (candidate features: radial power-spectrum shape, Minkowski functionals, defect/junction statistics, two-point anisotropy) trained and scored on synthetic ground truth from four generators (Gray-Scott/Schnakenberg RD, Eden colonisation, Cahn-Hilliard phase separation, elastic buckling) with a pre-registered held-out accuracy threshold and published confusion matrix; then applied blind to the 10 coat-masked Holstein specimens, reported either way.

**Tool route.** models/1981-murray-bos-taurus-turing-wavelength.py (RD + Eden generators, coat masks, paired harness); models/1952-turing-morpho.py (Pearson classes); morphoeval/measure_outputs.py; predictions/ registry for the hash lock. Cahn-Hilliard and buckling generators must be written fresh (the catalog *_morphogen_condensation / *_elastic_buckling scripts are alpha-template scaffolds, scout-048).

**Deciding data.** In repo: 10 Holstein masks, RD/Eden outputs. Synthetic ground truth generated in-run. No wet data needed.

**Prior attempts.** OP-11 (static Turing rejected; growing RD vs Eden p=0.91) is the direct precursor; scout-040 OP-11 Delphi rescore pending.

**Resolvability 77/100 — RESOLVE NOW.** tool_fit 17/25: RD + Eden generators and the paired scale-matched harness exist (OP-11); phase-separation and buckling generators do not; data_in_hand 15/20: 10 coat-masked specimens and both generators in repo; synthetic truth is free; falsifiable_done 13/15: held-out accuracy vs registered threshold + confusion matrix is taste-free; feature choice is not; compute_in_repo 13/15: 2D RD/Eden/CH runs at 256^2 are minutes each on the repo stack; isolation 9/10: new generator scripts + one registry entry; touches no tracked file; time_to_verdict 6/10: two loops: build+validate discriminator, then blind application; not_blocked 4/5: OP-11 null is a warning, not a refutation of a multi-feature discriminator

**Viability 55.7% (prior 50%, +7 dB for, -6 dB against, net +1 dB, BF 1.2589, confidence 0.6) — ACHIEVABLE_IN_TWO_LOOPS_HARD_CASE_KNOWN.**

*Hypothesis scored:* A pre-registered multi-feature discriminator reaches its registered held-out accuracy on synthetic four-generator ground truth AND returns a definite class for the Holstein set.

- [+3 dB] RD and Eden generators plus the paired, scale-matched measurement harness already exist (OP-11 pipeline)
- [+2 dB] Minkowski functionals / spectral shape are standard, cheap and already separate RD labyrinths from spinodal fields in the physics literature
- [+2 dB] Synthetic ground truth makes the discriminator's own error rate decidable regardless of what the cows turn out to be
- [-3 dB] OP-11: adult patch statistics did not separate growing RD from Eden at n=9 (p=0.91) — the hard case is already known to be hard
- [-2 dB] Phase-separation and buckling generators in the catalog are template scaffolds (scout-048); both must be written and validated first
- [-1 dB] A discriminator fitted to our own four generators may not transfer to the true generator (misspecification)

*Sources:* [The Chemical Basis of Morphogenesis (1952)](https://doi.org/10.1098/rstb.1952.0012); [A Pre-pattern Formation Mechanism for Animal Coat Markings (1981)](https://doi.org/10.1016/0022-5193(81)90334-9); [Interactions between zebrafish pigment cells responsible for the generation of Turing patterns (2009)](https://doi.org/10.1073/pnas.0808622106); [Emergent cellular self-organization and mechanosensation initiate follicle pattern in the avian skin (2017)](https://doi.org/10.1126/science.aai7868); [Reconciling diverse mammalian pigmentation patterns with a fundamental mathematical model (2016)](https://doi.org/10.1038/ncomms10288); [From models to molecules: self-organized and instructed modes of developmental patterning (2026)](https://doi.org/10.1038/s41576-025-00925-z)

### TOP-09 — Turing's neglected case: is there a wave (Turing-Hopf) dispersion signature in the embryonic Ca2+ wave field?

*Anchor:* Turing 1952 Sec. 10 (oscillatory case) -> Ca2+ wave-vector pipeline

**Gap.** Turing 1952 also derived oscillatory, travelling-wave instabilities; essentially no developmental system has been shown to use one. The repo's Ca2+ spark-track data (x, y, t per event) is the only place here that can ask whether an observed wave field carries a dispersion ridge omega(k) distinct from a single driven front (omega = v k).

**Definition of done.** A pre-registered space-time spectral test on existing spark_tracks CSVs: estimate the (k, omega) power surface per embryo, fit the driven-front line, and test for a secondary ridge against phase-randomised surrogates at a locked false-positive rate; report per-embryo outcomes and the pooled verdict either way.

**Tool route.** wave-vector-analysis/ (spark_tracks.csv schema, embryo_units.py for em/s conversion); new spectral script; predictions/ registry for the surrogate threshold.

**Deciding data.** In repo: wave-vector-analysis/test_data/spark_tracks_qc_folder31.csv, _folder33.csv and analysis_results/*.csv; orientation and coordinate caveats recorded (COORDINATE_SYSTEM_WARNING.md, OP-07).

**Prior attempts.** scout-036 proposed binned AP spark spectra vs Turing k_c and a travelling-front vs static-labyrinth gate; neither run. Blind tests OP-07/08 found orientation, not wave features, carried signal.

**Resolvability 71/100 — RESOLVE NOW.** tool_fit 15/25: pipeline and unit conversion exist; the spectral estimator is new but small; data_in_hand 14/20: several spark_tracks CSVs are in the repo; falsifiable_done 10/15: surrogate-tested secondary ridge at a locked rate is checkable; ridge definition needs care; compute_in_repo 13/15: FFTs of event fields: seconds; isolation 9/10: new script + registry row; time_to_verdict 6/10: one loop; not_blocked 4/5: OP-07/08 negatives concern neighbour prediction, not dispersion

**Viability 38.7% (prior 50%, +4 dB for, -6 dB against, net -2 dB, BF 0.631, confidence 0.55) — TESTABLE_NOW_LIKELY_NULL.**

*Hypothesis scored:* The registered surrogate test returns a decisive per-embryo verdict (secondary ridge present/absent at the locked rate) on the in-repo spark-track data.

- [+2 dB] Event (x, y, t) data are in repo and space-time spectra are cheap
- [+2 dB] The null (single driven front, omega = v k) is sharp and the surrogate test is standard
- [-3 dB] Poke-triggered waves are driven, not spontaneous; a wave instability would have to appear as a weak secondary ridge, and prior blind tests found wave features carry little signal
- [-2 dB] Short, sparse event fields give poor (k, omega) resolution; the ridge criterion can slide into taste unless locked tightly
- [-1 dB] Orientation/coordinate confounds documented in OP-07 and COORDINATE_SYSTEM_WARNING.md

*Sources:* [The Chemical Basis of Morphogenesis (1952)](https://doi.org/10.1098/rstb.1952.0012); [Pattern formation mechanisms of self-organizing reaction-diffusion systems (2020)](https://doi.org/10.1016/j.ydbio.2019.10.031)

### TOP-04 — Growing domains: an independent ontogenetic stripe-count series to test the insertion law blind

*Anchor:* Kondo-Asai 1995 -> Crampin-Gaffney-Maini 1999 -> Krause 2021

**Gap.** Classical Turing conditions are neither necessary nor sufficient on growing domains and wavelength selection under growth has no closed theory (Krause et al. 2021). Empirically, Kondo-Asai's Pomacanthus series remains nearly the only ontogenetic stripe-count dataset, and the repo's Kondo stripe table is calibrated on it (circularity recorded in scout-034/038; OP-06 growing-domain solver debt still open).

**Definition of done.** A hash-locked prediction N_stripes(L) = round(L/lambda_c) with lambda_c fixed from one stage, scored on an independently sourced Pomacanthus (or other stripe-inserting fish) ontogenetic series of >=6 stages with body length metadata, each image SHA-256 pinned before counting; result reported either way.

**Tool route.** python3 morphoeval/scout_literature.py --taxon 'Pomacanthus semicirculatus' (Commons + OpenAlex); verify_image_fitness.py / the Y-junction skeleton detector for counting; predictions/ registry for the lock.

**Deciding data.** Not in repo. Needs public juvenile photographs with size metadata or a published series other than Kondo-Asai 1995 Fig. 1.

**Prior attempts.** scout-031 (Kondo ontogenetic series beyond one adult photo), scout-034/038 (circular calibration), scout-032 (Y-junction over-count) — all record the gap, none closed it.

**Resolvability 66/100 — RESOLVABLE WITH WORK.** tool_fit 16/25: literature scout + CV counter exist; the counter over-counts adult junctions (scout-032); data_in_hand 8/20: no independent series in repo; juvenile images with sizes are scarce on Commons; falsifiable_done 13/15: integer stripe counts vs a locked formula are taste-free once images are pinned; compute_in_repo 12/15: counting is cheap; 2D growing-domain solver would be the expensive add-on; isolation 9/10: new dataset folder + registry row; time_to_verdict 5/10: data hunt dominates; verdict in one loop only if a series is found; not_blocked 3/5: circularity of the current table is recorded; no refutation of an independent test

**Viability 33.4% (prior 50%, +4 dB for, -7 dB against, net -3 dB, BF 0.5012, confidence 0.6) — BLOCKED_ON_DATA_NOT_ON_TOOLS.**

*Hypothesis scored:* An independent >=6-stage series with size metadata is found and the locked insertion-law prediction is scored decisively (all stages within +-1 stripe, or not).

- [+2 dB] The insertion law is sharp: stripe count doubles when body length doubles, so +-1 tolerance is a real test
- [+2 dB] A stripe/junction detector already exists and only needs re-tuning for juveniles
- [-4 dB] No independent ontogenetic series is known; the only published one calibrated the repo's own table
- [-2 dB] Commons juvenile photographs lack body-length metadata
- [-1 dB] The existing detector over-counts on adult ROIs (scout-032/034)

*Sources:* [A reaction-diffusion wave on the skin of the marine angelfish Pomacanthus (1995)](https://doi.org/10.1038/376765a0); [Reaction and diffusion on growing domains: scenarios for robust pattern formation (1999)](https://doi.org/10.1006/bulm.1999.0131); [Introduction to 'Recent progress and open frontiers in Turing's theory of morphogenesis' (2021)](https://doi.org/10.1098/rsta.2020.0280)

### TOP-06 — Reproducible topology: why five digits every time when RD from noise gives defective labyrinths

*Anchor:* Sheth 2012 -> Raspopovic 2014 -> Green-Sharpe 2015

**Gap.** RD from random initial conditions yields labyrinths with defects; embryos give five digits almost every time. The Turing + positional-information hybrid (Hox dosage sets wavelength, FGF/Wnt gradients orient) explains cases, but there is no general, tested account of how prepattern and boundaries lock topology.

**Definition of done.** A Bmp-Sox9-Wnt three-node RD on a growing fan-shaped limb-bud domain with published parameters and gradient modulation, pre-registered to produce exactly five rays in >=95% of 200 random-IC runs (and the Hox-dose polydactyly series of Sheth 2012 on dose reduction); scored as registered.

**Tool route.** New script (no BSW model exists in models/); 2D RD solver base from models/1952-turing-morpho.py; registry row for the 95% threshold.

**Deciding data.** Published parameters (Raspopovic 2014 SOM). No wet data.

**Prior attempts.** None in repo.

**Resolvability 60/100 — RESOLVABLE WITH WORK.** tool_fit 9/25: only the generic 2D RD base exists; the three-node growing-fan model must be built; data_in_hand 9/20: parameters are published; validation target (five rays, polydactyly series) is qualitative; falsifiable_done 12/15: ray count over 200 runs is a number; domain shape and growth schedule are modelling choices; compute_in_repo 11/15: 200 2D runs on a growing domain: hours; isolation 10/10: one new script; time_to_verdict 4/10: model build + validation is more than one loop; not_blocked 5/5: no prior attempt

**Viability 28.5% (prior 50%, +3 dB for, -7 dB against, net -4 dB, BF 0.3981, confidence 0.55) — MODEL_MUST_BE_BUILT_FIRST.**

*Hypothesis scored:* A faithful BSW growing-domain model reproduces five rays in >=95% of random-IC runs and the Hox-dose polydactyly ordering.

- [+2 dB] Equations and parameters are published and the original work reproduced digit patterns in silico
- [+1 dB] The repo has a validated 2D RD base and the Lean d_c machinery
- [-3 dB] No BSW script exists; building and validating it precedes any test
- [-3 dB] The 95% topological reproducibility criterion is stronger than what the source papers claimed; growth schedule and boundary choices can decide it either way
- [-1 dB] More than one loop before a verdict

*Sources:* [Hox genes regulate digit patterning by controlling the wavelength of a Turing-type mechanism (2012)](https://doi.org/10.1126/science.1226804); [Digit patterning is controlled by a Bmp-Sox9-Wnt Turing network modulated by morphogen gradients (2014)](https://doi.org/10.1126/science.1252960); [Orientation of Turing-like Patterns by Morphogen Gradients and Tissue Anisotropies (2015)](https://doi.org/10.1016/j.cels.2015.12.001)

### TOP-05 — Scaling: does biological pattern wavelength scale with body size across taxa, against Turing's intrinsic wavelength?

*Anchor:* Turing 1952 (intrinsic wavelength) -> Werner 2015 (scaling)

**Gap.** Turing wavelength is set by kinetics and diffusion, not domain size, yet many biological patterns scale with the organism (planarian regeneration, Werner 2015; expansion-repression schemes). No cross-taxon regression of measured wavelength against body size exists. The repo's 976 *_turing_wavelength dataset folders look like such an asset but their empirical_series are alpha-template curves (x_control 0.5-5.0, r=0.999 fits), not measurements (scout-048 / scout-062 data-quality findings), so they cannot decide this.

**Definition of done.** A pre-registered log-log regression of measured pattern wavelength on body size for >=10 taxa with real, DOI-cited measurements (hair-follicle spacing, fish stripe period, digit spacing, palatal ruga spacing), slope and CI locked before extraction; verdict = slope CI excludes 0 (scaling) or excludes 1 (intrinsic) or neither.

**Tool route.** scout_literature.py for measurements; a new extraction table under datasets/; registry row. The catalog templates must NOT be used as evidence.

**Deciding data.** Not usable in repo (template artefacts). Public literature extraction required.

**Prior attempts.** scout-048 BIO-TPL-DQ-01 and scout-062 established the catalog empirical curves are template artefacts; no scaling test attempted.

**Resolvability 54/100 — RESOLVABLE WITH WORK.** tool_fit 12/25: regression is trivial; the tool gap is a trustworthy measurement table; data_in_hand 4/20: the 976 catalog series are template curves, not data; everything must be extracted fresh; falsifiable_done 12/15: locked slope CI test is taste-free; wavelength definitions across taxa are not; compute_in_repo 12/15: negligible compute; isolation 9/10: new table + registry row; time_to_verdict 4/10: literature extraction across >=10 taxa is days, not a loop; not_blocked 1/5: the obvious in-repo route is recorded as an artefact (scout-048/062)

**Viability 16.6% (prior 50%, +2 dB for, -9 dB against, net -7 dB, BF 0.1995, confidence 0.65) — DATA_ASSET_IS_AN_ARTEFACT.**

*Hypothesis scored:* A >=10-taxon measured wavelength-vs-size table is assembled and the locked slope test returns a decisive class.

- [+2 dB] The question is sharp (slope 0 vs 1) and a few real measured systems exist (fish stripes across sizes, planarian scaling, follicle spacing)
- [-5 dB] The repo's apparent asset (976 series) is void: template curves, recorded as such
- [-3 dB] Heterogeneous wavelength definitions across taxa make a single regression contestable
- [-1 dB] Growth-time and developmental-stage confounds

*Sources:* [The Chemical Basis of Morphogenesis (1952)](https://doi.org/10.1098/rstb.1952.0012); [Scaling and Regeneration of Self-Organized Patterns (2015)](https://doi.org/10.1103/PhysRevLett.114.138101); [A reaction-diffusion wave on the skin of the marine angelfish Pomacanthus (1995)](https://doi.org/10.1038/376765a0); [WNT and DKK determine hair follicle spacing through a reaction-diffusion mechanism (2006)](https://doi.org/10.1126/science.1130088)

### TOP-10 — Ecology-scale Turing: fairy circles as scale-dependent-feedback RD vs termite engineering

*Anchor:* Klausmeier 1999 -> Getzin 2016 -> Tarnita 2017

**Gap.** Namibian and Australian fairy circles are still contested between vegetation reaction-diffusion (scale-dependent feedback) and termite engineering; Tarnita 2017 shows the two are not exclusive. Pattern statistics alone (hexagonal order, pair correlation) have not settled it — the same lesson as TOP-01.

**Definition of done.** A pre-registered comparison of a vegetation RD model and a termite-territory model against public aerial-imagery pair-correlation and Voronoi statistics with locked discrimination thresholds; reported either way.

**Tool route.** No ecological model or imagery in repo; new scripts and a public data fetch would be required.

**Deciding data.** Public aerial imagery / published point patterns (Getzin et al.). Nothing in repo.

**Prior attempts.** None in repo.

**Resolvability 51/100 — RESOLVABLE WITH WORK.** tool_fit 8/25: generic RD base only; no vegetation or termite model, no imagery tooling; data_in_hand 5/20: all data external; falsifiable_done 10/15: locked thresholds are checkable but the mechanisms are not exclusive; compute_in_repo 10/15: feasible compute once models exist; isolation 10/10: new files only; time_to_verdict 3/10: several loops; not_blocked 5/5: no prior attempt

**Viability 28.5% (prior 50%, +2 dB for, -6 dB against, net -4 dB, BF 0.3981, confidence 0.5) — OUTSIDE_THE_REPO_AND_LIKELY_INDECISIVE.**

*Hypothesis scored:* Pattern statistics from public imagery decisively favour one mechanism at the locked thresholds.

- [+2 dB] Public imagery and standard point-pattern statistics exist
- [-4 dB] No repo data or model; everything must be built
- [-2 dB] Mechanisms are not mutually exclusive (Tarnita 2017), so pattern statistics are unlikely to be decisive — the TOP-01 lesson

*Sources:* [Discovery of fairy circles in Australia supports self-organization theory (2016)](https://doi.org/10.1073/pnas.1522130113); [A theoretical foundation for multi-scale regular vegetation patterns (2017)](https://doi.org/10.1038/nature20801)

### TOP-08 — Synthetic realisation: a robust, deterministic periodic Turing pattern in eukaryotic cells

*Anchor:* Sekine 2018 -> Tica 2024

**Gap.** Bacterial three-node circuits now give reproducible concentric stripes but remain condition-sensitive (Tica 2024); the mammalian Nodal-Lefty circuit most likely produced solitary, not periodic, patterns (Sekine 2018). A robust eukaryotic periodic Turing pattern has not been built.

**Definition of done.** Wet-lab: a eukaryotic circuit producing periodic patterns with pre-registered wavelength reproducibility across >=3 biological replicates. The repo can only contribute the Turing-vs-solitary regime map of the published circuit model.

**Tool route.** None adequate: no wet lab. Modelling satellite only (regime map of the Sekine competitive-inhibition model).

**Deciding data.** Not obtainable here.

**Prior attempts.** None in repo.

**Resolvability 37/100 — HARD HERE.** tool_fit 3/25: no instrument in this repo can run a wet-lab experiment; data_in_hand 4/20: only the published model equations; falsifiable_done 12/15: replicate wavelength reproducibility is checkable — elsewhere; compute_in_repo 2/15: the deciding evidence is not computational; isolation 10/10: nothing to conflict with; time_to_verdict 1/10: months to years, in a lab; not_blocked 5/5: no prior attempt

**Viability 11.2% (prior 50%, +1 dB for, -10 dB against, net -9 dB, BF 0.1259, confidence 0.8) — NOT_WITH_OUR_TOOLS.**

*Hypothesis scored:* This repo brings the synthetic eukaryotic Turing problem to a decisive verdict.

- [+1 dB] A regime map of the published Sekine model (Turing vs solitary) is a tractable modelling satellite
- [-10 dB] The deciding evidence is wet-lab; none of the repo's instruments reach it

*Sources:* [Synthetic mammalian pattern formation driven by differential diffusivity of Nodal and Lefty (2018)](https://doi.org/10.1038/s41467-018-07847-x); [A three-node Turing gene circuit forms periodic spatial patterns in bacteria (2024)](https://doi.org/10.1016/j.cels.2024.11.002)

## Artefacts

- Ledger: `morphoeval/audits/2026-10-27-turing-open-problems-assessment.yaml`
- Graph feed: `docs/turing-open-problems.json` (injected by `docs/build_problem_graph_html.py`)
- Leaderboard input: `morphoeval/problem_scouts/scout-101.json` (ground `turing-forward`, lens `literature`)
- Nanopublications: `morphoeval/kg/2026-10-27-turing-open-problems-assessment/` (one `pk:Claim` per problem carrying a `pk:LLMClaimAssessment` for resolvability and a `pk:BayesianAssessment` for viability); trusty URIs in `published.json` after signing.

## Three-provider Delphi panel (2026-10-27) — supersedes the provisional viabilities above

Three assessors scored the same blinded packet in parallel: GPT 5.6 Sol Medium (OpenAI), Gemini 4 Argon Medium (Google), Claude Fable 5.1 Thinking High (fresh context) (Anthropic). Blinding: the packet (`morphoeval/audits/2026-10-27-turing-open-problems-panel/packet.json`) carried no scores and the assessors were told which files not to open; exposure disclosures: gpt-5.6-sol: none; gemini-4-argon: none; claude-fable-5-1: None of the listed contaminating files were opened. One filename-only grep hit (rg -l for 'cahn.hilliard') listed morphoeval/build_turing_open_problems.py among.

Resolvability totals are recomputed from each assessor's factor scores; viability is pooled in log-odds (equal priors 50%, so the pooled net dB is the mean). The spread is part of the result. Raw returns are vendored next to the packet; the signed superseding nanopubs are in `morphoeval/kg/2026-10-27-turing-open-problems-panel/`.

| Rank | ID | Problem | Resolvability panel mean (range) | Pooled viability (range) | Pooled net dB | Per-assessor net dB (GPT / Gemini / Fable) |
|---|---|---|---|---|---|---|
| 1 | TOP-07 | When does continuous reaction-diffusion collapse to a discrete automaton? Meinhardt mantle PDE vs the Lean-proved Rule 30 gate | **89.7** (86–96) · RESOLVE NOW | **77.33%** (71.53–79.92%) | +5.33 | +4 / +6 / +6 |
| 2 | TOP-09 | Turing's neglected case: is there a wave (Turing-Hopf) dispersion signature in the embryonic Ca2+ wave field? | **79.7** (70–88) · RESOLVE NOW | **57.6%** (38.69–69.12%) | +1.33 | +2.5 / +3.5 / -2 |
| 3 | TOP-03 | The fine-tuning objection: replicate the two-node vs three-node Turing parameter-volume ratio and the 1%-uncertainty retention claim | **78.7** (71–93) · RESOLVE NOW | **69.95%** (61.31–79.92%) | +3.67 | +2 / +6 / +3 |
| 4 | TOP-01 | Mechanism identification: a pre-registered statistic that separates reaction-diffusion from colonisation, phase separation and buckling on static patterns | **76** (70–84) · RESOLVE NOW | **59.5%** (55.73–61.31%) | +1.67 | +2 / +1 / +2 |
| 5 | TOP-02 | The diffusion-ratio problem: which molecularly identified Turing systems can be two-species at all? | **74.3** (61–89) · RESOLVE NOW | **66.61%** (58.55–78.01%) | +3 | +1.5 / +5.5 / +2 |
| 6 | TOP-06 | Reproducible topology: why five digits every time when RD from noise gives defective labyrinths | **69.7** (62–79) · RESOLVABLE WITH WORK | **49.02%** (35.99–71.53%) | -0.17 | -2.5 / -2 / +4 |
| 7 | TOP-05 | Scaling: does biological pattern wavelength scale with body size across taxa, against Turing's intrinsic wavelength? | **60.7** (53–75) · RESOLVABLE WITH WORK | **39.62%** (26.19–61.31%) | -1.83 | -4.5 / +2 / -3 |
| 8 | TOP-04 | Growing domains: an independent ontogenetic stripe-count series to test the insertion law blind | **57.7** (49–68) · RESOLVABLE WITH WORK | **22.67%** (16.63–28.47%) | -5.33 | -4 / -5 / -7 |
| 9 | TOP-10 | Ecology-scale Turing: fairy circles as scale-dependent-feedback RD vs termite engineering | **53** (49–61) · RESOLVABLE WITH WORK | **24.75%** (24.03–26.19%) | -4.83 | -5 / -4.5 / -5 |
| 10 | TOP-08 | Synthetic realisation: a robust, deterministic periodic Turing pattern in eukaryotic cells | **20** (14–26) · NOT WITH OUR TOOLS (yet) | **8.18%** (5.94–10.09%) | -10.5 | -12 / -9.5 / -10 |

### Packet corrections found by the panel

- **TOP-06**: The packet's tool_route said 'no BSW model exists in models/'. False: models/2014-raspopovic-bmp-sox9-wnt-digits.py (2,066 lines, Pass 3: Bmp-Sox9-Wnt 3-node RD on a growing polar-fan limb-bud mesh, Crampin-Gaffney-Maini dilution) and models/2012-sheth-hox-turing-polydactyly.py (1,619 lines) already exist. Caught by the Claude Fable panellist, verified by the aggregator. The provisional tool_fit (9/25) under-priced this problem; the panel scores stand as recorded, and a re-score with the corrected route is the next action on TOP-06.

### Per-assessor verdicts and cruxes

**TOP-07**
- GPT 5.6 Sol Medium: resolvability 87, posterior 71.53% (+4 dB, confidence 0.91), `HIGHLY_DECISIVE`. This is the strongest in-repo route: it can decisively reject as well as support the proposed reduction, and current chirality diagnostics already favor rejection.
- Gemini 4 Argon Medium: resolvability 96, posterior 79.92% (+6 dB, confidence 0.91), `HIGHLY_TRACTABLE_EXPECTED_NEGATIVE`. The crux is fundamental symmetry: Meinhardt-Klingler's 1D diffusion operator d^2/dx^2 is reflection-symmetric (left-right invariant), whereas Rule 30 breaks left-right symmetry via a_{i-1} XOR (a_i OR a_{i+1}). Thus the transducer test will reach a fast, decisive verdict—almost certainly proving that isotropic Meinhardt PDE reduces to a symmetric elementary CA class (like Rule 90/18) rather than chiral Rule 30 unless oblique wave travel or asymmetric coupling is present.
- Claude Fable 5.1 Thinking High (fresh context): resolvability 86, posterior 79.92% (+6 dB, confidence 0.75), `DECISIVE_LIKELY_NEGATIVE_IN_REPO`. Crux: a symmetric PDE should map to a symmetric (Rule 90-like or additive) rule, so expect a decisive negative on Rule 30; lock the coarse-graining scale and threshold in the registry before running.

**TOP-09**
- GPT 5.6 Sol Medium: resolvability 81, posterior 64.01% (+2.5 dB, confidence 0.79), `LIKELY_DECISIVE`. The route can return a calibrated detection result, but an absent ridge should only count as decisive if power against simulated secondary ridges is preregistered.
- Gemini 4 Argon Medium: resolvability 88, posterior 69.12% (+3.5 dB, confidence 0.85), `READY_IN_REPO_SPECTRAL_TEST`. The crux is converting sparse, tracked Ca2+ spark centroids into a well-conditioned space-time density field along ap_norm without introducing binning artifacts before computing S(k, omega). Note that COORDINATE_SYSTEM_WARNING.md is cited in the packet and AGENTS.md, though that file path is not currently present in wave-vector-analysis/ (the relevant caveats live in head_tail_comparison.csv and mask_tiff_mismatch_cases.csv).
- Claude Fable 5.1 Thinking High (fresh context): resolvability 70, posterior 38.69% (-2 dB, confidence 0.55), `PROCEDURALLY_DECISIVE_UNDERPOWERED`. Crux: the test will return a number, but with ~3 embryos and 1 fps sparse events the 'absent' outcome is weak evidence; pre-register a power analysis on synthetic Turing-Hopf fields so a negative is interpretable.

**TOP-03**
- GPT 5.6 Sol Medium: resolvability 72, posterior 61.31% (+2 dB, confidence 0.74), `LIKELY_DECISIVE`. The result can be decisive if the published sampling measure and topology are reproduced rather than replaced by convenient repo defaults.
- Gemini 4 Argon Medium: resolvability 93, posterior 79.92% (+6 dB, confidence 0.92), `IMMEDIATE_COMPUTATIONAL_WIN`. This is the cleanest computational replication in the packet because it requires only vectorized Jacobian eigenvalue algebra and published parameter bounds. The packet should explicitly note that finding non-trivial positive steady states for Hill kinetics at each Latin-hypercube sample is part of the Scholes/Tica denominator definition.
- Claude Fable 5.1 Thinking High (fresh context): resolvability 71, posterior 66.61% (+3 dB, confidence 0.65), `DECISIVE_IF_PUBLISHED_KINETICS_ADOPTED`. Crux: adopt Tica's exact ODEs and bounds, not the repo's Gray-Scott; otherwise the ratio test is a comparison of different models. The done should specify log-uniform sampling as in the paper.

**TOP-01**
- GPT 5.6 Sol Medium: resolvability 70, posterior 61.31% (+2 dB, confidence 0.72), `LIKELY_DECISIVE`. The synthetic benchmark is closeable, but the packet overstates data-in-hand: the checked dataset has pinned references rather than the ten image files. A reject or abstention rule is also needed to make the Holstein class assignment scientifically definite.
- Gemini 4 Argon Medium: resolvability 84, posterior 55.73% (+1 dB, confidence 0.78), `SYNTHETIC_SEPARABLE_EMPIRICAL_AMBIGUOUS`. The crux is whether higher-order morphological statistics (Minkowski functionals, chord-length distributions, or boundary curvature spectra) can break the OP-11 degeneracy between growing-domain RD and Eden colonisation after binary Otsu thresholding. The packet understates how severely the low patch count per Holstein flank limits spectral and defect-based features on the empirical specimens.
- Claude Fable 5.1 Thinking High (fresh context): resolvability 74, posterior 61.31% (+2 dB, confidence 0.6), `SYNTHETIC_LEG_DECISIVE_HOLSTEIN_FRAGILE`. Crux: the synthetic confusion matrix is cheap and decisive; the blind Holstein application is where the hypothesis will be fought over. Packet says CH/buckling generators 'must be written fresh' but non-template phase-separation (2002-sumper) and buckling (1975-richman) scripts exist to adapt.

**TOP-02**
- GPT 5.6 Sol Medium: resolvability 73, posterior 58.55% (+1.5 dB, confidence 0.67), `PLAUSIBLY_DECISIVE`. A useful census is achievable, but class rules must distinguish unmeasured transport from evidence for a third immobile node.
- Gemini 4 Argon Medium: resolvability 89, posterior 78.01% (+5.5 dB, confidence 0.88), `HIGHLY_TRACTABLE_DECISIVE`. The crux is distinguishing truly unmeasured in vivo diffusivities from indirect effective range estimates, since all three classification buckets are already guaranteed to be populated by Nodal/Lefty, Bmp-Sox9-Wnt, and WNT/DKK. Note that d_c = 3.9271 is specific to Turing's 1952 numerical Jacobian example, whereas each biological system has its own kinetic Jacobian bounds on D_inh/D_act.
- Claude Fable 5.1 Thinking High (fresh context): resolvability 61, posterior 61.31% (+2 dB, confidence 0.55), `CENSUS_REACHABLE_CLASSES_DISPUTABLE`. Crux: the table will be assembled, but 'two-node-feasible' needs a per-system Jacobian or a stated generic bound; the packet's reliance on the GS d_c is the weak link.

**TOP-06**
- GPT 5.6 Sol Medium: resolvability 62, posterior 35.99% (-2.5 dB, confidence 0.78), `DOUBTFUL`. The criterion is decisive, but substantial model reconstruction and underdetermined boundary choices make a decisive faithful result uncertain.
- Gemini 4 Argon Medium: resolvability 68, posterior 38.69% (-2 dB, confidence 0.74), `SOLVER_HEAVY_LIKELY_REFUTATION`. The crux is that achieving >=95% topological locking of exactly 5 rays from unseeded random initial conditions on a finite-difference fan grid is notoriously sensitive to boundary no-flux discretization and growth rate schedules. Even if the script finishes, the hypothesis as stated (>=95% five-ray success) will very likely be refuted unless pre-existing AP polarity (Shh/Gli3 gradient) is also included.
- Claude Fable 5.1 Thinking High (fresh context): resolvability 79, posterior 71.53% (+4 dB, confidence 0.65), `DECISIVE_SIM_FAITHFULNESS_CONTESTED`. Crux: the packet's tool_route is out of date; BSW + Hox-dose scripts exist. Likely outcome is failure of the 95% gate (RD from noise rarely locks topology), which is still decisive provided the model is accepted as faithful.

**TOP-05**
- GPT 5.6 Sol Medium: resolvability 53, posterior 26.19% (-4.5 dB, confidence 0.9), `UNLIKELY_DECISIVE`. The decision rule must be repaired before preregistration: test slope=0 for intrinsic wavelength and slope=1 for proportional scaling, with an explicit rule for rejecting neither or both.
- Gemini 4 Argon Medium: resolvability 75, posterior 61.31% (+2 dB, confidence 0.76), `FEASIBLE_LITERATURE_META_ANALYSIS`. The crux is normalizing body size to the exact embryonic stage when the pattern is laid down rather than adult size; otherwise an embryonic mouse hair follicle and an adult angelfish stripe lie on incomparable size axes. The packet rightly forbids the 976 catalog template folders but should specify that length L must be measured at the developmental stage of pattern initiation.
- Claude Fable 5.1 Thinking High (fresh context): resolvability 54, posterior 33.39% (-3 dB, confidence 0.5), `TABLE_HEAVY_DESIGN_CONTESTABLE`. Crux: the regression is cheap but the inferential design (cross-taxon, mixed pattern types) does not cleanly test Turing's intrinsic wavelength; restricting to one pattern type (e.g. fish stripe period) would sharpen it at the cost of taxon count.

**TOP-04**
- GPT 5.6 Sol Medium: resolvability 56, posterior 28.47% (-4 dB, confidence 0.88), `UNLIKELY_DECISIVE`. The bottleneck is not computation but finding a genuinely independent, size-annotated ontogenetic series.
- Gemini 4 Argon Medium: resolvability 68, posterior 24.03% (-5 dB, confidence 0.82), `DATA_STARVED_HIGH_RISK`. The crux is data availability: wildlife photographs on Wikimedia Commons and iNaturalist almost never include ruler scale bars or standard lengths across six juvenile-to-adult stages. Switching the taxon to zebrafish (Danio rerio SL vs interstripe count in Parichy's staging series) or archival museum fish plates with scale bars is much more realistic than sourcing Pomacanthus from Commons.
- Claude Fable 5.1 Thinking High (fresh context): resolvability 49, posterior 16.63% (-7 dB, confidence 0.65), `DATA_ACQUISITION_BLOCKS_VERDICT`. Crux: the series does not exist in any named public place; without it the route stalls before scoring. The packet omits the existing models/1995-kondo-pomacanthus-stripe-insertion.py and its polar-sector solver, which would be the natural prediction engine.

**TOP-10**
- GPT 5.6 Sol Medium: resolvability 49, posterior 24.03% (-5 dB, confidence 0.87), `UNLIKELY_DECISIVE`. The binary hypothesis is stronger than the biology supports; a mixture-model or intervention-linked dataset would be needed for mechanism identification.
- Gemini 4 Argon Medium: resolvability 61, posterior 26.19% (-4.5 dB, confidence 0.84), `EQUIFINAL_STATISTICS_UNLIKELY_DECISIVE`. The crux—explicitly proved by Tarnita et al. (2017)—is that macro-scale fairy-circle centroids from both termite territory competition and scale-dependent water-vegetation feedback converge to the same hexagonal packing limit (6 neighbors, identical g(r) first peak). Only multi-scale sub-meter vegetation patterning between circles or temporal birth-death trajectories can separate the two mechanisms.
- Claude Fable 5.1 Thinking High (fresh context): resolvability 49, posterior 24.03% (-5 dB, confidence 0.65), `STATISTICS_NON_DISCRIMINATING`. Crux: the packet's own gap statement undermines the done — static spatial statistics do not separate RD from termite engineering; a decisive test would need dynamics (e.g. rainfall-response or gap-recovery series), which are not in scope here.

**TOP-08**
- GPT 5.6 Sol Medium: resolvability 20, posterior 5.94% (-12 dB, confidence 0.99), `NOT_RESOLVABLE`. This is a clear category error for repo-only resolution; modelling can support experimental design but cannot deliver the stated verdict.
- Gemini 4 Argon Medium: resolvability 26, posterior 10.09% (-9.5 dB, confidence 0.98), `WET_LAB_BLOCKED_OUT_OF_SCOPE`. This problem serves as the negative control calibration anchor for the panel: its definition of done explicitly requires wet-lab eukaryotic synthetic biology across >=3 replicates. Only if the problem were reframed purely around the theoretical Turing-vs-solitary bifurcation boundary of the Sekine (2018) competitive-inhibition PDE could this repository close it.
- Claude Fable 5.1 Thinking High (fresh context): resolvability 14, posterior 9.09% (-10 dB, confidence 0.95), `OUT_OF_SCOPE_WET_LAB`. Crux: scope. Keep as a tracked external problem; the only in-repo contribution is a modelling regime map, which should be registered as a separate, non-deciding satellite.

