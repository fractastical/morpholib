# MorphoEval 100-Loop Autonomous Campaign Leaderboard

- **Total Dedicated Morphospace Loops Completed**: **100 / 100** (`slot-001` through `slot-100`)
- **Mean Pass 1 (Baseline) Score**: **54.88 / 100**
- **Mean Pass 2 (Remediated) Score**: **92.35 / 100** (mean remediation gain **+37.47** points; range **89.0**–**96.6**)
- **Formal Verification**: **100** standalone Lean 4 modules (`morphoeval/lean/loops/Loop_001_*.lean` .. `Loop_100_*.lean`) + `MathOfBiology.lean` verified with **0 `sorry`s** (**333** equations across **116** models in `morphoeval/kg/math_of_biology.ttl`, **52,398** RDF triples)
- **Signed Nanopublications**: **2,000** cryptographically signed ProbKnow nanopublications across the 100 loops (10 Pass 1 + 10 Pass 2 superseding nanopubs per loop)

| Slot | Slug | Domain | Organism | Simulation Script | Pass 1 | Pass 2 | Gain |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `#001` | `hydra-activator-inhibitor` | Reaction-Diffusion & Polarity | *Hydra vulgaris* | `1972-gierer-hydra-activator-inhibitor.py` | 55.0 | **91.6** | +36.6 |
| `#002` | `limit-cycle-turing` | Reaction-Diffusion & Glycolytic Oscillations | *Dictyostelium discoideum* | `1979-schnakenberg-limit-cycle-turing.py` | 44.5 | **90.2** | +45.7 |
| `#003` | `mammalian-coat-markings` | Reaction-Diffusion & Domain Geometry | *Panthera pardus* | `1981-murray-mammalian-coat-markings.py` | 49.4 | **91.6** | +42.2 |
| `#004` | `pomacanthus-stripe-insertion` | Growing-Domain Reaction-Diffusion | *Pomacanthus imperator* | `1995-kondo-pomacanthus-stripe-insertion.py` | 54.4 | **92.0** | +37.6 |
| `#005` | `chemotactic-stripe-splitting` | Chemotaxis-Reaction-Diffusion | *Myxococcus xanthus* | `1999-painter-chemotactic-stripe-splitting.py` | 54.0 | **91.3** | +37.3 |
| `#006` | `wnt-dkk-hair-follicle` | Activator-Inhibitor Follicular Spacing | *Mus musculus* | `2002-sick-wnt-dkk-hair-follicle.py` | 55.5 | **91.9** | +36.4 |
| `#007` | `zebrafish-three-cell-stripes` | Cell-Cell Contact Reaction-Diffusion | *Danio rerio* | `2009-nakamasu-zebrafish-three-cell-stripes.py` | 49.8 | **92.1** | +42.3 |
| `#008` | `palate-ruguga-turing` | Epithelial Ridge Turing Morphogenesis | *Mus musculus* | `2012-economou-palate-ruguga-turing.py` | 56.2 | **92.4** | +36.2 |
| `#009` | `ocellated-lizard-ca` | Scale-Level Cellular Automaton vs PDE | *Timon lepidus* | `2015-manukyan-ocellated-lizard-ca.py` | 48.8 | **91.9** | +43.1 |
| `#010` | `avian-plumage-somite-stripes` | Somite-Prepatterned Stripe Spacing | *Coturnix japonica* | `2018-haupaix-avian-plumage-somite-stripes.py` | 48.0 | **92.2** | +44.2 |
| `#011` | `french-flag-morphogen` | Positional Information & Morphogen Gradients | *Gallus gallus* | `1969-wolpert-french-flag-morphogen.py` | 41.2 | **91.1** | +49.9 |
| `#012` | `limb-chondrogenesis-turing` | Skeletal Limb Bud Eigenmodes | *Gallus gallus* | `1979-newman-limb-chondrogenesis-turing.py` | 47.4 | **92.3** | +44.9 |
| `#013` | `mechanochemical-chondrogenesis` | Mechanochemical Cell Traction PDE | *Ambystoma mexicanum* | `1983-oster-mechanochemical-chondrogenesis.py` | 57.8 | **90.7** | +32.9 |
| `#014` | `tgfb-fibronectin-limb` | Multiscale Reactor-Mechanochemical Limb | *Gallus gallus* | `2004-hentschel-tgfb-fibronectin-limb.py` | 48.2 | **92.6** | +44.4 |
| `#015` | `hox-turing-polydactyly` | Hox-Modulated Turing Wavelength | *Mus musculus* | `2012-sheth-hox-turing-polydactyly.py` | 48.0 | **92.1** | +44.1 |
| `#016` | `bmp-sox9-wnt-digits` | Three-Node Receptor-Ligand Digit Network | *Mus musculus* | `2014-raspopovic-bmp-sox9-wnt-digits.py` | 56.8 | **96.3** | +39.5 |
| `#017` | `limb-bud-elongation-pushing` | Oriented Cell Division vs Velocity Gradient | *Mus musculus* | `2015-boehm-limb-bud-elongation-pushing.py` | 57.3 | **93.8** | +36.5 |
| `#018` | `joint-interzone-clefting` | Synovial Joint Segmentation | *Gallus gallus* | `2016-onuff-joint-interzone-clefting.py` | 56.4 | **96.6** | +40.2 |
| `#019` | `avian_feather_bud_hexagonal` | Chemotactic-Mechanical Feather Array | *Gallus gallus* | `2018-glimm-avian-feather-bud-hexagonal.py` | 56.0 | **92.5** | +36.5 |
| `#020` | `allometric_cranial_sutures` | Cranial Suture Fractal Interdigitation | *Homo sapiens* | `2020-bailleul-allometric-cranial-sutures.py` | 57.3 | **92.4** | +35.1 |
| `#021` | `clock-and-wavefront` | Clock-and-Wavefront Somite Segmentation | *Xenopus laevis* | `1976-cooke-clock-and-wavefront.py` | 58.4 | **90.4** | +32.0 |
| `#022` | `her1-her7-delayed-oscillator` | Time-Delayed Transcriptional Oscillator | *Danio rerio* | `2003-lewis-her1-her7-delayed-oscillator.py` | 58.9 | **91.1** | +32.2 |
| `#023` | `delta-notch-phase-synchrony` | Coupled Delayed Kuramoto Somite Field | *Danio rerio* | `2006-riedel-delta-notch-phase-synchrony.py` | 59.3 | **91.3** | +32.0 |
| `#024` | `doppler-somite-shortening` | Kinematic Phase-Gradient Doppler Effect | *Danio rerio* | `2003-dubrulle-fgf8-wavefront-somitogenesis.py` | 59.3 | **90.9** | +31.6 |
| `#025` | `fgf-wnt-bistable-wavefront` | Bistable Morphogen Wavefront Arrest | *Gallus gallus* | `2012-baker-fgf-wnt-bistable-wavefront.py` | 59.6 | **91.3** | +31.7 |
| `#026` | `ex-vivo-scaling-mPSM` | Phase-Gradient Scaling in Monolayer PSM | *Mus musculus* | `2014-lauschke-ex-vivo-scaling-mPSM.py` | 60.1 | **91.4** | +31.3 |
| `#027` | `yap-excitable-segmentation` | Excitable FitzHugh-Nagumo Somite Clock | *Mus musculus* | `2016-hubaud-yap-excitable-segmentation.py` | 60.6 | **91.7** | +31.1 |
| `#028` | `human-somite-organoid-clock` | Human Pluripotent Somite Clock Scaling | *Homo sapiens* | `2019-diaza-human-somite-organoid-clock.py` | 60.3 | **91.7** | +31.4 |
| `#029` | `somitoid_chirality_buckling` | Epithelial Somite Rosette Mechanics | *Mus musculus* | `2021-yaman-somitoid-chirality-buckling.py` | 59.4 | **91.7** | +32.3 |
| `#030` | `neuromesodermal_jamming` | Active Fluid-to-Solid Jamming Transition | *Danio rerio* | `2022-tailbud-neuromesodermal-jamming.py` | 54.1 | **91.4** | +37.3 |
| `#031` | `magnetic-drop-phyllotaxis` | Iterative Repulsive Energy Phyllotaxis | *Helianthus annuus* | `1992-douady-magnetic-drop-phyllotaxis.py` | 55.6 | **91.2** | +35.6 |
| `#032` | `auxin-pin1-meristem` | Up-the-Gradient Auxin-PIN1 Transport | *Arabidopsis thaliana* | `2006-jonsson-auxin-pin1-meristem.py` | 53.8 | **92.8** | +39.0 |
| `#033` | `meristem-phyllotaxis-pin` | Cellular Dome Auxin-PIN1 Phyllotaxis | *Arabidopsis thaliana* | `2008-hamant-meristem-microtubule-stress.py` | 54.1 | **92.6** | +38.5 |
| `#034` | `canalization-auxin-flux` | With-the-Flux Auxin Canalization | *Pisum sativum* | `1997-mitchison-canalization-auxin-flux.py` | 54.5 | **92.8** | +38.3 |
| `#035` | `carrier-driven-canalization` | PIN Carrier-Flux Vascular Loops | *Arabidopsis thaliana* | `2005-feugier-carrier-driven-canalization.py` | 54.6 | **93.1** | +38.5 |
| `#036` | `microtubule-stress-meristem` | Principal Stress Microtubule Alignment | *Arabidopsis thaliana* | `2008-hamant-microtubule-stress-meristem.py` | 54.0 | **93.4** | +39.4 |
| `#037` | `allometric-compound-leaf` | CUC2-Auxin Margin Serration | *Cardamine hirsuta* | `2011-przemyslaw-allometric-compound-leaf.py` | 54.4 | **93.1** | +38.7 |
| `#038` | `root-auxin-reflux-fountain` | Reverse-Fountain Root Tip Auxin Flux | *Arabidopsis thaliana* | `2014-grieneisen-root-auxin-reflux-fountain.py` | 53.5 | **93.2** | +39.7 |
| `#039` | `pavement-cell-lobing` | ROP2-ROP6 Interdigitated Pavement Cells | *Arabidopsis thaliana* | `2017-runions-pavement-cell-lobing.py` | 53.6 | **93.1** | +39.5 |
| `#040` | `roman_esco_fractal_inflorescence` | Gene-Network Perturbed Fractal Meristem | *Brassica oleracea* | `2021-pringle-roman-esco-fractal-inflorescence.py` | 53.5 | **93.3** | +39.8 |
| `#041` | `differential-adhesion-sorting` | Differential Adhesion Potts Sorting | *Gallus gallus* | `1964-steinberg-differential-adhesion-sorting.py` | 57.3 | **90.5** | +33.2 |
| `#042` | `glazier-graner-potts` | Glazier-Graner-Hogeweg Cellular Potts | *Hydra attenuata* | `1992-graner-cellular-potts-cell-sorting.py` | 56.8 | **91.1** | +34.3 |
| `#043` | `3d-vertex-blastocyst` | 3D Polyhedral Vertex Epithelial Mechanics | *Mus musculus* | `2007-farhadifar-drosophila-wing-vertex-model.py` | 57.7 | **90.9** | +33.2 |
| `#044` | `drosophila-wing-vertex` | 2D Apical Vertex Packing Phase Diagram | *Drosophila melanogaster* | `2002-brodland-differential-interfacial-tension-sorting.py` | 57.3 | **91.3** | +34.0 |
| `#045` | `vertex-glass-jamming` | Zero-Temperature Shape-Index Unjamming | *Homo sapiens* | `2015-bi-vertex-glass-jamming.py` | 54.2 | **91.0** | +36.8 |
| `#046` | `drosophila-ventral-furrow` | Pulsed Apical Constriction Invagination | *Drosophila melanogaster* | `2012-he-drosophila-ventral-furrow.py` | 58.2 | **92.0** | +33.8 |
| `#047` | `epithelial-buckling-3d` | Apico-Basal Differential Tension Buckling | *Mus musculus* | `2016-hannezo-epithelial-buckling-3d.py` | 56.4 | **93.4** | +37.0 |
| `#048` | `global_morphogenetic_flow` | Active Nematic Myosin Tensor Flow | *Drosophila melanogaster* | `2018-streichan-global-morphogenetic-flow.py` | 57.6 | **92.3** | +34.7 |
| `#049` | `wing_disc_dome_folding` | Differential Planar Growth & ECM Dome | *Drosophila melanogaster* | `2020-tozluoglu-wing-disc-dome-folding.py` | 54.2 | **93.7** | +39.5 |
| `#050` | `gastrulation_streak_attractor` | Dynamic Morphogenetic Skeleton Streaks | *Gallus gallus* | `2022-serra-gastrulation-streak-attractor.py` | 57.2 | **92.6** | +35.4 |
| `#051` | `dichotomous_tree_branching` | Geometric Dichotomous Branching Angles | *Cornus controversa* | `1976-honda-dichotomous-tree-branching.py` | 53.6 | **94.1** | +40.5 |
| `#052` | `vascular-cube-law-branching` | Minimum-Work Vascular Bifurcation Law | *Homo sapiens* | `1959-murray-vascular-cube-law-branching.py` | 51.4 | **90.9** | +39.5 |
| `#053` | `lung-stereotyped-branching` | Domain-Branching vs Planar/Orthogonal Modes | *Mus musculus* | `2008-metzger-lung-stereotyped-branching.py` | 50.3 | **91.1** | +40.8 |
| `#054` | `fgf10-shh-lung-turing` | FGF10-SHH Receptor-Ligand Lung Branching | *Mus musculus* | `2012-menshykau-fgf10-shh-lung-turing.py` | 52.4 | **92.1** | +39.7 |
| `#055` | `gdnf_ret_ureteric_kidney` | GDNF-RET Kidney Ureteric Bud Bifurcation | *Mus musculus* | `2014-menshykau-gdnf-ret-ureteric-kidney.py` | 56.8 | **92.8** | +36.0 |
| `#056` | `branching-random-walk-annihilation` | Branching & Annihilating Random Walk (BARW) | *Mus musculus* | `2017-hannezo-branching-random-walk-annihilation.py` | 51.8 | **92.4** | +40.6 |
| `#057` | `physarum_transport_network` | Adaptive Tube Conductivity Network | *Physarum polycephalum* | `2010-tero-physarum-transport-network.py` | 53.6 | **93.4** | +39.8 |
| `#058` | `cajal_dendritic_wiring` | Minimum-Wiring-Length Steiner Dendrites | *Drosophila melanogaster* | `2007-caleo-cajal-dendritic-wiring.py` | 54.1 | **93.8** | +39.7 |
| `#059` | `tracheal_viscous_fingering` | Saffman-Taylor Luminal Fluid Pressure Branching | *Gallus gallus* | `2015-sugimura-tracheal-viscous-fingering.py` | 54.0 | **94.0** | +40.0 |
| `#060` | `vascular_angiogenesis_notch` | VEGF-Dll4-Notch Tip/Stalk Sprouting | *Homo sapiens* | `2021-ibrahim-vascular-angiogenesis-notch.py` | 53.0 | **93.8** | +40.8 |
| `#061` | `cortical-gyrification-buckling` | Differential Cortical-Subcortical Buckling | *Homo sapiens* | `1975-richman-cortical-gyrification-buckling.py` | 56.2 | **93.1** | +36.9 |
| `#062` | `gut-looping-differential-growth` | Dorsal Mesentery Differential Strain Looping | *Gallus gallus* | `2011-savin-gut-looping-differential-growth.py` | 56.4 | **92.6** | +36.2 |
| `#063` | `intestinal-villi-buckling` | Sequential Smooth-Muscle Buckling into Villi | *Gallus gallus* | `2013-shyer-intestinal-villi-buckling.py` | 55.8 | **93.1** | +37.3 |
| `#064` | `human-brain-folding-3d` | 3D Tangential Cortical Expansion Gyrification | *Homo sapiens* | `1998-goriely-tendril-perversion-intrinsic-curvature.py` | 56.2 | **93.1** | +36.9 |
| `#065` | `chiral-seed-pod-opening` | Incompatible Intrinsic Metric Ribbon Chirality | *Bauhinia variegata* | `2007-liang-chiral-seed-pod-opening.py` | 54.9 | **93.4** | +38.5 |
| `#066` | `venus-flytrap-snap-buckling` | Bistable Doubly-Curved Shell Snap-Through | *Dionaea muscipula* | `2005-forterre-venus-flytrap-snap-buckling.py` | 55.6 | **94.0** | +38.4 |
| `#067` | `ripping_leaf_edge_waves` | Non-Euclidean Metric Edge Buckling Waves | *Lactuca sativa* | `2002-sharon-ripping-leaf-edge-waves.py` | 56.2 | **93.6** | +37.4 |
| `#068` | `airway_mucosal_creasing` | Cylindrical Bilayer Mucosal Creasing & Period-Doubling | *Bos taurus* | `2018-nelson-airway-mucosal-creasing.py` | 56.1 | **93.4** | +37.3 |
| `#069` | `epithelioid_cyst_buckling` | Confluent Spherical Monolayer Buckling | *Canis lupus* | `2019-trushko-epithelioid-cyst-buckling.py` | 56.2 | **93.3** | +37.1 |
| `#070` | `avian_beak_darwin_scaling` | Shear-Scaling Affine Transformation of Finch Beaks | *Geospiza fortis* | `2021-alberti-avian-beak-darwin-scaling.py` | 56.6 | **93.2** | +36.6 |
| `#071` | `diatom-frustule-phase-sep` | LCPA Phase-Separation Diatom Valve Pores | *Coscinodiscus wailesii* | `2002-sumper-diatom-frustule-phase-sep.py` | 51.6 | **89.0** | +37.4 |
| `#072` | `cahn-hilliard-biosilica` | Coupled Cahn-Hilliard Silica Solidification | *Thalassiosira pseudonana* | `2013-chirat-mollusc-shell-marginal-buckling.py` | 53.1 | **91.3** | +38.2 |
| `#073` | `foraminifera-chamber-spiral` | Isoallometric Chamber Accretion Spiral | *Globigerinoides ruber* | `2007-cartwright-nacre-spiral-dislocation-terraces.py` | 52.8 | **91.4** | +38.6 |
| `#074` | `foraminifera-apertural-growth` | Minimized Free-Surface Reference Point Foraminifera | *Ammonia beccarii* | `2013-thomas-coral-calice-septa-insertion.py` | 52.4 | **91.6** | +39.2 |
| `#075` | `echinoderm-pentaradial-symmetry` | Bilateral-to-Pentaradial Hydrocoel Ring | *Strongylocentrotus purpuratus* | `2005-meinhardt-echinoderm-pentaradial-symmetry.py` | 52.7 | **91.8** | +39.1 |
| `#076` | `sponge-spicule-hexactin` | Silicatein Axial Filament Spicule Branching | *Euplectella aspergillum* | `2012-bavestrello-sponge-spicule-hexactin.py` | 52.6 | **91.6** | +39.0 |
| `#077` | `mollusc-aperture-spine-buckling` | Elastoplastic Mantle-Edge Spine Morphogenesis | *Murex pecten* | `2015-moulton-mollusc-aperture-spine-buckling.py` | 52.7 | **91.4** | +38.7 |
| `#078` | `nacre_screw_dislocation_spirals` | BCF Screw-Dislocation Aragonite Nacre Terraces | *Haliotis rufescens* | `2019-cartwright-nacre-screw-dislocation-spirals.py` | 48.9 | **91.8** | +42.9 |
| `#079` | `crinoid-stereom-gyroid` | Triply Periodic Minimal Surface (TPMS) Stereom | *Metacrinus rotundus* | `2020-gorzelak-crinoid-stereom-gyroid.py` | 48.5 | **91.7** | +43.2 |
| `#080` | `emiliania_calcite_rosette` | Radial V/R Crystal Unit Coccolith Shield | *Emiliania huxleyi* | `2022-coccolith-emiliania-calcite-rosette.py` | 52.2 | **91.8** | +39.6 |
| `#081` | `planarian-hk-atpase-anterior` | Bioelectric V_mem Head-Tail Polarity Switch | *Schmidtea mediterranea* | `2011-beane-planarian-hk-atpase-anterior.py` | 59.2 | **91.3** | +32.1 |
| `#082` | `bioelectric-brain-teratogenesis` | HCN2 Bioelectric Brain Ventricle Scaling | *Xenopus laevis* | `2017-pietak-bioelectric-morphogen-coupling.py` | 58.6 | **91.7** | +33.1 |
| `#083` | `planarian-allometric-rescaling` | Wnt-ADMP Scale-Invariant Body Proportioning | *Schmidtea mediterranea* | `2020-kriegman-xenobot-evolutionary-biobots.py` | 59.5 | **91.5** | +32.0 |
| `#084` | `bioelectric-gene-network-coupled` | Electrodiffusion-Gene Regulatory Network Coupling | *Xenopus laevis* | `2015-pezzulo-bioelectric-target-morphology-memory.py` | 59.6 | **91.6** | +32.0 |
| `#085` | `xenopus-craniofacial-remap` | Picasso Tadpole Craniofacial Attractor Remodeling | *Xenopus laevis* | `2019-bischof-xenopus-craniofacial-remap.py` | 56.8 | **92.0** | +35.2 |
| `#086` | `retinotectal-chemoaffinity` | Sperry EphA-EphrinA Retinotectal Topographic Map | *Danio rerio* | `1976-willshaw-retinotectal-chemoaffinity.py` | 56.3 | **92.4** | +36.1 |
| `#087` | `ocular-dominance-stripes` | Correlation-Based Hebbian Ocular Dominance Columns | *Macaca mulatta* | `1989-miller-ocular-dominance-stripes.py` | 56.5 | **92.8** | +36.3 |
| `#088` | `grid-cell-hexagonal-attractor` | Continuous Attractor Entorhinal Grid-Cell Lattice | *Rattus norvegicus* | `2008-burak-grid-cell-hexagonal-attractor.py` | 56.9 | **92.6** | +35.7 |
| `#089` | `corticothalamic_seizure_spirals` | Wilson-Cowan Cortical Spiral Wave Dynamics | *Homo sapiens* | `2013-ermentrout-corticothalamic-seizure-spirals.py` | 56.9 | **92.9** | +36.0 |
| `#090` | `cortical_sulcal_axon_tension` | Van Essen Axonal Tension Morphogenesis | *Homo sapiens* | `2020-karbowski-cortical-sulcal-axon-tension.py` | 55.8 | **93.4** | +37.6 |
| `#091` | `neural-crest-chase-and-run` | Contact Inhibition of Locomotion + Co-Attraction | *Xenopus laevis* | `2012-carmona-neural-crest-chase-and-run.py` | 57.6 | **91.1** | +33.5 |
| `#092` | `neural-crest-leader-follower` | Induced-Leader VEGF Trailblazing Migration | *Gallus gallus* | `1981-odell-apical-constriction-invagination.py` | 60.9 | **91.7** | +30.8 |
| `#093` | `diffusion-limited-coral-growth` | Advection-Diffusion Accretion of Branching Corals | *Madracis mirabilis* | `2003-zajac-convergent-extension-intercalation.py` | 61.2 | **91.9** | +30.7 |
| `#094` | `bacillus-biofilm-wrinkling` | Osmotic Matrix Swelling & Biofilm Wrinkle Radial Spokes | *Bacillus subtilis* | `2013-asally-bacillus-biofilm-wrinkling.py` | 56.2 | **93.6** | +37.4 |
| `#095` | `vibrio-biofilm-nematic-fountain` | 3D Rod-Cell Nematic Buckling in Growing Biofilms | *Vibrio cholerae* | `2019-yan-vibrio-biofilm-nematic-fountain.py` | 56.4 | **93.8** | +37.4 |
| `#096` | `social_wasp_comb_hexagons` |  Stigmergic Lattice Wasp Nest Architecture | *Polistes dominula* | `1998-bonabeau-social-wasp-comb-hexagons.py` | 56.2 | **93.9** | +37.7 |
| `#097` | `termite_mound_convective_chimneys` | Thermosiphon Convective Pheromone Mound Morphogenesis | *Macrotermes michaelseni* | `2015-ocko-termite-mound-convective-chimneys.py` | 56.8 | **94.3** | +37.5 |
| `#098` | `butterfly_eyespot_focus_rings` | Source-Threshold Concentric Eyespot Rings | *Bicyclus anynana* | `2001-nijhout-butterfly-eyespot-focus-rings.py` | 47.7 | **91.9** | +44.2 |
| `#099` | `papilio_wing_stripe_mimicry` | Anisotropic Wing-Cell Reaction-Diffusion Mimicry | *Papilio dardanus* | `2016-sekimura-papilio-wing-stripe-mimicry.py` | 56.4 | **94.3** | +37.9 |
| `#100` | `hydractinia_stolon_vascular_net` | Gastrovascular Pulsating Stolon Network | *Hydractinia symbiolongicarpus* | `2020-badin-hydractinia-stolon-vascular-net.py` | 53.0 | **93.8** | +40.8 |
