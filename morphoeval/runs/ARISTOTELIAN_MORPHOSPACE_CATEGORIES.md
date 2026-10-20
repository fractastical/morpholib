# Aristotelian Categories & Taxonomic Coverage of Infinite Morphospace

In *Categories* (*Katēgoriai*), *Historia Animalium*, and *De Partibus Animalium* (Book I), Aristotle established that biological form cannot be understood by a single dichotomous tree alone: every organismal phenomenon must be cross-indexed by its **Substance / Taxonomic Genus (*Ousia / Genos*)** and its **Four Causes & Categories**:

1. ***Ousia* (Substance — `Summum Genus → Megista Genē → Subaltern Class → Genus → Species`)**: Which living taxon embodies the form (**55 biological genera** across **116 models** and **111 dataset directories**).
2. ***Poion* (Quality — Morphological Topology)**: What qualitative geometric form emerges (stripes/spots, logarithmic shells, bifurcating trees, buckled sheets/gyri, polygonal tilings, parastichy spirals, target polarity).
3. ***Poson* (Quantity — Dimensionless Scaling Law)**: What mathematical invariant governs the form (**333 machine-checked Lean 4 equations** and **954 measurements**, e.g. Zoelly $(h/R_0)^2$, Murray's $r_0^3 = r_1^3 + r_2^3$, Golden Angle $137.508^\circ$, Vertex jamming $p_0^* = 3.81$).
4. ***Poiein & Paschein* (Action & Affection — Efficient Biophysical Cause)**: Which of the 10 dynamical mechanism classes generates the morphology.

![Aristotelian Categories & Porphyrian Taxonomic Atlas](../../docs/media/morphospace_aristotelian_atlas.png)

![Porphyrian Specimen Gallery across Aristotle's 10 Subaltern Classes](../../docs/media/morphospace_porphyrian_specimens.png)

---

## 1. Summary by Aristotle's *Megista Genē* (Greatest Genera)

| Aristotelian *Megiston Genos* | Subaltern Class (*Genos*) | Models | Unique Genera | Representative Genera (Dataset Count) |
| :--- | :--- | :---: | :---: | :--- |
| **I. Enaima (Vertebrata)** | Mammalia (Zootoka) | **29** | **7** | *Mus* (14), *Homo* (10), *Panthera* (1), *Bos* (1), *Canis* (1), *Macaca* (1), +1 more |
| **I. Enaima (Vertebrata)** | Aves (Ornithes) | **14** | **3** | *Gallus* (12), *Coturnix* (1), *Geospiza* (1) |
| **I. Enaima (Vertebrata)** | Amphibia & Reptilia (Ootoka) | **9** | **3** | *Xenopus* (6), *Ambystoma* (2), *Timon* (1) |
| **I. Enaima (Vertebrata)** | Pisces (Ichthyes) | **9** | **3** | *Danio* (6), *Pomacanthus* (2), *Scarus* (1) |
| **II. Anaima (Invertebrata)** | Entoma (Arthropoda) | **11** | **7** | *Drosophila* (5), *Polistes* (1), *Macrotermes* (1), *Bicyclus* (1), *Papilio* (1), *Arigomphus* (1), +1 more |
| **II. Anaima (Invertebrata)** | Ostrakoderma (Mollusca & Echinodermata) | **6** | **6** | *Strongylocentrotus* (1), *Murex* (1), *Haliotis* (1), *Metacrinus* (1), *Nautilus* (1), *Conus* (1) |
| **II. Anaima (Invertebrata)** | Zoophyta & Vermes (Cnidaria/Platyhelminthes) | **11** | **7** | *Schmidtea* (4), *Hydra* (2), *Euplectella* (1), *Madracis* (1), *Hydractinia* (1), *Pocillopora* (1), +1 more |
| **III. Phyta (Embryophyta)** | Angiospermae & Gymnospermae | **16** | **10** | *Arabidopsis* (6), *Helianthus* (2), *Pisum* (1), *Cardamine* (1), *Brassica* (1), *Cornus* (1), +4 more |
| **IV. Protista & Monera** | Slime Molds & Bacteria | **6** | **6** | *Dictyostelium* (1), *Myxococcus* (1), *Physarum* (1), *Bacillus* (1), *Vibrio* (1), *Cyanobacteria* (1) |
| **IV. Protista & Monera** | Mineralized Protists (Silica/Calcite) | **5** | **5** | *Coscinodiscus* (1), *Thalassiosira* (1), *Globigerinoides* (1), *Ammonia* (1), *Emiliania* (1) |

---

## 2. Complete Index of All 55 Biological Genera (*Infima Genera*)

| Genus (*Ousia*) | Common Name | Aristotelian Division (*Megiston Genos*) | Subaltern Class | Models | Formal Topologies (*Poion*) | Simulation / Dataset Slugs |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| ***Mus*** | Mouse | I. Enaima (Vertebrata) | Mammalia (Zootoka) | **14** | Bifurcating Vascular/Lung Trees; Periodic Somites & Segments | `wnt-dkk-hair-follicle`, `palate-ruguga-turing`, `hox-turing-polydactyly`, `bmp-sox9-wnt-digits` (+10 more) |
| ***Gallus*** | Chick | I. Enaima (Vertebrata) | Aves (Ornithes) | **12** | Bifurcating Vascular/Lung Trees; Biofilm Wrinkles & Hex Combs | `french-flag-morphogen`, `limb-chondrogenesis-turing`, `tgfb-fibronectin-limb`, `joint-interzone-clefting` (+8 more) |
| ***Homo*** | Human | I. Enaima (Vertebrata) | Mammalia (Zootoka) | **10** | Bifurcating Vascular/Lung Trees; Gyri, Villi, Loops & Snap-Shells | `allometric-cranial-sutures`, `human-somite-organoid-clock`, `vertex-glass-jamming`, `vascular-cube-law-branching` (+6 more) |
| ***Arabidopsis*** | Thale Cress | III. Phyta (Embryophyta) | Angiospermae & Gymnospermae | **6** | Spirals, Veins & Puzzle Cells | `auxin-pin1-meristem`, `meristem-phyllotaxis-pin`, `carrier-driven-canalization`, `microtubule-stress-meristem` (+2 more) |
| ***Danio*** | Zebrafish | I. Enaima (Vertebrata) | Pisces (Ichthyes) | **6** | Periodic Somites & Segments; Stripes, Spots & Ocelli | `zebrafish-three-cell-stripes`, `her1-her7-delayed-oscillator`, `delta-notch-phase-synchrony`, `doppler-somite-shortening` (+2 more) |
| ***Xenopus*** | Clawed Frog | I. Enaima (Vertebrata) | Amphibia & Reptilia (Ootoka) | **6** | Biofilm Wrinkles & Hex Combs; Embryonic Ca2+ Spark Waves | `clock-and-wavefront`, `bioelectric-brain-teratogenesis`, `bioelectric-gene-network-coupled`, `xenopus-craniofacial-remap` (+2 more) |
| ***Drosophila*** | Fruit Fly | II. Anaima (Invertebrata) | Entoma (Arthropoda) | **5** | Bifurcating Vascular/Lung Trees; Polygonal Packing & Furrows | `drosophila-wing-vertex`, `drosophila-ventral-furrow`, `global-morphogenetic-flow`, `wing-disc-dome-folding` (+1 more) |
| ***Schmidtea*** | Planarian Flatworm | II. Anaima (Invertebrata) | Zoophyta & Vermes (Cnidaria/Platyhelminthes) | **4** | Bistable V_mem Polarity; Regenerative Body-Plan Graph | `planarian-hk-atpase-anterior`, `planarian-allometric-rescaling`, `planform-morphology-animation`, `cervera-levin-mafe` |
| ***Ambystoma*** | Axolotl | I. Enaima (Vertebrata) | Amphibia & Reptilia (Ootoka) | **2** | Segmented Appendage Graph; Skeletal Rays & Joints | `mechanochemical-chondrogenesis`, `lobo-limbform-morphology` |
| ***Helianthus*** | Sunflower | III. Phyta (Embryophyta) | Angiospermae & Gymnospermae | **2** | Fibonacci Parastichy Spiral; Spirals, Veins & Puzzle Cells | `magnetic-drop-phyllotaxis`, `vogel-phyllotaxis` |
| ***Hydra*** | Freshwater Polyp | II. Anaima (Invertebrata) | Zoophyta & Vermes (Cnidaria/Platyhelminthes) | **2** | Polygonal Packing & Furrows; Stripes, Spots & Ocelli | `hydra-activator-inhibitor`, `glazier-graner-potts` |
| ***Pomacanthus*** | Angelfish | I. Enaima (Vertebrata) | Pisces (Ichthyes) | **2** | Stripes, Spots & Ocelli; Stripes, Spots & Y-Junctions | `pomacanthus-stripe-insertion`, `turing-gray-scott` |
| ***Ammonia*** | Benthic Foraminifer | IV. Protista & Monera | Mineralized Protists (Silica/Calcite) | **1** | Coiled Shells, Spicules & Gyroids | `foraminifera-apertural-growth` |
| ***Arigomphus*** | Clubtail Dragonfly | II. Anaima (Invertebrata) | Entoma (Arthropoda) | **1** | Polygonal Wing Venation | `hoffmann-wing-venation` |
| ***Bacillus*** | Subtilis Biofilm | IV. Protista & Monera | Slime Molds & Bacteria | **1** | Biofilm Wrinkles & Hex Combs | `bacillus-biofilm-wrinkling` |
| ***Bauhinia*** | Orchid Tree Pod | III. Phyta (Embryophyta) | Angiospermae & Gymnospermae | **1** | Gyri, Villi, Loops & Snap-Shells | `chiral-seed-pod-opening` |
| ***Bicyclus*** | Squinting Bush Brown | II. Anaima (Invertebrata) | Entoma (Arthropoda) | **1** | Biofilm Wrinkles & Hex Combs | `butterfly-eyespot-focus-rings` |
| ***Bos*** | Bovine | I. Enaima (Vertebrata) | Mammalia (Zootoka) | **1** | Gyri, Villi, Loops & Snap-Shells | `airway-mucosal-creasing` |
| ***Brassica*** | Romanesco Cauliflower | III. Phyta (Embryophyta) | Angiospermae & Gymnospermae | **1** | Spirals, Veins & Puzzle Cells | `roman-esco-fractal-inflorescence` |
| ***Caenorhabditis*** | Nematode Worm | II. Anaima (Invertebrata) | Zoophyta & Vermes (Cnidaria/Platyhelminthes) | **1** | 4D Embryonic Lineage Tree | `santella-wormguides-celegans` |
| ***Canis*** | Canine (MDCK) | I. Enaima (Vertebrata) | Mammalia (Zootoka) | **1** | Gyri, Villi, Loops & Snap-Shells | `epithelioid-cyst-buckling` |
| ***Cardamine*** | Hairy Bittercress | III. Phyta (Embryophyta) | Angiospermae & Gymnospermae | **1** | Spirals, Veins & Puzzle Cells | `allometric-compound-leaf` |
| ***Conus*** | Textile Cone Snail | II. Anaima (Invertebrata) | Ostrakoderma (Mollusca & Echinodermata) | **1** | Cellular Automaton Triangles | `wolfram-conus-textile-rule30` |
| ***Cornus*** | Pagoda Dogwood | III. Phyta (Embryophyta) | Angiospermae & Gymnospermae | **1** | Bifurcating Vascular/Lung Trees | `dichotomous-tree-branching` |
| ***Coscinodiscus*** | Centric Diatom | IV. Protista & Monera | Mineralized Protists (Silica/Calcite) | **1** | Coiled Shells, Spicules & Gyroids | `diatom-frustule-phase-sep` |
| ***Coturnix*** | Quail | I. Enaima (Vertebrata) | Aves (Ornithes) | **1** | Stripes, Spots & Ocelli | `avian-plumage-somite-stripes` |
| ***Cyanobacteria*** | Stromatolite Mat | IV. Protista & Monera | Slime Molds & Bacteria | **1** | Conical Microbial Laminae | `petroff-stromatolite` |
| ***Dictyostelium*** | Social Amoeba | IV. Protista & Monera | Slime Molds & Bacteria | **1** | Stripes, Spots & Ocelli | `limit-cycle-turing` |
| ***Dionaea*** | Venus Flytrap | III. Phyta (Embryophyta) | Angiospermae & Gymnospermae | **1** | Gyri, Villi, Loops & Snap-Shells | `venus-flytrap-snap-buckling` |
| ***Emiliania*** | Coccolithophore | IV. Protista & Monera | Mineralized Protists (Silica/Calcite) | **1** | Coiled Shells, Spicules & Gyroids | `emiliania-calcite-rosette` |
| ***Euplectella*** | Venus Flower Basket | II. Anaima (Invertebrata) | Zoophyta & Vermes (Cnidaria/Platyhelminthes) | **1** | Coiled Shells, Spicules & Gyroids | `sponge-spicule-hexactin` |
| ***Geospiza*** | Darwin's Finch | I. Enaima (Vertebrata) | Aves (Ornithes) | **1** | Gyri, Villi, Loops & Snap-Shells | `avian-beak-darwin-scaling` |
| ***Ginkgo*** | Maidenhair Tree | III. Phyta (Embryophyta) | Angiospermae & Gymnospermae | **1** | Open & Reticulate Venation | `runions-leaf-venation` |
| ***Globigerinoides*** | Planktonic Foraminifer | IV. Protista & Monera | Mineralized Protists (Silica/Calcite) | **1** | Coiled Shells, Spicules & Gyroids | `foraminifera-chamber-spiral` |
| ***Haliotis*** | Red Abalone | II. Anaima (Invertebrata) | Ostrakoderma (Mollusca & Echinodermata) | **1** | Coiled Shells, Spicules & Gyroids | `nacre-screw-dislocation-spirals` |
| ***Hydractinia*** | Colonial Hydroid | II. Anaima (Invertebrata) | Zoophyta & Vermes (Cnidaria/Platyhelminthes) | **1** | Biofilm Wrinkles & Hex Combs | `hydractinia-stolon-vascular-net` |
| ***Lactuca*** | Ruffled Leaf Lettuce | III. Phyta (Embryophyta) | Angiospermae & Gymnospermae | **1** | Gyri, Villi, Loops & Snap-Shells | `ripping-leaf-edge-waves` |
| ***Macaca*** | Macaque | I. Enaima (Vertebrata) | Mammalia (Zootoka) | **1** | Target Polarity & Neural Maps | `ocular-dominance-stripes` |
| ***Macrotermes*** | Mound Termite | II. Anaima (Invertebrata) | Entoma (Arthropoda) | **1** | Biofilm Wrinkles & Hex Combs | `termite-mound-convective-chimneys` |
| ***Madracis*** | Branching Coral | II. Anaima (Invertebrata) | Zoophyta & Vermes (Cnidaria/Platyhelminthes) | **1** | Biofilm Wrinkles & Hex Combs | `diffusion-limited-coral-growth` |
| ***Metacrinus*** | Stalked Sea Lily | II. Anaima (Invertebrata) | Ostrakoderma (Mollusca & Echinodermata) | **1** | Coiled Shells, Spicules & Gyroids | `crinoid-stereom-gyroid` |
| ***Murex*** | Venus Comb Murex | II. Anaima (Invertebrata) | Ostrakoderma (Mollusca & Echinodermata) | **1** | Coiled Shells, Spicules & Gyroids | `mollusc-aperture-spine-buckling` |
| ***Myxococcus*** | Predatory Myxobacterium | IV. Protista & Monera | Slime Molds & Bacteria | **1** | Stripes, Spots & Ocelli | `chemotactic-stripe-splitting` |
| ***Nautilus*** | Chambered Nautilus | II. Anaima (Invertebrata) | Ostrakoderma (Mollusca & Echinodermata) | **1** | Logarithmic Coiled Shell | `raup-shell-coiling` |
| ***Odonata*** | Dragonflies & Damselflies | II. Anaima (Invertebrata) | Entoma (Arthropoda) | **1** | Voronoi Wing Cell Polygons | `eshghi-odonata-wing-cells` |
| ***Panthera*** | Leopard | I. Enaima (Vertebrata) | Mammalia (Zootoka) | **1** | Stripes, Spots & Ocelli | `mammalian-coat-markings` |
| ***Papilio*** | Swallowtail Butterfly | II. Anaima (Invertebrata) | Entoma (Arthropoda) | **1** | Biofilm Wrinkles & Hex Combs | `papilio-wing-stripe-mimicry` |
| ***Physarum*** | Plasmodial Slime Mold | IV. Protista & Monera | Slime Molds & Bacteria | **1** | Bifurcating Vascular/Lung Trees | `physarum-transport-network` |
| ***Pisum*** | Garden Pea | III. Phyta (Embryophyta) | Angiospermae & Gymnospermae | **1** | Spirals, Veins & Puzzle Cells | `canalization-auxin-flux` |
| ***Pocillopora*** | Cauliflower Coral | II. Anaima (Invertebrata) | Zoophyta & Vermes (Cnidaria/Platyhelminthes) | **1** | Flow-Symmetrized Branches | `chindapol-coral-flow` |
| ***Polistes*** | Paper Wasp | II. Anaima (Invertebrata) | Entoma (Arthropoda) | **1** | Biofilm Wrinkles & Hex Combs | `social-wasp-comb-hexagons` |
| ***Rattus*** | Rat | I. Enaima (Vertebrata) | Mammalia (Zootoka) | **1** | Target Polarity & Neural Maps | `grid-cell-hexagonal-attractor` |
| ***Scarus*** | Parrotfish | I. Enaima (Vertebrata) | Pisces (Ichthyes) | **1** | Affine / Biharmonic Grid | `thompson-scarus-pomacanthus` |
| ***Strongylocentrotus*** | Purple Sea Urchin | II. Anaima (Invertebrata) | Ostrakoderma (Mollusca & Echinodermata) | **1** | Coiled Shells, Spicules & Gyroids | `echinoderm-pentaradial-symmetry` |
| ***Thalassiosira*** | Marine Diatom | IV. Protista & Monera | Mineralized Protists (Silica/Calcite) | **1** | Coiled Shells, Spicules & Gyroids | `cahn-hilliard-biosilica` |
| ***Timon*** | Ocellated Lizard | I. Enaima (Vertebrata) | Amphibia & Reptilia (Ootoka) | **1** | Stripes, Spots & Ocelli | `ocellated-lizard-ca` |
| ***Vibrio*** | Cholera Biofilm | IV. Protista & Monera | Slime Molds & Bacteria | **1** | Biofilm Wrinkles & Hex Combs | `vibrio-biofilm-nematic-fountain` |

---

## 3. Sparse / Unexplored Regions Revealed by the Aristotelian Matrix

Cross-tabulating *Ousia* (Taxonomic Class) against *Poiein* (Biophysical Mechanism) immediately reveals four high-value empirical gaps:

1. **Bioelectric Signaling ($V_{\text{mem}}$) in *Phyta* (Land Plants)**: We have 17 plant models in Auxin/Phyllotaxis, Venation, and Mechanical Snap-Buckling (*Dionaea*, *Bauhinia*, *Lactuca*), but **0** models of plant bioelectric action potentials / glutamate-receptor calcium waves (`GLR3.3`/`GLR3.6` wound signaling).
2. **Clock-and-Wavefront Oscillators in *Anaima* (*Entoma* / *Vermes*)**: All 10 segmentation clock loops currently model vertebrate (*Enaima*) somitogenesis (*Danio*, *Mus*, *Gallus*, *Xenopus*, *Homo*); short-germ arthropod (*Tribolium*) and annelid (*Platynereis*) segmentation clocks are prime next targets.
3. ***Mycota* (Fungi)**: Filamentous fungi (hyphal tip growth, *Neurospora* circadian banding, mycelial transport networks, and basidiocarp gill buckling) represent an entire kingdom currently unpopulated.
4. ***Malakia* (Cephalopods — Aristotle's Soft-Bodied *Anaima*)**: Neural-muscular chromatophore pattern waves in *Octopus* and *Sepia* bridge *Anaima* and *Neural Patterning*.
