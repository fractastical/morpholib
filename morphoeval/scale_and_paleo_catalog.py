#!/usr/bin/env python3
"""
morphoeval/scale_and_paleo_catalog.py — Multi-Scale Phenomenon Sizes, 100-Run Zebrafish
(*Danio rerio* / Zebrahub) Deep-Dive Matrix, and 90 Novel / Fossil Chronospecies Taxa
for Slots #1101–#2100 (and scale annotation of Slots #0001–#1100 + 16 Foundational Models).
"""

from __future__ import annotations

import math
from typing import Any

# ============================================================================
# 1. Eight Universal Logarithmic Spatial Scale Tiers (Smallest Bit -> Largest Bit)
# ============================================================================
SCALE_TIERS: list[dict[str, Any]] = [
    {
        "tier_id": "1_molecular_nm",
        "tier_num": 1,
        "label": "1. Molecular & Ion-Channel Complexes (1–50 nm)",
        "short": "1–50 nm (Molecular)",
        "min_m": 1e-9,
        "max_m": 5e-8,
    },
    {
        "tier_id": "2_macromolecular_100nm",
        "tier_num": 2,
        "label": "2. Subcellular Nanostructures & Junctions (50 nm–1 µm)",
        "short": "50 nm–1 µm (Nanostructure)",
        "min_m": 5e-8,
        "max_m": 1e-6,
    },
    {
        "tier_id": "3_subcellular_um",
        "tier_num": 3,
        "label": "3. Organelles, Cilia & Cytoskeletal Arrays (1–10 µm)",
        "short": "1–10 µm (Subcellular)",
        "min_m": 1e-6,
        "max_m": 1e-5,
    },
    {
        "tier_id": "4_cellular_10um",
        "tier_num": 4,
        "label": "4. Single Cells, Melanosomes & Vertex Polygons (10–100 µm)",
        "short": "10–100 µm (Single Cell)",
        "min_m": 1e-5,
        "max_m": 1e-4,
    },
    {
        "tier_id": "5_primordium_100um",
        "tier_num": 5,
        "label": "5. Tissue Primordia, Somites, Neuromasts & Crypts (100 µm–1 mm)",
        "short": "100 µm–1 mm (Primordium)",
        "min_m": 1e-4,
        "max_m": 1e-3,
    },
    {
        "tier_id": "6_organ_mm",
        "tier_num": 6,
        "label": "6. Developing Organs, Embryo Axes & Suture Lobes (1–10 mm)",
        "short": "1–10 mm (Organ / Embryo)",
        "min_m": 1e-3,
        "max_m": 1e-2,
    },
    {
        "tier_id": "7_organism_cm",
        "tier_num": 7,
        "label": "7. Whole Organisms, Adult Fins, Shells & Skulls (1–25 cm)",
        "short": "1–25 cm (Whole Organism)",
        "min_m": 1e-2,
        "max_m": 0.25,
    },
    {
        "tier_id": "8_colonial_m",
        "tier_num": 8,
        "label": "8. Macro-Skeletal, Shoaling & Colonial Architecture (0.25–15 m)",
        "short": "0.25–15 m (Macro / Shoal)",
        "min_m": 0.25,
        "max_m": 15.0,
    },
]


def classify_scale_tier(size_m: float) -> tuple[str, str, int]:
    """Return (tier_id, short_label, tier_num) for a physical size in meters."""
    for t in SCALE_TIERS:
        if size_m < t["max_m"]:
            return t["tier_id"], t["short"], int(t["tier_num"])
    last = SCALE_TIERS[-1]
    return last["tier_id"], last["short"], int(last["tier_num"])


def format_size_human(size_m: float) -> str:
    """Format physical size in meters into human-readable nm / µm / mm / cm / m."""
    if size_m < 1e-6:
        val = size_m * 1e9
        return f"{val:.1f} nm" if val < 10 else f"{val:.0f} nm"
    if size_m < 1e-3:
        val = size_m * 1e6
        return f"{val:.2f} µm" if val < 1 else (f"{val:.1f} µm" if val < 10 else f"{val:.0f} µm")
    if size_m < 1e-2:
        val = size_m * 1e3
        return f"{val:.2f} mm" if val < 1 else f"{val:.1f} mm"
    if size_m < 1.0:
        val = size_m * 1e2
        return f"{val:.1f} cm"
    return f"{size_m:.2f} m"


# ============================================================================
# 2. Zebrafish (*Danio rerio*) 10 Spatial Scales × 10 Biophysical Cohorts = 100 Runs
#    (Slots #1101–#1200) Linked to Zebrahub, ZFIN, Daniocell & MapZebrain
# ============================================================================
ZEBRAHUB_EXTERNAL_DATASETS: list[dict[str, str]] = [
    {
        "name": "Zebrahub (CZ Biohub SF — 4D Light-Sheet Lineage + scRNA-seq Timecourse Atlas)",
        "url": "https://zebrahub.ds.czbiohub.org/",
        "doi": "10.1016/j.cell.2024.09.047",
    },
    {
        "name": "ZFIN (Zebrafish Information Network — Anatomical Ontology & Mutant Phenotypes)",
        "url": "https://zfin.org/",
        "doi": "10.1093/genetics/iyad068",
    },
    {
        "name": "Daniocell (NICHD / Farrell Lab — Whole-Embryo Single-Cell Transcriptome Atlas)",
        "url": "https://daniocell.nichd.nih.gov/",
        "doi": "10.1016/j.devcel.2023.09.008",
    },
    {
        "name": "MapZebrain (Max Planck Neurobiology — Whole-Brain Cellular & Circuit Atlas)",
        "url": "https://mapzebrain.org/",
        "doi": "10.1038/s41592-019-0471-2",
    },
]

# 10 spatial scale rungs for Danio rerio (from 4.2 nm molecular complexes to 42 cm collective shoals)
DANIO_SCALE_RUNGS: list[dict[str, Any]] = [
    {
        "rung_idx": 0,
        "slug_tag": "nm4-connexin-her-complex",
        "base_size_m": 4.2e-9,
        "stage": "10–14 hpf (Bud to 10-Somite Stage, ZFA:0001094)",
        "structure": "Connexin-41.8 hemichannel pore & Her1/Her7 bHLH dimer complex",
        "zebrahub_cluster": "Zebrahub PSM_presomitic_mesoderm & neural_plate_progenitors (10–14 hpf)",
        "author": "lange",
        "year": 2024,
        "doi": "10.1016/j.cell.2024.09.047",
    },
    {
        "rung_idx": 1,
        "slug_tag": "nm28-delta-notch-synapse",
        "base_size_m": 2.8e-8,
        "stage": "12–16 hpf (6–14 Somite Stage, ZFA:0000115)",
        "structure": "DeltaC–Notch1a trans-endocytosis & EphA4–EphrinB2 boundary complex",
        "zebrahub_cluster": "Zebrahub somite_boundary_epithelium & notochord_sheath (12–16 hpf)",
        "author": "riedel",
        "year": 2013,
        "doi": "10.1038/nature11805",
    },
    {
        "rung_idx": 2,
        "slug_tag": "nm180-iridophore-bragg-crystal",
        "base_size_m": 1.8e-7,
        "stage": "72 hpf–Adult (Iridophore Reflector Platelet, ZFA:0009128)",
        "structure": "Guanine Bragg-stack nanocrystal platelet & tight-junction ZO-1 belt",
        "zebrahub_cluster": "Zebrahub neural_crest_iridophore_lineage & EVL_periderm",
        "author": "gur",
        "year": 2015,
        "doi": "10.1038/ncomms7393",
    },
    {
        "rung_idx": 3,
        "slug_tag": "um2-kupffer-motile-cilium",
        "base_size_m": 2.4e-6,
        "stage": "11–14 hpf (Kupffer's Vesicle L-R Organizer, ZFA:0000049)",
        "structure": "Motile 9+2 nodal cilium & sarcomeric Z-disc myofibril bundle",
        "zebrahub_cluster": "Zebrahub kupffers_vesicle_ciliated_epithelium & fast_muscle_myotome",
        "author": "supatto",
        "year": 2008,
        "doi": "10.1073/pnas.0803596105",
    },
    {
        "rung_idx": 4,
        "slug_tag": "um18-melanophore-xanthophore-cell",
        "base_size_m": 1.8e-5,
        "stage": "14–21 dpf (Metamorphic Pigment & Tailbud Progenitor, ZFA:0009096)",
        "structure": "Single melanophore–xanthophore pseudopodial contact & NMP cell body",
        "zebrahub_cluster": "Zebrahub tailbud_neuromesodermal_progenitors & pigment_chromatophores",
        "author": "yamanaka",
        "year": 2014,
        "doi": "10.1126/science.1247664",
    },
    {
        "rung_idx": 5,
        "slug_tag": "um55-somite-rosette-neuromast",
        "base_size_m": 5.5e-5,
        "stage": "16–24 hpf (Somite Epithelial Somitomere & Neuromast Rosette, ZFA:0000155)",
        "structure": "Single nascent somite block (55 µm AP) & deposited lateral-line neuromast",
        "zebrahub_cluster": "Zebrahub lateral_line_neuromast_rosette & segmented_paraxial_somite",
        "author": "soroldoni",
        "year": 2014,
        "doi": "10.1126/science.1253089",
    },
    {
        "rung_idx": 6,
        "slug_tag": "um220-pllp-heart-primordium",
        "base_size_m": 2.2e-4,
        "stage": "24–36 hpf (Pharyngula PLLP & Linear Heart Tube, ZFA:0001279)",
        "structure": "Migrating posterior lateral line primordium (PLLP) & looping heart tube",
        "zebrahub_cluster": "Zebrahub migrating_PLLP_stream & myocardial_endocardial_tube (24 hpf)",
        "author": "gilmour",
        "year": 2017,
        "doi": "10.1038/nature21576",
    },
    {
        "rung_idx": 7,
        "slug_tag": "mm1-2-embryo-trunk-vascular-axis",
        "base_size_m": 1.2e-3,
        "stage": "24–48 hpf (Whole Embryonic AP Trunk & ISV Arcade, ZFA:0001093)",
        "structure": "Whole embryonic AP trunk axis (30 somites), dorsal aorta & ISV plexus",
        "zebrahub_cluster": "Zebrahub whole_embryo_4D_lightsheet_trajectory (10–24 hpf)",
        "author": "mongera",
        "year": 2018,
        "doi": "10.1038/s41586-018-0479-2",
    },
    {
        "rung_idx": 8,
        "slug_tag": "mm32-adult-caudal-fin-flank-stripes",
        "base_size_m": 3.2e-2,
        "stage": "60–90 dpf (Adult Body Flank & Regenerating Caudal Fin, ZFA:0000016)",
        "structure": "Adult 5-stripe flank epidermis (32 mm SL) & bifurcating lepidotrichia rays",
        "zebrahub_cluster": "Daniocell / ZFIN adult_dermal_chromatophore & osteoblast_fin_ray",
        "author": "volkening",
        "year": 2015,
        "doi": "10.1038/ncomms9312",
    },
    {
        "rung_idx": 9,
        "slug_tag": "cm42-collective-shoal-vortex-wake",
        "base_size_m": 0.42,
        "stage": "Adult Collective Shoal (15–30 Fish Polarized School, 0.42 m arena)",
        "structure": "Reverse Kármán vortex dipole wake & visual-mechanosensory shoal lattice",
        "zebrahub_cluster": "MapZebrain optic_tectum_loom_circuit & lateral_line_rheotaxis",
        "author": "couzin",
        "year": 2013,
        "doi": "10.1126/science.1229965",
    },
]


# ============================================================================
# 3. 90 Novel Extant Species & Fossil-Record Chronospecies (Slots #1201–#2100)
#    Includes evolutionary variations of the same lineage across geological time!
# ============================================================================
# Tuple layout:
# (genus, species, common_name, megiston_genos, subaltern_genos,
#  base_author, base_year, doi_prefix, base_size_m, age_ma, geol_period, lineage_group)
NINETY_NOVEL_AND_FOSSIL_TAXA: list[tuple[str, str, str, str, str, str, int, str, float, float, str, str]] = [
    # --- Lineage 1: Sister Danionin & Fossil Teleost Trunk/Pigment Series (6 taxa) ---
    ("Danio", "margaritatus", "Celestial Pearl Danio (Galaxy Rasbora)", "I. Enaima (Vertebrata)", "Pisces (Ichthyes)", "parichy", 2015, "10.7554/elife", 0.018, 0.0, "Holocene (Extant)", "Danionin Pigment Evolution Series"),
    ("Danio", "albolineatus", "Pearl Danio (Uniform Interstripe)", "I. Enaima (Vertebrata)", "Pisces (Ichthyes)", "parichy", 2014, "10.1371/pgen", 0.036, 0.0, "Holocene (Extant)", "Danionin Pigment Evolution Series"),
    ("Danio", "nigrofasciatus", "Spotted Danio (broken ventral stripes)", "I. Enaima (Vertebrata)", "Pisces (Ichthyes)", "quigley", 2005, "10.1242/dev", 0.028, 0.0, "Holocene (Extant)", "Danionin Pigment Evolution Series"),
    ("Danio", "aesculapii", "Panther Danio (Vertical Bar Transition)", "I. Enaima (Vertebrata)", "Pisces (Ichthyes)", "spiewak", 2018, "10.7554/elife", 0.026, 0.0, "Holocene (Extant)", "Danionin Pigment Evolution Series"),
    ("Knightia", "eocaena", "Green River Eocene Clupeomorph Fossil", "I. Enaima (Vertebrata)", "Pisces (Ichthyes)", "grande", 1982, "10.1017/jpaleo", 0.095, 51.5, "Early Eocene (Ypresian, 51.5 Ma)", "Actinopterygian Axial Fossil Series"),
    ("Diplomystus", "dentatus", "Green River Eocene Armored Herring Fossil", "I. Enaima (Vertebrata)", "Pisces (Ichthyes)", "grande", 1984, "10.1017/jpaleo", 0.180, 51.5, "Early Eocene (Ypresian, 51.5 Ma)", "Actinopterygian Axial Fossil Series"),

    # --- Lineage 2: Stickleback (*Gasterosteus*) Miocene Fossil -> Marine -> Freshwater Chronospecies Series (5 taxa) ---
    ("Gasterosteus", "doryssus_early", "Miocene Stickleback Chronospecies I (Full Pelvic Girdle, T=0 kyr)", "I. Enaima (Vertebrata)", "Pisces (Ichthyes)", "bell", 2006, "10.1111/evo", 0.052, 10.02, "Late Miocene (Truckee Fm, 10.02 Ma)", "Gasterosteus Pelvic Reduction Chronospecies"),
    ("Gasterosteus", "doryssus_mid", "Miocene Stickleback Chronospecies II (Vestigial Spine, T=+9 kyr)", "I. Enaima (Vertebrata)", "Pisces (Ichthyes)", "bell", 2006, "10.1111/evo", 0.049, 10.01, "Late Miocene (Truckee Fm, 10.01 Ma)", "Gasterosteus Pelvic Reduction Chronospecies"),
    ("Gasterosteus", "doryssus_late", "Miocene Stickleback Chronospecies III (Pelvic Loss, T=+22 kyr)", "I. Enaima (Vertebrata)", "Pisces (Ichthyes)", "bell", 2006, "10.1111/evo", 0.047, 9.99, "Late Miocene (Truckee Fm, 9.99 Ma)", "Gasterosteus Pelvic Reduction Chronospecies"),
    ("Gasterosteus", "aculeatus_marine", "Ancestral Anadromous Complete-Morph Stickleback (32 Lateral Plates)", "I. Enaima (Vertebrata)", "Pisces (Ichthyes)", "colosimo", 2005, "10.1126/science", 0.065, 0.015, "Late Pleistocene Marine Ancestor (15 ka)", "Gasterosteus Pelvic Reduction Chronospecies"),
    ("Gasterosteus", "aculeatus_benthic", "Post-Glacial Pitx1-Deleted Freshwater Benthic Stickleback (Low Plate)", "I. Enaima (Vertebrata)", "Pisces (Ichthyes)", "chan", 2010, "10.1126/science", 0.058, 0.0, "Holocene Post-Glacial Lake (Extant)", "Gasterosteus Pelvic Reduction Chronospecies"),

    # --- Lineage 3: Fin-to-Limb Sarcopterygian -> Tetrapod Transition Series (6 taxa) ---
    ("Eusthenopteron", "foordi", "Late Devonian Tristichopterid Lobe-Fin Fish", "I. Enaima (Vertebrata)", "Pisces (Ichthyes)", "jarvik", 1980, "10.1038/nature", 0.140, 379.0, "Late Devonian (Frasnian, 379 Ma)", "Tetrapod Fin-to-Limb Fossil Transition"),
    ("Panderichthys", "rhombolepis", "Late Devonian Elpistostegalian Flat-Skulled Fish", "I. Enaima (Vertebrata)", "Pisces (Ichthyes)", "boisvert", 2008, "10.1038/nature", 0.165, 380.0, "Late Devonian (Frasnian, 380 Ma)", "Tetrapod Fin-to-Limb Fossil Transition"),
    ("Tiktaalik", "roseae", "Late Devonian Wristed Fin-Limb Transitional Fossil", "I. Enaima (Vertebrata)", "Amphibia & Reptilia (Ootoka)", "shubin", 2006, "10.1038/nature04639", 0.190, 375.0, "Late Devonian (Frasnian, 375 Ma)", "Tetrapod Fin-to-Limb Fossil Transition"),
    ("Acanthostega", "gunnari", "Late Devonian Eight-Digited Polydactyl Stem-Tetrapod", "I. Enaima (Vertebrata)", "Amphibia & Reptilia (Ootoka)", "coates", 1996, "10.1098/rstb", 0.115, 365.0, "Late Devonian (Famennian, 365 Ma)", "Tetrapod Fin-to-Limb Fossil Transition"),
    ("Ichthyostega", "stensioei", "Late Devonian Seven-Digited Weight-Bearing Tetrapod", "I. Enaima (Vertebrata)", "Amphibia & Reptilia (Ootoka)", "ahlberg", 2005, "10.1038/nature03893", 0.155, 363.0, "Late Devonian (Famennian, 363 Ma)", "Tetrapod Fin-to-Limb Fossil Transition"),
    ("Eryops", "megacephalus", "Early Permian Temnospondyl Pentadactyl Amphibian", "I. Enaima (Vertebrata)", "Amphibia & Reptilia (Ootoka)", "dilkes", 2015, "10.1017/jpaleo", 0.220, 295.0, "Early Permian (Cisuralian, 295 Ma)", "Tetrapod Fin-to-Limb Fossil Transition"),

    # --- Lineage 4: Equid (*Eohippus* -> *Equus*) Digit Reduction & Hypsodont Molar Chronospecies Series (5 taxa) ---
    ("Eohippus", "angustidens", "Early Eocene Four-Toed Dawn Horse (Hyracotherium)", "I. Enaima (Vertebrata)", "Mammalia (Zootoka)", "macfadden", 1992, "10.1017/cbo", 0.085, 52.0, "Early Eocene (Wasatchian, 52.0 Ma)", "Equid Autopod & Dental Chronospecies"),
    ("Mesohippus", "bairdi", "Late Eocene Three-Toed Sub-Hypsodont Horse", "I. Enaima (Vertebrata)", "Mammalia (Zootoka)", "macfadden", 1992, "10.1017/cbo", 0.125, 35.0, "Late Eocene (Chadronian, 35.0 Ma)", "Equid Autopod & Dental Chronospecies"),
    ("Merychippus", "insignis", "Miocene Grazing Spring-Footed Three-Toed Horse", "I. Enaima (Vertebrata)", "Mammalia (Zootoka)", "janis", 2008, "10.1111/evo", 0.180, 15.0, "Middle Miocene (Barstovian, 15.0 Ma)", "Equid Autopod & Dental Chronospecies"),
    ("Pliohippus", "pernix", "Late Miocene Single-Hoofed Monodactyl Ancestor", "I. Enaima (Vertebrata)", "Mammalia (Zootoka)", "kavanagh", 2020, "10.1038/s41586", 0.215, 8.5, "Late Miocene (Hemphillian, 8.5 Ma)", "Equid Autopod & Dental Chronospecies"),
    ("Equus", "scotti", "Pleistocene Giant Ice-Age Horse Chronospecies", "I. Enaima (Vertebrata)", "Mammalia (Zootoka)", "forsten", 1999, "10.1017/pal", 0.260, 0.6, "Middle Pleistocene (Irvingtonian, 0.6 Ma)", "Equid Autopod & Dental Chronospecies"),

    # --- Lineage 5: Cetacean Land-to-Sea Hindlimb Reduction & Fluke Evolution Series (5 taxa) ---
    ("Pakicetus", "attocki", "Early Eocene Terrestrial Wolf-Sized Stem-Cetacean", "I. Enaima (Vertebrata)", "Mammalia (Zootoka)", "thewissen", 2001, "10.1038/35095005", 0.160, 49.0, "Early Eocene (Ypresian, 49.0 Ma)", "Cetacean Limb-to-Flipper Fossil Series"),
    ("Ambulocetus", "natans", "Early Eocene Amphibious Walking Whale", "I. Enaima (Vertebrata)", "Mammalia (Zootoka)", "thewissen", 1994, "10.1126/science", 0.280, 47.5, "Early Eocene (Lutetian, 47.5 Ma)", "Cetacean Limb-to-Flipper Fossil Series"),
    ("Rodhocetus", "kasranii", "Middle Eocene Pelvic-Swimming Protocetid Whale", "I. Enaima (Vertebrata)", "Mammalia (Zootoka)", "gingerich", 2001, "10.1126/science", 0.240, 46.0, "Middle Eocene (Lutetian, 46.0 Ma)", "Cetacean Limb-to-Flipper Fossil Series"),
    ("Dorudon", "atrox", "Late Eocene Basilosaurid Whale (Vestigial Hindlimb)", "I. Enaima (Vertebrata)", "Mammalia (Zootoka)", "uhlen", 2004, "10.1017/pal", 0.340, 37.0, "Late Eocene (Priabonian, 37.0 Ma)", "Cetacean Limb-to-Flipper Fossil Series"),
    ("Monodon", "monoceros", "Arctic Narwhal (Chiral Helical Left Incisor Tusk)", "I. Enaima (Vertebrata)", "Mammalia (Zootoka)", "nweeia", 2014, "10.1002/ar", 1.850, 0.0, "Holocene (Extant)", "Cetacean Limb-to-Flipper Fossil Series"),

    # --- Lineage 6: Theropod-to-Avian Feather & Melanosome Evolution Series (6 taxa) ---
    ("Sinosauropteryx", "prima", "Early Cretaceous Proto-Feathered Compsognathid Dinosaur", "I. Enaima (Vertebrata)", "Aves (Ornithes)", "zhang", 2010, "10.1038/nature08740", 0.0012, 124.6, "Early Cretaceous (Yixian Fm, 124.6 Ma)", "Theropod-Avian Feather Fossil Series"),
    ("Anchiornis", "huxleyi", "Late Jurassic Four-Winged Paravian Dinosaur (Melanosome Array)", "I. Enaima (Vertebrata)", "Aves (Ornithes)", "li", 2010, "10.1126/science.1186290", 8.5e-7, 160.0, "Late Jurassic (Tiaojishan Fm, 160.0 Ma)", "Theropod-Avian Feather Fossil Series"),
    ("Archaeopteryx", "lithographica", "Late Jurassic Solnhofen Urvogel (Asymmetric Flight Vane)", "I. Enaima (Vertebrata)", "Aves (Ornithes)", "feoing", 2012, "10.1038/ncomms", 0.082, 150.0, "Late Jurassic (Tithonian, 150.0 Ma)", "Theropod-Avian Feather Fossil Series"),
    ("Microraptor", "gui", "Early Cretaceous Iridescent Black Four-Winged Dromaeosaur", "I. Enaima (Vertebrata)", "Aves (Ornithes)", "li", 2012, "10.1126/science.1213780", 9.2e-7, 120.0, "Early Cretaceous (Jiufotang Fm, 120.0 Ma)", "Theropod-Avian Feather Fossil Series"),
    ("Confuciusornis", "sanctus", "Early Cretaceous Pygostyle Beaked Bird (Ribbon Tail Streamers)", "I. Enaima (Vertebrata)", "Aves (Ornithes)", "chiappe", 1999, "10.1206/amnh", 0.140, 125.0, "Early Cretaceous (Yixian Fm, 125.0 Ma)", "Theropod-Avian Feather Fossil Series"),
    ("Hesperornis", "regalis", "Late Cretaceous Toothed Diving Bird", "I. Enaima (Vertebrata)", "Aves (Ornithes)", "bell", 2015, "10.1080/jvp", 0.195, 83.5, "Late Cretaceous (Campanian, 83.5 Ma)", "Theropod-Avian Feather Fossil Series"),

    # --- Lineage 7: Cephalopod Nautiloid -> Goniatitic -> Ceratitic -> Ammonitic & Heteromorph Suture Series (8 taxa) ---
    ("Orthoceras", "regulare", "Middle Ordovician Straight-Cone Nautiloid (Simple Ortho-Septa)", "II. Anaima (Invertebrata)", "Ostrakoderma (Mollusca & Echinodermata)", "kroger", 2008, "10.1017/pal", 0.045, 464.0, "Middle Ordovician (Darriwilian, 464 Ma)", "Cephalopod Fractal Suture Fossil Series"),
    ("Goniatites", "crenistria", "Mississippian Goniatitic Ammonoid (Angular Zigzag Suture, D_f=1.18)", "II. Anaima (Invertebrata)", "Ostrakoderma (Mollusca & Echinodermata)", "saunders", 1999, "10.1126/science", 0.018, 333.0, "Early Carboniferous (Viséan, 333 Ma)", "Cephalopod Fractal Suture Fossil Series"),
    ("Ceratites", "nodosus", "Middle Triassic Muschelkalk Ammonoid (Serrated Lobes, D_f=1.34)", "II. Anaima (Invertebrata)", "Ostrakoderma (Mollusca & Echinodermata)", "klug", 2004, "10.1111/let", 0.032, 240.0, "Middle Triassic (Ladinian, 240 Ma)", "Cephalopod Fractal Suture Fossil Series"),
    ("Dactylioceras", "commune", "Early Jurassic Whitby Bifurcating-Rib Ammonite (Fractal Suture, D_f=1.55)", "II. Anaima (Invertebrata)", "Ostrakoderma (Mollusca & Echinodermata)", "garcia", 2002, "10.1073/pnas", 0.042, 181.0, "Early Jurassic (Toarcian, 181 Ma)", "Cephalopod Fractal Suture Fossil Series"),
    ("Placenticeras", "meeki", "Late Cretaceous Iridescent Ammolite Ammonite (Hyper-Fractal Suture, D_f=1.68)", "II. Anaima (Invertebrata)", "Ostrakoderma (Mollusca & Echinodermata)", "inoue", 2016, "10.1126/sciadv", 0.068, 75.0, "Late Cretaceous (Campanian, 75 Ma)", "Cephalopod Fractal Suture Fossil Series"),
    ("Baculites", "compressus", "Late Cretaceous Uncoiled Straight-Shaft Heteromorph Ammonite", "II. Anaima (Invertebrata)", "Ostrakoderma (Mollusca & Echinodermata)", "landman", 2018, "10.1206/amnh", 0.110, 73.5, "Late Cretaceous (Pierre Shale, 73.5 Ma)", "Cephalopod Fractal Suture Fossil Series"),
    ("Nipponites", "mirabilis", "Late Cretaceous Meandering 3D Trefoil-Knot Heteromorph Ammonite", "II. Anaima (Invertebrata)", "Ostrakoderma (Mollusca & Echinodermata)", "okamoto", 1988, "10.1017/pal", 0.054, 88.0, "Late Cretaceous (Coniacian, 88 Ma)", "Cephalopod Fractal Suture Fossil Series"),
    ("Nautilus", "pompilius", "Extant Chambered Nautilus (Equiangular Logarithmic Siphuncle)", "II. Anaima (Invertebrata)", "Ostrakoderma (Mollusca & Echinodermata)", "ward", 1987, "10.1017/pal", 0.165, 0.0, "Holocene (Extant Living Fossil)", "Cephalopod Fractal Suture Fossil Series"),

    # --- Lineage 8: Trilobite Ontogenetic Segmentation (*Aulacopleura*) & Schizochroal Eye Chronospecies Series (7 taxa) ---
    ("Aulacopleura", "koninckii_meraspis", "Silurian Trilobite Meraspid Larva (12 Thoracic Segments, Early Morph)", "II. Anaima (Invertebrata)", "Entoma (Arthropoda)", "hughes", 1999, "10.1126/science", 0.0042, 428.5, "Silurian (Wenlock Epoch, 428.5 Ma)", "Trilobite Segmentation & Eye Fossil Series"),
    ("Aulacopleura", "koninckii_holaspis18", "Silurian Trilobite Holaspid Morphotype I (18 Thoracic Segments)", "II. Anaima (Invertebrata)", "Entoma (Arthropoda)", "fusco", 2004, "10.1038/nature", 0.016, 428.3, "Silurian (Wenlock Epoch, 428.3 Ma)", "Trilobite Segmentation & Eye Fossil Series"),
    ("Aulacopleura", "koninckii_holaspis22", "Silurian Trilobite Holaspid Morphotype II (22 Thoracic Segments)", "II. Anaima (Invertebrata)", "Entoma (Arthropoda)", "fusco", 2004, "10.1038/nature", 0.022, 428.1, "Silurian (Wenlock Epoch, 428.1 Ma)", "Trilobite Segmentation & Eye Fossil Series"),
    ("Eldredgeops", "rana_18file", "Middle Devonian Phacopid Trilobite (18-File Schizochroal Calcite Eye)", "II. Anaima (Invertebrata)", "Entoma (Arthropoda)", "eldredge", 1972, "10.1017/paleo", 0.00048, 387.0, "Middle Devonian (Cazenovian, 387 Ma)", "Trilobite Segmentation & Eye Fossil Series"),
    ("Eldredgeops", "rana_17file", "Middle Devonian Punctuated-Equilibrium Chronospecies (17-File Eye)", "II. Anaima (Invertebrata)", "Entoma (Arthropoda)", "eldredge", 1972, "10.1017/paleo", 0.00046, 384.0, "Middle Devonian (Tioughniogan, 384 Ma)", "Trilobite Segmentation & Eye Fossil Series"),
    ("Walliserops", "trifurcatus", "Middle Devonian Long-Trident Moroccan Trilobite", "II. Anaima (Invertebrata)", "Entoma (Arthropoda)", "gishlick", 2023, "10.1073/pnas", 0.048, 394.0, "Middle Devonian (Eifelian, 394 Ma)", "Trilobite Segmentation & Eye Fossil Series"),
    ("Anomalocaris", "canadensis", "Cambrian Burgess Shale Radiodont Apex Predator (Compound Eye & Oral Cone)", "II. Anaima (Invertebrata)", "Entoma (Arthropoda)", "paterson", 2011, "10.1038/nature10689", 0.028, 508.0, "Middle Cambrian (Burgess Shale, 508 Ma)", "Trilobite Segmentation & Eye Fossil Series"),

    # --- Lineage 9: Ediacaran Rangeomorph & Cambrian Stem-Metazoan Fractal Series (5 taxa) ---
    ("Charnia", "masoni", "Ediacaran Self-Similar Glide-Reflected Rangeomorph Frond (Mistaken Point)", "II. Anaima (Invertebrata)", "Zoophyta & Vermes (Cnidaria/Platyhelminthes)", "hoyall", 2004, "10.1126/science", 0.145, 565.0, "Late Ediacaran (565 Ma)", "Ediacaran-Cambrian Bodyplan Fossil Series"),
    ("Dickinsonia", "costata", "Ediacaran Glide-Symmetric Segmented Proarticulatan (Cholesterol Biomarker)", "II. Anaima (Invertebrata)", "Zoophyta & Vermes (Cnidaria/Platyhelminthes)", "bobrovskiy", 2018, "10.1126/science", 0.065, 558.0, "Late Ediacaran (White Sea, 558 Ma)", "Ediacaran-Cambrian Bodyplan Fossil Series"),
    ("Fractofusus", "misrai", "Ediacaran Four-Order Self-Similar Spindle Rangeomorph", "II. Anaima (Invertebrata)", "Zoophyta & Vermes (Cnidaria/Platyhelminthes)", "mitchell", 2015, "10.1038/nature14646", 0.092, 565.0, "Late Ediacaran (Newfoundland, 565 Ma)", "Ediacaran-Cambrian Bodyplan Fossil Series"),
    ("Cloudina", "hartmannae", "Terminal Ediacaran Cone-in-Cone Biomineralized Calcite Tube", "II. Anaima (Invertebrata)", "Zoophyta & Vermes (Cnidaria/Platyhelminthes)", "grant", 1990, "10.2475/ajs", 0.0028, 549.0, "Terminal Ediacaran (Nama Group, 549 Ma)", "Ediacaran-Cambrian Bodyplan Fossil Series"),
    ("Hallucigenia", "sparsa", "Middle Cambrian Burgess Shale Armored Lobopod", "II. Anaima (Invertebrata)", "Zoophyta & Vermes (Cnidaria/Platyhelminthes)", "smith", 2015, "10.1038/nature14573", 0.019, 508.0, "Middle Cambrian (Burgess Shale, 508 Ma)", "Ediacaran-Cambrian Bodyplan Fossil Series"),

    # --- Lineage 10: Early Land Plant Vascular Stele, Root & Leaf Evolution Series (7 taxa) ---
    ("Cooksonia", "pertoni", "Silurian Dichotomous Sporophyte Stem (Oldest Vascular Tracheids)", "III. Phyta (Embryophyta)", "Angiospermae & Gymnospermae", "edwards", 1992, "10.1038/nature", 0.0018, 425.0, "Late Silurian (Wenlock-Ludlow, 425 Ma)", "Paleobotanical Vascular & Leaf Fossil Series"),
    ("Rhynia", "gwynne_vaughanii", "Early Devonian Rhynie Chert Leafless Protostele Sporophyte", "III. Phyta (Embryophyta)", "Angiospermae & Gymnospermae", "kenrick", 1997, "10.1038/nature", 0.0032, 407.0, "Early Devonian (Pragian Rhynie Chert, 407 Ma)", "Paleobotanical Vascular & Leaf Fossil Series"),
    ("Asteroxylon", "mackiei", "Early Devonian Rhynie Chert Lycopsid (Actinostele & Fibonacci Microphylls)", "III. Phyta (Embryophyta)", "Angiospermae & Gymnospermae", "turner", 2023, "10.1126/science", 0.0065, 407.0, "Early Devonian (Pragian Rhynie Chert, 407 Ma)", "Paleobotanical Vascular & Leaf Fossil Series"),
    ("Archaeopteris", "hibernica", "Late Devonian Progymnosperm Tree (Planated Megaphyll Fronds)", "III. Phyta (Embryophyta)", "Angiospermae & Gymnospermae", "meyer", 1999, "10.1038/43662", 0.045, 370.0, "Late Devonian (Famennian, 370 Ma)", "Paleobotanical Vascular & Leaf Fossil Series"),
    ("Lepidodendron", "aculeatum", "Carboniferous Giant Scale-Tree (Helical Parastichy Leaf Cushions)", "III. Phyta (Embryophyta)", "Angiospermae & Gymnospermae", "dimichele", 1992, "10.2307/ajb", 0.038, 310.0, "Late Carboniferous (Pennsylvanian, 310 Ma)", "Paleobotanical Vascular & Leaf Fossil Series"),
    ("Glossopteris", "brownii", "Permian Gondwanan Seed Fern (Reticulate Closed-Loop Leaf Venation)", "III. Phyta (Embryophyta)", "Angiospermae & Gymnospermae", "mcloughlin", 2001, "10.1071/sb", 0.055, 260.0, "Late Permian (Gondwana, 260 Ma)", "Paleobotanical Vascular & Leaf Fossil Series"),
    ("Prototaxites", "logani", "Early Devonian Giant Nematophyte Mycelial Trunk (Concentric Hyphal Rings)", "IV. Protista & Monera", "Slime Molds & Bacteria", "boyce", 2007, "10.1130/geol", 0.350, 400.0, "Early Devonian (Emsian, 400 Ma)", "Paleobotanical Vascular & Leaf Fossil Series"),

    # --- Lineage 11: Ginkgoales Jurassic -> Paleogene -> Extant Leaf Dissection Chronospecies Series (4 taxa) ---
    ("Ginkgo", "yimaensis", "Middle Jurassic Deeply-Divided 6-Lobe Ginkgo Chronospecies", "III. Phyta (Embryophyta)", "Angiospermae & Gymnospermae", "zhou", 2009, "10.1098/rstb", 0.036, 170.0, "Middle Jurassic (Yima Fm, 170 Ma)", "Ginkgoales Leaf Dissection Chronospecies"),
    ("Ginkgo", "apodes", "Early Cretaceous Multi-Lobed Jehol Biota Ginkgo Chronospecies", "III. Phyta (Embryophyta)", "Angiospermae & Gymnospermae", "zheng", 2003, "10.1038/nature", 0.042, 121.0, "Early Cretaceous (Yixian Fm, 121 Ma)", "Ginkgoales Leaf Dissection Chronospecies"),
    ("Ginkgo", "adiantoides", "Paleocene-Eocene Bilobed Coalescent Ginkgo Chronospecies", "III. Phyta (Embryophyta)", "Angiospermae & Gymnospermae", "royer", 2003, "10.1086/ijps", 0.052, 56.0, "Paleocene-Eocene Boundary (56 Ma)", "Ginkgoales Leaf Dissection Chronospecies"),
    ("Ginkgo", "biloba", "Extant Maidenhair Tree (Dichotomous Open-Vein Fan Leaf)", "III. Phyta (Embryophyta)", "Angiospermae & Gymnospermae", "crane", 2013, "10.2307/yale", 0.058, 0.0, "Holocene (Extant Living Fossil)", "Ginkgoales Leaf Dissection Chronospecies"),

    # --- Lineage 12: Planktonic Foraminifera (*Globorotalia*) Miocene-Holocene Spiral Chronospecies Series (4 taxa) ---
    ("Globorotalia", "praefohsi", "Middle Miocene Unkeeled Rounded-Margin Foraminifer Chronospecies", "IV. Protista & Monera", "Mineralized Protists (Silica/Calcite)", "norris", 1996, "10.1017/pal", 0.00034, 13.4, "Middle Miocene (Serravallian, 13.4 Ma)", "Globorotalia Spire & Keel Chronospecies"),
    ("Globorotalia", "fohsi", "Middle Miocene Peripheral-Keel Punctuated Foraminifer Chronospecies", "IV. Protista & Monera", "Mineralized Protists (Silica/Calcite)", "malmgren", 1983, "10.1017/pal", 0.00042, 12.7, "Middle Miocene (Serravallian, 12.7 Ma)", "Globorotalia Spire & Keel Chronospecies"),
    ("Globorotalia", "plesiotumida", "Late Miocene Moderately-Swollen Trochospiral Foraminifer Chronospecies", "IV. Protista & Monera", "Mineralized Protists (Silica/Calcite)", "malmgren", 1983, "10.1017/pal", 0.00051, 6.2, "Late Miocene (Messinian, 6.2 Ma)", "Globorotalia Spire & Keel Chronospecies"),
    ("Globorotalia", "tumida", "Pliocene-Holocene Heavy-Keeled Biconvex Foraminifer Chronospecies", "IV. Protista & Monera", "Mineralized Protists (Silica/Calcite)", "hull", 2011, "10.1038/nature", 0.00068, 0.0, "Pliocene-Holocene (5.3–0 Ma)", "Globorotalia Spire & Keel Chronospecies"),

    # --- Lineage 13: Echinoderm Helicoplacoid -> Blastoid -> Crinoid -> Echinoid Symmetry Series (5 taxa) ---
    ("Helicoplacus", "gilberti", "Early Cambrian Triradial Helical-Plated Stem-Echinoderm", "II. Anaima (Invertebrata)", "Ostrakoderma (Mollusca & Echinodermata)", "durham", 1963, "10.1126/science", 0.032, 516.0, "Early Cambrian (Poleta Fm, 516 Ma)", "Echinoderm Symmetry Evolution Series"),
    ("Pentremites", "godoni", "Mississippian Pentaradial Hydrospire Blastoid Calyx", "II. Anaima (Invertebrata)", "Ostrakoderma (Mollusca & Echinodermata)", "waters", 1998, "10.1017/pal", 0.018, 330.0, "Mississippian (Chesterian, 330 Ma)", "Echinoderm Symmetry Evolution Series"),
    ("Encrinus", "liliiformis", "Middle Triassic Sea-Lily Crinoid (Stereom Gyroid Ossicles)", "II. Anaima (Invertebrata)", "Ostrakoderma (Mollusca & Echinodermata)", "gorzelak", 2014, "10.1130/geol", 0.000018, 240.0, "Middle Triassic (Muschelkalk, 240 Ma)", "Echinoderm Symmetry Evolution Series"),
    ("Micraster", "coranguinum", "Late Cretaceous Chalk Heart-Urchin Chronospecies (Petaloid Burrowing)", "II. Anaima (Invertebrata)", "Ostrakoderma (Mollusca & Echinodermata)", "kermack", 1954, "10.1098/rstb", 0.048, 85.0, "Late Cretaceous (Santonian, 85 Ma)", "Echinoderm Symmetry Evolution Series"),
    ("Strongylocentrotus", "purpuratus", "Purple Sea Urchin (Larval Calcite Spicule Triradiate Skeleton)", "II. Anaima (Invertebrata)", "Ostrakoderma (Mollusca & Echinodermata)", "davidson", 2002, "10.1126/science", 0.000085, 0.0, "Holocene (Extant)", "Echinoderm Symmetry Evolution Series"),

    # --- Lineage 14: Unseen Iconic Extant & Fossil Taxa Across All 4 Aristotelian Kingdoms (17 taxa) ---
    ("Ornithorhynchus", "anatinus", "Duck-Billed Platypus (Electroreceptive Bill Pore Hexagonal Array)", "I. Enaima (Vertebrata)", "Mammalia (Zootoka)", "pettigrew", 1999, "10.1242/jeb", 0.00014, 0.0, "Holocene (Extant)", "Iconic Vertebrate Sensory & Dermal Specializations"),
    ("Manis", "javanica", "Sunda Pangolin (Imbricated Keratin Scale Stress-Shielding Lattice)", "I. Enaima (Vertebrata)", "Mammalia (Zootoka)", "wang", 2016, "10.1016/actbio", 0.035, 0.0, "Holocene (Extant)", "Iconic Vertebrate Sensory & Dermal Specializations"),
    ("Dasypus", "novemcinctus", "Nine-Banded Armadillo (Hexagonal Osteoderm Rosette Tiles)", "I. Enaima (Vertebrata)", "Mammalia (Zootoka)", "chen", 2011, "10.1016/jmbbm", 0.0065, 0.0, "Holocene (Extant)", "Iconic Vertebrate Sensory & Dermal Specializations"),
    ("Loxodonta", "africana", "African Bush Elephant (Skin Thermal Mud-Cracking Channel Network)", "I. Enaima (Vertebrata)", "Mammalia (Zootoka)", "martins", 2018, "10.1038/s41467", 0.0012, 0.0, "Holocene (Extant)", "Iconic Vertebrate Sensory & Dermal Specializations"),
    ("Varanus", "komodoensis", "Komodo Dragon (Chainmail Cephalic Osteoderm Meshwork)", "I. Enaima (Vertebrata)", "Amphibia & Reptilia (Ootoka)", "maisano", 2019, "10.1002/ar", 0.0028, 0.0, "Holocene (Extant)", "Iconic Vertebrate Sensory & Dermal Specializations"),
    ("Moloch", "horridus", "Australian Thorny Devil (Capillary Drinking Skin Micro-Groove Network)", "I. Enaima (Vertebrata)", "Amphibia & Reptilia (Ootoka)", "comanns", 2016, "10.1098/rsif", 0.000095, 0.0, "Holocene (Extant)", "Iconic Vertebrate Sensory & Dermal Specializations"),
    ("Astyanax", "mexicanus", "Mexican Blind Cavefish (Optic Cup Degeneration vs Neuromast Expansion)", "I. Enaima (Vertebrata)", "Pisces (Ichthyes)", "jeffery", 2009, "10.1146/annurev", 0.00065, 0.0, "Holocene Cave Morph (Extant)", "Iconic Vertebrate Sensory & Dermal Specializations"),
    ("Hypsibius", "exemplaris", "Water Bear Tardigrade (Cryptobiotic Tun Buckling & Stylet Morphogenesis)", "II. Anaima (Invertebrata)", "Entoma (Arthropoda)", "gross", 2019, "10.1016/cub", 0.00022, 0.0, "Holocene (Extant)", "Panarthropod & Invertebrate Extremophiles"),
    ("Euperipatoides", "rowelli", "Australian Velvet Worm (Onychophoran Slime-Papilla & Lobopod Segments)", "II. Anaima (Invertebrata)", "Entoma (Arthropoda)", "mayer", 2015, "10.1186/evodevo", 0.0014, 0.0, "Holocene (Extant)", "Panarthropod & Invertebrate Extremophiles"),
    ("Odontodactylus", "scyllarus", "Peacock Mantis Shrimp (Bouligand Helicoidal Dactyl Club Nanocomposite)", "II. Anaima (Invertebrata)", "Entoma (Arthropoda)", "weaver", 2012, "10.1126/science.1218764", 6.5e-7, 0.0, "Holocene (Extant)", "Panarthropod & Invertebrate Extremophiles"),
    ("Chrysomallon", "squamiferum", "Scaly-Foot Hydrothermal Vent Gastropod (Greigite Iron-Sulfide Armor)", "II. Anaima (Invertebrata)", "Ostrakoderma (Mollusca & Echinodermata)", "yao", 2010, "10.1073/pnas", 4.5e-7, 0.0, "Holocene (Extant)", "Panarthropod & Invertebrate Extremophiles"),
    ("Euplectella", "aspergillum", "Venus' Flower Basket Glass Sponge (Hierarchical Silica Lattice Cage)", "II. Anaima (Invertebrata)", "Zoophyta & Vermes (Cnidaria/Platyhelminthes)", "aizenberg", 2005, "10.1126/science.1112255", 0.0015, 0.0, "Holocene (Extant)", "Panarthropod & Invertebrate Extremophiles"),
    ("Physalia", "physalis", "Portuguese Man o' War (Chiral Left/Right Dimorphic Pneumatophore Sail)", "II. Anaima (Invertebrata)", "Zoophyta & Vermes (Cnidaria/Platyhelminthes)", "munro", 2019, "10.1038/s41598", 0.120, 0.0, "Holocene (Extant)", "Panarthropod & Invertebrate Extremophiles"),
    ("Welwitschia", "mirabilis", "Namib Desert Welwitschia (Indeterminate Basal-Meristem Strap Leaves)", "III. Phyta (Embryophyta)", "Angiospermae & Gymnospermae", "wan", 2021, "10.1038/s41467", 0.850, 0.0, "Holocene (Extant)", "Extreme Plant & Protist Morphologies"),
    ("Victoria", "amazonica", "Giant Amazon Waterlily (Load-Bearing Vascular Girder Underside Ribs)", "III. Phyta (Embryophyta)", "Angiospermae & Gymnospermae", "box", 2022, "10.1126/sciadv", 1.450, 0.0, "Holocene (Extant)", "Extreme Plant & Protist Morphologies"),
    ("Nepenthes", "rajah", "Giant Bornean Pitcher Plant (Micro-Grooved Peristome Aquaplaning Rim)", "III. Phyta (Embryophyta)", "Angiospermae & Gymnospermae", "bohn", 2004, "10.1073/pnas", 0.000045, 0.0, "Holocene (Extant)", "Extreme Plant & Protist Morphologies"),
    ("Pyrocystis", "fusiformis", "Bioluminescent Marine Dinoflagellate (Shear-Induced Scintillon Flash)", "IV. Protista & Monera", "Mineralized Protists (Silica/Calcite)", "jabusch", 2020, "10.1103/prl", 0.00038, 0.0, "Holocene (Extant)", "Extreme Plant & Protist Morphologies"),
]

assert len(NINETY_NOVEL_AND_FOSSIL_TAXA) == 90, f"Expected 90 novel/fossil taxa, got {len(NINETY_NOVEL_AND_FOSSIL_TAXA)}"


# ============================================================================
# 4. Deterministic Phenomenon Scale Resolver for Existing Models (#0001–#1100 + Foundational)
# ============================================================================
COHORT_SCALE_MULTIPLIER: dict[int, tuple[float, str]] = {
    0: (0.04, "Turing pigment stripe / ocellus wavelength"),
    1: (0.18, "Morphogen gradient & pre-cartilage condensation zone"),
    2: (0.08, "Segmentation clock somite / metamere unit"),
    3: (0.25, "Auxin flux vein / phyllotactic primordium"),
    4: (0.0025, "Apical epithelial cell vertex polygon"),
    5: (0.35, "Bifurcating vascular / bronchial / hyphal branch"),
    6: (0.22, "Elastic bilayer fold / wrinkle / suture wavelength"),
    7: (0.55, "Biomineralized shell whorl / spicule / test"),
    8: (0.0008, "Gap-junction bioelectric domain & ion-channel patch"),
    9: (1.80, "Collective swarm / biofilm / colonial array"),
}

GENUS_ADULT_SIZE_M: dict[str, float] = {
    "Mus": 0.085, "Homo": 1.70, "Felis": 0.45, "Equus": 2.20, "Giraffa": 4.80,
    "Sus": 1.40, "Ovis": 1.50, "Cavia": 0.22, "Myotis": 0.09, "Tursiops": 2.80,
    "Gallus": 0.35, "Taeniopygia": 0.10, "Pavo": 1.10, "Columba": 0.32, "Struthio": 2.10,
    "Anas": 0.55, "Melopsittacus": 0.18, "Tyto": 0.36, "Aptenodytes": 1.15, "Calypte": 0.09,
    "Xenopus": 0.045, "Ambystoma": 0.18, "Chamaeleo": 0.35, "Anolis": 0.16, "Pleurodeles": 0.20,
    "Rana": 0.075, "Notophthalmus": 0.10, "Gekko": 0.25, "Trachemys": 0.22, "Alligator": 2.60,
    "Danio": 0.035, "Tetraodon": 0.45, "Oryzias": 0.032, "Hippocampus": 0.14, "Scyliorhinus": 0.55,
    "Gasterosteus": 0.055, "Poecilia": 0.035, "Salmo": 0.75, "Latimeria": 1.50, "Polypterus": 0.35,
    "Drosophila": 0.003, "Tribolium": 0.0035, "Apis": 0.014, "Morpho": 0.12, "Parhyale": 0.012,
    "Bombyx": 0.04, "Heliconius": 0.075, "Schistocerca": 0.07, "Daphnia": 0.003, "Limulus": 0.45,
    "Sepia": 0.30, "Octopus": 0.40, "Patella": 0.045, "Asterias": 0.20, "Lytechinus": 0.07,
    "Argonauta": 0.15, "Pecten": 0.11, "Turritella": 0.05, "Ophiocoma": 0.12, "Echinarachnius": 0.075,
    "Platynereis": 0.035, "Nematostella": 0.025, "Ciona": 0.08, "Aurelia": 0.22, "Acropora": 0.45,
    "Dugesia": 0.012, "Schmidtea": 0.012, "Hydra": 0.008, "Botryllus": 0.04, "Clytia": 0.015,
    "Hirudo": 0.08, "Mnemiopsis": 0.09, "Conus": 0.075, "Caenorhabditis": 0.001,
    "Arabidopsis": 0.15, "Cardamine": 0.18, "Brassica": 0.25, "Helianthus": 1.60,
    "Marchantia": 0.05, "Physcomitrium": 0.008, "Zea": 2.10, "Solanum": 0.85, "Pinus": 8.0,
    "Antirrhinum": 0.45, "Utricularia": 0.03, "Mimosa": 0.40, "Selaginella": 0.15, "Ceratopteris": 0.25,
    "Dionaea": 0.03, "Bauhinia": 0.12, "Passiflora": 0.08, "Lactuca": 0.20,
    "Neurospora": 0.04, "Aspergillus": 0.015, "Coprinopsis": 0.08, "Pseudomonas": 0.002,
    "Anabaena": 0.0005, "Saccharomyces": 0.00008, "Schizosaccharomyces": 0.00012,
    "Streptomyces": 0.004, "Paenibacillus": 0.035, "Proteus": 0.045, "Bacillus": 0.025,
    "Physarum": 0.12, "Dictyostelium": 0.004,
    "Volvox": 0.0008, "Acetabularia": 0.04, "Acanthometra": 0.0003, "Micrasterias": 0.0002,
    "Chlamydomonas": 0.000015, "Stentor": 0.0012, "Tetrahymena": 0.00005, "Gephyrocapsa": 0.000008,
    "Emiliania": 0.000006, "Arachnoidiscus": 0.00025, "Orbulina": 0.0006, "Thalassiosira": 0.00004,
}


def compute_phenomenon_scale_for_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Compute or refine physical phenomenon scale metadata for any model entry."""
    if "phenomenon_size_m" in entry and entry["phenomenon_size_m"]:
        size_m = float(entry["phenomenon_size_m"])
        tier_id, tier_short, tier_num = classify_scale_tier(size_m)
        return {
            "phenomenon_size_m": signif(size_m, 4),
            "log10_size_m": round(math.log10(max(size_m, 1e-10)), 3),
            "size_human": format_size_human(size_m),
            "scale_tier": tier_id,
            "scale_tier_short": tier_short,
            "scale_tier_num": tier_num,
            "anatomical_structure": entry.get("anatomical_structure", entry.get("poion_form", "")),
            "geological_age_ma": float(entry.get("geological_age_ma", 0.0)),
            "geological_period": entry.get("geological_period", "Holocene (Extant)"),
            "fossil_lineage_group": entry.get("fossil_lineage_group", ""),
            "external_datasets": entry.get("external_datasets", []),
        }

    genus = entry.get("genus", "Danio")
    cause = entry.get("efficient_cause", "1. Chemical Turing / RD")
    slot = entry.get("slot") or 1
    c_idx = max(0, min(9, int(cause.split(".")[0]) - 1)) if "." in cause else (int(slot) % 10)
    mult, struct_desc = COHORT_SCALE_MULTIPLIER.get(c_idx, (0.1, "Morphogenetic feature"))
    adult_m = GENUS_ADULT_SIZE_M.get(genus, 0.05)

    # Add deterministic fine variation by slot so multiple runs on the same genus span distinct physical scales
    jitter = 0.72 + 0.07 * ((int(slot) * 13) % 9)
    size_m = max(2e-9, min(12.0, adult_m * mult * jitter))
    tier_id, tier_short, tier_num = classify_scale_tier(size_m)

    ext_ds = ZEBRAHUB_EXTERNAL_DATASETS if genus == "Danio" else []
    return {
        "phenomenon_size_m": signif(size_m, 4),
        "log10_size_m": round(math.log10(size_m), 3),
        "size_human": format_size_human(size_m),
        "scale_tier": tier_id,
        "scale_tier_short": tier_short,
        "scale_tier_num": tier_num,
        "anatomical_structure": f"{struct_desc} in {entry.get('species', genus)}",
        "geological_age_ma": 0.0,
        "geological_period": "Holocene (Extant)",
        "fossil_lineage_group": "Danionin Pigment Evolution Series" if genus == "Danio" else "",
        "external_datasets": ext_ds,
    }


def signif(x: float, digits: int = 4) -> float:
    if x == 0 or not math.isfinite(x):
        return 0.0
    return round(x, digits - int(math.floor(math.log10(abs(x)))) - 1)
