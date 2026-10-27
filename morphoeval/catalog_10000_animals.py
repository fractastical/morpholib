#!/usr/bin/env python3
"""
morphoeval/catalog_10000_animals.py — Deterministic 10,000-Loop Animal Catalog.

Extends the existing atlas (slots #0001-#3100, 3,116 entries in
``morphoeval/runs/aristotelian_categories_index.json``) to slot #10000:

  * Slots #3101-#10000 = 6,900 NEW loops. Every loop is one
    animal x spatial-scale x biophysical-cohort combination ("basic look"):
    the same 12-step closed loop used by slots #0001-#0100, run at a
    deliberately different physical scale (8 logarithmic tiers, 1 nm - 15 m)
    and, for fossil taxa, at a different point in evolutionary history
    (chronospecies ordered oldest -> youngest within each lineage).
  * Taxa pool is built only from data already in the repo:
    ``datasets/species_gallery/species_manifest.json`` (273 taxa),
    ``NINETY_NOVEL_AND_FOSSIL_TAXA`` (90 extant + fossil chronospecies,
    scale_and_paleo_catalog), the 40-taxon Drosophila radiation and the
    60-rung *D. melanogaster* ladder (drosophila_1000_catalog).
  * Each taxon receives ~10 cohorts x >=2 scale variants, so every animal
    is looked at at multiple scales, not just its adult body size.
  * Each slot spec carries the remediation contract consumed by
    ``morphoeval/cloud_campaign_10000.py``: Pass 1 baseline, remediation
    loop 1 (Pass 2, always), remediation loop 2 (Pass 3, conditional on
    failing the image-fitness / audit gates), and the image-fitness gate
    itself (threshold 70, CV similarity >= 58, animated GIF, from
    ``morphoeval/verify_image_fitness.py``).

Stdlib-only (plus the stdlib-only ``scale_and_paleo_catalog`` /
``drosophila_1000_catalog``) so it runs on a bare cloud VM before
``requirements.txt`` finishes installing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    from morphoeval.scale_and_paleo_catalog import (
        COHORT_SCALE_MULTIPLIER,
        GENUS_ADULT_SIZE_M,
        NINETY_NOVEL_AND_FOSSIL_TAXA,
        classify_scale_tier,
        format_size_human,
        signif,
    )
    from morphoeval.drosophila_1000_catalog import (
        DROSOPHILA_MELANOGASTER_60_RUNGS,
        DROSOPHILA_RADIATION_40_TAXA,
    )
except ImportError:  # run as a plain script inside morphoeval/
    from scale_and_paleo_catalog import (  # type: ignore[no-redef]
        COHORT_SCALE_MULTIPLIER,
        GENUS_ADULT_SIZE_M,
        NINETY_NOVEL_AND_FOSSIL_TAXA,
        classify_scale_tier,
        format_size_human,
        signif,
    )
    from drosophila_1000_catalog import (  # type: ignore[no-redef]
        DROSOPHILA_MELANOGASTER_60_RUNGS,
        DROSOPHILA_RADIATION_40_TAXA,
    )

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

START_SLOT = 3101
END_SLOT = 10000
TOTAL_NEW_SLOTS = END_SLOT - START_SLOT + 1  # 6,900
CAMPAIGN_TARGET_SLOTS = END_SLOT  # slot #10000 == the 10,000-loop target

TEN_COHORTS: list[dict[str, str]] = [
    {"slug_suffix": "turing-wavelength", "domain": "Turing Reaction-Diffusion Wavelength Selection", "cause": "1. Chemical Turing / RD", "poion": "Stripes, Spots & Ocelli", "poson": "Turing Wavenumber k_c"},
    {"slug_suffix": "morphogen-condensation", "domain": "Morphogen Gradient Positional Condensation", "cause": "2. Morphogen + Condensations", "poion": "Skeletal Rays & Joints", "poson": "French-Flag / PSHM Thresholds"},
    {"slug_suffix": "clock-wavefront", "domain": "Phase-Coupled Oscillator & Wavefront Arrest", "cause": "3. Phase/Delay Oscillators", "poion": "Periodic Somites & Segments", "poson": "Somite Length S = u*T_clock"},
    {"slug_suffix": "flux-canalization", "domain": "Polar Flux Canalization & Anisotropic Growth", "cause": "4. Polar Auxin & Turgor Stress", "poion": "Spirals, Veins & Puzzle Cells", "poson": "Golden Angle / Auxin Flux"},
    {"slug_suffix": "vertex-packing", "domain": "Epithelial Vertex Energy & Topological Packing", "cause": "5. Vertex / Potts / Nematics", "poion": "Polygonal Packing & Furrows", "poson": "Shape Index p_0* = 3.813"},
    {"slug_suffix": "branching-tree", "domain": "Bifurcating Tree Murray Power-Law & Tip Splitting", "cause": "6. Tip-Splitting & Flow Trees", "poion": "Bifurcating Vascular/Lung Trees", "poson": "Murray Law r_0^3 = r_1^3 + r_2^3"},
    {"slug_suffix": "elastic-buckling", "domain": "Differential Growth Elastic Bilayer Buckling", "cause": "7. Elastic Shell/Sheet Buckling", "poion": "Gyri, Villi, Loops & Snap-Shells", "poson": "Wrinkling lambda ~ h(E_f/E_s)^1/3"},
    {"slug_suffix": "biomineral-accretion", "domain": "Logarithmic Accretion & Minimal-Surface Biomineralization", "cause": "8. Crystal Accretion & TPMS", "poion": "Coiled Shells, Spicules & Gyroids", "poson": "Raup (W,D,S) / Gyroid Mean H=0"},
    {"slug_suffix": "bioelectric-polarity", "domain": "Gap-Junction Bioelectric Bistability & Target Polarity", "cause": "9. Bioelectric V_mem & Attractors", "poion": "Target Polarity & Neural Maps", "poson": "Bistable V_mem / G_ij Threshold"},
    {"slug_suffix": "collective-stigmergy", "domain": "Chemotactic & Stigmergic Collective Pattern Formation", "cause": "10. Stigmergy & Active Swarms", "poion": "Biofilm Wrinkles & Hex Combs", "poson": "Chemotactic / Stigmergic Pe"},
]

SCALE_VARIANT_FACTORS: list[tuple[str, float]] = [
    ("detail", 0.02),
    ("characteristic", 1.0),
    ("context", 25.0),
]

IMAGE_FITNESS_THRESHOLD = 70.0
CV_SIM_MIN = 58.0
REMEDIATION_LOOPS = "2 always-planned (Pass 2), 3rd (Pass 3) conditional on gate failure"


def _fallback_size_m(genus: str) -> float:
    if genus in GENUS_ADULT_SIZE_M:
        return float(GENUS_ADULT_SIZE_M[genus])
    h = sum((i + 1) * ord(c) for i, c in enumerate(genus.lower()))
    return float(signif(0.01 * (10 ** ((h % 160) / 40.0 - 1.0)), 4))


def load_taxa_pool() -> list[dict[str, Any]]:
    """All animals available in-repo, de-duplicated by (genus, epithet)."""
    pool: dict[tuple[str, str], dict[str, Any]] = {}

    def add(t: dict[str, Any]) -> None:
        key = (t["genus"].lower(), t["species_epithet"].lower())
        if key not in pool or (t.get("geological_age_ma", 0) > 0 and not pool[key].get("geological_age_ma")):
            pool[key] = t

    manifest_path = ROOT / "datasets" / "species_gallery" / "species_manifest.json"
    if manifest_path.exists():
        for m in json.loads(manifest_path.read_text(encoding="utf-8")):
            genus = m["genus"]
            epithet = m["species"].split()[-1] if " " in m.get("species", "") else m["species"]
            add({
                "genus": genus, "species_epithet": epithet,
                "organism": m.get("organism") or f"{genus} {epithet}",
                "common_name": m.get("common_name", ""),
                "megiston_genos": m.get("megiston_genos", ""),
                "subaltern_genos": m.get("subaltern_genos", ""),
                "base_size_m": float(m.get("characteristic_size_m") or _fallback_size_m(genus)),
                "geological_age_ma": float(m.get("geological_age_ma") or 0.0),
                "geological_period": m.get("geological_period", "Holocene (Extant)"),
                "fossil_lineage_group": m.get("fossil_lineage_group", ""),
                "author": "", "year": 2026, "doi": "",
                "gallery_path": m.get("gallery_path", ""),
            })

    for (genus, species, common, megiston, subaltern, author, year, doi_pfx,
         base_size_m, age_ma, period, lineage) in NINETY_NOVEL_AND_FOSSIL_TAXA:
        add({
            "genus": genus, "species_epithet": species,
            "organism": f"{genus} {species.replace('_', ' ')}",
            "common_name": common, "megiston_genos": megiston, "subaltern_genos": subaltern,
            "base_size_m": float(base_size_m), "geological_age_ma": float(age_ma),
            "geological_period": period, "fossil_lineage_group": lineage,
            "author": author, "year": int(year),
            "doi": doi_pfx if len(doi_pfx.split("/")[-1]) > 6 else "",
            "gallery_path": "",
        })

    for (epithet, common, lineage, base_size_m, age_ma, period, author, year, doi) in DROSOPHILA_RADIATION_40_TAXA:
        add({
            "genus": "Drosophila", "species_epithet": epithet,
            "organism": f"Drosophila {epithet}", "common_name": common,
            "megiston_genos": "II. Anaima (Invertebrata)", "subaltern_genos": "Entoma (Arthropoda)",
            "base_size_m": float(base_size_m), "geological_age_ma": float(age_ma),
            "geological_period": period, "fossil_lineage_group": lineage,
            "author": author, "year": int(year), "doi": doi, "gallery_path": "",
        })

    add({
        "genus": "Drosophila", "species_epithet": "melanogaster",
        "organism": "Drosophila melanogaster", "common_name": "Fruit Fly",
        "megiston_genos": "II. Anaima (Invertebrata)", "subaltern_genos": "Entoma (Arthropoda)",
        "base_size_m": 2.5e-3, "geological_age_ma": 0.0,
        "geological_period": "Holocene (Extant)",
        "fossil_lineage_group": "Drosophila melanogaster Multi-Scale FlyBase & FlyWire Connectome Atlas (2.5 nm -> 1.2 m)",
        "author": "lewis", "year": 1978, "doi": "10.1038/276565a0", "gallery_path": "",
        "scale_rungs": [r[0] for r in DROSOPHILA_MELANOGASTER_60_RUNGS],
    })

    taxa = sorted(pool.values(), key=lambda t: (t["genus"].lower(), -t["geological_age_ma"], t["species_epithet"].lower()))
    _annotate_evolutionary_history(taxa)
    return taxa


def _annotate_evolutionary_history(taxa: list[dict[str, Any]]) -> None:
    by_genus: dict[str, list[dict[str, Any]]] = {}
    for t in taxa:
        by_genus.setdefault(t["genus"], []).append(t)
    for genus, group in by_genus.items():
        chrono = sorted(group, key=lambda t: (-t["geological_age_ma"], t["species_epithet"].lower()))
        for rank, t in enumerate(chrono):
            t["chrono_rank_oldest_first"] = rank + 1
            t["chrono_total_in_genus"] = len(chrono)
            t["chrono_predecessor"] = chrono[rank - 1]["organism"] if rank > 0 else ""
            t["chrono_successor"] = chrono[rank + 1]["organism"] if rank + 1 < len(chrono) else ""
            t["evolutionary_axis"] = (
                f"{genus} chronoseries {rank + 1}/{len(chrono)}: "
                f"{t['geological_period']} ({t['geological_age_ma']:g} Ma)"
                + (f"; predecessor {t['chrono_predecessor']}" if t["chrono_predecessor"] else "")
                + (f"; successor {t['chrono_successor']}" if t["chrono_successor"] else "")
            )


def build_slot_spec(slot: int, taxa: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if not START_SLOT <= slot <= END_SLOT:
        raise ValueError(f"slot {slot} outside {START_SLOT}..{END_SLOT}")
    taxa = taxa if taxa is not None else load_taxa_pool()
    i = slot - START_SLOT
    taxon = taxa[i % len(taxa)]
    run_per_taxon = i // len(taxa)
    cohort = TEN_COHORTS[run_per_taxon % len(TEN_COHORTS)]
    variant_name, variant_factor = SCALE_VARIANT_FACTORS[(run_per_taxon // len(TEN_COHORTS)) % len(SCALE_VARIANT_FACTORS)]
    c_idx = run_per_taxon % len(TEN_COHORTS)
    mult, struct_desc = COHORT_SCALE_MULTIPLIER[c_idx]
    jitter = 0.72 + 0.07 * ((slot * 13) % 9)
    size_m = float(signif(max(2e-9, min(12.0, taxon["base_size_m"] * mult * variant_factor * jitter)), 4))
    tier_id, tier_short, tier_num = classify_scale_tier(size_m)
    size_human = format_size_human(size_m)

    epithet_slug = taxon["species_epithet"].lower().replace("_", "-")
    slug = f"{taxon['genus'].lower()}-{epithet_slug}-{cohort['slug_suffix']}-s{run_per_taxon // len(TEN_COHORTS)}"
    snake = slug.replace("-", "_")
    author = taxon["author"] or "delphi10000"
    year = int(taxon["year"]) if taxon["author"] else 2026
    sim_id = f"{author}-{slug}"
    return {
        "slot": slot,
        "slot_id": f"slot-{slot:04d}-{slug}",
        "campaign": "10000-animals-basic-look",
        "genus": taxon["genus"],
        "species_epithet": taxon["species_epithet"],
        "organism": taxon["organism"],
        "common_name": taxon["common_name"],
        "megiston_genos": taxon["megiston_genos"],
        "subaltern_genos": taxon["subaltern_genos"],
        "cohort_idx": c_idx,
        "cohort_domain": cohort["domain"],
        "efficient_cause": cohort["cause"],
        "poion_form": cohort["poion"],
        "poson_law": cohort["poson"],
        "slug": slug,
        "snake": snake,
        "sim_id": sim_id,
        "script": f"models/{year}-{author}-{slug}.py",
        "dataset_dir": f"datasets/{snake}",
        "lean_module": f"morphoeval/lean/loops/Loop_{slot:04d}_{snake}.lean",
        "phenomenon_size_m": size_m,
        "log10_size_m": round(math.log10(size_m), 3),
        "size_human": size_human,
        "scale_tier": tier_id,
        "scale_tier_short": tier_short,
        "scale_tier_num": tier_num,
        "scale_variant": variant_name,
        "anatomical_structure": f"{struct_desc} in {taxon['organism']} ({size_human}, {variant_name} scale)",
        "geological_age_ma": taxon["geological_age_ma"],
        "geological_period": taxon["geological_period"],
        "fossil_lineage_group": taxon["fossil_lineage_group"],
        "evolutionary_axis": taxon["evolutionary_axis"],
        "chrono_rank_oldest_first": taxon["chrono_rank_oldest_first"],
        "chrono_total_in_genus": taxon["chrono_total_in_genus"],
        "seed_doi": taxon["doi"],
        "gallery_path": taxon["gallery_path"],
        "remediation": {
            "pass1": "baseline 12-step loop (scout/paper/source/rules/script/output)",
            "remediation_loop_1": "Pass 2 always: audit v2 + measure + parameter sweep + ablation + specimen CV segmentation",
            "remediation_loop_2": "Pass 3 conditional: audit v3, ONLY if any gate below fails after Pass 2",
            "max_remediation_loops": 2,
            "image_fitness_gate": {
                "image_fitness_min": IMAGE_FITNESS_THRESHOLD,
                "cv_sim_similarity_min": CV_SIM_MIN,
                "animated_gif_min_frames": 6,
                "min_resolution_px": 340,
                "non_circular_specimen_metrics_min": 1,
                "verifier": "morphoeval/verify_image_fitness.py (per-loop, index-rewrite reverted)",
            },
            "stop_when": "image_fitness>=70 AND cv_sim>=58 AND pearson_r>0.98 AND audit>=85; else Pass 3, then stop and flag",
        },
    }


def build_all_specs() -> list[dict[str, Any]]:
    taxa = load_taxa_pool()
    return [build_slot_spec(s, taxa) for s in range(START_SLOT, END_SLOT + 1)]


def campaign_summary(specs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    specs = specs if specs is not None else build_all_specs()
    tiers: dict[str, int] = {}
    cohorts: dict[str, int] = {}
    for s in specs:
        tiers[s["scale_tier"]] = tiers.get(s["scale_tier"], 0) + 1
        cohorts[s["efficient_cause"]] = cohorts.get(s["efficient_cause"], 0) + 1
    return {
        "campaign": "10000-animals-basic-look",
        "slot_range": [START_SLOT, END_SLOT],
        "new_loops": len(specs),
        "campaign_target_slots": CAMPAIGN_TARGET_SLOTS,
        "unique_animals": len({(s["genus"], s["species_epithet"]) for s in specs}),
        "unique_genera": len({s["genus"] for s in specs}),
        "fossil_loops_age_gt_0": sum(1 for s in specs if s["geological_age_ma"] > 0),
        "extant_loops": sum(1 for s in specs if s["geological_age_ma"] == 0),
        "oldest_age_ma": max(s["geological_age_ma"] for s in specs),
        "scale_tiers_covered": sorted({s["scale_tier_num"] for s in specs}),
        "loops_by_scale_tier": tiers,
        "loops_by_cohort": cohorts,
        "remediation_loops_per_animal": REMEDIATION_LOOPS,
        "image_fitness_threshold": IMAGE_FITNESS_THRESHOLD,
        "cv_sim_min": CV_SIM_MIN,
    }


if __name__ == "__main__":
    print(json.dumps(campaign_summary(), indent=2))
