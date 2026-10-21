#!/usr/bin/env python3
"""
planform_bioelectric.py - L1 evidence layer for the Cervera-Levin-Mafe v3 reproduction.

Extracts the nine bioelectric experiment series S1-S9 from the SHA-pinned
PlanformDB 2.5.0 database (Lobo, Malone & Levin 2013) into a notice-compliant
derived fixture:

    datasets/planform/planform_bioelectric_series_v1.json

The record shape is MesmerPrism/rusty-matter's
``rusty.matter.fields.planformdb_derived_fixture.v1`` (see
datasets/planform/rusty_matter_planformdb_derived_v0.json) extended with

    series_id, mechanism_class, drug_start_days, drug_end_days, drug_exposures,
    rnai, series_axis, literature_anchors, row_flags

so that WP2-WP4 can fit a link function on S1 and test on S2-S9 without ever
reading the database again. Only identifiers, normalized labels, sample counts,
observation periods, drug windows and outcome frequencies are written; region
geometry, organ tables, experiment free text and morphology images stay in the
database (Lobo Lab notice: raw rows are not redistributed).

Damaged negatives (bad frequency sum, missing notice, unknown morphology id,
SHA mismatch, unknown mechanism class) are written next to the fixture under
datasets/planform/damaged/ so a consumer's self-test fails closed.

Series (PlanformDB experiment ids; publication ids in parentheses)
    S1  Oviedo 2010 Fig. 1A   415-419      octanol x AP crop position            (pub 1)
    S2  Oviedo 2010 Fig. 3D   446-450 +445 Dj-inx-12 + Dj-inx-5/13 RNAi x crop   (pub 1)
    S3  Nogi & Levin 2005 10IB 325-329 +322-324 heptanol x AP crop position      (pub 20)
    S4  Oviedo 2010 Fig. 2A   434-440      octanol onset delay, post-phar. crop  (pub 1)
    S5  Oviedo 2010 Fig. 2B   441-444      octanol + VNC disruption timing       (pub 1)
    S6  Oviedo 2010 Fig. 1B   420-423      head-retained fragments under octanol (pub 1)
    S7  Beane 2011 Fig. 1D    391-395      SCH-28080 x AP crop position          (pub 4)
    S8  Beane 2011 Fig. 6B    411-413      ivermectin / SCH-28080 / Cl-, pharynx (pub 4)
    S9  Nogi 2009 Fig. 1D-E   516-536      praziquantel dose and exposure window (pub 25)

Usage
    # extract (DB from --db, $PLANFORM_DB_PATH, or the default local path)
    python3 planform_bioelectric.py --db /path/to/planformDB_2.5.0.edb

    # validate an existing fixture without the database
    python3 planform_bioelectric.py --validate datasets/planform/planform_bioelectric_series_v1.json

    # self-test: validator, rusty-matter parity, census expectations, damaged
    # negatives, and (if the pinned DB is present) byte-identical regeneration
    python3 planform_bioelectric.py --self-test

Library use (WP2+):
    from planform_bioelectric import load_fixture, series_records, outcome_table
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parent.parent
PLANFORM_DATASET_DIR = REPO_ROOT / "datasets" / "planform"
DEFAULT_OUT = PLANFORM_DATASET_DIR / "planform_bioelectric_series_v1.json"
DEFAULT_DAMAGED_DIR = PLANFORM_DATASET_DIR / "damaged"
RUSTY_MATTER_V0 = PLANFORM_DATASET_DIR / "rusty_matter_planformdb_derived_v0.json"
DEFAULT_DB_CANDIDATES = [
    Path("/Users/jdietz/Documents/Levin/PlanformDB/planformDB_2.5.0.edb"),
]

# --------------------------------------------------------------------------
# Database pin (same digest as 2013-lobo-planform-morphology.py and
# datasets/planform/planformDB.sha256)
# --------------------------------------------------------------------------
PLANFORM_DB_SHA256 = "9effd13ddb87664b9ef7a9b6c9c1959b502fcdd6c1b06ee016501b2d0be83b89"
PLANFORM_DB_BYTES = 744448
PLANFORM_DB_STRUCT_VERSION = 2

SCHEMA_ID = "rusty.matter.fields.planformdb_derived_fixture.v1"
SCHEMA_EXTENSION_ID = "infinitemorphospace.planform_bioelectric_series.v1"
FIXTURE_ID = "planform-bioelectric-series-v1"
SELECTED_ON = "2026-10-18"

# --------------------------------------------------------------------------
# Lobo Lab notice. The numbered restrictions, warranty paragraph and citation
# are the Planform / PlanformDB notice text and are reproduced verbatim (they
# are the same clauses carried by the vendored rusty-matter fixture). Only the
# first paragraph, describing what this particular fixture is, differs.
# --------------------------------------------------------------------------
LOBO_NOTICE_COPYRIGHT = (
    "All contents of Planform, Copyright 2012-2016 Daniel Lobo. Use for research and "
    "educational purposes is free of charge, provided proper citation is given to the "
    "database and the functional work that gave rise to the datasets used."
)
NOTICE_TEXT = (
    "Planform / PlanformDB Notice\n\n"
    "Source: Lobo Lab PlanformDB 2.5.0\n"
    "Source page: https://lobolab.umbc.edu/planform/download/\n\n"
    "This infinitemorphospace fixture is a small transformed subset of PlanformDB metadata "
    "(experiment, result-set and morphology identifiers, sample counts, observation periods, "
    "drug exposure windows and outcome frequencies). It does not redistribute the raw SQLite "
    "database, region geometry, organ tables, paper figures, or morphology images.\n\n"
    f"{LOBO_NOTICE_COPYRIGHT}\n\n"
    "Planform and PlanformDB are provided as-is, without any express or implied warranty. "
    "The authors are not liable for damages arising from use of this software or database.\n\n"
    "Permission is granted to use and redistribute Planform and PlanformDB freely, subject to "
    "these restrictions:\n\n"
    "1. The origin of the software and database must not be misrepresented.\n"
    "2. Works using the software or database require acknowledgment and citation of the "
    "Planform publications.\n"
    "3. This notice may not be removed or altered from any distribution.\n\n"
    "Citation for the database/application:\n\n"
    "Lobo D, Malone TJ, Levin M. Planform: an application and database of graph-encoded "
    "planarian regenerative experiments. Bioinformatics 29(8), 1098-1100, 2013. "
    "DOI: 10.1093/bioinformatics/btt088"
)
# A fixture whose notice lacks any of these is rejected by the validator.
NOTICE_REQUIRED_CLAUSES = (
    "Lobo Lab PlanformDB 2.5.0",
    "The origin of the software and database must not be misrepresented.",
    "This notice may not be removed or altered from any distribution.",
    "10.1093/bioinformatics/btt088",
)

# --------------------------------------------------------------------------
# Controlled vocabularies (validated against the database where applicable)
# --------------------------------------------------------------------------
PUBLICATIONS: Dict[int, Dict[str, Any]] = {
    1: dict(
        source_id="oviedo_2010_devbiol",
        label="Oviedo et al. 2010 - long-range neural and gap-junction cues",
        doi="10.1016/j.ydbio.2009.12.012",
        year=2010,
    ),
    20: dict(
        source_id="nogi_2005_devbiol",
        label="Nogi & Levin 2005 - innexin expression and gap-junctional communication in planarian regeneration",
        doi="10.1016/j.ydbio.2005.09.002",
        year=2005,
    ),
    4: dict(
        source_id="beane_2011_chembiol",
        label="Beane et al. 2011 - H,K-ATPase-mediated membrane voltage and planarian head regeneration",
        doi="10.1016/j.chembiol.2010.11.012",
        year=2011,
    ),
    25: dict(
        source_id="nogi_2009_plosntd",
        label="Nogi et al. 2009 - praziquantel, voltage-operated Ca2+ channel beta subunits and regenerative polarity",
        doi="10.1371/journal.pntd.0000464",
        year=2009,
    ),
}
LOBO_CITATION = dict(
    id="lobo_2013_planform",
    label="Lobo, Malone, and Levin 2013 - Planform",
    doi="10.1093/bioinformatics/btt088",
    url="https://doi.org/10.1093/bioinformatics/btt088",
)

SPECIES: Dict[int, Tuple[str, str]] = {1: ("Dugesia japonica", "dugesia_japonica")}

# PlanformDB Morphology.Id -> (database name, normalized outcome label)
MORPHOLOGIES: Dict[int, Tuple[str, str]] = {
    1: ("Wild type", "wild_type_like"),
    2: ("Double head, two pharynxes", "double_head_two_pharynxes"),
    20: ("Headless", "headless"),
    21: ("Cyclop", "cyclops"),
    24: ("Double head, one pharynx", "double_head_one_pharynx"),
    74: ("Tailless, two pharynxes", "tailless_two_pharynxes"),
    101: ("Double head, incomplete, two pharynxes", "double_head_incomplete_two_pharynxes"),
    112: ("Ectopic pharynxes, 2 posterior, ectopic brains", "ectopic_posterior_pharynxes_ectopic_brains"),
}

# PlanformDB Manipulation.Id -> (database name, normalized label)
MANIPULATIONS: Dict[int, Tuple[str, str]] = {
    2: ("Head crop", "head_crop"),
    3: ("Pre-pharyngeal crop", "pre_pharyngeal_crop"),
    4: ("Pharyngeal crop", "pharyngeal_crop"),
    7: ("Head plus pre-pharyngeal crop", "head_plus_pre_pharyngeal_crop"),
    8: ("Head plus pharyngeal crop", "head_plus_pharyngeal_crop"),
    9: ("Head plus post-pharyngeal crop", "head_plus_post_pharyngeal_crop"),
    20: ("Post-pharyngeal crop", "post_pharyngeal_crop"),
    23: ("Wild type", "uncut_wild_type"),
    33: ("Tail crop", "tail_crop"),
    55: ("Pharynx fragment with slanted anterior", "pharynx_fragment_slanted_anterior"),
    161: ("Head plus post-pharyngeal crop, VNC disruption", "head_plus_post_pharyngeal_crop_with_vnc_disruption"),
    274: ("Trunk crop", "trunk_crop"),
}

# Drug.Name -> (perturbation label, mechanism class or None for co-treatments)
DRUGS: Dict[str, Tuple[str, Optional[str]]] = {
    "Octanol": ("octanol_gap_junction_blockade", "gap_junction_block_pharmacological"),
    "Heptanol": ("heptanol_gap_junction_blockade", "gap_junction_block_pharmacological"),
    "Hexanol": ("hexanol_gap_junction_blockade", "gap_junction_block_pharmacological"),
    "SCH-28080": ("sch_28080_hk_atpase_inhibition", "hyperpolarization_hk_atpase"),
    "Ivermectin": ("ivermectin_glucl_depolarization", "depolarization_glucl_ivermectin"),
    "PZQ": ("praziquantel_vgcc_depolarization", "depolarization_vgcc"),
    "Chloride": ("chloride_cotreatment", None),
}
RNAI: Dict[str, str] = {
    "Dj-inx-12": "dj_inx_12_rnai",
    "Dj-inx-5 + 13": "dj_inx_5_13_rnai",
}

MECHANISM_CLASSES: Dict[str, str] = {
    "gap_junction_block_pharmacological": "Alcohol gap-junction blockers (octanol, heptanol, hexanol); coupling reduction label only, no conductance constants.",
    "gap_junction_rnai": "Innexin RNAi (Dj-inx-12 + Dj-inx-5/13); genetic coupling reduction kept distinct from pharmacological block.",
    "hyperpolarization_hk_atpase": "SCH-28080 H,K-ATPase inhibition; anterior hyperpolarization label (Beane 2011).",
    "depolarization_vgcc": "Praziquantel acting through voltage-operated Ca2+ channel beta subunits; depolarization label (Nogi 2009).",
    "depolarization_glucl_ivermectin": "Ivermectin-gated glutamate/chloride channel opening; depolarization label distinct from the VGCC route (Beane 2011).",
    "neural_vnc_disruption": "Ventral nerve cord disruption on top of octanol; neural-cue label kept separate from generic coupling reduction (Oviedo 2010).",
    "untreated_control": "Amputation without drug or RNAi; baseline for the series' publication.",
}

ROW_FLAGS: Dict[str, str] = {
    "placeholder_sample_count": "ResultSet.Num is 0 or 1 in the database; treat as a placeholder, no sampling interval can be derived.",
    "regen_period_zero": "ResultSet.RegenPeriod is 0 in the database; the observation time is not recorded and must come from the paper.",
    "conflicting_control_semantics": "Row conflicts with another row of the same manipulation and drug; its control/treated semantics must be confirmed against the source figure before use.",
    "drug_window_unspecified": "ExperimentDrug StartTime = EndTime = 0; the exposure window is not recorded in the database.",
    "drug_window_end_unspecified": "ExperimentDrug StartTime > 0 with EndTime = 0; the onset is recorded but the end of exposure is not (PlanformDB convention for open-ended exposure).",
    "drug_window_inconsistent_within_series": "Stored start/end days are the transpose of a sibling row in the same figure panel; the window must be read from the paper's methods before use.",
    "drug_window_pre_amputation": "ExperimentDrug StartTime < 0; exposure began before amputation.",
    "multi_drug_cotreatment": "More than one ExperimentDrug row; perturbation labels and drug_exposures list every agent, mechanism_class follows the single mechanism-bearing agent.",
    "zero_frequency_outcome_row": "At least one ResultantMorphology row is stored with Frequency = 0 (an explicitly recorded absence).",
    "control_record": "Untreated or uncut control; off the series axis, kept for baseline comparison.",
    "supplementary_record": "Same publication as the series but not part of the figure panel that defines the series axis.",
}

PLANFORMDB_ANCHOR = "source:planformdb_250::target:planformdb_curated_subset"
ANCHOR_RE = re.compile(r"^source:[a-z0-9_]+::target:[a-z0-9_]+(::[a-z0-9_]+)*$")

# AP crop position ordinals for the five classical fragments and for the
# head-retaining fragments of Oviedo 2010 Fig. 1B.
CROP_ORDINAL: Dict[int, int] = {2: 0, 3: 1, 4: 2, 20: 3, 33: 4}
HEAD_RETAINED_ORDINAL: Dict[int, int] = {2: 0, 7: 1, 8: 2, 9: 3}

# --------------------------------------------------------------------------
# Series definitions
# --------------------------------------------------------------------------
SERIES: List[Dict[str, Any]] = [
    dict(
        series_id="S1",
        label="Octanol x AP crop position",
        publication_id=1,
        figure="Fig. 1A",
        role="train",
        axis="ap_crop_position",
        experiments=[415, 416, 417, 418, 419],
        supplementary=[],
        teaching_target="gap_block_conductance",
        literature_anchors=["source:oviedo_2010_devbiol::target:gap_block_conductance"],
        primary_outcome_morphology_ids=[2],
        notes=[
            "Octanol gap-junction blockade; the only series WP4 fits the link function on.",
            "Shared with the vendored rusty-matter v0 fixture (records 415-419).",
        ],
    ),
    dict(
        series_id="S2",
        label="Triple innexin RNAi x AP crop position",
        publication_id=1,
        figure="Fig. 3D",
        role="held-out",
        axis="ap_crop_position",
        experiments=[446, 447, 448, 449, 450],
        supplementary=[445],
        teaching_target="innexin_gap_junction_label",
        literature_anchors=["source:oviedo_2010_devbiol::target:gap_block_conductance"],
        primary_outcome_morphology_ids=[2],
        notes=[
            "Dj-inx-12 + Dj-inx-5/13 RNAi; genetic coupling reduction tested as transfer, not pooled with octanol.",
            "Experiment 445 is the uncut RNAi animal (Fig. 3B, #112) and is kept as a control record off the axis.",
            "Shared with the vendored rusty-matter v0 fixture (records 446-450).",
        ],
    ),
    dict(
        series_id="S3",
        label="Heptanol x AP crop position",
        publication_id=20,
        figure="Fig. 10IB",
        role="held-out",
        axis="ap_crop_position",
        experiments=[325, 326, 327, 328, 329],
        supplementary=[322, 323, 324],
        teaching_target="gap_block_conductance",
        literature_anchors=["source:nogi_2005_devbiol::target:gap_block_conductance"],
        primary_outcome_morphology_ids=[2, 74, 101],
        notes=[
            "Heptanol pulse; posterior identity shift is the composite of #2, #74 and #101 frequencies.",
            "Experiments 322 (heptanol), 323 (untreated) and 324 (hexanol) are the Fig. 10III pharyngeal-crop comparison and are supplementary.",
            "Rows 322 and 324-328 store the drug window as start 2.0 / end 0.0 days while 329 stores 0.0 / 2.0; flagged drug_window_inconsistent_within_series, not corrected.",
        ],
    ),
    dict(
        series_id="S4",
        label="Octanol onset delay window",
        publication_id=1,
        figure="Fig. 2A",
        role="calibrate",
        axis="drug_onset_hours",
        experiments=[434, 435, 436, 437, 438, 439, 440],
        supplementary=[],
        teaching_target="ap_transient_memory",
        literature_anchors=["source:oviedo_2010_devbiol::target:ap_transient_memory"],
        primary_outcome_morphology_ids=[2],
        notes=[
            "Post-pharyngeal crop; octanol added at 0, 3, 6, 12, 24, 48 or 72 h post-amputation.",
            "Experiment 434 (0 h) is stored at 0 % double head, conflicting with experiment 418 (same crop and drug, 100 %); flagged conflicting_control_semantics and excluded from fits by default.",
            "RegenPeriod is 0 for every row; the 14-day observation comes from the paper, not the database.",
        ],
    ),
    dict(
        series_id="S5",
        label="VNC disruption timing under octanol",
        publication_id=1,
        figure="Fig. 2B",
        role="held-out",
        axis="vnc_disruption_hours",
        experiments=[441, 442, 443, 444],
        supplementary=[],
        teaching_target="gap_block_vnc_disruption_boundary",
        mechanism_class_override="neural_vnc_disruption",
        literature_anchors=["source:oviedo_2010_devbiol::target:gap_block_conductance::vnc_disruption_boundary"],
        primary_outcome_morphology_ids=[2],
        notes=[
            "Head plus post-pharyngeal crop with VNC cut at 0, 3, 12 or 24 h; neural cue class kept separate from coupling reduction.",
            "Shared with the vendored rusty-matter v0 fixture (records 441-444).",
        ],
    ),
    dict(
        series_id="S6",
        label="Head-retained fragments under octanol",
        publication_id=1,
        figure="Fig. 1B",
        role="held-out",
        axis="ap_crop_position",
        axis_ordinals=HEAD_RETAINED_ORDINAL,
        experiments=[420, 421, 422, 423],
        supplementary=[],
        teaching_target="gap_block_conductance",
        literature_anchors=["source:oviedo_2010_devbiol::target:gap_block_conductance"],
        primary_outcome_morphology_ids=[2],
        notes=[
            "Fragments keep their own head; the axis is the posterior cut position (head only, +pre-pharyngeal, +pharyngeal, +post-pharyngeal).",
            "Tests whether an existing anterior head suppresses the posterior ectopic head under the same blocker.",
        ],
    ),
    dict(
        series_id="S7",
        label="SCH-28080 x AP crop position",
        publication_id=4,
        figure="Fig. 1D",
        role="held-out (new pole)",
        axis="ap_crop_position",
        experiments=[391, 392, 393, 394, 395],
        supplementary=[],
        teaching_target="head_vs_tail_voltage",
        literature_anchors=["source:beane_2011_chembiol::target:head_vs_tail_voltage"],
        primary_outcome_morphology_ids=[20, 21],
        notes=[
            "H,K-ATPase inhibition (hyperpolarization) for 3 days; anterior loss outcomes #20 headless and #21 cyclops.",
            "Rows 391 and 392 store #20 and #21 at Frequency 0 explicitly.",
        ],
    ),
    dict(
        series_id="S8",
        label="Bidirectional voltage perturbation, pharynx fragment",
        publication_id=4,
        figure="Fig. 6B",
        role="held-out (new pole)",
        axis="voltage_perturbation",
        experiments=[411, 412, 413],
        supplementary=[],
        teaching_target="head_vs_tail_voltage",
        literature_anchors=["source:beane_2011_chembiol::target:head_vs_tail_voltage"],
        primary_outcome_morphology_ids=[24, 20],
        notes=[
            "Pharynx fragment with slanted anterior: ivermectin (depolarize) vs SCH-28080 (hyperpolarize) vs ivermectin + chloride.",
            "Mechanism classes differ per record (depolarization_glucl_ivermectin vs hyperpolarization_hk_atpase); no series-level class.",
            "Drug windows are stored as 0/0 (unspecified).",
        ],
    ),
    dict(
        series_id="S9",
        label="Praziquantel dose and exposure window",
        publication_id=25,
        figure="Fig. 1D-E",
        role="calibrate",
        axis="pzq_dose_or_window",
        experiments=list(range(516, 537)),
        supplementary=[],
        teaching_target="ap_transient_memory",
        literature_anchors=["source:nogi_2009_plosntd::target:ap_transient_memory"],
        primary_outcome_morphology_ids=[2],
        notes=[
            "Trunk crop; 516-522 are the dose series (1-110 uM, 0-2 d), 523-536 the exposure-window series.",
            "ResultSet.Num is 0 or 1 for every row except 520 (285); no sampling interval is available for this series.",
            "The dose used in the window series is not recorded in the database.",
        ],
    ),
]

# Experiments present in the queried id ranges but outside S1-S9 (ids only).
EXCLUDED_EXPERIMENTS = [
    dict(
        experiment_ids=list(range(424, 434)),
        publication_source_id="oviedo_2010_devbiol",
        reason="Oviedo 2010 Fig. 1C lateral notch, CNS and centre-removal geometries under octanol; not an AP crop-position or timing series.",
    ),
]

# Hand-reviewed row issues keyed by experiment id.
KNOWN_ROW_ISSUES: Dict[int, List[str]] = {
    # 0 h octanol onset stored at 0 % double head vs 100 % in experiment 418 (same crop, same drug).
    434: ["conflicting_control_semantics"],
    # Nogi & Levin 2005 heptanol/hexanol rows store start 2.0 / end 0.0 d while sibling 329 stores 0.0 / 2.0 d.
    **{e: ["drug_window_inconsistent_within_series"] for e in (322, 324, 325, 326, 327, 328)},
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_db_path(cli_path: Optional[str]) -> Optional[Path]:
    """Explicit --db must exist; otherwise try $PLANFORM_DB_PATH, then the default local copy."""
    if cli_path:
        p = Path(cli_path)
        if not p.is_file():
            sys.exit(f"--db {cli_path}: no such file")
        return p
    candidates: List[Path] = []
    env = os.getenv("PLANFORM_DB_PATH")
    if env:
        if Path(env).is_file():
            return Path(env)
        print(f"WARNING: PLANFORM_DB_PATH={env} is not a file; trying the default location")
    candidates.extend(DEFAULT_DB_CANDIDATES)
    for c in candidates:
        if c.is_file():
            return c
    return None


def check_db_pin(db_path: Path, strict: bool = True) -> str:
    digest = sha256_of_file(db_path)
    size = db_path.stat().st_size
    if digest == PLANFORM_DB_SHA256 and size == PLANFORM_DB_BYTES:
        print(f"PlanformDB sha256 ok ({size} bytes; planformDB_2.5.0.edb as pinned)")
        return digest
    msg = (f"{db_path}: sha256 {digest} ({size} bytes) is not the pinned planformDB_2.5.0.edb "
           f"({PLANFORM_DB_SHA256}, {PLANFORM_DB_BYTES} bytes)")
    if strict:
        sys.exit("ERROR: " + msg + " (pass --allow-unpinned-db to extract anyway)")
    print("WARNING: " + msg)
    return digest


def round_days(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 4)


def hours_label(hours: float) -> str:
    return f"{int(round(hours))} h"


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
class PlanformDB:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def one(self, sql: str, *args: Any) -> sqlite3.Row:
        row = self.conn.execute(sql, args).fetchone()
        if row is None:
            raise LookupError(f"no row for {sql!r} {args!r}")
        return row

    def all(self, sql: str, *args: Any) -> List[sqlite3.Row]:
        return self.conn.execute(sql, args).fetchall()

    def verify_vocabularies(self) -> None:
        """Fail closed if the database disagrees with the hard-coded vocabularies."""
        ver = self.one("SELECT DBStructVer FROM Meta")["DBStructVer"]
        if ver != PLANFORM_DB_STRUCT_VERSION:
            raise ValueError(f"Meta.DBStructVer {ver} != {PLANFORM_DB_STRUCT_VERSION}")
        for pid, pub in PUBLICATIONS.items():
            row = self.one("SELECT Title, URL, Year FROM Publication WHERE Id=?", pid)
            if pub["doi"] not in (row["URL"] or ""):
                raise ValueError(f"Publication {pid}: URL {row['URL']!r} does not contain DOI {pub['doi']}")
            if row["Year"] != pub["year"]:
                raise ValueError(f"Publication {pid}: Year {row['Year']} != {pub['year']}")
        for sid, (name, _) in SPECIES.items():
            if self.one("SELECT Name FROM Species WHERE Id=?", sid)["Name"] != name:
                raise ValueError(f"Species {sid} is not {name!r}")
        for mid, (name, _) in MORPHOLOGIES.items():
            got = self.one("SELECT Name FROM Morphology WHERE Id=?", mid)["Name"]
            if got != name:
                raise ValueError(f"Morphology {mid}: {got!r} != {name!r}")
        for mid, (name, _) in MANIPULATIONS.items():
            got = self.one("SELECT Name FROM Manipulation WHERE Id=?", mid)["Name"]
            if got != name:
                raise ValueError(f"Manipulation {mid}: {got!r} != {name!r}")
        for dname in DRUGS:
            self.one("SELECT Id FROM Drug WHERE Name=?", dname)
        for rname in RNAI:
            self.one("SELECT Id FROM RNAi WHERE Name=?", rname)

    def experiment(self, exp_id: int) -> Dict[str, Any]:
        e = self.one("SELECT Id, Name, Manipulation, Publication, Species FROM Experiment WHERE Id=?", exp_id)
        drugs = [
            dict(name=r["Name"], start=r["StartTime"], end=r["EndTime"])
            for r in self.all(
                "SELECT d.Name, ed.StartTime, ed.EndTime FROM ExperimentDrug ed JOIN Drug d ON d.Id=ed.Drug "
                "WHERE ed.Experiment=? ORDER BY ed.Id", exp_id)
        ]
        rnai = sorted(
            r["Name"] for r in self.all(
                "SELECT r.Name FROM ExperimentRNAi er JOIN RNAi r ON r.Id=er.RNAi WHERE er.Experiment=?", exp_id)
        )
        result_sets = []
        for rs in self.all("SELECT Id, Num, RegenPeriod FROM ResultSet WHERE Experiment=? ORDER BY Id", exp_id):
            outcomes = [
                dict(id=rm["Id"], morphology=rm["Morphology"], frequency=rm["Frequency"])
                for rm in self.all(
                    "SELECT Id, Morphology, Frequency FROM ResultantMorphology WHERE ResultSet=? ORDER BY Id", rs["Id"])
            ]
            result_sets.append(dict(id=rs["Id"], num=rs["Num"], regen=rs["RegenPeriod"], outcomes=outcomes))
        return dict(
            id=e["Id"], name=e["Name"], manipulation=e["Manipulation"], publication=e["Publication"],
            species=e["Species"], drugs=drugs, rnai=rnai, result_sets=result_sets,
        )


def parse_figure(exp_name: str) -> str:
    """'Oviedo 2010 Fig. 2B, 2 (VNC disruption at t=0.125days)' -> 'Fig. 2B, 2'."""
    m = re.search(r"(Fig\.\s*[^(]*?)\s*(?:\(|$)", exp_name)
    return m.group(1).strip().rstrip(",") if m else exp_name


def parse_vnc_days(exp_name: str) -> float:
    m = re.search(r"t\s*=\s*([0-9.]+)\s*(?:days?)?", exp_name)
    if not m:
        raise ValueError(f"cannot parse VNC disruption time from {exp_name!r}")
    return float(m.group(1))


def parse_dose_um(exp_name: str) -> Optional[int]:
    m = re.search(r"\((\d+)\s*uM\)", exp_name)
    return int(m.group(1)) if m else None


def classify_mechanism(series: Dict[str, Any], exp: Dict[str, Any]) -> str:
    if series.get("mechanism_class_override"):
        return series["mechanism_class_override"]
    classes = [DRUGS[d["name"]][1] for d in exp["drugs"] if DRUGS[d["name"]][1]]
    if exp["rnai"]:
        classes.append("gap_junction_rnai")
    classes = sorted(set(classes))
    if not classes:
        return "untreated_control"
    if len(classes) > 1:
        raise ValueError(f"experiment {exp['id']}: ambiguous mechanism classes {classes}")
    return classes[0]


def series_axis(series: Dict[str, Any], exp: Dict[str, Any], supplementary: bool) -> Optional[Dict[str, Any]]:
    if supplementary:
        return None
    axis = series["axis"]
    if axis == "ap_crop_position":
        ordinals = series.get("axis_ordinals", CROP_ORDINAL)
        if exp["manipulation"] not in ordinals:
            raise ValueError(f"experiment {exp['id']}: manipulation {exp['manipulation']} has no crop ordinal")
        return dict(name=axis, ordinal=ordinals[exp["manipulation"]], label=MANIPULATIONS[exp["manipulation"]][1])
    if axis == "drug_onset_hours":
        start_days = float(exp["drugs"][0]["start"])
        hours = round(start_days * 24.0, 2)
        return dict(name=axis, value=hours, value_days=round_days(start_days), label=hours_label(hours))
    if axis == "vnc_disruption_hours":
        days = parse_vnc_days(exp["name"])
        hours = round(days * 24.0, 2)
        return dict(name=axis, value=hours, value_days=round_days(days), label=hours_label(hours))
    if axis == "voltage_perturbation":
        labels = [DRUGS[d["name"]][0] for d in exp["drugs"]]
        return dict(name=axis, label="+".join(labels))
    if axis == "pzq_dose_or_window":
        dose = parse_dose_um(exp["name"])
        d0 = exp["drugs"][0]
        if dose is not None:
            return dict(name="pzq_dose_uM", value=dose, start_days=round_days(d0["start"]), end_days=round_days(d0["end"]))
        return dict(name="pzq_exposure_window_days", start_days=round_days(d0["start"]), end_days=round_days(d0["end"]),
                    dose_uM=None)
    raise ValueError(f"unknown axis {axis!r}")


def build_record(series: Dict[str, Any], exp: Dict[str, Any], rs: Dict[str, Any], supplementary: bool) -> Dict[str, Any]:
    pub = PUBLICATIONS[exp["publication"]]
    species_label = SPECIES[exp["species"]][1]
    manip_label = MANIPULATIONS[exp["manipulation"]][1]
    if exp["manipulation"] == 161:
        days = parse_vnc_days(exp["name"])
        manip_label += "_t" + (f"{days:g}".replace(".", "_")) + "d"

    perturbations = [DRUGS[d["name"]][0] for d in exp["drugs"]] + [RNAI[r] for r in exp["rnai"]]
    if exp["manipulation"] == 161:
        perturbations.append("vnc_disruption")

    mechanism_class = classify_mechanism(series, exp)
    flags: List[str] = []
    if (rs["num"] or 0) <= 1:
        flags.append("placeholder_sample_count")
    if float(rs["regen"] or 0.0) == 0.0:
        flags.append("regen_period_zero")
    if exp["drugs"]:
        d0 = exp["drugs"][0]
        if d0["start"] == 0 and d0["end"] == 0:
            flags.append("drug_window_unspecified")
        elif d0["start"] is not None and d0["start"] > 0 and d0["end"] == 0:
            flags.append("drug_window_end_unspecified")
        if d0["start"] is not None and d0["start"] < 0:
            flags.append("drug_window_pre_amputation")
        if len(exp["drugs"]) > 1:
            flags.append("multi_drug_cotreatment")
    if any(float(o["frequency"]) == 0.0 for o in rs["outcomes"]):
        flags.append("zero_frequency_outcome_row")
    if mechanism_class == "untreated_control" or exp["manipulation"] == 23:
        flags.append("control_record")
    elif supplementary:
        flags.append("supplementary_record")
    flags.extend(KNOWN_ROW_ISSUES.get(exp["id"], []))

    outcomes = []
    for o in rs["outcomes"]:
        if o["morphology"] not in MORPHOLOGIES:
            raise ValueError(f"experiment {exp['id']}: morphology {o['morphology']} is not in the vocabulary")
        outcomes.append(dict(
            resultant_morphology_id=o["id"],
            morphology_id=o["morphology"],
            planformdb_morphology_name=MORPHOLOGIES[o["morphology"]][0],
            normalized_outcome=MORPHOLOGIES[o["morphology"]][1],
            frequency=round(float(o["frequency"]), 4),
        ))

    notes = [
        f"Extracted by planform_bioelectric.py {SCRIPT_VERSION} from PlanformDB 2.5.0 (sha256 {PLANFORM_DB_SHA256[:12]}...).",
        "PlanformDB IDs are preserved; raw database rows remain private.",
        "Normalized labels are for link-function fitting and validation metadata only.",
        "No calibrated bioelectric physiology is inferred from this record.",
        f"Part of series {series['series_id']} ({series['label']}; {pub['source_id']} {series['figure']}).",
    ]
    if flags:
        notes.append("Row flags: " + ", ".join(flags) + ".")

    drugs_out = [dict(drug=d["name"], perturbation=DRUGS[d["name"]][0],
                      start_days=round_days(d["start"]), end_days=round_days(d["end"])) for d in exp["drugs"]]
    d0 = exp["drugs"][0] if exp["drugs"] else None

    return dict(
        record_id=f"planformdb:experiment:{exp['id']}:resultset:{rs['id']}",
        evidence_type="derived_planformdb_record",
        source_citation_ids=["planformdb_250", LOBO_CITATION["id"], pub["source_id"]],
        source_ids=dict(
            publication_id=exp["publication"],
            species_id=exp["species"],
            experiment_id=exp["id"],
            manipulation_id=exp["manipulation"],
            result_set_id=rs["id"],
        ),
        normalized_labels=dict(
            publication_source_id=pub["source_id"],
            species=species_label,
            manipulation=manip_label,
            perturbations=perturbations,
            teaching_target=series["teaching_target"],
        ),
        assay_context=dict(
            sample_count=int(rs["num"] or 0),
            regeneration_period_days=round_days(rs["regen"] or 0.0),
        ),
        resultant_morphologies=outcomes,
        transform_notes=notes,
        # ---- infinitemorphospace extension -------------------------------
        series_id=series["series_id"],
        source_figure=parse_figure(exp["name"]),
        mechanism_class=mechanism_class,
        drug_start_days=round_days(d0["start"]) if d0 else None,
        drug_end_days=round_days(d0["end"]) if d0 else None,
        drug_exposures=drugs_out,
        rnai=[RNAI[r] for r in exp["rnai"]],
        series_axis=series_axis(series, exp, supplementary),
        literature_anchors=list(series["literature_anchors"]) + [PLANFORMDB_ANCHOR],
        row_flags=flags,
    )


def extract(db_path: Path, db_sha256: str) -> Dict[str, Any]:
    db = PlanformDB(db_path)
    try:
        db.verify_vocabularies()
        records: List[Dict[str, Any]] = []
        series_out: List[Dict[str, Any]] = []
        for s in SERIES:
            series_records: List[Dict[str, Any]] = []
            for exp_id in list(s["experiments"]) + list(s["supplementary"]):
                exp = db.experiment(exp_id)
                if exp["publication"] != s["publication_id"]:
                    raise ValueError(f"experiment {exp_id} belongs to publication {exp['publication']}, "
                                     f"series {s['series_id']} expects {s['publication_id']}")
                if not exp["result_sets"]:
                    raise ValueError(f"experiment {exp_id} has no result set")
                for rs in exp["result_sets"]:
                    series_records.append(build_record(s, exp, rs, supplementary=exp_id in s["supplementary"]))
            on_axis = [r for r in series_records if r["series_axis"] is not None]
            off_axis = [r for r in series_records if r["series_axis"] is None]
            mech = sorted({r["mechanism_class"] for r in on_axis})
            series_out.append(dict(
                series_id=s["series_id"],
                label=s["label"],
                publication_source_id=PUBLICATIONS[s["publication_id"]]["source_id"],
                publication_id=s["publication_id"],
                source_figure=s["figure"],
                role=s["role"],
                axis=s["axis"],
                mechanism_class=mech[0] if len(mech) == 1 else None,
                mechanism_classes=mech,
                teaching_target=s["teaching_target"],
                literature_anchors=list(s["literature_anchors"]) + [PLANFORMDB_ANCHOR],
                primary_outcome_morphology_ids=list(s["primary_outcome_morphology_ids"]),
                experiment_ids=list(s["experiments"]),
                supplementary_experiment_ids=list(s["supplementary"]),
                record_ids=[r["record_id"] for r in on_axis] + [r["record_id"] for r in off_axis],
                n_specimens_on_axis=sum(r["assay_context"]["sample_count"] for r in on_axis),
                n_specimens_supplementary=sum(r["assay_context"]["sample_count"] for r in off_axis),
                notes=list(s["notes"]),
            ))
            records.extend(on_axis + off_axis)
    finally:
        db.close()

    citations = [
        dict(id="planformdb_250", label="Lobo Lab PlanformDB 2.5.0", url="https://lobolab.umbc.edu/planform/download/"),
        LOBO_CITATION,
    ] + [
        dict(id=p["source_id"], label=p["label"], doi=p["doi"], url=f"https://doi.org/{p['doi']}")
        for p in PUBLICATIONS.values()
    ]

    return dict(
        schema_id=SCHEMA_ID,
        schema_extension_id=SCHEMA_EXTENSION_ID,
        fixture_id=FIXTURE_ID,
        schema_version=1,
        evidence_type="derived_planformdb_record",
        scope=("PlanformDB 2.5.0-derived bioelectric experiment series S1-S9 for fitting and testing the "
               "pole-identity link of 2021-Cervera-Levin-Mafe.py (infinitemorphospace calibration, not "
               "rusty-matter runtime dynamics)."),
        source_database=dict(
            id="planformdb_250",
            version="2.5.0",
            source_url="https://lobolab.umbc.edu/planform/download/",
            raw_sha256=PLANFORM_DB_SHA256.upper(),
            raw_bytes=PLANFORM_DB_BYTES,
            sqlite_schema_version=PLANFORM_DB_STRUCT_VERSION,
            queried_tables=["Experiment", "ExperimentDrug", "Drug", "ExperimentRNAi", "RNAi", "ResultSet",
                            "ResultantMorphology", "Morphology", "Manipulation", "Publication", "Species", "Meta"],
        ),
        notice_text=NOTICE_TEXT,
        citations=citations,
        selection_policy=dict(
            selected_on=SELECTED_ON,
            selection_basis=("Every PlanformDB 2.5.0 experiment series from Oviedo 2010, Nogi & Levin 2005, Beane 2011 and "
                             "Nogi 2009 that perturbs planarian AP polarity through gap-junction coupling, membrane "
                             "voltage or neural cues and reports outcome frequencies against a crop-position, timing "
                             "or dose axis."),
            fit_policy=("Link parameters may be fitted on S1 only; S2, S3, S6 test gap-junction transfer, S7-S8 the "
                        "hyperpolarization pole, S4 and S9 calibrate the time scale, S5 tests it. Rows carrying "
                        "conflicting_control_semantics or placeholder_sample_count are excluded from fits by default."),
            non_scope=[
                "calibrated physiology",
                "Matter runtime dynamics",
                "PlanformDB row dump",
                "region geometry or organ tables",
                "paper figure redistribution",
                "morphology image redistribution",
            ],
        ),
        extraction=dict(
            script="planform_bioelectric.py",
            script_version=SCRIPT_VERSION,
            db_sha256_verified=(db_sha256 == PLANFORM_DB_SHA256),
            record_count=len(records),
            series_count=len(series_out),
        ),
        vocabularies=dict(
            mechanism_class=MECHANISM_CLASSES,
            row_flags=ROW_FLAGS,
            morphologies={str(k): dict(planformdb_name=v[0], normalized_outcome=v[1]) for k, v in MORPHOLOGIES.items()},
            manipulations={str(k): dict(planformdb_name=v[0], normalized_label=v[1]) for k, v in MANIPULATIONS.items()},
            perturbations={DRUGS[d][0]: dict(planformdb_drug=d, mechanism_class=DRUGS[d][1]) for d in DRUGS}
            | {RNAI[r]: dict(planformdb_rnai=r, mechanism_class="gap_junction_rnai") for r in RNAI}
            | {"vnc_disruption": dict(planformdb_manipulation=161, mechanism_class="neural_vnc_disruption")},
        ),
        excluded_experiments=EXCLUDED_EXPERIMENTS,
        series=series_out,
        records=records,
    )


# --------------------------------------------------------------------------
# Validation (fails closed; used on the fixture and on the damaged negatives)
# --------------------------------------------------------------------------
FREQ_SUM_TOL = 0.015  # frequencies are stored to two decimals


def validate_fixture(doc: Dict[str, Any]) -> List[str]:
    errs: List[str] = []

    def err(code: str, msg: str) -> None:
        errs.append(f"{code}: {msg}")

    if doc.get("schema_id") != SCHEMA_ID:
        err("E_SCHEMA_ID", f"schema_id {doc.get('schema_id')!r} != {SCHEMA_ID!r}")
    if doc.get("fixture_id") != FIXTURE_ID:
        err("E_FIXTURE_ID", f"fixture_id {doc.get('fixture_id')!r} != {FIXTURE_ID!r}")
    sdb = doc.get("source_database") or {}
    if str(sdb.get("raw_sha256", "")).lower() != PLANFORM_DB_SHA256:
        err("E_SHA_MISMATCH", f"source_database.raw_sha256 {sdb.get('raw_sha256')!r} is not the pinned digest")

    notice = doc.get("notice_text")
    if not isinstance(notice, str) or not notice.strip():
        err("E_NOTICE_MISSING", "notice_text is missing or empty")
    else:
        for clause in NOTICE_REQUIRED_CLAUSES:
            if clause not in notice:
                err("E_NOTICE_MISSING", f"notice_text lacks required clause {clause!r}")

    series = doc.get("series")
    records = doc.get("records")
    if not isinstance(series, list) or not series:
        err("E_SERIES_MISSING", "series list is missing or empty")
        series = []
    if not isinstance(records, list) or not records:
        err("E_RECORDS_MISSING", "records list is missing or empty")
        records = []

    series_ids = {s.get("series_id") for s in series}
    record_ids = [r.get("record_id") for r in records]
    if len(set(record_ids)) != len(record_ids):
        err("E_DUPLICATE_RECORD", "duplicate record_id")
    record_id_set = set(record_ids)

    for s in series:
        sid = s.get("series_id")
        for rid in s.get("record_ids", []):
            if rid not in record_id_set:
                err("E_SERIES_RECORD_MISSING", f"{sid}: record {rid} not in records")
        for a in s.get("literature_anchors", []):
            if not ANCHOR_RE.match(a):
                err("E_BAD_ANCHOR", f"{sid}: anchor {a!r}")
        for mc in s.get("mechanism_classes", []):
            if mc not in MECHANISM_CLASSES:
                err("E_UNKNOWN_MECHANISM_CLASS", f"{sid}: {mc!r}")

    required_record_keys = (
        "record_id", "evidence_type", "source_citation_ids", "source_ids", "normalized_labels", "assay_context",
        "resultant_morphologies", "transform_notes", "series_id", "mechanism_class", "drug_start_days",
        "drug_end_days", "literature_anchors", "row_flags",
    )
    for r in records:
        rid = r.get("record_id", "<no id>")
        for k in required_record_keys:
            if k not in r:
                err("E_RECORD_KEY", f"{rid}: missing {k}")
        if r.get("evidence_type") != "derived_planformdb_record":
            err("E_EVIDENCE_TYPE", f"{rid}: {r.get('evidence_type')!r}")
        if r.get("series_id") not in series_ids:
            err("E_UNKNOWN_SERIES", f"{rid}: series {r.get('series_id')!r}")
        if r.get("mechanism_class") not in MECHANISM_CLASSES:
            err("E_UNKNOWN_MECHANISM_CLASS", f"{rid}: {r.get('mechanism_class')!r}")
        for f in r.get("row_flags", []):
            if f not in ROW_FLAGS:
                err("E_UNKNOWN_ROW_FLAG", f"{rid}: {f!r}")
        for a in r.get("literature_anchors", []):
            if not ANCHOR_RE.match(a):
                err("E_BAD_ANCHOR", f"{rid}: anchor {a!r}")
        if PLANFORMDB_ANCHOR not in r.get("literature_anchors", []):
            err("E_BAD_ANCHOR", f"{rid}: missing {PLANFORMDB_ANCHOR}")
        sids = r.get("source_ids") or {}
        for k in ("publication_id", "species_id", "experiment_id", "manipulation_id", "result_set_id"):
            if not isinstance(sids.get(k), int):
                err("E_SOURCE_ID", f"{rid}: source_ids.{k}")
        exp_id = sids.get("experiment_id")
        rs_id = sids.get("result_set_id")
        if isinstance(exp_id, int) and isinstance(rs_id, int) and rid != f"planformdb:experiment:{exp_id}:resultset:{rs_id}":
            err("E_RECORD_ID", f"{rid}: does not match source_ids")
        ac = r.get("assay_context") or {}
        if not isinstance(ac.get("sample_count"), int) or ac["sample_count"] < 0:
            err("E_SAMPLE_COUNT", f"{rid}: sample_count {ac.get('sample_count')!r}")
        if ac.get("sample_count", 2) <= 1 and "placeholder_sample_count" not in r.get("row_flags", []):
            err("E_FLAG_MISSING", f"{rid}: sample_count <= 1 without placeholder_sample_count flag")
        outcomes = r.get("resultant_morphologies") or []
        if not outcomes:
            err("E_NO_OUTCOMES", f"{rid}: no resultant morphologies")
        total = 0.0
        for o in outcomes:
            mid = o.get("morphology_id")
            if mid not in MORPHOLOGIES:
                err("E_UNKNOWN_MORPHOLOGY", f"{rid}: morphology_id {mid!r}")
            elif o.get("normalized_outcome") != MORPHOLOGIES[mid][1]:
                err("E_OUTCOME_LABEL", f"{rid}: morphology {mid} labelled {o.get('normalized_outcome')!r}")
            f = o.get("frequency")
            if not isinstance(f, (int, float)) or not (0.0 <= float(f) <= 1.0):
                err("E_FREQ_RANGE", f"{rid}: frequency {f!r}")
            else:
                total += float(f)
        if outcomes and abs(total - 1.0) > FREQ_SUM_TOL:
            err("E_FREQ_SUM", f"{rid}: frequencies sum to {total:.3f}")
    return errs


def error_codes(errs: Iterable[str]) -> List[str]:
    return sorted({e.split(":", 1)[0] for e in errs})


# --------------------------------------------------------------------------
# Damaged negatives
# --------------------------------------------------------------------------
def make_damaged(doc: Dict[str, Any]) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """Return (file stem, injected fault, expected error code, damaged document)."""
    out = []

    d = copy.deepcopy(doc)
    rec = d["records"][0]
    o = min(rec["resultant_morphologies"], key=lambda x: x["frequency"])
    o["frequency"] = round(o["frequency"] + 0.3, 4)
    out.append(("bad_frequency_sum", f"{rec['record_id']}: morphology {o['morphology_id']} frequency +0.30 (sum 1.30)", "E_FREQ_SUM", d))

    d = copy.deepcopy(doc)
    d["notice_text"] = d["notice_text"].replace("3. This notice may not be removed or altered from any distribution.\n\n", "")
    out.append(("missing_notice", "Lobo Lab restriction 3 removed from notice_text", "E_NOTICE_MISSING", d))

    d = copy.deepcopy(doc)
    rec = d["records"][0]
    rec["resultant_morphologies"][0]["morphology_id"] = 999
    rec["resultant_morphologies"][0]["normalized_outcome"] = "unknown_morphology"
    out.append(("unknown_morphology_id", f"{rec['record_id']}: morphology_id 999", "E_UNKNOWN_MORPHOLOGY", d))

    d = copy.deepcopy(doc)
    d["source_database"]["raw_sha256"] = "0" * 64
    out.append(("sha_mismatch", "source_database.raw_sha256 zeroed", "E_SHA_MISMATCH", d))

    d = copy.deepcopy(doc)
    rec = d["records"][0]
    rec["mechanism_class"] = "octanol_and_rnai_pooled"
    out.append(("unknown_mechanism_class", f"{rec['record_id']}: mechanism_class pooled label", "E_UNKNOWN_MECHANISM_CLASS", d))

    return out


def write_damaged(doc: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        schema_id=SCHEMA_EXTENSION_ID + ".damaged_negatives",
        purpose=("Negative fixtures for a fail-closed self-test: each file is the full fixture with exactly one "
                 "injected fault. A validator that accepts any of them is broken."),
        notice="Each file carries the Lobo Lab notice unchanged except missing_notice.json, whose fault is the removal of restriction 3.",
        source_fixture=DEFAULT_OUT.name,
        files=[],
    )
    for stem, fault, code, d in make_damaged(doc):
        p = out_dir / f"{FIXTURE_ID}.{stem}.json"
        p.write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        got = error_codes(validate_fixture(d))
        if code not in got:
            raise RuntimeError(f"damaged negative {stem} does not trigger {code} (got {got})")
        manifest["files"].append(dict(file=p.name, injected_fault=fault, expected_error_code=code,
                                      sha256=sha256_of_file(p)))
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return mpath


# --------------------------------------------------------------------------
# Library API for WP2+
# --------------------------------------------------------------------------
def dumps_fixture(doc: Dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def load_fixture(path: Path = DEFAULT_OUT, strict: bool = True) -> Dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    errs = validate_fixture(doc)
    if errs and strict:
        raise ValueError(f"{path}: {len(errs)} validation errors, first: {errs[0]}")
    return doc


def series_records(doc: Dict[str, Any], series_id: str, on_axis_only: bool = True,
                   exclude_flags: Sequence[str] = ("conflicting_control_semantics",)) -> List[Dict[str, Any]]:
    """Records of one series in axis order, optionally dropping flagged rows."""
    recs = [r for r in doc["records"] if r["series_id"] == series_id]
    if on_axis_only:
        recs = [r for r in recs if r["series_axis"] is not None]
    recs = [r for r in recs if not set(r["row_flags"]) & set(exclude_flags)]
    return recs


def outcome_table(doc: Dict[str, Any], series_id: str, morphology_ids: Optional[Sequence[int]] = None,
                  **kw: Any) -> List[Dict[str, Any]]:
    """Per-record axis value, n, and summed frequency of the requested morphologies
    (default: the series' primary outcomes)."""
    s = next(x for x in doc["series"] if x["series_id"] == series_id)
    mids = set(morphology_ids or s["primary_outcome_morphology_ids"])
    rows = []
    for r in series_records(doc, series_id, **kw):
        freq = sum(o["frequency"] for o in r["resultant_morphologies"] if o["morphology_id"] in mids)
        rows.append(dict(record_id=r["record_id"], experiment_id=r["source_ids"]["experiment_id"],
                         axis=r["series_axis"], n=r["assay_context"]["sample_count"],
                         mechanism_class=r["mechanism_class"], frequency=round(freq, 4), row_flags=r["row_flags"]))
    return rows


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
# Census expectations (PlanformDB 2.5.0 Frequency x 100 as read on 2026-10-18).
CENSUS = {
    "S1": dict(mids=[2], freq=[0.05, 0.28, 0.5, 1.0, 0.0], n=688),
    "S2": dict(mids=[2], freq=[0.0, 0.0, 0.2, 0.8, 0.0], n=99),
    "S3": dict(mids=[2, 74, 101], freq=[0.17, 0.52, 0.68, 0.19, 0.0], n=379),
    "S4": dict(mids=[2], freq=[0.85, 0.9, 0.58, 0.1, 0.1, 0.1], n=60),  # 434 excluded by flag
    "S5": dict(mids=[2], freq=[0.18, 0.75, 0.2, 0.0], n=40),
    "S6": dict(mids=[2], freq=[0.04, 0.0, 0.0, 0.05], n=503),
    "S7": dict(mids=[20], freq=[0.0, 0.0, 0.67, 0.16, 0.4], n=650),
    "S8": dict(mids=[24, 20], freq=[0.92, 1.0, 0.37], n=30),
}


def self_test(fixture_path: Path, damaged_dir: Path, db_cli: Optional[str]) -> int:
    failures = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal failures
        print(("  ok   " if ok else "  FAIL ") + msg)
        if not ok:
            failures += 1

    print(f"[planform_bioelectric] self-test on {fixture_path.relative_to(REPO_ROOT)}")
    doc = json.loads(fixture_path.read_text(encoding="utf-8"))
    errs = validate_fixture(doc)
    check(not errs, f"validator: {len(errs)} errors" + (f" (first: {errs[0]})" if errs else ""))
    check(len(doc["series"]) == 9, f"9 series present ({len(doc['series'])})")
    check(doc["notice_text"] == NOTICE_TEXT, "Lobo Lab notice text is byte-identical to the script constant")

    # Parity with the vendored rusty-matter v0 fixture on the 14 shared records.
    if RUSTY_MATTER_V0.is_file():
        v0 = json.loads(RUSTY_MATTER_V0.read_text(encoding="utf-8"))
        ours = {r["record_id"]: r for r in doc["records"]}
        mism = []
        for r0 in v0["records"]:
            r1 = ours.get(r0["record_id"])
            if r1 is None:
                mism.append(f"{r0['record_id']} missing")
                continue
            if r1["source_ids"] != r0["source_ids"]:
                mism.append(f"{r0['record_id']} source_ids")
            if r1["assay_context"] != r0["assay_context"]:
                mism.append(f"{r0['record_id']} assay_context")
            if r1["normalized_labels"]["manipulation"] != r0["normalized_labels"]["manipulation"]:
                mism.append(f"{r0['record_id']} manipulation label")
            if sorted(r1["normalized_labels"]["perturbations"]) != sorted(r0["normalized_labels"]["perturbations"]):
                mism.append(f"{r0['record_id']} perturbations")
            o1 = sorted((o["resultant_morphology_id"], o["morphology_id"], o["normalized_outcome"], o["frequency"])
                        for o in r1["resultant_morphologies"])
            o0 = sorted((o["resultant_morphology_id"], o["morphology_id"], o["normalized_outcome"], o["frequency"])
                        for o in r0["resultant_morphologies"])
            if o1 != o0:
                mism.append(f"{r0['record_id']} outcomes")
        check(not mism, f"rusty-matter v0 parity on {len(v0['records'])} shared records" + (f": {mism[:3]}" if mism else ""))
        marker = "Planform and PlanformDB are provided as-is"
        check(marker in v0["notice_text"] and v0["notice_text"][v0["notice_text"].index(marker):] == NOTICE_TEXT[NOTICE_TEXT.index(marker):],
              "Lobo Lab notice clauses identical to the vendored rusty-matter fixture")
    else:
        print("  skip rusty-matter v0 parity (vendored fixture not found)")

    # Census expectations.
    for sid, exp in CENSUS.items():
        rows = outcome_table(doc, sid, exp["mids"])
        got = [r["frequency"] for r in rows]
        n = sum(r["n"] for r in rows)
        check(got == exp["freq"] and n == exp["n"], f"{sid}: {exp['mids']} -> {got}, n={n}")
    s9 = outcome_table(doc, "S9", exclude_flags=())
    dose = [r["frequency"] for r in s9 if r["axis"]["name"] == "pzq_dose_uM"]
    check(dose == [0.0, 0.1, 0.55, 0.75, 0.87, 0.95, 1.0], f"S9 dose series #2 -> {dose}")
    win = {(r["axis"]["start_days"], r["axis"]["end_days"]): r["frequency"] for r in s9 if r["axis"]["name"] == "pzq_exposure_window_days"}
    check(win.get((0.0, 0.25)) == 0.2 and win.get((0.0, 2.0)) == 0.95 and win.get((0.75, 2.75)) == 0.0,
          f"S9 window series 0-6h {win.get((0.0, 0.25))}, 0-48h {win.get((0.0, 2.0))}, 18h-66h {win.get((0.75, 2.75))}")
    check(all("placeholder_sample_count" in r["row_flags"] for r in s9 if r["experiment_id"] != 520),
          "S9 placeholder sample counts flagged on every row except 520")
    r434 = next(r for r in doc["records"] if r["source_ids"]["experiment_id"] == 434)
    check("conflicting_control_semantics" in r434["row_flags"] and "regen_period_zero" in r434["row_flags"],
          "S4 experiment 434 flagged (conflicting control, regen period 0)")
    check(len(series_records(doc, "S4")) == 6, "S4 default view excludes the flagged 434 row")
    s8 = {r["source_ids"]["experiment_id"]: r["mechanism_class"] for r in doc["records"] if r["series_id"] == "S8"}
    check(s8 == {411: "depolarization_glucl_ivermectin", 412: "hyperpolarization_hk_atpase", 413: "depolarization_glucl_ivermectin"},
          f"S8 per-record mechanism classes {s8}")
    check(all(r["mechanism_class"] == "neural_vnc_disruption" for r in doc["records"] if r["series_id"] == "S5"),
          "S5 records carry neural_vnc_disruption, not the octanol class")
    s3_inc = sorted(r["source_ids"]["experiment_id"] for r in doc["records"] if "drug_window_inconsistent_within_series" in r["row_flags"])
    check(s3_inc == [322, 324, 325, 326, 327, 328], f"S3 inconsistent drug windows flagged {s3_inc}")
    s4_open = sorted(r["source_ids"]["experiment_id"] for r in doc["records"]
                     if r["series_id"] == "S4" and "drug_window_end_unspecified" in r["row_flags"])
    check(s4_open == [435, 436, 437, 438, 439, 440], f"S4 open-ended onset windows flagged {s4_open}")
    r523 = next(r for r in doc["records"] if r["source_ids"]["experiment_id"] == 523)
    check("drug_window_pre_amputation" in r523["row_flags"] and r523["drug_start_days"] == -2.0,
          "S9 experiment 523 pre-amputation exposure flagged")
    r413 = next(r for r in doc["records"] if r["source_ids"]["experiment_id"] == 413)
    check("multi_drug_cotreatment" in r413["row_flags"] and "chloride_cotreatment" in r413["normalized_labels"]["perturbations"],
          "S8 experiment 413 chloride co-treatment flagged and labelled")

    # Damaged negatives.
    mpath = damaged_dir / "manifest.json"
    if mpath.is_file():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        for item in manifest["files"]:
            p = damaged_dir / item["file"]
            if not p.is_file():
                check(False, f"damaged/{item['file']} missing")
                continue
            codes = error_codes(validate_fixture(json.loads(p.read_text(encoding="utf-8"))))
            check(item["expected_error_code"] in codes, f"damaged/{item['file']} -> {codes}")
            check(sha256_of_file(p) == item["sha256"], f"damaged/{item['file']} sha256 matches manifest")
    else:
        check(False, f"{mpath} missing")

    # Regeneration against the pinned database, if available.
    db_path = resolve_db_path(db_cli)
    if db_path is None:
        print("  skip regeneration (PlanformDB not found; set PLANFORM_DB_PATH)")
    else:
        digest = sha256_of_file(db_path)
        if digest != PLANFORM_DB_SHA256:
            print(f"  skip regeneration ({db_path} is not the pinned database)")
        else:
            regenerated = dumps_fixture(extract(db_path, digest))
            check(regenerated == fixture_path.read_text(encoding="utf-8"),
                  "regenerated fixture is byte-identical to the file on disk")

    print(f"[planform_bioelectric] self-test {'PASSED' if failures == 0 else f'FAILED ({failures})'}")
    return 1 if failures else 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="path to planformDB_2.5.0.edb (default: $PLANFORM_DB_PATH or the local Levin/PlanformDB copy)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"fixture output path (default {DEFAULT_OUT.relative_to(REPO_ROOT)})")
    ap.add_argument("--damaged", type=Path, default=DEFAULT_DAMAGED_DIR, help="directory for damaged negatives")
    ap.add_argument("--no-damaged", action="store_true", help="do not write damaged negatives")
    ap.add_argument("--allow-unpinned-db", action="store_true", help="extract even if the database SHA-256 differs from the pin")
    ap.add_argument("--validate", type=Path, metavar="FIXTURE", help="validate an existing fixture and exit")
    ap.add_argument("--self-test", action="store_true", help="run the self-test against the fixture on disk")
    args = ap.parse_args(argv)

    if args.validate:
        doc = json.loads(args.validate.read_text(encoding="utf-8"))
        errs = validate_fixture(doc)
        for e in errs:
            print(e)
        print(f"{args.validate}: {'valid' if not errs else f'{len(errs)} errors'} "
              f"({len(doc.get('records', []))} records, {len(doc.get('series', []))} series)")
        return 1 if errs else 0

    if args.self_test:
        return self_test(args.out, args.damaged, args.db)

    db_path = resolve_db_path(args.db)
    if db_path is None:
        sys.exit("PlanformDB not found. Pass --db, set PLANFORM_DB_PATH, or place planformDB_2.5.0.edb at "
                 + str(DEFAULT_DB_CANDIDATES[0]))
    digest = check_db_pin(db_path, strict=not args.allow_unpinned_db)
    doc = extract(db_path, digest)
    errs = validate_fixture(doc)
    if errs:
        for e in errs:
            print(e)
        sys.exit(f"extraction produced an invalid fixture ({len(errs)} errors); nothing written")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps_fixture(doc), encoding="utf-8")
    n_on = sum(1 for r in doc["records"] if r["series_axis"] is not None)
    print(f"wrote {args.out.relative_to(REPO_ROOT) if args.out.is_relative_to(REPO_ROOT) else args.out}: "
          f"{len(doc['records'])} records ({n_on} on-axis) in {len(doc['series'])} series, "
          f"sha256 {sha256_of_file(args.out)[:16]}...")
    for s in doc["series"]:
        print(f"  {s['series_id']}  {s['label']:<48} {s['publication_source_id']:<22} {s['source_figure']:<10} "
              f"n={s['n_specimens_on_axis']:<4} +{s['n_specimens_supplementary']:<4} role={s['role']}")
    if not args.no_damaged:
        mpath = write_damaged(doc, args.damaged)
        print(f"wrote {len(json.loads(mpath.read_text())['files'])} damaged negatives + manifest under "
              f"{args.damaged.relative_to(REPO_ROOT) if args.damaged.is_relative_to(REPO_ROOT) else args.damaged}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
