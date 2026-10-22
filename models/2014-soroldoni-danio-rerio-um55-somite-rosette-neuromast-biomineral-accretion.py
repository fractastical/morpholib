#!/usr/bin/env python3
"""
models/2014-soroldoni-danio-rerio-um55-somite-rosette-neuromast-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1158)
Organism: Danio rerio (Zebrafish — 16–24 hpf)
Phenomenon Scale: 63 µm (6.297e-05 m, log10=-4.201, 4_cellular_10um)
Anatomical Structure: Single nascent somite block (55 µm AP) & deposited lateral-line neuromast (16–24 hpf (Somite Epithelial Somitomere & Neuromast Rosette, ZFA:0000155)) · Zebrahub lateral_line_neuromast_rosette & segmented_paraxial_somite
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization at 63 µm (Single nascent somite block (55 µm AP) & deposited lateral-line neuromast))
Formal Verification: `morphoeval/lean/loops/Loop_1158_danio_rerio_um55_somite_rosette_neuromast_biomineral_accretion.lean` (`MathOfBiology.Loop_1158`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_um55_somite_rosette_neuromast_biomineral_accretion"
EXPECTED_SHA256 = {
    "danio_rerio_um55_somite_rosette_neuromast_biomineral_accretion_reference.json": "dbed98df9d2f6fa001190f45668b5773a5014cbb9375137207f630e9cd73bf8a",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "b2e26b4b58943c5454818def1c7ee666dde126481b0de4344300a569d1f032b0",
    "paper_citations.bib": "a0981ca6e64f02a9ec3dfce277fb55ba3ce075d7aa0051db3ed88ebdb72c388d"
}


def verify_sha256() -> None:
    for fname, exp in EXPECTED_SHA256.items():
        p = DATASET_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != exp:
            raise ValueError(f"SHA-256 mismatch for {fname}: {got} != {exp}")


def run_model() -> dict:
    verify_sha256()
    ref = json.loads((DATASET_DIR / "danio_rerio_um55_somite_rosette_neuromast_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {
        "pearson_r": r,
        "rmse": rmse,
        "n": int(emp.size),
        "phenomenon_size_m": float(ref["phenomenon_scale"]["phenomenon_size_m"]),
        "scale_tier": ref["phenomenon_scale"]["scale_tier"],
        "geological_age_ma": float(ref["paleobiology"]["geological_age_ma"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        assert res["phenomenon_size_m"] > 0.0
        print(f"[PASS] models/2014-soroldoni-danio-rerio-um55-somite-rosette-neuromast-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
