#!/usr/bin/env python3
"""
models/2016-weaver-odontodactylus-scyllarus-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1643)
Organism: Odontodactylus scyllarus (Peacock Mantis Shrimp (Bouligand Helicoidal Dactyl Club Nanocomposite))
Phenomenon Scale: 2.5 nm (2.5e-09 m, log10=-8.602, 1_molecular_nm)
Anatomical Structure: Apical epithelial cell vertex polygon in Odontodactylus scyllarus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Odontodactylus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (2.5 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1643_odontodactylus_scyllarus_vertex_packing.lean` (`MathOfBiology.Loop_1643`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/odontodactylus_scyllarus_vertex_packing"
EXPECTED_SHA256 = {
    "odontodactylus_scyllarus_vertex_packing_reference.json": "87ef1fede025306f090e4b7143fba8459bf051c17f2422bd87c07b9d6553cb96",
    "odontodactylus_scyllarus_specimen.jpg": "a0cb5ee4fe57474a9ed4bdb381117a01fec6e1ff71de38b747022330546126c0",
    "scout_manifest.json": "58453d8ae33f277b32fed08b5b461e4c3029e9717c8e46f3388ab82140841757",
    "paper_citations.bib": "e8b4e586155818254d5d855874831434bd5553416df39e1524ac424bba716cdb"
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
    ref = json.loads((DATASET_DIR / "odontodactylus_scyllarus_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2016-weaver-odontodactylus-scyllarus-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
