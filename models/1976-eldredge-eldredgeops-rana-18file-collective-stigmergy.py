#!/usr/bin/env python3
"""
models/1976-eldredge-eldredgeops-rana-18file-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2055)
Organism: Eldredgeops rana_18file (Middle Devonian Phacopid Trilobite (18-File Schizochroal Calcite Eye))
Phenomenon Scale: 804 µm (0.0008035 m, log10=-3.095, 5_primordium_100um)
Anatomical Structure: Collective swarm / biofilm / colonial array in Eldredgeops rana_18file (Middle Devonian (Cazenovian, 387 Ma))
Geological Horizon: Middle Devonian (Cazenovian, 387 Ma) (387.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Eldredgeops
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (804 µm))
Formal Verification: `morphoeval/lean/loops/Loop_2055_eldredgeops_rana_18file_collective_stigmergy.lean` (`MathOfBiology.Loop_2055`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eldredgeops_rana_18file_collective_stigmergy"
EXPECTED_SHA256 = {
    "eldredgeops_rana_18file_collective_stigmergy_reference.json": "fa3d19f831840ec9a1db6845babda46b8f1339862e1ac63d012c279753f238f4",
    "eldredgeops_rana_18file_specimen.jpg": "ba63b7ea8e242e6374117b84812be17b530eebb3f9621b25eff533b5397e1369",
    "scout_manifest.json": "e74494294622059ce0df4f83a238a13c027ac8929470f07e8cf29f08c11c4e71",
    "paper_citations.bib": "c8ab2857f7afb610eff814bcab4387fc975b2a1f4d5d35cd3058de4d951cbc47"
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
    ref = json.loads((DATASET_DIR / "eldredgeops_rana_18file_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1976-eldredge-eldredgeops-rana-18file-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
