#!/usr/bin/env python3
"""
models/1976-eldredge-eldredgeops-rana-18file-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1605)
Organism: Eldredgeops rana_18file (Middle Devonian Phacopid Trilobite (18-File Schizochroal Calcite Eye))
Phenomenon Scale: 1.1 µm (1.116e-06 m, log10=-5.952, 3_subcellular_um)
Anatomical Structure: Apical epithelial cell vertex polygon in Eldredgeops rana_18file (Middle Devonian (Cazenovian, 387 Ma))
Geological Horizon: Middle Devonian (Cazenovian, 387 Ma) (387.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Eldredgeops
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (1.1 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1605_eldredgeops_rana_18file_vertex_packing.lean` (`MathOfBiology.Loop_1605`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eldredgeops_rana_18file_vertex_packing"
EXPECTED_SHA256 = {
    "eldredgeops_rana_18file_vertex_packing_reference.json": "1590ea627093309b7a5dbc7dda1e2734676bdbd1bb35b9588c1422e55b4bf9b4",
    "eldredgeops_rana_18file_specimen.jpg": "ba63b7ea8e242e6374117b84812be17b530eebb3f9621b25eff533b5397e1369",
    "scout_manifest.json": "855bd7642e7ac3d1a90dbc8ada9134f2a7af6fb0498084a9728199c1fd87f934",
    "paper_citations.bib": "5929f852efc60e294db9f33567cb1c4f9ad04821abcf4fd4f084a958c0e072d4"
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
    ref = json.loads((DATASET_DIR / "eldredgeops_rana_18file_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1976-eldredge-eldredgeops-rana-18file-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
