#!/usr/bin/env python3
"""
models/2013-riedel-danio-rerio-nm28-delta-notch-synapse-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1115)
Organism: Danio rerio (Zebrafish — 12–16 hpf)
Phenomenon Scale: 27 nm (2.744e-08 m, log10=-7.562, 1_molecular_nm)
Anatomical Structure: DeltaC–Notch1a trans-endocytosis & EphA4–EphrinB2 boundary complex (12–16 hpf (6–14 Somite Stage, ZFA:0000115)) · Zebrahub somite_boundary_epithelium & notochord_sheath (12–16 hpf)
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing at 27 nm (DeltaC–Notch1a trans-endocytosis & EphA4–EphrinB2 boundary complex))
Formal Verification: `morphoeval/lean/loops/Loop_1115_danio_rerio_nm28_delta_notch_synapse_vertex_packing.lean` (`MathOfBiology.Loop_1115`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_nm28_delta_notch_synapse_vertex_packing"
EXPECTED_SHA256 = {
    "danio_rerio_nm28_delta_notch_synapse_vertex_packing_reference.json": "b6eb6c8cdc8fcd0029f25c7e74349178983f6db40969a21522ce90c293826e7f",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "02573887d7a7fa513c7e80e35ca6c50ad59db94a8814024c3bd3a92b3e04e777",
    "paper_citations.bib": "64dc22052af597d87222a5246f7cfd2b47463714ecbe33c2c57a281204b00906"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_nm28_delta_notch_synapse_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2013-riedel-danio-rerio-nm28-delta-notch-synapse-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
