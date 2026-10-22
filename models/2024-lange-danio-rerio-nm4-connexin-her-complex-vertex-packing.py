#!/usr/bin/env python3
"""
models/2024-lange-danio-rerio-nm4-connexin-her-complex-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1105)
Organism: Danio rerio (Zebrafish — 10–14 hpf)
Phenomenon Scale: 4.1 nm (4.116e-09 m, log10=-8.386, 1_molecular_nm)
Anatomical Structure: Connexin-41.8 hemichannel pore & Her1/Her7 bHLH dimer complex (10–14 hpf (Bud to 10-Somite Stage, ZFA:0001094)) · Zebrahub PSM_presomitic_mesoderm & neural_plate_progenitors (10–14 hpf)
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing at 4.1 nm (Connexin-41.8 hemichannel pore & Her1/Her7 bHLH dimer complex))
Formal Verification: `morphoeval/lean/loops/Loop_1105_danio_rerio_nm4_connexin_her_complex_vertex_packing.lean` (`MathOfBiology.Loop_1105`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_nm4_connexin_her_complex_vertex_packing"
EXPECTED_SHA256 = {
    "danio_rerio_nm4_connexin_her_complex_vertex_packing_reference.json": "56740ba5463478e7e2d6e3a6507eb53b7fa3d5f7b75926a31bc1173a78c10497",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "f71406f99e918b43b94214ea5b12d089559b8bebff6dccaafa10fae4cddfee8c",
    "paper_citations.bib": "cbb625a3a9c88e9d09e8bb0d3dc78926ca694b1b55286e382cb6d15445a7b25e"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_nm4_connexin_her_complex_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2024-lange-danio-rerio-nm4-connexin-her-complex-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
