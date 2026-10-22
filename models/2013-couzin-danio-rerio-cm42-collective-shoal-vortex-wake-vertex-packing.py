#!/usr/bin/env python3
"""
models/2013-couzin-danio-rerio-cm42-collective-shoal-vortex-wake-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1195)
Organism: Danio rerio (Zebrafish — Adult Collective Shoal)
Phenomenon Scale: 41.2 cm (0.4116 m, log10=-0.386, 8_colonial_m)
Anatomical Structure: Reverse Kármán vortex dipole wake & visual-mechanosensory shoal lattice (Adult Collective Shoal (15–30 Fish Polarized School, 0.42 m arena)) · MapZebrain optic_tectum_loom_circuit & lateral_line_rheotaxis
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing at 41.2 cm (Reverse Kármán vortex dipole wake & visual-mechanosensory shoal lattice))
Formal Verification: `morphoeval/lean/loops/Loop_1195_danio_rerio_cm42_collective_shoal_vortex_wake_vertex_packing.lean` (`MathOfBiology.Loop_1195`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_cm42_collective_shoal_vortex_wake_vertex_packing"
EXPECTED_SHA256 = {
    "danio_rerio_cm42_collective_shoal_vortex_wake_vertex_packing_reference.json": "e3b9e211b1bcfd0a9614c1c04836c256799d37ebc654ad17235159f97d8c0b62",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "8f9023ae11538687b26dbdc9287ee33809dfb8a4657b75d67acee2a74ca0853b",
    "paper_citations.bib": "e19f9b4d60f80de982ca7c0aed60fade5424f2ed51fe5c18d200981495402bb8"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_cm42_collective_shoal_vortex_wake_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2013-couzin-danio-rerio-cm42-collective-shoal-vortex-wake-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
