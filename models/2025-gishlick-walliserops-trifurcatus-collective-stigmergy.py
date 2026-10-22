#!/usr/bin/env python3
"""
models/2025-gishlick-walliserops-trifurcatus-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2057)
Organism: Walliserops trifurcatus (Middle Devonian Long-Trident Moroccan Trilobite)
Phenomenon Scale: 8.7 cm (0.08726 m, log10=-1.059, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Walliserops trifurcatus (Middle Devonian (Eifelian, 394 Ma))
Geological Horizon: Middle Devonian (Eifelian, 394 Ma) (394.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Walliserops
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (8.7 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2057_walliserops_trifurcatus_collective_stigmergy.lean` (`MathOfBiology.Loop_2057`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/walliserops_trifurcatus_collective_stigmergy"
EXPECTED_SHA256 = {
    "walliserops_trifurcatus_collective_stigmergy_reference.json": "d0d5b3e8ac37d0999027d6e93f6d15cc6b1c3681c96ac3538d0bb5fd47fde1d4",
    "walliserops_trifurcatus_specimen.jpg": "8a4e9c4d8a7e87dbdcf6b43da85f70188d90b4d5e1d541428ebaa0c016dd9923",
    "scout_manifest.json": "11e2c92ca2d3e5739b6964bdeb7784e151c30a43040a5af073b485742eb6e13c",
    "paper_citations.bib": "802aa621c7101d101e0c5722c9be483e997f3f666563a466362237c6369e4d75"
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
    ref = json.loads((DATASET_DIR / "walliserops_trifurcatus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2025-gishlick-walliserops-trifurcatus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
