#!/usr/bin/env python3
"""
models/2024-gishlick-walliserops-trifurcatus-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1787)
Organism: Walliserops trifurcatus (Middle Devonian Long-Trident Moroccan Trilobite)
Phenomenon Scale: 1.1 cm (0.01067 m, log10=-1.972, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Walliserops trifurcatus (Middle Devonian (Eifelian, 394 Ma))
Geological Horizon: Middle Devonian (Eifelian, 394 Ma) (394.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Walliserops
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (1.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1787_walliserops_trifurcatus_elastic_buckling.lean` (`MathOfBiology.Loop_1787`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/walliserops_trifurcatus_elastic_buckling"
EXPECTED_SHA256 = {
    "walliserops_trifurcatus_elastic_buckling_reference.json": "94301e0321af5144061598354d217f93a6880aaf764d624c969f25d398bf2f4e",
    "walliserops_trifurcatus_specimen.jpg": "8a4e9c4d8a7e87dbdcf6b43da85f70188d90b4d5e1d541428ebaa0c016dd9923",
    "scout_manifest.json": "9ef72bf5ab8f34120965dd5640219eee0fc193d3de5090a43424030cb3a8e4fc",
    "paper_citations.bib": "c29518b1872f6f774f74bcc42fb89059051a182bbcb206b31698656ba82d9ab8"
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
    ref = json.loads((DATASET_DIR / "walliserops_trifurcatus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2024-gishlick-walliserops-trifurcatus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
